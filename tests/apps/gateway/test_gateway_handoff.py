"""Consolidated tests:

- test_gateway_handoff: Handoff to a successor run (design 16 WP-D2): gate and blockers, finalisation of the predecessor, the successor on
- test_gateway_handoff_price_card: The console's dry run reports the price-card refusal of ``handoff_run`` (design 16 WP-G, gap found in WP-D2).
- test_gateway_process_line: The gateway names the process and the checkout that serve each run (design 16 WP-G, real-arm handoff check).
"""

import asyncio
import json
import os
import signal
import tempfile
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import aiohttp
from ni_rua.apps.gateway.extensions import agent_settings, handoff
from ni_rua.apps.gateway.extensions.handoff import HANDOFF_OBSERVE_PROMPT, HandoffHost
from ni_rua.apps.gateway.server import ProcessSignals, serve
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import Config
from ni_rua.core.storage.store import open_read_store
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.profiles import apply_profile
from support.fake_agent import FakeAgentSession
from support.fixture_robot import FixtureRobot
from support.agent_fixtures import Listing
import sys
import ni_rua


NI_RUA_TEST_TIER = "component"

# --- test_gateway_handoff.py ---
FAKE_KEY = "sk-test-FAKE-d2-0123456789abcdefKEY"
LISTING = {"object": "list", "data": [
    {"id": "kimi-k2.6", "name": "Kimi K2.6", "protocols": ["MESSAGES", "RESPONSES"],
     "capabilities": {"tools": True, "image_input": True}},
    {"id": "gpt-a", "name": "GPT A", "protocols": ["RESPONSES"], "capabilities": {"tools": True, "image_input": True}},
]}
CAP = 0.5                      # operator speed cap carried across the handoff (fraction of deployment max speed)
WAIT_S = 30.0                  # upper bound of every wait in these tests (s)


def profile_document(wires, *, key=FAKE_KEY, name="Fixture"):
    document = {"schema": "ni_rua.provider_profile.v1", "name": name, "wires": wires}
    if key is not None:
        document["credential"] = {"api_key": key}
    return document


class HandoffRobot(FixtureRobot):
    """``FixtureRobot`` with an operator (``set_runtime_limits``, ``hold``), a deployment torque flag and every
    ``start`` / ``close`` / operator call appended to the shared ``log`` as ``(kind, index, ...)``."""

    def __init__(self, root, log, index, *, start_delay_s=0.0, close_delay_s=0.0, close_report=None):
        super().__init__(root)
        self.log, self.index = log, index
        self.start_delay_s, self.close_delay_s, self.close_report = start_delay_s, close_delay_s, close_report
        self.deployment = {"disable_torque_on_disconnect": False}
        self.capabilities = None

    def start(self):
        self.log.append(("start", self.index))
        time.sleep(self.start_delay_s)
        self.capabilities = super().start()
        return self.capabilities

    def operator(self, command, progress_callback=None, **arguments):
        self.log.append(("operator", self.index, command, dict(arguments)))
        if command == "set_runtime_limits":
            return {"command": command, "status": "applied", "speed_scale_cap": float(arguments["speed_scale_cap"])}
        if command == "hold":
            return {"command": command, "status": "held", "stop_confirmation": {"confirmed": True}}
        raise ValueError(f"unsupported_operator_command:{command}")

    def close(self):
        self.log.append(("close", self.index))
        time.sleep(self.close_delay_s)
        return dict(self.close_report or {"native_close_acknowledged": True, "process_exit_confirmed": True,
                                          "reader_exit_confirmed": True, "returncode": 0, "errors": []})


class LoggedSession(FakeAgentSession):
    """``FakeAgentSession`` of the configured backend that appends ``start`` / ``run_turn`` to the shared log."""

    def __init__(self, *arguments, log, index, fail_start=None, **options):
        super().__init__(*arguments, **options)
        self.log, self.index, self.fail_start = log, index, fail_start
        self.cli_executable, self.process = "/fixture/agent", SimpleNamespace(pid=-1)   # Codex manifest fields

    def start(self, tools, instructions, resume_thread_id=None):
        self.log.append(("session_start", self.index))
        if self.fail_start:
            raise RuntimeError(self.fail_start)
        return super().start(tools, instructions, resume_thread_id)

    def run_turn(self, prompt):
        self.log.append(("run_turn", self.index))
        return super().run_turn(prompt)


class GatewayFixture(unittest.IsolatedAsyncioTestCase):
    """Profiles ``anthro`` (anthropic_messages), ``resp`` (openai_responses), ``nokey``; patched session and robot
    factories; ``_say`` captured; one aiohttp client session."""

    async def asyncSetUp(self):
        agent_settings.reset_runtime_probes()
        self.addCleanup(agent_settings.reset_runtime_probes)
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        self.listing = Listing(LISTING)
        self.addCleanup(self.listing.close)
        self.profiles_dir = self.root / "profiles"
        self.profiles_dir.mkdir()
        base = self.listing.base
        self.write("anthro", profile_document({"anthropic_messages": {"base_url": base}}, name="Anthro"))
        self.write("resp", profile_document({"openai_responses": {"base_url": base + "/v1"}}, name="Resp"))
        self.write("nokey", profile_document({"anthropic_messages": {"base_url": base}}, key=None, name="No key"))
        for target, value in ((factory, "claude_code_runtime_problem"),):
            patcher = patch.object(target, value, return_value=None)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.log, self.robots, self.sessions, self.lines = [], [], [], []
        self.robot_options, self.session_options = {}, {}    # per instance index (1-based)

        def create_backend(**kwargs):
            index = len(self.robots) + 1
            robot = HandoffRobot(kwargs["run_dir"], self.log, index, **self.robot_options.get(index, {}))
            self.robots.append(robot)
            return robot

        def create_session(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
            index = len(self.sessions) + 1
            options = dict(self.session_options.get("all", {}), **self.session_options.get(index, {}))
            fail = options.pop("fail_start_backend", None)
            session = LoggedSession(run_dir, store, tool_handler, log=self.log, index=index,
                                    backend_kind=config.agent.backend, model=config.model.name,
                                    effort=config.model.effort, price_card=price_card,
                                    provider=getattr(factory.actor_upstream(config), "provider", None),
                                    fail_start=(fail[1] if fail and fail[0] == config.agent.backend else None),
                                    **options)
            self.sessions.append(session)
            return session

        for name, value in (("ni_rua.apps.gateway.workbench.create_backend", create_backend),
                            ("ni_rua.apps.gateway.workbench.create_session", create_session),
                            ("ni_rua.apps.gateway.server._say", self.lines.append)):
            patcher = patch(name, side_effect=value)
            patcher.start()
            self.addCleanup(patcher.stop)
        self.http = aiohttp.ClientSession()
        self.addAsyncCleanup(self.http.close)
        self.port = self.lease = None
        self.serving = None

    def write(self, profile_id, document):
        path = self.profiles_dir / f"{profile_id}.local.json"
        path.write_text(json.dumps(document))
        os.chmod(path, 0o600)
        return path

    def config(self, backend="claude_code", profile="anthro", model="kimi-k2.6", **sections):
        config = Config.from_dict({
            "model": {"name": model, "effort": "high"},
            "features": {"grasp_memory_dir": str(self.root / "memory"), "observation_profile": "rgb_proprio_v1",
                         "motion_review": False, "self_briefing": False, "terminal": False, "record_video": False},
            "agent": {"profiles_dir": str(self.profiles_dir)},
            "run": {"experiment_root": str(self.root / "exp"), "category": "handoff", "task": "fixture", "port": 0},
            **sections})
        return apply_profile(config, backend, self.profiles_dir / f"{profile}.local.json")

    # ---- the gateway -------------------------------------------------------------------------------------------
    async def start_gateway(self, config, prompt="Fixture startup."):
        args = handoff.launch_args(config.with_overrides({"run": {"prompt": prompt}}))
        self.serving = asyncio.create_task(serve(args))
        self.addAsyncCleanup(self.stop_gateway)
        run = await self.wait_run(lambda run: run.get("status") == "running", "first run running")
        self.port = run["_port"]
        return run

    async def stop_gateway(self):
        if self.serving is not None and not self.serving.done():
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.wait_for(asyncio.shield(self.serving), WAIT_S)

    def listening_ports(self):
        return [int(line.split(":")[2].split(";")[0]) for line in self.lines
                if isinstance(line, str) and line.startswith("Workbench http://127.0.0.1:")]

    async def get(self, path):
        async with self.http.get(f"http://127.0.0.1:{self.port}{path}") as response:
            return response.status, await response.json()

    async def wait_run(self, predicate, what, *, record=None):
        """Poll ``GET /v1/run`` until ``predicate(snapshot)``; connection errors while a listener is down are expected.
        ``record`` (list) collects ``(status, run_id)`` of every answer."""
        deadline = time.monotonic() + WAIT_S
        last = None
        while time.monotonic() < deadline:
            if self.serving is not None and self.serving.done():
                self.serving.result()
                raise AssertionError(f"gateway ended while waiting for {what}; last {last}")
            ports = self.listening_ports()
            if ports:
                port = self.port or ports[0]
                try:
                    async with self.http.get(f"http://127.0.0.1:{port}/v1/run") as response:
                        body = await response.json()
                        if record is not None:
                            record.append((response.status, body.get("run_id")))
                        if response.status == 200:
                            last = dict(body, _port=port)
                            if predicate(last):
                                return last
                except (aiohttp.ClientError, OSError, ValueError):
                    if record is not None:
                        record.append(("down", None))
            await asyncio.sleep(0.02)
        raise AssertionError(f"timed out waiting for {what}; last {last}; lines {self.lines}")

    async def control(self, command, **fields):
        base = f"http://127.0.0.1:{self.port}"
        headers = {"Origin": base}
        claim = dict(request_id="claim-" + os.urandom(4).hex(), command="claim_input", client_id="fixture-console")
        if self.lease:
            claim["writer_token"] = self.lease
        async with self.http.post(base + "/v1/control", json=claim, headers=headers) as response:
            body = await response.json()
            if response.status != 200:   # a new run has no lease yet: claim afresh
                claim.pop("writer_token", None)
                async with self.http.post(base + "/v1/control", json=claim, headers=headers) as again:
                    body = await again.json()
        self.lease = body["writer_token"]
        request_id = fields.pop("request_id", None) or f"{command}-{os.urandom(4).hex()}"
        async with self.http.post(base + "/v1/control", headers=headers, json=dict(
                request_id=request_id, command=command, client_id="fixture-console", writer_token=self.lease,
                **fields)) as response:
            return response.status, await response.json()

    async def finish_turn(self, index):
        session = self.sessions[index - 1]
        deadline = time.monotonic() + WAIT_S
        while session.state != "running" and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        session.complete(final_text="done")

    async def handoff_to(self, backend="codex", profile="resp", wire="openai_responses", model="gpt-a", **extra):
        return await self.control("handoff_run", backend=backend, profile=profile, wire_api=wire, model=model,
                                  effort="high", **extra)

    # ---- archive -----------------------------------------------------------------------------------------------
    def run_dir(self, run_id):
        matches = [path for path in (self.root / "exp").rglob(run_id) if path.is_dir()]
        self.assertEqual(len(matches), 1, run_id)
        return matches[0]

    def summary(self, run_id):
        return json.loads((self.run_dir(run_id) / "summary.json").read_text())

    def manifest(self, run_id):
        return json.loads((self.run_dir(run_id) / "manifest.json").read_text())

    def events(self, run_id, kind):
        with open_read_store(self.run_dir(run_id)) as store:
            return [event["payload"] for event in store.events_by_type([kind], limit=1000)]

    def assert_no_key_in_archive(self):
        for path in (self.root / "exp").rglob("*"):
            if path.is_file() and not path.is_symlink():
                self.assertNotIn(FAKE_KEY.encode(), path.read_bytes(), path)


class HandoffSequenceTests(GatewayFixture):
    async def test_successor_starts_on_the_same_port_with_the_new_backend_and_the_carried_speed_cap(self):
        first = await self.start_gateway(self.config())
        self.assertEqual(first["agent_backend"], "claude_code")
        await self.finish_turn(1)
        status, limits = await self.control("set_runtime_limits", speed_scale_cap=CAP)
        self.assertEqual((status, limits["speed_scale_cap"]), (200, CAP))
        _, options = await self.get("/v1/console/agent-options")
        self.assertEqual((options["handoff"]["allowed"], options["handoff"]["disable_torque_on_disconnect"]),
                         (True, False))
        # An event stream is open across the swap: the old listener ends it, it never holds the handoff up.
        stream = await self.http.get(f"http://127.0.0.1:{self.port}/v1/events")
        consoles, polling = [], asyncio.Event()

        async def poll_console():
            # The console's other reads during the swap: whole answers of one run, or a closed port; never a 5xx.
            while not polling.is_set():
                try:
                    async with self.http.get(f"http://127.0.0.1:{self.port}/v1/console/state") as response:
                        body = await response.json()
                        consoles.append((response.status, body.get("run_id"), "settings" in body))
                except (aiohttp.ClientError, OSError, ValueError):
                    consoles.append(("down", None, None))
                await asyncio.sleep(0.01)
        poller = asyncio.create_task(poll_console())
        status, receipt = await self.handoff_to(prompt="Look at the table.", request_id="handoff-1")
        self.assertEqual(status, 200, receipt)
        self.assertEqual((receipt["status"], receipt["predecessor_run_id"]), ("handoff_accepted", first["run_id"]))
        successor_id = receipt["successor_run_id"]
        answers = []
        second = await self.wait_run(lambda run: run["run_id"] == successor_id and run["status"] == "running",
                                     "successor running", record=answers)
        polling.set()
        await poller
        # Every answer during the swap was a whole run (the old or the new one) or a closed port; never a 5xx.
        self.assertTrue(all(item in (("down", None),) or (item[0] == 200 and item[1] in (first["run_id"], successor_id))
                            for item in answers), answers)
        self.assertTrue(all(item == ("down", None, None) or (item[0] == 200 and item[1] in (first["run_id"], successor_id)
                                                            and item[2]) for item in consoles), consoles)
        self.assertIn(successor_id, {item[1] for item in consoles})
        await asyncio.wait_for(stream.read(), WAIT_S)
        stream.close()
        self.assertEqual(self.listening_ports(), [self.port, self.port])
        self.assertEqual(second["agent_backend"], "codex")
        self.assertEqual(second["handoff"]["predecessor_run_id"], first["run_id"])
        self.assertIsNone(second["handoff"]["fallback"])

        # Predecessor: finalised like a clean shutdown, with the handoff record naming the successor.
        old = self.summary(first["run_id"])
        self.assertEqual((old["status"], old["cleanup_status"], old["successor_run_id"]),
                         ("stopped", "completed", successor_id))
        self.assertEqual((old["successor_handoff"]["status"], old["successor_handoff"]["from"]["backend"],
                          old["successor_handoff"]["to"]["backend"], old["successor_handoff"]["to"]["profile"]),
                         ("handed_off", "claude_code", "codex", "resp"))
        requested, = self.events(first["run_id"], "run.handoff_requested")
        self.assertEqual((requested["successor_run_id"], requested["carried"]), (successor_id, {"speed_scale_cap": CAP}))
        with open_read_store(self.run_dir(first["run_id"])) as store:
            self.assertEqual(store.read("handoff", "handoff-1")["successor_run_id"], successor_id)

        # Robot: the old worker closed through its close op before the new one started; no motion command; the
        # carried cap reached the new robot before the new agent session started or ran a turn.
        order = [entry[:2] if entry[0] != "operator" else entry for entry in self.log]
        self.assertLess(order.index(("close", 1)), order.index(("start", 2)))
        carried = ("operator", 2, "set_runtime_limits", {"speed_scale_cap": CAP})
        self.assertLess(order.index(("start", 2)), order.index(carried))
        self.assertLess(order.index(carried), order.index(("session_start", 2)))
        self.assertLess(order.index(carried), order.index(("run_turn", 2)))
        self.assertEqual([entry[2] for entry in self.log if entry[0] == "operator"],
                         ["set_runtime_limits", "set_runtime_limits"])

        # Successor: link back, operator prompt, cap recorded with origin handoff, console sees the cap.
        manifest = self.manifest(successor_id)
        self.assertEqual((manifest["handoff"]["predecessor_run_id"], manifest["handoff"]["request_id"],
                          manifest["handoff"]["prompt_source"]), (first["run_id"], "handoff-1", "operator"))
        self.assertEqual((manifest["agent_backend"]["kind"], manifest["effective_config"]["agent"]["backend"]),
                         ("codex", "codex"))
        self.assertEqual(self.sessions[1].calls[1], ("run_turn", "Look at the table."))
        applied, = self.events(successor_id, "operator.set_runtime_limits")
        self.assertEqual((applied["origin"], applied["arguments"]), ("handoff", {"speed_scale_cap": CAP}))
        started, = self.events(successor_id, "run.handoff_started")
        self.assertEqual(started["predecessor_run_id"], first["run_id"])
        _, state = await self.get("/v1/console/state")
        self.assertEqual(state["settings"]["runtime"]["speed_scale_cap"]["value"], CAP)
        _, options = await self.get("/v1/console/agent-options")
        self.assertEqual((options["handoff"]["last"]["status"], options["handoff"]["last"]["predecessor_run_id"]),
                         ("completed", first["run_id"]))

        # The same request id retried against the successor answers the same receipt (no second handoff).
        await self.finish_turn(2)
        status, replay = await self.handoff_to(request_id="handoff-1")
        self.assertEqual((status, replay["status"], replay["successor_run_id"], replay["replayed"]),
                         (200, "handoff_accepted", successor_id, True))
        self.assertEqual(len(self.robots), 2)

        await self.stop_gateway()
        new = self.summary(successor_id)
        self.assertEqual((new["status"], new["cleanup_status"], new["handoff"]["predecessor_run_id"]),
                         ("stopped", "completed", first["run_id"]))
        self.assert_no_key_in_archive()


class HandoffFallbackTests(GatewayFixture):
    async def test_a_successor_whose_session_start_raises_leads_to_one_fallback_run_on_the_old_settings(self):
        self.session_options["all"] = {"fail_start_backend": ("codex", "fixture codex start failure")}
        first = await self.start_gateway(self.config())
        await self.finish_turn(1)
        await self.control("set_runtime_limits", speed_scale_cap=CAP)
        status, changed = await self.control("set_model", model="kimi-k2.7")   # a live change is an old setting too
        self.assertEqual((status, changed["status"]), (200, "applied"))
        status, receipt = await self.handoff_to(prompt="Pick up the cube.")
        self.assertEqual(status, 200, receipt)
        failed_id = receipt["successor_run_id"]
        third = await self.wait_run(lambda run: run["run_id"] not in (first["run_id"], failed_id)
                                    and run["status"] == "running", "fallback running")
        self.assertEqual(third["agent_backend"], "claude_code")
        self.assertEqual((self.sessions[2].model, self.sessions[2].provider), ("kimi-k2.7", "anthro"))
        self.assertEqual(third["handoff"]["fallback"]["failed_run_id"], failed_id)
        self.assertIn("fixture codex start failure", third["handoff"]["fallback"]["reason"])
        _, options = await self.get("/v1/console/agent-options")
        self.assertEqual((options["handoff"]["last"]["status"], options["handoff"]["last"]["failed_run_id"]),
                         ("fallback", failed_id))
        self.assertIn("fixture codex start failure", options["handoff"]["last"]["reason"])

        failed = self.summary(failed_id)
        self.assertEqual((failed["status"], failed["handoff"]["predecessor_run_id"]), ("failed", first["run_id"]))
        self.assertIn("fixture codex start failure", failed["infrastructure_error"])
        self.assertEqual(self.summary(first["run_id"])["successor_run_id"], failed_id)
        fallback, = self.events(third["run_id"], "run.handoff_fallback")
        self.assertEqual((fallback["failed_run_id"], fallback["predecessor_run_id"]), (failed_id, first["run_id"]))
        manifest = self.manifest(third["run_id"])
        self.assertEqual((manifest["handoff"]["fallback"]["failed_run_id"], manifest["agent_backend"]["kind"],
                          manifest["handoff"]["prompt_source"]), (failed_id, "claude_code", "observe_only_default"))
        # The fallback observes only (never the operator's task for the other backend) and keeps the speed cap.
        self.assertEqual(self.sessions[2].calls[1], ("run_turn", HANDOFF_OBSERVE_PROMPT))
        self.assertEqual(self.events(third["run_id"], "operator.set_runtime_limits")[0]["origin"], "handoff")
        self.assertEqual([entry[:2] for entry in self.log if entry[0] in ("start", "close")],
                         [("start", 1), ("close", 1), ("start", 2), ("close", 2), ("start", 3)])
        # The fallback run is a working run: it completes a turn.
        await self.finish_turn(3)
        status, prompt = await self.control("prompt", prompt="Observe again.")
        self.assertEqual((status, prompt["status"]), (200, "running"))
        await self.finish_turn(3)
        await self.wait_run(lambda run: run["agent_status"] == "completed", "fallback turn completed")
        await self.stop_gateway()
        self.assertEqual(self.summary(third["run_id"])["status"], "stopped")
        self.assert_no_key_in_archive()

    async def test_a_successor_that_cannot_be_built_is_finalised_failed_and_the_fallback_starts(self):
        self.robot_options[1] = dict(close_delay_s=0.5)
        first = await self.start_gateway(self.config())
        await self.finish_turn(1)
        status, receipt = await self.handoff_to()
        self.assertEqual(status, 200, receipt)
        # The successor's profile loses its key after the gate accepted it (an operator edit, a race).
        self.write("resp", profile_document({"openai_responses": {"base_url": self.listing.base + "/v1"}}, key=None))
        failed_id = receipt["successor_run_id"]
        third = await self.wait_run(lambda run: run["run_id"] not in (first["run_id"], failed_id)
                                    and run["status"] == "running", "fallback running")
        failed = self.summary(failed_id)
        self.assertEqual((failed["status"], failed["handoff"]["request_id"]), ("failed", receipt["request_id"]))
        self.assertIn("profile_missing_key", failed["infrastructure_error"])
        self.assertFalse((self.run_dir(failed_id) / "manifest.json").exists())
        self.assertIn("profile_missing_key", third["handoff"]["fallback"]["reason"])
        self.assertEqual(len(self.robots), 2)    # the failed successor never created a robot


class HandoffShutdownTests(GatewayFixture):
    async def test_sigint_after_acceptance_finalises_both_runs_and_starts_nothing(self):
        self.robot_options[1] = dict(close_delay_s=1.0)
        first = await self.start_gateway(self.config())
        await self.finish_turn(1)
        status, receipt = await self.handoff_to()
        self.assertEqual(status, 200)
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.wait_for(asyncio.shield(self.serving), WAIT_S)
        old = self.summary(first["run_id"])
        self.assertEqual((old["status"], old["cleanup_status"], old["successor_run_id"]),
                         ("stopped", "completed", receipt["successor_run_id"]))
        never = self.summary(receipt["successor_run_id"])
        self.assertEqual((never["status"], never["termination_reason"]), ("cancelled", "gateway_shutdown_before_start"))
        self.assertEqual(never["handoff"]["predecessor_run_id"], first["run_id"])
        self.assertEqual(len(self.robots), 1)

    async def test_sigint_during_the_successor_start_stops_it_without_a_fallback(self):
        self.robot_options[2] = dict(start_delay_s=1.0)
        first = await self.start_gateway(self.config())
        await self.finish_turn(1)
        status, receipt = await self.handoff_to()
        self.assertEqual(status, 200)
        deadline = time.monotonic() + WAIT_S
        while ("start", 2) not in self.log and time.monotonic() < deadline:
            await asyncio.sleep(0.02)
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.wait_for(asyncio.shield(self.serving), WAIT_S)
        self.assertEqual(self.summary(first["run_id"])["status"], "stopped")
        stopped = self.summary(receipt["successor_run_id"])
        self.assertEqual((stopped["status"], stopped["handoff"]["predecessor_run_id"]), ("stopped", first["run_id"]))
        self.assertEqual(len(self.robots), 2)    # no fallback run
        self.assertNotIn(("session_start", 2), [entry[:2] for entry in self.log])

    async def test_a_predecessor_whose_robot_exit_is_unconfirmed_gets_no_successor(self):
        self.robot_options[1] = dict(close_report={"native_close_acknowledged": False, "process_exit_confirmed": False,
                                                   "reader_exit_confirmed": True, "errors": ["fixture: still running"]})
        first = await self.start_gateway(self.config())
        await self.finish_turn(1)
        status, receipt = await self.handoff_to()
        self.assertEqual(status, 200)
        await asyncio.wait_for(asyncio.shield(self.serving), WAIT_S)
        old = self.summary(first["run_id"])
        self.assertEqual((old["status"], old["cleanup_status"], old["successor_handoff"]["status"],
                          old["successor_handoff"]["reason"]),
                         ("unknown", "failed", "abandoned", "predecessor_cleanup_unconfirmed"))
        never = self.summary(receipt["successor_run_id"])
        self.assertEqual((never["status"], never["termination_reason"]),
                         ("cancelled", "predecessor_cleanup_unconfirmed"))
        self.assertEqual(len(self.robots), 1)


class SharedRemote:
    """``RemoteManager`` double: the services ``serve()`` owns for the whole process (one ``start``, one ``stop``)."""
    instances = []

    def __init__(self, config, **kwargs):
        self.config, self.record_path, self.calls = config, None, []
        self.supervisor = None
        SharedRemote.instances.append(self)

    def start(self):
        self.calls.append("start")
        self.supervisor = SimpleNamespace(record_path=self.record_path, record={"schema": "ni_rua.remote_services.v1",
                                                                                "host": "gpuhost"})

    def abort(self):
        self.calls.append("abort")

    def stop(self):
        self.calls.append("stop")

    def status(self):
        return {"enabled": True, "state": "ready"}


class HandoffRemoteServicesTests(GatewayFixture):
    async def test_the_successor_reuses_the_remote_services_and_records_them_in_its_own_run(self):
        SharedRemote.instances = []
        patcher = patch("ni_rua.apps.gateway.server.RemoteManager", side_effect=SharedRemote)
        patcher.start()
        self.addCleanup(patcher.stop)
        config = self.config(remote={"enabled": True, "host": "gpuhost", "services": [
            {"name": "vggt", "remote_port": 8331}]})
        first = await self.start_gateway(config)
        await self.finish_turn(1)
        status, receipt = await self.handoff_to()
        self.assertEqual(status, 200, receipt)
        await self.wait_run(lambda run: run["run_id"] == receipt["successor_run_id"] and run["status"] == "running",
                            "successor running")
        remote, = SharedRemote.instances
        self.assertEqual(remote.calls, ["start"])            # connected once for the process, not per run
        successor = self.run_dir(receipt["successor_run_id"])
        record = successor / "private" / "remote_services.json"
        self.assertEqual((Path(remote.record_path), Path(remote.supervisor.record_path)), (record, record))
        self.assertEqual(json.loads(record.read_text())["host"], "gpuhost")
        _, status_view = await self.get("/v1/remote/status")
        self.assertEqual(status_view["state"], "ready")
        await self.stop_gateway()
        self.assertEqual(remote.calls, ["start", "stop"])
        self.assertEqual(self.summary(first["run_id"])["successor_run_id"], receipt["successor_run_id"])


class FallbackRuleTests(unittest.TestCase):
    def test_only_a_successor_that_failed_to_start_gets_a_fallback(self):
        request = SimpleNamespace(base_config=None)
        for kind, started, received, expected in (("fallback", False, None, None), ("successor", True, None, None),
                                                  ("successor", False, signal.SIGINT, None)):
            launch = handoff.Launch(SimpleNamespace(received=received), request=request, kind=kind, started=started)
            self.assertIs(handoff.Successor(SimpleNamespace(), launch).fallback(RuntimeError("x")), expected)


class ProcessSignalsTests(unittest.IsolatedAsyncioTestCase):
    async def test_handlers_are_installed_once_and_follow_the_current_run(self):
        signals = ProcessSignals()
        first, second = asyncio.Event(), asyncio.Event()
        signals.bind(first)
        signals.bind(second)
        os.kill(os.getpid(), signal.SIGINT)
        await asyncio.wait_for(second.wait(), WAIT_S)
        self.assertFalse(first.is_set())
        self.assertEqual(signals.received, signal.SIGINT)
        third = asyncio.Event()
        signals.bind(third)            # a run bound after the signal starts in the shutdown state
        self.assertTrue(third.is_set())

    @unittest.skipUnless(hasattr(signal, "SIGHUP"), "no SIGHUP on this platform")
    async def test_a_sighup_ignored_at_launch_stays_ignored_for_every_later_run(self):
        previous = signal.signal(signal.SIGHUP, signal.SIG_IGN)
        try:
            signals = ProcessSignals()
            signals.bind(asyncio.Event())
            signals.bind(asyncio.Event())
            self.assertNotIn(signal.SIGHUP, signals.signals)
            self.assertIs(signal.getsignal(signal.SIGHUP), signal.SIG_IGN)
        finally:
            asyncio.get_running_loop().remove_signal_handler(signal.SIGINT)
            asyncio.get_running_loop().remove_signal_handler(signal.SIGTERM)
            signal.signal(signal.SIGHUP, previous)


class HandoffGateTests(GatewayFixture):
    """``handoff_run`` against a Workbench with a run-supervisor host attached by hand: every blocker and every
    selection refusal answers HTTP 400 with its code and changes nothing (no directory, no event, ``done`` unset)."""

    async def asyncSetUp(self):
        await super().asyncSetUp()
        from aiohttp.test_utils import TestClient, TestServer
        self.base_config = self.config()
        self.workbench = Workbench(self.root / "run", robot=HandoffRobot(self.root, self.log, 1),
                                   config=self.base_config)
        self.workbench.harness.start()
        self.workbench.starting, self.workbench.status = False, "running"
        self.sessions[0].start([], "instructions")
        self.done = asyncio.Event()
        self.workbench.handoff_host = HandoffHost(base_config=handoff.base_config(SimpleNamespace(
            resolved_config=self.base_config), self.workbench), port=4321, done=self.done)
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        self.addAsyncCleanup(self.workbench.close)
        self.addAsyncCleanup(self.client.close)
        self.origin = str(self.client.make_url("")).rstrip("/")
        status, owner = await self.post(dict(request_id="claim", command="claim_input", client_id="fixture"))
        self.assertEqual(status, 200)
        self.lease = dict(client_id="fixture", writer_token=owner["writer_token"])

    async def post(self, body):
        response = await self.client.post("/v1/control", headers={"Origin": self.origin}, json=body)
        return response.status, await response.json()

    async def request(self, identifier="h", **fields):
        selection = dict(backend="codex", profile="resp", wire_api="openai_responses", model="gpt-a", effort="high")
        selection.update(fields)
        return await self.post(dict(request_id=identifier, command="handoff_run", **self.lease, **selection))

    def allocated(self):
        return [path for path in (self.root / "exp").rglob("experiment.json")] if (self.root / "exp").exists() else []

    async def assertRefused(self, code, **fields):
        status, body = await self.request(identifier="refused-" + code + os.urandom(3).hex(), **fields)
        self.assertEqual((status, body.get("error", "").split(":")[0]), (400, code), body)
        self.assertFalse(self.done.is_set())
        self.assertEqual(self.allocated(), [])
        self.assertIsNone(self.workbench.successor)
        self.assertEqual(self.workbench.store.events_by_type(["run.handoff_requested"]), [])
        return body

    async def test_each_blocker_refuses_with_its_code(self):
        workbench, gate = self.workbench, self.workbench.operator_gate
        cases = [
            ("run_starting", lambda: setattr(workbench, "starting", True), lambda: setattr(workbench, "starting", False)),
            ("agent_turn_running", lambda: setattr(self.sessions[0], "state", "running"),
             lambda: setattr(self.sessions[0], "state", "idle")),
            ("action_in_progress", lambda: setattr(workbench.harness, "active_action", "act-1"),
             lambda: setattr(workbench.harness, "active_action", None)),
            ("program_active_or_unresolved", lambda: setattr(workbench.harness, "programs", SimpleNamespace(
                snapshot=lambda: {"active_jobs": ["p"]}, unresolved_jobs=lambda: [], close=lambda: None,
                cancel_all=lambda: None)), lambda: setattr(workbench.harness, "programs", None)),
            ("unknown_unacknowledged", lambda: gate["unknown"].append(dict(id="operator:x", source="operator_hold",
                                                                         reason="fixture")),
             lambda: gate["unknown"].clear()),
            ("torque_not_on", lambda: gate.update(torque="off"), lambda: gate.update(torque=None)),
            ("torque_not_on", lambda: gate.update(torque="unknown"), lambda: gate.update(torque=None)),
            ("benchmark_run_owned_by_queue", lambda: setattr(workbench, "config", workbench.config.with_overrides(
                {"run": {"benchmark": True}})), lambda: setattr(workbench, "config", self.base_config_running)),
        ]
        self.base_config_running = workbench.config
        for code, block, unblock in cases:
            with self.subTest(code=code):
                block()
                try:
                    await self.assertRefused(code)
                finally:
                    unblock()
        response = await self.client.get("/v1/console/agent-options")
        self.assertEqual((await response.json())["handoff"]["allowed"], True)

    async def test_selection_refusals_name_their_code(self):
        await self.assertRefused("profile_missing_key", backend="claude_code", profile="nokey",
                                 wire_api="anthropic_messages", model="kimi-k2.6")
        await self.assertRefused("profile_not_found", profile="absent")
        await self.assertRefused("backend_unavailable", backend="opencode", profile="anthro",
                                 wire_api="anthropic_messages", model="kimi-k2.6")
        await self.assertRefused("no_common_wire", backend="claude_code", profile="resp", wire_api=None)
        await self.assertRefused("invalid_agent_backend", backend="robot")
        await self.assertRefused("invalid_task_prompt", prompt="   ")
        # Price card: the run's card has no rate for the successor's provider and model.
        self.sessions[0].price_card = {"version": "1", "source": "fixture", "currency": "USD",
                                       "rates": [{"provider": "anthro", "model": "kimi-k2.6"}]}
        body = await self.assertRefused("price_card_mismatch")
        self.assertIn("'resp'", body["error"])
        self.sessions[0].price_card["rates"].append({"provider": "resp", "model": "gpt-a"})
        status, receipt = await self.request(identifier="priced")
        self.assertEqual((status, receipt["status"]), (200, "handoff_accepted"))

    async def test_a_handoff_to_the_external_backend_is_refused(self):
        # An external agent run starts only from the command line (design 20 section 2.1): never as a successor.
        body = await self.assertRefused("handoff_unsupported_by_backend", backend="external")
        self.assertIn("starts only from the command line", body["error"])

    async def test_a_profile_whose_key_the_provider_refuses_is_refused(self):
        denied = Listing({"error": "unauthorized"}, status=401)
        self.addCleanup(denied.close)
        self.write("denied", profile_document({"openai_responses": {"base_url": denied.base + "/v1"}}, name="Denied"))
        body = await self.assertRefused("handoff_profile_key_refused", profile="denied")
        self.assertIn("http_401", body["error"])
        self.assertNotIn(FAKE_KEY, json.dumps(body))
        # A listing that is merely unavailable does not block.
        down = Listing({"error": "busy"}, status=503)
        self.addCleanup(down.close)
        self.write("flaky", profile_document({"openai_responses": {"base_url": down.base + "/v1"}}, name="Flaky"))
        status, receipt = await self.request(identifier="flaky", profile="flaky")
        self.assertEqual((status, receipt["status"]), (200, "handoff_accepted"))

    async def test_without_a_run_supervisor_or_in_replay_nothing_hands_off(self):
        self.workbench.handoff_host = None
        await self.assertRefused("handoff_not_available")
        response = await self.client.get("/v1/console/agent-options")
        self.assertIn("handoff_not_available", [item["code"] for item in (await response.json())["handoff"]["blockers"]])
        self.workbench.replay = True
        status, body = await self.request(identifier="replayed")
        self.assertEqual((status, body["error"]), (400, "replay_is_read_only"))
        self.workbench.replay = False

    async def test_acceptance_allocates_records_closes_admission_and_refuses_later_mutations(self):
        status, receipt = await self.request(identifier="accept", prompt="Observe the cube.")
        self.assertEqual(status, 200, receipt)
        self.assertEqual((receipt["status"], receipt["predecessor_run_id"]),
                         ("handoff_accepted", self.workbench.identity["run_id"]))
        self.assertTrue(self.done.is_set())
        directory, = [path.parent for path in self.allocated()]
        self.assertEqual(directory.name, receipt["successor_run_id"])
        self.assertEqual([path.name for path in directory.iterdir()], ["experiment.json"])
        record = self.workbench.store.read("handoff", "accept")
        self.assertEqual((record["to"]["backend"], record["to"]["provider"], record["prompt"], record["port"]),
                         ("codex", "resp", "Observe the cube.", 4321))
        self.assertFalse(self.workbench.harness.accepting)
        request = self.workbench.successor
        self.assertEqual((request.config.agent.backend, request.config.run.run_dir, request.config.run.port,
                          request.config.run.prompt), ("codex", str(directory), 4321, "Observe the cube."))
        self.assertEqual(request.base_config.agent.backend, "claude_code")
        response = await self.client.get("/v1/run")
        self.assertEqual((await response.json())["handoff"]["successor"],
                         dict(request_id="accept", successor_run_id=directory.name, status="accepted"))
        status, again = await self.request(identifier="accept", prompt="Observe the cube.")
        self.assertEqual((status, again), (200, receipt))    # same request id: the archived receipt, once
        status, second = await self.request(identifier="other")
        self.assertEqual((status, second["error"].split(":")[0]), (400, "handoff_in_progress"))
        status, prompt = await self.post(dict(request_id="late", command="prompt", prompt="Move.", **self.lease))
        self.assertEqual((status, prompt["error"].split(":")[0]), (400, "handoff_in_progress"))
        status, hold = await self.post(dict(request_id="hold", command="hold"))
        self.assertEqual((status, hold["status"]), (200, "held"))
        self.assertNotIn(FAKE_KEY, json.dumps(record))


# --- test_gateway_handoff_price_card.py ---
SUCCESSOR = dict(backend="codex", profile="resp", wire_api="openai_responses", model="gpt-a", effort="high")
CARD = {"version": "1", "source": "fixture", "currency": "USD",
        "rates": [{"provider": "anthro", "model": "kimi-k2.6"}]}


class PriceCardDryRunTests(GatewayFixture):
    async def asyncSetUp(self):
        await super().asyncSetUp()
        from aiohttp.test_utils import TestClient, TestServer
        self.base_config = self.config()
        self.workbench = Workbench(self.root / "run", robot=HandoffRobot(self.root, self.log, 1),
                                   config=self.base_config)
        self.workbench.harness.start()
        self.workbench.starting, self.workbench.status = False, "running"
        self.sessions[0].start([], "instructions")
        self.done = asyncio.Event()
        self.workbench.handoff_host = HandoffHost(base_config=handoff.base_config(SimpleNamespace(
            resolved_config=self.base_config), self.workbench), port=4321, done=self.done)
        self.client = TestClient(TestServer(self.workbench.app()))
        await self.client.start_server()
        self.addAsyncCleanup(self.workbench.close)
        self.addAsyncCleanup(self.client.close)
        self.origin = str(self.client.make_url("")).rstrip("/")
        status, owner = await self.post_control(dict(request_id="claim", command="claim_input", client_id="fixture"))
        self.assertEqual(status, 200)
        self.lease = dict(client_id="fixture", writer_token=owner["writer_token"])

    async def post_control(self, body):
        response = await self.client.post("/v1/control", headers={"Origin": self.origin}, json=body)
        return response.status, await response.json()

    async def check(self, **fields):
        response = await self.client.post("/v1/console/agent-options/check", json=dict(SUCCESSOR, **fields))
        self.assertEqual(response.status, 200)
        body = await response.json()
        self.assertNotIn(FAKE_KEY, json.dumps(body))
        return body

    async def handoff(self, identifier, **fields):
        return await self.post_control(dict(request_id=identifier, command="handoff_run", **self.lease,
                                            **dict(SUCCESSOR, **fields)))

    @staticmethod
    def codes(body):
        return [item["code"] for item in body["handoff"]["blockers"]]

    async def test_a_selection_the_price_card_does_not_cover_fails_the_dry_run_with_the_handoff_code(self):
        body = await self.check()
        self.assertEqual((body["ok"], body["reason"], body["handoff"]["allowed"], self.codes(body)),
                         (True, None, True, []))                       # no price card: nothing to contradict
        self.sessions[0].price_card = json.loads(json.dumps(CARD))
        before = (self.workbench.store.event_cursor(), len(self.workbench.store.records("control_request")))
        body = await self.check()
        self.assertEqual((body["ok"], body["reason"], body["wire_api"]), (False, "price_card_mismatch",
                                                                           "openai_responses"))
        self.assertIn("'resp'", body["message"])
        self.assertIn("'gpt-a'", body["message"])
        self.assertEqual((body["handoff"]["allowed"], self.codes(body)), (False, ["price_card_mismatch"]))
        blocker, = body["handoff"]["blockers"]
        self.assertEqual(blocker["message"], handoff.HANDOFF_BLOCKERS["price_card_mismatch"])
        self.assertIn("'resp'", blocker["detail"])
        self.assertIsNotNone(body["reasoning"])                        # the selection's controls are still shown
        self.assertEqual((self.workbench.store.event_cursor(), len(self.workbench.store.records("control_request"))),
                         before)                                       # the dry run has no side effect
        # handoff_run refuses the very selection the dry run refused, with the same code.
        status, refused = await self.handoff("priced-out")
        self.assertEqual((status, refused["error"].split(":")[0]), (400, "price_card_mismatch"))
        self.assertFalse(self.done.is_set())
        # A selection the card covers passes both, and the running pair is covered by its own rate.
        body = await self.check(backend="claude_code", profile="anthro", wire_api="anthropic_messages",
                                model="kimi-k2.6")
        self.assertEqual((body["ok"], body["handoff"]["allowed"]), (True, True), body)
        self.sessions[0].price_card["rates"].append({"provider": "resp", "model": "gpt-a"})
        body = await self.check()
        self.assertEqual((body["ok"], body["reason"], self.codes(body)), (True, None, []))
        status, receipt = await self.handoff("priced-in")
        self.assertEqual((status, receipt["status"]), (200, "handoff_accepted"))

    async def test_a_missing_model_is_the_running_one_as_in_handoff_run(self):
        self.sessions[0].price_card = json.loads(json.dumps(CARD))
        # Same backend and profile, no model and effort named: the running kimi-k2.6, which the card covers.
        body = await self.check(backend="claude_code", profile="anthro", wire_api="anthropic_messages", model=None,
                                effort=None)
        self.assertEqual((body["ok"], body["reason"]), (True, None), body)

    async def test_run_blockers_and_selection_blockers_are_listed_together(self):
        self.sessions[0].price_card = json.loads(json.dumps(CARD))
        self.sessions[0].state = "running"
        try:
            body = await self.check()
        finally:
            self.sessions[0].state = "idle"
        self.assertEqual(self.codes(body), ["agent_turn_running", "price_card_mismatch"])
        self.assertEqual((body["ok"], body["reason"]), (False, "price_card_mismatch"))
        # agent-options has no selection: its blockers are the run's own, unchanged.
        response = await self.client.get("/v1/console/agent-options")
        options = await response.json()
        self.assertEqual((options["handoff"]["allowed"], options["handoff"]["blockers"]), (True, []))

    async def test_a_gate_failure_is_reported_before_the_price_card(self):
        self.sessions[0].price_card = json.loads(json.dumps(CARD))
        body = await self.check(backend="claude_code", profile="nokey", wire_api="anthropic_messages",
                                model="kimi-k2.6")
        self.assertEqual((body["ok"], body["reason"]), (False, "profile_missing_key"))
        self.assertNotIn("price_card_mismatch", self.codes(body))


# --- test_gateway_process_line.py ---
class GatewayProcessLineTests(GatewayFixture):
    async def test_each_run_names_its_process_and_checkout_after_the_workbench_line(self):
        await self.start_gateway(self.config())
        lines = [line for line in self.lines if isinstance(line, str)]
        workbench = next(index for index, line in enumerate(lines) if line.startswith("Workbench http://127.0.0.1:"))
        self.assertEqual(lines[workbench + 1], f"gateway_process pid={os.getpid()} python={sys.executable} "
                                               f"ni_rua={Path(ni_rua.__file__).resolve()}")
        self.assertTrue(lines[0].startswith("Workbench http://"), lines[0])   # still the first line



if __name__ == "__main__":
    unittest.main()
