"""Merged cortex tests:
- tests/test_agent_opencode_session_smoke.py: One real-binary smoke run of OpenCodeSession (plan 07 §4; the full conformance suite is package T12).
- tests/test_agent_opencode_session_unit.py: OpenCodeSession against the scripted fake ``opencode serve`` 2.2.7 (design 11 §1-§2; plan 07 §3, §3.9 failure
- tests/test_agent_opencode_terminal_smoke.py: Real-binary check of the OpenCode native TUI attach (work package NATIVE-TUI).
"""

NI_RUA_TEST_TIER = "component"

import json
import os
import shutil
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock
from support.fake_model_servers import FakeAnthropicServer, Gate, Step, text_step, tool_step
from ni_rua.core.config import Config
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.isolation import process_tree_snapshot
import signal
import stat
import sys
import threading
from support.fake_mcp_client import FakeMcpClient
from support.fake_opencode import (DIRECTORY, PASSWORD, FakeOpenCodeServer, model_info, text_item, tool_item)
from ni_rua.core.config import Config, ConfigError
from ni_rua.cortex.agents.base import AgentRejected, AgentTransportUnknown
from ni_rua.cortex.agents.isolation import VersionMismatch, process_tree_snapshot
from ni_rua.cortex.agents.model_catalogue import CatalogueModel, ModelCatalogue
from ni_rua.cortex.agents.opencode import config as opencode_config
from ni_rua.cortex.agents.opencode.client import OpenCodeClient, OpenCodeTransportError
from ni_rua.cortex.agents.opencode.config import OpenCodeConfigError
from ni_rua.cortex.agents.opencode.session import OpenCodeSession, naming_note
import asyncio
import re
import subprocess
from support.fake_model_servers import FakeAnthropicServer, text_step, tool_step
from ni_rua.cortex.session.terminal import Terminal

# ==== tests/test_agent_opencode_session_smoke.py ====

OC_SMOKE_EXECUTABLE = os.environ.get("NI_RUA_OPENCODE_EXECUTABLE")
OC_SMOKE_PROBLEM = (factory.opencode_runtime_problem(OC_SMOKE_EXECUTABLE) if OC_SMOKE_EXECUTABLE else
           "set NI_RUA_OPENCODE_EXECUTABLE to the pinned opencode 2.2.7 platform binary")
if OC_SMOKE_PROBLEM is None and shutil.which("rg") is None:
    PROBLEM = "ripgrep (rg) is not installed"
OC_SMOKE_IDENTITY = dict(run_id="smoke", episode_id="ep", env_instance_id="env", env_generation=1)
OC_SMOKE_KEY_ENV = "NI_RUA_SMOKE_OPENCODE_UPSTREAM_KEY"
HOST_KEY = "host-secret-opencode-smoke-key"
OC_SMOKE_MARKER = "NI-RUA-OPENCODE-SMOKE-INSTRUCTIONS-5c1d"
OC_SMOKE_SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x position in m, default 0"}},
          "additionalProperties": False}
OC_SMOKE_TOOLS = [{"type": "function", "name": "look", "description": "Read-only view of the scene.", "inputSchema": OC_SMOKE_SCHEMA}]
DETACH_COMMAND = ("perl -e 'sleep 600' >/dev/null 2>&1 & echo $! > orphan.pid; "
                  "perl -MPOSIX -e 'POSIX::setsid(); sleep 600' >/dev/null 2>&1 & echo $! > detached.pid")


def oc_smoke_system_text(body):
    system = body.get("system")
    if isinstance(system, list):
        return "\n".join(block.get("text", "") for block in system if isinstance(block, dict))
    return system if isinstance(system, str) else ""


def oc_smoke_last_user(body):
    """``(text, has_tool_result)`` of the last user message of an Anthropic Messages body."""
    for message in reversed(body.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content, False
        texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        results = any(isinstance(block, dict) and block.get("type") == "tool_result" for block in content)
        return "\n".join(texts), results
    return "", False


class OcSmokeRouter:
    """Actor-loop requests (tools declared + the instructions marker in the system text) are routed on the last user
    message; anything else (none expected: the session has an explicit title) gets a short text."""

    def __init__(self):
        self.gate_steer, self.gate_abort = Gate(), Gate()
        self.main = []

    def __call__(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        if not (body.get("tools") and OC_SMOKE_MARKER in oc_smoke_system_text(body)):
            return text_step("side-ok")
        self.main.append(request)
        text, has_result = oc_smoke_last_user(body)
        if has_result:
            return text_step("FINAL")
        if "STEER-NOW" in text:
            return text_step("FINAL-STEER")
        if "PLAIN" in text:        # checked before GATE-A: the aborted prompt merges into the next user message
            return text_step("FINAL-2")
        if "DETACH" in text:       # conformance item 15 (b): two children outlive the shell that started them
            return tool_step("shell", {"command": DETACH_COMMAND})
        if "SHELL" in text:
            return tool_step("shell", {"command": "sleep 600", "timeout": 600000})
        if "GATE-S" in text:
            return Step(text="late answer", gate=self.gate_steer)
        if "GATE-A" in text:
            return Step(text="never delivered", gate=self.gate_abort)
        return tool_step("ni_rua_look", {"x": 1})


def bodies_text(request):
    return json.dumps(request.json)


@unittest.skipIf(OC_SMOKE_PROBLEM is not None, f"OpenCode runtime unavailable: {OC_SMOKE_PROBLEM}")
class OpenCodeRealBinarySmoke(unittest.TestCase):
    def wait(self, predicate, timeout, message):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            if predicate():
                return
            time.sleep(0.05)
        self.fail(message)

    def test_tool_turn_steer_interrupt_and_close(self):
        from ni_rua.cortex.agents.opencode.session import OpenCodeSession
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        router = OcSmokeRouter()
        server = FakeAnthropicServer([], default=router, gate_timeout_s=60).start()
        self.addCleanup(server.close)
        self.addCleanup(router.gate_steer.release)
        self.addCleanup(router.gate_abort.release)
        environment = mock.patch.dict(os.environ, {OC_SMOKE_KEY_ENV: HOST_KEY})
        environment.start()
        self.addCleanup(environment.stop)
        store = Store(root, OC_SMOKE_IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({
            "agent": {"backend": "opencode", "opencode": {"executable": OC_SMOKE_EXECUTABLE, "upstream": {
                "wire_api": "anthropic_messages", "base_url": server.base_url, "api_key_env": OC_SMOKE_KEY_ENV,
                "auth_scheme": "x-api-key", "provider": "fake"}}},
            "model": {"name": "claude-smoke-model", "effort": "high"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        self.assertIsNone(factory.backend_requirements(config))
        calls, disconnects = [], []

        def handler(params):
            calls.append(params)
            return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: one red cube"}]}

        session = factory.create_session(config, root, store, handler)
        self.assertIsInstance(session, OpenCodeSession)
        session.on_disconnect = lambda: disconnects.append(session.state)
        session.tool_parallel_hints = {"look": True}
        self.addCleanup(session.close)

        thread = session.start(OC_SMOKE_TOOLS, "Robot instructions. " + OC_SMOKE_MARKER)
        self.assertEqual(session.state, "idle")
        self.assertTrue(thread.startswith("ses"))
        (role, process), = session.owned_processes()
        self.assertEqual(role, "opencode")
        identity = store.read_state("runtime", "agent")
        self.assertEqual((identity["backend"], identity["thread_id"], identity["cli_version"], identity["wire_api"]),
                         ("opencode", thread, "2.2.7", "anthropic_messages"))
        provenance = session.provenance()
        self.assertIsNotNone(provenance["route_list_sha256"])
        self.assertIsNone(provenance["native_tools"]["enabled"])  # known from the first main request the relay forwards

        # 1. One turn with one registry tool call ends completed with the final answer.
        turn = session.run_turn("Look at the scene, then answer.")
        self.assertTrue(session.turn_done.wait(120), "tool turn did not end")
        self.assertEqual((session.state, session.turn_result["status"]), ("completed", "completed"),
                         session.turn_result)
        self.assertEqual(session.completed_final_answer(), "FINAL")
        params, = calls
        self.assertEqual((params["threadId"], params["turnId"], params["tool"], params["callIdSource"]),
                         (thread, turn, "look", "mcp_request"))
        row = store.transcript_item(params["callId"])
        self.assertEqual((row["type"], row["status"], row["success"]), ("dynamicToolCall", "completed", True))
        self.assertTrue(row.get("native_id"), row)        # the projector attached OpenCode's callID
        # Only relayed model requests carry the host key; stray loopback probes from unrelated local tools (seen on
        # Linux: keyless `GET /`) can reach the fake upstream port and must not fail the test.
        self.assertTrue(all(request.header("x-api-key") == HOST_KEY for request in server.generation_requests()))
        self.assertEqual(router.main[0].json["model"], "claude-smoke-model")
        self.assertIn("scene: one red cube", " ".join(result.text for result in server.find_tool_results(router.main[1])))
        self.assertNotIn("Instructions from:", bodies_text(router.main[0]))   # egress neutralises the file path
        self.assertEqual(store.read_state("runtime", "model_requests")["counting"],
                         "delivered_usable_generation_completions_v1")
        enabled = session.provenance()["native_tools"]["enabled"]       # the tools of the first main request
        self.assertTrue({"shell", "read", "subagent"} <= set(enabled), enabled)
        self.assertFalse({"webfetch", "websearch", "question"} & set(enabled), enabled)
        self.assertEqual(set(enabled), {tool["name"] for tool in router.main[0].json["tools"]
                                        if not tool["name"].startswith("ni_rua_")})
        # Written by the relay's callback thread after the answer was delivered: settle before reading it.
        self.assertTrue(session.model_proxy.settle(), "the relay's records did not settle")
        self.assertGreaterEqual(store.read_state("runtime", "model_usage_relay")["calls"], 2)

        # 2. A steer during a gated model call joins the running turn; the turn completes once, with its answer.
        steered = session.run_turn("GATE-S: wait for the gate.")
        self.assertTrue(router.gate_steer.wait_entered(60), "gated model call never arrived")
        self.assertEqual(session.steer("STEER-NOW: answer the steer."), steered)
        router.gate_steer.release()
        self.assertTrue(session.turn_done.wait(120), "steered turn did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        self.assertEqual(session.completed_final_answer(), "FINAL-STEER")
        self.assertTrue(any("STEER-NOW" in oc_smoke_last_user(request.json)[0] for request in router.main))
        steered_rows = [row for row in store.states(store.TRANSCRIPT_KIND)
                        if row.get("turn_id") == steered and row["type"] == "userMessage"]
        self.assertEqual([row.get("steer", False) for row in steered_rows], [False, True])
        self.assertTrue(all(row.get("native_id") for row in steered_rows), steered_rows)

        # 3. An interrupt during a gated model call ends the turn interrupted; the next turn reopens the relay thread.
        session.run_turn("GATE-A: wait for the gate.")
        self.assertTrue(router.gate_abort.wait_entered(60), "gated model call never arrived")
        started = time.monotonic()
        session.interrupt()
        self.assertTrue(session.turn_done.wait(45), "interrupt did not end the turn")
        self.assertLess(time.monotonic() - started, 45)
        self.assertEqual(session.state, "interrupted", session.turn_result)
        session.run_turn("PLAIN: answer directly.")
        self.assertTrue(session.turn_done.wait(120), "turn after the interrupt did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        self.assertEqual(session.completed_final_answer(), "FINAL-2")
        self.assertTrue(store.events_by_type(["model.model_thread_reopened"]))
        self.assertEqual([event["payload"]["status"] for event in store.events_by_type(["agent.turn_completed"])],
                         ["completed", "completed", "interrupted", "completed"])
        self.assertEqual(disconnects, [])

        # 4. Close while the native shell tool runs (its process is detached into its own group) leaves no process.
        session.run_turn("SHELL: wait in the shell.")
        server_pgid = os.getpgid(process.pid)

        def detached_shell():
            members = process_tree_snapshot(process.pid).members
            return [item for item in members if item.pgid != server_pgid] or None

        self.wait(detached_shell, 60, "the shell tool's process never started")
        members = [item.pid for item in process_tree_snapshot(process.pid).members]
        report = session.close()
        self.assertEqual(session.state, "closed")
        self.assertFalse(report["clean_shutdown"])        # a turn was running at close
        self.assertTrue(report["backend"]["agent_processes"]["confirmed"], report)
        for member in members:
            with self.assertRaises(ProcessLookupError, msg=f"pid {member} survived close"):
                os.kill(member, 0)
        self.assertEqual(session.owned_processes(), [])
        self.assertEqual(disconnects, [])
        self.assertFalse(store.read_state("runtime", "agent")["clean_shutdown"])
        commands = [record["message"]["command"] for record in map(json.loads, (root / "agent-raw.jsonl").read_text()
                    .splitlines()) if record["channel"] == "control"]
        self.assertEqual(commands, ["submit_prompt", "submit_prompt", "steer", "submit_prompt", "interrupt",
                                    "submit_prompt", "submit_prompt"])
        self.assertNotIn(HOST_KEY, (root / "private" / "agent.log").read_text(errors="replace"))


@unittest.skipIf(OC_SMOKE_PROBLEM is not None, f"OpenCode runtime unavailable: {OC_SMOKE_PROBLEM}")
@unittest.skipIf(shutil.which("perl") is None, "perl is not installed")
class OpenCodeDetachedChildren(unittest.TestCase):
    """Conformance item 15 (b) with the real server: a shell call leaves ``cmd &`` and a setsid child behind (their
    shell exits at once, both are re-parented away from the server's tree); close kills both and reports a clean,
    confirmed teardown only after the marker re-scan verified it."""

    def test_close_kills_children_a_shell_call_left_behind(self):
        from ni_rua.cortex.agents.opencode.session import OpenCodeSession
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        server = FakeAnthropicServer([], default=OcSmokeRouter(), gate_timeout_s=60).start()
        self.addCleanup(server.close)
        environment = mock.patch.dict(os.environ, {OC_SMOKE_KEY_ENV: HOST_KEY})
        environment.start()
        self.addCleanup(environment.stop)
        store = Store(root, OC_SMOKE_IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({
            "agent": {"backend": "opencode", "opencode": {"executable": OC_SMOKE_EXECUTABLE, "upstream": {
                "wire_api": "anthropic_messages", "base_url": server.base_url, "api_key_env": OC_SMOKE_KEY_ENV,
                "auth_scheme": "x-api-key", "provider": "fake"}}},
            "model": {"name": "claude-smoke-model", "effort": "high"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        session = factory.create_session(config, root, store, lambda params: {"success": True, "contentItems": []})
        self.assertIsInstance(session, OpenCodeSession)
        self.addCleanup(session.close)
        session.start(OC_SMOKE_TOOLS, "Robot instructions. " + OC_SMOKE_MARKER)
        session.run_turn("DETACH: leave two sleepers.")
        self.assertTrue(session.turn_done.wait(120), "the shell turn did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        actor = session.actor_dir
        children = {name: int((actor / f"{name}.pid").read_text().strip()) for name in ("orphan", "detached")}
        for pid in children.values():
            self.addCleanup(lambda pid=pid: _kill_quietly(pid))
        self.assertTrue(all(_alive(pid) for pid in children.values()), children)
        report = session.close()
        survivors = {name: pid for name, pid in children.items() if _alive(pid)}
        self.assertEqual(survivors, {}, f"shell children survived close: {report['backend']}")
        self.assertTrue(report["backend"]["agent_processes"]["confirmed"], report["backend"])
        self.assertTrue(report["clean_shutdown"], report)


def _alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def _kill_quietly(pid):
    try:
        os.kill(pid, 9)
    except OSError:
        pass

# ==== tests/test_agent_opencode_session_unit.py ====

IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
KEY_ENV = "NI_RUA_TEST_OPENCODE_UPSTREAM_KEY"
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
TOOLS = [{"type": "function", "name": "look", "description": "Read-only view.", "inputSchema": SCHEMA},
         {"type": "function", "name": "move", "description": "Stateful motion.", "inputSchema": SCHEMA}]
PROVIDER_400 = {"type": "provider.invalid-request", "message": "provider text that must not be stored", "status": 400}
MODEL = "claude-test-model"

SERVE_MODES = {
    "sleep": "time.sleep(3600)",
    "exit": "sys.exit(3)",
    # A shell tool call that leaves `cmd &` and a setsid child behind and returns at once: the shell exits, both are
    # re-parented away from the server tree before any refresh can see them; their pids land in the cwd (actor dir).
    "orphans": ("subprocess.run(['/bin/sh', '-c', 'sleep 3600 >/dev/null 2>&1 & echo $! > orphan.pid; '"
                " + sys.executable + ' -c \"import os, time; os.setsid(); time.sleep(3600)\" >/dev/null 2>&1 &"
                " echo $! > detached.pid'], start_new_session=True)\ntime.sleep(3600)"),
    # Like OpenCode's shell tool: a child in its own session and process group, then the server keeps running.
    "detached": ("subprocess.Popen([sys.executable, '-c', 'import time; time.sleep(3600)'], start_new_session=True,"
                 " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)\ntime.sleep(3600)"),
}


def fake_binary(directory, *, version="2.2.7", serve="sleep"):
    """An executable standing in for the OpenCode platform binary (``--version`` prints ``opencode v<version>`` as
    2.x does; ``serve`` per ``SERVE_MODES``)."""
    path = Path(directory) / f"opencode-{version}-{serve}"
    path.write_text(f"#!{sys.executable}\nimport subprocess, sys, time\n"
                    f"if sys.argv[1:2] == ['--version']:\n    print('opencode v' + {version!r})\n    sys.exit(0)\n"
                    f"{SERVE_MODES[serve]}\n")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return str(path)


def relay_post(session, thread):
    """POST one Anthropic Messages generation request to the session's relay as OpenCode would (session header
    ``x-opencode-session-id``); returns ``(status, body)``. Used only where the relay must refuse before
    forwarding (the upstream of these tests is a closed port)."""
    import http.client
    from urllib.parse import urlsplit
    url = urlsplit(session.model_proxy.base_url)
    body = json.dumps({"model": MODEL, "max_tokens": 8, "stream": True, "tools": [
        {"name": "ni_rua_look", "description": "d", "input_schema": {"type": "object"}}],
        "messages": [{"role": "user", "content": "hi"}]}).encode()
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=10)
    try:
        connection.request("POST", "/v1/messages", body=body, headers={
            "Content-Type": "application/json", "x-opencode-session-id": thread, "x-api-key": "ni-rua-local-relay",
            "anthropic-version": "2023-06-01"})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def alive(pid):
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    return True


def served_models(document):
    """``GET /api/model`` entries 2.2.7 reports for the relay models of a generated config document."""
    provider = document["providers"]["relay"]
    return [model_info(model_id, package=provider["package"], variants=entry["variants"],
                       input_modalities=tuple(entry["capabilities"]["input"]), context=entry["limit"]["context"],
                       output=entry["limit"]["output"], base_url=provider["settings"]["baseURL"])
            for model_id, entry in provider["models"].items()]


class ScriptedFake(FakeOpenCodeServer):
    """The support fake whose ``GET /api/model`` serves the models declared in ``config_path`` (the session's
    generated ``opencode.json``): re-read on every request (OpenCode's hot reload) or, with ``hot_reload=False``, only
    at the first request and at ``POST /api/location/reload``. ``models_override`` (a list) is served instead."""
    config_path = None
    hot_reload = True
    models_override = None
    _loaded = None

    def _read_config(self):
        self._loaded = json.loads(Path(self.config_path).read_text())

    def _r_models(self, handler, query, body, session_id, rest, outcome):
        with self.lock:
            override = self.models_override
            if override is None and self.config_path is not None:
                if self.hot_reload or self._loaded is None:
                    self._read_config()
                models = served_models(self._loaded)
            else:
                models = list(override if override is not None else self.models)
        return self._json(handler, 200, {"location": {"directory": DIRECTORY}, "data": models})

    def _r_reload(self, handler, query, body, session_id, rest, outcome):
        with self.lock:
            if self.config_path is not None:
                self._read_config()
        return super()._r_reload(handler, query, body, session_id, rest, outcome)


class SessionCase(unittest.TestCase):
    maxDiff = None
    client_timeout = 2.0

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.dir = Path(directory.name).resolve()
        self.root = self.dir / "run"
        self.root.mkdir()
        self.store = Store(self.root, IDENTITY)
        self.addCleanup(self.store.close)
        environment = mock.patch.dict(os.environ, {KEY_ENV: "host-secret"})
        environment.start()
        self.addCleanup(environment.stop)
        self.fake = ScriptedFake().start()
        self.addCleanup(self.fake.close)
        self.executable = fake_binary(self.dir)
        self.handler_calls, self.disconnects, self.resets = [], [], []
        self.clients = []

    # ---- helpers --------------------------------------------------------------------------------------------
    def config(self, *, executable=None, effort="high", upstream=None, **opencode):
        return Config.from_dict({
            "agent": {"backend": "opencode", "opencode": {
                "executable": executable or self.executable,
                "upstream": upstream or {"wire_api": "anthropic_messages", "base_url": "http://127.0.0.1:9",
                                         "api_key_env": KEY_ENV, "auth_scheme": "x-api-key", "provider": "fake"},
                **opencode}},
            "model": {"name": MODEL, "effort": effort},
            "features": {"self_briefing": False, "visual_review_model": None}})

    def client_factory(self, base_url, password, unix_socket=None):
        self.clients.append((base_url, password))
        self.fake.config_path = self.session.launch.config_path      # written just before the client is built
        return OpenCodeClient(self.fake.url, PASSWORD, timeout_s=self.client_timeout)

    def tool_handler(self, params):
        self.handler_calls.append(params)
        return {"success": True, "contentItems": [{"type": "inputText", "text": "ok " + params["tool"]}]}

    def make(self, config=None, *, launch_options=None, client_factory=None, **overrides):
        # The --version probe of the stand-in binary is exercised by the version tests only (it costs a process).
        options = dict(ripgrep="/bin/echo", managed_config_paths=[], check_version=False)
        options.update(launch_options or {})
        session = OpenCodeSession(self.root, self.store, self.tool_handler, config=config or self.config(),
                                  client_factory=client_factory or self.client_factory, launch_options=options)
        session.on_disconnect = lambda: self.disconnects.append(session.state)
        session.on_context_reset = self.resets.append
        session.tool_parallel_hints = {"look": True, "move": False}
        session.tool_parallel_safe = lambda tool, arguments: tool == "look"
        session.watchdog_s = 30.0
        for key, value in overrides.items():
            setattr(session, key, value)
        if "incomplete_gap_s" in overrides:
            session._guard.min_gap_s = overrides["incomplete_gap_s"]
        self.session = session
        self.addCleanup(session.close)
        return session

    def started(self, resume=None, *, launch_options=None, **overrides):
        session = self.make(launch_options=launch_options, **overrides)
        session.start(TOOLS, "Be careful.", resume)
        return session

    def events(self, kind):
        return self.store.events_by_type([kind], limit=1000)

    def raw(self, channel):
        path = self.root / "agent-raw.jsonl"
        records = [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []
        return [record for record in records if record["channel"] == channel]

    def rows(self, kind=None, thread=None):
        return [row for row in self.store.states(self.store.TRANSCRIPT_KIND)
                if (kind is None or row["type"] == kind) and (thread is None or row.get("thread_id") == thread)]

    def requests(self, suffix, method="POST"):
        with self.fake.lock:
            return [item for item in self.fake.requests if item["method"] == method and item["path"].endswith(suffix)]

    def prompts(self):
        return self.requests("/prompt")

    def interrupts(self):
        return [item["path"] for item in self.requests("/interrupt")]

    def finish(self, session, timeout=5.0):
        self.assertTrue(session.turn_done.wait(timeout), "turn did not end")
        return session.state

    def running_turn(self, session, text="Do it."):
        turn = session.run_turn(text)
        self.assertTrue(wait_until(lambda: session.thread_id in self.fake.active), "prompt never admitted")
        return turn

    def mcp(self, session):
        client = FakeMcpClient(session.bridge.url, headers=session.bridge.headers)
        client.handshake()
        return client

    def generated(self, session):
        return json.loads(session.launch.config_path.read_text())

    def wake(self, sid, text="<shell id=\"bg\" state=\"completed\">BG-DONE</shell>"):
        """A background shell's completion report: a ``synthetic`` inbox item that starts the idle main session."""
        self.fake.enqueue(sid, "synthetic", {"text": text, "description": "background shell"})
        if self.fake.start_execution(sid):
            self.fake.deliver(sid)


class StartTests(SessionCase):
    def test_fresh_start_writes_identity_config_and_threads(self):
        session = self.started(launch_options=dict(check_version=True))
        self.assertEqual(session.state, "idle")
        thread = session.thread_id
        self.assertIn(thread, self.fake.sessions)
        identity = self.store.read_state("runtime", "agent")
        self.assertEqual((identity["backend"], identity["thread_id"], identity["cli_version"], identity["wire_api"],
                          identity["counting"], identity["clean_shutdown"], identity["resumed"], identity["provider"]),
                         ("opencode", thread, "2.2.7", "anthropic_messages",
                          "delivered_usable_generation_completions_v1", False, False, "fake"))
        self.assertEqual(identity["pid"], session.process.pid)
        (role, process), = session.owned_processes()
        self.assertEqual((role, process.poll()), ("opencode", None))
        created, = [item for item in self.fake.requests if item["method"] == "POST" and item["path"] == "/api/session"]
        self.assertEqual(created["body"], {"title": "ni-rua-actor", "model": {
            "id": MODEL, "providerID": "relay", "variant": "high"}})
        paths = [item["path"] for item in self.fake.requests]
        # Start order (design 11 §5 OC2-D): ready -> routes -> served models -> MCP connected -> session -> history.
        self.assertEqual(paths[:6], ["/api/info", "/openapi.json", "/api/model", "/api/mcp", "/api/session",
                                     f"/api/session/{thread}/message"])
        # Launch: the generated config file names the relay and the MCP endpoint; read-only instructions file.
        config = self.generated(session)
        self.assertEqual(config, session.launch.config)
        provider = config["providers"]["relay"]
        self.assertEqual(provider["settings"]["baseURL"], session.model_proxy.base_url + "/v1")
        self.assertEqual(config["mcp"]["servers"]["ni_rua"]["url"], session.bridge.url)
        self.assertEqual(config["model"], f"relay/{MODEL}")
        self.assertEqual(list(provider["models"]), [MODEL])
        text = session.launch.instructions_path.read_text()
        self.assertEqual(text, "Be careful.\n\n" + naming_note() + "\n")
        self.assertEqual(stat.S_IMODE(session.launch.instructions_path.stat().st_mode), 0o444)
        launch = session.launch
        self.assertNotIn("host-secret", json.dumps(launch.env))
        self.assertNotIn("host-secret", session.launch.config_path.read_text())
        self.assertEqual(launch.argv[1:4], ("serve", "--hostname", "127.0.0.1"))
        self.assertEqual(self.clients[0][0], "http://127.0.0.1:" + launch.argv[-1])
        self.assertEqual(launch.env["OPENCODE_PASSWORD"], self.clients[0][1])
        self.assertEqual(sorted(thread.name for thread in session._threads),
                         ["opencode-events", "opencode-reconcile", "opencode-tree", "opencode-watch"])
        registered = self.store.events_by_type(["model.model_thread_registered"])
        self.assertEqual([event["payload"]["native_thread_id"] for event in registered], [thread])
        self.assertEqual(session.model_proxy.default_thread_id, thread)
        self.assertTrue(self.fake.wait_subscribers(1))

    def test_mcp_pending_is_awaited_before_the_session(self):
        self.fake.mcp_pending_polls = 4
        session = self.started()
        self.assertEqual(session.state, "idle")
        paths = [item["path"] for item in self.fake.requests]
        self.assertEqual(paths.count("/api/mcp"), 5)
        self.assertLess(len(paths) - 1 - paths[::-1].index("/api/mcp"), paths.index("/api/session"))

    def test_provenance_records_runtime_routes_effort_limits_and_declarations(self):
        session = self.started()
        block = session.provenance()
        self.assertEqual((block["kind"], block["cli"]["version"], block["wire_api"], block["native_model_id"],
                          block["counting"], block["isolation"], block["context_delivery"], block["subagents"]),
                         ("opencode", "2.2.7", "anthropic_messages", MODEL,
                          "delivered_usable_generation_completions_v1", "disabled", "delta", True))
        self.assertIsNone(block["native_tools"]["enabled"])          # known once the first main request was relayed
        self.assertEqual({item["name"] for item in block["native_tools"]["disabled"]},
                         {"webfetch", "websearch", "question"})
        self.assertEqual(len(block["route_list_sha256"]), 64)
        self.assertEqual(block["native_effort"], dict(
            variant="high", body={"thinking": {"type": "adaptive"}, "output_config": {"effort": "high"}},
            thinking_display=None, declared_variants=["low", "medium", "high", "xhigh", "max"],
            wire_field="output_config.effort"))
        limits = dict(context=200000, output=32000, source=dict(context="fallback", output="fallback"))
        self.assertEqual(block["effective_limits"], limits)
        self.assertEqual(block["declared_models"], [dict(id=MODEL, limits=limits, input=["text", "image"])])
        self.assertEqual(block["prompt_channel"]["instructions_channel"], "config_instructions_file")
        self.assertEqual(block["prompt_channel"]["tool_name_prefix"], "ni_rua_")
        self.assertEqual(block["tool_transport"]["read_only_tools"], ["look"])
        self.assertNotIn(PASSWORD, json.dumps(block))
        self.assertNotIn(self.clients[0][1], json.dumps(block))
        self.assertTrue(block["capabilities"]["steer"] and block["capabilities"]["native_wakeup"])
        unstarted = OpenCodeSession(self.root, self.store, None, config=self.config()).provenance()
        self.assertEqual((unstarted["kind"], unstarted["wire_api"]), ("opencode", "anthropic_messages"))

    def test_native_tools_come_from_the_first_main_request_the_relay_forwarded(self):
        session = self.started()
        sid = session.thread_id
        names = [{"name": name} for name in ("shell", "ni_rua_look", "read", "subagent")]
        session._note_request_tools({"tools": names[:2]}, dict(logical_call_id="t:1", native_thread_id=sid))
        session._model_event(dict(kind="model_request_started", logical_call_id="t:1", call_purpose="subagent",
                                  native_thread_id=sid))
        self.assertIsNone(session.provenance()["native_tools"]["enabled"])     # a subagent's list never counts
        session._note_request_tools({"tools": names}, dict(logical_call_id="t:2", native_thread_id=sid))
        session._note_request_tools({"messages": []}, dict(logical_call_id="t:3", native_thread_id=sid))
        session._model_event(dict(kind="model_request_started", logical_call_id="t:2", call_purpose="main",
                                  native_thread_id=sid))
        self.assertEqual(session.provenance()["native_tools"]["enabled"], ["read", "shell", "subagent"])
        session._note_request_tools({"tools": [{"name": "other"}]}, dict(logical_call_id="t:4", native_thread_id=sid))
        self.assertEqual(session._call_tools, {})
        session._model_event(dict(kind="model_call_completed", logical_call_id="t:4", call_purpose="main",
                                  native_thread_id=sid))
        recorded, = self.events("agent.native_tools")                  # the manifest predates it: recorded once
        self.assertEqual(recorded["payload"], dict(enabled=["read", "shell", "subagent"], registry=["look"],
                                                   source="first_main_request"))

    def test_start_twice_is_refused(self):
        session = self.started()
        with self.assertRaisesRegex(ValueError, "^session_already_started$"):
            session.start(TOOLS, "again")

    def assert_start_fails(self, session, error, pattern):
        with self.assertRaisesRegex(error, pattern):
            session.start(TOOLS, "Be careful.")
        self.assertEqual(session.state, "closed")
        self.assertEqual(session.owned_processes(), [])
        if session.process is not None:
            self.assertTrue(wait_until(lambda: session.process.poll() is not None))
        self.assertIsNone(self.store.read_state("runtime", "agent"))
        self.assertEqual(self.disconnects, [])

    def test_runtime_and_launch_failures_close_everything(self):
        cases = (
            (dict(config=self.config(executable=fake_binary(self.dir, version="2.2.6")),
                  launch_options=dict(check_version=True)), VersionMismatch, "^opencode_version_mismatch"),
            (dict(launch_options=dict(managed_config_paths=[str(self.dir)])), OpenCodeConfigError,
             "^opencode_managed_config_present$"),
            (dict(launch_options=dict(ripgrep="/nonexistent/rg")), OpenCodeConfigError, "^opencode_ripgrep_missing$"),
            # The readiness check keeps failing (nothing listens) until the process has exited.
            (dict(config=self.config(executable=fake_binary(self.dir, serve="exit")), startup_timeout_s=20.0,
                  client_factory=lambda url, password: OpenCodeClient("http://127.0.0.1:9", PASSWORD, timeout_s=0.2)),
             RuntimeError, "^opencode_exited_during_startup:3$"),
        )
        for options, error, pattern in cases:
            with self.subTest(pattern=pattern):
                self.assert_start_fails(self.make(**options), error, pattern)

    def test_server_checks_kill_the_server(self):
        for setup, error, pattern in (
                (lambda: setattr(self.fake, "version", "2.2.6"), VersionMismatch, "^opencode_version_mismatch"),
                (lambda: setattr(self.fake, "routes", self.fake.routes[:-1]), OpenCodeConfigError,
                 "^opencode_route_list_mismatch$"),
                (lambda: setattr(self.fake, "models_override", [model_info(MODEL)]), OpenCodeConfigError,
                 "^opencode_served_models_mismatch$"),
                (lambda: setattr(self.fake, "mcp_servers", [{"name": "ni_rua", "status": {
                    "status": "failed", "error": "connection refused"}}]), RuntimeError, "^opencode_mcp_failed$")):
            with self.subTest(pattern=pattern):
                saved = (self.fake.version, list(self.fake.routes), self.fake.models_override,
                         list(self.fake.mcp_servers))
                setup()
                try:
                    session = self.make()
                    self.assert_start_fails(session, error, pattern)
                    self.assertIsNotNone(session.process)
                finally:
                    (self.fake.version, self.fake.routes, self.fake.models_override, self.fake.mcp_servers) = saved

    def test_startup_and_mcp_timeouts(self):
        closed = OpenCodeClient("http://127.0.0.1:9", PASSWORD, timeout_s=0.2)
        session = self.make(client_factory=lambda url, password: closed, startup_timeout_s=0.3)
        self.assert_start_fails(session, RuntimeError, "^opencode_startup_timeout$")
        self.fake.mcp_pending_polls = 1000
        self.assert_start_fails(self.make(mcp_connect_timeout_s=0.3), RuntimeError, "^opencode_mcp_connect_timeout$")

    def test_configuration_failures_before_any_process(self):
        cases = (
            (self.config(executable="/nonexistent/opencode"), OpenCodeConfigError, "^opencode_executable_invalid$"),
            (Config.from_dict({"agent": {"backend": "opencode"}}), ValueError, "^opencode_executable_not_set"),
            (self.config(upstream={"base_url": "http://127.0.0.1:9", "api_key_env": KEY_ENV}), ValueError,
             "^opencode_upstream_incomplete"),
            (self.config(upstream={"wire_api": "openai_chat", "base_url": "http://127.0.0.1:9", "api_key_env": KEY_ENV}),
             ValueError, "^opencode_upstream_wire_unsupported:openai_chat"),
            (self.config(upstream={"wire_api": "anthropic_messages", "base_url": "http://127.0.0.1:9",
                                   "api_key_env": "NI_RUA_TEST_UNSET_KEY"}), ValueError, "^opencode_upstream_key_missing"),
            (self.config(effort="minimal"), ValueError, "^agent_effort_unsupported:opencode:minimal"),
        )
        for config, error, pattern in cases:
            with self.subTest(pattern=pattern):
                session = self.make(config)
                self.assert_start_fails(session, error, pattern)
                self.assertIsNone(session.process)

    def test_close_before_start_and_in_replay(self):
        session = self.make()
        report = session.close()
        self.assertEqual((session.state, report["clean_shutdown"]), ("closed", False))
        viewer = self.make()
        viewer.enter_replay("ses_recorded")
        viewer.close()
        self.assertEqual((viewer.state, viewer.thread_id, viewer.owned_processes()), ("closed", "ses_recorded", []))


class TurnTests(SessionCase):
    def test_admission_returns_before_completion_and_idle_ends_the_turn(self):
        session = self.started()
        sid = session.thread_id
        turn = session.run_turn("Do the task.")
        self.assertEqual((session.state, session.turn_id), ("running", turn))
        self.assertFalse(session.turn_done.is_set())
        prompt, = self.prompts()
        self.assertEqual(prompt["body"], {"text": "Do the task."})     # no model, variant or system field
        self.assertEqual(self.requests("/model"), [])                   # the session already has the model
        self.fake.finish_turn(sid, "FINAL")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.turn_result, {"id": turn, "status": "completed", "error": None})
        self.assertEqual(session.completed_final_answer(), "FINAL")
        verdict, = self.events("agent.turn_verdict")
        self.assertEqual((verdict["payload"]["status"], verdict["payload"]["reason"]), ("completed", "succeeded"))
        user, = self.rows("userMessage")
        native = [message for message in self.fake.messages_of(sid) if message["type"] == "user"][-1]["id"]
        self.assertEqual((user["turn_id"], user.get("native_id")), (turn, native))
        command, = self.raw("control")
        self.assertEqual(command["message"]["command"], "submit_prompt")
        self.assertEqual(self.disconnects, [])
        # A second turn on the same session.
        second = session.run_turn("Again.")
        self.fake.finish_turn(sid, "SECOND")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.turn_id, session.completed_final_answer()), (second, "SECOND"))

    def test_startup_inline_grounding_attaches_inline_images_and_proves_native_evidence(self):
        import base64
        import hashlib
        from types import SimpleNamespace
        from ni_rua.apps.evaluation.native_evidence import AgentRawEvidence

        session = self.started()
        self.assertTrue(session.supports_turn_images())
        # Text-only model declaration disables inline image grounding cleanly
        orig_models = session._models
        try:
            session._models = [opencode_config.model_spec(session.model, catalogue_model=SimpleNamespace(image_input=False))]
            self.assertFalse(session.supports_turn_images())
        finally:
            session._models = orig_models

        png_b64 = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        img_url = f"data:image/png;base64,{png_b64}"
        turn = session.run_turn(
            "Initial scene grounding prompt.",
            images=({"type": "image", "url": img_url},),
        )
        prompt, = self.prompts()
        self.assertEqual(prompt["body"]["text"], "Initial scene grounding prompt.")
        self.assertEqual(prompt["body"]["files"], [{"uri": img_url}])
        self.fake.finish_turn(session.thread_id, "DONE")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.turn_id, turn)

        # AgentRawEvidence extracts sent_image_sha256 from control submit_prompt input items
        evidence = AgentRawEvidence(
            self.root,
            {"thread_id": session.thread_id, "agent_backend": {"kind": "opencode"}},
            self.store,
        ).load()
        self.assertIn(hashlib.sha256(base64.b64decode(png_b64)).hexdigest(), evidence.sent_image_sha256)

    def test_run_turn_refusals(self):
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("x")
        session.start(TOOLS, "Be careful.")
        self.running_turn(session)
        with self.assertRaisesRegex(ValueError, "^turn_already_active$"):
            session.run_turn("y")
        self.assertEqual(len(self.prompts()), 1)

    def test_prompt_4xx_is_a_definite_rejection_that_restores_the_previous_turn(self):
        session = self.started()
        first = session.run_turn("First.")
        self.fake.finish_turn(session.thread_id, "ONE")
        self.finish(session)
        result = session.turn_result
        self.fake.script("session.prompt", 400)
        with self.assertRaises(AgentRejected) as raised:
            session.run_turn("Refused.")
        self.assertEqual((raised.exception.code, raised.exception.message), (400, "opencode_prompt_rejected"))
        self.assertEqual((session.state, session.turn_id, session.turn_result), ("completed", first, result))
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(session.completed_final_answer(), "ONE")
        rejected, = self.events("agent.turn_start_rejected")
        self.assertEqual(rejected["payload"]["code"], 400)
        withdrawn = [row for row in self.rows("userMessage") if row.get("withdrawn")]
        self.assertEqual([row["content"][0]["text"] for row in withdrawn], ["Refused."])
        self.assertEqual(self.disconnects, [])
        # The session stays usable.
        session.run_turn("Third.")
        self.fake.finish_turn(session.thread_id, "THREE")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "THREE")

    def assert_unknown_admission(self, session, outcome, pattern):
        self.fake.script("session.prompt", outcome)
        with self.assertRaisesRegex(AgentTransportUnknown, pattern):
            session.run_turn("Maybe admitted.")
        self.assertEqual(session.state, "disconnected")
        self.assertEqual(session.turn_result["status"], "failed")
        self.assertTrue(session.turn_result["error"]["message"].startswith("upstream_connection_lost: "))
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(self.disconnects, ["disconnected"])
        self.assertEqual(len(self.prompts()), 1)          # never retried
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("again")
        lost, = self.events("agent.connection_lost")
        self.assertFalse(lost["payload"]["history_complete"])

    def test_prompt_5xx_is_unknown_admission(self):
        self.assert_unknown_admission(self.started(), 500, "^opencode_prompt_unknown:status_500$")

    def test_prompt_timeout_is_unknown_admission(self):
        self.fake.hang_s = 1.5
        self.client_timeout = 0.4
        self.assert_unknown_admission(self.started(), "hang", "^opencode_prompt_unknown:OpenCodeTransportError$")

    def test_dropped_prompt_connection_is_unknown_admission(self):
        self.assert_unknown_admission(self.started(), "drop", "^opencode_prompt_unknown:OpenCodeTransportError$")

    def test_provider_failure_fails_the_turn_without_provider_text(self):
        session = self.started()
        session.run_turn("Go.")
        self.fake.finish_turn(session.thread_id, "", error=PROVIDER_400)
        self.assertEqual(self.finish(session), "failed")
        self.assertEqual(session.turn_result["error"], {"message": "opencode_turn_failed:provider.invalid-request:400"})
        self.assertIsNone(session.completed_final_answer())
        self.assertEqual(self.disconnects, [])
        self.assertNotIn("provider text", json.dumps(self.rows()))
        self.assertNotIn("provider text", json.dumps(self.events("agent.turn_completed")))

    def test_an_execution_that_ends_without_idle_fails_the_turn_as_incomplete(self):
        session = self.started(incomplete_gap_s=0.3)
        sid = session.thread_id
        self.running_turn(session)
        self.fake.assistant(sid, [tool_item("read", {"filePath": "a"})])   # the server dies mid-tool: no idle
        with self.fake.lock:
            self.fake.active.pop(sid)
        session._request_reconcile("test")
        self.assertEqual(self.finish(session, 5.0), "failed")
        self.assertEqual(session.turn_result["error"], {"message": "opencode_turn_incomplete"})
        verdict, = self.events("agent.turn_verdict")
        self.assertEqual(verdict["payload"]["reason"], "incomplete")
        self.assertIn("recheck", verdict["payload"]["signals"])
        self.assertEqual(self.disconnects, [])

    def test_unrelated_session_events_are_ignored(self):
        session = self.started()
        other = self.fake.add_session("other")
        self.fake.start_execution(other)
        self.fake.assistant(other, [text_item("foreign")], finish="stop")
        self.running_turn(session)
        time.sleep(0.2)
        self.assertEqual(session.state, "running")
        self.assertFalse([row for row in self.rows() if row.get("thread_id") == other])


class SteerTests(SessionCase):
    def test_steer_joins_the_running_turn(self):
        session = self.started()
        turn = self.running_turn(session)
        self.assertEqual(session.steer("Also check the left side."), turn)
        first, second = self.prompts()
        self.assertEqual(second["body"], {"text": "Also check the left side."})
        self.assertEqual(len(self.fake.inbox_of(session.thread_id)), 1)     # waits for the next step boundary
        self.fake.finish_turn(session.thread_id, "BOTH")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "BOTH")
        rows = self.rows("userMessage")
        self.assertEqual([(row["turn_id"], row.get("steer", False), bool(row.get("native_id"))) for row in rows],
                         [(turn, False, True), (turn, True, True)])
        self.assertEqual([record["message"]["command"] for record in self.raw("control")], ["submit_prompt", "steer"])
        self.assertEqual(len(self.events("agent.turn_steered")), 1)
        self.assertEqual(self.requests("/model"), [])                      # steers never switch the model

    def test_steer_requires_a_running_turn(self):
        session = self.started()
        with self.assertRaisesRegex(ValueError, "^no_active_turn_to_steer$"):
            session.steer("nobody listens")
        self.assertEqual(self.prompts(), [])

    def test_steer_4xx_is_rejected_and_the_turn_continues_without_it(self):
        session = self.started()
        self.running_turn(session)
        self.fake.script("session.prompt", 400)
        with self.assertRaises(AgentRejected) as raised:
            session.steer("refused text")
        self.assertEqual(raised.exception.code, 400)
        self.assertEqual(session.state, "running")
        withdrawn = [row for row in self.rows("userMessage") if row.get("withdrawn")]
        self.assertEqual([row["content"][0]["text"] for row in withdrawn], ["refused text"])
        self.fake.finish_turn(session.thread_id, "ORIGINAL")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "ORIGINAL")
        self.assertEqual(self.disconnects, [])

    def test_steer_transport_failure_disconnects(self):
        session = self.started()
        self.running_turn(session)
        self.fake.script("session.prompt", 503)
        with self.assertRaisesRegex(AgentTransportUnknown, "^opencode_steer_unknown:status_503$"):
            session.steer("lost")
        self.assertEqual((session.state, self.disconnects), ("disconnected", ["disconnected"]))

    def test_steer_written_while_a_reconcile_holds_the_final_snapshot_keeps_the_turn(self):
        """The reconcile at the end of the first answer fetched its snapshot just before a steer was admitted: the
        boundary re-read under the session lock is the steer's (not in that snapshot), so the turn keeps running for
        its answer. (The test holds the session's projector lock to park that reconcile between its snapshot and its
        verdict.)"""
        session = self.started()
        sid = session.thread_id
        turn = self.running_turn(session)
        served = threading.Event()
        self.fake.after("session.message.list", served.set)
        session._projector_lock.acquire()
        released = False
        try:
            self.fake.finish_turn(sid, "FIRST")               # idle -> reconcile fetches the final snapshot
            self.assertTrue(served.wait(5))
            done = []
            steering = threading.Thread(target=lambda: done.append(session.steer("STEER-LATE")))
            steering.start()                                   # admitted after the snapshot was taken
            self.assertTrue(wait_until(lambda: len(self.prompts()) == 2))
            session._projector_lock.release()
            released = True
            steering.join(5)
            self.assertEqual(done, [turn])
        finally:
            if not released:
                session._projector_lock.release()
        time.sleep(0.3)
        self.assertEqual(session.state, "running")
        self.assertEqual(self.events("agent.turn_completed"), [])
        self.fake.finish_turn(sid, "STEER-ANSWER")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.turn_id, session.completed_final_answer()), (turn, "STEER-ANSWER"))
        self.assertEqual(len(self.events("agent.turn_completed")), 1)
        phases = {row["text"]: row["phase"] for row in self.rows("agentMessage")}
        self.assertEqual(phases, {"FIRST": "commentary", "STEER-ANSWER": "final_answer"})

    def test_a_reconcile_during_a_submission_never_finishes_the_turn(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        with session.lock:
            session._submitting += 1                          # a steer request is on its way
        self.fake.finish_turn(sid, "EARLY")
        time.sleep(0.3)
        self.assertEqual(session.state, "running")
        with session.lock:
            session._submitting -= 1
        session._request_reconcile("test")
        self.assertEqual(self.finish(session), "completed")


class InterruptTests(SessionCase):
    def test_interrupt_ends_the_turn_interrupted(self):
        session = self.started()
        self.running_turn(session)
        with mock.patch.object(session.model_proxy, "cancel_pending", wraps=session.model_proxy.cancel_pending) as cancel:
            session.interrupt()
        cancel.assert_called_once_with("turn_interrupted", threads=(session.thread_id,))
        self.assertEqual(self.finish(session), "interrupted")
        self.assertEqual(session.turn_result["status"], "interrupted")
        self.assertEqual(self.interrupts(), [f"/api/session/{session.thread_id}/interrupt"])
        self.assertEqual([record["message"]["command"] for record in self.raw("control")],
                         ["submit_prompt", "interrupt"])
        self.assertEqual(self.disconnects, [])
        # The next turn reopens the relay thread first and runs normally.
        with mock.patch.object(session.model_proxy, "reopen_thread", wraps=session.model_proxy.reopen_thread) as reopen:
            session.run_turn("Next.")
        reopen.assert_called_once_with(session.thread_id)
        self.fake.finish_turn(session.thread_id, "AFTER")
        self.assertEqual(self.finish(session), "completed")

    def test_interrupt_blocks_the_relay_thread_before_the_native_interrupt(self):
        """cancel_pending (with the session's thread tombstoned even when no model call is in flight) happens BEFORE
        POST .../interrupt, so a model call the agent still makes is denied."""
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        order = []
        real_cancel = session.model_proxy.cancel_pending

        def cancel(reason, **keywords):
            order.append(("cancel_pending", reason, tuple(keywords.get("threads", ())), len(self.interrupts())))
            return real_cancel(reason, **keywords)

        with mock.patch.object(session.model_proxy, "cancel_pending", side_effect=cancel):
            session.interrupt()
        self.assertEqual(order, [("cancel_pending", "turn_interrupted", (sid,), 0)])   # nothing sent before it
        self.assertIn(sid, session.model_proxy.cancelled_threads)
        self.assertEqual(self.finish(session), "interrupted")
        status, _body = relay_post(session, sid)              # denied by the relay, never forwarded
        self.assertEqual(status, 400)
        rejected = lambda: self.store.events_by_type(["model.model_thread_request_rejected"])  # noqa: E731
        self.assertTrue(wait_until(rejected))                 # relay events arrive through its callback worker
        self.assertEqual(rejected()[-1]["payload"]["native_thread_id"], sid)
        # The next turn lifts the tombstone before anything is sent.
        session.run_turn("Next.")
        self.assertNotIn(sid, session.model_proxy.cancelled_threads)
        reopened = lambda: self.store.events_by_type(["model.model_thread_reopened"])  # noqa: E731
        self.assertTrue(wait_until(reopened))
        self.assertEqual([event["payload"]["native_thread_id"] for event in reopened()], [sid])

    def test_interrupt_wins_over_an_answer_that_completed_before_it_landed(self):
        session = self.started()
        self.running_turn(session)
        with session.lock:
            session._submitting += 1                          # hold the verdict until the interrupt is marked
        self.fake.finish_turn(session.thread_id, "RACED")
        session.interrupt()                                   # the main session is idle: a no-op natively
        self.assertTrue(self.events("agent.interrupt_not_accepted"))
        with session.lock:
            session._submitting -= 1
        session._request_reconcile("test")
        self.assertEqual(self.finish(session), "interrupted")

    def test_interrupt_while_idle_sends_nothing(self):
        session = self.started()
        session.interrupt()
        self.assertEqual(self.interrupts(), [])
        self.assertEqual(self.raw("control"), [])

    def test_interrupt_http_failure_disconnects(self):
        session = self.started()
        self.running_turn(session)
        self.fake.script("session.interrupt", 500)
        with self.assertRaisesRegex(AgentTransportUnknown, "^opencode_interrupt_unknown:status_500$"):
            session.interrupt()
        self.assertEqual((session.state, session.turn_result["status"]), ("disconnected", "failed"))
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(self.disconnects, ["disconnected"])

    def test_interrupt_store_failure_still_disconnects_and_steer_decrements_submitting_on_base_exception(self):
        session = self.started()
        self.running_turn(session)

        class SyntheticAbort(BaseException):
            pass

        with mock.patch.object(session.client, "prompt", side_effect=SyntheticAbort("abort")):
            with self.assertRaises(SyntheticAbort):
                session.steer("aborted steer")
        self.assertEqual(session._submitting, 0)

        self.fake.script("session.interrupt", 500)
        orig_event = session.transcript.event
        session.transcript.event = lambda kind, *a, **kw: (
            (_ for _ in ()).throw(RuntimeError("store_closed"))
            if kind == "agent.interrupt_failed"
            else orig_event(kind, *a, **kw)
        )
        with self.assertRaisesRegex(AgentTransportUnknown, "^opencode_interrupt_unknown:status_500$"):
            session.interrupt()
        self.assertEqual((session.state, self.disconnects), ("disconnected", ["disconnected"]))
        self.assertTrue(session.turn_done.is_set())

    def test_steer_is_refused_once_an_interrupt_was_requested(self):
        session = self.started()
        self.running_turn(session)
        self.fake.script("session.interrupt", "hang")         # the turn stays running after the interrupt
        self.fake.hang_s = 0.2
        session.interrupt_timeout_s = 1.0
        with self.assertRaises(AgentTransportUnknown):
            session.interrupt()
        self.assertEqual(session.state, "disconnected")
        with self.assertRaisesRegex(ValueError, "^no_active_turn_to_steer$"):
            session.steer("too late")
        other = self.make()
        other.start(TOOLS, "Be careful.")
        self.running_turn(other)
        with other.lock:
            other._interrupt_requested = True
        before = len(self.prompts())
        with self.assertRaisesRegex(ValueError, "^turn_interrupt_requested"):
            other.steer("too late")
        self.assertEqual(len(self.prompts()), before)


class LifecycleTests(SessionCase):
    def test_server_exit_mid_turn_disconnects_once_and_abandons_open_tools(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        self.fake.assistant(sid, [tool_item("shell", {"command": "sleep 9"})])
        self.assertTrue(wait_until(lambda: self.rows("commandExecution")))
        session.process.kill()
        self.assertEqual(self.finish(session), "disconnected")
        self.assertEqual(session.turn_result["status"], "failed")
        self.assertEqual(session.turn_result["error"]["message"], "upstream_connection_lost: ProcessLookupError")
        self.assertEqual(self.disconnects, ["disconnected"])
        row, = self.rows("commandExecution")
        self.assertEqual((row["status"], row.get("interrupted")), ("failed", True))
        lost, = self.events("agent.connection_lost")
        self.assertTrue(lost["payload"]["detail"].startswith("opencode_exited:"))
        report = session.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertEqual(self.disconnects, ["disconnected"])

    def test_server_exit_while_idle_disconnects(self):
        session = self.started()
        session.process.kill()
        self.assertTrue(wait_until(lambda: session.state == "disconnected"))
        # _disconnected sets the state first, records agent.connection_lost, then calls on_disconnect.
        self.assertTrue(wait_until(lambda: self.disconnects))
        self.assertEqual(self.disconnects, ["disconnected"])
        with self.assertRaisesRegex(ValueError, "^session_not_connected$"):
            session.run_turn("x")

    def test_event_stream_gap_is_reconciled_from_the_api(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        self.fake.blocked["event.subscribe"] = 503
        self.fake.drop_event_streams()
        self.assertTrue(wait_until(lambda: self.events("agent.event_stream_gap")))
        self.fake.finish_turn(sid, "UNSEEN")                  # every event of the finish is lost
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "UNSEEN")
        self.assertEqual(self.disconnects, [])
        time.sleep(0.5)                                       # several failed reconnects: one gap event
        self.assertEqual(len(self.events("agent.event_stream_gap")), 1)
        del self.fake.blocked["event.subscribe"]
        self.assertTrue(self.fake.wait_subscribers(1))
        # The reconnect reconciles again; a second outage is recorded again.
        self.fake.drop_event_streams()
        self.assertTrue(wait_until(lambda: len(self.events("agent.event_stream_gap")) == 2))

    def test_watchdog_reconciles_a_silently_lost_stream(self):
        session = self.started(watchdog_s=0.3)
        sid = session.thread_id
        self.fake.muted = True
        self.running_turn(session)
        # The reconcile requested by the admission saw the session active; nothing else will ask again.
        self.assertTrue(wait_until(lambda: self.fake.request_paths("GET").count("/api/session/active") >= 1))
        self.fake.finish_turn(sid, "QUIET")
        self.assertEqual(self.finish(session, 5.0), "completed")
        self.assertEqual(self.events("agent.turn_verdict")[0]["payload"]["signals"], ["watchdog"])

    def test_reconcile_failures_are_logged_and_retried(self):
        session = self.started(watchdog_s=0.2)
        self.running_turn(session)
        failing = mock.patch.object(session.client, "status", side_effect=OpenCodeTransportError("ConnectionResetError"))
        failing.start()
        try:
            # One reconcile thread: once a pass has failed, a pass that began before the patch (its status read
            # succeeded) is over and cannot read the finished turn below and complete it.
            self.assertTrue(wait_until(lambda: len(self.events("agent.reconcile_failed")) >= 1, 5.0))
            self.fake.finish_turn(session.thread_id, "LATER")
            self.assertTrue(wait_until(lambda: len(self.events("agent.reconcile_failed")) >= 3, 5.0))
            self.assertEqual(session.state, "running")    # the server is alive: not a disconnect
        finally:
            failing.stop()
        self.assertEqual(self.finish(session, 5.0), "completed")
        counts = [event["payload"]["consecutive"] for event in self.events("agent.reconcile_failed")]
        self.assertEqual(counts[:3], [1, 2, 3])
        self.assertEqual(self.disconnects, [])

    def test_three_failed_reconciles_with_a_dead_server_disconnect(self):
        """Backup to the watch thread: the server is gone but its exit was not observed (the watch thread is
        blocked), and the API keeps failing -> disconnected after ``reconcile_failure_limit`` failures."""
        session = self.started(watchdog_s=0.1)
        self.running_turn(session)
        with mock.patch.object(session.client, "status", side_effect=OpenCodeTransportError("ConnectionRefusedError")), \
                mock.patch.object(session.process, "poll", return_value=-9):
            self.assertEqual(self.finish(session, 5.0), "disconnected")
        self.assertEqual(len(self.events("agent.reconcile_failed")), 3)
        self.assertEqual(session.turn_result["error"]["message"], "upstream_connection_lost: ProcessLookupError")
        self.assertEqual(self.disconnects, ["disconnected"])

    def test_close_kills_detached_children_and_records_a_clean_shutdown(self):
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="detached")))
        pid = session.process.pid
        self.assertTrue(wait_until(lambda: len(process_tree_snapshot(pid).members) >= 2, 10.0))
        members = process_tree_snapshot(pid).members
        self.assertTrue(any(item.pgid != os.getpgid(pid) for item in members))   # a detached group
        session.run_turn("x")
        self.fake.finish_turn(session.thread_id, "done")
        self.finish(session)
        report = session.close()
        self.assertTrue(report["clean_shutdown"], report)
        self.assertTrue(report["backend"]["agent_processes"]["confirmed"])
        self.assertEqual(report["backend"]["interrupt"], "completed")   # sent whenever the server lives
        for item in members:
            self.assertFalse(alive(item.pid), f"pid {item.pid} survived close")
        self.assertEqual((session.state, session.owned_processes()), ("closed", []))
        self.assertTrue(self.store.read_state("runtime", "agent")["clean_shutdown"])
        self.assertEqual(session.close(), report)             # idempotent
        self.assertFalse([thread for thread in session._threads if thread.is_alive()])

    def test_close_after_an_unknown_admission_still_interrupts_the_native_turn(self):
        session = self.started()
        sid = session.thread_id
        self.fake.script("session.prompt", "drop")            # admitted by the server, answer lost
        with self.assertRaises(AgentTransportUnknown):
            session.run_turn("Maybe running.")
        self.assertIn(sid, session.model_proxy.cancelled_threads)
        self.assertEqual(relay_post(session, sid)[0], 400)    # the admitted execution's model call is denied
        session.interrupt()                                   # the server lives and the execution runs
        self.assertEqual(self.interrupts(), [f"/api/session/{sid}/interrupt"])
        self.assertEqual(self.raw("control")[-1]["message"]["host_turn"], False)
        report = session.close()
        self.assertEqual(report["backend"]["interrupt"], "completed")
        self.assertFalse(report["clean_shutdown"])
        self.assertEqual(self.disconnects, ["disconnected"])

    def test_close_during_a_turn_interrupts_first_and_is_unclean(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        self.fake.assistant(sid, [tool_item("read", {"filePath": "a"})])
        self.assertTrue(wait_until(lambda: self.rows("mcpToolCall")))
        report = session.close()
        self.assertEqual(report["backend"]["interrupt"], "completed")
        self.assertFalse(report["clean_shutdown"])
        self.assertEqual(self.disconnects, [])
        self.assertIn(f"/api/session/{sid}/interrupt", self.interrupts())
        row, = self.rows("mcpToolCall")
        self.assertEqual(row["status"], "failed")
        self.assertEqual(session.turn_result, None)
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])

    def test_a_close_that_finds_a_session_active_is_unclean(self):
        session = self.started()
        sid = session.thread_id
        session.run_turn("x")
        self.fake.finish_turn(sid, "done")
        self.finish(session)
        with self.fake.lock:                                  # active, but no event says so
            self.fake.active[sid] = {"type": "running"}
        report = session.close()
        self.assertEqual((report["clean_shutdown"], report["native_active_at_close"]), (False, True))
        self.assertEqual(report["backend"]["interrupt"], "completed")

    def test_crash_teardown_kills_detached_children_and_is_verified_by_the_marker(self):
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="detached")),
                               tree_refresh_s=0.1)
        pid = session.process.pid
        self.assertTrue(wait_until(lambda: any(row.pgid != pid for row in session._known_tree.values()), 10.0))
        children = [row.pid for row in session._known_tree.values() if row.pid != pid]
        os.kill(pid, signal.SIGKILL)
        self.assertTrue(wait_until(lambda: session.state == "disconnected", 5.0))
        report = session.close()
        backend = report["backend"]
        self.assertTrue(backend["agent_exited_before_teardown"])
        self.assertTrue(set(children) <= set(backend["agent_processes"]["members"]), backend)
        for child in children:
            self.assertTrue(wait_until(lambda: not alive(child), 5.0), f"detached child {child} survived")
        self.assertTrue(backend["agent_processes"]["confirmed"], backend)
        self.assertNotIn("agent_process_exit_unconfirmed", backend)
        self.assertFalse(report["clean_shutdown"])                    # a crash is never a clean boundary
        self.assertEqual(session.owned_processes(), [])                 # confirmed gone

    def test_a_store_failure_writing_the_verdict_event_does_not_delay_the_terminal(self):
        session = self.started()
        sid = session.thread_id
        real = session.transcript.event

        def event(kind, *args, **kwargs):
            if kind == "agent.turn_verdict":
                raise OSError("store unavailable")
            return real(kind, *args, **kwargs)

        session.transcript.event = event
        session.run_turn("x")
        self.fake.finish_turn(sid, "FINAL")
        self.assertEqual(self.finish(session, 2.0), "completed")
        self.assertEqual(session.completed_final_answer(), "FINAL")
        failures = [e["payload"] for e in self.events("agent.native_handler_error")]
        self.assertIn("turn_verdict", [failure["handler"] for failure in failures])

    def test_a_rejected_turn_after_an_interrupt_keeps_the_relay_blocked(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        session.interrupt()
        self.assertEqual(self.finish(session), "interrupted")
        self.assertIn(sid, session.model_proxy.cancelled_threads)
        self.fake.script("session.prompt", 400)
        with self.assertRaises(AgentRejected):
            session.run_turn("again")
        self.assertEqual(session.state, "interrupted")
        self.assertIn(sid, session.model_proxy.cancelled_threads)   # put back: nothing was admitted
        self.assertEqual(relay_post(session, sid)[0], 400)
        session.run_turn("admitted")                                 # an admitted turn lifts it
        self.assertNotIn(sid, session.model_proxy.cancelled_threads)
        self.fake.finish_turn(sid, "OK")
        self.assertEqual(self.finish(session), "completed")


class ToolAndContextTests(SessionCase):
    def test_responses_wire_and_subagent_context_delivery(self):
        config = self.config(upstream={"wire_api": "openai_responses", "base_url": "http://127.0.0.1:9",
                                       "api_key_env": KEY_ENV}, context_delivery="delta", effort="medium")
        pre_session = self.make(config)
        self.assertTrue(pre_session.capabilities.service_tier)
        self.assertEqual(
            pre_session.provenance()["context_delivery_policy"],
            dict(effective="delta", configured="delta", subagents=True),
        )
        session = self.started(config=config)
        self.assertEqual((session.wire.name, session.wire.client), ("openai_responses", "opencode"))
        self.assertTrue(session.capabilities.service_tier)
        self.assertEqual(self.generated(session)["providers"]["relay"]["package"],
                         "@opencode/ai/providers/openai-compatible/responses")
        # 主链与子智能体隔离后，不再因 subagents=True 强覆写 context_delivery 为 full
        self.assertEqual(session.context_delivery, "delta")
        self.assertEqual(session.provenance()["subagents"], True)
        self.assertEqual(
            session.provenance()["context_delivery_policy"],
            dict(effective="delta", configured="delta", subagents=True),
        )
        self.assertTrue(self.config().agent.opencode.subagents)
        self.assertEqual(self.make(self.config()).context_delivery, "delta")
        self.assertEqual(self.make(self.config(context_delivery="full")).context_delivery, "full")
        alone = self.make(self.config(context_delivery="delta", subagents=False))
        self.assertEqual(alone.context_delivery, "delta")
        self.assertFalse(alone.capabilities.service_tier)
        # The tool catalogue is rendered from config.agent.context_delivery before the session exists.
        self.assertEqual((config.agent.context_delivery, alone.config.agent.context_delivery), ("delta", "delta"))
        alone.start(TOOLS, "Be careful.")
        block = alone.provenance()
        self.assertEqual((block["context_delivery"], block["subagents"]), ("delta", False))
        self.assertEqual(block["context_delivery_policy"], dict(effective="delta", configured="delta", subagents=False))
        self.assertIn({"name": "subagent", "reason": "subagents_disabled"}, block["native_tools"]["disabled"])
        self.assertIn({"action": "subagent", "resource": "*", "effect": "deny"}, self.generated(alone)["permissions"])

    def test_multimodal_run_turn_egress_cache_verbosity_and_host_limits(self):
        """验证 OpenCodeSession 多模态 run_turn、模型识图门控、Egress 缓存、Responses verbosity 与截断阈值。"""
        from ni_rua.cortex.session.egress_cache import EgressProjectionCache

        cfg = self.config(
            upstream={"wire_api": "openai_responses", "base_url": "http://127.0.0.1:9", "api_key_env": KEY_ENV},
            effort="medium",
        ).with_overrides({"model": {"verbosity": "low"}})
        session = self.started(config=cfg)
        self.assertIsInstance(session._egress_cache, EgressProjectionCache)
        self.assertTrue(session.capabilities.turn_images)
        self.assertTrue(session.capabilities.service_tier)
        self.assertTrue(session.supports_turn_images())
        self.assertEqual(session.host_tool_output_truncation_limit(), (16_000_000, 100_000, False))

        # 启动生成的 opencode.json 与动态 add_model 均透传 Responses verbosity
        self.assertEqual(
            self.generated(session)["providers"]["relay"]["models"][MODEL]["settings"],
            {"textVerbosity": "low"},
        )
        session.set_model("gpt-extra")
        self.assertEqual(
            self.generated(session)["providers"]["relay"]["models"]["gpt-extra"]["settings"],
            {"textVerbosity": "low"},
        )

        # Egress 投影走会话级 LRU 缓存
        req = {
            "model": MODEL,
            "tools": [{"type": "function", "name": "ni_rua_look", "description": "Look.", "parameters": {"type": "object"}}],
            "input": [{"type": "message", "role": "user", "content": [{"type": "input_text", "text": "hello"}]}],
        }
        self.assertEqual(session._project_request(req), session._project_request(req))
        self.assertGreaterEqual(session._egress_cache.bucket_size("tools"), 1)

        # 多模态 run_turn 将图片以 files=[{"uri": ...}] 透传并记入 raw_agent 控制日志
        img_uri = "data:image/png;base64,iVBORw0KGgo="
        session.run_turn("Look at initial view.", images=[{"type": "image", "url": img_uri}])
        prompt_req = [item for item in self.fake.requests if item["path"].endswith("/prompt")][-1]
        self.assertEqual(prompt_req["body"], {"text": "Look at initial view.", "files": [{"uri": img_uri}]})
        control = [
            json.loads(line)
            for line in (self.root / "agent-raw.jsonl").read_text().splitlines()
            if json.loads(line).get("channel") == "control"
        ]
        self.assertEqual(control[-1]["message"]["input"][1], {"type": "image", "url": img_uri})
        self.fake.finish_turn(session.thread_id, "DONE")
        self.assertEqual(self.finish(session), "completed")

        # 非法图片 URI 触发确定拒绝（AgentRejected），不断链
        with self.assertRaises(AgentRejected):
            session.run_turn("Bad uri.", images=[{"type": "image", "url": "https://evil.invalid/x.png"}])
        self.assertEqual(self.disconnects, [])

        # 当声明模型标记 image_input=False 时，supports_turn_images() 返回 False
        text_only = opencode_config.ModelSpec(id="text-only", limits=session._models[0].limits, image_input=False)
        with session.lock:
            session._models.append(text_only)
            session.model = "text-only"
        self.assertFalse(session.supports_turn_images())

    def test_codex_home_upstream_resolves_the_codex_credentials(self):
        home = self.dir / "codex-home"
        home.mkdir()
        (home / "config.toml").write_text('model_provider = "lab"\n[model_providers.lab]\n'
                                          'base_url = "http://127.0.0.1:9/v1"\nenv_key = "NI_RUA_TEST_OPENCODE_UPSTREAM_KEY"\n')
        config = Config.from_dict({**self.config().to_dict(), "model": {"name": "gpt-test", "codex_home": str(home)},
                                   "agent": {"backend": "opencode", "opencode": {"executable": self.executable,
                                                                                 "upstream": {"source": "codex_home"}}}})
        session = self.make(config)
        self.assertEqual(session.wire_api, "openai_responses")
        url, headers = session._upstream()
        self.assertEqual((url, headers), ("http://127.0.0.1:9/v1", {"Authorization": "Bearer host-secret"}))
        self.assertEqual((session.wire.name, session.wire.client), ("openai_responses", "opencode"))

    def test_relay_discontinuity_changes_the_epoch_and_resets_the_context(self):
        session = self.started()
        self.assertEqual(session.model_proxy.on_context_discontinuity, session._relay_discontinuity)
        # The relay announces each model step's tool calls to the bridge (emitted order of concurrent calls).
        self.assertEqual(session.model_proxy.on_tool_calls, session._model_tool_calls)
        self.assertEqual(session.tool_name_prefix, "ni_rua_")
        session.model_proxy.on_context_discontinuity("history_rewritten")
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((epoch["source"], epoch["signal"]), ("relay_context_discontinuity", "history_rewritten"))
        self.assertEqual(self.resets, ["relay_context_discontinuity"])

    def test_declared_effort_variants_reach_the_config_file(self):
        session = self.started()
        entry = self.generated(session)["providers"]["relay"]["models"][MODEL]
        self.assertEqual(entry["variants"], opencode_config.effort_variants("anthropic_messages"))
        self.assertEqual(entry["limit"], {"context": 200000, "output": 32000})

    def test_registry_calls_carry_the_host_turn_and_stale_calls_are_refused(self):
        session = self.started()
        client = self.mcp(session)
        stale = client.call_tool("look", {"x": 1})
        self.assertTrue(stale.result["isError"])
        self.assertEqual(self.handler_calls, [])
        turn = self.running_turn(session)
        result = client.call_tool("look", {"x": 2})
        self.assertFalse(result.result.get("isError"))
        params, = self.handler_calls
        self.assertEqual((params["threadId"], params["turnId"], params["tool"], params["callIdSource"]),
                         (session.thread_id, turn, "look", "mcp_request"))
        row = self.store.transcript_item(params["callId"])
        self.assertEqual((row["type"], row["status"], row["turn_id"]), ("dynamicToolCall", "completed", turn))

    def test_effort_is_switched_at_the_next_turn_never_by_a_steer(self):
        session = self.started()
        sid = session.thread_id
        self.assertEqual(session.effort_choices, ("low", "medium", "high", "xhigh", "max"))
        session.run_turn("one")
        session.set_effort("max")                             # selected mid-turn: applies from the next turn
        self.assertEqual(session.steer("also this"), session.turn_id)
        self.assertEqual(self.requests("/model"), [])
        self.fake.finish_turn(sid)
        self.finish(session)
        session.run_turn("two")
        switch, = self.requests("/model")
        self.assertEqual(switch["body"], {"model": {"id": MODEL, "providerID": "relay", "variant": "max"}})
        paths = [item["path"] for item in self.fake.requests if item["method"] == "POST"]
        self.assertLess(paths.index(f"/api/session/{sid}/model"), len(paths) - 1)
        self.assertEqual(paths[-1], f"/api/session/{sid}/prompt")         # switched before the prompt
        self.assertEqual(self.fake.info(sid)["model"]["variant"], "max")
        changed, = self.events("agent.model_changed")
        self.assertEqual((changed["payload"]["variant"], changed["payload"]["previous_variant"]), ("max", "high"))
        with self.assertRaisesRegex(ValueError, "^invalid_effort$"):
            session.set_effort("minimal")                     # an openai_responses tier
        native = session.provenance()["native_effort"]
        self.assertEqual((native["variant"], native["body"]["output_config"]["effort"]), ("max", "max"))
        self.fake.finish_turn(sid)
        self.finish(session)
        self.assertTrue(wait_until(lambda: self.events("agent.model_switched")))

    def test_responses_wire_declares_its_own_tiers(self):
        config = self.config(upstream={"wire_api": "openai_responses", "base_url": "http://127.0.0.1:9",
                                       "api_key_env": KEY_ENV}, effort="minimal")
        session = self.make(config)
        self.assertEqual(session.effort_choices, ("none", "minimal", "low", "medium", "high", "xhigh"))
        session.start(TOOLS, "Be careful.")
        entry = self.generated(session)["providers"]["relay"]["models"][MODEL]
        self.assertEqual(entry["variants"], opencode_config.effort_variants("openai_responses"))
        created, = [item for item in self.fake.requests if item["path"] == "/api/session"]
        self.assertEqual(created["body"]["model"]["variant"], "minimal")
        native = session.provenance()["native_effort"]
        self.assertEqual((native["wire_field"], native["body"]), ("reasoning.effort", {"reasoning": {"effort": "minimal"}}))

    def test_native_compaction_signals_once_per_compaction(self):
        # Each compaction reaches the session twice under the same message id, as the event stream's
        # ``session.compaction.started`` and as the ``compaction`` message of a reconcile snapshot, on two threads
        # (either first under load). The first starts the one epoch and names itself in ``signal`` (observer data);
        # the second is a no-op.
        signals = ("compaction_started", "compaction_message")
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        first = self.fake.compact(sid)
        self.assertTrue(wait_until(lambda: self.resets))
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((epoch["source"], epoch["thread_id"]), ("native_context_compaction", sid))
        self.assertIn(epoch["signal"], signals)
        second = self.fake.compact(sid)
        self.assertTrue(wait_until(lambda: len(self.resets) == 2))
        self.fake.finish_turn(sid, "AFTER")
        self.finish(session)       # its reconcile reads both compaction messages once more: no further epoch
        self.assertEqual(self.resets, ["native_context_compaction"] * 2)
        compactions = [event["payload"] for event in self.events("agent.compaction")]
        self.assertEqual([payload["item_id"] for payload in compactions], [first["id"], second["id"]])
        self.assertEqual([(payload["source"], payload["signal"] in signals) for payload in compactions],
                         [("native", True)] * 2, compactions)
        rows = self.rows("contextCompaction")
        self.assertEqual(sorted(row["id"] for row in rows), sorted([first["id"], second["id"]]))
        self.assertEqual(session.completed_final_answer(), "AFTER")

    def test_a_compaction_missed_by_the_events_is_signalled_by_the_snapshot(self):
        session = self.started()
        sid = session.thread_id
        self.running_turn(session)
        self.fake.blocked["event.subscribe"] = 503
        self.fake.drop_event_streams()
        compaction = self.fake.compact(sid)
        self.fake.finish_turn(sid, "DONE")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(self.resets, ["native_context_compaction"])
        event, = self.events("agent.compaction")
        self.assertEqual((event["payload"]["signal"], event["payload"]["item_id"]), ("compaction_message",
                                                                                     compaction["id"]))


class ModelSwitchTests(SessionCase):
    """Design 11 §2.4: set_model declares an undeclared model in the config file and switches at the next turn."""

    def test_an_undeclared_model_is_declared_and_switched_at_the_next_turn_never_mid_turn(self):
        session = self.started()
        sid = session.thread_id
        turn = self.running_turn(session)
        session.set_model("glm-5.3")
        document = self.generated(session)
        self.assertEqual(list(document["providers"]["relay"]["models"]), [MODEL, "glm-5.3"])
        self.assertEqual(document["model"], f"relay/{MODEL}")               # the default is unchanged
        self.assertEqual(stat.S_IMODE(session.launch.config_path.parent.stat().st_mode), 0o555)
        declared, = self.events("agent.model_declared")
        self.assertEqual((declared["payload"]["id"], declared["payload"]["method"], declared["payload"]["input"]),
                         ("glm-5.3", "hot_reload", ["text", "image"]))
        self.assertEqual((session.model, session.turn_id, session.state), ("glm-5.3", turn, "running"))
        self.assertEqual(self.requests("/model"), [])                          # never mid-turn
        self.assertEqual(self.fake.info(sid)["model"]["id"], MODEL)
        self.fake.finish_turn(sid, "ON-THE-OLD-MODEL")
        self.finish(session)
        second = session.run_turn("Now on the new model.")
        switch, = self.requests("/model")
        self.assertEqual(switch["body"], {"model": {"id": "glm-5.3", "providerID": "relay", "variant": "high"}})
        self.assertEqual(self.fake.info(sid)["model"]["id"], "glm-5.3")
        self.assertEqual(session.active_turn_model, "glm-5.3")
        self.fake.finish_turn(sid, "ON-THE-NEW-MODEL")
        self.finish(session)
        self.assertEqual(session.completed_final_answer(), "ON-THE-NEW-MODEL")
        changed, = self.events("agent.model_changed")
        self.assertEqual((changed["payload"]["model"], changed["payload"]["previous"],
                          changed["correlation"]["turn_id"]), ("glm-5.3", MODEL, second))
        switched, = self.events("agent.model_switched")
        self.assertEqual(switched["payload"]["model"]["id"], "glm-5.3")
        block = session.provenance()
        self.assertEqual([item["id"] for item in block["declared_models"]], [MODEL, "glm-5.3"])
        self.assertEqual(block["native_model_id"], "glm-5.3")
        # Back to a declared model: no new declaration, one more switch.
        session.set_model(MODEL)
        self.assertEqual(len(self.events("agent.model_declared")), 1)
        session.run_turn("Back.")
        self.assertEqual(len(self.requests("/model")), 2)

    def test_the_provider_listing_gives_limits_and_image_input(self):
        session = self.started()
        session._catalogue = ModelCatalogue("ok", models=(
            CatalogueModel(id="text-only-model", image_input=False, context_window=64000, max_output_tokens=8192),
            CatalogueModel(id="huge-model", context_window=1000000, max_output_tokens=131072)))
        session.set_model("text-only-model")
        session.set_model("huge-model")
        models = self.generated(session)["providers"]["relay"]["models"]
        self.assertEqual((models["text-only-model"]["capabilities"]["input"], models["text-only-model"]["limit"]),
                         (["text"], {"context": 64000, "output": 8192}))
        self.assertEqual(models["huge-model"]["limit"], {"context": 1000000, "output": 32000})
        effective = session.provenance()["effective_limits"]
        self.assertEqual(effective["source"], {"context": "catalogue", "output": "catalogue_capped"})
        self.assertEqual(session.model_options(), [dict(value="text-only-model", name=None, image_input=False),
                                                   dict(value="huge-model", name=None, image_input=None)])

    def test_explicit_limits_apply_to_every_declared_model(self):
        session = self.started(config=self.config(context_limit_tokens=50000, output_limit_tokens=4000))
        session._catalogue = ModelCatalogue("ok", models=(CatalogueModel(id="other", context_window=9,
                                                                          max_output_tokens=9),))
        session.set_model("other")
        models = self.generated(session)["providers"]["relay"]["models"]
        self.assertEqual({model: entry["limit"] for model, entry in models.items()},
                         {MODEL: {"context": 50000, "output": 4000}, "other": {"context": 50000, "output": 4000}})

    def test_a_server_that_reloads_only_on_request_is_reloaded_once(self):
        self.fake.hot_reload = False
        session = self.started(model_declare_timeout_s=0.3)
        session.set_model("glm-5.3")
        declared, = self.events("agent.model_declared")
        self.assertEqual(declared["payload"]["method"], "reload")
        self.assertEqual(len(self.requests("/api/location/reload")), 1)

    def test_a_model_the_server_never_lists_is_refused_and_the_file_restored(self):
        session = self.started(model_declare_timeout_s=0.3)
        before = session.launch.config_path.read_text()
        self.fake.models_override = served_models(session.launch.config)      # the new entry never appears
        with self.assertRaisesRegex(ValueError, "^opencode_model_declare_failed:glm-5.3$"):
            session.set_model("glm-5.3")
        self.assertEqual(session.launch.config_path.read_text(), before)
        self.assertEqual(session.model, MODEL)
        self.assertEqual(len(self.requests("/api/location/reload")), 1)
        failed, = self.events("agent.model_declare_failed")
        self.assertIn("missing:glm-5.3", failed["payload"]["problems"])
        self.assertEqual([item["id"] for item in session.provenance()["declared_models"]], [MODEL])

    def test_before_start_only_the_selection_changes_and_ids_are_validated(self):
        session = self.make()
        session.set_model("picked-before-start")
        with self.assertRaisesRegex(ValueError, "^invalid_model$"):
            session.set_model("bad#variant")
        with self.assertRaisesRegex(ValueError, "^invalid_model$"):
            session.set_model("")
        session.start(TOOLS, "Be careful.")
        self.assertEqual(list(self.generated(session)["providers"]["relay"]["models"]), ["picked-before-start"])

    def test_a_refused_switch_keeps_the_native_model(self):
        session = self.started()
        sid = session.thread_id
        session.set_model("glm-5.3")
        self.fake.script("session.switchModel", 400)
        session.run_turn("go")
        self.assertEqual((session.model, session.active_turn_model), (MODEL, MODEL))
        rejected, = self.events("agent.model_change_rejected")
        self.assertEqual((rejected["payload"]["model"], rejected["payload"]["kept"], rejected["payload"]["code"]),
                         ("glm-5.3", MODEL, 400))
        self.fake.finish_turn(sid, "OLD")
        self.assertEqual(self.finish(session), "completed")

    def test_an_unanswered_switch_is_unknown_and_nothing_is_prompted(self):
        session = self.started()
        session.set_effort("low")
        self.fake.script("session.switchModel", "drop")
        with self.assertRaisesRegex(AgentTransportUnknown, "^opencode_switch_model_unknown:OpenCodeTransportError$"):
            session.run_turn("go")
        self.assertEqual(self.prompts(), [])
        self.assertEqual((session.state, self.disconnects), ("disconnected", ["disconnected"]))


class SubagentTests(SessionCase):
    """Design R4.3: child sessions are followed, their rows hang under the subagent tool row, never final answer."""

    def child_turn(self, session, *, background=False):
        sid = session.thread_id
        turn = self.running_turn(session)
        child = self.fake.add_child(sid)
        status = "completed" if background else "running"
        tool = tool_item("subagent", {"agent": "general", "description": "explore", "prompt": "look around",
                                      **({"background": True} if background else {})},
                         status=status, metadata={"sessionID": child, "status": "running"},
                         content=[text_item("working in the background")] if background else None)
        message = self.fake.assistant(sid, [tool], finish="tool-calls" if background else None)
        self.fake.start_execution(child)
        self.fake.append(child, {"type": "user", "text": "look around"})
        self.fake.assistant(child, [text_item("CHILD-REPORT")], finish="stop")
        return turn, child, message, tool

    def test_child_rows_hang_under_the_subagent_row_and_are_never_final(self):
        session = self.started()
        sid = session.thread_id
        turn, child, message, tool = self.child_turn(session)
        self.fake.idle(child)                                  # a child's idle never ends the turn
        self.fake.finish_tool(sid, message, tool["id"], content=[text_item("CHILD-REPORT")],
                              metadata={"sessionID": child})
        self.fake.complete(sid, message, finish="tool-calls")
        self.fake.finish_turn(sid, "MAIN-FINAL")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "MAIN-FINAL")
        item_id = f"{message['id']}:{tool['id']}"
        texts = self.rows("agentMessage", thread=child)
        self.assertEqual([(row["text"], row["parent_item_id"], row["turn_id"], row["phase"]) for row in texts],
                         [("CHILD-REPORT", item_id, turn, None)])
        child_user, = self.rows("userMessage", thread=child)
        self.assertEqual((child_user["parent_item_id"], child_user.get("source")), (item_id, "native"))
        tool_row = self.store.transcript_item(item_id)
        self.assertEqual((tool_row["type"], tool_row["tool"], tool_row["thread_id"]),
                         ("mcpToolCall", "subagent", sid))
        linked, = self.events("agent.subagent_session")
        self.assertEqual(linked["payload"], {"session_id": child, "parent_session_id": sid})
        self.assertEqual(linked["correlation"]["item_id"], item_id)

    def test_child_rows_are_recovered_from_snapshots_after_an_event_gap(self):
        session = self.started()
        self.fake.blocked["event.subscribe"] = 503
        self.fake.drop_event_streams()
        self.assertTrue(wait_until(lambda: self.events("agent.event_stream_gap")))
        sid = session.thread_id
        turn, child, message, tool = self.child_turn(session)
        self.fake.idle(child)
        self.fake.finish_tool(sid, message, tool["id"], content=[text_item("r")], metadata={"sessionID": child})
        self.fake.complete(sid, message, finish="tool-calls")
        self.fake.finish_turn(sid, "MAIN-FINAL")
        self.assertEqual(self.finish(session), "completed")
        texts = self.rows("agentMessage", thread=child)
        self.assertEqual([(row["text"], row["parent_item_id"]) for row in texts],
                         [("CHILD-REPORT", f"{message['id']}:{tool['id']}")])
        self.assertIn(f"/api/session/{child}/message", self.fake.request_paths("GET"))

    def test_a_background_subagent_holds_the_turn_until_its_report_is_answered(self):
        session = self.started(settle_gap_s=0.2)
        session._settle.min_gap_s = 0.2
        sid = session.thread_id
        turn, child, message, tool = self.child_turn(session, background=True)
        self.fake.finish_turn(sid, "MAIN-BEFORE-CHILD-DONE")    # the main session goes idle; the child still runs
        time.sleep(0.4)
        self.assertEqual((session.state, session.turn_id), ("running", turn))
        # A registry call of the child runs inside the held turn.
        result = self.mcp(session).call_tool("look", {"x": 3})
        self.assertFalse(result.result.get("isError"))
        self.assertEqual(self.handler_calls[-1]["turnId"], turn)
        self.fake.idle(child)
        self.fake.report_to_parent(child, "CHILD-RESULT")       # starts the idle parent again
        self.fake.finish_turn(sid, "MAIN-AFTER-CHILD-REPORT")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.turn_id, session.completed_final_answer()), (turn, "MAIN-AFTER-CHILD-REPORT"))
        self.assertEqual(len(self.events("agent.turn_completed")), 1)
        self.assertEqual([event["payload"]["origin"] for event in self.events("agent.turn_started")], ["host"])
        report, = [row for row in self.rows("userMessage", thread=sid) if row.get("synthetic")]
        self.assertEqual(report["turn_id"], turn)

    def test_execution_started_mid_reconcile_guards_against_stale_idle_snapshot(self):
        session = self.started(settle_gap_s=30.0)
        session._settle.min_gap_s = 30.0
        sid = session.thread_id
        turn, child, _message, _tool = self.child_turn(session, background=True)
        self.fake.finish_turn(sid, "MAIN-BEFORE-CHILD-DONE")
        self.fake.idle(child)
        self.assertTrue(wait_until(lambda: session._settle.key is not None))

        real_messages = session.client.messages
        real_status = session.client.status
        injected = False
        recheck_started = threading.Event()
        allow_recheck = threading.Event()

        def hooked_messages(session_id, **keywords):
            nonlocal injected
            result = real_messages(session_id, **keywords)
            if not injected and session_id == sid:
                injected = True
                session._settle.min_gap_s = 0.0
                seq0 = session._activity_seq
                self.fake.report_to_parent(child, "CHILD-RESULT")
                self.assertTrue(wait_until(lambda: session._activity_seq > seq0))
                self.assertTrue(session._native_busy)
                self.assertIsNone(session._settle.key)
            return result

        def hooked_status():
            if injected and not recheck_started.is_set():
                recheck_started.set()
                self.assertTrue(allow_recheck.wait(5.0))
            return real_status()

        with mock.patch.object(session.client, "messages", side_effect=hooked_messages), \
                mock.patch.object(session.client, "status", side_effect=hooked_status):
            session._request_reconcile("recheck")
            self.assertTrue(recheck_started.wait(5.0))
            self.assertTrue(session._native_busy)
            self.assertIsNone(session._settle.key)
            self.assertEqual((session.state, session.turn_id), ("running", turn))
            self.assertEqual(self.events("agent.turn_completed"), [])
            mcp_res = self.mcp(session).call_tool("look", {"x": 9})
            self.assertFalse(mcp_res.result.get("isError"))
            self.assertEqual(self.handler_calls[-1]["turnId"], turn)
            allow_recheck.set()

        self.fake.finish_turn(sid, "MAIN-AFTER-CHILD-REPORT")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.turn_id, session.completed_final_answer()), (turn, "MAIN-AFTER-CHILD-REPORT"))
        self.assertEqual(len(self.events("agent.turn_completed")), 1)

    def test_interrupt_interrupts_the_main_session_then_its_active_children(self):
        session = self.started()
        sid = session.thread_id
        _turn, child, _message, _tool = self.child_turn(session)
        self.assertTrue(wait_until(lambda: child in session._children and session._children[child]["projector"]))
        order = []
        real_cancel = session.model_proxy.cancel_pending
        with mock.patch.object(session.model_proxy, "cancel_pending", side_effect=lambda reason, **keywords: (
                order.append((tuple(keywords.get("threads", ())), list(self.interrupts()))),
                real_cancel(reason, **keywords))):
            session.interrupt()
        self.assertEqual(order, [((sid, child), [])])         # both threads blocked before any interrupt
        self.assertEqual(self.finish(session), "interrupted")
        self.assertEqual(self.interrupts(), [f"/api/session/{sid}/interrupt", f"/api/session/{child}/interrupt"])
        for thread in (sid, child, "ses_unknownChildAbc"):
            with self.subTest(thread=thread):
                self.assertEqual(relay_post(session, thread)[0], 400)
        # The next turn reopens the main and the child thread (a child can be continued with its sessionID).
        session.run_turn("Resume the exploration.")
        self.assertFalse({sid, child} & set(session.model_proxy.cancelled_threads))


class NativeWakeupTests(SessionCase):
    """Design R4.2 / 11 §1.8: native work that starts while no host turn runs is adopted as a NI-RUA turn."""

    def host_turn(self, session):
        turn = session.run_turn("Start the background job.")
        self.fake.finish_turn(session.thread_id, "HOST-DONE")
        self.assertEqual(self.finish(session), "completed")
        return turn

    def test_a_wakeup_is_adopted_and_a_run_turn_joins_it(self):
        session = self.started()
        sid = session.thread_id
        host = self.host_turn(session)
        self.wake(sid)
        self.assertTrue(wait_until(lambda: session.state == "running" and session.turn_id != host))
        adopted = session.turn_id
        self.assertEqual(session.turn_origin, "native_wakeup")
        result = self.mcp(session).call_tool("look", {"x": 1})       # registry calls run inside the adopted turn
        self.assertFalse(result.result.get("isError"))
        self.assertEqual(self.handler_calls[-1]["turnId"], adopted)
        self.assertEqual(session.run_turn("JOIN-PROMPT"), adopted)
        self.assertEqual(self.prompts()[-1]["body"], {"text": "JOIN-PROMPT"})
        self.fake.finish_turn(sid, "JOINED-ANSWER")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.turn_id, session.completed_final_answer()), (adopted, "JOINED-ANSWER"))
        self.assertEqual([event["payload"]["origin"] for event in self.events("agent.turn_started")],
                         ["host", "native_wakeup"])
        completed = [event["correlation"]["turn_id"] for event in self.events("agent.turn_completed")]
        self.assertEqual(completed, [host, adopted])
        joined, = self.events("agent.turn_joined")
        self.assertEqual(joined["payload"], {"origin": "native_wakeup"})
        report, = [row for row in self.rows("userMessage") if row.get("synthetic")]
        self.assertEqual(report["turn_id"], adopted)

    def test_an_adopted_turn_ends_at_its_own_idle(self):
        session = self.started()
        sid = session.thread_id
        host = self.host_turn(session)
        self.wake(sid)
        self.assertTrue(wait_until(lambda: session.state == "running" and session.turn_id != host))
        adopted = session.turn_id
        self.fake.finish_turn(sid, "WAKE-ANSWER")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual((session.turn_id, session.completed_final_answer()), (adopted, "WAKE-ANSWER"))
        self.assertEqual(self.prompts()[1:], [])

    def test_an_empty_native_wakeup_after_a_filtered_step_preserves_its_error(self):
        session = self.started()
        sid = session.thread_id
        host = self.running_turn(session)
        error = {"type": "provider.content-filter", "message": "private provider text"}
        self.fake.finish_turn(sid, error=error)
        self.assertEqual(self.finish(session), "failed")
        self.fake.start_execution(sid)
        self.assertTrue(wait_until(lambda: session.state == "running" and session.turn_id != host))
        self.fake.idle(sid, "succeeded")
        self.assertEqual(self.finish(session), "failed")
        self.assertEqual(session.turn_result["error"], {"message": "opencode_turn_failed:provider.content-filter"})
        self.assertIsNone(session.completed_final_answer())
        self.assertEqual(len(self.prompts()), 1)

    def test_a_wakeup_missed_by_the_events_is_adopted_by_the_idle_watchdog(self):
        session = self.started(watchdog_s=0.3)
        sid = session.thread_id
        host = self.host_turn(session)
        self.fake.muted = True
        self.wake(sid)
        self.assertTrue(wait_until(lambda: session.state == "running" and session.turn_id != host, 5.0))
        self.fake.finish_turn(sid, "WAKE-ANSWER")
        self.assertEqual(self.finish(session, 5.0), "completed")
        self.assertEqual(session.completed_final_answer(), "WAKE-ANSWER")

    def test_native_work_that_cannot_be_adopted_is_recorded_once(self):
        session = self.started()
        sid = session.thread_id
        self.fake.script("session.prompt", "drop")
        with self.assertRaises(AgentTransportUnknown):
            session.run_turn("Maybe running.")
        for _ in range(2):
            self.fake.push("session.execution.started", {"sessionID": sid}, session_id=sid)
        self.assertTrue(wait_until(lambda: self.events("agent.unsolicited_native_turn")))
        time.sleep(0.1)
        event, = self.events("agent.unsolicited_native_turn")
        # Whichever detector sees the dropped (admitted) prompt's execution first records it: the pushed
        # execution.started event, or a reconcile pass that finds the native work unaccounted (seen on a loaded host).
        self.assertEqual(event["payload"]["host_state"], "disconnected")
        self.assertIn(event["payload"]["signal"], ("execution_started", "reconcile"))
        self.assertEqual([event["payload"]["origin"] for event in self.events("agent.turn_started")], ["host"])


class PermissionTests(SessionCase):
    """Design 11 §1.3: a permission request (none with the generated config) is rejected at once."""

    def ask(self, session):
        sid = session.thread_id
        tool = tool_item("read", {"filePath": ".env"})
        message = self.fake.assistant(sid, [tool])
        request = self.fake.ask_permission(sid, tool_message_id=message["id"], tool_call_id=tool["id"])
        return request

    def test_a_permission_request_is_rejected_and_fails_the_turn(self):
        session = self.started()
        turn = self.running_turn(session)
        request = self.ask(session)
        self.assertEqual(self.finish(session), "failed")
        self.assertEqual(session.turn_result["error"], {"message": "opencode_permission_rejected"})
        reply, = self.requests("/reply")
        self.assertEqual(reply["body"], {"decision": "reject"})
        rejected, = self.events("agent.permission_auto_rejected")
        self.assertEqual((rejected["payload"]["request_id"], rejected["payload"]["action"],
                          rejected["correlation"]["turn_id"]), (request, "read", turn))
        self.assertEqual(self.fake.messages_of(session.thread_id)[-1]["type"], "assistant")   # no idle written
        # The session accepts the next prompt normally.
        session.run_turn("Next.")
        self.fake.finish_turn(session.thread_id, "AFTER")
        self.assertEqual(self.finish(session), "completed")

    def test_a_request_missed_by_the_events_is_found_by_the_watchdog(self):
        session = self.started(watchdog_s=0.3)
        self.running_turn(session)
        self.fake.muted = True
        self.ask(session)
        self.assertEqual(self.finish(session, 5.0), "failed")
        self.assertEqual(session.turn_result["error"], {"message": "opencode_permission_rejected"})
        self.assertTrue(self.requests("/permission", method="GET"))


class ResumeTests(SessionCase):
    def first_run(self, *, clean=True):
        session = self.started()
        session.run_turn("Earlier.")
        self.fake.finish_turn(session.thread_id, "EARLIER")
        self.finish(session)
        thread = session.thread_id
        session.close()
        if not clean:
            identity = self.store.read_state("runtime", "agent")
            self.store.set_state("runtime", "agent", dict(identity, clean_shutdown=False))
        self.disconnects.clear()
        return thread

    def test_clean_resume_adopts_the_history_without_moving_it(self):
        thread = self.first_run()
        rows_before = {row["id"]: row["turn_id"] for row in self.rows()}
        session = self.started(resume=thread)
        self.assertEqual((session.state, session.thread_id), ("idle", thread))
        self.assertEqual({row["id"]: row["turn_id"] for row in self.rows()}, rows_before)
        self.assertEqual(self.resets, [])
        recovered, = self.events("agent.history_recovered")
        self.assertTrue(recovered["payload"]["clean"])
        self.assertTrue(self.store.read_state("runtime", "agent")["resumed"])
        session.run_turn("Now.")
        self.fake.finish_turn(thread, "NOW")
        self.assertEqual(self.finish(session), "completed")
        self.assertEqual(session.completed_final_answer(), "NOW")
        self.assertEqual([item for item in self.fake.requests if item["path"] == "/api/session"][1:], [])
        self.assertEqual(self.requests("/model"), [])          # the native session kept the model and variant

    def test_unclean_resume_invalidates_the_context(self):
        thread = self.first_run(clean=False)
        self.started(resume=thread)
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual(epoch["source"], "agent_resume_after_unclean_exit")
        self.assertEqual(self.resets, ["agent_resume_after_unclean_exit"])

    def test_resume_identity_and_native_session_checks(self):
        thread = self.first_run()
        identity = self.store.read_state("runtime", "agent")
        for key, value in (("model", "other-model"), ("tools_digest", "x"), ("backend", "claude_code"),
                           ("wire_api", "openai_responses"), ("cli_version", "1.19.5"), ("thread_id", "ses_other")):
            with self.subTest(key=key):
                self.store.set_state("runtime", "agent", dict(identity, **{key: value}))
                session = self.make()
                with self.assertRaisesRegex(ValueError, f"^resume_agent_identity_conflict:{key}"):
                    session.start(TOOLS, "Be careful.", thread)
                self.assertIsNone(session.process)
        self.store.set_state("runtime", "agent", identity)
        with self.fake.lock:
            del self.fake.sessions[thread]
        session = self.make()
        with self.assertRaisesRegex(ValueError, "^resume_native_session_missing"):
            session.start(TOOLS, "Be careful.", thread)
        self.assertEqual(session.owned_processes(), [])
        self.store.set_state("runtime", "agent", None)
        with self.assertRaisesRegex(ValueError, "^resume_agent_identity_missing"):
            self.make().start(TOOLS, "Be careful.", thread)


class DetachedChildTests(SessionCase):
    """Conformance item 15 (b) for OpenCode with a fake server: children a shell call left behind (``cmd &``, setsid)
    are re-parented out of the server's tree; close must still kill them and confirm only after verifying."""

    def orphans(self, session):
        actor = session.actor_dir
        self.assertTrue(wait_until(lambda: all((actor / f"{name}.pid").is_file() and (actor / f"{name}.pid")
                                               .read_text().strip() for name in ("orphan", "detached")), 10.0))
        pids = [int((actor / f"{name}.pid").read_text()) for name in ("orphan", "detached")]
        for pid in pids:
            self.addCleanup(lambda pid=pid: os.kill(pid, signal.SIGKILL) if alive(pid) else None)
        self.assertTrue(all(alive(pid) for pid in pids))
        return pids

    def test_close_kills_children_outside_the_server_tree(self):
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="orphans")))
        pids = self.orphans(session)
        session._refresh_tree()
        self.assertFalse(set(pids) & set(session._known_tree), "the orphans should be outside the server tree")
        report = session.close()
        for pid in pids:
            self.assertTrue(wait_until(lambda: not alive(pid), 5.0), f"detached child {pid} survived close")
        backend = report["backend"]
        self.assertTrue(set(pids) <= {item["pid"] for item in backend["agent_processes"]["marker"]["found"]}, backend)
        self.assertTrue(backend["agent_processes"]["confirmed"], backend)
        self.assertTrue(report["clean_shutdown"], report)
        marker = self.store.read_state("runtime", "agent")["instance_id"]
        self.assertEqual(session.launch.env["NI_RUA_AGENT_INSTANCE"], marker)

    def test_an_unverifiable_teardown_is_never_confirmed(self):
        from ni_rua.cortex.agents import isolation
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="orphans")))
        pids = self.orphans(session)
        with mock.patch.object(isolation, "_marker_rows", side_effect=OSError("ps unavailable")):
            report = session.close()
        backend = report["backend"]
        self.assertTrue(backend["agent_process_exit_unconfirmed"], backend)
        self.assertFalse(report["clean_shutdown"])
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])
        self.assertTrue(all(alive(pid) for pid in pids))   # nothing found them; the receipt says so

    def test_resume_stops_what_the_previous_instance_left_running(self):
        session = self.started(config=self.config(executable=fake_binary(self.dir, serve="orphans")))
        thread = session.thread_id
        pids = self.orphans(session)
        session._teardown_confirmed = True
        with mock.patch.object(type(session), "_stop_backend", lambda self: None):   # a gateway that died
            session.close()
        self.assertTrue(all(alive(pid) for pid in pids))
        resumed = self.started(resume=thread)
        for pid in pids:
            self.assertTrue(wait_until(lambda: not alive(pid), 5.0), f"previous instance's child {pid} survived")
        stopped, = self.events("agent.previous_process_stopped")
        self.assertTrue(set(pids) <= set(stopped["payload"]["pids"]))
        self.assertTrue(stopped["payload"]["stop_confirmed"])
        self.assertEqual(resumed.state, "idle")


class TeardownSnapshotTests(unittest.TestCase):
    def test_shared_teardown_snapshot_rules(self):
        """``isolation.teardown_snapshot`` (both sessions reach it through ``isolation.stop_agent_tree``)."""
        from ni_rua.cortex.agents import isolation
        from ni_rua.cortex.agents.claude_code import session as claude_session
        from ni_rua.cortex.agents.opencode import session as opencode_session
        from ni_rua.cortex.agents.isolation import ProcessInfo, teardown_snapshot
        self.assertIs(claude_session.stop_agent_tree, isolation.stop_agent_tree)
        self.assertIs(opencode_session.stop_agent_tree, isolation.stop_agent_tree)
        me, own = os.getpid(), os.getpgrp()
        table = [ProcessInfo(100, me, own), ProcessInfo(101, 100, 100), ProcessInfo(102, 1, 102),
                 ProcessInfo(103, 102, 102), ProcessInfo(200, me, own), ProcessInfo(me, 1, own)]
        known = [ProcessInfo(101, 100, 100), ProcessInfo(102, 101, 102), ProcessInfo(200, 100, own),
                 ProcessInfo(300, 100, 300)]
        for alive_root, expected in ((True, [100, 101, 102, 103]), (False, [101, 102, 103])):
            with self.subTest(root_alive=alive_root):
                members = [m.pid for m in teardown_snapshot(100, alive_root, known, table).members]
                self.assertEqual(members, expected)


class RelayBlockTests(unittest.TestCase):
    """``ModelRequestProxy.cancel_pending(reason, threads=...)``: the reopenable block an interrupting session puts on
    its own threads even when no call is in flight (non-Codex profiles only)."""

    def proxy(self, wire):
        from ni_rua.cortex.relay.server import ModelRequestProxy
        events = []
        proxy = ModelRequestProxy("http://127.0.0.1:9", wire=wire, on_event=events.append).start()
        self.addCleanup(proxy.close)
        return proxy, events

    def test_block_threads_tombstones_only_the_named_threads(self):
        from ni_rua.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire
        proxy, _events = self.proxy(AnthropicMessagesWire(client="opencode"))
        self.assertEqual(proxy.block_threads(["ses_a", "", None, "ses_a"], "turn_interrupted"), ["ses_a"])
        self.assertEqual(proxy.cancelled_threads, {"ses_a": "turn_interrupted"})
        self.assertTrue(proxy.reopen_thread("ses_a"))
        codex, _ = self.proxy(OpenAIResponsesWire())
        self.assertEqual(codex.block_threads(["thread-1"], "x"), [])
        self.assertEqual(codex.cancelled_threads, {})

    def test_threads_are_tombstoned_reopenably_on_the_opencode_profile(self):
        from ni_rua.cortex.session.wire import AnthropicMessagesWire
        proxy, events = self.proxy(AnthropicMessagesWire(client="opencode"))
        proxy.cancel_pending("turn_interrupted", threads=("ses_main", "ses_child", "", None))
        self.assertEqual(proxy.cancelled_threads, {"ses_main": "turn_interrupted", "ses_child": "turn_interrupted"})
        self.assertTrue(wait_until(lambda: len([e for e in events if e["kind"] == "model_thread_cancelled"]) == 2))
        self.assertTrue(all(e["reopenable"] for e in events if e["kind"] == "model_thread_cancelled"))
        proxy.cancel_pending("again", threads=("ses_main",))            # already blocked: kept, not re-recorded
        self.assertEqual(proxy.cancelled_threads["ses_main"], "turn_interrupted")
        self.assertTrue(proxy.reopen_thread("ses_main"))
        self.assertFalse(proxy.reopen_thread("ses_main"))
        proxy.register_thread("ses_child", "actor", "ses_child")         # registering lifts it as well
        self.assertEqual(proxy.cancelled_threads, {})

    def test_the_codex_profile_ignores_threads(self):
        from ni_rua.cortex.session.wire import OpenAIResponsesWire
        proxy, _events = self.proxy(OpenAIResponsesWire())
        proxy.cancel_pending("turn_interrupted", threads=("thread-1",))
        self.assertEqual(proxy.cancelled_threads, {})


class FactoryTests(SessionCase):
    def test_create_session_builds_an_opencode_session_without_starting_anything(self):
        card, planning = {"input_per_million": 1.0}, object()
        session = factory.create_session(self.config(), self.root, self.store, None, price_card=card,
                                         task_planning=planning)
        self.assertIsInstance(session, OpenCodeSession)
        self.assertEqual((session.price_card, session.task_planning, session.model, session.backend_kind,
                          session.state, session.process), (card, planning, MODEL, "opencode", "not_started", None))
        session.enter_replay("ses_recorded")
        session.close()

    def test_requirements_pass_with_the_pinned_binary_and_refuse_others(self):
        import re
        # 在临时目录提供同目录伪 rg，确保无系统 rg 的纯净环境也能稳定验证预检通过路径
        sibling_rg = self.dir / "rg"
        sibling_rg.write_text("#!/bin/sh\nexit 0\n", encoding="utf-8")
        sibling_rg.chmod(0o755)
        self.assertIsNone(factory.backend_requirements(self.config()))
        hint = re.escape(opencode_config.OPENCODE_INSTALL_HINT)
        for executable, reason in ((fake_binary(self.dir, version="1.19.5"), "opencode_version_mismatch"),
                                   ("/nonexistent/opencode", "opencode_executable_invalid"),
                                   ("relative/opencode", "opencode_executable_invalid")):
            with self.subTest(reason=reason):
                if not executable.startswith("/"):
                    self.assertEqual(factory.opencode_runtime_problem(executable), reason)
                    continue
                config = Config.from_dict({**self.config().to_dict(), "agent": {
                    "backend": "opencode", "opencode": {**self.config().agent.opencode.to_dict(),
                                                        "executable": executable}}})
                with self.assertRaisesRegex(ConfigError, f"^agent_backend_unavailable:opencode: {reason}"
                                                         f"(:[^;]*)?; {hint}"):
                    factory.backend_requirements(config)
        self.assertEqual(factory.opencode_runtime_problem(None), "opencode_executable_not_set")

    def test_requirement_order_and_upstream(self):
        quiet = {"self_briefing": False, "visual_review_model": None}
        cases = (
            # features.sandbox_enabled is accepted (R4-8): the configuration checks after it still apply.
            ({"features": {**quiet, "sandbox_enabled": True}, "agent": {"opencode": {}}}, "opencode_upstream_incomplete"),
            ({"features": quiet, "agent": {"opencode": {}}}, "opencode_upstream_incomplete"),
            ({"features": quiet, "agent": {"opencode": {"upstream": {"wire_api": "openai_chat", "base_url": "http://x",
                                                                     "api_key_env": "K"}}}},
             "opencode_upstream_wire_unsupported:openai_chat"),
            ({"features": quiet, "agent": {"opencode": {"upstream": {"source": "codex_home",
                                                                     "wire_api": "anthropic_messages"}}}},
             "opencode_upstream_codex_home_requires_openai_responses"),
            ({"features": quiet, "agent": {"opencode": {"upstream": {"wire_api": "openai_responses",
                                                                     "base_url": "http://x", "api_key_env": "K"}}}},
             "agent_backend_unavailable:opencode"),
        )
        for document, code in cases:
            with self.subTest(code=code), mock.patch.object(factory, "opencode_runtime_problem",
                                                            return_value="opencode_executable_not_set") as probe:
                agent = {"backend": "opencode", **document.pop("agent", {})}
                config = Config.from_dict({**document, "agent": agent})
                with self.assertRaises(ConfigError) as raised:
                    factory.backend_requirements(config)
                self.assertTrue(str(raised.exception).startswith(code), str(raised.exception))
                self.assertEqual(probe.called, code.startswith("agent_backend_unavailable"))
        with mock.patch.object(factory, "opencode_runtime_problem", return_value=None):
            codex_home = Config.from_dict({"features": quiet, "agent": {"backend": "opencode", "opencode": {
                "upstream": {"source": "codex_home"}}}})
            self.assertIsNone(factory.backend_requirements(codex_home))

# ==== tests/test_agent_opencode_terminal_smoke.py ====

OC_TERM_EXECUTABLE = os.environ.get("NI_RUA_OPENCODE_EXECUTABLE")
OC_TERM_PROBLEM = (factory.opencode_runtime_problem(OC_TERM_EXECUTABLE) if OC_TERM_EXECUTABLE else
           "set NI_RUA_OPENCODE_EXECUTABLE to the pinned opencode 2.2.7 platform binary")
if OC_TERM_PROBLEM is None and shutil.which("rg") is None:
    PROBLEM = "ripgrep (rg) is not installed"
OC_TERM_IDENTITY = dict(run_id="tui", episode_id="ep", env_instance_id="env", env_generation=1)
OC_TERM_KEY_ENV = "NI_RUA_TUI_OPENCODE_UPSTREAM_KEY"
OC_TERM_MARKER = "NI-RUA-OPENCODE-TUI-INSTRUCTIONS-9e2b"
FIRST_ANSWER, SECOND_ANSWER = "ZEBRA-FIRST-ANSWER", "OKAPI-SECOND-ANSWER"
OC_TERM_SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x position in m, default 0"}},
          "additionalProperties": False}
OC_TERM_TOOLS = [{"type": "function", "name": "look", "description": "Read-only view of the scene.", "inputSchema": OC_TERM_SCHEMA}]
_ESCAPES = re.compile(rb"\x1b\][\s\S]*?(?:\x07|\x1b\\)|\x1bP[\s\S]*?\x1b\\|\x1b\[[0-?]*[ -/]*[@-~]|\x1b[()][A-Z0-9]"
                      rb"|\x1b[@-_]")


def screen_text(data):
    """The terminal bytes without escape sequences and whitespace (cursor-positioned TUI output joins up)."""
    return re.sub(r"\s+", "", _ESCAPES.sub(b"", data).decode("utf-8", "replace"))


def oc_term_system_text(body):
    system = body.get("system")
    if isinstance(system, list):
        return "\n".join(block.get("text", "") for block in system if isinstance(block, dict))
    return system if isinstance(system, str) else ""


def oc_term_last_user(body):
    for message in reversed(body.get("messages") or []):
        if message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content, False
        texts = [block.get("text", "") for block in content if isinstance(block, dict) and block.get("type") == "text"]
        return "\n".join(texts), any(isinstance(block, dict) and block.get("type") == "tool_result"
                                     for block in content)
    return "", False


class OcTermRouter:
    """Actor requests (tools + the instructions marker): turn 1 calls ``ni_rua_look`` then answers; turn 2 answers.
    Anything else is recorded in ``other`` (none expected: no title agent, and attaching makes no request)."""

    def __init__(self):
        self.main, self.other = [], []

    def __call__(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        if not (body.get("tools") and OC_TERM_MARKER in oc_term_system_text(body)):
            self.other.append(request)
            return text_step("side-ok")
        self.main.append(request)
        text, has_result = oc_term_last_user(body)
        if "SECOND" in text:
            return text_step(SECOND_ANSWER)
        if has_result:
            return text_step(FIRST_ANSWER)
        return tool_step("ni_rua_look", {"x": 1})


def inet_connections(pid):
    """``[(local, remote)]`` of the process's inet sockets (``lsof``), or None when lsof is unavailable."""
    if shutil.which("lsof") is None:
        return None
    result = subprocess.run(["lsof", "-a", "-nP", "-p", str(pid), "-i"], capture_output=True, text=True, timeout=10)
    found = []
    for line in result.stdout.splitlines()[1:]:
        match = re.search(r"(TCP|UDP) (\S+?)(?:->(\S+))?(?: \(\w+\))?$", line)
        if match:
            found.append((match.group(2), match.group(3)))
    return found


@unittest.skipIf(OC_TERM_PROBLEM is not None, f"OpenCode runtime unavailable: {OC_TERM_PROBLEM}")
class OpenCodeTerminalAttachSmoke(unittest.IsolatedAsyncioTestCase):
    async def wait_for_text(self, terminal, needles, timeout=45.0, after=0):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            text = screen_text(terminal.read(after, 1048576)[0])
            if all(needle in text for needle in needles):
                return text
            self.sample_network(terminal)
            await asyncio.sleep(0.25)
        self.fail(f"the attached TUI never showed {needles}: ...{screen_text(terminal.read(after, 1048576)[0])[-600:]}")

    def sample_network(self, terminal):
        if terminal.process is None or terminal.process.poll() is not None:
            return
        connections = inet_connections(terminal.process.pid)
        if connections is not None:
            self.connections.extend(connections)

    async def test_attach_renders_the_live_session_without_model_requests(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve() / "run"
        root.mkdir()
        router = OcTermRouter()
        server = FakeAnthropicServer([], default=router, gate_timeout_s=60).start()
        self.addCleanup(server.close)
        environment = mock.patch.dict(os.environ, {OC_TERM_KEY_ENV: "host-secret-opencode-tui-key"})
        environment.start()
        self.addCleanup(environment.stop)
        store = Store(root, OC_TERM_IDENTITY)
        self.addCleanup(store.close)
        config = Config.from_dict({
            "agent": {"backend": "opencode", "opencode": {"executable": OC_TERM_EXECUTABLE, "upstream": {
                "wire_api": "anthropic_messages", "base_url": server.base_url, "api_key_env": OC_TERM_KEY_ENV,
                "auth_scheme": "x-api-key", "provider": "fake"}}},
            "model": {"name": "claude-tui-model", "effort": "high"},
            "features": {"self_briefing": False, "visual_review_model": None}})
        calls = []

        def handler(params):
            calls.append(params)
            return {"success": True, "contentItems": [{"type": "inputText", "text": "scene: one red cube"}]}

        session = factory.create_session(config, root, store, handler)
        session.tool_parallel_hints = {"look": True}
        self.addCleanup(session.close)
        self.connections = []

        thread = await asyncio.to_thread(session.start, OC_TERM_TOOLS, "Robot instructions. " + OC_TERM_MARKER)
        await asyncio.to_thread(session.run_turn, "Look at the scene, then answer.")
        self.assertTrue(await asyncio.to_thread(session.turn_done.wait, 120), "tool turn did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        self.assertEqual(len(calls), 1)
        requests_before, actor_before = len(server.requests), len(router.main)
        sessions_before = session.client._request("GET", "/api/session")
        messages_before = session.client.messages(thread)

        spec = session.terminal_attach_spec()
        self.assertIsNotNone(spec)
        self.assertNotIn(spec["env"]["OPENCODE_PASSWORD"], " ".join(spec["command"]))
        terminal = Terminal(root, spec["cwd"], command_prefix=spec["command_prefix"], env=spec["env"],
                            command=spec["command"])
        self.addAsyncCleanup(terminal.close)
        await terminal.start(spec["attach_uri"], spec["thread_id"])
        pid = terminal.process.pid

        # (a) The TUI renders the live session: the prompt, the registry tool call made through MCP, the answer.
        await self.wait_for_text(terminal, ["Lookatthescene", "ni_rua_look", FIRST_ANSWER])
        scroll_cursor = terminal.cursor
        self.assertTrue(terminal.scroll(lines=-3))
        self.assertTrue(terminal.scroll(action="page_up"))
        self.assertTrue(terminal.scroll(action="bottom"))
        for _ in range(8):                      # keep it attached a little longer: no request may follow later
            self.sample_network(terminal)
            await asyncio.sleep(0.25)
        self.assertGreater(terminal.cursor, scroll_cursor)
        # (b) Attaching made no model (or title) request, created no session and changed no session record.
        self.assertEqual((len(server.requests), len(router.main), router.other), (requests_before, actor_before, []))
        self.assertEqual(session.client._request("GET", "/api/session"), sessions_before)
        self.assertEqual(session.client.messages(thread), messages_before)
        self.assertIsNone(terminal.process.poll())

        # (a) ... and follows a later turn live.
        cursor = terminal.cursor
        await asyncio.to_thread(session.run_turn, "SECOND: answer directly.")
        self.assertTrue(await asyncio.to_thread(session.turn_done.wait, 120), "second turn did not end")
        self.assertEqual(session.completed_final_answer(), SECOND_ANSWER)
        await self.wait_for_text(terminal, [SECOND_ANSWER], after=cursor)
        self.assertEqual(len(router.main), actor_before + 1)                  # the host's turn, nothing else
        self.assertEqual(router.other, [])
        if shutil.which("lsof") is not None:                                  # (b) loopback to the server only
            self.assertTrue(self.connections, "lsof saw no connection of the attached TUI")
            port = str(session.client.port)
            for local, remote in self.connections:
                self.assertIsNotNone(remote, f"listening or unconnected socket {local}")
                self.assertEqual(remote, f"127.0.0.1:{port}", self.connections)

        # (c) Terminal.close ends the TUI.
        await terminal.close()
        self.assertIsNotNone(terminal.status()["exit_code"])
        with self.assertRaises(ProcessLookupError):
            os.kill(pid, 0)

        # A TUI the gateway failed to close does not outlive the session: close() stops it via the process marker.
        spare = Terminal(root / "spare", spec["cwd"], command_prefix=spec["command_prefix"], env=spec["env"],
                         command=spec["command"])
        self.addAsyncCleanup(spare.close)
        await spare.start(spec["attach_uri"], spec["thread_id"])
        await self.wait_for_text(spare, [SECOND_ANSWER])
        report = await asyncio.to_thread(session.close)
        self.assertTrue(report["clean_shutdown"], report)
        await asyncio.wait_for(asyncio.shield(spare._exit_task), 15)
        self.assertIsNotNone(spare.process.poll())
        self.assertEqual(len(router.main), actor_before + 1)

if __name__ == "__main__":
    unittest.main()
