"""OpenCode binding of the shared backend conformance suite (design 04 §10.2; plan 07 §4.2; design 11 §5 OC2-D), on
both relay wires.

Drives the pinned ``opencode serve`` 2.2.7 named by ``NI_RUA_OPENCODE_EXECUTABLE`` through the real relay
(``ModelRequestProxy`` + the ``opencode`` client profile of ``AnthropicMessagesWire`` / ``OpenAIResponsesWire``;
upstream = ``FakeAnthropicServer`` / ``FakeResponsesServer`` on loopback) and the real ``NiRuaMcpServer``. One small
``TestCase`` per wire (``OpenCodeAnthropicConformance``, ``OpenCodeResponsesConformance``). No real model and no
network: OpenCode's XDG dirs, HOME and TMPDIR are inside the temporary run directory and its only credential is the
relay placeholder. Skipped with the reason unless the variable names the pinned binary (``requires_opencode``) and
``rg`` is installed (OpenCode refuses to start without it).

Run: ``NI_RUA_OPENCODE_EXECUTABLE=~/.local/share/ni-rua/opencode-2.2.7/opencode PYTHONPATH=tests mamba run -n ni-rua
python -m unittest tests.test_agent_opencode_conformance -v``. Item 22 (330 s provider delay) additionally needs
``NI_RUA_SLOW_TESTS=1``.

OpenCode-specific behaviour pinned on top of the shared items (design 11 §1.3-§1.9):

* 9: the relay tombstone of the main session is in place before every ``POST .../interrupt`` (relay block first).
* 10: the definite rejection is a real HTTP 400 of the running server (payload validation of ``POST .../prompt``).
* 17/18/26: the effort variant reaches every generation request - main, compaction and subagent - in the wire's body
  field (``output_config.effort`` / ``reasoning.effort``).
* 18/24: OpenCode 2.x compacts with a tool-declaring request on the main chain (same tools and system) whose last
  user text is its summary instruction; a summary that misses the template fails the turn, so the fake answers with
  the template.
* 20: a steer joins at the next step boundary: it is the newest user input of the request right after the step it
  was sent during, next to that step's tool result.
* 21: after a restart the dangling tool call reaches the model as OpenCode's own ``Tool execution interrupted``
  error result.
* 23: deferred delivery - OpenCode records the relay's verdict as a ``provider.invalid-request`` 400 (not retried):
  the first and only bytes it received were the final error status, never a 200 stream.
* 27: the shared item (a background shell finishing after the turn ended) plus a background subagent: it holds the
  turn while it runs (its registry call runs inside a turn) and its report is answered in a turn; one
  ``agent.turn_completed`` per turn either way.
* ``set_model`` to an undeclared model mid-turn declares it and takes effect at the next turn, never mid-turn; a
  300 kB PNG tool result reaches the provider byte-identical (``media.image.auto_resize=false``).
* 28 (sandbox) runs where bubblewrap namespaces work (Linux): the server is reachable only through the inbound unix
  socket (no host TCP listener on its port) on top of the shared checks.
"""

NI_RUA_TEST_TIER = "component"

import contextlib
import os
import random
import shutil
import struct
import time
import unittest
import zlib
from unittest import mock
from support.agent_conformance import (MODEL, UPSTREAM_KEY_ENV, AgentBackendConformance, AnthropicMessagesView,
                                       INSTRUCTIONS_MARKER, OpenAIResponsesView, TURN_TIMEOUT_S, body_text,
                                       image_item, sha256, text_item)
from support.fake_model_servers import (FakeAnthropicServer, FakeResponsesServer, Gate, Step, ToolCall, Usage,
                                        text_step, tool_step)
from support.requirements import OPENCODE_EXECUTABLE_ENV, requires_opencode
from ni_rua.core.config import Config
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.opencode.client import OpenCodeClient, OpenCodeTransportError
from ni_rua.cortex.agents.opencode.session import OpenCodeSession

EXECUTABLE = os.environ.get(OPENCODE_EXECUTABLE_ENV)
RIPGREP_PROBLEM = None if shutil.which("rg") else "ripgrep (rg) is not installed (OpenCode refuses to start without it)"
SESSION_HEADER = "x-opencode-session-id"     # OpenCode's per-session request header (child requests: the child id)
# The last user text of OpenCode 2.2.7's compaction request (its first form and the retry after a summary that missed
# the template).
COMPACTION_MARKS = ("You MUST summarize the conversation above into a structured summary",
                    "The previous response did not fill in the required summary template")
# A summary in OpenCode 2.2.7's template (a summary without the headings is retried once, then fails the turn).
SUMMARY = ("## Objective\n- Scripted conformance session.\n\n## Requirements\n- (none)\n\n## Decisions\n- (none)\n\n"
           "## Work State\n### Completed\n- scripted tool steps ran\n\n### Active\n- (none)\n\n### Blocked\n- (none)\n\n"
           "## Next Move\n1. continue\n\n## Relevant Files\n(none)\n\n## Important Context\n- (none)\n")
INTERRUPTED_TEXT = "Tool execution interrupted"   # OpenCode's repair of a dangling tool call after a restart
EFFORT = "low"
# Auto-compaction: OpenCode 2.x compacts once a finished step reports tokens above context minus its 15% buffer.
CONTEXT_LIMIT, OUTPUT_LIMIT = 40000, 8000
OTHER_MODEL = "conformance-model-two"


def noise_png(width, seed=7):
    """A ``width`` x ``width`` RGB PNG of random pixels (incompressible: ~3 bytes per pixel)."""
    generator = random.Random(seed)
    rows = b"".join(b"\x00" + generator.randbytes(width * 3) for _ in range(width))

    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, width, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 1)) + chunk(b"IEND", b""))


class ResponsesServer(FakeResponsesServer):
    """``FakeResponsesServer`` that reports ``total_tokens: null`` when a scripted counter is invalid (negative):
    the stock fake would fold the invalid value into a plausible total. The pinned client requires a number for
    ``input_tokens``, so the unknown-counter case of item 16 reports an invalid number instead."""

    def _usage(self, u):
        usage = super()._usage(u)
        if min(u.input_tokens, u.cache_read_input_tokens, u.output_tokens) < 0:
            usage["total_tokens"] = None
        return usage


class RecordingClient(OpenCodeClient):
    """The real client; every ``interrupt`` records whether the relay thread of the interrupted session was
    tombstoned (``ModelRequestProxy.cancelled_threads``) when it was sent."""

    def __init__(self, session, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session, self.interrupts = session, []

    def interrupt(self, session_id, *, timeout_s=None):
        proxy = self.session.model_proxy
        self.interrupts.append(dict(session_id=session_id,
                                    tombstoned=proxy is not None and session_id in proxy.cancelled_threads))
        return super().interrupt(session_id, timeout_s=timeout_s)


@requires_opencode
@unittest.skipIf(RIPGREP_PROBLEM is not None, RIPGREP_PROBLEM)
class _OpenCodeConformance(AgentBackendConformance):
    """Binding shared by both wires; a subclass names the wire (``view``, ``fake_model_class``, ``auth_scheme``)."""
    backend_kind = "opencode"
    auth_scheme = None
    effort_field = None          # body field of the effort variant on this wire
    tool_prefix = "ni_rua_"
    expect_steer = True
    expect_native_wakeup = True
    expect_subagents = True
    forbidden_tool_names = frozenset({"webfetch", "websearch", "codesearch", "question"})
    identity_fields = ("cli_version", "cli_executable", "cli_sha256", "native", "resumed")
    native_call_id_source = None          # OpenCode sends no call id over MCP: callIdSource is "mcp_request"
    concurrent_step_calls = True          # 2.2.7 sends every tool call of a step at once, ordered ones included
    # OpenCode renders the instructions file as "Instructions from: <private path>"; egress keeps the path out.
    private_markers = ("Instructions from:", "private/opencode")
    compaction_usage = Usage(input_tokens=CONTEXT_LIMIT)
    compaction_trigger_steps = 1

    def setUp(self):
        self._sessions = []
        super().setUp()
        self.router.defaults["compaction"] = text_step(SUMMARY)

    def make_session(self, run_dir, store, tool_handler, fake_model, *, model=MODEL, compact=False, sandbox=False):
        opencode = {"executable": EXECUTABLE, "upstream": {
            "wire_api": self.view.name, "base_url": fake_model.base_url, "api_key_env": UPSTREAM_KEY_ENV,
            "auth_scheme": self.auth_scheme, "provider": "conformance"}}
        if compact:
            opencode.update(context_limit_tokens=CONTEXT_LIMIT, output_limit_tokens=OUTPUT_LIMIT)
        config = Config.from_dict({"agent": {"backend": "opencode", "opencode": opencode},
                                   "model": {"name": model, "effort": EFFORT},
                                   "features": {"self_briefing": False, "visual_review_model": None,
                                                "sandbox_enabled": sandbox}})
        if sandbox:   # item 28: the launch gate the gateway and the queue apply accepts the sandboxed config
            factory.backend_requirements(config)
        holder = []

        def client_factory(base_url, password, unix_socket=None):
            return RecordingClient(holder[0], base_url, password, timeout_s=holder[0].request_timeout_s,
                                   unix_socket=unix_socket)
        session = OpenCodeSession(run_dir, store, tool_handler, config=config, client_factory=client_factory)
        holder.append(session)
        self._sessions.append(session)
        return session

    def classify_request(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        if not self.view.tools(body):
            return "helper"
        mains = {session.thread_id for session in list(self._sessions) if session.thread_id}
        native = request.header(SESSION_HEADER)
        if native and native not in mains:
            return "subagent"
        if any(mark in self.view.last_user_text(body) for mark in COMPACTION_MARKS):
            return "compaction"
        if INSTRUCTIONS_MARKER in self.view.system_text(body):
            return "main"
        return "helper"

    # native tool calls in OpenCode 2.2.7's vocabulary (build agent: shell/write/subagent)
    def shell_call(self, command, *, background=False):
        """``shell``; ``background``: the tool's own background mode (returns at once, the completion is reported to
        the agent later)."""
        arguments = {"command": command}
        if background:
            arguments["background"] = True
        return ToolCall("shell", arguments)

    def write_call(self, path, content):
        return ToolCall("write", {"path": str(path), "content": content})

    def subagent_call(self, prompt, *, background=False):
        arguments = {"agent": "general", "description": "Inspect the scene", "prompt": prompt}
        if background:
            arguments["background"] = True
        return ToolCall("subagent", arguments)

    # provenance readers
    def runtime_record(self, provenance):
        from ni_rua.cortex.agents.opencode.config import OPENCODE_VERSION, probe_cli_version
        self.assertIsNone(provenance["sdk"])
        cli = provenance["cli"]
        self.assertEqual(probe_cli_version(cli["executable"]), cli["version"], "recorded version != the binary's")
        return dict(version=cli["version"], executable=cli["executable"], sha256=cli["sha256"],
                    pinned_version=OPENCODE_VERSION)

    def denied_tools(self, provenance):
        return {item["name"] for item in provenance["native_tools"]["disabled"]}

    def init_tool_names(self, provenance):
        return set(provenance["native_tools"]["enabled"] or ())

    def interrupted_result(self, result):
        """OpenCode's own repair text; flagged as an error where the wire has a flag."""
        return INTERRUPTED_TEXT in result.text and self.view.flagged_error(result) is not False

    def unknown_usage(self):
        """An input counter reported as an invalid number: the pinned client refuses a null ``input_tokens`` on
        both wires (stream parse failure, re-requested), the relay treats a negative counter as unknown."""
        return Usage(input_tokens=-1, output_tokens=6)

    def await_history_persisted(self, session, native_call_id, timeout=15):
        """Wait until OpenCode's own session history (``GET /api/session/{id}/message``, its database) has the
        model's tool call ``native_call_id`` as a tool item."""
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            for message in session.client.messages(session.thread_id):
                if any(isinstance(item, dict) and item.get("type") == "tool" and item.get("id") == native_call_id
                       for item in message.get("content") or ()):
                    return
            time.sleep(0.05)
        self.fail(f"OpenCode never recorded the tool call {native_call_id} in its history")

    # fault injection
    @contextlib.contextmanager
    def definite_rejection(self, session):
        """The prompt reaches the running server with a ``text`` of the wrong type: OpenCode's payload validation
        answers HTTP 400 ``InvalidRequestError`` (a definite rejection, nothing admitted)."""
        client = session.client

        def rejected(session_id, text=None, **_options):
            client._request("POST", client._session_path(session_id) + "/prompt", {"text": 0})
            raise AssertionError("OpenCode admitted a prompt with an invalid text")
        with mock.patch.object(client, "prompt", rejected):
            yield

    @contextlib.contextmanager
    def unknown_admission(self, session):
        """``prompt`` times out without an answer (an ``OSError`` of the client: admission unknown). The request is
        never written, so any later model call would be a retry."""
        with mock.patch.object(session.client, "prompt", side_effect=OpenCodeTransportError("TimeoutError")):
            yield

    @contextlib.contextmanager
    def route_outside_traffic(self, proxy_url):
        """Every proxy variable of the server environment points at ``proxy_url`` (loopback stays direct)."""
        from ni_rua.cortex.agents.opencode import config as opencode_config
        values = {name: proxy_url for name in ("HTTPS_PROXY", "HTTP_PROXY", "ALL_PROXY", "https_proxy", "http_proxy",
                                               "all_proxy")}
        values.update(NO_PROXY="127.0.0.1,localhost", no_proxy="127.0.0.1,localhost")
        original = opencode_config.build_launch

        def build_launch(*args, **kwargs):
            launch = original(*args, **kwargs)
            launch.env.update(values)
            return launch
        with mock.patch.object(opencode_config, "build_launch", build_launch):
            yield

    @contextlib.contextmanager
    def default_tool_set(self):
        """Launch without the host's permission denials: the build agent's own default tool set."""
        from ni_rua.cortex.agents.opencode import config as opencode_config
        with mock.patch.dict(opencode_config._DENIED_TOOLS, {}, clear=True):
            yield

    # -- OpenCode-specific checks on top of the shared items ----------------------------------------------------
    def _effort(self, request):
        node = request.json if isinstance(request.json, dict) else {}
        for key in self.effort_field.split("."):
            node = node.get(key) if isinstance(node, dict) else None
        return node

    def assert_effort(self, requests, kind):
        self.assertTrue(requests, f"no {kind} request")
        self.assertEqual([self._effort(request) for request in requests], [EFFORT] * len(requests),
                         f"a {kind} request lacks {self.effort_field}={EFFORT}")

    def test_interrupt_model_and_tool(self):
        """Item 9 + relay block first: every ``POST .../interrupt`` of the main session was sent with its thread
        already tombstoned in the relay."""
        super().test_interrupt_model_and_tool()
        (session,) = self._sessions
        interrupts = [item for item in session.client.interrupts if item["session_id"] == session.thread_id]
        self.assertGreaterEqual(len(interrupts), 2, interrupts)          # one per interrupt
        self.assertTrue(all(item["tombstoned"] for item in interrupts), interrupts)

    def test_provenance(self):
        """Item 17 + the effort variant: provenance names it and its body; every main request carries it."""
        super().test_provenance()
        (session,) = self._sessions
        native = session.provenance()["native_effort"]
        self.assertEqual((native["variant"], native["wire_field"]), (EFFORT, self.effort_field))
        self.assertIsNotNone(native["body"])
        self.assert_effort(self.main_requests(), "main")

    def test_compaction_changes_epoch(self):
        """Item 18 + the effort variant on the native compaction request and on every main request."""
        super().test_compaction_changes_epoch()
        self.assert_effort(self.router.requests["compaction"], "compaction")
        self.assert_effort(self.main_requests(), "main")

    def test_steer_joins_turn(self):
        """Item 20 + OpenCode's join point: each steer is the newest user input of the request right after the step
        it was sent during (the gated model call, then the slow tool), next to that step's tool result."""
        super().test_steer_joins_turn()
        mains = self.main_requests()
        firsts = [next(index for index, request in enumerate(mains) if marker in body_text(request))
                  for marker in ("STEER-ONE-20", "STEER-TWO-20")]
        self.assertEqual(firsts, [1, 2], "a steer did not join at the next step boundary")
        for index, marker, output in ((1, "STEER-ONE-20", "scene: one red cube at x=1"),
                                      (2, "STEER-TWO-20", "slow motion finished")):
            self.assertIn(marker, self.view.last_user_text(mains[index].json))
            self.assertIn(output, self.server.find_tool_results(mains[index])[-1].text)

    def test_truncated_upstream_no_storm(self):
        """Item 23 + deferred delivery: OpenCode's step error is the relay's HTTP 400 (not retried)."""
        super().test_truncated_upstream_no_storm()
        (session,) = self._sessions
        errors = [message["error"] for message in session.client.messages(session.thread_id)
                  if message.get("type") == "assistant" and message.get("error")]
        self.assertEqual([(error.get("type"), error.get("status")) for error in errors],
                         [("provider.invalid-request", 400)], errors)

    def test_subagent_tool_call(self):
        """Item 26 + the effort variant on every subagent request."""
        super().test_subagent_tool_call()
        self.assert_effort(self.router.requests["subagent"], "subagent")

    def test_background_subagent_wakeup(self):
        """Item 27 for a background subagent (design 11 §1.8): the main agent answers and goes idle while the child
        runs; the turn is held (or the child's report adopted), the child's registry call runs inside a turn, the
        report is answered, and every turn completes exactly once."""
        session = self.started_session()
        gate = Gate()
        spawn = self.subagent_call("BG-SUB-27: call the look tool once, then report.", background=True)
        spawn.id = self.call_id()
        self.router.script("main", Step(tool_calls=[spawn]), text_step("MAIN-BEFORE-CHILD-27"),
                           text_step("MAIN-AFTER-REPORT-27"))
        self.router.script("subagent", Step(tool_calls=[ToolCall(self.name("look"), {"x": 2.7})], gate=gate),
                           text_step("SUB-REPORT-27"))
        host = session.run_turn("Delegate the inspection to a background subagent and answer right away.")
        self.assertTrue(gate.wait_entered(60), "the background subagent never called the model")
        self.wait_for(lambda: any("MAIN-BEFORE-CHILD-27" in body_text(request) for request in self.main_requests())
                      or len(self.main_requests()) >= 2, 60, "the main agent did not answer while the child ran")
        time.sleep(1.0)
        self.assertEqual(session.state, "running", "nothing holds or adopts the work of the running child")
        gate.release()
        self.wait_for(lambda: session.state == "completed" and session.completed_final_answer()
                      == "MAIN-AFTER-REPORT-27", TURN_TIMEOUT_S, "the child's report was never answered in a turn")
        (call,) = self.bench.calls_of("look")
        started = [event["correlation"]["turn_id"] for event in self.events("agent.turn_started")]
        self.assertIn(call["params"]["turnId"], started)
        self.assertEqual(started[0], host)
        self.assertEqual(self.completed_per_turn(), {turn: 1 for turn in started})
        self.assertTrue(any("SUB-REPORT-27" in body_text(request) for request in self.main_requests()))
        self.assertEqual(self.disconnects, [])

    def test_set_model_applies_at_the_next_turn(self):
        """Design 11 §2.4: ``set_model`` to an undeclared id while a turn runs declares it (config file rewrite, served
        by ``GET /api/model``); the running turn stays on its model, the next turn uses the new one."""
        session = self.started_session()
        gate = Gate()
        self.router.script("main", Step(tool_calls=[ToolCall(self.name("look"), {"x": 1})], gate=gate),
                           text_step("FIRST-MODEL-ANSWER"), text_step("SECOND-MODEL-ANSWER"))
        session.run_turn("Look, then answer.")
        self.assertTrue(gate.wait_entered(60))
        session.set_model(OTHER_MODEL)
        gate.release()
        self.assertTrue(session.turn_done.wait(TURN_TIMEOUT_S))
        self.assertEqual((session.state, session.completed_final_answer()), ("completed", "FIRST-MODEL-ANSWER"),
                         session.turn_result)
        first = self.main_requests()
        self.assertEqual({request.json.get("model") for request in first}, {MODEL}, "a switch happened mid-turn")
        self.turn(session, "Answer on the new model.")
        self.assertEqual(session.completed_final_answer(), "SECOND-MODEL-ANSWER")
        later = self.main_requests()[len(first):]
        self.assertTrue(later)
        self.assertEqual({request.json.get("model") for request in later}, {OTHER_MODEL})
        self.assertEqual({self._effort(request) for request in later}, {EFFORT})
        provenance = session.provenance()
        self.assertEqual([item["id"] for item in provenance["declared_models"]], [MODEL, OTHER_MODEL])
        self.assertEqual(provenance["native_model_id"], OTHER_MODEL)
        self.assertTrue(self.events("agent.model_declared"))
        self.assertTrue(self.events("agent.model_changed"))

    def test_large_png_tool_result_is_byte_identical(self):
        """Design 11 F11: a 300 kB PNG tool result reaches the provider with the exact bytes the host sent (OpenCode's
        own image re-encoding is off, ``media.image.auto_resize=false``)."""
        session = self.started_session()
        image = noise_png(320)
        self.assertGreater(len(image), 300000)
        self.bench.behaviour["look"] = lambda params: {"success": True, "contentItems": [
            text_item("scene: a large frame"), image_item(image)]}
        look = self.call_id()
        self.router.script("main", tool_step(self.name("look"), {"x": 1}, id=look), text_step("LARGE-DONE"))
        self.turn(session, "Look at the scene.")
        _, result = self.results_for(look)
        self.assertEqual([item.sha256 for item in result.images], [sha256(image)])
        self.assertEqual(session.completed_final_answer(), "LARGE-DONE")

    def test_sandbox_namespace(self):
        """Item 28 + the inbound direction: the host reached the sandboxed server only through its unix socket."""
        original = OpenCodeSession._prepare_sandbox
        inbound = []

        def recording(session, port):
            prefix, env, unix_socket = original(session, port)
            inbound.append((port, unix_socket))
            return prefix, env, unix_socket
        with mock.patch.object(OpenCodeSession, "_prepare_sandbox", recording):
            super().test_sandbox_namespace()
        (session,) = self._sessions
        ((port, unix_socket),) = inbound
        self.assertEqual(session.client.unix_socket, unix_socket)
        self.assertTrue(unix_socket.endswith(f"/inbound/{port}.sock"), unix_socket)
        self.assertIn("--unshare-net", session.process.args)


class OpenCodeAnthropicConformance(_OpenCodeConformance, unittest.TestCase):
    fake_model_class = FakeAnthropicServer
    view = AnthropicMessagesView()
    auth_scheme = "x-api-key"
    effort_field = "output_config.effort"


class OpenCodeResponsesConformance(_OpenCodeConformance, unittest.TestCase):
    fake_model_class = ResponsesServer
    view = OpenAIResponsesView()
    auth_scheme = "bearer"
    effort_field = "reasoning.effort"

if __name__ == "__main__":
    unittest.main()
