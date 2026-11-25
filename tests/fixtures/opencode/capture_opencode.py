"""Capture OpenCode 2.2.7 model requests and client retry behaviour for the relay fixtures (work package OC2-E1).

The pinned binary is launched as ``OpenCodeSession`` launches it: ``opencode.config.build_launch`` (private config dir,
instructions file, isolation flags, the shipped permission rules, one relay model with the default effort variant
``high``), ``opencode serve``, the 2.x ``OpenCodeClient``. The relay provider's base URL points at a loopback fake model
server of ``support.fake_model_servers``, so the captured bodies are what OpenCode itself sends (before the relay's
egress projection). The MCP server is the real ``NiRuaMcpServer`` with five fixture tools (``FIXTURE_TOOLS``). No
real model, no network; every process started here is stopped through ``isolation.terminate_agent_processes``.

Scenarios. Each conversation scenario writes ``<evidence>/<scenario>-<wire>/requests.json`` (every generation
request in arrival order: ``path``, ``headers``, ``body``, ``status``, ``delivered``, ``note``) and ``result.json``
(session ids, run directory, message timeline outline, tool calls):

* ``basic``: the S6 shape of report 06: text, MCP image results, an ``isError`` result, two parallel calls, a native
  ``shell`` call, a reasoning block, then a second user turn.
* ``steer``: a steer while a tool runs, a late steer while the final text streams, an interrupt during a model call
  (that request is ``delivered: false``) and an interrupt during a tool call, each followed by a new prompt.
* ``subagent``: a foreground ``subagent`` (agent ``general``) whose child session calls two NI-RUA tools, then a
  second user turn.
* ``compact``: three user turns of large text and image results with ``limit.context`` 40000 tokens and usage =
  body bytes / 4, so native auto-compaction runs (answered with ``COMPACTION_SUMMARY``).

``attempts`` (relay wires only) writes ``<evidence>/attempts-<wire>.json``: for each answer form of ``attempt_forms``
(the relay's own denial / failure replies, other HTTP errors, unusable streams), served verbatim on every request of
one session, the number of requests OpenCode made before it stopped, the session outcome and the elapsed time.
All forms of one wire run concurrently on one server; a session still retrying after ``ATTEMPT_CAP_S`` is
interrupted and recorded with ``stopped: "cap"``.

Wires: ``anthropic_messages`` and ``openai_responses`` (the relay wires, ``config.WIRE_PACKAGES``) and, for the
conversation scenarios only, ``openai_chat``. The shipped config has no Chat package yet (relay adapter: OC2-E2), so
for that wire the capture swaps the built config's provider package to ``CHAT_PACKAGE`` and drops the effort variants
before the server starts; everything else is the shipped launch.

Usage, from the worktree root (the evidence directory must not exist yet)::

    PYTHONPATH=tests mamba run -n ni-rua python tests/fixtures/opencode/capture_opencode.py \\
        --opencode ~/.local/share/ni-rua/opencode-2.2.7/opencode --evidence /tmp/oc2-e1-evidence

Then ``build_opencode_bodies.py --evidence <dir>`` and ``tests/fixtures/continuity/build_fixture.py --opencode <dir>``.
"""
import argparse
import copy
import dataclasses
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path[:0] = [str(ROOT), str(ROOT / "tests")]

from ni_rua.cortex.agents import isolation  # noqa: E402
from ni_rua.cortex.agents.mcp_server import NiRuaMcpServer  # noqa: E402
from ni_rua.cortex.agents.opencode import config as oc_config  # noqa: E402
from ni_rua.cortex.agents.opencode.client import OpenCodeClient, OpenCodeError, model_ref  # noqa: E402
from ni_rua.cortex.session.wire import base as wire_base, wire_from_name  # noqa: E402
from support.fake_model_servers import (  # noqa: E402
    FakeAnthropicServer, FakeChatServer, FakeResponsesServer, Gate, Step, ToolCall, Usage, text_step, tool_step)

WIRES = ("anthropic_messages", "openai_responses", "openai_chat")
RELAY_WIRES = tuple(oc_config.WIRE_PACKAGES)
FAKES = {"anthropic_messages": FakeAnthropicServer, "openai_responses": FakeResponsesServer,
         "openai_chat": FakeChatServer}
MODELS = {"anthropic_messages": "claude-sonnet-4-5", "openai_responses": "gpt-5.4", "openai_chat": "kimi-k2.6"}
CHAT_PACKAGE = "@opencode/ai/providers/openai-compatible"   # bundled Chat Completions package of the 2.2.7 binary
EFFORT = "high"                                             # model.effort default -> the session's model ref variant
INSTRUCTIONS = ("NI-RUA-INSTRUCTIONS-BEGIN\nYou control a robot through the NI-RUA tools. Tool names are bare "
                "registry names.\nNI-RUA-INSTRUCTIONS-END\n")
THREAD_HEADER = wire_base.OPENCODE_THREAD_HEADER            # x-opencode-session-id (child id on child requests)
PARENT_HEADER = "x-opencode-parent-session-id"              # present only on a child session's requests
# Markers of OpenCode 2.2.7's compaction instructions, the last user item of a summary request: the first summary
# ("You MUST summarize the conversation above ..."), a checkpoint update ("Update the existing checkpoint ...") and the
# one retry after a summary that missed the template. COMPACTION_SUMMARY follows the template (a summary missing it
# is retried once, then the compaction and the turn fail).
COMPACTION_MARKS = ("You MUST use this format for your response", "did not fill in the required summary template")
COMPACTION_SUMMARY = "\n".join([
    "## Objective", "- Fixture objective FIXTURE-SUMMARY.", "", "## Requirements", "- (none)", "", "## Decisions",
    "- (none)", "", "## Work State", "### Completed", "- (none)", "", "### Active", "- observing", "", "### Blocked",
    "- (none)", "", "## Next Move", "1. continue", "", "## Relevant Files", "(none)", "", "## Important Context",
    "- (none)", ""])
COMPACT_CONTEXT_TOKENS = 40000   # limit.context of the compact scenario (threshold = 85 % with the default buffer)
COMPACT_OUTPUT_TOKENS = 4000     # limit.output of the compact scenario
ATTEMPT_CAP_S = 200.0            # s: a session still retrying after this is interrupted (stopped: "cap")
TURN_TIMEOUT_S = 120.0           # s: one user turn of a conversation scenario must end within this
PNG_4X3 = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="

FIXTURE_TOOLS = [
    {"name": "fast", "description": "Fixture tool: return the note as text.",
     "inputSchema": {"type": "object", "properties": {"note": {"type": "string", "description": "text to echo"}},
                     "additionalProperties": False}},
    {"name": "image", "description": "Fixture camera: return one text item and one 4x3 PNG image.",
     "inputSchema": {"type": "object", "properties": {"label": {"type": "string", "description": "view name"}},
                     "additionalProperties": False}},
    {"name": "fail", "description": "Fixture tool that always fails with a typed error text.",
     "inputSchema": {"type": "object", "properties": {}, "additionalProperties": False}},
    {"name": "big", "description": "Fixture tool: return kb kibibytes of numbered text lines.",
     "inputSchema": {"type": "object", "properties": {"kb": {"type": "integer", "description": "size in KiB"}},
                     "required": ["kb"], "additionalProperties": False}},
    {"name": "wait_s", "description": "Fixture tool: sleep seconds (s), then return.",
     "inputSchema": {"type": "object", "properties": {"seconds": {"type": "number", "description": "duration in s"}},
                     "required": ["seconds"], "additionalProperties": False}},
]


class _Plain:
    """Identity presentation (the fixture tools carry no host data to scrub)."""

    def present_text(self, text):
        return text

    def present_data(self, data):
        return data


def _free_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return probe.getsockname()[1]


def _wait(predicate, timeout, what):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        value = predicate()
        if value:
            return value
        time.sleep(0.1)
    raise TimeoutError(what)


def _remove_tree(path):
    """``shutil.rmtree`` after lifting the read-only modes ``build_launch`` sets (config dir 0555, files 0444)."""
    for directory, _, _ in os.walk(path):
        os.chmod(directory, 0o700)
    shutil.rmtree(path, ignore_errors=True)


def _text(value):
    return {"type": "inputText", "text": value}


def _ok(*items):
    return {"success": True, "contentItems": list(items)}


def _last_item_text(body):
    items = body.get("messages") or body.get("input") or []
    return json.dumps(items[-1:]) if isinstance(items, list) else ""


class Server:
    """One ``opencode serve`` launched by ``build_launch`` against ``fake`` (base URL) with the fixture MCP server."""

    def __init__(self, binary, wire, fake_url, *, name, context_limit=None, output_limit=None):
        self.run_dir = Path(tempfile.mkdtemp(prefix="oc2e1-" + name + "-"))
        self.tool_calls, self.release = [], threading.Event()
        self.mcp = NiRuaMcpServer(tools=FIXTURE_TOOLS, presentation=_Plain(), call_tool=self.call_tool,
                                   raw_log=lambda direction, message: None,
                                   on_delivery_failure=lambda call_id, reason: None)
        self.mcp.start()
        spec = oc_config.model_spec(MODELS[wire], context_limit_tokens=context_limit,
                                    output_limit_tokens=output_limit)
        port, password = _free_port(), secrets.token_urlsafe(24)
        self.instance_id = "oc2e1-" + name + "-" + secrets.token_hex(4)
        self.launch = oc_config.build_launch(
            self.run_dir, executable=binary, models=[spec], wire_api=wire if wire in RELAY_WIRES else RELAY_WIRES[0],
            relay_base_url=fake_url, instructions_text=INSTRUCTIONS, port=port, password=password,
            mcp_url=self.mcp.url, mcp_headers=self.mcp.headers, base_env={"LANG": "C.UTF-8"},
            instance_id=self.instance_id)
        self.variant = EFFORT if wire in RELAY_WIRES else None
        if wire not in RELAY_WIRES:
            document = copy.deepcopy(self.launch.config)
            provider = document["providers"][oc_config.RELAY_PROVIDER_ID]
            provider["package"] = CHAT_PACKAGE
            for entry in provider["models"].values():
                entry.pop("variants", None)
            oc_config.write_config(self.launch.config_path, document)
        self.log = open(self.run_dir / "serve.log", "ab")
        self.process = subprocess.Popen(list(self.launch.argv), env=self.launch.env, cwd=self.launch.cwd,
                                        stdin=subprocess.DEVNULL, stdout=self.log, stderr=subprocess.STDOUT,
                                        start_new_session=True, close_fds=True)
        self.client = OpenCodeClient(f"http://127.0.0.1:{port}", password)
        try:
            info = _wait(self._health, 60, "opencode_not_ready")
            self.version = oc_config.check_health_version(info)
            _wait(lambda: (self.client.mcp_status().get("ni_rua") or {}).get("status") == "connected", 30,
                  "mcp_not_connected")
        except BaseException:
            self.close()
            raise

    def _health(self):
        if self.process.poll() is not None:
            raise RuntimeError("opencode_exited: " + (self.run_dir / "serve.log").read_text()[-2000:])
        try:
            return self.client.health(timeout=2)
        except OpenCodeError:
            return None

    def call_tool(self, name, arguments, *, call_id, native_call_id, rpc_id, native_thread_id=None):
        self.tool_calls.append(name)
        if name == "fast":
            return _ok(_text("fast ok: " + str(arguments.get("note", ""))))
        if name == "image":
            return _ok(_text("observation " + str(arguments.get("label", ""))),
                       {"type": "inputImage", "imageUrl": "data:image/png;base64," + PNG_4X3})
        if name == "fail":
            return {"success": False, "contentItems": [_text("error: fixture_tool_failed")]}
        if name == "big":
            size = int(arguments.get("kb", 1)) * 1024
            lines, total, k = [], 0, 0
            while total < size:
                line = "fixture output line %05d of the big tool\n" % k
                lines.append(line)
                total, k = total + len(line), k + 1
            return _ok(_text("".join(lines)[:size]))
        if name == "wait_s":
            self.release.wait(float(arguments.get("seconds", 0)))
            return _ok(_text("waited"))
        return {"success": False, "contentItems": [_text("error: unknown fixture tool")]}

    def session(self, title="ni-rua-actor"):
        return self.client.create_session(title, model=model_ref(oc_config.RELAY_PROVIDER_ID,
                                                                 self.launch.models[0].id, self.variant))

    def idle_count(self, sid):
        return sum(1 for message in self.client.messages(sid) if message.get("type") == "idle")

    def wait_quiet(self, sid, idles, timeout=TURN_TIMEOUT_S):
        """Wait until ``sid`` has ``idles`` idle messages, no session of the server is active and the inbox is
        empty."""
        _wait(lambda: self.idle_count(sid) >= idles and not self.client.status() and not self.client.inbox(sid),
              timeout, "turn_not_finished")

    def close(self):
        """Stop the server tree and every process carrying this launch's marker (taken as late as possible)."""
        snapshot = isolation.process_tree_snapshot(self.process.pid)
        report = isolation.terminate_agent_processes(snapshot, self.instance_id, cwd_roots=[str(self.run_dir)])
        if not report.confirmed:
            print("warning: teardown not confirmed for", self.instance_id, report.to_dict(), file=sys.stderr)
        self.process.poll()
        self.release.set()
        self.mcp.close()
        self.log.close()


class Scenario:
    """Fake model + server for one conversation scenario; the fake answers by request content (``route``)."""

    def __init__(self, binary, wire, name, *, main=(), child=(), usage_from_body=False, **server_kwargs):
        self.wire, self.name = wire, name
        self.main, self.child = list(main), list(child)
        self.usage_from_body = usage_from_body
        self.lock = threading.Lock()
        self.fake = FAKES[wire]((), default=self.route, gate_timeout_s=180).start()
        try:
            self.server = Server(binary, wire, self.fake.base_url, name=f"{name}-{wire}", **server_kwargs)
        except BaseException:
            self.fake.close()
            raise

    def route(self, request):
        body = request.json if isinstance(request.json, dict) else {}
        usage = Usage(input_tokens=len(request.raw_body) // 4, output_tokens=20) if self.usage_from_body else Usage()
        if not body.get("tools"):           # a tool-less side call (title, ...): answered without using the queues
            request.capture_kind = "helper"
            return Step(text="Fixture side answer", usage=usage)
        if any(mark in _last_item_text(body) for mark in COMPACTION_MARKS):
            request.capture_kind = "compaction"
            return Step(text=COMPACTION_SUMMARY, usage=usage)
        child = bool(request.headers.get(PARENT_HEADER))
        with self.lock:
            queue = self.child if child else self.main
            step = queue.pop(0) if queue else text_step("CHILD-DEFAULT" if child else "MAIN-DEFAULT")
        request.capture_kind = "child" if child else "main"
        return dataclasses.replace(step, usage=usage) if self.usage_from_body else step

    def write(self, evidence, result, aborted=()):
        rows = []
        for k, request in enumerate(self.fake.generation_requests()):
            note = aborted[k] if isinstance(aborted, dict) and k in aborted else None
            rows.append(dict(path=request.path, headers=dict(request.headers.items()), body=request.json,
                             status=request.response_status, delivered=bool(
                                 request.response_status == 200 and not request.client_disconnected and note is None),
                             note=note))
        out = Path(evidence) / f"{self.name}-{self.wire}"
        out.mkdir(parents=True)
        (out / "requests.json").write_text(json.dumps(rows, indent=1))
        result = dict(result, wire=self.wire, model=MODELS[self.wire], run_dir=str(self.server.run_dir),
                      run_dir_real=os.path.realpath(self.server.run_dir), version=self.server.version,
                      tool_calls=self.server.tool_calls, requests=len(rows))
        (out / "result.json").write_text(json.dumps(result, indent=1))
        return out

    def outline(self, sid):
        out = []
        for message in self.server.client.messages(sid):
            kind = message.get("type")
            line = kind
            if kind == "assistant":
                line += ":" + ",".join(c.get("type") + ("=" + c.get("name", "") if c.get("type") == "tool" else "")
                                       for c in message.get("content") or []) + f" finish={message.get('finish')}"
                if message.get("error"):
                    line += " error=" + str(message["error"].get("type"))
            elif kind == "idle":
                line += ":" + str(message.get("outcome"))
            elif kind == "compaction":
                line += ":" + str(message.get("status"))
            elif kind in ("user", "synthetic"):
                line += ":" + (message.get("text") or "")[:40]
            out.append(line)
        return out

    def close(self):
        self.server.close()
        self.fake.close()
        _remove_tree(self.server.run_dir)


# ------------------------------------------------------------------------------------------- conversation scenarios
def scenario_basic(binary, wire, evidence):
    main = [tool_step("ni_rua_fast", {"note": "hello"}, text="Calling fast."),
            tool_step("ni_rua_image", {"label": "cam"}, thinking="Planning the observation."),
            tool_step("ni_rua_fail", {}),
            Step(tool_calls=[ToolCall("ni_rua_fast", {"note": "par1"}), ToolCall("ni_rua_image", {"label": "par2"})]),
            tool_step("shell", {"command": "echo native-shell-ok"}),
            text_step("FINAL-ANSWER-ONE"), text_step("FINAL-ANSWER-TWO")]
    run = Scenario(binary, wire, "basic", main=main)
    try:
        sid = run.server.session()
        run.server.client.prompt(sid, "USER-TURN-ONE: please work.")
        run.server.wait_quiet(sid, 1)
        run.server.client.prompt(sid, "USER-TURN-TWO: continue.")
        run.server.wait_quiet(sid, 2)
        return run.write(evidence, dict(session=sid, outline=run.outline(sid)))
    finally:
        run.close()


def scenario_steer(binary, wire, evidence):
    late, held = Gate(), Gate()
    main = [tool_step("ni_rua_wait_s", {"seconds": 3}), text_step("ANSWER-AFTER-STEER"),
            Step(text="FIRST-FINAL", gate=late), text_step("ANSWER-AFTER-LATE-STEER"),
            Step(text="HELD", gate=held), text_step("ANSWER-AFTER-MODEL-INTERRUPT"),
            tool_step("ni_rua_wait_s", {"seconds": 90}), text_step("ANSWER-AFTER-TOOL-INTERRUPT")]
    run = Scenario(binary, wire, "steer", main=main)
    server, aborted = run.server, {}
    try:
        sid = server.session()
        client = server.client
        client.prompt(sid, "USER-TURN-ONE: wait, then answer.")
        _wait(lambda: server.tool_calls.count("wait_s") >= 1, 30, "first_wait_not_started")
        client.prompt(sid, "STEER-ONE: report the time too.")
        server.wait_quiet(sid, 1)
        client.prompt(sid, "USER-TURN-TWO: answer.")
        assert late.wait_entered(30)
        client.prompt(sid, "STEER-TWO: one more thing.")
        time.sleep(0.5)
        late.release()
        server.wait_quiet(sid, 2)
        client.prompt(sid, "USER-TURN-THREE: answer slowly.")
        assert held.wait_entered(30)
        aborted[len(run.fake.generation_requests()) - 1] = "interrupted during the model call (no answer reached " \
                                                            "OpenCode)"
        client.interrupt(sid)
        server.wait_quiet(sid, 3)
        held.release()
        client.prompt(sid, "USER-TURN-FOUR: after the interrupt.")
        server.wait_quiet(sid, 4)
        client.prompt(sid, "USER-TURN-FIVE: long wait.")
        _wait(lambda: server.tool_calls.count("wait_s") >= 2, 30, "second_wait_not_started")
        time.sleep(0.5)
        client.interrupt(sid)
        server.wait_quiet(sid, 5)
        server.release.set()
        client.prompt(sid, "USER-TURN-SIX: final.")
        server.wait_quiet(sid, 6)
        return run.write(evidence, dict(session=sid, outline=run.outline(sid)), aborted)
    finally:
        run.close()


def scenario_subagent(binary, wire, evidence):
    main = [tool_step("subagent", {"agent": "general", "description": "look around",
                                   "prompt": "Observe the scene once with the image tool and report."}),
            text_step("MAIN-FINAL-ONE"), text_step("MAIN-FINAL-TWO")]
    child = [tool_step("ni_rua_image", {"label": "child"}), tool_step("ni_rua_fast", {"note": "child"}),
             text_step("CHILD-REPORT")]
    run = Scenario(binary, wire, "subagent", main=main, child=child)
    try:
        sid = run.server.session()
        run.server.client.prompt(sid, "USER-TURN-ONE: delegate the look.")
        run.server.wait_quiet(sid, 1)
        run.server.client.prompt(sid, "USER-TURN-TWO: continue.")
        run.server.wait_quiet(sid, 2)
        children = sorted({r.headers.get(THREAD_HEADER) for r in run.fake.generation_requests()
                           if r.headers.get(PARENT_HEADER)})
        return run.write(evidence, dict(session=sid, children=children, outline=run.outline(sid),
                                        child_outlines={c: run.outline(c) for c in children}))
    finally:
        run.close()


def scenario_compact(binary, wire, evidence):
    main = []
    for turn in (1, 2, 3):
        for k in range(1, 7):
            main += [tool_step("ni_rua_big", {"kb": 12}), tool_step("ni_rua_image", {"label": f"t{turn}-{k}"})]
        main.append(text_step(f"TURN-{turn}-DONE"))
    run = Scenario(binary, wire, "compact", main=main, usage_from_body=True, context_limit=COMPACT_CONTEXT_TOKENS,
                   output_limit=COMPACT_OUTPUT_TOKENS)
    try:
        sid = run.server.session()
        for turn in (1, 2, 3):
            run.server.client.prompt(sid, f"USER-TURN-{turn}: observe and measure.")
            run.server.wait_quiet(sid, turn)
        compactions = sum(1 for r in run.fake.generation_requests() if getattr(r, "capture_kind", "") == "compaction")
        return run.write(evidence, dict(session=sid, compaction_requests=compactions, outline=run.outline(sid)))
    finally:
        run.close()


# ------------------------------------------------------------------------------------------------ client attempts
def _anthropic_error(message, error_type="invalid_request_error"):
    return json.dumps({"type": "error", "error": {"type": error_type, "message": message}}).encode()


def _openai_error(message, error_type="invalid_request_error"):
    return json.dumps({"error": {"message": message, "type": error_type, "param": None, "code": None}}).encode()


def _sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


JSON = {"Content-Type": "application/json"}
SSE = {"Content-Type": "text/event-stream", "Cache-Control": "no-cache"}


def attempt_forms(wire_name):
    """``[(form name, status, headers, body bytes)]`` served verbatim by ``measure_attempts``."""
    wire = wire_from_name(wire_name, client="opencode")
    forms = [("relay_denial:" + kind, wire.denial(kind)) for kind in wire_base.OPENCODE_DENIAL_MESSAGES]
    forms.append(("relay_error", wire.relay_error(streaming=False, evidence=None, manual_retry_allowed=False)))
    out = [(name, reply.status, dict(reply.headers), reply.body) for name, reply in forms]
    error = _anthropic_error if wire_name == "anthropic_messages" else _openai_error
    for status, message in ((400, "ni_rua relay denied: rate limit reached"), (400, "ni_rua relay denied call 503"),
                            (400, "ni_rua relay denied: connection refused"),
                            (400, "provider overloaded, try again later"),
                            (409, "ni_rua relay denied this request"), (429, "ni_rua relay denied this request"),
                            (500, "provider internal problem"), (529, "provider is busy")):
        out.append((f"http_{status}:{message}", status, dict(JSON), error(message)))
    if wire_name == "anthropic_messages":
        start = _sse("message_start", {"type": "message_start", "message": {
            "id": "msg_fixture", "type": "message", "role": "assistant", "model": "fixture", "content": [],
            "stop_reason": None, "usage": {"input_tokens": 1, "output_tokens": 0}}})
        unusable = _sse("error", {"type": "error", "error": {"type": "invalid_request_error",
                                                             "message": "ni_rua_model_response_unusable"}})
        overloaded = _sse("error", {"type": "error", "error": {"type": "overloaded_error",
                                                               "message": "mid stream failure"}})
        ping = b'event: ping\ndata: {"type": "ping"}\n\n'
        streams = [("stream:start+unusable", start + unusable), ("stream:unusable", unusable),
                   ("stream:ping+unusable", ping + unusable), ("stream:start+overloaded", start + overloaded),
                   ("stream:start_no_terminal", start)]
    else:
        created = _sse("response.created", {"type": "response.created", "response": {
            "id": "resp_fixture", "object": "response", "status": "in_progress", "output": []}})
        failed = _sse("response.failed", {"type": "response.failed", "response": {
            "id": "resp_fixture", "object": "response", "status": "failed", "output": [],
            "error": {"code": "ni_rua_model_response_unusable", "message": "ni_rua_model_response_unusable"}}})
        incomplete = _sse("response.incomplete", {"type": "response.incomplete", "response": {
            "id": "resp_fixture", "object": "response", "status": "incomplete", "output": [],
            "incomplete_details": {"reason": "max_output_tokens"}}})
        streams = [("stream:created+failed", created + failed), ("stream:incomplete", incomplete),
                   ("stream:created_no_terminal", created)]
    out += [(name, 200, dict(SSE), body) for name, body in streams]
    return out


class RawModel:
    """Loopback model endpoint answering every POST with the raw form assigned to the request's session."""

    def __init__(self):
        self.lock, self.forms, self.seen = threading.Lock(), {}, {}
        owner = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_POST(self):
                self.rfile.read(int(self.headers.get("Content-Length") or 0))
                session = self.headers.get(THREAD_HEADER, "")
                with owner.lock:
                    owner.seen.setdefault(session, []).append(time.time())
                    status, headers, body = owner.forms.get(session, (200, dict(SSE), b""))
                try:
                    self.send_response(status)
                    for key, value in headers.items():
                        self.send_header(key, value)
                    self.send_header("Content-Length", str(len(body)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(body)
                except OSError:
                    pass
                self.close_connection = True

            do_GET = do_POST

        self.httpd = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.httpd.daemon_threads = True
        threading.Thread(target=self.httpd.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True).start()
        self.base_url = "http://127.0.0.1:%d" % self.httpd.server_address[1]

    def close(self):
        self.httpd.shutdown()
        self.httpd.server_close()


def measure_attempts(binary, wire, evidence):
    model = RawModel()
    server = None
    try:
        server = Server(binary, wire, model.base_url, name="attempts-" + wire)
        forms, sessions = attempt_forms(wire), {}
        for name, status, headers, body in forms:
            sid = server.session()
            with model.lock:
                model.forms[sid] = (status, headers, body)
            sessions[name] = sid
        started = time.time()
        for sid in sessions.values():
            server.client.prompt(sid, "USER-TURN-ONE: answer.")
        pending = set(sessions)
        while pending and time.time() - started < ATTEMPT_CAP_S:
            time.sleep(1.0)
            active = server.client.status()
            pending = {name for name in pending
                       if sessions[name] in active or server.idle_count(sessions[name]) == 0}
        for name in pending:
            server.client.interrupt(sessions[name])
        rows = []
        for name, status, headers, body in forms:
            sid = sessions[name]
            messages = server.client.messages(sid)
            idle = [m for m in messages if m.get("type") == "idle"]
            errors = [m.get("error") for m in messages if m.get("type") == "assistant" and m.get("error")]
            times = model.seen.get(sid, [])
            rows.append(dict(form=name, status=status, headers=headers, body=body.decode(), attempts=len(times),
                             stopped="cap" if name in pending else "idle",
                             outcome=idle[-1].get("outcome") if idle else None,
                             error_type=(errors[-1] or {}).get("type") if errors else None,
                             assistant_messages=sum(1 for m in messages if m.get("type") == "assistant"),
                             synthetic_messages=sum(1 for m in messages if m.get("type") == "synthetic"),
                             span_s=round(times[-1] - times[0], 1) if len(times) > 1 else 0.0))
        out = Path(evidence) / f"attempts-{wire}.json"
        out.write_text(json.dumps(dict(wire=wire, version=server.version, cap_s=ATTEMPT_CAP_S, forms=rows),
                                  indent=1))
        return out
    finally:
        if server is not None:
            server.close()
            _remove_tree(server.run_dir)
        model.close()


SCENARIOS = {"basic": scenario_basic, "steer": scenario_steer, "subagent": scenario_subagent,
             "compact": scenario_compact}


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--opencode", default=os.environ.get("NI_RUA_OPENCODE_EXECUTABLE"),
                        help="pinned opencode 2.2.7 binary (default: $NI_RUA_OPENCODE_EXECUTABLE)")
    parser.add_argument("--evidence", required=True, help="output directory (must not exist)")
    parser.add_argument("--only", default="", help="comma-separated subset of: " + ", ".join([*SCENARIOS, "attempts"]))
    args = parser.parse_args()
    binary = os.path.abspath(os.path.expanduser(args.opencode or ""))
    evidence = Path(args.evidence)
    evidence.mkdir(parents=True)
    only = {name for name in args.only.split(",") if name}
    for name, function in SCENARIOS.items():
        if only and name not in only:
            continue
        for wire in WIRES:
            started = time.time()
            out = function(binary, wire, evidence)
            print(f"{name} {wire}: {json.loads((out / 'result.json').read_text())['requests']} requests, "
                  f"{time.time() - started:.1f} s", flush=True)
    if not only or "attempts" in only:
        threads, outs = [], {}
        for wire in RELAY_WIRES:
            thread = threading.Thread(target=lambda wire=wire: outs.setdefault(wire, measure_attempts(
                binary, wire, evidence)))
            thread.start()
            threads.append(thread)
        for thread in threads:
            thread.join()
        for wire, out in outs.items():
            for row in json.loads(out.read_text())["forms"]:
                print(f"attempts {wire} {row['form']}: {row['attempts']} ({row['stopped']}, {row['outcome']}, "
                      f"{row['error_type']}, {row['span_s']} s)", flush=True)
    (evidence / "capture.json").write_text(json.dumps(dict(
        opencode_version=oc_config.OPENCODE_VERSION, executable=binary, captured_at=time.strftime(
            "%Y-%m-%dT%H:%M:%SZ", time.gmtime())), indent=1))


if __name__ == "__main__":
    main()
