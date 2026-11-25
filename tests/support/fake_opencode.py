"""Scripted fake ``opencode serve`` 2.2.7 for client and session tests.  Loopback TCP (+ optional unix socket), no
model, no binary.

Every shape is taken from captures of the pinned 2.2.7 binary (``/tmp/ni-rua-agent-harness/oc2/runs/*/out``, the
OC2-B probes ``b01``..``b07``; OpenAPI ``openapi-2.2.7.json``), not from the documentation:

* Basic auth ``opencode:<password>`` on every route, ``/api/info`` and ``/openapi.json`` included; a missing/wrong
  password answers 401 ``{"_tag": "UnauthorizedError", "message": "Authentication required"}`` with
  ``WWW-Authenticate: Basic realm="Secure Area"``.  An unknown route answers 404 with an empty body.
* Typed errors: ``{"_tag": "InvalidRequestError", "message", "kind": "Params" | "Payload" | "Query"}`` (400),
  ``SessionNotFoundError`` (404, ``sessionID``), ``PermissionNotFoundError`` (404, ``requestID``),
  ``InvalidCursorError`` (400), ``ConflictError`` (409), ``UnknownError`` (500), ``ServiceUnavailableError`` (503).
* ``{"data": ...}`` envelopes; ``GET /api/mcp`` and ``GET /api/model`` add ``location``; message pages add
  ``cursor {previous, next}`` (base64url JSON ``{"id", "order", "direction"}``).
* SSE ``GET /api/event``: chunked, one chunk per frame ``data: <json>\\n\\n``; ``server.connected`` first, then a
  ``: heartbeat`` comment at once and every ``heartbeat_s`` (real: 15 s; default here: none); repeated ``types``
  query filter (unknown type -> 400).  Frames: ``{"id": "evt_...", "created", "type", "location"?, "data",
  "durable"?}``.  A message created by an event shares that event's id suffix (``evt_X`` -> ``msg_X``), as real.

Routes (operation ids of the OpenAPI document): ``server.info`` ``GET /api/info`` (``starting_polls`` -> 503 with the
starting body first), ``GET /openapi.json`` (built from ``routes``), ``mcp.list`` ``GET /api/mcp``, ``model.list``
``GET /api/model``, ``location.reload`` ``POST /api/location/reload``, ``event.subscribe`` ``GET /api/event``,
``session.active`` ``GET /api/session/active``, ``session.create`` ``POST /api/session``, ``session.get``,
``session.prompt``, ``session.switchModel``, ``session.interrupt``, ``session.message.list``, ``session.inbox.list``,
``session.permission.list``, ``session.permission.reply`` (all under ``/api/session/{id}``).

Behaviour, driven from the test thread (all state under ``lock``):

* Prompt admission (``session.prompt``): an idle session starts an execution at once (``inbox.enqueued``,
  ``execution.started``, ``inbox.delivered``, the ``user`` message); on an active session the message waits in the
  inbox (steer) until ``deliver``/``finish_turn``; ``delivery="queue"`` waits until the execution ended.  The
  receipt ``{"data": {"id": "msg_...", "sessionID", "time", "type": "user", "payload": {"text"}, "delivery"}}`` id
  is the user message id.
* ``script(operation_id, *outcomes)`` queues per-request outcomes for one operation: an ``int`` HTTP status (typed
  error body, nothing done), ``"hang"`` (read the request, stall ``hang_s``, close unanswered, nothing done),
  ``"drop"`` (state-changing routes: do the operation, then close the socket unanswered: admitted but
  unacknowledged), and for ``session.prompt`` also ``"accept"`` (the default) and ``"auto"`` (accept +
  ``finish_turn(auto_text)``).
  ``prompt_script`` is the list of ``session.prompt``.  ``blocked[operation_id] = status`` fails every request.
* Turn helpers emitting the real events: ``deliver``, ``assistant`` / ``add_content`` / ``finish_tool`` /
  ``complete`` (content items built with ``text_item``, ``reasoning_item``, ``tool_item``), ``finish_turn``, ``idle``,
  ``interrupt_session`` (tool ``aborted`` "Tool execution interrupted", assistant ``aborted`` "Step interrupted",
  ``idle interrupted``), ``add_child`` (``parentID``), ``report_to_parent`` (background subagent's ``synthetic``
  report; starts an idle parent), ``ask_permission`` (a reject fails the tool, ends the execution with
  ``execution.interrupted reason=shutdown`` and writes **no** ``idle``, as real), ``compact``; ``append`` adds any
  raw timeline message.  ``model_info`` builds a ``GET /api/model`` entry for ``models``.
* Knobs: ``health_stalls``/``health_stall_s`` stall the first N ``GET /api/info``; ``mcp_pending_polls`` answers the
  first N ``GET /api/mcp`` with ``pending``; ``busy_gate`` (Event) holds ``session.active``/``session.message.list``
  until set; ``after(operation_id, fn)`` runs ``fn()`` once right after that operation's next answer was sent;
  ``muted`` (events recorded, not sent); ``reload_hook`` (called by ``POST /api/location/reload``).
* ``push``/``comment``/``send_raw_event``/``drop_event_streams``/``end_event_streams`` drive ``/api/event``; events
  emitted while no stream is connected are lost (no replay), like the real server.
* ``requests``: ``{method, path, query, body, headers}`` of every HTTP request except the event stream.
"""
from __future__ import annotations

import base64
import itertools
import json
import os
import re
import socket
import socketserver
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

PIN = "2.2.7"
ROUTES_JSON = Path(__file__).resolve().parents[2] / "ni_rua" / "cortex" / "agents" / "opencode" / "opencode_routes.json"
PASSWORD = "fake-opencode-password"
DIRECTORY = "/tmp/fake-opencode/actor"            # location.directory the server reports (the real one: its cwd)
PROJECT_ID = "0f0a0e0d0c0b0a090807060504030201fa4e0001"
DEFAULT_MODEL = "fake-model"
DEFAULT_PAGE = 50                                 # page size of GET .../message without `limit` (binary: lee=50)
MAX_PAGE = 200
PACKAGE = "@opencode/ai/providers/anthropic-compatible"
EVENT_TYPES = frozenset((
    "location.shutdown", "models-dev.refreshed", "credential.updated", "credential.switched", "integration.updated",
    "provider.updated", "model.updated", "agent.updated", "session.created", "session.agent.selected",
    "session.model.selected", "session.moved", "session.renamed", "session.metadata.updated", "session.permissions",
    "session.viewed", "session.archived", "session.unarchived", "session.usage.updated", "session.deleted",
    "session.forked", "session.inbox.delivered", "session.inbox.enqueued", "session.inbox.cancelled",
    "session.inbox.delivery.changed", "session.execution.started", "session.execution.succeeded",
    "session.execution.failed", "session.execution.interrupted", "session.instructions.updated", "session.synthetic",
    "session.skill.activated", "session.shell.started", "session.shell.ended", "session.step.started",
    "session.step.streamed", "session.step.ended", "session.step.failed", "session.text.started", "session.text.delta",
    "session.text.ended", "session.reasoning.started", "session.reasoning.delta", "session.reasoning.ended",
    "session.tool.input.started", "session.tool.input.delta", "session.tool.input.ended", "session.tool.called",
    "session.tool.progress", "session.tool.success", "session.tool.failed", "session.retry.scheduled",
    "session.compaction.started", "session.compaction.delta", "session.compaction.ended", "session.compaction.failed",
    "session.revert.staged", "session.revert.cleared", "session.revert.committed", "filesystem.changed",
    "reference.updated", "permission.asked", "permission.replied", "plugin.updated", "project.updated",
    "worktree.updated", "worktree.resolved", "command.updated", "config.updated", "skill.updated", "pty.created",
    "pty.updated", "pty.exited", "pty.deleted", "persistent-pty.added", "persistent-pty.removed",
    "persistent-pty.updated", "shell.created", "shell.exited", "shell.deleted", "form.created", "form.replied",
    "form.cancelled", "websearch.updated", "artifact.updated", "session.status", "session.idle", "tui.prompt.append",
    "tui.command.execute", "tui.toast.show", "tui.session.select", "installation.updated",
    "installation.update-available", "vcs.branch.updated", "mcp.status.changed", "mcp.resources.changed"))
# Events the real server sends without `location` / without `durable` (captured).
_NO_LOCATION = frozenset(("server.connected", "session.execution.started", "session.execution.succeeded",
                          "session.execution.failed", "session.execution.interrupted", "session.usage.updated",
                          "session.model.selected"))
_VOLATILE = frozenset(("server.connected", "session.text.delta", "session.reasoning.delta", "session.tool.input.delta",
                       "session.usage.updated", "session.compaction.delta", "permission.asked", "permission.replied"))
_ZERO_TOKENS = {"input": 0, "output": 0, "reasoning": 0, "cache": {"read": 0, "write": 0}}
_STEP_TOKENS = {"input": 10, "output": 5, "reasoning": 0, "cache": {"read": 0, "write": 0}}
_SUFFIX_COUNTER = itertools.count(1)
_SESSION_COUNTER = itertools.count(1)
_SESSION_ROUTE = re.compile(r"/api/session/([^/]+)(?:/(.*))?\Z")


def _suffix() -> str:
    """Id body shared by an event and the message it creates; increasing, so ids sort in creation order."""
    return f"{next(_SUFFIX_COUNTER):012x}FakeIdAbcdefgh"


def _now() -> int:
    return int(time.time() * 1000)


def text_item(text: str) -> dict:
    """Assistant content item ``{"type": "text", "text"}`` (2.x text items carry no part id)."""
    return {"type": "text", "text": text}


def reasoning_item(text: str) -> dict:
    """Assistant content item ``{"type": "reasoning", "text", "time"}`` (the provider ``native`` field is omitted)."""
    now = _now()
    return {"type": "reasoning", "text": text, "time": {"created": now, "completed": now}}


def tool_item(name: str, tool_input: dict | None = None, *, call_id: str | None = None, status: str = "running",
              content: list | None = None, error: dict | None = None, metadata: dict | None = None) -> dict:
    """Assistant content item ``{"type": "tool", "id": <provider call id>, "name", "executed": false, "state":
    {"status": streaming|running|completed|error, "input", "content"?, "error"?, "metadata"?}, "time"}``."""
    now = _now()
    state: dict = {"status": status, "input": dict(tool_input or {})}
    if content is not None or status == "completed":
        state["content"] = list(content or [])
    if error is not None:
        state["error"] = dict(error)
    if metadata is not None or status == "completed":
        state["metadata"] = dict(metadata if metadata is not None else {"truncated": False})
    item = {"type": "tool", "id": call_id or f"call_fake_{next(_SUFFIX_COUNTER)}", "name": name, "executed": False,
            "state": state, "time": {"created": now, "ran": now}}
    if status in ("completed", "error"):
        item["time"]["completed"] = now
    return item


def model_info(model_id: str = DEFAULT_MODEL, *, provider_id: str = "relay", package: str = PACKAGE,
               variants: list | None = None, input_modalities: tuple = ("text", "image"), context: int = 200000,
               output: int = 32000, base_url: str = "http://127.0.0.1:9/v1") -> dict:
    """One ``GET /api/model`` entry (``Model.Info``) as 2.2.7 reports a config-declared model; ``variants`` is the
    declared list (``[{"id", "body"}]``, reported verbatim) or None = no explicit variants (then 2.2.7 reports the
    package's generated ones; here: an empty list)."""
    return {"id": model_id, "modelID": model_id, "providerID": provider_id, "name": model_id, "package": package,
            "settings": {"apiKey": "ni-rua-local-relay", "baseURL": base_url},
            "capabilities": {"tools": True, "input": list(input_modalities), "output": ["text"]},
            "variants": [dict(item) for item in (variants or [])], "time": {"released": 0}, "cost": [],
            "status": "active", "enabled": True, "limit": {"context": context, "output": output}}


def _error(status: int, tag: str, message: str, **fields) -> tuple[int, dict]:
    return status, {"_tag": tag, "message": message, **fields}


def _invalid(message: str, kind: str) -> tuple[int, dict]:
    return _error(400, "InvalidRequestError", message, kind=kind)


def _scripted_error(status: int, session_id: str | None) -> tuple[int, dict]:
    if status == 400:
        return _invalid("scripted rejection", "Payload")
    if status == 401:
        return _error(401, "UnauthorizedError", "Authentication required")
    if status == 404:
        return _error(404, "SessionNotFoundError", f"Session not found: {session_id}", sessionID=session_id or "")
    if status == 409:
        return _error(409, "ConflictError", "scripted conflict", resource=None)
    if status == 503:
        return _error(503, "ServiceUnavailableError", "scripted unavailable", service=None)
    return _error(status, "UnknownError", "scripted failure", ref=None)


def _encode_cursor(message_id: str, order: str, direction: str) -> str:
    raw = json.dumps({"id": message_id, "order": order, "direction": direction}, separators=(",", ":"))
    return base64.urlsafe_b64encode(raw.encode()).decode().rstrip("=")


def _decode_cursor(cursor: str) -> dict | None:
    try:
        value = json.loads(base64.urlsafe_b64decode(cursor + "=" * (-len(cursor) % 4)))
    except (ValueError, TypeError):
        return None
    if not isinstance(value, dict) or value.get("order") not in ("asc", "desc") \
            or value.get("direction") not in ("next", "previous") or not isinstance(value.get("id"), str):
        return None
    return value


class _Answered(Exception):
    """Raised inside a route after it wrote (or deliberately did not write) its answer."""


class FakeOpenCodeServer:
    def __init__(self, *, password: str = PASSWORD, version: str = PIN, routes: list[str] | None = None,
                 auto_text: str = "done", models: list | None = None, default_model: str = DEFAULT_MODEL,
                 unix_socket: str | None = None, directory: str = DIRECTORY):
        self.password, self.version, self.auto_text = password, version, auto_text
        self.routes = list(routes) if routes is not None else _checked_in_routes()
        self.default_model, self.directory = default_model, directory
        self.models = list(models) if models is not None else [model_info(default_model)]
        self.mcp_servers: list = [{"name": "ni_rua", "status": {"status": "connected"}}]
        self.unix_socket = unix_socket
        self.hang_s = 2.0
        self.starting_polls = 0          # first N GET /api/info answer 503 (server still starting)
        self.health_stalls = 0           # first N GET /api/info stall health_stall_s before answering
        self.health_stall_s = 2.0
        self.mcp_pending_polls = 0       # first N GET /api/mcp report every server `pending`
        self.heartbeat_s: float | None = None
        self.busy_gate: threading.Event | None = None
        self.muted = False
        self.reload_hook = None
        self.blocked: dict[str, int] = {}
        self.scripts: dict[str, list] = {}
        self.hooks: dict[str, list] = {}
        self.requests: list[dict] = []
        self.sessions: dict[str, dict] = {}
        self.active: dict[str, dict] = {}
        self.event_log: list[dict] = []
        self.lock = threading.RLock()
        self._send_lock = threading.RLock()
        self._seq: dict[str, int] = {}
        self._subscribers: list = []
        self._servers: list = []
        self._threads: list = []
        self.url = ""
        self.pid = os.getpid()

    # ------------------------------------------------------------------------------------------- lifecycle
    def start(self) -> "FakeOpenCodeServer":
        fake = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                fake._handle(self)

            do_POST = do_DELETE = do_PUT = do_PATCH = do_GET

        class Httpd(ThreadingHTTPServer):
            daemon_threads = True
            allow_reuse_address = True
            request_queue_size = 64

        class UnixHttpd(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
            daemon_threads = True
            request_queue_size = 64

        tcp = Httpd(("127.0.0.1", 0), Handler)
        self._servers.append(tcp)
        self.url = f"http://127.0.0.1:{tcp.server_address[1]}"
        if self.unix_socket is not None:
            self._servers.append(UnixHttpd(self.unix_socket, Handler))
        for server in self._servers:
            thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
            thread.start()
            self._threads.append(thread)
        return self

    def close(self) -> None:
        self.drop_event_streams()
        for server in self._servers:
            server.shutdown()
            server.server_close()
        self._servers = []
        for thread in self._threads:
            thread.join(5)
        self._threads = []
        if self.unix_socket is not None:
            try:
                os.unlink(self.unix_socket)
            except OSError:
                pass

    def __enter__(self):
        return self.start()

    def __exit__(self, *exc):
        self.close()

    # ------------------------------------------------------------------------------------- scripting API
    @property
    def prompt_script(self) -> list:
        """Per-request outcomes of ``session.prompt`` (see ``script``)."""
        return self.scripts.setdefault("session.prompt", [])

    def script(self, operation_id: str, *outcomes) -> None:
        with self.lock:
            self.scripts.setdefault(operation_id, []).extend(outcomes)

    def after(self, operation_id: str, hook) -> None:
        """Run ``hook()`` once, right after the next answer of ``operation_id`` was written (handler thread)."""
        with self.lock:
            self.hooks.setdefault(operation_id, []).append(hook)

    def request_paths(self, method: str | None = None) -> list[str]:
        with self.lock:
            return [item["path"] for item in self.requests if method in (None, item["method"])]

    def info(self, session_id: str) -> dict:
        with self.lock:
            return json.loads(json.dumps(self.sessions[session_id]["info"]))

    def messages_of(self, session_id: str) -> list:
        with self.lock:
            return json.loads(json.dumps(self.sessions[session_id]["messages"]))

    def inbox_of(self, session_id: str) -> list:
        with self.lock:
            return json.loads(json.dumps(self.sessions[session_id]["inbox"]))

    # ---------------------------------------------------------------------------------------------- events
    def push(self, event_type: str, data: dict | None = None, *, event_id: str | None = None,
             session_id: str | None = None) -> dict:
        """Emit one event to every connected ``/api/event`` stream whose filter admits it (lost if none).  A
        ``durable`` block (per-session ``seq``) is added for durable session events of ``session_id``.  Pushes from
        different threads reach every stream in ``event_log`` order (``_send_lock``; the state ``lock`` is not held
        while writing to sockets)."""
        event = {"id": event_id or "evt_" + _suffix(), "created": _now(), "type": event_type}
        if event_type not in _NO_LOCATION:
            event["location"] = {"directory": self.directory}
        event["data"] = dict(data or {})
        with self._send_lock:
            with self.lock:
                if session_id is not None and event_type not in _VOLATILE:
                    seq = self._seq.get(session_id, -1) + 1
                    self._seq[session_id] = seq
                    event["durable"] = {"aggregateID": session_id, "seq": seq, "version": 1}
                self.event_log.append(event)
                subscribers = [] if self.muted else list(self._subscribers)
            frame = ("data: " + json.dumps(event, separators=(",", ":")) + "\n\n").encode()
            for subscriber in subscribers:
                if subscriber.accepts(event_type):
                    subscriber.send(frame)
        return event

    def comment(self, text: str = "heartbeat") -> None:
        with self.lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            subscriber.send(f": {text}\n\n".encode())

    def send_raw_event(self, raw: bytes, *, chunks: list[int] | None = None) -> None:
        """Send ``raw`` SSE bytes to every stream, as one chunk or split at the byte offsets ``chunks``."""
        cuts = [0, *(chunks or []), len(raw)]
        with self.lock:
            subscribers = list(self._subscribers)
        for subscriber in subscribers:
            for start, end in zip(cuts, cuts[1:]):
                if end > start:
                    subscriber.send(raw[start:end])

    def wait_subscribers(self, count: int = 1, timeout: float = 5.0) -> bool:
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            with self.lock:
                if len(self._subscribers) >= count:
                    return True
            time.sleep(0.01)
        return False

    def drop_event_streams(self) -> None:
        """Close every stream's socket without the terminating chunk (a dropped connection)."""
        with self.lock:
            subscribers, self._subscribers = list(self._subscribers), []
        for subscriber in subscribers:
            subscriber.close()

    def end_event_streams(self) -> None:
        """End every stream cleanly (terminating ``0`` chunk), then close it."""
        with self.lock:
            subscribers, self._subscribers = list(self._subscribers), []
        for subscriber in subscribers:
            subscriber.finish()
            subscriber.close()

    # ---------------------------------------------------------------------------------------------- state
    def add_session(self, title: str = "ni-rua-test", *, parent_id: str | None = None, agent: str | None = None,
                    model: dict | None = None) -> str:
        """Create a session (``session.created``); ``parent_id``/``agent`` make it a subagent session."""
        session_id = f"ses_{next(_SESSION_COUNTER):012x}FakeSesAbcdef"
        now = _now()
        info: dict = {"id": session_id, "projectID": PROJECT_ID, "cost": 0, "tokens": dict(_ZERO_TOKENS),
                      "time": {"created": now, "updated": now}, "title": title,
                      "location": {"directory": self.directory}}
        if parent_id is not None:
            info["parentID"] = parent_id
        if agent is not None:
            info["agent"] = agent
        if model is not None:
            info["model"] = dict(model, variant=model.get("variant") or "default")
        with self.lock:
            self.sessions[session_id] = {"info": info, "messages": [], "inbox": [], "permissions": []}
        created = {"sessionID": session_id, "slug": "fake-slug", "version": self.version, "projectID": PROJECT_ID,
                   "location": {"directory": self.directory}, "subpath": "", "title": title}
        if parent_id is not None:
            created["parentID"] = parent_id
        if agent is not None:
            created["agent"] = agent
        if model is not None:
            created["model"] = dict(model)
        self.push("session.created", created, session_id=session_id)
        return session_id

    def add_child(self, parent_id: str, *, agent: str = "general", title: str = "subagent work") -> str:
        """A subagent session of ``parent_id`` (``session.created`` carries ``parentID`` and ``agent``)."""
        return self.add_session(title, parent_id=parent_id, agent=agent)

    def append(self, session_id: str, message: dict, *, suffix: str | None = None) -> dict:
        """Append one raw timeline message (``id`` = ``msg_<suffix>`` and ``time.created`` filled in if absent)."""
        message = dict(message)
        message.setdefault("id", "msg_" + (suffix or _suffix()))
        message.setdefault("time", {"created": _now()})
        with self.lock:
            self.sessions[session_id]["messages"].append(message)
            self.sessions[session_id]["info"]["time"]["updated"] = _now()
        return message

    def start_execution(self, session_id: str) -> bool:
        """Mark the session active (``session.execution.started``); False when it already was."""
        with self.lock:
            if session_id in self.active:
                return False
            self.active[session_id] = {"type": "running"}
            self.sessions[session_id]["info"]["time"]["busy"] = _now()
        self.push("session.execution.started", {"sessionID": session_id}, session_id=session_id)
        return True

    def enqueue(self, session_id: str, item_type: str, payload: dict, *, delivery: str = "steer") -> dict:
        """Admit one inbox item (``session.inbox.enqueued``); returns ``Session.Inbox.Info``."""
        item = {"id": "msg_" + _suffix(), "sessionID": session_id, "time": {"created": _now()}, "type": item_type,
                "payload": dict(payload), "delivery": delivery}
        with self.lock:
            self.sessions[session_id]["inbox"].append(item)
        self.push("session.inbox.enqueued", {"inboxID": item["id"], "sessionID": session_id,
                                             "item": {"type": item_type, "payload": dict(payload),
                                                      "delivery": delivery}}, session_id=session_id)
        return item

    def deliver(self, session_id: str, *, delivery: str = "steer") -> list:
        """Deliver every waiting inbox item of ``delivery`` (``session.inbox.delivered``) as timeline messages, as
        the real server does at a step boundary; returns the new messages."""
        with self.lock:
            inbox = self.sessions[session_id]["inbox"]
            items = [item for item in inbox if item["delivery"] == delivery]
            inbox[:] = [item for item in inbox if item["delivery"] != delivery]
        delivered = []
        for item in items:
            self.push("session.inbox.delivered", {"sessionID": session_id, "inboxID": item["id"]},
                      session_id=session_id)
            payload, synthetic = item["payload"], item["type"] == "synthetic"
            message: dict = {"id": item["id"]}          # the inbox id becomes the message id
            if synthetic and "metadata" in payload:
                message["metadata"] = payload["metadata"]
            message["time"] = {"created": _now()}
            message["text"] = payload.get("text", "")
            if synthetic and "description" in payload:
                message["description"] = payload["description"]
            message["type"] = item["type"]
            delivered.append(self.append(session_id, message))
        return delivered

    def admit_prompt(self, session_id: str, text: str, *, delivery: str = "steer") -> dict:
        """What ``POST /api/session/{id}/prompt`` does once validated: enqueue; an idle session starts at once."""
        receipt = self.enqueue(session_id, "user", {"text": text}, delivery=delivery)
        if self.start_execution(session_id):
            self.deliver(session_id, delivery=delivery)
        return receipt

    def assistant(self, session_id: str, content: list | None = None, *, finish: str | None = None,
                  error: dict | None = None, model: dict | None = None, agent: str = "build") -> dict:
        """Add an assistant message (``session.step.started``, then the stream events of ``content`` via
        ``add_content``); ``finish=None`` leaves it in progress."""
        with self.lock:
            session_model = self.sessions[session_id]["info"].get("model")
        if model is None:
            model = dict(session_model) if session_model else {"id": self.default_model, "providerID": "relay"}
        now = _now()
        message = {"time": {"created": now, "streamed": now}, "type": "assistant", "agent": agent,
                   "model": dict(model), "content": []}
        message = self.append(session_id, message)
        self.push("session.step.started", {"sessionID": session_id, "agent": agent, "model": dict(model),
                                           "assistantMessageID": message["id"], "started": now},
                  session_id=session_id)
        self.add_content(session_id, message, content or [])
        if finish is not None or error is not None:
            self.complete(session_id, message, finish=finish or "error", error=error)
        return message

    def add_content(self, session_id: str, message: dict, items: list) -> None:
        """Append content items to an in-progress assistant message with the events the real server streams:
        text ``text.started/delta/ended``, reasoning ``reasoning.started/delta/ended``, tool ``tool.input.started/
        ended`` + ``tool.called`` (+ ``tool.success``/``tool.failed`` for a finished one), then ``step.streamed``."""
        base = {"sessionID": session_id, "assistantMessageID": message["id"]}
        with self.lock:
            ordinals = {kind: sum(1 for item in message["content"] if item.get("type") == kind)
                        for kind in ("text", "reasoning")}
            message["content"].extend(items)
        for item in items:
            kind = item.get("type")
            if kind in ("text", "reasoning"):
                ordinal = ordinals[kind]
                ordinals[kind] += 1
                self.push(f"session.{kind}.started", dict(base, ordinal=ordinal), session_id=session_id)
                self.push(f"session.{kind}.delta", dict(base, ordinal=ordinal, delta=item["text"]))
                self.push(f"session.{kind}.ended", dict(base, ordinal=ordinal, text=item["text"]),
                          session_id=session_id)
            elif kind == "tool":
                state = item["state"]
                self.push("session.tool.input.started", dict(base, id=item["id"], name=item["name"]),
                          session_id=session_id)
                self.push("session.tool.input.ended", dict(base, id=item["id"], text=json.dumps(state["input"])),
                          session_id=session_id)
                self.push("session.tool.called", dict(base, id=item["id"], input=state["input"], executed=False),
                          session_id=session_id)
                if state["status"] in ("completed", "error"):
                    self._tool_finished(session_id, message, item)
        self.push("session.step.streamed", dict(base), session_id=session_id)

    def finish_tool(self, session_id: str, message: dict, call_id: str, *, content: list | None = None,
                    error: dict | None = None, metadata: dict | None = None) -> dict:
        """A running tool of ``message`` ends: ``completed`` with ``content`` (``[{"type": "text", "text"} |
        {"type": "file", "uri", "mime"}]``) or ``error`` (``{"type", "message"}``); emits ``session.tool.success`` /
        ``session.tool.failed``."""
        with self.lock:
            item = next(entry for entry in message["content"] if entry.get("type") == "tool" and entry["id"] == call_id)
            state: dict = {"status": "error" if error is not None else "completed", "input": item["state"]["input"]}
            if error is not None:
                state["error"] = dict(error)
            else:
                state["content"] = list(content or [])
                state["metadata"] = dict(metadata if metadata is not None else {"truncated": False})
            item["state"] = state
            item["time"]["completed"] = _now()
        self._tool_finished(session_id, message, item)
        return item

    def _tool_finished(self, session_id: str, message: dict, item: dict) -> None:
        base = {"sessionID": session_id, "assistantMessageID": message["id"], "id": item["id"]}
        state = item["state"]
        if state["status"] == "error":
            self.push("session.tool.failed", dict(base, error=state.get("error"), executed=False),
                      session_id=session_id)
        else:
            self.push("session.tool.success", dict(base, content=state.get("content", []),
                                                   metadata=state.get("metadata", {}), executed=False),
                      session_id=session_id)

    def complete(self, session_id: str, message: dict, *, finish: str = "stop", error: dict | None = None) -> dict:
        """Finish an assistant message (``session.step.ended`` / ``session.step.failed``, ``session.usage.updated``)."""
        with self.lock:
            message["time"]["completed"] = _now()
            message["finish"] = "error" if error is not None else finish
            if error is None:
                message["rawFinish"] = {"stop": "end_turn", "tool-calls": "tool_use",
                                        "length": "max_tokens"}.get(finish, finish)
            message.update(cost=0, tokens=dict(_STEP_TOKENS))
            if error is not None:
                message["error"] = dict(error)
            info = self.sessions[session_id]["info"]
            info["tokens"] = {key: info["tokens"].get(key, 0) + value for key, value in _STEP_TOKENS.items()
                              if key != "cache"}
            info["tokens"]["cache"] = {"read": 0, "write": 0}
            usage = {"sessionID": session_id, "cost": 0, "tokens": json.loads(json.dumps(info["tokens"]))}
        if error is None:
            self.push("session.step.ended", {"sessionID": session_id, "assistantMessageID": message["id"],
                                             "finish": message["finish"], "rawFinish": message["rawFinish"],
                                             "tokens": dict(_STEP_TOKENS), "cost": 0}, session_id=session_id)
        else:
            self.push("session.step.failed", {"sessionID": session_id, "assistantMessageID": message["id"],
                                              "error": dict(error)}, session_id=session_id)
        self.push("session.usage.updated", usage)
        return message

    def idle(self, session_id: str, outcome: str = "succeeded", error: dict | None = None,
             reason: str = "user") -> dict:
        """End the execution: the ``idle`` message, inactive, ``session.execution.succeeded|failed|interrupted``
        (event and message share their id suffix); then a waiting ``queue`` prompt starts the next execution."""
        suffix = _suffix()
        message = {"time": {"created": _now()}, "type": "idle", "outcome": outcome}
        if error is not None:
            message["error"] = dict(error)
        message = self.append(session_id, message, suffix=suffix)
        with self.lock:
            self.active.pop(session_id, None)
            info = self.sessions[session_id]["info"]
            info["outcome"], info["time"]["idle"] = outcome, _now()
        kind = {"succeeded": "succeeded", "failed": "failed"}.get(outcome, "interrupted")
        data: dict = {"sessionID": session_id}
        if kind == "failed" and error is not None:
            data["error"] = dict(error)
        if kind == "interrupted":
            data["reason"] = reason
        self.push(f"session.execution.{kind}", data, event_id="evt_" + suffix, session_id=session_id)
        with self.lock:
            queued = any(item["delivery"] == "queue" for item in self.sessions[session_id]["inbox"])
        if queued and self.start_execution(session_id):
            self.deliver(session_id, delivery="queue")
        return message

    def finish_turn(self, session_id: str, text: str = "done", *, finish: str = "stop", error: dict | None = None,
                    outcome: str | None = None) -> dict:
        """Deliver waiting steers, add the final assistant message (``text``; with ``error``: no content,
        ``finish="error"``) and end the execution (``idle``, outcome ``succeeded``/``failed``)."""
        self.deliver(session_id)
        content = [] if error is not None or not text else [text_item(text)]
        message = self.assistant(session_id, content, finish=finish, error=error)
        self.idle(session_id, outcome or ("failed" if error is not None else "succeeded"), error)
        return message

    def interrupt_session(self, session_id: str) -> bool:
        """``POST /interrupt`` of an active session: running tools -> error ``aborted`` "Tool execution
        interrupted", the open assistant message -> ``finish: error`` ``aborted`` "Step interrupted" (or a new empty
        one when none is open), ``idle interrupted``.  False (idle no-op) when inactive."""
        with self.lock:
            if session_id not in self.active:
                return False
            messages = self.sessions[session_id]["messages"]
            last = messages[-1] if messages else None
            open_message = last if last and last.get("type") == "assistant" and "finish" not in last else None
        aborted = {"type": "aborted", "message": "Step interrupted"}
        if open_message is None:
            self.assistant(session_id, [], error=aborted)
        else:
            with self.lock:
                tools = [item["id"] for item in open_message["content"] if item.get("type") == "tool"
                         and item["state"].get("status") in ("streaming", "running")]
            for call_id in tools:
                self.finish_tool(session_id, open_message, call_id,
                                 error={"type": "aborted", "message": "Tool execution interrupted"})
            self.complete(session_id, open_message, error=aborted)
        self.idle(session_id, "interrupted")
        return True

    def report_to_parent(self, child_id: str, text: str, *, description: str = "subagent work",
                         state: str = "completed", agent: str = "General") -> dict:
        """A background subagent's report: a ``synthetic`` inbox item of the parent (an idle parent starts a new
        execution and receives it at once; an active one at its next step boundary / ``finish_turn``)."""
        with self.lock:
            parent_id = self.sessions[child_id]["info"]["parentID"]
        body = f'<subagent sessionID="{child_id}" state="{state}" description="{description}">\n{text}\n</subagent>'
        item = self.enqueue(parent_id, "synthetic", {"text": body, "description": description, "metadata": {
            "source": "subagent", "childID": child_id, "agent": agent, "state": state}})
        if self.start_execution(parent_id):
            self.deliver(parent_id)
        return item

    def ask_permission(self, session_id: str, *, action: str = "read", resources: tuple = (".env",),
                       tool_message_id: str | None = None, tool_call_id: str | None = None) -> str:
        """A pending permission request (``permission.asked``), optionally owned by a tool call."""
        request = {"id": "per_" + _suffix(), "sessionID": session_id, "action": action, "resources": list(resources),
                   "save": ["*"]}
        if tool_message_id is not None and tool_call_id is not None:
            request["source"] = {"type": "tool", "messageID": tool_message_id, "id": tool_call_id}
        with self.lock:
            self.sessions[session_id]["permissions"].append(request)
        self.push("permission.asked", dict(request))
        return request["id"]

    def compact(self, session_id: str, *, status: str = "completed", reason: str = "auto",
                summary: str = "## Objective\n- fake summary\n") -> dict:
        """A compaction: ``session.compaction.started`` (id suffix = the ``compaction`` message id suffix), the
        message, then ``session.compaction.ended`` / ``failed``."""
        suffix = _suffix()
        self.push("session.compaction.started", {"sessionID": session_id, "reason": reason},
                  event_id="evt_" + suffix, session_id=session_id)
        message = {"time": {"created": _now()}, "type": "compaction", "status": status, "reason": reason,
                   "model": {"id": self.default_model, "providerID": "relay"}, "cost": 0, "tokens": dict(_STEP_TOKENS)}
        if status == "completed":
            message["summary"] = summary
        message = self.append(session_id, message, suffix=suffix)
        ended = "session.compaction.ended" if status == "completed" else "session.compaction.failed"
        self.push(ended, {"sessionID": session_id, "reason": reason}, session_id=session_id)
        return message

    # ------------------------------------------------------------------------------------------------ HTTP
    def _handle(self, handler: BaseHTTPRequestHandler) -> None:
        length = int(handler.headers.get("Content-Length") or 0)
        raw = handler.rfile.read(length) if length else b""
        try:
            body = json.loads(raw) if raw else None
            parse_error = False
        except ValueError:
            body, parse_error = {"_unparsable": raw[:200].decode("latin1")}, True
        split = urlsplit(handler.path)
        path, query = split.path, parse_qs(split.query, keep_blank_values=True)
        if path != "/api/event":
            with self.lock:
                self.requests.append({"method": handler.command, "path": path, "query": split.query, "body": body,
                                      "headers": dict(handler.headers)})
        expected = "Basic " + base64.b64encode(f"opencode:{self.password}".encode()).decode()
        try:
            if handler.headers.get("Authorization") != expected:
                return self._json(handler, 401, {"_tag": "UnauthorizedError", "message": "Authentication required"},
                                  extra={"WWW-Authenticate": 'Basic realm="Secure Area"'})
            route = self._route(handler.command, path)
            if route is None:
                return self._empty(handler, 404)
            operation, method, session_id, rest = route
            if parse_error:
                return self._json(handler, *_invalid("Invalid JSON", "Payload"))
            outcome = self._next_outcome(operation)
            if isinstance(outcome, int):
                return self._json(handler, *_scripted_error(outcome, session_id))
            if outcome == "hang":
                time.sleep(self.hang_s)
                handler.close_connection = True
                return
            try:
                method(handler, query, body, session_id, rest, outcome)
            except _Answered:
                pass
            self._run_hooks(operation)
        except (BrokenPipeError, ConnectionError):
            return

    def _route(self, command: str, path: str):
        fixed = {("GET", "/api/info"): ("server.info", self._r_info),
                 ("GET", "/openapi.json"): ("openapi", self._r_openapi),
                 ("GET", "/api/mcp"): ("mcp.list", self._r_mcp),
                 ("GET", "/api/model"): ("model.list", self._r_models),
                 ("POST", "/api/location/reload"): ("location.reload", self._r_reload),
                 ("GET", "/api/event"): ("event.subscribe", self._r_event),
                 ("GET", "/api/session/active"): ("session.active", self._r_active),
                 ("POST", "/api/session"): ("session.create", self._r_create)}
        if (command, path) in fixed:
            operation, method = fixed[(command, path)]
            return operation, method, None, None
        match = _SESSION_ROUTE.match(path)
        if not match:
            return None
        session_id, rest = match.group(1), match.group(2) or ""
        tail = rest.split("/")
        table = {("GET", ""): ("session.get", self._r_session),
                 ("POST", "prompt"): ("session.prompt", self._r_prompt),
                 ("POST", "model"): ("session.switchModel", self._r_switch),
                 ("POST", "interrupt"): ("session.interrupt", self._r_interrupt),
                 ("GET", "message"): ("session.message.list", self._r_messages),
                 ("GET", "inbox"): ("session.inbox.list", self._r_inbox),
                 ("GET", "permission"): ("session.permission.list", self._r_permissions)}
        if (command, rest) in table:
            operation, method = table[(command, rest)]
            return operation, method, session_id, rest
        if command == "POST" and len(tail) == 3 and tail[0] == "permission" and tail[2] == "reply":
            return "session.permission.reply", self._r_permission_reply, session_id, tail[1]
        return None

    def _next_outcome(self, operation: str):
        with self.lock:
            if operation in self.blocked:
                return self.blocked[operation]
            queue = self.scripts.get(operation)
            outcome = queue.pop(0) if queue else None
        if isinstance(outcome, tuple) and outcome[:1] == ("status",):     # the 1.19.5 fake's ("status", 400) form
            outcome = int(outcome[1])
        return outcome

    def _run_hooks(self, operation: str) -> None:
        with self.lock:                                # one hook per answer, in registration order
            hooks = self.hooks.get(operation)
            hook = hooks.pop(0) if hooks else None
        if hook is not None:
            hook()

    def _session(self, handler, session_id: str) -> dict:
        """The session record, or answer 400 (malformed id) / 404 (unknown) and stop the route."""
        if not session_id.startswith("ses"):
            self._json(handler, *_invalid('Expected a string starting with "ses"\n  at ["sessionID"]', "Params"))
            raise _Answered
        with self.lock:
            record = self.sessions.get(session_id)
        if record is None:
            self._json(handler, *_error(404, "SessionNotFoundError", f"Session not found: {session_id}",
                                        sessionID=session_id))
            raise _Answered
        return record

    def _drop(self, handler) -> None:
        handler.close_connection = True
        try:
            handler.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        raise _Answered

    def _wait_gate(self) -> None:
        gate = self.busy_gate
        if gate is not None:
            gate.wait(10)

    # ---- routes ----------------------------------------------------------------------------------------------
    def _r_info(self, handler, query, body, session_id, rest, outcome):
        with self.lock:
            stall = self.health_stalls > 0
            if stall:
                self.health_stalls -= 1
            starting = self.starting_polls > 0
            if starting:
                self.starting_polls -= 1
        if stall:
            time.sleep(self.health_stall_s)
        info = {"version": self.version, "pid": self.pid, "urls": [self.url],
                "paths": {"tmp": "/tmp/fake-opencode/tmp/opencode"}}
        if starting:                                   # real 503 body: version/pid/urls/paths only
            return self._json(handler, 503, info)
        info = {"version": self.version, "pid": self.pid, "hostname": "fake-host", "urls": [self.url],
                "paths": info["paths"], "capabilities": {"persistentPty": True}}
        return self._json(handler, 200, info)

    def _r_openapi(self, handler, query, body, session_id, rest, outcome):
        paths: dict = {}
        for route in self.routes:
            method, path = route.split(" ", 1)
            paths.setdefault(path, {})[method.lower()] = {"operationId": route}
        return self._json(handler, 200, {"openapi": "3.1.0", "info": {"title": "opencode", "version": self.version},
                                         "paths": paths})

    def _r_mcp(self, handler, query, body, session_id, rest, outcome):
        with self.lock:
            pending = self.mcp_pending_polls > 0
            if pending:
                self.mcp_pending_polls -= 1
            servers = json.loads(json.dumps(self.mcp_servers))
        if pending:
            servers = [dict(server, status={"status": "pending"}) for server in servers]
        return self._json(handler, 200, {"location": {"directory": self.directory}, "data": servers})

    def _r_models(self, handler, query, body, session_id, rest, outcome):
        with self.lock:
            models = json.loads(json.dumps(self.models))
        return self._json(handler, 200, {"location": {"directory": self.directory}, "data": models})

    def _r_reload(self, handler, query, body, session_id, rest, outcome):
        hook = self.reload_hook
        if hook is not None:
            hook()
        self.push("location.shutdown", {})
        if outcome == "drop":
            self._drop(handler)
        return self._empty(handler, 204)

    def _r_active(self, handler, query, body, session_id, rest, outcome):
        self._wait_gate()
        with self.lock:
            active = json.loads(json.dumps(self.active))
        return self._json(handler, 200, {"data": active})

    def _r_create(self, handler, query, body, session_id, rest, outcome):
        body = body if isinstance(body, dict) else {}
        model = body.get("model")
        if model is not None and not (isinstance(model, dict) and isinstance(model.get("id"), str)
                                      and isinstance(model.get("providerID"), str)):
            return self._json(handler, *_invalid('Missing key\n  at ["model"]["providerID"]', "Payload"))
        title = body.get("title") or "New session"
        new_id = self.add_session(title, parent_id=body.get("parentID"), agent=body.get("agent"), model=model)
        if outcome == "drop":
            self._drop(handler)
        return self._json(handler, 200, {"data": self.info(new_id)})

    def _r_session(self, handler, query, body, session_id, rest, outcome):
        self._session(handler, session_id)
        return self._json(handler, 200, {"data": self.info(session_id)})

    def _r_prompt(self, handler, query, body, session_id, rest, outcome):
        self._session(handler, session_id)
        if not isinstance(body, dict) or "text" not in body:
            return self._json(handler, *_invalid('Missing key\n  at ["text"]', "Payload"))
        if not isinstance(body["text"], str):
            return self._json(handler, *_invalid('Expected string\n  at ["text"]', "Payload"))
        delivery = body.get("delivery") or "steer"
        if delivery not in ("steer", "queue"):
            return self._json(handler, *_invalid('Expected Session.Inbox.Delivery\n  at ["delivery"]', "Payload"))
        receipt = self.admit_prompt(session_id, body["text"], delivery=delivery)
        if outcome == "drop":
            self._drop(handler)
        self._json(handler, 200, {"data": receipt})
        if outcome == "auto":
            self.finish_turn(session_id, self.auto_text)

    def _r_switch(self, handler, query, body, session_id, rest, outcome):
        record = self._session(handler, session_id)
        model = body.get("model") if isinstance(body, dict) else None
        if not isinstance(model, dict):
            return self._json(handler, *_invalid('Missing key\n  at ["model"]', "Payload"))
        for key in ("id", "providerID"):
            if not isinstance(model.get(key), str):
                return self._json(handler, *_invalid(f'Missing key\n  at ["model"]["{key}"]', "Payload"))
        suffix = _suffix()
        with self.lock:
            previous = dict(record["info"].get("model") or {"id": self.default_model, "providerID": "relay"})
            record["info"]["model"] = dict(model, variant=model.get("variant") or "default")
        self.push("session.model.selected", {"sessionID": session_id, "model": dict(model), "previous": previous},
                  event_id="evt_" + suffix, session_id=session_id)
        self.append(session_id, {"time": {"created": _now()}, "type": "model-switched", "model": dict(model),
                                 "previous": previous}, suffix=suffix)
        if outcome == "drop":
            self._drop(handler)
        return self._empty(handler, 204)

    def _r_interrupt(self, handler, query, body, session_id, rest, outcome):
        self._session(handler, session_id)
        interrupted = self.interrupt_session(session_id)
        if outcome == "drop":
            self._drop(handler)
        return self._json(handler, 200, {"interrupted": interrupted})

    def _r_messages(self, handler, query, body, session_id, rest, outcome):
        record = self._session(handler, session_id)
        limit_values, order_values = query.get("limit"), query.get("order")
        cursor_values, type_values = query.get("cursor"), query.get("type")
        limit = DEFAULT_PAGE
        if limit_values:
            try:
                limit = int(limit_values[0])
            except ValueError:
                return self._json(handler, *_invalid('Expected a finite number\n  at ["limit"]', "Query"))
            if limit > MAX_PAGE:
                return self._json(handler, *_invalid(f'Expected a value less than or equal to {MAX_PAGE}\n'
                                                     '  at ["limit"]', "Query"))
            if limit < 1:
                return self._json(handler, *_invalid('Expected a value greater than or equal to 1\n  at ["limit"]',
                                                     "Query"))
        if order_values and order_values[0] not in ("asc", "desc"):
            return self._json(handler, *_invalid('Expected "asc" | "desc"\n  at ["order"]', "Query"))
        if cursor_values and order_values is not None:
            return self._json(handler, *_error(400, "InvalidCursorError", "Cursor cannot be combined with order"))
        cursor = None
        if cursor_values:
            cursor = _decode_cursor(cursor_values[0])
            if cursor is None:
                return self._json(handler, *_error(400, "InvalidCursorError", "Invalid cursor"))
        order = cursor["order"] if cursor else (order_values[0] if order_values else "desc")
        self._wait_gate()
        with self.lock:
            items = json.loads(json.dumps(record["messages"]))
        if type_values:
            items = [item for item in items if item.get("type") == type_values[0]]
        if order == "desc":
            items.reverse()
        if cursor is not None:
            ids = [item["id"] for item in items]
            if cursor["id"] not in ids:
                return self._json(handler, *_error(400, "InvalidCursorError", "Invalid cursor"))
            position = ids.index(cursor["id"])
            items = items[position + 1:][:limit] if cursor["direction"] == "next" else \
                items[max(0, position - limit):position]
        else:
            items = items[:limit]
        page = {"data": items, "cursor": {
            "previous": _encode_cursor(items[0]["id"], order, "previous") if items else None,
            "next": _encode_cursor(items[-1]["id"], order, "next") if items else None}}
        return self._json(handler, 200, page)

    def _r_inbox(self, handler, query, body, session_id, rest, outcome):
        self._session(handler, session_id)
        return self._json(handler, 200, {"data": self.inbox_of(session_id)})

    def _r_permissions(self, handler, query, body, session_id, rest, outcome):
        record = self._session(handler, session_id)
        with self.lock:
            pending = json.loads(json.dumps(record["permissions"]))
        return self._json(handler, 200, {"data": pending})

    def _r_permission_reply(self, handler, query, body, session_id, request_id, outcome):
        record = self._session(handler, session_id)
        if not isinstance(body, dict) or "decision" not in body:
            return self._json(handler, *_invalid('Missing key\n  at ["decision"]', "Payload"))
        decision = body["decision"]
        if decision not in ("once", "always", "reject"):
            return self._json(handler, *_invalid('Expected Permission.Reply\n  at ["decision"]', "Payload"))
        with self.lock:
            request = next((item for item in record["permissions"] if item["id"] == request_id), None)
            if request is not None:
                record["permissions"].remove(request)
        if request is None:
            return self._json(handler, *_error(404, "PermissionNotFoundError",
                                               f"Permission request not found: {request_id}", requestID=request_id))
        self.push("permission.replied", {"sessionID": session_id, "requestID": request_id, "reply": decision})
        if decision == "reject":
            self._reject_tool(session_id, request.get("source"))
        if outcome == "drop":
            self._drop(handler)
        return self._empty(handler, 204)

    def _reject_tool(self, session_id: str, source: dict | None) -> None:
        """A rejected permission (2.2.7, probe b07): the tool fails ``aborted`` "The user declined this tool call",
        the step fails "Step interrupted", the execution ends ``interrupted`` (reason ``shutdown``) with no ``idle``
        message and no outcome change; the session accepts the next prompt normally."""
        with self.lock:
            message = next((item for item in self.sessions[session_id]["messages"]
                            if source and item.get("id") == source.get("messageID")), None)
            tool = next((item for item in (message or {}).get("content", ())
                         if item.get("type") == "tool" and item.get("id") == source.get("id")), None)
        if tool is not None:
            self.finish_tool(session_id, message, tool["id"],
                             error={"type": "aborted", "message": "The user declined this tool call"})
        if message is not None and "finish" not in message:
            self.complete(session_id, message, error={"type": "aborted", "message": "Step interrupted"})
        with self.lock:
            was_active = self.active.pop(session_id, None) is not None
        if was_active:
            self.push("session.execution.interrupted", {"sessionID": session_id, "reason": "shutdown"},
                      session_id=session_id)

    def _r_event(self, handler, query, body, session_id, rest, outcome):
        types = query.get("types")
        if types is not None:
            unknown = [item for item in types if item not in EVENT_TYPES]
            if unknown:
                return self._json(handler, *_invalid("Expected " + " | ".join(sorted(EVENT_TYPES))[:200]
                                                     + '\n  at ["types"]', "Query"))
        subscriber = _Subscriber(handler, None if types is None else frozenset(types))
        handler.send_response(200)
        handler.send_header("cache-control", "no-cache, no-transform")
        handler.send_header("x-accel-buffering", "no")
        handler.send_header("Content-Type", "text/event-stream")
        handler.send_header("Transfer-Encoding", "chunked")
        handler.end_headers()
        handler.close_connection = True
        connected = {"id": "evt_" + _suffix(), "type": "server.connected", "data": {}}
        subscriber.send(("data: " + json.dumps(connected, separators=(",", ":")) + "\n\n").encode())
        subscriber.send(b": heartbeat\n\n")
        with self.lock:
            self._subscribers.append(subscriber)
        last = time.monotonic()
        while not subscriber.closed.wait(0.05):
            if self.heartbeat_s and time.monotonic() - last >= self.heartbeat_s:
                last = time.monotonic()
                subscriber.send(b": heartbeat\n\n")
        with self.lock:
            if subscriber in self._subscribers:
                self._subscribers.remove(subscriber)
        raise _Answered

    # ---- answers ----------------------------------------------------------------------------------------------
    def _json(self, handler, status: int, payload, *, extra: dict | None = None) -> None:
        data = json.dumps(payload).encode()
        handler.send_response(status)
        handler.send_header("Content-Type", "application/json")
        handler.send_header("Content-Length", str(len(data)))
        for key, value in (extra or {}).items():
            handler.send_header(key, value)
        handler.end_headers()
        handler.wfile.write(data)

    def _empty(self, handler, status: int) -> None:
        handler.send_response(status)
        handler.send_header("Content-Length", "0")
        handler.end_headers()


class _Subscriber:
    def __init__(self, handler, types: frozenset | None):
        self.handler, self.types = handler, types
        self.closed = threading.Event()
        self._lock = threading.Lock()

    def accepts(self, event_type: str) -> bool:
        return self.types is None or event_type in self.types

    def send(self, data: bytes) -> None:
        with self._lock:
            if self.closed.is_set():
                return
            try:
                self.handler.wfile.write(f"{len(data):x}\r\n".encode() + data + b"\r\n")
                self.handler.wfile.flush()
            except OSError:
                self.closed.set()

    def finish(self) -> None:
        with self._lock:
            if not self.closed.is_set():
                try:
                    self.handler.wfile.write(b"0\r\n\r\n")
                    self.handler.wfile.flush()
                except OSError:
                    pass

    def close(self) -> None:
        self.closed.set()
        try:
            self.handler.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass


def _checked_in_routes() -> list[str]:
    """The checked-in ``opencode_routes.json`` operation list (served as ``GET /openapi.json``)."""
    if ROUTES_JSON.exists():
        return list(json.loads(ROUTES_JSON.read_text())["routes"])
    return []
