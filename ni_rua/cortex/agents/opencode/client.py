"""Blocking HTTP client for ``opencode serve`` 2.2.7 (design 11 §1.5, §1.10; the server API lives under ``/api/*``).

Stdlib ``http.client`` only, HTTP basic auth (user ``opencode`` + the per-run ``OPENCODE_PASSWORD`` of the launch
environment), one short-lived connection per request so any number of threads may call it concurrently.  The client
never retries anything.  Every shape below was captured from the pinned 2.2.7 binary (probe outputs under
``/tmp/ni-rua-agent-harness/oc2/runs/``; OpenAPI document ``GET /openapi.json``, 170 operations).

Error model (the only thing callers need to decide whether a request may have been admitted):

* ``OpenCodeHttpError``       the server answered with a non-2xx status.  ``definite_rejection`` is True for 4xx:
                              OpenCode explicitly refused the request and nothing was admitted (2.x error bodies are
                              typed: ``{"_tag": "SessionNotFoundError" | "InvalidRequestError" | ..., "message"}``,
                              exposed as ``tag``).  5xx is *not* a definite rejection (the request may have been
                              processed before the failure).
* ``OpenCodeTransportError``  no usable HTTP answer (connection refused/reset, timeout, truncated or unparsable
                              answer).  It is an ``OSError``.  For ``prompt`` this means *unknown admission*: the
                              caller must stop, never resend.
* ``OpenCodeStreamEnded``     the ``/api/event`` stream ended (a transport error subclass).

``/api/event`` is advisory only (volatile by contract: no replay, events during a disconnection are lost);
``event_stream`` reconnects with backoff and reports every gap so the caller can reconcile from ``status()``,
``messages()`` and ``inbox()``.
"""
from __future__ import annotations

import base64
import http.client
import json
import re
import socket
import threading
from typing import Callable, Iterator, Mapping, Sequence
from urllib.parse import quote, urlencode, urlsplit

__all__ = [
    "OpenCodeError", "OpenCodeHttpError", "OpenCodeTransportError", "OpenCodeStreamEnded", "OpenCodeClient",
    "is_definite_rejection", "routes_from_openapi", "model_ref", "OPENCODE_USERNAME", "PROMPT_DELIVERIES",
    "PERMISSION_DECISIONS", "MESSAGE_PAGE_MAX",
]

OPENCODE_USERNAME = "opencode"          # fixed basic-auth user name of `opencode serve` (password: OPENCODE_PASSWORD)
PROMPT_DELIVERIES = ("steer", "queue")  # Session.Inbox.Delivery
PERMISSION_DECISIONS = ("once", "always", "reject")  # Permission.Reply
MESSAGE_PAGE_MAX = 200                  # largest `limit` GET /api/session/{id}/message accepts (201 -> HTTP 400)
_SESSION_ID = re.compile(r"ses[A-Za-z0-9_]*\Z")      # Session.ID pattern ^ses
_MESSAGE_ID = re.compile(r"msg_[A-Za-z0-9_]*\Z")     # Session.Message.ID pattern ^msg_
_PERMISSION_ID = re.compile(r"per[A-Za-z0-9_]*\Z")   # Permission.ID pattern ^per
_DATA_URI = re.compile(r"^data:([A-Za-z0-9.+-]+/[A-Za-z0-9.+-]+);base64,([A-Za-z0-9+/=]+)\Z")
_HTTP_METHODS = ("get", "put", "post", "delete", "options", "head", "patch", "trace")
_MAX_BODY_BYTES = 256 << 20             # cap of one non-streaming answer (a message page carries inline images)
_MAX_EVENT_BYTES = 64 << 20             # cap of one SSE frame (session.tool.success carries the tool's images)
_READ_SIZE = 1 << 16


class OpenCodeError(Exception):
    """Base class of the client errors."""


class OpenCodeHttpError(OpenCodeError):
    """Non-2xx HTTP answer.  ``status`` is the HTTP status, ``body`` the decoded JSON body (else text, else None)."""

    def __init__(self, status: int, body: object = None, *, method: str = "", path: str = ""):
        super().__init__(f"opencode_http_{status}")
        self.status, self.body, self.method, self.path = status, body, method, path

    @property
    def definite_rejection(self) -> bool:
        """True for 4xx: the server refused the request, nothing was admitted (safe to treat as rejected)."""
        return 400 <= self.status < 500

    @property
    def tag(self) -> str | None:
        """The typed error name of a 2.x error body (``_tag``, e.g. ``"SessionNotFoundError"``); None otherwise."""
        tag = self.body.get("_tag") if isinstance(self.body, dict) else None
        return tag if isinstance(tag, str) else None


class OpenCodeTransportError(OpenCodeError, OSError):
    """No usable HTTP answer; admission of a state-changing request is unknown.  ``__cause__`` has the original."""

    admission_unknown = True

    def __init__(self, reason: str):
        OpenCodeError.__init__(self, reason)
        self.reason = reason


class OpenCodeStreamEnded(OpenCodeTransportError):
    """The ``/api/event`` stream reached its end (server closed it, the connection dropped or was shut down)."""


def is_definite_rejection(error: BaseException) -> bool:
    """True only for an HTTP 4xx answer.  Everything else (5xx, transport error, timeout) is unknown admission."""
    return isinstance(error, OpenCodeHttpError) and error.definite_rejection


def routes_from_openapi(document: Mapping) -> list[str]:
    """Sorted ``"METHOD /path"`` operation list of an OpenAPI document (the format of ``opencode_routes.json``)."""
    paths = document.get("paths") if isinstance(document, Mapping) else None
    if not isinstance(paths, Mapping):
        raise ValueError("openapi_paths_missing")
    routes = []
    for path, item in paths.items():
        if isinstance(item, Mapping):
            routes.extend(f"{method.upper()} {path}" for method in item if method in _HTTP_METHODS)
    return sorted(routes)


def model_ref(provider_id: str, model_id: str, variant: str | None = None) -> dict:
    """An OpenCode ``Model.Ref``: ``{"id": <model id>, "providerID": <provider id>[, "variant": <variant id>]}``.

    ``provider_id``: the config provider key (NI-RUA: ``"relay"``).  ``model_id``: a model declared under that
    provider.  ``variant``: one of the model's declared effort variants; None = no variant field (OpenCode then
    reports ``"variant": "default"`` in the session info).  The server does not validate the model when it is set
    (an undeclared id fails the next prompt with ``provider.no-route``)."""
    ref = {"id": model_id, "providerID": provider_id}
    if variant is not None:
        ref["variant"] = variant
    return _checked_model_ref(ref)


def _checked_file_uris(files: Sequence[str]) -> list[str]:
    """校验 canonical base64 data: URI（禁止换行与非规范载荷），返回 URI 列表。"""
    if isinstance(files, (str, bytes)) or not isinstance(files, Sequence):
        raise ValueError("opencode_prompt_files_invalid")
    uris = []
    for uri in files:
        if not isinstance(uri, str) or not _DATA_URI.match(uri):
            raise ValueError("opencode_prompt_file_uri_invalid")
        uris.append(uri)
    return uris


def _checked_file_attachments(files: Sequence[str | Mapping[str, object]]) -> list[dict[str, object]]:
    """校验并规范化 ``PromptInput.FileAttachment`` 列表（支持规范 base64 ``data:`` URI 字符串或含 ``uri`` 的字典）。"""
    if isinstance(files, (str, bytes)) or not isinstance(files, Sequence):
        raise ValueError("opencode_prompt_files_invalid")
    attachments: list[dict[str, object]] = []
    for entry in files:
        if isinstance(entry, str):
            if not _DATA_URI.match(entry):
                raise ValueError("opencode_prompt_file_uri_invalid")
            attachments.append({"uri": entry})
        elif isinstance(entry, Mapping):
            uri = entry.get("uri")
            if not isinstance(uri, str) or not uri:
                raise ValueError("opencode_prompt_files_invalid")
            if not _DATA_URI.match(uri):
                raise ValueError("opencode_prompt_file_uri_invalid")
            attachments.append(dict(entry))
        else:
            raise ValueError("opencode_prompt_file_uri_invalid")
    return attachments


def _checked_model_ref(value: object) -> dict:
    if not isinstance(value, Mapping) or set(value) - {"id", "providerID", "variant"}:
        raise ValueError("opencode_model_ref_invalid")
    ref = {key: value[key] for key in ("id", "providerID", "variant") if key in value}
    for key in ("id", "providerID", "variant"):
        if (key in ref or key != "variant") and not (isinstance(ref.get(key), str) and ref[key].strip()):
            raise ValueError("opencode_model_ref_invalid")
    return ref


class _UnixHTTPConnection(http.client.HTTPConnection):
    """``HTTPConnection`` whose transport is an ``AF_UNIX`` stream socket (the host side of an inbound socket)."""

    def __init__(self, host: str, port: int, unix_socket: str, timeout: float):
        super().__init__(host, port, timeout=timeout)
        self._unix_socket = unix_socket

    def connect(self) -> None:
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(self.timeout)
        try:
            sock.connect(self._unix_socket)
        except BaseException:
            sock.close()
            raise
        self.sock = sock


class OpenCodeClient:
    """Typed wrapper over the ``/api/*`` routes of OpenCode 2.2.7 that NI-RUA uses.  Thread-safe; no retries.

    Parameters
    ----------
    base_url:
        ``http://127.0.0.1:<port>`` of the server (no credentials, no path).
    password:
        The per-run server password (``OPENCODE_PASSWORD`` of the launch environment).  ``username`` defaults to the
        fixed ``opencode``.  Every route, ``/api/info`` and ``/openapi.json`` included, answers 401 without it.
    timeout_s:
        Socket timeout (s) of every non-streaming request, default 30.  A timeout is a transport error.
    stream_timeout_s:
        Maximum silence (s) on the ``/api/event`` stream before it counts as dropped, default 35 (the 2.2.7 server
        sends a ``: heartbeat`` comment right after connecting and then every 15 s).
    unix_socket:
        Host path of a Unix socket that reaches the server (``AgentSandbox.inbound_socket_path`` for a sandboxed
        server).  Default ``None``: plain TCP to ``base_url``.  With a socket the connection goes there; ``base_url``
        still supplies the ``Host`` header (``127.0.0.1:<port>``, the in-sandbox listen address).  The API, the
        authentication and the error model are identical.
    """

    def __init__(self, base_url: str, password: str, *, username: str = OPENCODE_USERNAME, timeout_s: float = 30.0,
                 stream_timeout_s: float = 35.0, unix_socket: str | None = None):
        parts = urlsplit(base_url)
        if parts.scheme != "http" or not parts.hostname or parts.username or parts.path not in ("", "/") \
                or parts.query or parts.fragment:
            raise ValueError("opencode_base_url_invalid")
        self.host, self.port = parts.hostname, parts.port or 80
        self.timeout_s, self.stream_timeout_s = float(timeout_s), float(stream_timeout_s)
        self.unix_socket = None if unix_socket is None else str(unix_socket)
        token = base64.b64encode(f"{username}:{password}".encode("utf-8")).decode("ascii")
        self._auth = "Basic " + token
        self._streams: set = set()
        self._lock = threading.Lock()
        self.invalid_event_frames = 0

    # ------------------------------------------------------------------------------------------ transport
    def _connect(self, timeout: float) -> http.client.HTTPConnection:
        if self.unix_socket is not None:
            return _UnixHTTPConnection(self.host, self.port, self.unix_socket, timeout)
        return http.client.HTTPConnection(self.host, self.port, timeout=timeout)

    def _headers(self, extra: Mapping[str, str] | None = None) -> dict:
        headers = {"Authorization": self._auth, "Accept": "application/json"}
        headers.update(extra or {})
        return headers

    def _request(self, method: str, path: str, body: object = None, *, timeout: float | None = None,
                 want_body: bool = True, allow_status: Sequence[int] = ()) -> object:
        """One request.  Returns decoded JSON (None for an empty body).  A status in ``allow_status`` returns
        ``(status, decoded body)`` instead of raising.  Raises OpenCodeHttpError / OpenCodeTransportError."""
        data = None if body is None else json.dumps(body, ensure_ascii=False).encode("utf-8")
        headers = self._headers({"Content-Type": "application/json"} if data is not None else None)
        connection = self._connect(self.timeout_s if timeout is None else timeout)
        try:
            try:
                connection.request(method, path, body=data, headers=headers)
                response = connection.getresponse()
                status = response.status
                raw = response.read(_MAX_BODY_BYTES + 1)
            except (OSError, http.client.HTTPException) as error:
                raise OpenCodeTransportError(type(error).__name__) from error
        finally:
            connection.close()
        if len(raw) > _MAX_BODY_BYTES:
            raise OpenCodeTransportError("response_too_large")
        if status in allow_status:
            return status, _decode_loose(raw)
        if not 200 <= status < 300:
            raise OpenCodeHttpError(status, _decode_loose(raw), method=method, path=path)
        if not want_body or not raw.strip():
            return None
        try:
            return json.loads(raw)
        except ValueError as error:
            raise OpenCodeTransportError("response_not_json") from error

    @staticmethod
    def _session_path(session_id: str) -> str:
        if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
            raise ValueError("opencode_session_id_invalid")
        return "/api/session/" + quote(session_id, safe="")

    @staticmethod
    def _expect_mapping(result: object) -> dict:
        if not isinstance(result, dict):
            raise OpenCodeTransportError("response_not_object")
        return result

    def _data(self, result: object, kind: type) -> object:
        """The ``data`` member of a 2.x ``{"data": ...}`` envelope, which must be a ``kind``."""
        data = self._expect_mapping(result).get("data")
        if not isinstance(data, kind):
            raise OpenCodeTransportError("response_data_invalid")
        return data

    # ------------------------------------------------------------------------------------------- server
    def health(self, *, timeout: float | None = None) -> dict | None:
        """Readiness probe ``GET /api/info``.

        Returns the ``ServerInfo`` (``{"version": "2.2.7", "pid", "hostname", "urls", "paths": {"tmp"},
        "capabilities"}``) when the server answers 200, i.e. is ready.  Returns ``None`` while it answers HTTP 503
        (still starting; the 503 body carries only version/pid/urls/paths).  Any other non-2xx answer raises
        ``OpenCodeHttpError`` (401 = wrong password); no answer raises ``OpenCodeTransportError``.

        ``timeout`` (s): per-request socket timeout override, used by startup polling so one stalled early
        connection (observed on Linux, ~30 s) does not consume the whole startup budget.  Default ``None`` keeps
        the client's configured ``timeout_s``.
        """
        result = self._request("GET", "/api/info", timeout=timeout, allow_status=(503,))
        if isinstance(result, tuple):
            return None
        return self._expect_mapping(result)

    def routes(self) -> list[str]:
        """``GET /openapi.json`` reduced to the sorted ``"METHOD /path"`` operation list (170 for 2.2.7)."""
        try:
            return routes_from_openapi(self._expect_mapping(self._request("GET", "/openapi.json")))
        except ValueError as error:
            raise OpenCodeTransportError(str(error)) from error

    def mcp_status(self) -> dict[str, dict]:
        """``GET /api/mcp`` -> ``{server name: status}`` with status ``{"status": "connected" | "pending" |
        "disabled" | "failed" | "needs_auth"[, "error": <text>]}``.  Right after start a configured remote server
        is ``pending`` (and may be missing for a few hundred ms); NI-RUA waits for ``ni_rua`` to be ``connected``
        before the first prompt."""
        servers = {}
        for entry in self._data(self._request("GET", "/api/mcp"), list):
            if not isinstance(entry, dict) or not isinstance(entry.get("name"), str) \
                    or not isinstance(entry.get("status"), dict):
                raise OpenCodeTransportError("mcp_status_invalid")
            servers[entry["name"]] = entry["status"]
        return servers

    def models(self) -> list[dict]:
        """``GET /api/model`` -> the chat models the server offers, each a ``Model.Info``: ``{"id", "modelID",
        "providerID", "name", "package", "settings", "capabilities": {"tools", "input", "output"}, "variants":
        [{"id", "body" | "settings"}], "limit": {"context", "output"}, "status", "enabled", "time", "cost"}``.
        With the ``provider.use`` policy only ``relay`` models are listed."""
        return self._data(self._request("GET", "/api/model"), list)

    def reload(self, *, timeout_s: float | None = None) -> None:
        """``POST /api/location/reload`` (204): rebuild every loaded location (re-reads the config; pending
        permissions are cancelled; running sessions continue with fresh services at the next step boundary).
        It answers once the rebuild settled, so ``timeout_s`` (s, default ``timeout_s`` of the client) may need to
        cover an MCP reconnect.  Fallback when a rewritten config file was not hot-reloaded (design 11 §2.4)."""
        self._request("POST", "/api/location/reload", timeout=timeout_s, want_body=False)

    # ------------------------------------------------------------------------------------------- sessions
    def create_session(self, title: str, *, model: Mapping[str, str] | None = None) -> str:
        """``POST /api/session {title[, model]}`` -> the new session id (``ses_...``).

        ``title``: explicit non-empty title (no title-model call).  ``model``: the session's initial model as a
        ``Model.Ref`` (``model_ref(...)``); None = the config's default ``model``."""
        if not isinstance(title, str) or not title.strip():
            raise ValueError("opencode_session_title_required")
        body: dict = {"title": title}
        if model is not None:
            body["model"] = _checked_model_ref(model)
        session_id = self._data(self._request("POST", "/api/session", body), dict).get("id")
        if not isinstance(session_id, str) or not _SESSION_ID.match(session_id):
            raise OpenCodeTransportError("session_id_missing")
        return session_id

    def session_info(self, session_id: str) -> dict:
        """``GET /api/session/{id}`` -> ``Session.Info``: ``{"id", "parentID"?, "agent"?, "model"?: {"id",
        "providerID", "variant"}, "outcome"?: "succeeded" | "failed" | "interrupted", "time": {"created", "updated",
        "busy"?, "idle"?}, "title", "tokens", "cost", "location": {"directory"}, ...}``.  A missing session raises
        ``OpenCodeHttpError`` 404 (``SessionNotFoundError``)."""
        return self._data(self._request("GET", self._session_path(session_id)), dict)

    def session_exists(self, session_id: str) -> bool:
        """``GET /api/session/{id}``: True on 200, False on 404; any other failure raises."""
        try:
            self.session_info(session_id)
        except OpenCodeHttpError as error:
            if error.status == 404:
                return False
            raise
        return True

    def prompt(self, session_id: str, text: str, *, delivery: str | None = None,
               files: Sequence[str | Mapping[str, object]] | None = None) -> dict:
        """``POST /api/session/{id}/prompt {text[, delivery][, files]}``: durably admit one user message.

        Returns the admission receipt (``Session.Inbox.User``): ``{"id": "msg_...", "sessionID", "time":
        {"created"}, "type": "user", "payload": {"text"[, "files"]}, "delivery"}``.  ``id`` is the id the user message
        will carry in ``messages()`` once delivered (the turn boundary).  The execution runs in the background: an idle
        session starts at once; on an active session the message waits in ``inbox()`` and joins at the next step
        boundary (``delivery="steer"``, the server default when None) or after the current execution ends
        (``"queue"``).  ``files`` 为规范 base64 ``data:`` URI 字符串序列或 ``PromptInput.FileAttachment`` 字典序列
        （每项至少包含合法 ``uri``），非空时作为 ``body["files"]`` 发送。
        No model/variant/agent/system field exists per prompt in 2.x: the model is session state
        (``switch_model``) and instructions travel in the config ``instructions`` file.

        Raises ``OpenCodeHttpError`` for a non-2xx answer (4xx = definite rejection, e.g. 404
        ``SessionNotFoundError``, 400 ``InvalidRequestError``, 409 ``ConflictError``) and ``OpenCodeTransportError``
        for socket errors/timeouts or a 2xx answer without a receipt id (unknown admission).  Never retried.
        """
        if not isinstance(text, str) or not text:
            raise ValueError("opencode_prompt_text_required")
        if delivery is not None and delivery not in PROMPT_DELIVERIES:
            raise ValueError("opencode_prompt_delivery_invalid")
        attachments = _checked_file_attachments(files) if files is not None else []
        body: dict = {"text": text}
        if delivery is not None:
            body["delivery"] = delivery
        if attachments:
            body["files"] = attachments
        result = self._request("POST", self._session_path(session_id) + "/prompt", body)
        receipt = result.get("data") if isinstance(result, dict) else None
        if not isinstance(receipt, dict) or not isinstance(receipt.get("id"), str) \
                or not _MESSAGE_ID.match(receipt["id"]):
            raise OpenCodeTransportError("prompt_receipt_invalid")
        return receipt

    def switch_model(self, session_id: str, model: Mapping[str, str]) -> None:
        """``POST /api/session/{id}/model {model}`` (204): the model (and variant) of subsequent provider turns.
        Writes a ``model-switched`` message; while busy it applies at the next step.  ``model`` is a ``Model.Ref``
        (``model_ref(...)``); OpenCode accepts an undeclared id here and fails the next prompt instead."""
        self._request("POST", self._session_path(session_id) + "/model", {"model": _checked_model_ref(model)},
                      want_body=False)

    def interrupt(self, session_id: str, *, timeout_s: float | None = None) -> bool:
        """``POST /api/session/{id}/interrupt`` -> ``interrupted``: True when an active execution was interrupted
        (synchronous; cascades to foreground subagents, kills running tool shells), False for the idle no-op.
        ``timeout_s`` (s): socket timeout override, default the client's ``timeout_s``."""
        result = self._expect_mapping(self._request("POST", self._session_path(session_id) + "/interrupt",
                                                    timeout=timeout_s))
        interrupted = result.get("interrupted")
        if not isinstance(interrupted, bool):
            raise OpenCodeTransportError("interrupt_result_invalid")
        return interrupted

    def status(self) -> dict:
        """``GET /api/session/active`` -> ``{session_id: {"type": "running"}}`` for every session with an active
        execution in this server (subagent sessions appear under their own ids); a session absent from it is
        inactive.  ``{}`` = everything idle."""
        return self._data(self._request("GET", "/api/session/active"), dict)

    def messages(self, session_id: str, *, page_size: int = MESSAGE_PAGE_MAX) -> list:
        """``GET /api/session/{id}/message`` -> the whole timeline, oldest first: a flat list of typed messages
        (``Session.Message.Info``: ``type`` ``user`` | ``synthetic`` | ``assistant`` | ``compaction`` | ``idle`` |
        ``model-switched`` | ``agent-switched`` | ``location-switched`` | ``system`` | ``skill`` | ``shell``).

        Pages of ``page_size`` messages (1..200, default 200): the first request sends ``order=asc``, every later
        one only ``cursor=<cursor.next>`` (combining both is HTTP 400).  Reading stops at a short or empty page.
        The pages are separate reads, so a message written meanwhile may appear in a later page; the result is not
        an atomic snapshot (callers that need one compare two reads)."""
        if not isinstance(page_size, int) or isinstance(page_size, bool) or not 1 <= page_size <= MESSAGE_PAGE_MAX:
            raise ValueError("opencode_message_page_size_invalid")
        base = self._session_path(session_id) + "/message?"
        query: dict = {"limit": page_size, "order": "asc"}
        seen: set = set()
        result: list = []
        while True:
            page = self._expect_mapping(self._request("GET", base + urlencode(query)))
            data, cursor = page.get("data"), page.get("cursor")
            if not isinstance(data, list):
                raise OpenCodeTransportError("messages_data_invalid")
            result.extend(data)
            following = cursor.get("next") if isinstance(cursor, dict) else None
            if len(data) < page_size or not isinstance(following, str) or not following:
                return result
            if following in seen:
                raise OpenCodeTransportError("messages_cursor_loop")
            seen.add(following)
            query = {"limit": page_size, "cursor": following}

    def inbox(self, session_id: str) -> list:
        """``GET /api/session/{id}/inbox`` -> admitted work not yet delivered, in enqueue order: ``[{"id", "type":
        "user" | "synthetic" | "compaction" | "move", "payload", "delivery": "steer" | "queue", "time"}]``.  A steer
        sent while the session runs, or a background subagent's report, waits here until the next step boundary."""
        return self._data(self._request("GET", self._session_path(session_id) + "/inbox"), list)

    def permissions(self, session_id: str) -> list:
        """``GET /api/session/{id}/permission`` -> pending permission requests of that session (``Permission.Request``:
        ``{"id": "per_...", "sessionID", "action", "resources": [...], "save", "source": {"type": "tool",
        "messageID", "id"}}``).  The snapshot counterpart of the ``permission.asked`` event."""
        return self._data(self._request("GET", self._session_path(session_id) + "/permission"), list)

    def permission_reply(self, session_id: str, request_id: str, decision: str) -> None:
        """``POST /api/session/{id}/permission/{request id}/reply {decision}`` (204).

        ``decision``: ``"once"`` | ``"always"`` | ``"reject"``.  A reject fails the asking tool call (error
        ``aborted`` "The user declined this tool call") and ends the execution (``session.execution.interrupted``,
        reason ``shutdown``) without an ``idle`` message (2.2.7, probed).  An unknown or already answered request
        raises ``OpenCodeHttpError`` 404 (``PermissionNotFoundError``)."""
        if not isinstance(request_id, str) or not _PERMISSION_ID.match(request_id):
            raise ValueError("opencode_permission_id_invalid")
        if decision not in PERMISSION_DECISIONS:
            raise ValueError("opencode_permission_decision_invalid")
        self._request("POST", self._session_path(session_id) + "/permission/" + quote(request_id, safe="")
                      + "/reply", {"decision": decision}, want_body=False)

    # --------------------------------------------------------------------------------------------- event stream
    def events(self, *, types: Sequence[str] | None = None) -> Iterator[dict]:
        """One ``GET /api/event`` connection: yields decoded frames ``{"id", "created"?, "type", "location"?,
        "data": {...}, "durable"?: {"aggregateID", "seq", "version"}}``; the first is ``server.connected``.

        ``types``: event types to receive (repeated ``types`` query parameter; an unknown type is HTTP 400); None =
        all.  ``server.connected`` arrives regardless.  The body is chunked (``Transfer-Encoding: chunked``) and
        decoded by ``http.client``; a frame may span chunks.  Comment lines (``: heartbeat``) and fields other than
        ``data`` are skipped; undecodable or oversized (> 64 MiB) frames are skipped and counted in
        ``invalid_event_frames``.  Raises ``OpenCodeHttpError`` for a non-200 answer, ``OpenCodeStreamEnded`` at the
        end of the stream and ``OpenCodeTransportError`` on errors or ``stream_timeout_s`` of silence.
        ``close_streams()`` ends it from another thread."""
        path = "/api/event"
        if types is not None:
            if isinstance(types, str) or not all(isinstance(item, str) and item for item in types):
                raise ValueError("opencode_event_types_invalid")
            path += "?" + urlencode([("types", item) for item in types])
        connection = self._connect(self.stream_timeout_s)
        with self._lock:
            self._streams.add(connection)
        try:
            try:
                connection.request("GET", path, headers=self._headers({"Accept": "text/event-stream"}))
                response = connection.getresponse()
                if response.status != 200:
                    raise OpenCodeHttpError(response.status, _decode_loose(response.read(1 << 16)), method="GET",
                                            path=path)
                yield from self._frames(response)
            except OpenCodeError:
                raise
            except http.client.IncompleteRead as error:
                raise OpenCodeStreamEnded("event_stream_incomplete") from error
            except (OSError, http.client.HTTPException, ValueError) as error:
                raise OpenCodeTransportError(type(error).__name__) from error
        finally:
            with self._lock:
                self._streams.discard(connection)
            connection.close()

    def _frames(self, response: http.client.HTTPResponse) -> Iterator[dict]:
        """SSE decoding over the de-chunked body (``read1`` returns what the current chunk holds)."""
        buffer, scan, skipping = bytearray(), 0, False
        data_lines: list[bytes] = []
        data_size, oversized = 0, False
        while True:
            newline = buffer.find(b"\n", scan)
            if newline < 0:
                if len(buffer) > _MAX_EVENT_BYTES:      # one line beyond the frame cap: drop it up to its newline
                    del buffer[:]
                    skipping = True
                scan = len(buffer)
                chunk = response.read1(_READ_SIZE)
                if not chunk:
                    raise OpenCodeStreamEnded("event_stream_eof")
                buffer += chunk
                continue
            line = bytes(buffer[:newline])
            del buffer[:newline + 1]
            scan = 0
            if skipping:
                skipping, oversized, data_lines, data_size = False, True, [], 0
                continue
            if line.endswith(b"\r"):
                line = line[:-1]
            if line:
                if line.startswith(b":"):
                    continue                            # comment, e.g. ": heartbeat"
                field, _, value = line.partition(b":")
                if field == b"data" and not oversized:
                    value = value[1:] if value.startswith(b" ") else value
                    data_lines.append(value)
                    data_size += len(value)
                    if data_size > _MAX_EVENT_BYTES:
                        oversized, data_lines, data_size = True, [], 0
                continue
            if oversized:
                oversized = False
                self.invalid_event_frames += 1
                continue
            if not data_lines:
                continue
            payload, data_lines, data_size = b"\n".join(data_lines), [], 0
            try:
                frame = json.loads(payload)
            except ValueError:
                self.invalid_event_frames += 1
                continue
            if isinstance(frame, dict):
                yield frame
            else:
                self.invalid_event_frames += 1

    def event_stream(self, stop: threading.Event, *, on_gap: Callable[[BaseException], None] | None = None,
                     initial_backoff_s: float = 0.1, max_backoff_s: float = 2.0,
                     types: Sequence[str] | None = None) -> Iterator[dict]:
        """Reconnecting ``/api/event`` reader for a reader thread: yields frames until ``stop`` is set.

        After every drop or failed connect ``on_gap(error)`` is called (the caller reconciles from ``status()``,
        ``messages()`` and ``inbox()``, because events missed while disconnected are not replayed), then the reader
        waits ``backoff`` (s, doubling from ``initial_backoff_s`` to ``max_backoff_s``; reset by the next received
        frame) and reconnects.  ``types`` is passed to ``events``.  ``on_gap`` exceptions propagate.
        """
        backoff = initial_backoff_s
        while not stop.is_set():
            try:
                for frame in self.events(types=types):
                    backoff = initial_backoff_s
                    yield frame
                    if stop.is_set():
                        return
            except OpenCodeError as error:
                if stop.is_set():
                    return
                if on_gap is not None:
                    on_gap(error)
            if stop.wait(backoff):
                return
            backoff = min(backoff * 2, max_backoff_s)

    def close_streams(self) -> None:
        """Shut down every open ``/api/event`` connection (unblocks readers in other threads)."""
        with self._lock:
            streams = list(self._streams)
        for connection in streams:
            sock = connection.sock
            if sock is not None:
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


def _decode_loose(raw: bytes) -> object:
    text = raw.decode("utf-8", "replace").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except ValueError:
        return text[:2000]
