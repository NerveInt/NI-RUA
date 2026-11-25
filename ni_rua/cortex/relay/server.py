"""Buffered HTTP model relay server: connection pool, worker attempts, upstream bindings, and service_tier."""

from __future__ import annotations

import base64
import hashlib
import http.client
import inspect
import json
import os
from pathlib import Path
import queue
import re
import select
import signal
import socket
import subprocess
import sys
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlsplit

if __package__:
    from ...core.roboprobe_source import joint_policy
    from ...core.storage.store import Store
    from ..session.wire import OpenAIResponsesWire, WireAdapter, wire_from_name
    from ..session.wire.base import provider_error_message
    from .continuity import MAIN as MAIN_CALL, ContinuityTracker
    from .guard import (
        CODEX_PROFILE_NAME,
        DEFAULT_MAX_RETRIES,
        RETRY_POLICY,
        TRANSPORT_RECOVERY_KEYS,
        ContinuityAdmission,
        GuardMixin,
        IdleRun,
    )
    from .payload import (
        ACTOR_UPSTREAM,
        AUTH_HEADERS,
        CLIENT_AUTH_HEADERS,
        HOP_HEADERS,
        MIN_TRIM_IMAGE_LIMIT,
        PRIVATE_HEADERS,
        PROVIDER_CONTEXT_EDIT,
        TOOL_RESULT_IMAGE_PLACEMENTS,
        UPSTREAM_NAME,
        UPSTREAM_PREFIX,
        PayloadMixin,
        UpstreamBinding,
        _RESPONSES,
        _request_target,
        _terminal_sse_block,
        _terminal_sse_type,
        load_request_body,
        upstream_auth_headers_copy,
    )
else:  # ``--http-worker`` runs this file as a script: import adapters from this same checkout.
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from ni_rua.core.roboprobe_source import joint_policy
    from ni_rua.core.storage.store import Store
    from ni_rua.cortex.relay.continuity import MAIN as MAIN_CALL, ContinuityTracker
    from ni_rua.cortex.relay.guard import (
        CODEX_PROFILE_NAME,
        DEFAULT_MAX_RETRIES,
        RETRY_POLICY,
        TRANSPORT_RECOVERY_KEYS,
        ContinuityAdmission,
        GuardMixin,
        IdleRun,
    )
    from ni_rua.cortex.relay.payload import (
        ACTOR_UPSTREAM,
        AUTH_HEADERS,
        CLIENT_AUTH_HEADERS,
        HOP_HEADERS,
        MIN_TRIM_IMAGE_LIMIT,
        PRIVATE_HEADERS,
        PROVIDER_CONTEXT_EDIT,
        TOOL_RESULT_IMAGE_PLACEMENTS,
        UPSTREAM_NAME,
        UPSTREAM_PREFIX,
        PayloadMixin,
        UpstreamBinding,
        _RESPONSES,
        _request_target,
        _terminal_sse_block,
        _terminal_sse_type,
        load_request_body,
        upstream_auth_headers_copy,
    )
    from ni_rua.cortex.session.wire import OpenAIResponsesWire, WireAdapter, wire_from_name
    from ni_rua.cortex.session.wire.base import provider_error_message

MAX_CLIENT_BODY_BYTES = 128 * 1024 * 1024  # declared client request body bound (bytes)
REFUSAL_DRAIN_TIMEOUT_S = 30.0
AUX_REQUEST_TIMEOUT_S = 120.0


def _invoke_on_tool_calls(callback, thread, calls, call_purpose):
    """调用 on_tool_calls 并透传 call_purpose，同时兼容仅接受两参数的既有测试桩。"""
    try:
        params = list(inspect.signature(callback).parameters.values())
    except (TypeError, ValueError):
        return callback(thread, calls, call_purpose)
    positional = 0
    for param in params:
        if param.kind == inspect.Parameter.VAR_POSITIONAL:
            return callback(thread, calls, call_purpose)
        if param.kind in (inspect.Parameter.POSITIONAL_ONLY, inspect.Parameter.POSITIONAL_OR_KEYWORD):
            positional += 1
    if positional >= 3:
        return callback(thread, calls, call_purpose)
    return callback(thread, calls)


def _kill(process):
    if hasattr(process, "kill_inprocess"):
        process.kill_inprocess()
        return
    if process is not None and process.poll() is None:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass


_DEFAULT_POPEN = subprocess.Popen


class _InProcessAttempt:
    """In-process HTTP attempt handle matching subprocess cancellation and wait APIs."""

    def __init__(self):
        self.lock = threading.Lock()
        self.conn = None
        self.killed = False
        self.done = threading.Event()
        self.returncode = None

    def set_connection(self, conn):
        with self.lock:
            self.conn = conn
            if self.killed and conn is not None:
                self._force_close(conn)

    @staticmethod
    def _force_close(conn):
        sock = getattr(conn, "sock", None)
        if sock is not None:
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
        try:
            conn.close()
        except OSError:
            pass

    def kill_inprocess(self):
        with self.lock:
            self.killed = True
            conn, self.conn = self.conn, None
        if conn is not None:
            self._force_close(conn)

    def poll(self):
        return self.returncode if self.done.is_set() else None

    def wait(self, timeout=None):
        self.done.wait(timeout)
        return self.returncode

    def communicate(self, *_args, **_kwargs):
        self.done.wait(0.2)
        return "", ""


def _http_worker():
    """One HTTP attempt in an independently killable process; stdout emits JSON event lines."""
    payload = json.load(sys.stdin)
    wire = wire_from_name(payload.get("wire", OpenAIResponsesWire.name), client=payload.get("client"))
    started = time.monotonic()

    def event(kind, **fields):
        print(
            json.dumps(dict(kind=kind, wall_time=time.time(), elapsed_s=time.monotonic() - started, **fields)),
            flush=True,
        )

    url = urlsplit(payload["url"])
    connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
    connection = connection_type(url.hostname, url.port, timeout=None)
    body = base64.b64decode(payload["body"])
    phase, incoming, first_byte = "connecting_upstream", 0, None
    response = None
    raw = bytearray()
    try:
        event("model_attempt_phase", phase=phase)
        connection.connect()
        event("model_upstream_connected")
        phase = "sending_upstream_body"
        event("model_attempt_phase", phase=phase)
        connection.request("POST", _request_target(url), body=body, headers=payload["headers"])
        event("model_request_forwarded", body_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest())
        phase = "awaiting_response_headers"
        event("model_attempt_phase", phase=phase)
        response = connection.getresponse()
        event("model_response_headers", http_status=response.status)
        phase = "streaming_response"
        event("model_attempt_phase", phase=phase)
        pending, terminal_seen = b"", False
        while chunk := response.read1(65536):
            event("model_response_bytes_received")
            raw.extend(chunk)
            incoming += len(chunk)
            if first_byte is None:
                first_byte = time.monotonic() - started
                event("model_response_first_byte")
            pending += chunk
            while match := re.search(br"\r?\n\r?\n", pending):
                block, pending = pending[:match.start()], pending[match.end():]
                if wire.terminal_sse_type(block) is not None:
                    terminal_seen = True
                    if pending:
                        del raw[-len(pending):]
                    break
            if terminal_seen:
                event("model_response_terminal_received")
                break
        if not terminal_seen and response.length not in (None, 0):
            raise http.client.IncompleteRead(b"", response.length)
        result = dict(
            status=response.status,
            headers=dict(response.getheaders()),
            body=base64.b64encode(raw).decode(),
            terminal_seen=terminal_seen,
        )
    except Exception as error:
        result = dict(
            error_type=type(error).__name__,
            phase=phase,
            body=base64.b64encode(raw).decode(),
            status=response.status if response is not None else None,
            headers=dict(response.getheaders()) if response is not None else {},
        )
    finally:
        connection.close()
    print(json.dumps(dict(result=result, incoming_bytes=incoming, first_upstream_byte_s=first_byte)), flush=True)


class ModelRequestProxy(GuardMixin, PayloadMixin):
    """Loopback model relay owning generic budget, retry/recovery, continuity, and upstream bindings."""

    RUNAWAY_MAX_IDENTICAL_ADMISSIONS = 8
    RUNAWAY_WINDOW_S = 2.0
    CONSUME_GUARD_MAX_IDLE_COMPLETIONS = 3
    SHUTDOWN_TERMINAL_GRACE_S = 2.0
    SETTLE_TIMEOUT_S = Store.LOCK_WAIT_S

    def __init__(
        self,
        upstream_url,
        *,
        max_requests=180,
        request_timeout_s=None,
        max_retries=None,
        on_event=None,
        on_limit=None,
        metrics_path=None,
        image_reuse=False,
        image_trim=True,
        request_projection=None,
        upstream_auth_headers=None,
        request_context=None,
        max_inline_images=0,
        transport_recovery=None,
        recovery_gate=None,
        wire=None,
        default_thread_id=None,
        on_context_discontinuity=None,
        tool_result_images="inline",
        on_tool_calls=None,
        policy=None,
        classify_call_purpose=False,
    ):
        if wire is not None and not isinstance(wire, WireAdapter):
            raise ValueError("model_proxy_wire_requires_adapter")
        if tool_result_images not in TOOL_RESULT_IMAGE_PLACEMENTS:
            raise ValueError("model_proxy_invalid_tool_result_images")
        if type(classify_call_purpose) is not bool:
            raise ValueError("model_proxy_classify_call_purpose_requires_boolean")
        self.tool_result_images = tool_result_images
        self.classify_call_purpose = classify_call_purpose
        self.wire = OpenAIResponsesWire() if wire is None else wire
        if default_thread_id is not None and (
            not isinstance(default_thread_id, str)
            or not default_thread_id
            or len(default_thread_id) > 256
            or "\n" in default_thread_id
            or "\r" in default_thread_id
        ):
            raise ValueError("model_proxy_invalid_default_thread_id")
        self.default_thread_id = default_thread_id
        if on_context_discontinuity is not None and not callable(on_context_discontinuity):
            raise ValueError("model_proxy_context_discontinuity_requires_callable")
        self.on_context_discontinuity = on_context_discontinuity
        if on_tool_calls is not None and not callable(on_tool_calls):
            raise ValueError("model_proxy_tool_calls_requires_callable")
        self.on_tool_calls = on_tool_calls
        self.continuity = None if self.wire.client == CODEX_PROFILE_NAME else ContinuityTracker()
        self._upstream_auth_headers = upstream_auth_headers_copy(upstream_auth_headers, self.wire.auth_header_names)
        self._client_auth_headers = frozenset(CLIENT_AUTH_HEADERS) | frozenset(self.wire.auth_header_names)
        self._check_upstream_url(upstream_url)
        if type(max_requests) is not int or max_requests < 1:
            raise ValueError("model_proxy_invalid_request_budget")
        if request_timeout_s is not None:
            raise ValueError("model_proxy_request_deadlines_disabled")
        self.policy = joint_policy() if policy is None else policy
        self.max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        if type(self.max_retries) is not int or not 0 <= self.max_retries <= DEFAULT_MAX_RETRIES:
            raise ValueError("model_proxy_invalid_retry_budget")
        if type(image_reuse) is not bool:
            raise ValueError("model_proxy_image_reuse_requires_boolean")
        self.image_reuse = image_reuse
        if type(image_trim) is not bool:
            raise ValueError("model_proxy_image_trim_requires_boolean")
        self.image_trim = image_trim
        if type(max_inline_images) is not int or not (max_inline_images == 0 or max_inline_images >= MIN_TRIM_IMAGE_LIMIT):
            raise ValueError("model_proxy_invalid_inline_image_limit")
        self.max_inline_images = max_inline_images
        if transport_recovery is not None:
            if not isinstance(transport_recovery, dict) or transport_recovery.keys() != TRANSPORT_RECOVERY_KEYS:
                raise ValueError("model_proxy_invalid_transport_recovery")
            transport_recovery = dict(transport_recovery) if transport_recovery["max_rounds"] > 0 else None
        if recovery_gate is not None and not callable(recovery_gate):
            raise ValueError("model_proxy_recovery_gate_requires_callable")
        self.transport_recovery, self.recovery_gate = transport_recovery, recovery_gate
        if request_projection is not None and not callable(request_projection):
            raise ValueError("model_proxy_request_projection_requires_callable")
        self.request_projection = request_projection
        if request_context is not None and not callable(request_context):
            raise ValueError("model_proxy_request_context_requires_callable")
        self.request_context = request_context
        self.upstream_url = upstream_url.rstrip("/")
        self._upstreams = {}
        self.max_requests, self.request_timeout_s = max_requests, None
        self.on_event, self.on_limit = on_event, on_limit
        self.metrics_path = Path(metrics_path) if metrics_path else None
        if self.metrics_path:
            self.metrics_path.parent.mkdir(parents=True, exist_ok=True)
            self.metrics_path.touch(exist_ok=False)
            self.metrics_path.chmod(0o600)
        self.lock = threading.RLock()
        self.request_count = self.attempt_count = self.logical_call_count = self.reserved_calls = 0
        self.transport_id = uuid.uuid4().hex
        self.image_context_epoch = None
        self.limit_notified = False
        self.events, self.active = [], {}
        self.thread_owners, self.cancelled_threads = {}, {}
        self._image_trim_watermarks = {}
        self._image_trim_starts = {}
        self._runaway_runs = {}
        self._idle_runs = {}
        self._reopenable_threads = set()
        self._aux_connections = set()
        self.aux_request_count = 0
        self._recovery_rounds = self._recovered_calls = self._recovery_denials = 0
        self._recovery_delay_s = 0.0
        self._wait_started_at = None
        self._wait_total_s = 0.0
        self._last_activity_at = None
        self.closed = False
        self.blocked_error = None
        self.server = self.thread = self.base_url = None
        self.callbacks = queue.Queue()
        self._callback_counts = threading.Lock()
        self._callbacks_queued = self._callbacks_run = 0
        self.callback_thread = None
        self._conn_pool = {}
        self._conn_pool_lock = threading.Lock()
        self.service_tier = None
        self.service_tier_probes = {}
        self._service_tier_probes_model = None
        self._service_tier_probes_by_model = {}

    @staticmethod
    def _check_upstream_url(upstream_url):
        url = urlsplit(upstream_url) if isinstance(upstream_url, str) else None
        if (
            url is None
            or url.scheme not in ("http", "https")
            or not url.hostname
            or url.username
            or url.password
            or url.query
            or url.fragment
        ):
            raise ValueError("model_proxy_requires_http_base_url_without_credentials_or_query")

    @property
    def actor_binding(self):
        """The default ``UpstreamBinding`` (constructor wire, ``upstream_url`` and credentials)."""
        return UpstreamBinding(
            ACTOR_UPSTREAM,
            self.wire,
            self.upstream_url,
            self._upstream_auth_headers,
            self._client_auth_headers,
        )

    def add_upstream(self, name, *, wire, upstream_url, upstream_auth_headers, presentation):
        """Front one more upstream on this relay under ``/_ni_rua/u/<name>``."""
        if not isinstance(name, str) or not UPSTREAM_NAME.fullmatch(name) or name == ACTOR_UPSTREAM:
            raise ValueError("model_proxy_invalid_upstream_name")
        if not isinstance(wire, WireAdapter):
            raise ValueError("model_proxy_wire_requires_adapter")
        if presentation is None:
            raise ValueError("model_proxy_upstream_requires_presentation")
        self._check_upstream_url(upstream_url)
        binding = UpstreamBinding(
            name,
            wire,
            upstream_url.rstrip("/"),
            upstream_auth_headers_copy(upstream_auth_headers, wire.auth_header_names),
            frozenset(CLIENT_AUTH_HEADERS) | frozenset(wire.auth_header_names),
            presentation,
            UPSTREAM_PREFIX + name,
        )
        with self.lock:
            if self.closed:
                raise ValueError("model_proxy_already_started_or_closed")
            if name in self._upstreams:
                raise ValueError("model_proxy_upstream_already_bound")
            self._upstreams[name] = binding
        self._record(
            "model_upstream_bound",
            upstream=name,
            wire_api=wire.name,
            relay_profile=wire.client,
            credentials=sorted(binding.auth_headers or ()),
        )
        return self.upstream_base_url(name)

    def upstream_base_url(self, name):
        """Relay base URL a host-side client uses for binding ``name``, or None before ``start()``."""
        if name != ACTOR_UPSTREAM:
            with self.lock:
                if name not in self._upstreams:
                    raise KeyError(name)
        if self.base_url is None:
            return None
        return self.base_url if name == ACTOR_UPSTREAM else self.base_url + UPSTREAM_PREFIX + name

    def upstream_binding(self, name):
        """``UpstreamBinding`` for ``name`` (``"actor"`` included); ``KeyError`` when unknown."""
        if name == ACTOR_UPSTREAM:
            return self.actor_binding
        with self.lock:
            return self._upstreams[name]

    def _binding_for_path(self, raw_path):
        """``(binding, path_within_binding)`` for a client request target; ``(None, None)`` when unknown."""
        if not isinstance(raw_path, str) or not raw_path.startswith(UPSTREAM_PREFIX):
            return self.actor_binding, raw_path
        name, slash, rest = raw_path[len(UPSTREAM_PREFIX):].partition("/")
        with self.lock:
            binding = self._upstreams.get(name)
        if binding is None or not slash:
            return None, None
        return binding, "/" + rest

    @staticmethod
    def _socket_alive(conn):
        sock = getattr(conn, "sock", None)
        if sock is None:
            return False
        try:
            readable, _, _ = select.select([sock], [], [], 0)
            return not readable
        except (OSError, ValueError):
            return False

    def _acquire_upstream_conn(self, url):
        key = (url.scheme, url.hostname, url.port)
        with self._conn_pool_lock:
            bucket = self._conn_pool.get(key)
            while bucket:
                conn = bucket.pop()
                if self._socket_alive(conn):
                    return conn, True
                try:
                    conn.close()
                except OSError:
                    pass
        connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        return connection_type(url.hostname, url.port, timeout=None), False

    def _release_upstream_conn(self, url, conn, response):
        if conn is None or response is None:
            if conn is not None:
                try:
                    conn.close()
                except OSError:
                    pass
            return
        if getattr(response, "will_close", True):
            try:
                conn.close()
            except OSError:
                pass
            return
        sock = getattr(conn, "sock", None)
        if not response.isclosed() and sock is not None:
            try:
                sock.settimeout(0.005)
                response.read()
            except Exception:
                pass
            finally:
                try:
                    sock.settimeout(None)
                except OSError:
                    pass
        if not response.isclosed() or not self._socket_alive(conn):
            try:
                conn.close()
            except OSError:
                pass
            return
        key = (url.scheme, url.hostname, url.port)
        with self._conn_pool_lock:
            if self.closed:
                try:
                    conn.close()
                except OSError:
                    pass
                return
            self._conn_pool.setdefault(key, []).append(conn)

    def _close_conn_pool(self):
        with self._conn_pool_lock:
            buckets = list(self._conn_pool.values())
            self._conn_pool.clear()
        for bucket in buckets:
            for conn in bucket:
                try:
                    conn.close()
                except OSError:
                    pass

    def _inprocess_http(self, attempt, identity, url_str, body, headers, started, wire=None):
        wire = self.wire if wire is None else wire

        def emit(kind, **fields):
            elapsed = time.monotonic() - started
            if kind == "model_response_bytes_received":
                with self.lock:
                    self._last_activity_at = time.monotonic()
                return
            self._record(kind, **identity, elapsed_s=elapsed, **fields)

        url = urlsplit(url_str)
        req_headers = dict(headers)
        if not any(k.lower() == "connection" for k in req_headers):
            req_headers["Connection"] = "keep-alive"
        phase, incoming, first_byte = "connecting_upstream", 0, None
        response = None
        connection = None
        reused_ok = False
        raw = bytearray()
        try:
            emit("model_attempt_phase", phase=phase)
            if identity.get("attempt_index", 1) > 1:
                self._close_conn_pool()
            connection, was_reused = self._acquire_upstream_conn(url)
            attempt.set_connection(connection)
            if not was_reused:
                connection.connect()
            emit("model_upstream_connected")
            phase = "sending_upstream_body"
            emit("model_attempt_phase", phase=phase)
            forwarded_emitted = False
            awaiting_emitted = False
            try:
                connection.request("POST", _request_target(url), body=body, headers=req_headers)
                emit("model_request_forwarded", body_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest())
                forwarded_emitted = True
                phase = "awaiting_response_headers"
                emit("model_attempt_phase", phase=phase)
                awaiting_emitted = True
                response = connection.getresponse()
            except Exception:
                if not was_reused or attempt.killed:
                    raise
                try:
                    connection.close()
                except OSError:
                    pass
                phase = "connecting_upstream"
                connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
                connection = connection_type(url.hostname, url.port, timeout=None)
                attempt.set_connection(connection)
                connection.connect()
                phase = "sending_upstream_body"
                connection.request("POST", _request_target(url), body=body, headers=req_headers)
                if not forwarded_emitted:
                    emit("model_request_forwarded", body_bytes=len(body), body_sha256=hashlib.sha256(body).hexdigest())
                phase = "awaiting_response_headers"
                if not awaiting_emitted:
                    emit("model_attempt_phase", phase=phase)
                response = connection.getresponse()
            emit("model_response_headers", http_status=response.status)
            phase = "streaming_response"
            emit("model_attempt_phase", phase=phase)
            pending, terminal_seen, terminal_type = b"", False, None
            while chunk := response.read1(65536):
                emit("model_response_bytes_received")
                raw.extend(chunk)
                incoming += len(chunk)
                if first_byte is None:
                    first_byte = time.monotonic() - started
                    emit("model_response_first_byte")
                pending += chunk
                while match := re.search(br"\r?\n\r?\n", pending):
                    block, pending = pending[:match.start()], pending[match.end():]
                    if (t_type := wire.terminal_sse_type(block)) is not None:
                        terminal_seen = True
                        terminal_type = t_type
                        if pending:
                            del raw[-len(pending):]
                        break
                if terminal_seen:
                    emit("model_response_terminal_received")
                    break
            if not terminal_seen and response.length not in (None, 0):
                raise http.client.IncompleteRead(b"", response.length)
            result = dict(
                status=response.status,
                headers=dict(response.getheaders()),
                body=base64.b64encode(raw).decode(),
                terminal_seen=terminal_seen,
            )
            is_sse = wire.is_event_stream(raw)
            reused_ok = (
                not attempt.killed
                and 200 <= response.status < 400
                and (wire.terminal_allows_reuse(terminal_type) if is_sse else not terminal_seen)
            )
        except Exception as error:
            result = dict(
                error_type=type(error).__name__,
                phase=phase,
                body=base64.b64encode(raw).decode(),
                status=response.status if response is not None else None,
                headers=dict(response.getheaders()) if response is not None else {},
            )
        finally:
            attempt.set_connection(None)
            if reused_ok:
                self._release_upstream_conn(url, connection, response)
            elif connection is not None:
                try:
                    connection.close()
                except OSError:
                    pass
            attempt.returncode = 0
            attempt.done.set()
        return dict(result=result, incoming_bytes=incoming, first_upstream_byte_s=first_byte), phase

    def _record(self, kind, **fields):
        event = {"kind": kind, "wall_time": time.time(), "transport_schema_version": "2", **fields}
        with self.lock:
            if kind in {
                "model_call_started",
                "model_request_started",
                "model_request_body_received",
                "model_upstream_connected",
                "model_request_forwarded",
                "model_response_headers",
                "model_response_first_byte",
                "model_response_terminal_received",
                "model_request_finished",
                "model_call_completed",
            }:
                self._last_activity_at = time.monotonic()
            if (
                kind == "model_call_started"
                and "upstream" not in fields
                and self.wire.route_for_endpoint(fields.get("endpoint")).compaction
            ):
                self.image_context_epoch = fields.get("logical_call_id") or uuid.uuid4().hex
                thread_id = fields.get("native_thread_id") or ""
                self._image_trim_watermarks.pop(thread_id, None)
                self._image_trim_starts.pop(thread_id, None)
            self.events.append(event)
            if self.metrics_path:
                with self.metrics_path.open("a") as stream:
                    stream.write(json.dumps(event, allow_nan=False) + "\n")
        if self.on_event:
            self._queue_callback(self.on_event, event)
        return event

    def _queue_callback(self, callback, event):
        with self._callback_counts:
            self._callbacks_queued += 1
        self.callbacks.put((callback, event, True))

    def _callbacks(self):
        while (work := self.callbacks.get()) is not None:
            callback, event, counted = work
            try:
                callback(event)
            except Exception:
                pass
            finally:
                if counted:
                    with self._callback_counts:
                        self._callbacks_run += 1

    def _callbacks_all_run(self):
        with self._callback_counts:
            return self._callbacks_queued == self._callbacks_run

    def settle(self, timeout_s=None):
        """Wait until the owner's records have caught up with every answer the relay has delivered so far."""
        deadline = time.monotonic() + (self.SETTLE_TIMEOUT_S if timeout_s is None else max(0.0, timeout_s))
        with self.lock:
            delivering = [state for state in self.active.values() if state.get("delivering")]
        for state in delivering:
            if not state["finished"].wait(max(0.0, deadline - time.monotonic())):
                return False
        thread = self.callback_thread
        if thread is None:
            return self._callbacks_all_run()
        if thread is threading.current_thread():
            return False
        reached = threading.Event()
        self.callbacks.put((lambda _event: reached.set(), None, False))
        while not reached.wait(min(0.05, max(0.0, deadline - time.monotonic()))):
            if not thread.is_alive():
                return self._callbacks_all_run()
            if time.monotonic() >= deadline:
                return False
        return True

    @staticmethod
    def _shutdown(connection):
        if connection is not None:
            try:
                connection.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass
            try:
                connection.close()
            except OSError:
                pass

    def _cancel(self, state, reason):
        with self.lock:
            if state["cancel"].is_set() or state["finished"].is_set():
                return
            if reason == "client_disconnected" and state.get("delivered"):
                return
            state["cancel_reason"] = reason
            state["cancel"].set()
            if reason == "client_disconnected":
                state["provider_explicit_error"] = False
                state["failure_evidence"] = dict(
                    classification="downstream_disconnected",
                    response_completion="unknown",
                )
                self._block_after_failure(state, "ClientDisconnected")
            _kill(state.get("process"))
            if state.get("reading_client_body"):
                self._shutdown(state["client"])
        self._record("model_call_cancelled", **state["identity"], reason=reason)

    def select_model(self, model):
        """Switch ``service_tier_probes`` view to ``model``; clear ``service_tier`` if proven unsupported."""
        if not isinstance(model, str) or not model:
            return self.service_tier_probes
        with self.lock:
            if self._service_tier_probes_model is None and self.service_tier_probes and not self._service_tier_probes_by_model:
                self._service_tier_probes_by_model[model] = self.service_tier_probes
            self._service_tier_probes_model = model
            self.service_tier_probes = self._service_tier_probes_by_model.setdefault(model, {})
            if self.service_tier and (self.service_tier_probes.get(self.service_tier) or {}).get("supported") is False:
                self.service_tier = None
            return self.service_tier_probes

    def probe_service_tier(self, model, tier, timeout_s=60):
        """Send one tiny Responses request asking for ``tier`` and report the tier actually served."""
        probes = self.select_model(model) if isinstance(model, str) and model else self.service_tier_probes
        if tier in probes:
            return probes[tier]
        import urllib.error
        import urllib.request

        body = json.dumps({
            "model": model,
            "input": "Reply with: ok",
            "max_output_tokens": 16,
            "reasoning": {"effort": "low"},
            "service_tier": tier,
            "stream": True,
        }).encode()
        headers = {
            "Content-Type": "application/json",
            "Accept": "text/event-stream",
            "User-Agent": "ni-rua-native-model-relay/1",
            **(self._upstream_auth_headers or {}),
        }
        served = error = provider_error = None
        completed = False
        try:
            request = urllib.request.Request(self.upstream_url + "/responses", data=body, method="POST", headers=headers)
            with urllib.request.urlopen(request, timeout=timeout_s) as response:
                for line in response:
                    line = line.decode(errors="replace").strip()
                    if not line.startswith("data:"):
                        continue
                    try:
                        event = json.loads(line[5:])
                    except ValueError:
                        continue
                    if not isinstance(event, dict):
                        continue
                    if event.get("type") in ("response.completed", "response.failed", "response.incomplete", "error"):
                        served = (event.get("response") or {}).get("service_tier")
                        completed = event.get("type") == "response.completed"
                        if not completed:
                            error = f"probe_{event['type']}"
                        break
        except urllib.error.HTTPError as caught:
            try:
                payload = json.loads(caught.read(65536))
            except (ValueError, OSError, http.client.HTTPException):
                payload = None
            finally:
                caught.close()
            detail = payload.get("error") if isinstance(payload, dict) else None
            tier_rejected = isinstance(detail, dict) and (
                detail.get("param") == "service_tier"
                or re.search(r"\bservice_tier\b", str(detail.get("message", "")), re.I)
            )
            if caught.code in (400, 422) and tier_rejected:
                result = dict(
                    tier=tier,
                    supported=False,
                    served_tier=None,
                    error=None,
                    rejection=f"HTTP {caught.code}: service_tier rejected",
                )
                probes[tier] = result
                return result
            if isinstance(detail, dict):
                mapped = self._failed_response_error(payload)
                provider_error = dict(
                    http_status=caught.code,
                    code=mapped["code"],
                    type=mapped["type"],
                    classification=mapped["classification"],
                )
            label = (provider_error or {}).get("code") or (provider_error or {}).get("type")
            error = f"HTTP {caught.code}{' ' + label if label else ''}: service_tier acceptance unknown"
        except Exception as caught:
            scrubbed = re.sub(r"([A-Za-z][A-Za-z0-9+.-]*://)[^/\s@]+@", r"\1[redacted]@", str(caught))
            detail = provider_error_message(dict(message=scrubbed), limit=200)
            error = f"{type(caught).__name__}: {detail}" if detail else type(caught).__name__
        if error is None and not completed:
            error = "probe_response_not_completed"
        result = dict(tier=tier, supported=True if error is None else None, served_tier=served, error=error)
        if provider_error is not None:
            result["provider_error"] = provider_error
        if error is None:
            probes[tier] = result
        return result

    def _forward_auxiliary(self, route, body, headers, upstream_url=None):
        """Forward one auxiliary POST without retries under ``AUX_REQUEST_TIMEOUT_S``."""
        url = urlsplit((self.upstream_url if upstream_url is None else upstream_url) + route.upstream_path)
        connection_type = http.client.HTTPSConnection if url.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(url.hostname, url.port, timeout=AUX_REQUEST_TIMEOUT_S)
        with self.lock:
            if self.closed:
                raise ConnectionError("model proxy closed")
            self._aux_connections.add(connection)
        try:
            connection.request("POST", _request_target(url), body=body, headers=headers)
            response = connection.getresponse()
            data = response.read()
            return response.status, response.reason, response.getheaders(), data
        finally:
            with self.lock:
                self._aux_connections.discard(connection)
            connection.close()

    def _watch_client(self, state):
        while not state["finished"].wait(0.05):
            try:
                readable, _, _ = select.select([state["client"]], [], [], 0)
                if readable and not state["client"].recv(1, socket.MSG_PEEK | socket.MSG_DONTWAIT):
                    self._cancel(state, "client_disconnected")
                    return
            except (OSError, ValueError):
                self._cancel(state, "client_disconnected")
                return

    def _attempt(self, state, body, headers, endpoint, retry_index):
        binding = state.get("binding") or self.actor_binding
        wire = binding.wire
        route = state.get("route") or wire.route_for_endpoint(endpoint)
        body, moved_images = self._place_tool_result_images(binding, body)
        for key in (
            "failed_response",
            "provider_explicit_error",
            "capability_failure",
            "failure_evidence",
            "connect_failure",
            "stream_incomplete",
        ):
            state.pop(key, None)
        with self.lock:
            if state["cancel"].is_set():
                raise self.policy.InfrastructureFailure("model call cancelled")
            self.attempt_count += 1
            number = self.attempt_count
        identity = dict(**state["identity"], request_index=number, attempt_index=retry_index + 1)
        if moved_images is not None:
            identity["tool_result_images_moved"] = moved_images
        payload = dict(
            url=binding.upstream_url + endpoint,
            body=base64.b64encode(body).decode(),
            headers=headers,
            wire=wire.name,
        )
        if wire.client != CODEX_PROFILE_NAME:
            payload["client"] = wire.client
        serialized = json.dumps(payload)
        started = time.monotonic()
        process = None
        result, phase = None, "worker_initialization"
        try:
            if subprocess.Popen is _DEFAULT_POPEN:
                process = _InProcessAttempt()
                with self.lock:
                    state["process"] = process
                    if state["cancel"].is_set():
                        _kill(process)
                self._record(
                    "model_request_started",
                    **identity,
                    timeout_s=None,
                    max_retries=self.max_retries,
                    deadline_boundary=None,
                    phase="worker_initialization",
                )
                started = time.monotonic()
                result, phase = self._inprocess_http(
                    process,
                    identity,
                    binding.upstream_url + endpoint,
                    body,
                    headers,
                    started,
                    wire=wire,
                )
            else:
                process = subprocess.Popen(
                    [sys.executable, str(Path(__file__).resolve()), "--http-worker"],
                    stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.DEVNULL,
                    text=True,
                    start_new_session=True,
                )
                with self.lock:
                    state["process"] = process
                    if state["cancel"].is_set():
                        _kill(process)
                self._record(
                    "model_request_started",
                    **identity,
                    timeout_s=None,
                    max_retries=self.max_retries,
                    deadline_boundary=None,
                    phase="worker_initialization",
                )
                started = time.monotonic()
                process.stdin.write(serialized)
                try:
                    process.stdin.close()
                finally:
                    process.stdin = None
                for line in process.stdout:
                    try:
                        event = json.loads(line)
                    except ValueError:
                        continue
                    if "result" in event:
                        result = event
                    elif "kind" in event:
                        phase = event.get("phase", phase)
                        kind = event.pop("kind")
                        if kind == "model_response_bytes_received":
                            with self.lock:
                                self._last_activity_at = time.monotonic()
                            continue
                        self._record(kind, **identity, **event)
                process.wait()
            if state["cancel"].is_set():
                raise self.policy.InfrastructureFailure("model call cancelled")
            if process.returncode != 0 or result is None:
                raise self.policy.InfrastructureFailure("planner HTTP worker failed", retryable=False)
            response = result["result"]
            try:
                _, state["attempt_evidence"], _ = wire.response_evidence(
                    base64.b64decode(response.get("body", "")),
                    route,
                )
            except (ValueError, EOFError, TypeError, AttributeError):
                pass
            raw = base64.b64decode(response["body"])
            status = response.get("status")
            if status is not None and status >= 400:
                state["provider_explicit_error"] = True
                try:
                    error_body = json.loads(raw)
                except ValueError:
                    error_body = None
                detail = error_body if isinstance(error_body, dict) else {}
                state["attempt_evidence"] = dict(
                    usage=None,
                    usage_known=False,
                    response_error=wire.operator_response_error(detail),
                    http_error_status=status,
                )
                state["failure_evidence"] = dict(http_status=status, **state["attempt_evidence"]["response_error"])
                raise wire.classify_http_failure(status, response["headers"], error_body, policy=self.policy)
            if response.get("error_type"):
                phase = response.get("phase", phase)
                connect_failure = phase == "connecting_upstream"
                state["failure_evidence"] = dict(
                    classification="transport_connect_failed" if connect_failure else "transport_incomplete",
                    transport_error_type=response["error_type"],
                    phase=phase,
                )
                state["connect_failure"] = connect_failure
                self._record(
                    "model_request_transport_error",
                    **identity,
                    error_type=response["error_type"],
                    phase=phase,
                )
                raise self.policy.InfrastructureFailure(
                    "model upstream connect failed" if connect_failure else "model upstream transport incomplete",
                    retryable=connect_failure,
                )

            def stream_incomplete(message, detail):
                state["failure_evidence"] = dict(classification="stream_incomplete", http_status=status, detail=detail)
                state["stream_incomplete"] = True
                return self.policy.InfrastructureFailure(message, retryable=True)

            try:
                complete, evidence, terminal = wire.response_evidence(raw, route)
            except EOFError:
                raise stream_incomplete("model response terminal missing", "terminal_missing") from None
            except (ValueError, TypeError, AttributeError):
                raise stream_incomplete("model response malformed", "malformed") from None
            state["attempt_evidence"] = evidence
            outcome = wire.completion_outcome(complete, evidence)
            if outcome in ("failed", "incomplete"):
                failed_raw = wire.failed_payload(terminal, streaming=wire.is_event_stream(raw)) if wire.early_stream else None
                if failed_raw is not None:
                    state["failed_response"] = (failed_raw, response["headers"])
                error = evidence.get("response_error", dict(classification="incomplete"))
                state["failure_evidence"] = dict(http_status=status, **error)
                state["provider_explicit_error"] = outcome == "failed"
                raise self.policy.InfrastructureFailure(
                    "model response failed: " + error["classification"],
                    retryable=bool(error.get("retryable")),
                    retry_after_s=error.get("retry_after_s"),
                )
            if outcome != "usable":
                raise stream_incomplete("model response ended without completed status", "not_completed")
            if self.continuity is not None:
                state["context_edit_applied"] = wire.context_edit_applied(terminal)
            return raw, response["headers"]
        finally:
            _kill(process)
            if process is not None:
                process.communicate()
            with self.lock:
                state["process"] = None
            info = result or {}
            response = info.get("result", {})
            self._record(
                "model_request_finished",
                **identity,
                http_status=response.get("status"),
                elapsed_s=time.monotonic() - started,
                first_upstream_byte_s=info.get("first_upstream_byte_s"),
                incoming_bytes=info.get("incoming_bytes"),
                outgoing_bytes=0,
                transport_complete=bool(result and not response.get("error_type")),
                timed_out=False,
                phase=phase,
                **state.pop("attempt_evidence", dict(usage=None, usage_known=False)),
            )

    def _complete(self, state, body, headers, endpoint):
        binding = state.get("binding") or self.actor_binding
        wire = binding.wire
        image_trim = self.image_trim and binding.name == ACTOR_UPSTREAM
        route = state.get("route") or wire.route_for_endpoint(endpoint)
        if (
            image_trim
            and route.kind == "generation"
            and not route.compaction
            and "base_request" not in state
            and len(body) > wire.body_limit_bytes
        ):
            body = wire.trim_oldest_images(body, 16)
        capability_failure = getattr(self.policy, "CapabilityFailure", ())
        retry_index = rounds = 0
        while True:
            try:
                result = self._attempt(state, body, headers, endpoint, retry_index)
                if rounds:
                    with self.lock:
                        self._recovered_calls += 1
                    self._record(
                        "model_transport_recovered",
                        **state["identity"],
                        recovery_rounds=rounds,
                        attempt_index=retry_index + 1,
                    )
                return result
            except capability_failure as failure:
                if state["cancel"].is_set():
                    raise
                state["provider_explicit_error"] = True
                state["capability_failure"] = True
                return self._terminal_failure(state, failure)
            except self.policy.InfrastructureFailure as failure:
                if state["cancel"].is_set():
                    raise
                image_rejected = wire.image_rejected(state.get("failure_evidence") or {})
                explicit_retryable = bool(
                    (state.get("provider_explicit_error") or state.get("connect_failure") or state.get("stream_incomplete"))
                    and failure.retryable
                    and not (image_rejected and not image_trim)
                )
                recovering = not (explicit_retryable and retry_index < self.max_retries)
                delay = self._recovery_delay(state, rounds, explicit_retryable) if recovering else None
                if recovering and delay is None:
                    return self._terminal_failure(state, failure)
                if image_rejected:
                    if "base_request" in state:
                        _, body = self._prepare_responses_body(
                            state,
                            state["base_request"],
                            state["base_body"],
                            force_advance=True,
                            retry_index=retry_index,
                        )
                    else:
                        body = wire.trim_oldest_images(body, max(8, 16 - retry_index * 4))
                if recovering:
                    rounds += 1
                    self._record(
                        "model_transport_recovery_scheduled",
                        **state["identity"],
                        recovery_round=rounds,
                        attempt_index=retry_index + 1,
                        delay_s=delay,
                        http_status=failure.status,
                        episode_cooldown_s=self.transport_recovery_summary["cooldown_s"],
                        error=state.get("failure_evidence"),
                    )
                else:
                    delay = self.policy._retry_delay_seconds(retry_index, failure)
                    self._record(
                        "model_request_retry",
                        **state["identity"],
                        attempt_index=retry_index + 1,
                        delay_s=delay,
                        http_status=failure.status,
                        error=state.get("failure_evidence"),
                    )
                if state["cancel"].wait(delay):
                    raise self.policy.InfrastructureFailure("model call cancelled") from None
                if recovering and (reason := self._recovery_gate_reason()) is not None:
                    self._deny_recovery(state, reason, rounds)
                    return self._terminal_failure(state, failure)
                retry_index += 1

    def _terminal_failure(self, state, failure):
        if "failed_response" in state:
            self._block_after_failure(state, "ModelResponseFailed")
            state["delivering_failed_response"] = True
            return state["failed_response"]
        raise failure

    def start(self):
        if self.server is not None or self.closed:
            raise ValueError("model_proxy_already_started_or_closed")
        self.callback_thread = threading.Thread(target=self._callbacks, daemon=True)
        self.callback_thread.start()
        owner = self
        wire = self.wire

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *_):
                pass

            def send_reply(self, reply):
                if reply.body is None:
                    self.send_error(reply.status, reply.reason)
                    return
                self.send_response(reply.status, reply.reason)
                for key, value in reply.headers.items():
                    self.send_header(key, value)
                self.send_header("Content-Length", str(len(reply.body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(reply.body)
                self.wfile.flush()

            def content_length(self):
                try:
                    length = int(self.headers.get("Content-Length", "-1"))
                except ValueError:
                    self.send_error(400)
                    return None
                if not 0 <= length <= MAX_CLIENT_BODY_BYTES or self.headers.get("Transfer-Encoding"):
                    self.send_error(400, "Content-Length required within model input bound")
                    return None
                return length

            def drain_body(self, length):
                remaining = length
                try:
                    self.connection.settimeout(REFUSAL_DRAIN_TIMEOUT_S)
                    while remaining > 0 and (chunk := self.rfile.read(min(remaining, 1 << 16))):
                        remaining -= len(chunk)
                    self.connection.settimeout(None)
                except OSError:
                    pass

            def refuse(self, reply, length):
                if length:
                    self.drain_body(length)
                self.send_reply(reply)

            def declared_length(self):
                try:
                    length = int(self.headers.get("Content-Length", "0"))
                except ValueError:
                    return 0
                return length if 0 < length <= MAX_CLIENT_BODY_BYTES and not self.headers.get("Transfer-Encoding") else 0

            def wire_thread(self, wire=wire):
                key = self.headers.get("X-Ni-Rua-Thread") or wire.thread_key(self.headers)
                return key if isinstance(key, str) and len(key) <= 256 else ""

            def attribution(self, wire=wire):
                native_thread_id = self.wire_thread(wire)
                if (
                    owner.default_thread_id is not None
                    and native_thread_id not in owner.thread_owners
                    and native_thread_id not in owner.cancelled_threads
                ):
                    native_thread_id = owner.default_thread_id
                return native_thread_id, owner.thread_owners.get(native_thread_id)

            def do_POST(self):
                binding, path = owner._binding_for_path(self.path)
                if binding is None:
                    self.refuse(wire.not_found(), self.declared_length())
                    return
                route = binding.wire.route(self.command, path)
                if route is not None and route.kind == "generation":
                    self.generation(route, binding)
                elif route is not None and route.kind == "auxiliary":
                    self.auxiliary(route, binding)
                else:
                    self.refuse(binding.wire.not_found(), self.declared_length())

            def binding_projection(self, binding):
                if binding.name == ACTOR_UPSTREAM:
                    return owner.request_projection
                return lambda request: binding.wire.project_request(
                    request,
                    binding.presentation,
                    registered_tools=frozenset(),
                    tool_name_prefix="",
                )

            def auxiliary(self, route, binding):
                wire = binding.wire
                projection = self.binding_projection(binding)
                length = self.content_length()
                if length is None:
                    return
                with owner.lock:
                    native_thread_id, attribution = self.attribution(wire)
                    attribution = attribution or dict(
                        source=self.headers.get("X-Ni-Rua-Source", "actor")[:128],
                        caller_request_id=self.headers.get("X-Ni-Rua-Request-Id", "")[:128],
                    )
                    owner.aux_request_count += 1
                    record = dict(
                        aux_request_index=owner.aux_request_count,
                        endpoint=route.endpoint,
                        native_thread_id=native_thread_id,
                        **attribution,
                    )
                    if binding.name != ACTOR_UPSTREAM:
                        record["upstream"] = binding.name
                    denial = (
                        "thread_cancelled"
                        if native_thread_id in owner.cancelled_threads
                        else "transport_blocked" if owner.blocked_error is not None or owner.closed else None
                    )
                started, phase = time.monotonic(), "reading_client_body"
                try:
                    if denial is not None:
                        record["denied"] = denial
                        self.refuse(wire.denial(denial), length)
                        return
                    if projection is None:
                        record["denied"] = "projection_unavailable"
                        self.refuse(wire.relay_error(streaming=False, evidence=None, manual_retry_allowed=False), length)
                        return
                    self.connection.settimeout(None)
                    received = self.rfile.read(length)
                    if len(received) != length:
                        raise ConnectionError("incomplete client request")
                    record["body_bytes"] = len(received)
                    phase = "projecting"
                    body = wire.decode_body(self.headers, received)
                    decoded = body is not received
                    request = load_request_body(body)
                    if not isinstance(request, dict):
                        raise ValueError("request_body_must_be_json_object")
                    request = projection(request)
                    body = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
                    body, moved = owner._place_tool_result_images(binding, body)
                    if moved is not None:
                        record["tool_result_images_moved"] = moved
                    record["requested_model"] = wire.requested_model(request) if isinstance(request, dict) else None
                    headers = owner._upstream_headers(self.headers, decoded=decoded, binding=binding)
                    phase = "forwarding"
                    record.update(
                        forwarded_body_bytes=len(body),
                        forwarded_body_sha256=hashlib.sha256(body).hexdigest(),
                    )
                    status, reason, response_headers, data = owner._forward_auxiliary(
                        route,
                        body,
                        headers,
                        binding.upstream_url,
                    )
                    record.update(http_status=status, response_bytes=len(data))
                    phase = "delivering"
                    if status >= 400:
                        try:
                            detail = json.loads(data)
                        except ValueError:
                            detail = None
                        record["response_error"] = wire.operator_response_error(
                            detail if isinstance(detail, dict) else {}
                        )
                        self.send_reply(wire.relay_error(streaming=False, evidence=None, manual_retry_allowed=False))
                        record["delivered"] = True
                        return
                    self.send_response(status, reason)
                    for key, value in response_headers:
                        if key.lower() not in HOP_HEADERS:
                            self.send_header(key, value)
                    self.send_header("Content-Length", str(len(data)))
                    self.send_header("Connection", "close")
                    self.end_headers()
                    self.wfile.write(data)
                    self.wfile.flush()
                    record["delivered"] = True
                except (Exception, owner.policy.InfrastructureFailure) as error:
                    record.update(error_type=type(error).__name__, phase=phase)
                    if phase != "delivering":
                        try:
                            self.send_reply(wire.relay_error(streaming=False, evidence=None, manual_retry_allowed=False))
                        except OSError:
                            pass
                finally:
                    self.close_connection = True
                    owner._record("model_aux_request", **record, elapsed_s=time.monotonic() - started)

            def generation(self, route, binding):
                wire = binding.wire
                actor = binding.name == ACTOR_UPSTREAM
                length = self.content_length()
                if length is None:
                    return
                refusal, denied, exhausted, notify_limit = None, False, False, False
                with owner.lock:
                    native_thread_id, registered = self.attribution(wire)
                    chain_thread = self.wire_thread(wire) or owner.default_thread_id or ""
                    if native_thread_id in owner.cancelled_threads:
                        owner._record(
                            "model_thread_request_rejected",
                            native_thread_id=native_thread_id,
                            reason=owner.cancelled_threads[native_thread_id],
                        )
                        refusal = "thread_cancelled"
                    elif owner.blocked_error is not None:
                        refusal = "transport_blocked"
                    else:
                        exhausted = owner.request_count >= owner.max_requests
                        denied = owner.closed or owner.request_count + owner.reserved_calls >= owner.max_requests
                        if not denied:
                            owner.reserved_calls += 1
                            owner.logical_call_count += 1
                            number = owner.logical_call_count
                            attribution = registered or dict(
                                source=self.headers.get("X-Ni-Rua-Source", "actor")[:128],
                                caller_request_id=self.headers.get("X-Ni-Rua-Request-Id", "")[:128],
                            )
                            identity = dict(
                                logical_call_index=number,
                                logical_call_id=f"{owner.transport_id}:{number}",
                                endpoint=route.endpoint,
                                native_thread_id=native_thread_id,
                                **attribution,
                            )
                            if not actor:
                                identity["upstream"] = binding.name
                            state = dict(
                                identity=identity,
                                client=self.connection,
                                process=None,
                                cancel=threading.Event(),
                                finished=threading.Event(),
                                cancel_reason=None,
                                reading_client_body=True,
                                route=route,
                            )
                            if not actor:
                                state["binding"] = binding
                            if not owner.active:
                                owner._wait_started_at = time.monotonic()
                            owner.active[number] = state
                        notify_limit = exhausted and not owner.limit_notified
                        if notify_limit:
                            owner.limit_notified = True
                if refusal is not None:
                    self.refuse(wire.denial(refusal), length)
                    return
                if denied:
                    event = owner._record(
                        "model_budget_exhausted" if exhausted else "model_budget_reserved",
                        completed_calls=owner.request_count,
                        reserved_calls=owner.reserved_calls,
                        max_requests=owner.max_requests,
                    )
                    self.refuse(wire.denial("budget"), length)
                    if notify_limit and owner.on_limit:
                        owner._queue_callback(owner.on_limit, event)
                    return
                started, streaming, delivered, charged = time.monotonic(), False, False, False
                continuity = None
                call_purpose = None
                idle = None
                stream_open = False
                io_lock = threading.Lock()
                heartbeat_stop = threading.Event()
                keepalive = wire.keepalive_frame()

                def heartbeat():
                    while not heartbeat_stop.wait(wire.keepalive_interval_s):
                        try:
                            with io_lock:
                                self.wfile.write(keepalive)
                                self.wfile.flush()
                        except OSError:
                            owner._cancel(state, "client_disconnected")
                            return

                try:
                    owner._record("model_call_started", **identity, phase="reading_client_body")
                    self.connection.settimeout(None)
                    try:
                        received = self.rfile.read(length)
                        client_read_error = None
                    except OSError as read_error:
                        received = b""
                        client_read_error = read_error
                    with owner.lock:
                        state["reading_client_body"] = False
                    if client_read_error is not None or len(received) != length:
                        state["cancel"].set()
                        state["cancel_reason"] = "client_disconnected_reading_body"
                        owner._record(
                            "model_call_failed",
                            **identity,
                            error_type=(
                                type(client_read_error).__name__
                                if client_read_error is not None
                                else "ConnectionError"
                            ),
                            phase="reading_client_body",
                            cancelled=True,
                            completed_call_charged=False,
                            manual_retry_required=False,
                            error=None,
                        )
                        return
                    owner._record(
                        "model_request_body_received",
                        **identity,
                        body_bytes=len(received),
                        elapsed_s=time.monotonic() - started,
                        body_sha256=hashlib.sha256(received).hexdigest(),
                    )
                    try:
                        body = wire.decode_body(self.headers, received)
                        decoded = body is not received
                        if wire.client_loop_guard and (runaway := owner._runaway_admission(native_thread_id, body)):
                            owner._block_runaway(state, runaway)
                            self.send_reply(wire.denial("client_retry_runaway"))
                            return
                        request = load_request_body(body)
                        if not isinstance(request, dict):
                            raise ValueError("request_body_must_be_json_object")
                        projected = False
                        if actor and owner.request_context is not None:
                            request = owner.request_context(request, dict(identity))
                            if not isinstance(request, dict):
                                raise ValueError("request_body_must_be_json_object")
                            projected = True
                        projection = self.binding_projection(binding)
                        if projection is not None:
                            request = projection(request)
                            if not isinstance(request, dict):
                                raise ValueError("request_body_must_be_json_object")
                            projected = True
                    except ValueError as rejected_error:
                        raw_code = str(rejected_error).strip().split(":", 1)[0]
                        reason_code = (
                            raw_code if re.fullmatch(r"[a-z][a-z0-9_]{1,95}", raw_code) else "invalid_request_body"
                        )
                        owner._record(
                            "model_request_rejected",
                            **identity,
                            error_type=type(rejected_error).__name__,
                            reason=reason_code,
                        )
                        owner._record(
                            "model_call_failed",
                            **identity,
                            error_type=type(rejected_error).__name__,
                            cancelled=False,
                            completed_call_charged=False,
                            manual_retry_required=False,
                            error=None,
                        )
                        try:
                            self.send_reply(wire.relay_error(streaming=False, evidence=None, manual_retry_allowed=False))
                        except OSError:
                            pass
                        return
                    subagent_request = False
                    if actor and owner.continuity is not None and isinstance(request, dict):
                        subagent_request = wire.is_subagent_request(self.headers)
                        continuity = wire.continuity_inputs(request)
                        flags = wire.history_model_output_flags(request) if wire.client_loop_guard else None
                        admitted = owner._observe_continuity(
                            state,
                            chain_thread,
                            continuity,
                            flags,
                            subagent_request=subagent_request,
                        )
                        call_purpose, idle = admitted.call_purpose, admitted.idle
                        if admitted.refused is not None:
                            owner._block_unconsumed(state, admitted.refused)
                            self.send_reply(wire.denial("client_retry_runaway"))
                            return
                    elif actor and owner.classify_call_purpose and isinstance(request, dict):
                        subagent_request = wire.is_subagent_request(self.headers)
                        call_purpose = owner._classify_call_purpose(
                            state,
                            chain_thread,
                            request,
                            subagent_request=subagent_request,
                        )
                    tier = owner.service_tier if actor else None
                    if tier is not None and wire.supports_service_tier and not route.compaction and isinstance(request, dict):
                        request = dict(request, service_tier=tier)
                    if projected or tier is not None:
                        body = json.dumps(request, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
                    if actor and not route.compaction:
                        state["base_request"] = request
                        state["base_body"] = body
                        request, body = owner._prepare_responses_body(state, request, body)
                    identity["requested_model"] = wire.requested_model(request)
                    streaming = wire.is_streaming(request)
                    self.connection.settimeout(None)
                    if streaming and wire.early_stream:
                        self.send_response(200)
                        self.send_header("Content-Type", "text/event-stream")
                        self.send_header("Connection", "close")
                        self.end_headers()
                        stream_open = True
                        if keepalive is not None:
                            threading.Thread(target=heartbeat, daemon=True).start()
                    threading.Thread(target=owner._watch_client, args=(state,), daemon=True).start()
                    headers = owner._upstream_headers(self.headers, decoded=decoded, binding=binding)
                    raw, response_headers = owner._complete(state, body, headers, route.upstream_path)
                    with owner.lock:
                        if state["cancel"].is_set() or owner.closed:
                            raise owner.policy.InfrastructureFailure("model call cancelled before delivery")
                    if call_purpose == MAIN_CALL and state.get("context_edit_applied"):
                        owner._context_discontinuity(state, PROVIDER_CONTEXT_EDIT)
                    if actor and owner.on_tool_calls is not None and not state.get("delivering_failed_response"):
                        try:
                            _invoke_on_tool_calls(owner.on_tool_calls, chain_thread, wire.emitted_tool_calls(raw), call_purpose)
                        except Exception as error:  # noqa: BLE001 - ordering information only; deliver as usual
                            owner._record(
                                "model_tool_call_order_unavailable",
                                **identity,
                                error_type=type(error).__name__,
                            )
                    heartbeat_stop.set()
                    with io_lock:
                        state["delivering"] = True
                        if not stream_open:
                            self.send_response(200)
                            for key, value in response_headers.items():
                                if key.lower() not in HOP_HEADERS:
                                    self.send_header(key, value)
                            self.send_header("Content-Length", str(len(raw)))
                            self.send_header("Connection", "close")
                            self.end_headers()
                        self.wfile.write(raw)
                        self.wfile.flush()
                        delivered = True
                        state["delivered"] = True
                    if state.get("delivering_failed_response"):
                        owner._block_after_failure(state, "ModelResponseFailed")
                        owner._record(
                            "model_call_failed",
                            **identity,
                            error_type="ModelResponseFailed",
                            cancelled=False,
                            completed_call_charged=False,
                            outgoing_bytes=len(raw),
                            manual_retry_required=True,
                            error=state.get("failure_evidence"),
                        )
                    else:
                        with owner.lock:
                            owner.request_count += 1
                            owner.reserved_calls -= 1
                            charged = True
                            if continuity is not None:
                                main_thread = owner._continuity_main_thread(chain_thread) and not subagent_request
                                owner.continuity.commit(
                                    chain_thread,
                                    main_thread=main_thread,
                                    call_purpose=call_purpose,
                                    **continuity,
                                )
                                if wire.client_loop_guard:
                                    owner._commit_idle_run(chain_thread, call_purpose, continuity["fingerprints"], idle)
                        owner._record(
                            "model_call_completed",
                            **identity,
                            completed_calls=owner.request_count,
                            outgoing_bytes=len(raw),
                        )
                except (Exception, owner.policy.InfrastructureFailure) as error:
                    heartbeat_stop.set()
                    capability_refusal = bool(state.get("capability_failure"))
                    if not capability_refusal:
                        owner._block_after_failure(state, type(error).__name__)
                    owner._record(
                        "model_call_failed",
                        **identity,
                        error_type=type(error).__name__,
                        cancelled=state["cancel"].is_set(),
                        completed_call_charged=charged,
                        manual_retry_required=(
                            not capability_refusal
                            and (not state["cancel"].is_set() or state["cancel_reason"] == "client_disconnected")
                        ),
                        error=state.get("failure_evidence"),
                    )
                    try:
                        reply = wire.relay_error(
                            streaming=stream_open,
                            evidence=state.get("failure_evidence"),
                            manual_retry_allowed=bool(state.get("provider_explicit_error")),
                        )
                        with io_lock:
                            if stream_open:
                                self.wfile.write(reply.body)
                                self.wfile.flush()
                            else:
                                self.send_reply(reply)
                    except OSError:
                        pass
                finally:
                    heartbeat_stop.set()
                    state["finished"].set()
                    self.close_connection = True
                    with owner.lock:
                        if not charged:
                            owner.reserved_calls -= 1
                        owner.active.pop(number, None)
                        if not owner.active and owner._wait_started_at is not None:
                            owner._wait_total_s += time.monotonic() - owner._wait_started_at
                            owner._wait_started_at = None
                    owner._record(
                        "model_call_finished",
                        **identity,
                        completed_calls=owner.request_count,
                        attempt_count=owner.attempt_count,
                        delivered=delivered,
                        elapsed_s=time.monotonic() - started,
                        cancelled=state["cancel"].is_set(),
                        failed_response_delivered=bool(delivered and state.get("delivering_failed_response")),
                    )

        if "GET" in wire.methods:
            Handler.do_GET = Handler.do_POST

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        self.server.block_on_close = False
        self.base_url = f"http://127.0.0.1:{self.server.server_port}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.05}, daemon=True)
        self.thread.start()
        return self

    def close(self):
        with self.lock:
            self.closed = True
            auxiliary = list(self._aux_connections)
        for connection in auxiliary:
            _InProcessAttempt._force_close(connection)
        self.cancel_pending("proxy_closed")
        with self.lock:
            active = list(self.active.values())
        if self.wire.client_loop_guard:
            deadline = time.monotonic() + self.SHUTDOWN_TERMINAL_GRACE_S
            for state in active:
                state["finished"].wait(max(0.0, deadline - time.monotonic()))
        for state in active:
            self._shutdown(state["client"])
        if self.server:
            self.server.shutdown()
            self.server.server_close()
        if self.thread:
            self.thread.join(1)
        for state in active:
            state["finished"].wait(1)
        self._close_conn_pool()
        self.callbacks.put(None)
        if self.callback_thread and self.callback_thread is not threading.current_thread():
            self.callback_thread.join(1)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()


if __name__ == "__main__":
    _http_worker()
