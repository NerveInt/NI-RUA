"""Thread-safe HTTP/1.1 Keep-Alive connection pool and per-endpoint image-cache tracker for external perception.

Shared by ``object_table.py`` and ``scene3d.py`` when calling the local/tunnelled VGGT service over plain ``http://``
with the unpatched stdlib ``urllib.request.urlopen``. Preserves:
- Exclusive socket checkout per in-flight request (safe for concurrent ``/object_table`` + ``/points``).
- Automatic ``TCP_NODELAY``, proxy-bypass check, and per-request socket timeout updates.
- Non-blocking idle socket liveness check (``setblocking(False)`` + ``MSG_PEEK``) plus time-bounded single-retry
  recovery when a pooled keep-alive socket hits EOF/reset during ``request()`` or within 1.0 s of send (never retrying
  fresh sockets, late mid-flight crashes, timeouts, or HTTP status responses).
- Strict respect for ``response.will_close`` (HTTP/1.0 test fixtures and ``Connection: close`` error responses are
  closed immediately and never pooled).
- Bounded per-endpoint tracking of recently uploaded ``(image_sha256, K, T, support_z)`` keys and server
  ``points_image_sha256`` capability so ``/points`` omits duplicate ``image_b64`` uploads when ``/object_table`` has
  already sent the frame, while seamlessly falling back on cache misses or legacy servers.
"""
from __future__ import annotations

import atexit
import collections
import contextlib
import http.client
import json
import socket
import threading
import time
import urllib.error
import urllib.parse
import urllib.request

import numpy as np

_MAX_CONNS_PER_HOST = 8
_MAX_POOL_HOSTS = 16
_MAX_UPLOADED_KEYS_PER_ENDPOINT = 32
_STALE_SOCKET_RETRY_WINDOW_S = 1.0
_POINTS_SHA_CAP = "points_image_sha256"

_POOL_LOCK = threading.Lock()
_IDLE_CONNS: collections.OrderedDict[tuple[str, int], list[http.client.HTTPConnection]] = collections.OrderedDict()

_STATE_LOCK = threading.Lock()


class _EndpointCacheState:
    __slots__ = ("sha_mode", "uploaded", "inflight")

    def __init__(self) -> None:
        self.sha_mode: str = "unknown"  # "unknown" | "supported" | "unsupported"
        self.uploaded: collections.OrderedDict[tuple[str, bytes, bytes, bytes], None] = collections.OrderedDict()
        self.inflight: dict[tuple[str, bytes, bytes, bytes], int] = {}


_ENDPOINT_STATES: collections.OrderedDict[str, _EndpointCacheState] = collections.OrderedDict()


def _safe_close(conn: http.client.HTTPConnection | None) -> None:
    if conn is None:
        return
    with contextlib.suppress(Exception):
        conn.close()


def _is_socket_alive(sock: socket.socket | None) -> bool:
    """Return True iff ``sock`` is open with zero unread bytes and no pending EOF/error."""
    if sock is None:
        return False
    if hasattr(socket, "MSG_PEEK"):
        prev_timeout = None
        has_prev = False
        try:
            prev_timeout = sock.gettimeout()
            has_prev = True
            sock.setblocking(False)
            _ = sock.recv(1, socket.MSG_PEEK)
            # Empty bytes => peer sent FIN (EOF); non-empty bytes => unexpected unread data on idle socket.
            return False
        except BlockingIOError:
            return True
        except OSError:
            return False
        finally:
            if has_prev:
                with contextlib.suppress(OSError):
                    sock.settimeout(prev_timeout)
    return True


def _checkout_conn(key: tuple[str, int]) -> http.client.HTTPConnection | None:
    stale: list[http.client.HTTPConnection] = []
    chosen: http.client.HTTPConnection | None = None
    with _POOL_LOCK:
        bucket = _IDLE_CONNS.get(key)
        while bucket:
            cand = bucket.pop()  # LIFO: prefer most recently active warm socket
            if _is_socket_alive(getattr(cand, "sock", None)):
                chosen = cand
                break
            stale.append(cand)
        if bucket is not None and not bucket:
            _IDLE_CONNS.pop(key, None)
    for dead in stale:
        _safe_close(dead)
    return chosen


def _checkin_conn(key: tuple[str, int], conn: http.client.HTTPConnection) -> None:
    if getattr(conn, "sock", None) is None:
        _safe_close(conn)
        return
    evicted: list[http.client.HTTPConnection] = []
    with _POOL_LOCK:
        bucket = _IDLE_CONNS.get(key)
        if bucket is not None:
            _IDLE_CONNS.move_to_end(key)
        else:
            while len(_IDLE_CONNS) >= _MAX_POOL_HOSTS:
                _, old_bucket = _IDLE_CONNS.popitem(last=False)
                evicted.extend(old_bucket)
            bucket = []
            _IDLE_CONNS[key] = bucket
        if len(bucket) >= _MAX_CONNS_PER_HOST:
            evicted.append(bucket.pop(0))
        bucket.append(conn)
    for old_conn in evicted:
        _safe_close(old_conn)


def close_pool() -> None:
    """Close all idle pooled HTTP connections and reset per-endpoint cache state."""
    to_close: list[http.client.HTTPConnection] = []
    with _POOL_LOCK:
        for bucket in _IDLE_CONNS.values():
            to_close.extend(bucket)
        _IDLE_CONNS.clear()
    for conn in to_close:
        _safe_close(conn)
    with _STATE_LOCK:
        _ENDPOINT_STATES.clear()


atexit.register(close_pool)


def parse_http_endpoint(endpoint: str, path: str) -> tuple[str, int, str] | None:
    """Parse ``endpoint + path`` when scheme is direct (non-proxied) ``http://`` with valid host/port; else ``None``."""
    full_url = endpoint.rstrip("/") + path
    parsed = urllib.parse.urlsplit(full_url)
    if parsed.scheme.lower() != "http" or not parsed.hostname:
        return None
    try:
        port = parsed.port or 80
    except ValueError:
        return None
    if urllib.request.getproxies().get("http") and not urllib.request.proxy_bypass(parsed.hostname):
        return None
    target = parsed.path or "/"
    if parsed.query:
        target = f"{target}?{parsed.query}"
    return parsed.hostname, port, target


def post_json_keepalive(
    host: str,
    port: int,
    target: str,
    payload_bytes: bytes,
    timeout_s: float,
    *,
    on_sent=None,
    service_error_cls=RuntimeError,
) -> tuple[int, dict]:
    """Send a JSON POST over a pooled HTTP/1.1 connection and return ``(http_status, parsed_json_dict_or_val)``.

    Fires ``on_sent`` at most once (after request bytes are written to the socket and before ``getresponse()``, or on
    exit if connection/send fails). Retries at most once using remaining timeout when a pooled socket closes with
    EOF/reset during ``request()`` or within ``_STALE_SOCKET_RETRY_WINDOW_S`` of ``request()``.
    """
    eff_timeout = max(0.01, float(timeout_s))
    t_call_start = time.monotonic()
    headers = {
        "Content-Type": "application/json",
        "Content-Length": str(len(payload_bytes)),
        "Connection": "keep-alive",
    }
    sent_fired = False

    def _fire_sent() -> None:
        nonlocal sent_fired
        if not sent_fired and callable(on_sent):
            sent_fired = True
            on_sent()

    key = (host, int(port))
    conn = _checkout_conn(key)
    reused = conn is not None

    try:
        for _attempt in range(2):
            cur_timeout = max(0.01, eff_timeout - (time.monotonic() - t_call_start))
            if conn is None:
                reused = False
                conn = http.client.HTTPConnection(host, int(port), timeout=cur_timeout)
                try:
                    conn.connect()
                except OSError as err:
                    _safe_close(conn)
                    conn = None
                    _fire_sent()
                    url_err = urllib.error.URLError(err)
                    raise service_error_cls(f"URLError:{str(url_err.reason)[:200]}") from None
                if conn.sock is not None:
                    with contextlib.suppress(OSError):
                        conn.sock.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            else:
                conn.timeout = cur_timeout
                if conn.sock is not None:
                    try:
                        conn.sock.settimeout(cur_timeout)
                    except OSError:
                        _safe_close(conn)
                        conn = None
                        continue

            t_sent = time.monotonic()
            try:
                conn.request("POST", target, body=payload_bytes, headers=headers)
            except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError) as err:
                _safe_close(conn)
                conn = None
                if reused:
                    reused = False
                    continue
                _fire_sent()
                url_err = urllib.error.URLError(err)
                raise service_error_cls(f"URLError:{str(url_err.reason)[:200]}") from None
            except TimeoutError as err:
                _safe_close(conn)
                conn = None
                _fire_sent()
                raise service_error_cls(f"TimeoutError:{str(err) or 'timed out'}"[:200]) from None
            except OSError as err:
                _safe_close(conn)
                conn = None
                _fire_sent()
                url_err = urllib.error.URLError(err)
                raise service_error_cls(f"URLError:{str(url_err.reason)[:200]}") from None
            except http.client.HTTPException as err:
                _safe_close(conn)
                conn = None
                _fire_sent()
                raise service_error_cls(f"{type(err).__name__}:{str(err)[:200]}") from None

            _fire_sent()

            try:
                resp = conn.getresponse()
            except TimeoutError as err:
                _safe_close(conn)
                conn = None
                raise service_error_cls(f"TimeoutError:{str(err) or 'timed out'}"[:200]) from None
            except (
                http.client.RemoteDisconnected,
                BrokenPipeError,
                ConnectionResetError,
                ConnectionAbortedError,
            ) as err:
                elapsed_since_send = time.monotonic() - t_sent
                _safe_close(conn)
                conn = None
                if reused and elapsed_since_send <= _STALE_SOCKET_RETRY_WINDOW_S:
                    reused = False
                    continue
                raise service_error_cls(f"{type(err).__name__}:{str(err)[:200]}") from None
            except (OSError, http.client.HTTPException) as err:
                _safe_close(conn)
                conn = None
                raise service_error_cls(f"{type(err).__name__}:{str(err)[:200]}") from None

            try:
                status = int(resp.status)
                raw_body = resp.read()
                will_close = bool(resp.will_close)
            except TimeoutError as err:
                _safe_close(conn)
                conn = None
                raise service_error_cls(f"TimeoutError:{str(err) or 'timed out'}"[:200]) from None
            except (OSError, http.client.HTTPException) as err:
                _safe_close(conn)
                conn = None
                raise service_error_cls(f"{type(err).__name__}:{str(err)[:200]}") from None

            if not will_close and getattr(conn, "sock", None) is not None:
                _checkin_conn(key, conn)
            else:
                _safe_close(conn)
            conn = None

            if status >= 400:
                try:
                    return status, json.loads(raw_body)
                except ValueError:
                    raise service_error_cls(f"http_{status}") from None
            try:
                return status, json.loads(raw_body)
            except ValueError as err:
                raise service_error_cls(f"{type(err).__name__}:{str(err)[:200]}") from None
    finally:
        if conn is not None:
            _safe_close(conn)
        _fire_sent()

    raise service_error_cls("http_keepalive_retry_exhausted")


def make_image_cache_key(image_sha256: str, K, T, support_z: float) -> tuple[str, bytes, bytes, bytes]:
    return (
        str(image_sha256).lower(),
        np.asarray(K, dtype=np.float64).reshape(3, 3).tobytes(),
        np.asarray(T, dtype=np.float64).reshape(4, 4).tobytes(),
        np.float64(support_z).tobytes(),
    )


def _get_endpoint_state(endpoint: str) -> _EndpointCacheState:
    ep = str(endpoint or "").rstrip("/")
    st = _ENDPOINT_STATES.get(ep)
    if st is not None:
        _ENDPOINT_STATES.move_to_end(ep)
        return st
    while len(_ENDPOINT_STATES) >= _MAX_POOL_HOSTS:
        _ENDPOINT_STATES.popitem(last=False)
    st = _EndpointCacheState()
    _ENDPOINT_STATES[ep] = st
    return st


def _apply_server_caps(st: _EndpointCacheState, answer: dict | None) -> None:
    if not isinstance(answer, dict):
        return
    caps = answer.get("caps")
    if isinstance(caps, list) and _POINTS_SHA_CAP in caps:
        st.sha_mode = "supported"
    elif str(answer.get("version") or "").startswith("object-table-v"):
        st.sha_mode = "unsupported"


def _record_uploaded(st: _EndpointCacheState, cache_key: tuple[str, bytes, bytes, bytes]) -> None:
    st.uploaded[cache_key] = None
    st.uploaded.move_to_end(cache_key)
    while len(st.uploaded) > _MAX_UPLOADED_KEYS_PER_ENDPOINT:
        st.uploaded.popitem(last=False)


def note_image_upload_started(endpoint: str, cache_key: tuple[str, bytes, bytes, bytes]) -> None:
    with _STATE_LOCK:
        st = _get_endpoint_state(endpoint)
        st.inflight[cache_key] = st.inflight.get(cache_key, 0) + 1


def note_image_upload_finished(
    endpoint: str,
    cache_key: tuple[str, bytes, bytes, bytes],
    *,
    ok: bool,
    answer: dict | None = None,
) -> None:
    with _STATE_LOCK:
        st = _get_endpoint_state(endpoint)
        rem = st.inflight.get(cache_key, 0) - 1
        if rem > 0:
            st.inflight[cache_key] = rem
        else:
            st.inflight.pop(cache_key, None)
        _apply_server_caps(st, answer)
        if ok:
            _record_uploaded(st, cache_key)


def note_image_uploaded(
    endpoint: str,
    cache_key: tuple[str, bytes, bytes, bytes],
    *,
    answer: dict | None = None,
) -> None:
    """Record a completed upload (e.g. from ``scene3d_query``) without touching concurrent ``inflight`` counts."""
    with _STATE_LOCK:
        st = _get_endpoint_state(endpoint)
        _apply_server_caps(st, answer)
        _record_uploaded(st, cache_key)


def note_server_response(endpoint: str, answer: dict | None) -> None:
    if not isinstance(answer, dict):
        return
    caps = answer.get("caps")
    if isinstance(caps, list) and _POINTS_SHA_CAP in caps:
        with _STATE_LOCK:
            st = _get_endpoint_state(endpoint)
            st.sha_mode = "supported"


def get_endpoint_sha_mode(endpoint: str) -> str:
    with _STATE_LOCK:
        st = _ENDPOINT_STATES.get(str(endpoint or "").rstrip("/"))
        return st.sha_mode if st is not None else "unknown"


def should_try_points_sha256(endpoint: str, cache_key: tuple[str, bytes, bytes, bytes]) -> bool:
    """Return True when ``endpoint`` has not rejected ``image_sha256``-only ``/points`` and ``cache_key`` was
    already uploaded or is currently in flight to ``endpoint``."""
    with _STATE_LOCK:
        st = _ENDPOINT_STATES.get(str(endpoint or "").rstrip("/"))
        if st is None or st.sha_mode == "unsupported":
            return False
        return cache_key in st.uploaded or cache_key in st.inflight


def evict_cached_image_key(
    endpoint: str,
    cache_key: tuple[str, bytes, bytes, bytes],
    *,
    legacy_unsupported: bool = False,
) -> None:
    with _STATE_LOCK:
        st = _get_endpoint_state(endpoint)
        st.uploaded.pop(cache_key, None)
        if legacy_unsupported:
            st.sha_mode = "unsupported"
