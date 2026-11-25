"""Relay egress projection, upstream header/credential binding, continuity observation, and image windowing."""

from __future__ import annotations

import hashlib
import json
import math
import re
from typing import NamedTuple

from ..session.wire.openai_responses import OpenAIResponsesWire
from .continuity import HELPER as HELPER_CALL, MAIN as MAIN_CALL, SUBAGENT as SUBAGENT_CALL, is_helper_request
from .guard import ContinuityAdmission

PROVIDER_CONTEXT_EDIT = "provider_context_edit"

HOP_HEADERS = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}
PRIVATE_HEADERS = {
    "x-ni-rua-deadline-unix-ms",
    "x-ni-rua-source",
    "x-ni-rua-request-id",
    "x-ni-rua-thread",
}
AUTH_HEADERS = dict(OpenAIResponsesWire.auth_header_names)
CLIENT_AUTH_HEADERS = set(AUTH_HEADERS) | {
    "cookie",
    "x-api-key",
    "api-key",
    "openai-organization",
    "openai-project",
}
MIN_TRIM_IMAGE_LIMIT = 24
ACTOR_UPSTREAM = "actor"
UPSTREAM_PREFIX = "/_ni_rua/u/"
UPSTREAM_NAME = re.compile(r"[a-z][a-z0-9_-]{0,31}")
TOOL_RESULT_IMAGE_PLACEMENTS = ("inline", "user_blocks")

_RESPONSES = OpenAIResponsesWire()


class UpstreamBinding(NamedTuple):
    """One upstream the relay fronts (``ModelRequestProxy.add_upstream``)."""

    name: str
    wire: object
    upstream_url: str
    auth_headers: object
    client_auth_headers: frozenset
    presentation: object = None
    prefix: str = ""


def _reject_json_constant(token):
    raise ValueError("non_finite_json_number")


def _finite_json_float(token):
    value = float(token)
    if not math.isfinite(value):  # a literal that overflows a double, for example 1e999
        raise ValueError("non_finite_json_number")
    return value


_SURROGATE_ESCAPE = re.compile(r"\\u[dD][89a-fA-F]")
_SURROGATE_ESCAPE_BYTES = re.compile(rb"\\u[dD][89a-fA-F]")


#: Deepest nesting of objects and arrays a request body may have (outermost container = 1). The projections and the
#: continuity counters walk a request recursively, some with two interpreter frames per level, so a body a few hundred
#: levels deep failed there with RecursionError after admission. Recorded request bodies nest at most about 25 levels
#: (a tool catalogue with its JSON Schemas); 128 leaves both sides a wide margin.
MAX_REQUEST_NESTING = 128


def _nesting_exceeds(value, limit):
    """True when a parsed JSON value nests objects / arrays deeper than ``limit`` (outermost container = 1)."""
    pending = [(value, 1)]
    while pending:
        item, depth = pending.pop()
        if isinstance(item, dict):
            children = item.values()
        elif isinstance(item, list):
            children = item
        else:
            continue
        if depth > limit:
            return True
        pending.extend((child, depth + 1) for child in children if isinstance(child, (dict, list)))
    return False


def _strings_encode(value):
    """True when every string (keys included) of a parsed JSON value encodes as UTF-8, i.e. has no unpaired surrogate."""
    pending = [value]
    while pending:
        item = pending.pop()
        if isinstance(item, str):
            try:
                item.encode("utf-8")
            except UnicodeEncodeError:
                return False
        elif isinstance(item, dict):
            pending.extend(item)
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return True


def load_request_body(body):
    """Parse a client request body as strict JSON (RFC 8259) that the relay can serialise again.

    ``json.loads`` accepts three kinds of body that the relay cannot forward once it has projected the request
    (it re-serialises with ``allow_nan=False`` and encodes as UTF-8), or cannot even parse without an error that is
    not a ``ValueError``. Each of them used to fail after admission, as a relay failure that blocks the transport;
    refused here it is an invalid request body like any other malformed one (nothing forwarded, relay usable):

    * a non-finite number: the tokens ``NaN``, ``Infinity``, ``-Infinity`` or a literal that overflows a double
      (``1e999``) -> ``ValueError("non_finite_json_number")``;
    * a string with an unpaired UTF-16 surrogate (the escape ``\\ud800`` alone; a client that cuts text between the
      two halves of a character outside the BMP produces it) -> ``ValueError("unpaired_surrogate_in_json_string")``;
    * objects / arrays nested deeper than ``MAX_REQUEST_NESTING`` levels (the relay's own recursive walkers and, far
      beyond that, the parser itself raise ``RecursionError``) -> ``ValueError("json_nesting_too_deep")``.

    Every other body parses exactly as ``json.loads`` parses it. The strings are only walked when the body contains a
    surrogate escape at all (or is not UTF-8 bytes), so an ordinary body costs one regular-expression scan.

    ``body``: the decoded request body (bytes or str). Example: ``load_request_body(b'{"temperature":NaN}')`` raises.
    """
    try:
        request = json.loads(body, parse_constant=_reject_json_constant, parse_float=_finite_json_float)
    except RecursionError:
        raise ValueError("json_nesting_too_deep") from None
    if _nesting_exceeds(request, MAX_REQUEST_NESTING):
        raise ValueError("json_nesting_too_deep")
    if isinstance(body, str):
        suspect = True
    elif json.detect_encoding(body) != "utf-8":
        suspect = True
    else:
        suspect = _SURROGATE_ESCAPE_BYTES.search(body) is not None
    if suspect and not _strings_encode(request):
        raise ValueError("unpaired_surrogate_in_json_string")
    return request


def upstream_auth_headers_copy(headers, names=None):
    """Freeze host credentials without putting their values in validation errors."""
    names = AUTH_HEADERS if names is None else names
    if headers is None:
        return None
    if not isinstance(headers, dict):
        raise ValueError("invalid_upstream_auth_headers")
    copied = {}
    for name, value in headers.items():
        key = names.get(name.lower()) if isinstance(name, str) else None
        if (
            key is None
            or key in copied
            or not isinstance(value, str)
            or not 0 < len(value) <= 16384
            or any(ord(char) < 32 or ord(char) > 126 for char in value)
        ):
            raise ValueError("invalid_upstream_auth_headers")
        copied[key] = value
    return copied


def _request_target(url):
    """Request target for ``http.client``: path plus any wire query string."""
    return url.path + ("?" + url.query if url.query else "")


def _terminal_sse_type(block):
    """Compatibility delegate: ``OpenAIResponsesWire.terminal_sse_type``."""
    return _RESPONSES.terminal_sse_type(block)


def _terminal_sse_block(block):
    """Compatibility delegate: a Responses terminal SSE block (used by ``reference_api``)."""
    return _RESPONSES.terminal_sse_type(block) is not None


class PayloadMixin:
    """Header/credential scoping, continuity tracking, image trim/reuse, and image placement."""

    def _upstream_headers(self, client_headers, *, decoded, binding=None):
        """Build forwarded headers with hop/private headers and client credentials stripped."""
        binding = self.actor_binding if binding is None else binding
        headers = {k: v for k, v in client_headers.items() if k.lower() not in HOP_HEADERS | PRIVATE_HEADERS}
        headers = {k: v for k, v in headers.items() if k.lower() != "accept-encoding"}
        if decoded:
            headers = {k: v for k, v in headers.items() if k.lower() != "content-encoding"}
        if binding.auth_headers is not None:
            headers = {k: v for k, v in headers.items() if k.lower() not in binding.client_auth_headers}
            headers.update(binding.auth_headers)
        elif binding.name != ACTOR_UPSTREAM:
            headers = {k: v for k, v in headers.items() if k.lower() not in binding.client_auth_headers}
        headers["Accept-Encoding"] = "identity"
        return headers

    def _context_discontinuity(self, state, reason):
        """Record a main-chain discontinuity, reset the image-trim window, and notify outside lock."""
        identity = state["identity"]
        with self.lock:
            self.image_context_epoch = identity["logical_call_id"]
            thread_id = identity.get("native_thread_id") or ""
            self._image_trim_watermarks.pop(thread_id, None)
            self._image_trim_starts.pop(thread_id, None)
            self._record("model_context_discontinuity", **identity, reason=reason)
        callback = self.on_context_discontinuity
        if callback is None:
            return
        try:
            callback(reason)
        except Exception as error:
            self._record(
                "model_context_discontinuity_callback_failed",
                **identity,
                reason=reason,
                error_type=type(error).__name__,
            )

    def _continuity_main_thread(self, thread):
        """Return True when ``thread`` tracks the main conversation chain (call with ``self.lock`` held)."""
        actors = [key for key, owner in self.thread_owners.items() if owner.get("source") == "actor"]
        if self.default_thread_id is None and not actors:
            return True
        return thread == self.default_thread_id or thread in actors

    def _observe_continuity(self, state, thread, inputs, output_flags=None, *, subagent_request=False):
        """Admit continuity inputs on ``thread``, compute consume-guard verdict, and return ContinuityAdmission."""
        refused = idle = None
        with self.lock:
            main_thread = self._continuity_main_thread(thread) and not subagent_request
            admission = self.continuity.admit(thread, main_thread=main_thread, **inputs)
            state["identity"]["call_purpose"] = admission.call_purpose
            if output_flags is not None:
                idle = self._idle_request(thread, admission, inputs["fingerprints"], output_flags)
                run = self._idle_runs.get(thread)
                if idle and run.idle >= self.CONSUME_GUARD_MAX_IDLE_COMPLETIONS:
                    refused = run.idle
        if admission.reason is not None:
            self._context_discontinuity(state, admission.reason)
        return ContinuityAdmission(admission.call_purpose, idle, refused)

    def _classify_call_purpose(self, state, thread, request, *, subagent_request=False):
        """在未启用 ContinuityTracker 时（如 Codex），按统一口径为 actor 上游请求归属 call_purpose。"""
        with self.lock:
            main_thread = self._continuity_main_thread(thread) and not subagent_request
        source = state["identity"].get("source")
        if source not in (None, "", "actor") or is_helper_request(request):
            purpose = HELPER_CALL
        elif main_thread:
            purpose = MAIN_CALL
        else:
            purpose = SUBAGENT_CALL
        state["identity"]["call_purpose"] = purpose
        return purpose

    @staticmethod
    def _failed_response_error(response):
        """Compatibility delegate: ``OpenAIResponsesWire.failed_response_error``."""
        return _RESPONSES.failed_response_error(response)

    @staticmethod
    def _response_evidence(raw, endpoint):
        """Compatibility delegate: ``OpenAIResponsesWire.response_evidence`` for ``endpoint``."""
        return _RESPONSES.response_evidence(raw, _RESPONSES.route_for_endpoint(endpoint))

    def _place_tool_result_images(self, binding, body):
        """Move tool-result images into labelled user blocks when ``tool_result_images == 'user_blocks'``."""
        if self.tool_result_images != "user_blocks" or binding.name != ACTOR_UPSTREAM:
            return body, None
        try:
            request = json.loads(body)
        except ValueError:
            return body, 0
        placed, moved = binding.wire.move_tool_result_images(request)
        if placed is request:
            return body, moved
        return json.dumps(placed, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode(), moved

    @staticmethod
    def _part_observation_labels(parts):
        """Delegate: ``OpenAIResponsesWire.part_observation_labels``."""
        return OpenAIResponsesWire.part_observation_labels(parts)

    @classmethod
    def _inline_image_locations(cls, request, *, include_labels=False):
        """Delegate: ``OpenAIResponsesWire.image_locations``."""
        return _RESPONSES.image_locations(request, include_labels=include_labels)

    @classmethod
    def _step_image_boundaries(cls, request):
        """Delegate: ``OpenAIResponsesWire.step_image_boundaries``."""
        return _RESPONSES.step_image_boundaries(request)

    @classmethod
    def _trim_image_range(cls, request, start_idx, end_idx):
        """Delegate: ``WireAdapter.trim_image_range`` on the Responses wire."""
        return _RESPONSES.trim_image_range(request, start_idx, end_idx)

    @classmethod
    def _trim_prefix_images(cls, request, prefix_count):
        """Delegate: ``WireAdapter.trim_prefix_images`` on the Responses wire."""
        return _RESPONSES.trim_prefix_images(request, prefix_count)

    @classmethod
    def _trim_oldest_images(cls, body, keep_latest=24):
        """Delegate: ``WireAdapter.trim_oldest_images`` on the Responses wire."""
        return _RESPONSES.trim_oldest_images(body, keep_latest)

    def _prepare_responses_body(self, state, base_request, base_body, *, force_advance=False, retry_index=0):
        """Image-trim sliding window state machine for one actor generation request."""
        wire = self.wire
        identity = state["identity"]
        thread_key = identity.get("native_thread_id") or ""
        locations, step_boundaries = wire.step_image_boundaries(base_request)
        total_images = len(locations)
        with self.lock:
            watermark, last_total = self._image_trim_watermarks.get(thread_key, (0, 0))
            start_idx = self._image_trim_starts.get(thread_key, 0)
            if total_images < last_total or watermark > total_images or start_idx > watermark:
                start_idx, watermark = 0, 0
            previous = (start_idx, watermark)
            if force_advance and total_images > 0:
                keep = max(8, 16 - retry_index * 4)
                watermark = min(total_images, max(watermark + 4, total_images - keep))

        def render(s_idx, wm):
            working = base_request
            if wm > s_idx:
                working, _ = wire.trim_image_range(working, s_idx, wm)
            image_stats = None
            if self.image_reuse:
                rewritten, image_stats = wire.reuse_exact_images(working)
                if image_stats["replaced"]:
                    working = rewritten
            if wm <= s_idx and not (image_stats and image_stats["replaced"]):
                encoded = base_body
            else:
                encoded = json.dumps(working, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()
            return working, encoded, image_stats

        working, encoded, image_stats = render(start_idx, watermark)
        limit = self.max_inline_images

        def image_count(request):
            return len(wire.image_locations(request))

        bytes_over = len(encoded) > wire.body_limit_bytes
        count_over = bool(limit) and image_count(working) > limit
        reason = "body_bytes_and_image_count" if bytes_over and count_over else "image_count" if count_over else "body_bytes"
        if self.image_trim and (bytes_over or count_over) and total_images > 8:
            target_bytes = wire.trim_target_bytes
            target_images = limit * 2 // 3
            tail_candidates = [b for b in step_boundaries if b <= total_images - 12]
            new_wm = max(watermark, tail_candidates[-1] if tail_candidates else max(0, total_images - 12))
            if watermark == 0 and new_wm > 0:
                need_bytes = len(encoded) - target_bytes if bytes_over else 0
                need_images = image_count(working) - target_images if count_over else 0
                saved, removed, raw_start = 0, 0, new_wm
                for idx in range(new_wm - 1, -1, -1):
                    parts, part_idx = locations[idx]
                    saved += wire.image_part_size(parts[part_idx])
                    removed += 1
                    raw_start = idx
                    if saved >= need_bytes and removed >= need_images:
                        break
                prefix_candidates = [b for b in step_boundaries if 0 < b <= raw_start and b < new_wm]
                start_idx = prefix_candidates[-1] if prefix_candidates else 0
            watermark = new_wm
            working, encoded, image_stats = render(start_idx, watermark)
            while (
                (bytes_over and len(encoded) > target_bytes)
                or (count_over and image_count(working) > target_images)
            ):
                if start_idx > 0:
                    lower = [b for b in step_boundaries if 0 < b < start_idx]
                    start_idx = lower[-1] if lower else 0
                elif total_images - watermark > 8:
                    watermark = min(total_images - 8, max(watermark + 4, total_images - 8))
                else:
                    break
                working, encoded, image_stats = render(start_idx, watermark)
        with self.lock:
            self._image_trim_watermarks[thread_key] = (watermark, total_images)
            self._image_trim_starts[thread_key] = start_idx
        if (start_idx, watermark) != previous:
            self._record(
                "model_images_trimmed",
                **identity,
                reason="provider_rejected" if force_advance else reason,
                images_total=total_images,
                images_forwarded=image_count(working),
                window_start=start_idx,
                window_end=watermark,
                previous_window=list(previous),
                original_body_bytes=len(base_body),
                forwarded_body_bytes=len(encoded),
            )
        if not force_advance and image_stats and image_stats["replaced"]:
            self._record(
                "model_images_reused",
                **identity,
                **image_stats,
                original_body_bytes=len(base_body),
                forwarded_body_bytes=len(encoded),
                forwarded_body_sha256=hashlib.sha256(encoded).hexdigest(),
            )
        return working, encoded
