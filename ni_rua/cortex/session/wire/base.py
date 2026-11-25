"""Wire adapters: the provider-format half of the model egress relay.

``ModelRequestProxy`` owns every generic policy: budget admission and reservation, logical
call identity, thread attribution and cancellation, the blocked-transport gate, the retry
loop, transport recovery and its gate, the in-process/worker attempt, client-disconnect
watch, heartbeat thread, evidence events and clocks, and the image-trim *window state
machine*. A ``WireAdapter`` owns everything that depends on the provider wire format: which
paths are model calls, how a buffered upstream stream ends, when an answer is usable, how
provider errors are sanitised and classified, how relay-generated errors and denials look to
the client, where images live in a request and what replaces an omitted one.

Adapters are stateless. Every method is a pure function of its arguments (no I/O, no clock,
no model call), so one instance may be shared by threads, proxies and the ``--http-worker``
subprocess. Byte-level choices (key order, JSON separators, frame text) are part of each
adapter's contract because the relay forwards and delivers exactly the bytes they return.

Client profiles: how relay-generated errors, denials and the delivery of a buffered answer must
look depends on the agent *client* behind the wire, not only on the wire format (Claude Code
re-sends an HTTP 400 twice but a 403 once; OpenCode 2.2.7 re-sends a 409/429/5xx answer or a
stream without its terminal ten times). Each adapter is therefore constructed with
``client=<profile>`` and lists the profiles it implements in ``client_profiles`` (see
``ClientProfile``); the profile is fixed for the adapter's lifetime and recorded with the run.

Implemented wires: ``openai_responses`` (profiles ``codex`` and ``opencode``) and
``anthropic_messages`` (profiles ``claude_code`` and ``opencode``). Later wires subclass
``WireAdapter`` and implement the methods marked "new wires" below; the base raises
``NotImplementedError`` there so a missing method fails loudly, never silently.

The ``opencode`` profile (both wires; measured against opencode 2.2.7 by the OC2-E1 capture,
``tests/fixtures/opencode/capture_opencode.py``, first against 1.19.5 in spike report 06 S6) is
*deferred delivery*: no byte before the attempt is validated, then the complete answer or one
neutral HTTP 400 JSON (``OPENCODE_*`` constants below); thread key ``x-opencode-session-id``.
"""
from copy import deepcopy
import json
import re
from types import MappingProxyType
from typing import Mapping, NamedTuple

from ...relay import continuity as context_signal


class Route(NamedTuple):
    """One relay endpoint, resolved from the client's HTTP method and raw path.

    kind: ``"generation"`` — a model completion: budget-charged, retried, buffered until a
        validated terminal and released once (the full relay path); ``"auxiliary"`` — a
        side request that carries history (e.g. Anthropic ``count_tokens``): projected with
        the relay's ``request_projection`` hook, forwarded once with host credentials, never
        charged, never retried, recorded as one ``model_aux_request`` event; ``"local"`` —
        answered by the relay itself (``WireAdapter.not_found``), never forwarded.
    endpoint: normalised client path without query string; recorded as
        ``identity.endpoint`` in every evidence event of the call.
    upstream_path: path (plus any query string the wire preserves) appended to the
        upstream base URL for the forwarded request.
    compaction: True for a provider-side context-compaction endpoint (Responses
        ``/responses/compact``). Its admission starts a new image-context epoch and resets
        the thread's image-trim window; image trim/reuse and ``service_tier`` are not
        applied to its body; a non-streaming JSON answer is complete without a ``status``.
        Default False.
    """
    kind: str
    endpoint: str
    upstream_path: str
    compaction: bool = False


class WireReply(NamedTuple):
    """A relay-generated HTTP answer in the wire's own format.

    status: HTTP status code sent to the client.
    reason: short reason phrase. When ``body`` is None the relay answers with the standard
        ``BaseHTTPRequestHandler.send_error(status, reason)`` page (``None`` = the default
        phrase for ``status``); otherwise it is the status-line phrase (``None`` = default).
    headers: response headers in send order; the relay appends ``Content-Length`` and
        ``Connection: close`` itself. Ignored when the reply is written into an already
        open event stream.
    body: exact response bytes, or None for the standard error page.
    """
    status: int
    reason: str | None = None
    headers: Mapping[str, str] = MappingProxyType({})
    body: bytes | None = None


class ClientProfile(NamedTuple):
    """Delivery behaviour one agent client needs from the relay (selected by ``client=``).

    early_stream: True — a streaming request is answered at once with HTTP 200 and
        event-stream headers, and ``keepalive_frame()`` is written every
        ``keepalive_interval_s`` seconds while the relay buffers the upstream answer; a later
        failure is delivered inside that open stream. False — *deferred delivery*: the relay
        sends no byte (no status line, no keepalive) until the attempt, including relay retries
        and recovery cooldowns, is validated, then either the complete answer or one
        ``relay_error`` HTTP reply. Default True (the Codex behaviour).
    client_loop_guard: True — the client re-requests when a stream ends without a terminal or
        fails oddly (observed: OpenCode re-sent one body 45–970 times in seconds, and re-requested
        after every delivered answer whose usage it could not parse), so the relay (a) runs its
        client-retry runaway guard and its consume guard on this client's generation requests and
        (b) on shutdown lets every open client stream receive its ``relay_error`` before closing
        the socket. False keeps the Codex relay byte-identical (pinned by golden fixtures).
        Default False.
    record_provider_message: True — the relay's private failure evidence (``response_error``)
        also keeps a bounded operator copy of the provider's error message (``message``, see
        ``provider_error_message``), shown in operator events, ``runtime/model_requests`` and the
        console. It never enters client bytes: these profiles answer relay failures with a
        neutral ``relay_error``. False (Codex, whose ``relay_error`` carries the evidence to the
        client and whose evidence is pinned by golden fixtures): identifiers only. Default False.
    """
    early_stream: bool = True
    client_loop_guard: bool = False
    record_provider_message: bool = False


CODEX_PROFILE = ClientProfile(early_stream=True, client_loop_guard=False)
# OpenCode (all wires): deferred delivery and the runaway/consume guards. Measured on 2.2.7 (OC2-E1 capture, both
# wires, one answer form served on every request of a session):
# * 1 attempt: every relay denial and ``relay_error`` (neutral HTTP 400), any other 400 without the rate-limit text,
#   an Anthropic ``invalid_request_error`` stream event (first or later event);
# * 11 attempts (10 retries with backoff, 77-92 s): 409/429/500/529, a 400 with the rate-limit text, an Anthropic
#   ``overloaded_error`` event, a Responses ``response.failed`` with an unclassified code, and a stream that ends
#   without its terminal. A failure inside an early-opened stream is therefore retried or not by its error code;
#   deferred delivery never sends one;
# * 3 attempts: an empty ``response.incomplete`` (continued twice with a ``synthetic`` message).
# 1.19.5 (report 06 S6.9) differed: five retries, digit / "overloaded" / "connection" patterns retried at HTTP 400, a
# first-event stream error retried, and a stream without terminal re-sent in a tight loop (45-970 times), the reason
# for the runaway guard; it stays on for identical re-sent bodies and unconsumed completions.
OPENCODE_PROFILE = ClientProfile(early_stream=False, client_loop_guard=True, record_provider_message=True)
# Operator copy of a provider error message (``provider_error_message``): at most this many characters.
PROVIDER_MESSAGE_LIMIT = 300
# Credential-looking runs masked in that copy: sk-/pk-/rk-... keys, "Bearer <token>", and any run of 32 or more
# key-alphabet characters (long tokens, signatures, request digests).
_CREDENTIAL_LIKE = re.compile(r"(?i)\bbearer\s+\S+|\b(?:sk|pk|rk)[-_][A-Za-z0-9_-]{6,}|[A-Za-z0-9_+/=-]{32,}")
_CONTROL_CHARS = re.compile(r"[\x00-\x1f\x7f]")


# ``tool_result_images = "user_blocks"``: text left in a tool result whose only content was moved images, and the
# label put before each moved image (``k`` = 1-based position among that tool result's images).
MOVED_IMAGE_NOTE = "(the image of this tool result follows after the tool results)"


def tool_result_image_label(k, tool_id):
    """``"[image <k> of tool result <tool_id>]"``: the text block placed before a moved tool-result image."""
    return f"[image {k} of tool result {tool_id}]"


def provider_error_message(body, limit=PROVIDER_MESSAGE_LIMIT):
    """Bounded operator copy of the message of a provider error body, or None.

    body: decoded error JSON: ``{"error": {"message": ...}}`` (Anthropic, OpenAI), ``{"error": "<text>"}`` or
    ``{"message": ...}``; anything else gives None. Control characters become spaces, whitespace is collapsed,
    credential-looking runs are replaced by ``[redacted]`` and the text is cut to ``limit`` characters (``…``
    appended when cut). Example: ``{"type": "error", "error": {"type": "invalid_request_error", "message":
    "IMAGE_INPUT_UNSUPPORTED"}}`` -> ``"IMAGE_INPUT_UNSUPPORTED"``. Operator data only (never client bytes or model
    context).
    """
    if not isinstance(body, dict):
        return None
    error = body.get("error")
    message = (error.get("message") if isinstance(error, dict) else error if isinstance(error, str)
               else body.get("message"))
    if not isinstance(message, str):
        return None
    text = " ".join(_CREDENTIAL_LIKE.sub("[redacted]", _CONTROL_CHARS.sub(" ", message)).split())
    if not text:
        return None
    return text if len(text) <= limit else text[:limit].rstrip() + "…"
# Native thread key OpenCode sends on every model request, its compaction summary request included (2.2.7 sends no
# tool-less helper call). A child session's requests carry the child's own id here; 2.2.7 adds the root session id
# (``x-opencode-session``, ``x-session-id``, ``x-session-affinity``) and the parent id
# (``x-opencode-parent-session-id``), which the relay does not read.
OPENCODE_THREAD_HEADER = "x-opencode-session-id"
# Client-visible messages of relay-generated OpenCode answers (HTTP 400 for denials and failures,
# 404 for unknown routes). OpenCode 2.2.7 retries an HTTP 400 when its message or body matches the
# rate-limit pattern ("rate limit", "too many requests", ...) or ``x-should-retry: true`` is sent
# (``false`` forbids a retry); 1.19.5 also retried a 400 whose text held digits such as 503,
# "overloaded", "timeout", "server error", "connection", "try again", "capacity", "unavailable" or
# "exhausted". These strings contain none of those words or digits (checked by the tests).
OPENCODE_DENIAL_MESSAGES = MappingProxyType({
    "thread_cancelled": "ni_rua_model_thread_cancelled",
    "transport_blocked": "ni_rua_model_transport_blocked",
    "budget": "ni_rua_model_budget_spent",
    "client_retry_runaway": "ni_rua_client_retry_runaway",
})
OPENCODE_REQUEST_FAILED_MESSAGE = "ni_rua_model_request_failed"
OPENCODE_ROUTE_NOT_FOUND_MESSAGE = "ni_rua_relay_route_not_found"
# OpenCode renders every configured instruction file into the system text as
# "Instructions from: <absolute path>\n<content>"; the path names the run's private directory, so the
# egress walker of the opencode profile replaces that line (system text only) by a constant.
INSTRUCTION_HEADER = re.compile(r"^Instructions from: .*$", re.MULTILINE)
INSTRUCTION_HEADER_REPLACEMENT = "Instructions:"


def neutralize_instruction_header(text):
    """``text`` with every OpenCode ``Instructions from: <path>`` line replaced by ``Instructions:``."""
    return INSTRUCTION_HEADER.sub(INSTRUCTION_HEADER_REPLACEMENT, text) if isinstance(text, str) else text


def header_value(headers, name):
    """Case-insensitive request header lookup over ``http.client.HTTPMessage`` or a plain mapping;
    ``""`` when absent. ``name`` is lower case."""
    if headers is None:
        return ""
    getter = getattr(headers, "get_all", None)
    if getter is not None:
        values = getter(name) or []
        return values[0] if values else ""
    for key, value in headers.items():
        if isinstance(key, str) and key.lower() == name:
            return value
    return ""


class WireAdapter:
    """Format-specific relay operations; see the module docstring for the split.

    Constructor: ``WireAdapter(*, client=None)`` — ``client`` names the client profile (a key
    of ``client_profiles``); None selects the adapter's first (default) profile. An unknown
    profile raises ``ValueError("unsupported_wire_client:<wire>:<client>")``.

    Class attributes (constants of one wire; never tuned per task or per run):

    name: stable wire identifier recorded with requests and passed to the
        ``--http-worker`` subprocess (``"openai_responses"``, ``"anthropic_messages"``, later
        ``"openai_chat"``).
    client_profiles: profile name → ``ClientProfile`` for every client this adapter supports;
        the first key is the default. The base declares none (``client`` stays ``"codex"``
        with the Codex profile) so a bare ``WireAdapter()`` remains constructible.
    methods: HTTP methods the relay routes through ``route()`` for this wire. Requests with any
        other method keep the standard ``BaseHTTPRequestHandler`` 501 page (Codex behaviour).
    auth_header_names: lower-case → canonical names of the upstream credential headers the
        host may inject (``upstream_auth_headers``). Client copies of these headers are
        stripped before host credentials are added.
    supports_service_tier: whether the operator ``service_tier`` override may be written into
        generation request bodies on non-compaction routes.
    body_limit_bytes: forwarded body size (bytes) above which the relay trims the oldest
        inline images (when image trim is enabled).
    trim_target_bytes: body size (bytes) the image-trim window aims for once trimming started.
    keepalive_interval_s: seconds between two ``keepalive_frame()`` writes into an open client
        event stream while the relay buffers (default 1.0, the Codex heartbeat).
    evidence_path: relay path (str) a host-side evidence client posts ``build_evidence_request``
        bodies to (the wire's own non-compaction generation route); None for a wire without
        evidence requests.

    Instance attributes set from the profile: ``client`` (profile name), ``early_stream``,
    ``client_loop_guard`` and ``record_provider_message`` (see ``ClientProfile``).
    """

    name: str = ""
    client_profiles: Mapping[str, ClientProfile] = MappingProxyType({})
    methods: frozenset = frozenset({"POST"})
    auth_header_names: Mapping[str, str] = MappingProxyType({})
    supports_service_tier: bool = False
    body_limit_bytes: int = 0
    trim_target_bytes: int = 0
    keepalive_interval_s: float = 1.0
    evidence_path: str | None = None
    client: str = "codex"
    early_stream: bool = CODEX_PROFILE.early_stream
    client_loop_guard: bool = CODEX_PROFILE.client_loop_guard
    record_provider_message: bool = CODEX_PROFILE.record_provider_message

    def __init__(self, *, client=None):
        profiles = self.client_profiles
        if not profiles:
            if client not in (None, "codex"):
                raise ValueError("unsupported_wire_client:" + str(self.name) + ":" + str(client))
            return
        if client is None:
            client = next(iter(profiles))
        if not isinstance(client, str) or client not in profiles:
            raise ValueError("unsupported_wire_client:" + str(self.name) + ":" + str(client))
        profile = profiles[client]
        self.client = client
        self.early_stream = profile.early_stream
        self.client_loop_guard = profile.client_loop_guard
        self.record_provider_message = profile.record_provider_message

    # ------------------------------------------------------------------ request ingress
    def route(self, method, raw_path):
        """Resolve a client request to a ``Route``, or None for an unknown path.

        method: HTTP method (``"POST"``). raw_path: request target exactly as received,
        including any query string. Unknown paths are answered with ``not_found()``.
        """
        raise NotImplementedError

    def route_for_endpoint(self, endpoint):
        """``Route`` for a recorded ``identity.endpoint`` (POST); unknown → generation route.

        Used where only the endpoint string survives (evidence events, compatibility
        delegates). ``endpoint`` is returned unchanged as both paths of the fallback.
        """
        route = self.route("POST", endpoint) if isinstance(endpoint, str) else None
        return route if route is not None else Route("generation", endpoint, endpoint)

    def decode_body(self, headers, raw):
        """Return the identity-encoded request body (bytes) for a received body.

        headers: client request headers. raw: body bytes as received. The default is a
        passthrough (identity encoding only). A wire that accepts compressed bodies
        decodes them here and must also drop ``Content-Encoding`` from forwarded headers.
        """
        return raw

    def thread_key(self, headers, request=None):
        """Native thread key the client sent, or ``""`` when the wire carries none.

        headers: client request headers. request: decoded JSON body, or None when the key is
        needed for admission before the body is read (the relay always passes None today,
        so a wire whose key lives only in the body returns ``""`` there). The relay maps
        the key to a registered owner for attribution and cancellation.
        """
        return ""

    def is_streaming(self, request):
        """True when the client asked for an event stream. request: decoded JSON body. With an
        ``early_stream`` profile the relay then answers 200 SSE at once and sends
        ``keepalive_frame`` while buffering; a deferred profile sends the whole stream at the end."""
        raise NotImplementedError

    def requested_model(self, request):
        """Model name the client requested (recorded as ``identity.requested_model``), or None."""
        raise NotImplementedError

    # -------------------------------------------- continuity tracking (design §4.3, P1-3a)
    def history_view(self, request):
        """New wires: ``context_signal.HistoryView`` of a decoded request body, i.e. its system /
        instruction entries and its conversation items with the wire layout removed (for example
        ``context_signal.anthropic_messages_history``). Pure; never mutates ``request``."""
        raise NotImplementedError

    def conversation_key(self, request):
        """Hex digest of the request's canonical system/instruction text only (no conversation
        item). Equal keys mean equal system text; the relay passes it to
        ``ContinuityTracker`` as ``system_key``. Chains are keyed by ``(native thread key, system_key)``:
        on the actor's thread a different key is a subagent chain unless its history continues the main
        chain, in which case the change is reported as ``system_changed`` (``context_signal``, rules 2
        and 7)."""
        return context_signal.conversation_key(self.history_view(request))

    def history_fingerprints(self, request):
        """``[(item_head_digest, (block_digest, ...)), ...]``: one entry per conversation item in
        context order, content canonicalised (a plain string equals one text block) and
        transport-volatile fields (cache markers, client-regenerated ids) dropped
        (``context_signal.history_fingerprints``)."""
        return context_signal.history_fingerprints(self.history_view(request))

    def history_image_count(self, request):
        """Number of image parts in the conversation items (any depth, any source), the
        tracker's ``image_count`` input (``context_signal.image_count``)."""
        return context_signal.image_count(self.history_view(request))

    def is_subagent_request(self, headers):
        """True when request headers explicitly mark a native subagent/fork chain."""
        return False

    def is_helper_request(self, request):
        """True when the request declares no tools (a compaction-summary, title or side call of
        the client, or a relay evidence request). Helpers are attributed ``call_purpose="helper"``
        and charged like any generation call, but never compared or committed by the tracker. A request
        with tools is ``"main"`` or ``"subagent"`` (``ContinuityTracker.admit``)."""
        return context_signal.is_helper_request(request)

    def continuity_inputs(self, request):
        """Keyword inputs of ``ContinuityTracker.observe`` / ``commit`` for one decoded body:
        ``{helper, fingerprints, image_count, system_key}`` (see the methods above). The relay
        computes them once per logical call on the body after the egress projection and before
        its own image trim/reuse, and reuses the same dict for the commit."""
        view = self.history_view(request)
        # OpenCode 子会话已由独立会话头隔离；主会话跳过 system_key 比较以根除压缩与 system 同步变化时的粘滞失明
        system_key = None if self.client == "opencode" else context_signal.conversation_key(view)
        return dict(helper=self.is_helper_request(request), fingerprints=context_signal.history_fingerprints(view),
                    image_count=context_signal.image_count(view), system_key=system_key)

    def history_model_output_flags(self, request):
        """One bool per conversation item of ``history_view(request).items`` (same order and length as
        ``history_fingerprints``): True when the item is the model's own earlier output that the client
        echoes back, False for everything the client itself adds (user and system messages, tool
        results) and for any item the wire does not recognise. The default recognises
        ``role: "assistant"`` items, the whole model turn on the Anthropic Messages and Chat
        Completions wires; a wire with separate output item types extends it.

        Used only by the relay's consume guard (``ModelRequestProxy``, ``client_loop_guard``
        profiles): a request whose new items are all model output added no client input after the
        completion it follows. Unknown items count as client input, so a wire this method does not
        know can only weaken the guard, never trip it. Pure; never mutates ``request``."""
        return tuple(isinstance(item, dict) and item.get("role") == "assistant"
                     for item in self.history_view(request).items)

    def project_request(self, request, presentation, *, registered_tools, tool_name_prefix,
                        on_unknown_part=None, cache=None):
        """New wires: egress information-boundary re-scrub of one request body.

        request: decoded JSON body. presentation: the session's ``ActorPresentation``.
        registered_tools: frozenset of NI-RUA tool names the agent may call.
        tool_name_prefix: prefix the agent's client puts before NI-RUA tool names.
        on_unknown_part: optional callable ``(type_name)`` called once per content block of a
        type the walker does not know (it is still scrubbed by the generic key walker); the
        session records it as ``model.egress_unknown_part``. Default None (not reported).
        Returns a new body: prose scrubbed with ``present_text``, JSON schemas with
        ``present_data``, structured arguments with ``present``; ids, pairing fields,
        media bytes, signatures and sampling parameters untouched; forbidden tools or
        media raise ``ValueError``. Deterministic, so prompt caches survive. The Responses
        scrub used by Codex is ``openai_responses.present_responses_request``.
        """
        raise NotImplementedError

    # ------------------------------------------------------------------------- images
    def image_locations(self, request, *, include_labels=False):
        """Inline images of a request in context order.

        Returns a list of ``(parts, index)`` — the containing list and the part's index —
        or ``(parts, index, (observation_id, camera_id))`` when ``include_labels`` is True
        (either label may be None). Only inline image data counts; remote references do not.
        """
        raise NotImplementedError

    def step_image_boundaries(self, request):
        """``(locations, boundaries)``: ``image_locations(request)`` and the sorted image
        counts at which one agent step ends (trim windows only start/end on boundaries)."""
        raise NotImplementedError

    def image_part_size(self, part):
        """Characters of inline payload an image part contributes to the body (int)."""
        raise NotImplementedError

    def placeholder_part(self, text):
        """The text part (a new dict) that replaces an omitted image part."""
        raise NotImplementedError

    def move_tool_result_images(self, request):
        """``UpstreamConfig.tool_result_images = "user_blocks"``: the final upstream body with every image an agent
        returned inside a tool result moved, unchanged and in order, to a labelled user block
        (``tool_result_image_label``) right after the tool results (new wires). Returns ``(request, moved)``: the
        same object when nothing changed, else a new request (inputs are never mutated); ``moved`` = images moved.
        Deterministic (a function of the body only), so a repeated history gives the same bytes."""
        raise NotImplementedError

    def reuse_exact_images(self, request):
        """Replace later exact duplicates of an inline image part with a reference text part.

        Returns ``(new_request, stats)`` with ``stats = {images, replaced, saved_url_chars}``;
        the input is never mutated (``core.presentation.image_reuse.reuse_exact_images``).
        """
        raise NotImplementedError

    @staticmethod
    def omitted_image_text(observation_id, camera_id):
        """Placeholder prose for a trimmed image; names how to re-read the frame when known."""
        if observation_id and camera_id:
            return (f"[Earlier historical observation image omitted (observation_id={observation_id}, camera={camera_id}) "
                    f"to stay within upstream gateway payload limit; call history_view_observation(observation_id='{observation_id}') "
                    "if you need to inspect this frame.]")
        if observation_id:
            return (f"[Earlier historical observation image omitted (observation_id={observation_id}) "
                    f"to stay within upstream gateway payload limit; call history_view_observation(observation_id='{observation_id}') "
                    "if you need to inspect this frame.]")
        return "[Earlier historical observation image omitted to stay within upstream gateway payload limit.]"

    def trim_image_range(self, request, start_idx, end_idx):
        """Copy of ``request`` with inline images ``[start_idx, end_idx)`` replaced by placeholders.

        Indices count ``image_locations`` entries (clamped to the available range). Returns
        ``(request_copy, replaced_count)``; ``(request, 0)`` unchanged for an empty range.
        """
        if end_idx <= start_idx or end_idx <= 0:
            return request, 0
        cloned = deepcopy(request)
        locations = self.image_locations(cloned, include_labels=True)
        start = max(0, min(start_idx, len(locations)))
        end = max(start, min(end_idx, len(locations)))
        for parts, idx, (observation_id, camera_id) in locations[start:end]:
            parts[idx] = self.placeholder_part(self.omitted_image_text(observation_id, camera_id))
        return cloned, end - start

    def trim_prefix_images(self, request, prefix_count):
        """``trim_image_range(request, 0, prefix_count)``: omit the oldest ``prefix_count`` images."""
        return self.trim_image_range(request, 0, prefix_count)

    def trim_oldest_images(self, body, keep_latest=24):
        """Keep the newest ``keep_latest`` inline images of a JSON body (bytes); stateless.

        Returns the body unchanged when it is not JSON or already within the count, else the
        compact UTF-8 re-encoding (``separators=(",", ":")``, ``ensure_ascii=False``).
        """
        try:
            request = json.loads(body)
        except ValueError:
            return body
        total = len(self.image_locations(request))
        if total <= keep_latest:
            return body
        trimmed, _ = self.trim_prefix_images(request, total - keep_latest)
        return json.dumps(trimmed, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()

    def image_rejected(self, failure_evidence):
        """True when a sanitised failure (``failure_evidence`` dict) means the provider rejected
        the request's images, so a retry must first advance the image-trim window."""
        return False

    # ----------------------------------------------------------------- response stream
    def emitted_tool_calls(self, raw):
        """The tool calls of one validated usable answer, in the order the model emitted them.

        raw: the buffered answer bytes (event stream or JSON body) exactly as ``response_evidence`` accepted them.
        Returns a list of ``(call_id, name, arguments)``: ``call_id`` the model's own id of the call (str, or None
        when the wire carries none), ``name`` the tool name as the model wrote it (client prefix included),
        ``arguments`` the decoded JSON arguments (None when they do not decode). Read-only: the relay still
        forwards ``raw`` unchanged; the only use is ``ModelRequestProxy.on_tool_calls`` (ordering of one step's
        concurrent tool calls). The base returns ``[]`` (no order known); malformed input raises ``ValueError``.
        """
        return []

    @staticmethod
    def is_event_stream(raw):
        """True when a buffered upstream body (bytes) is a server-sent event stream."""
        return raw.lstrip().startswith((b"event:", b"data:", b":"))

    def keepalive_frame(self):
        """Bytes the relay writes every ``keepalive_interval_s`` seconds into an open client
        event stream while it buffers the upstream answer, or None to send no keepalive.
        Only used by ``early_stream`` profiles."""
        return None

    def terminal_sse_type(self, block):
        """Event type when one SSE block (bytes, without the blank-line separator) ends the
        upstream stream, else None. The relay stops reading at the first terminal block."""
        raise NotImplementedError

    def terminal_allows_reuse(self, terminal_type):
        """True when a stream that ended with ``terminal_type`` left the upstream keep-alive
        connection clean enough to pool for the next attempt."""
        return False

    def response_evidence(self, raw, route):
        """Validate a buffered upstream body and extract authoritative evidence.

        raw: complete upstream body bytes. route: the ``Route`` it answered. Returns
        ``(complete, evidence, terminal)``: ``complete`` — a terminal was received;
        ``evidence`` — dict recorded on ``model_request_finished`` (``response_id``,
        ``served_model``, ``response_status``, raw ``usage``, ``usage_known``, and a
        sanitised ``response_error`` for failures); ``terminal`` — the decoded final
        response object. Raises ``EOFError`` when the terminal is missing and
        ``ValueError``/``TypeError``/``AttributeError`` for a malformed body. Never builds
        output from deltas.
        """
        raise NotImplementedError

    def completion_outcome(self, complete, evidence):
        """Delivery decision for a validated answer (first two ``response_evidence`` values).

        Returns ``"usable"`` (deliver, charge one completion), ``"failed"`` (explicit
        provider failure: deliver ``failed_payload`` after the retry policy, block the
        transport, manual retry allowed), ``"incomplete"`` (provider stopped early: same
        delivery, no manual retry) or ``"not_completed"`` (no usable terminal: resent like
        an incomplete stream).
        """
        raise NotImplementedError

    def failed_response_error(self, body):
        """Sanitised provider error of a failed response or HTTP error body (dict).

        Returns ``{code, type, code_state, classification, retryable, retry_after_s}``:
        identifiers only (never the provider message), ``classification`` from the wire's
        table, ``retry_after_s`` in seconds or None.
        """
        raise NotImplementedError

    def operator_response_error(self, body):
        """``failed_response_error(body)`` plus, for profiles with ``record_provider_message``, the bounded
        provider message as ``message`` (``provider_error_message``; omitted when the body has none). This is the
        private evidence the relay records; it never reaches the client."""
        error = self.failed_response_error(body)
        message = provider_error_message(body) if self.record_provider_message else None
        return dict(error, message=message) if message is not None else error

    def classify_http_failure(self, status, headers, body, *, policy):
        """``policy.InfrastructureFailure`` (or ``CapabilityFailure``) for an upstream HTTP error.

        status: HTTP status (>= 400). headers: upstream response headers. body: decoded JSON
        error body or None. policy: the relay's pinned RoboProbe policy module (shared
        ``InfrastructureFailure`` type and status classifier); passed in so the adapter has
        no import-time dependency on it.
        """
        raise NotImplementedError

    def failed_payload(self, terminal, *, streaming):
        """Client-visible bytes for a failed/incomplete terminal, with all output removed.

        terminal: decoded final response object. streaming: the upstream body was an event
        stream (answer with one terminal event) rather than one JSON document. The bytes are
        delivered as the HTTP 200 answer (or into the open stream). Returns None when the
        wire/profile has no in-band failure form for this delivery (deferred profiles, or a
        non-streaming Messages answer): the relay then answers with ``relay_error`` after the
        retry policy, exactly as for any other terminal failure.
        """
        raise NotImplementedError

    def relay_error(self, *, streaming, evidence, manual_retry_allowed):
        """``WireReply`` for a model call the relay could not complete.

        streaming: the client event stream is already open (only ``body`` is written; always
        False for deferred profiles). evidence: sanitised ``failure_evidence`` dict or None.
        manual_retry_allowed: an operator may authorise one more attempt (explicit provider
        error). Profiles other than ``codex`` must not put digits, identifiers or provider
        text in the client-visible bytes (§4.2 rules for relay-generated errors).
        """
        raise NotImplementedError

    def denial(self, kind):
        """``WireReply`` refusing admission before any upstream byte is sent.

        kind: ``"thread_cancelled"`` (thread explicitly cancelled), ``"transport_blocked"``
        (an earlier failure awaits operator review), ``"budget"`` (episode completion budget
        exhausted or fully reserved) or ``"client_retry_runaway"`` (a client re-request loop: the
        client re-sent one body in a tight loop, or kept re-requesting after delivered completions
        without adding any input; the relay records which guard refused it; only requested for
        ``client_loop_guard`` profiles).
        """
        raise NotImplementedError

    def not_found(self):
        """``WireReply`` for an unknown or relay-local path (default: standard 404 page)."""
        return WireReply(404)

    def normalize_usage(self, usage):
        """Map a raw provider usage object to NI-RUA ``TOKEN_FIELDS`` counters.

        Returns a dict with every ``usage.TOKEN_FIELDS`` key; a counter the provider did not
        report (or reported as a non-integer / negative value) stays None, never 0.
        """
        raise NotImplementedError

    def context_edit_applied(self, terminal):
        """True when the provider reports it edited the context server-side (e.g. Anthropic
        ``context_management.applied_edits``); such a response starts a new image epoch."""
        return False

    # ------------------------------------------------- evidence helper requests (new wires)
    def build_evidence_request(self, *, model, base_instructions, developer_instructions, input_items,
                               output_schema, effort):
        """New wires: one fresh no-tools structured request for visual review/self-briefing (§9.3).

        model: model name requested (str). base_instructions / developer_instructions: instruction
        text (str), already presented. input_items: evidence items in the Codex user-input shape,
        already presented: ``{"type": "text", "text": str}`` and ``{"type": "image", "url":
        "data:<mime>;base64,<data>"}`` (inline images only). output_schema: JSON schema (dict) of the
        single answer, already presented. effort: reasoning effort label (str, e.g. ``"high"``).
        Returns ``(body, extra_headers)``: a streaming JSON body with no tools, no history and no
        cache markers, and the extra request headers the wire needs (dict). Raises ``ValueError``
        (``evidence_input_item_unsupported`` / ``evidence_image_requires_inline_data_url``) for an
        item it cannot express.
        """
        raise NotImplementedError

    def evidence_answer(self, terminal):
        """New wires: the single final answer text of an evidence request from the decoded
        ``terminal`` of ``response_evidence``. Raises ``ValueError``:
        ``sol_agent_unverified_or_incomplete_turn`` (not a usable completed answer),
        ``sol_agent_unexpected_tool_or_context_item`` (any tool or hosted-tool output),
        ``sol_agent_requires_one_structured_final_answer`` (zero or several answers)."""
        raise NotImplementedError


# Error strings shared by the evidence answer extractors (the same strings CodexEvidenceAgent raises).
EVIDENCE_UNVERIFIED = "sol_agent_unverified_or_incomplete_turn"
EVIDENCE_TOOL_ACTIVITY = "sol_agent_unexpected_tool_or_context_item"
EVIDENCE_ONE_ANSWER = "sol_agent_requires_one_structured_final_answer"


def evidence_parts(input_items):
    """Validate evidence items (Codex user-input shape) and yield ``("text", text)`` or
    ``("image", media_type, base64_data, data_url)`` per item, in order."""
    if not isinstance(input_items, (list, tuple)) or not input_items:
        raise ValueError("evidence_input_item_unsupported")
    for item in input_items:
        kind = item.get("type") if isinstance(item, dict) else None
        if kind == "text" and isinstance(item.get("text"), str):
            yield ("text", item["text"])
        elif kind == "image" and isinstance(item.get("url"), str):
            url = item["url"]
            head, sep, data = url.partition(";base64,")
            if not (url.startswith("data:image/") and sep and data):
                raise ValueError("evidence_image_requires_inline_data_url")
            yield ("image", head[len("data:"):], data, url)
        else:
            raise ValueError("evidence_input_item_unsupported")
