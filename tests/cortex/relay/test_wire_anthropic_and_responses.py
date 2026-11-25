"""Merged cortex tests:
- tests/test_wire_anthropic_messages.py: Unit tests of the Anthropic Messages wire adapter (P1-2) and its egress information-boundary walker.
- tests/test_wire_openai_responses.py: Unit tests of the WireAdapter base and the OpenAI Responses wire (P0-2 extraction).
"""

NI_RUA_TEST_TIER = "component"

from copy import deepcopy
import gzip
import json
from pathlib import Path
import re
import tempfile
from types import MappingProxyType
import unittest
from ni_rua.apps.reference.reference_api import upstream
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.core.storage.store import Store
from ni_rua.cortex.session.usage import RelayUsageTotal, TOKEN_FIELDS, account_usage
from ni_rua.cortex.session.wire import (AnthropicMessagesWire, CODEX_PROFILE, ClientProfile, OpenAIResponsesWire,
                                        Route, WireAdapter, WireReply, present_messages_request, wire_from_name)
from ni_rua.cortex.session.wire import anthropic_messages as am
import base64
import io
import sys
from contextlib import redirect_stdout
from unittest import mock
from ni_rua.core.presentation.image_reuse import reuse_exact_images, responses_image_slots
from ni_rua.cortex.relay import server as model_proxy
from ni_rua.cortex.relay.server import (AUTH_HEADERS, CLIENT_AUTH_HEADERS, ModelRequestProxy,
                                        upstream_auth_headers_copy)
from ni_rua.cortex.session.usage import TOKEN_FIELDS, account_responses_usage
from ni_rua.cortex.session.wire import (OpenAIResponsesWire, Route, WireAdapter, WireReply, wire_from_name)
import support.wire_golden_fixture as golden
from support.paths import FIXTURES_ROOT

# ==== tests/test_wire_anthropic_messages.py ====

ANTHROPIC_WIRE = AnthropicMessagesWire()
POLICY = upstream().joint
FIXTURE = FIXTURES_ROOT / "anthropic_egress_bodies.json"
DIGEST = "0123456789abcdef" * 4
ARTIFACT = "artifact:" + "a" * 64 + "-" + "b" * 16
LONG_OBSERVATION = "observation-" + "c" * 32
PLANTED = f"sha256={DIGEST} see {ARTIFACT} and {LONG_OBSERVATION}"
PRIVATE_IMAGE_BLOCK = ("[Image: source: " + "/private/run/claude-config/projects/-private-run-actor/"
                       + "9ff8e771-aa08-4c17-b00d-032e1cb2011f/tool-results/mcp-ni-rua-blob-1794338642782-z5ip6n.png]")
MODELS_ID_PATH = "/v1/models/" + "claude-opus-5-5"
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
PDF = "JVBERi0xLjQKJcfsj6IKMSAwIG9iago8PD4+CmVuZG9iagp0cmFpbGVyCjw8Pj4KJSVFT0YK"
PREFIX = "mcp__ni_rua__"
REGISTERED = frozenset({"robot_observe", "robot_move"})
FORBIDDEN_WORDS = ("rate limit", "overloaded", "timeout", "try again", "capacity", "server error", "connection",
                   "unavailable")


def sse(event, data):
    return f"event: {event}\ndata: {json.dumps(data)}\n\n".encode()


def message_start(**usage):
    return sse("message_start", {"type": "message_start", "message": {
        "id": "msg_1", "type": "message", "role": "assistant", "model": "claude-served", "content": [],
        "stop_reason": None, "usage": dict(dict(input_tokens=10, cache_read_input_tokens=4,
                                                cache_creation_input_tokens=2, output_tokens=1), **usage)}})


def block_start(index, block):
    return sse("content_block_start", {"type": "content_block_start", "index": index, "content_block": block})


def delta(index, value):
    return sse("content_block_delta", {"type": "content_block_delta", "index": index, "delta": value})


def block_stop(index):
    return sse("content_block_stop", {"type": "content_block_stop", "index": index})


def message_end(stop_reason, output_tokens=7, **extra):
    return (sse("message_delta", {"type": "message_delta", "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                                  "usage": {"output_tokens": output_tokens}, **extra})
            + sse("message_stop", {"type": "message_stop"}))


def tool_stream(stop_reason, arguments, *, closed=True):
    frames = [message_start(), block_start(0, {"type": "text", "text": ""}), delta(0, {"type": "text_delta", "text": "hi"}),
              block_stop(0), block_start(1, {"type": "tool_use", "id": "toolu_1", "name": "mcp__ni_rua__robot_observe",
                                             "input": {}})]
    frames += [delta(1, {"type": "input_json_delta", "partial_json": piece}) for piece in arguments]
    if closed:
        frames.append(block_stop(1))
    return b"".join(frames) + message_end(stop_reason)


def generation():
    return ANTHROPIC_WIRE.route("POST", "/v1/messages")


class ProfileTest(unittest.TestCase):
    def test_client_profiles(self):
        self.assertEqual((ANTHROPIC_WIRE.client, ANTHROPIC_WIRE.early_stream, ANTHROPIC_WIRE.client_loop_guard), ("claude_code", True, True))
        self.assertEqual(AnthropicMessagesWire(client="claude_code").client, "claude_code")
        for bad in ("codex", "gemini_cli", "", 3):  # "opencode" became a supported profile in P3-4
            with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "unsupported_wire_client:anthropic_messages"):
                AnthropicMessagesWire(client=bad)
        codex = OpenAIResponsesWire()
        self.assertEqual((codex.client, codex.early_stream, codex.client_loop_guard), ("codex", True, False))
        self.assertEqual(OpenAIResponsesWire(client="codex").client, "codex")
        with self.assertRaisesRegex(ValueError, "unsupported_wire_client:openai_responses:claude_code"):
            OpenAIResponsesWire(client="claude_code")
        base = WireAdapter()
        self.assertEqual((base.client, base.early_stream, base.client_loop_guard, base.keepalive_interval_s),
                         ("codex", True, False, 1.0))
        with self.assertRaisesRegex(ValueError, "unsupported_wire_client"):
            WireAdapter(client="claude_code")
        self.assertEqual(CODEX_PROFILE, ClientProfile(True, False))
        self.assertEqual(ClientProfile(), CODEX_PROFILE)

    def test_wire_from_name(self):
        self.assertIsInstance(wire_from_name("openai_responses"), OpenAIResponsesWire)
        restored = wire_from_name("anthropic_messages", client="claude_code")
        self.assertIsInstance(restored, AnthropicMessagesWire)
        self.assertEqual(restored.client, "claude_code")
        with self.assertRaisesRegex(ValueError, "unknown_model_wire:anthropic_messages:client_profile_required"):
            wire_from_name("anthropic_messages")
        with self.assertRaisesRegex(ValueError, "unsupported_wire_client:anthropic_messages:gemini_cli"):
            wire_from_name("anthropic_messages", client="gemini_cli")  # "opencode" is supported since P3-4
        with self.assertRaisesRegex(ValueError, "unknown_model_wire:openai_chat"):
            wire_from_name("openai_chat", client="opencode")

    def test_constants(self):
        self.assertEqual(ANTHROPIC_WIRE.name, "anthropic_messages")
        self.assertEqual(dict(ANTHROPIC_WIRE.auth_header_names), {"x-api-key": "x-api-key", "authorization": "Authorization"})
        self.assertEqual((ANTHROPIC_WIRE.body_limit_bytes, ANTHROPIC_WIRE.trim_target_bytes),
                         (OpenAIResponsesWire.body_limit_bytes, OpenAIResponsesWire.trim_target_bytes))
        self.assertFalse(ANTHROPIC_WIRE.supports_service_tier)
        self.assertEqual(ANTHROPIC_WIRE.methods, frozenset({"POST", "GET"}))
        self.assertEqual(OpenAIResponsesWire.methods, frozenset({"POST"}))
        # Claude Code resets its stream watchdog on pings spaced at most 10 s apart (spike S5).
        self.assertLessEqual(ANTHROPIC_WIRE.keepalive_interval_s, 10.0)
        self.assertGreater(ANTHROPIC_WIRE.keepalive_interval_s, 0)


class IngressTest(unittest.TestCase):
    def test_routes(self):
        self.assertEqual(generation(), Route("generation", "/v1/messages", "/v1/messages"))
        self.assertEqual(ANTHROPIC_WIRE.route("POST", "/v1/messages?beta=true"),
                         Route("generation", "/v1/messages", "/v1/messages?beta=true"))
        self.assertEqual(ANTHROPIC_WIRE.route("POST", "/v1/messages/count_tokens?beta=true"),
                         Route("auxiliary", "/v1/messages/count_tokens", "/v1/messages/count_tokens?beta=true"))
        self.assertEqual(ANTHROPIC_WIRE.route("GET", "/v1/models"), Route("local", "/v1/models", ""))
        self.assertEqual(ANTHROPIC_WIRE.route("GET", MODELS_ID_PATH).kind, "local")
        for method, path in (("GET", "/v1/messages"), ("POST", "/v1/messages/"), ("POST", "/messages"),
                             ("POST", "/v1/models"), ("PUT", "/v1/messages"), ("POST", "/v1/complete"),
                             ("POST", "/responses"), ("POST", None)):
            with self.subTest(method=method, path=path):
                self.assertIsNone(ANTHROPIC_WIRE.route(method, path))
        self.assertEqual(ANTHROPIC_WIRE.route_for_endpoint("/v1/messages"), generation())
        self.assertFalse(ANTHROPIC_WIRE.route_for_endpoint("/v1/messages").compaction)

    def test_request_fields_and_thread_key(self):
        import http.client
        import email
        message = email.message_from_string("x-claude-code-session-id: sess-1\n\n", _class=http.client.HTTPMessage)
        self.assertEqual(ANTHROPIC_WIRE.thread_key(message), "sess-1")
        self.assertEqual(ANTHROPIC_WIRE.thread_key({"X-Claude-Code-Session-Id": "sess-2"}), "sess-2")
        self.assertEqual(ANTHROPIC_WIRE.thread_key({"thread-id": "t"}), "")
        self.assertEqual(ANTHROPIC_WIRE.thread_key({}), "")
        self.assertTrue(ANTHROPIC_WIRE.is_streaming({"stream": True}))
        self.assertFalse(ANTHROPIC_WIRE.is_streaming({"stream": "true"}))
        self.assertFalse(ANTHROPIC_WIRE.is_streaming([]))
        self.assertEqual(ANTHROPIC_WIRE.requested_model({"model": "claude-opus-5-5"}), "claude-opus-5-5")
        self.assertIsNone(ANTHROPIC_WIRE.requested_model({"model": 3}))

    def test_gzip_decoding(self):
        raw = json.dumps({"model": "m", "messages": []}).encode()
        self.assertIs(ANTHROPIC_WIRE.decode_body({}, raw), raw)
        self.assertIs(ANTHROPIC_WIRE.decode_body({"Content-Encoding": "identity"}, raw), raw)
        self.assertEqual(ANTHROPIC_WIRE.decode_body({"content-encoding": "gzip"}, gzip.compress(raw)), raw)
        with self.assertRaisesRegex(ValueError, "unsupported_request_content_encoding"):
            ANTHROPIC_WIRE.decode_body({"Content-Encoding": "br"}, raw)
        with self.assertRaisesRegex(ValueError, "invalid_gzip_request_body"):
            ANTHROPIC_WIRE.decode_body({"Content-Encoding": "gzip"}, b"not gzip")
        with self.assertRaisesRegex(ValueError, "invalid_gzip_request_body"):
            ANTHROPIC_WIRE.decode_body({"Content-Encoding": "gzip"}, gzip.compress(raw)[:-12])
        small = AnthropicMessagesWire()
        small.max_decoded_body_bytes = 100
        with self.assertRaisesRegex(ValueError, "decoded_request_body_too_large"):
            small.decode_body({"Content-Encoding": "gzip"}, gzip.compress(b"x" * 10_000))


class StreamTest(unittest.TestCase):
    def test_terminal_detection_and_keepalive(self):
        self.assertEqual(ANTHROPIC_WIRE.terminal_sse_type(b'event: message_stop\ndata: {"type":"message_stop"}'), "message_stop")
        self.assertEqual(ANTHROPIC_WIRE.terminal_sse_type(b'event: error\ndata: {"type":"error","error":{}}'), "error")
        for block in (b'event: ping\ndata: {"type": "ping"}', b'event: message_delta\ndata: {"type":"message_delta"}',
                      b"data: [DONE]", b"event: message_stop", b"data: {not json", b": comment", b""):
            with self.subTest(block=block):
                self.assertIsNone(ANTHROPIC_WIRE.terminal_sse_type(block))
        self.assertTrue(ANTHROPIC_WIRE.terminal_allows_reuse("message_stop"))
        self.assertFalse(ANTHROPIC_WIRE.terminal_allows_reuse("error"))
        self.assertEqual(ANTHROPIC_WIRE.keepalive_frame(), b'event: ping\ndata: {"type": "ping"}\n\n')

    def test_usable_completion_and_evidence(self):
        raw = message_start() + block_start(0, {"type": "thinking", "thinking": ""}) + delta(0, {
            "type": "signature_delta", "signature": "sig"}) + block_stop(0) + sse("ping", {"type": "ping"}) + \
            block_start(1, {"type": "text", "text": ""}) + delta(1, {"type": "text_delta", "text": "ok"}) + \
            block_stop(1) + message_end("end_turn")
        complete, evidence, terminal = ANTHROPIC_WIRE.response_evidence(raw, generation())
        self.assertTrue(complete)
        self.assertEqual(evidence, dict(response_id="msg_1", served_model="claude-served", response_status="completed",
                                        stop_reason="end_turn", usage=dict(input_tokens=10, cache_read_input_tokens=4,
                                        cache_creation_input_tokens=2, output_tokens=7), usage_known=True))
        self.assertEqual(terminal["stop_reason"], "end_turn")
        self.assertEqual(ANTHROPIC_WIRE.completion_outcome(complete, evidence), "usable")
        self.assertFalse(ANTHROPIC_WIRE.context_edit_applied(terminal))

    def test_stop_reasons_and_tool_json_integrity(self):
        cases = [
            ("tool_use", ['{"camera":', ' "head"}'], True, "usable"),
            ("end_turn", ['{}'], True, "usable"),
            ("stop_sequence", ['{"a": 1}'], True, "usable"),
            ("pause_turn", [], True, "usable"),
            ("refusal", [], True, "usable"),
            ("max_tokens", ['{"camera": "head"}'], True, "usable"),
            ("model_context_window_exceeded", ['{"a": [1, 2]}'], True, "usable"),
            ("max_tokens", ['{"camera": "he'], True, "incomplete"),
            ("model_context_window_exceeded", ['{"a": [1,'], True, "incomplete"),
            ("max_tokens", ['{"camera": "head"}'], False, "incomplete"),
            ("tool_use", ['["not", "object"]'], True, "incomplete"),
            ("tool_use", ['{"a": 1}'], False, "incomplete"),
            ("future_reason", ['{"a": 1}'], True, "incomplete"),  # P1-2 review item 5: closed usable set
            ("future_reason", ['{"a"'], True, "incomplete"),
        ]
        for stop_reason, arguments, closed, expected in cases:
            with self.subTest(stop_reason=stop_reason, arguments=arguments, closed=closed):
                complete, evidence, _ = ANTHROPIC_WIRE.response_evidence(tool_stream(stop_reason, arguments, closed=closed),
                                                               generation())
                self.assertEqual(ANTHROPIC_WIRE.completion_outcome(complete, evidence), expected)
                self.assertEqual(evidence["stop_reason"], stop_reason)

    def test_failed_missing_and_malformed_streams(self):
        error = sse("error", {"type": "error", "error": {"type": "overloaded_error", "message": "fixture-private"}})
        for raw in (message_start() + block_start(0, {"type": "text", "text": ""}) + error, error):
            complete, evidence, terminal = ANTHROPIC_WIRE.response_evidence(raw, generation())
            self.assertTrue(complete)
            self.assertEqual(evidence["response_status"], "failed")
            self.assertEqual(evidence["response_error"]["classification"], "server_overloaded")
            self.assertTrue(evidence["response_error"]["retryable"])
            self.assertFalse(evidence["usage_known"])
            # Private evidence keeps a bounded operator copy of the provider message (record_provider_message);
            # client bytes never carry it (tests/test_relay_provider_message.py).
            self.assertEqual(evidence["response_error"]["message"], "fixture-private")
            self.assertEqual(ANTHROPIC_WIRE.completion_outcome(complete, evidence), "failed")
            self.assertEqual(terminal["status"], "failed")
        with self.assertRaises(EOFError):
            ANTHROPIC_WIRE.response_evidence(message_start() + block_start(0, {"type": "text", "text": ""}), generation())
        for raw in (sse("message_stop", {"type": "message_stop"}),
                    message_start() + message_start(),
                    message_start() + delta(5, {"type": "text_delta", "text": "x"}),
                    message_start() + block_stop(3),
                    message_start() + b"data: [1, 2]\n\n"):
            with self.subTest(raw=raw[:60]), self.assertRaises(ValueError):
                ANTHROPIC_WIRE.response_evidence(raw, generation())
        # message_stop without any stop_reason: not a usable completion (resent like a cut stream)
        complete, evidence, _ = ANTHROPIC_WIRE.response_evidence(message_start() + sse("message_stop", {"type": "message_stop"}),
                                                       generation())
        self.assertEqual(ANTHROPIC_WIRE.completion_outcome(complete, evidence), "not_completed")

    def test_non_streaming_json(self):
        body = dict(id="msg_2", type="message", role="assistant", model="claude-served", stop_reason="tool_use",
                    content=[dict(type="tool_use", id="toolu_1", name="x", input={"a": 1})],
                    usage=dict(input_tokens=3, output_tokens=2),
                    context_management=dict(applied_edits=[dict(type="clear_thinking_20251015")]))
        complete, evidence, terminal = ANTHROPIC_WIRE.response_evidence(json.dumps(body).encode(), generation())
        self.assertEqual(ANTHROPIC_WIRE.completion_outcome(complete, evidence), "usable")
        self.assertEqual(evidence["usage"], dict(input_tokens=3, output_tokens=2))
        self.assertTrue(ANTHROPIC_WIRE.context_edit_applied(terminal))
        error = dict(type="error", error=dict(type="invalid_request_error", message="prompt is too long: 9 > 8"))
        complete, evidence, _ = ANTHROPIC_WIRE.response_evidence(json.dumps(error).encode(), generation())
        self.assertEqual(ANTHROPIC_WIRE.completion_outcome(complete, evidence), "failed")
        self.assertEqual(evidence["response_error"]["classification"], "context_window_exceeded")
        for raw in (b"[]", b'{"type": "message"}', b"not json"):
            with self.subTest(raw=raw), self.assertRaises(ValueError):
                ANTHROPIC_WIRE.response_evidence(raw, generation())
        streamed_edit = message_start() + message_end("end_turn", context_management={"applied_edits": [{"type": "x"}]})
        self.assertTrue(ANTHROPIC_WIRE.context_edit_applied(ANTHROPIC_WIRE.response_evidence(streamed_edit, generation())[2]))


class AnthropicEmittedToolCallsTest(unittest.TestCase):
    """``emitted_tool_calls``: the model's tool calls in emitted (content index) order, read-only."""

    def test_stream_in_content_index_order(self):
        raw = b"".join([
            message_start(), block_start(0, {"type": "text", "text": ""}), delta(0, {"type": "text_delta", "text": "hi"}),
            block_stop(0),
            # Blocks may be interleaved on the stream; the content index is the emitted order.
            block_start(2, {"type": "tool_use", "id": "toolu_b", "name": PREFIX + "robot_move", "input": {}}),
            block_start(1, {"type": "tool_use", "id": "toolu_a", "name": PREFIX + "robot_move", "input": {}}),
            delta(2, {"type": "input_json_delta", "partial_json": '{"x": '}),
            delta(1, {"type": "input_json_delta", "partial_json": '{"x": 1}'}),
            delta(2, {"type": "input_json_delta", "partial_json": '2}'}), block_stop(1), block_stop(2),
            block_start(3, {"type": "server_tool_use", "id": "srv_1", "name": "web_search", "input": {}}), block_stop(3),
            block_start(4, {"type": "tool_use", "id": "toolu_c", "name": "Bash", "input": {"command": "ls"}}),
            block_stop(4), block_start(5, {"type": "tool_use", "id": "toolu_d", "name": "x", "input": {}}),
            delta(5, {"type": "input_json_delta", "partial_json": "{not json"}), block_stop(5)]) + message_end("tool_use")
        self.assertEqual(ANTHROPIC_WIRE.emitted_tool_calls(raw), [
            ("toolu_a", PREFIX + "robot_move", {"x": 1}), ("toolu_b", PREFIX + "robot_move", {"x": 2}),
            ("toolu_c", "Bash", {"command": "ls"}), ("toolu_d", "x", None)])
        self.assertEqual(ANTHROPIC_WIRE.emitted_tool_calls(message_start() + message_end("end_turn")), [])

    def test_json_body(self):
        body = dict(id="msg_2", type="message", role="assistant", stop_reason="tool_use", content=[
            dict(type="text", text="hi"), dict(type="tool_use", id="toolu_1", name="x", input={"a": 1}),
            dict(type="tool_use", id="toolu_2", name="y", input={})])
        self.assertEqual(ANTHROPIC_WIRE.emitted_tool_calls(json.dumps(body).encode()),
                         [("toolu_1", "x", {"a": 1}), ("toolu_2", "y", {})])
        with self.assertRaises(ValueError):
            ANTHROPIC_WIRE.emitted_tool_calls(b"not json")


class AnthropicErrorTest(unittest.TestCase):
    def test_failed_response_error_table(self):
        table = {
            ("overloaded_error", "fixture-private"): ("server_overloaded", True),
            ("rate_limit_error", "fixture-private"): ("rate_limit", True),
            ("api_error", "fixture-private"): ("retryable", True),
            ("timeout_error", "fixture-private"): ("retryable", True),
            ("authentication_error", "fixture-private"): ("authentication_error", False),
            ("permission_error", "fixture-private"): ("permission_denied", False),
            ("not_found_error", "model: claude-x"): ("configuration_error", False),
            ("billing_error", "fixture-private"): ("quota_exceeded", False),
            ("request_too_large", "Request exceeds the maximum size"): ("image_rejected", True),
            ("invalid_request_error", "messages: roles must alternate"): ("configuration_error", False),
            ("invalid_request_error", "prompt is too long: 250000 tokens > 200000 maximum"): ("context_window_exceeded", False),
            ("invalid_request_error", "Your credit balance is too low to access the API"): ("quota_exceeded", False),
            ("invalid_request_error", "messages.0.content.1.image.source.base64.data: image exceeds 5 MB maximum"):
                ("image_rejected", True),
            ("invalid_request_error", "Too many images in request"): ("image_rejected", True),
            ("future_error", "fixture-private"): ("retryable", True),
        }
        for (kind, message), (classification, retryable) in table.items():
            with self.subTest(kind=kind, message=message):
                result = ANTHROPIC_WIRE.failed_response_error({"type": "error", "error": {"type": kind, "message": message}})
                self.assertEqual((result["classification"], result["retryable"]), (classification, retryable))
                self.assertEqual(result["type"], kind)
                self.assertNotIn(message, json.dumps(result))
                self.assertEqual(set(result), {"code", "type", "code_state", "classification", "retryable", "retry_after_s"})
        for body in ({}, {"error": "text"}, {"error": {"message": "x"}}, None):
            with self.subTest(body=body):
                self.assertEqual(ANTHROPIC_WIRE.failed_response_error(body)["classification"], "stream")

    def test_http_failure_classification(self):
        def failure(status, error_type=None, message="x", headers=None):
            body = {"type": "error", "error": {"type": error_type, "message": message}} if error_type else None
            return ANTHROPIC_WIRE.classify_http_failure(status, headers or {}, body, policy=POLICY)
        cases = [
            (529, "overloaded_error", "x", True), (500, "api_error", "x", True), (503, None, "x", True),
            (429, "rate_limit_error", "x", True), (401, "authentication_error", "x", False),
            (403, "permission_error", "x", False), (404, "not_found_error", "x", False),
            (400, "invalid_request_error", "messages: bad", False),
            (400, "invalid_request_error", "prompt is too long", False),
            (400, "invalid_request_error", "Your credit balance is too low", False),
            (400, "invalid_request_error", "image exceeds 5 MB maximum", True),
            (413, "request_too_large", "x", True), (413, None, "x", True),
            (503, "not_found_error", "model: x", False),
        ]
        for status, error_type, message, retryable in cases:
            with self.subTest(status=status, error_type=error_type, message=message):
                result = failure(status, error_type, message)
                self.assertIsInstance(result, POLICY.InfrastructureFailure)
                self.assertEqual(result.retryable, retryable)
                self.assertEqual(result.status, status)
        self.assertEqual(failure(429, "rate_limit_error", headers={"Retry-After": "3"}).retry_after_s, 3.0)
        self.assertTrue(ANTHROPIC_WIRE.image_rejected(dict(classification="image_rejected")))
        self.assertTrue(ANTHROPIC_WIRE.image_rejected(dict(http_status=413, classification="stream")))
        self.assertFalse(ANTHROPIC_WIRE.image_rejected(dict(http_status=400, classification="configuration_error")))

    def assert_neutral(self, data):
        text = data.decode()
        self.assertIsNone(re.search(r"\d", text), text)
        for word in FORBIDDEN_WORDS:
            self.assertNotIn(word, text.lower())

    def test_claude_code_denials_relay_errors_and_failed_payload(self):
        for kind in ("thread_cancelled", "transport_blocked", "budget", "client_retry_runaway"):
            with self.subTest(kind=kind):
                reply = ANTHROPIC_WIRE.denial(kind)
                self.assertEqual(reply.status, 403)
                self.assertEqual(dict(reply.headers), {"Content-Type": "application/json", "x-should-retry": "false"})
                body = json.loads(reply.body)
                self.assertEqual(body["type"], "error")
                self.assertEqual(body["error"]["type"], "permission_error")
                self.assertTrue(body["error"]["message"].startswith("ni_rua_"))
                self.assert_neutral(reply.body)
        self.assertEqual(len({ANTHROPIC_WIRE.denial(kind).body for kind in am.DENIAL_MESSAGES}), 4)
        with self.assertRaises(KeyError):
            ANTHROPIC_WIRE.denial("unknown")
        evidence = dict(http_status=503, classification="retryable", code="abc123", detail="logical-call-9")
        streamed = ANTHROPIC_WIRE.relay_error(streaming=True, evidence=evidence, manual_retry_allowed=True)
        self.assertEqual(streamed.body, b'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error",'
                                        b'"message":"ni_rua_model_request_failed"}}\n\n')
        plain = ANTHROPIC_WIRE.relay_error(streaming=False, evidence=evidence, manual_retry_allowed=False)
        self.assertEqual((plain.status, dict(plain.headers)), (403, dict(am.NO_RETRY_HEADERS)))
        self.assertEqual(json.loads(plain.body)["error"], {"type": "permission_error", "message": "ni_rua_model_request_failed"})
        failed = ANTHROPIC_WIRE.failed_payload(dict(status="failed", error={"type": "overloaded_error", "message": "private 42"}),
                                     streaming=True)
        self.assertEqual(failed, b'event: error\ndata: {"type":"error","error":{"type":"invalid_request_error",'
                                 b'"message":"ni_rua_model_response_unusable"}}\n\n')
        self.assertIsNone(ANTHROPIC_WIRE.failed_payload(dict(status="incomplete"), streaming=False))
        for data in (streamed.body, plain.body, failed):
            self.assert_neutral(data)
        found = ANTHROPIC_WIRE.not_found()
        self.assertEqual(found.status, 404)
        self.assertEqual(json.loads(found.body)["error"]["type"], "not_found_error")
        self.assert_neutral(found.body)


class AnthropicUsageTest(unittest.TestCase):
    def test_normalize_usage(self):
        full = dict(input_tokens=100, cache_read_input_tokens=40, cache_creation_input_tokens=10, output_tokens=7)
        self.assertEqual(ANTHROPIC_WIRE.normalize_usage(full), dict(input_tokens=150, cached_input_tokens=40,
                         cache_write_input_tokens=10, output_tokens=7, reasoning_output_tokens=None, total_tokens=157))
        self.assertEqual(ANTHROPIC_WIRE.normalize_usage(dict(input_tokens=5, output_tokens=2)),
                         dict(input_tokens=5, cached_input_tokens=None, cache_write_input_tokens=None, output_tokens=2,
                              reasoning_output_tokens=None, total_tokens=7))
        self.assertEqual(ANTHROPIC_WIRE.normalize_usage(dict(input_tokens=5, cache_read_input_tokens=-1, output_tokens=2)),
                         dict(input_tokens=None, cached_input_tokens=None, cache_write_input_tokens=None,
                              output_tokens=2, reasoning_output_tokens=None, total_tokens=None))
        self.assertEqual(ANTHROPIC_WIRE.normalize_usage(dict(cache_read_input_tokens=3, output_tokens="2"))["input_tokens"], None)
        for usage in (None, {}, "x"):
            with self.subTest(usage=usage):
                self.assertEqual(ANTHROPIC_WIRE.normalize_usage(usage), {key: None for key in TOKEN_FIELDS})
        self.assertEqual(tuple(ANTHROPIC_WIRE.normalize_usage(full)), TOKEN_FIELDS)

    def test_normalize_usage_maps_thinking_tokens_and_keeps_unknown_honest(self):
        # 真实 claude_stack_bowls_s0_l0 实测归档中 6 次 model_request_finished 的完整 usage 样本
        real_samples = [
            {"cache_creation": {"ephemeral_1h_input_tokens": 77501, "ephemeral_5m_input_tokens": 31007},
             "cache_creation_input_tokens": 108508, "cache_read_input_tokens": 0, "inference_geo": "global",
             "input_tokens": 2, "output_tokens": 588, "output_tokens_details": {"thinking_tokens": 341},
             "service_tier": "standard", "speed": "standard"},
            {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 10641},
             "cache_creation_input_tokens": 10641, "cache_read_input_tokens": 105105, "inference_geo": "global",
             "input_tokens": 285, "output_tokens": 1560, "output_tokens_details": {"thinking_tokens": 1214},
             "service_tier": "standard", "speed": "standard"},
            {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 2673},
             "cache_creation_input_tokens": 2673, "cache_read_input_tokens": 115746, "inference_geo": "global",
             "input_tokens": 285, "output_tokens": 3437, "output_tokens_details": {"thinking_tokens": 3053},
             "service_tier": "standard", "speed": "standard"},
            {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 6653},
             "cache_creation_input_tokens": 6653, "cache_read_input_tokens": 118419, "inference_geo": "global",
             "input_tokens": 2, "output_tokens": 6376, "output_tokens_details": {"thinking_tokens": 5105},
             "service_tier": "standard", "speed": "standard"},
            {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 20126},
             "cache_creation_input_tokens": 20126, "cache_read_input_tokens": 118419, "inference_geo": "global",
             "input_tokens": 2, "output_tokens": 2448, "output_tokens_details": {"thinking_tokens": 1868},
             "service_tier": "standard", "speed": "standard"},
            {"cache_creation": {"ephemeral_1h_input_tokens": 0, "ephemeral_5m_input_tokens": 30557},
             "cache_creation_input_tokens": 30557, "cache_read_input_tokens": 118419, "inference_geo": "global",
             "input_tokens": 285, "output_tokens": 1435, "output_tokens_details": {"thinking_tokens": 840},
             "service_tier": "standard", "speed": "standard"},
        ]
        accumulator = RelayUsageTotal()
        record = None
        for sample in real_samples:
            record = accumulator.add(ANTHROPIC_WIRE.normalize_usage(sample), thread_id="c57b6754")
        self.assertEqual(record["calls"], 6)
        self.assertEqual(record["unknown_calls"], 0)
        self.assertEqual(record["total"], dict(
            input_tokens=756127,
            cached_input_tokens=576108,
            cache_write_input_tokens=179158,
            output_tokens=15844,
            reasoning_output_tokens=12421,
            total_tokens=771971,
        ))
        accounting = account_usage(record, model="claude-fable-5-2", provider="anthropic",
                                   source="relay.anthropic_messages.usage.sum")
        self.assertEqual(accounting["issues"], [])
        self.assertEqual(accounting["tokens"]["reasoning_output_tokens"], 12421)

        # SSE 流式 message_delta 中的 output_tokens_details.thinking_tokens 经 response_evidence 合并后可正确归一化
        raw = (message_start()
               + sse("message_delta", {"type": "message_delta",
                                       "delta": {"stop_reason": "end_turn", "stop_sequence": None},
                                       "usage": {"output_tokens": 7, "output_tokens_details": {"thinking_tokens": 3}}})
               + sse("message_stop", {"type": "message_stop"}))
        _, evidence, _ = ANTHROPIC_WIRE.response_evidence(raw, generation())
        self.assertEqual(ANTHROPIC_WIRE.normalize_usage(evidence["usage"])["reasoning_output_tokens"], 3)

        # 兼容 reasoning_tokens；未知键、null、负数、非法类型必须诚实保持 None 并计入 unknown_calls
        base = dict(input_tokens=10, cache_read_input_tokens=0, cache_creation_input_tokens=0, output_tokens=7)
        self.assertEqual(
            ANTHROPIC_WIRE.normalize_usage({**base, "output_tokens_details": {"reasoning_tokens": 4}})["reasoning_output_tokens"],
            4,
        )
        for bad_details in (None, {}, {"other_tokens": 5}, {"thinking_tokens": None},
                            {"thinking_tokens": -1}, {"thinking_tokens": "3"}, {"thinking_tokens": True}, "bad"):
            with self.subTest(bad_details=bad_details):
                normalized = ANTHROPIC_WIRE.normalize_usage({**base, "output_tokens_details": bad_details})
                self.assertIsNone(normalized["reasoning_output_tokens"])
                self.assertEqual(RelayUsageTotal().add(normalized)["unknown_calls"], 1)


def anthropic_observation_text(observation_id, cameras):
    return dict(type="text", text=json.dumps(dict(bundle=dict(
        observation_id=observation_id, cameras=[dict(camera_id=camera) for camera in cameras]))))


def anthropic_image(seed):
    return dict(type="image", source=dict(type="base64", media_type="image/png", data=seed * 8))


class AnthropicImageTest(unittest.TestCase):
    def request(self):
        return dict(messages=[
            dict(role="user", content=[dict(type="text", text="task"), anthropic_image("a")]),
            dict(role="assistant", content=[dict(type="tool_use", id="t1", name="mcp__ni_rua__robot_observe", input={})]),
            dict(role="user", content=[
                dict(type="tool_result", tool_use_id="t1", content=[anthropic_observation_text("obs_1", ["head", "wrist"]),
                                                                     anthropic_image("b"), dict(type="text", text=PRIVATE_IMAGE_BLOCK),
                                                                     anthropic_image("c")]),
                dict(type="tool_result", tool_use_id="t2", content=[anthropic_observation_text("obs_2", ["head"]), anthropic_image("b")])]),
            dict(role="user", content="string content"),
            dict(role="system", content="env"),
            dict(role="user", content=[anthropic_image("d"), dict(type="image", source=dict(type="url", url="https://x/y.png"))]),
            dict(role="assistant", content=[dict(type="text", text="ok")]),
        ])

    def test_locations_labels_and_boundaries(self):
        request = self.request()
        self.assertEqual(len(ANTHROPIC_WIRE.image_locations(request)), 5)
        labels = [label for _, _, label in ANTHROPIC_WIRE.image_locations(request, include_labels=True)]
        self.assertEqual(labels, [(None, None), ("obs_1", "head"), ("obs_1", "wrist"), ("obs_2", "head"), (None, None)])
        found, boundaries = ANTHROPIC_WIRE.step_image_boundaries(request)
        self.assertEqual([parts[index] for parts, index in found],
                         [parts[index] for parts, index in ANTHROPIC_WIRE.image_locations(request)])
        self.assertEqual(boundaries, [1, 4, 5])
        for value in (None, [], dict(messages="x"), dict(messages=[None, dict(content=3)])):
            self.assertEqual(ANTHROPIC_WIRE.image_locations(value), [])
            self.assertEqual(ANTHROPIC_WIRE.step_image_boundaries(value), ([], []))

    def test_trim_uses_the_generic_base_logic_with_text_placeholders(self):
        request = self.request()
        original = json.dumps(request)
        trimmed, count = ANTHROPIC_WIRE.trim_image_range(request, 1, 3)
        self.assertEqual(count, 2)
        self.assertEqual(json.dumps(request), original)
        nested = trimmed["messages"][2]["content"][0]["content"]
        self.assertEqual(nested[1], dict(type="text", text=ANTHROPIC_WIRE.omitted_image_text("obs_1", "head")))
        self.assertEqual(nested[3], dict(type="text", text=ANTHROPIC_WIRE.omitted_image_text("obs_1", "wrist")))
        self.assertEqual(len(ANTHROPIC_WIRE.image_locations(trimmed)), 3)
        self.assertEqual(ANTHROPIC_WIRE.placeholder_part("x"), dict(type="text", text="x"))
        self.assertEqual(ANTHROPIC_WIRE.image_part_size(anthropic_image("a")), 8)
        self.assertEqual(ANTHROPIC_WIRE.image_part_size(dict(type="image", source=dict(type="url"))), 0)
        body = json.dumps(request).encode()
        kept = ANTHROPIC_WIRE.trim_oldest_images(body, keep_latest=2)
        self.assertEqual(len(ANTHROPIC_WIRE.image_locations(json.loads(kept))), 2)
        self.assertIs(ANTHROPIC_WIRE.trim_oldest_images(body, keep_latest=5), body)
        # Same algorithm and same thresholds as the Responses wire (user decision, design rev 3.1).
        self.assertIs(type(ANTHROPIC_WIRE).trim_image_range, WireAdapter.trim_image_range)
        self.assertIs(type(ANTHROPIC_WIRE).trim_oldest_images, WireAdapter.trim_oldest_images)

    def test_exact_reuse_uses_the_shared_helper(self):
        request = self.request()
        result, stats = ANTHROPIC_WIRE.reuse_exact_images(request)
        self.assertEqual(stats, dict(images=5, replaced=1, saved_url_chars=8))
        replaced = result["messages"][2]["content"][1]["content"][1]
        self.assertEqual(replaced["type"], "text")
        self.assertIn("identical pixels to image 2", replaced["text"])
        self.assertEqual(request["messages"][2]["content"][1]["content"][1]["type"], "image")
        self.assertEqual(len(list(ANTHROPIC_WIRE.image_slots(request))), 6)  # the url image is located but never deduplicated


# ======================================================================== egress field table
class Presentation:
    """A real ActorPresentation over a temporary Store."""

    def __init__(self, test):
        directory = tempfile.TemporaryDirectory()
        test.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "run", dict(run_id="egress", episode_id="episode",
                                                              env_instance_id="environment", env_generation=1))
        test.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.unknown = []

    def project(self, request, wire=ANTHROPIC_WIRE):
        return wire.project_request(request, self.presentation, registered_tools=REGISTERED, tool_name_prefix=PREFIX,
                                    on_unknown_part=self.unknown.append)


def planted_request():
    """One body touching every row of the design §4.4 Anthropic table."""
    schema = {"type": "object", "properties": {"value_sha256": {"type": "string", "description": "digest " + DIGEST},
                                               "camera_id": {"type": "string", "description": "camera " + PLANTED}}}
    return {
        "model": "claude-opus-5-5", "max_tokens": 128000, "stream": True, "betas": ["context-management-2025-06-27"],
        "metadata": {"user_id": json.dumps({"device_id": DIGEST, "account_uuid": "", "session_id": "s"})},
        "thinking": {"type": "adaptive", "display": "updates"},
        "context_management": {"edits": [{"type": "clear_thinking_20251015", "keep": "all"}]},
        "output_config": {"effort": "medium", "format": {"type": "json_schema", "schema": deepcopy(schema)}},
        "tool_choice": {"type": "auto"},
        "system": [{"type": "text", "text": "Identity " + PLANTED},
                   {"type": "text", "text": "System " + PLANTED, "cache_control": {"type": "ephemeral"}}],
        "messages": [
            {"role": "user", "content": "plain string " + PLANTED},
            {"role": "system", "content": "environment block " + PLANTED},
            {"role": "user", "content": [
                {"type": "text", "text": "user text " + PLANTED, "cache_control": {"type": "ephemeral"},
                 "citations": [{"type": "char_location", "cited_text": "cited " + DIGEST, "document_index": 0,
                                "start_char_index": 0, "end_char_index": 3}]},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
                {"type": "document", "title": "title " + PLANTED, "context": "context " + PLANTED,
                 "source": {"type": "text", "media_type": "text/plain", "data": "document " + PLANTED},
                 "citations": {"enabled": True}},
                {"type": "document", "source": {"type": "base64", "media_type": "application/pdf", "data": PDF}},
                {"type": "document", "title": "chunks", "source": {"type": "content", "content": [
                    {"type": "text", "text": "chunk " + PLANTED}]}},
                {"type": "search_result", "title": "result " + PLANTED, "source": "source " + PLANTED,
                 "content": [{"type": "text", "text": "snippet " + PLANTED}], "citations": {"enabled": True}}]},
            {"role": "system", "content": [{"type": "text", "text": "steer framing " + PLANTED}]},
            {"role": "assistant", "content": [
                {"type": "thinking", "thinking": "thinking " + PLANTED, "signature": "sig-" + DIGEST},
                {"type": "redacted_thinking", "data": "redacted-" + DIGEST},
                {"type": "text", "text": "assistant " + PLANTED},
                {"type": "tool_use", "id": "toolu_01" + DIGEST[:20], "name": PREFIX + "robot_observe",
                 "input": {"note": PLANTED, "observation_id": LONG_OBSERVATION, "digest_sha256": DIGEST},
                 "cache_control": {"type": "ephemeral"}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_01" + DIGEST[:20], "cache_control": {"type": "ephemeral"},
                 "content": [
                     {"type": "text", "text": json.dumps({"observation_id": LONG_OBSERVATION, "frame_sha256": DIGEST,
                                                          "note": PLANTED})},
                     {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}},
                     {"type": "text", "text": PRIVATE_IMAGE_BLOCK, "cache_control": {"type": "ephemeral"}},
                     {"type": "tool_reference", "tool_name": PREFIX + "robot_move"},
                     {"type": "text", "text": "caption " + PLANTED}]},
                {"type": "tool_result", "tool_use_id": "toolu_02", "content": "failed " + PLANTED, "is_error": True},
                {"type": "text", "text": "after results " + PLANTED}]},
            {"role": "user", "content": [{"type": "future_block", "text": "future " + PLANTED, "data": "keep-" + DIGEST,
                                          "content": [{"type": "image", "source": {
                                              "type": "base64", "media_type": "image/png", "data": PNG}}]}]},
        ],
        "tools": [
            {"name": "Bash", "description": "Run " + PLANTED, "input_schema": deepcopy(schema)},
            {"name": PREFIX + "robot_observe", "description": "Observe " + PLANTED, "input_schema": deepcopy(schema),
             "cache_control": {"type": "ephemeral"}, "eager_input_streaming": True},
            {"type": "custom", "name": "custom_tool", "description": "Custom " + PLANTED,
             "input_schema": {"type": "object"}, "strict": True},
        ],
    }


def paths_differing(before, after, path=""):
    """Leaf paths (dotted) where two JSON values differ; a key present on one side only is a difference."""
    if isinstance(before, dict) and isinstance(after, dict):
        found = []
        for key in sorted(set(before) | set(after)):
            child = f"{path}.{key}" if path else str(key)
            if key not in before or key not in after:
                found.append(child)
            else:
                found.extend(paths_differing(before[key], after[key], child))
        return found
    if isinstance(before, list) and isinstance(after, list) and len(before) == len(after):
        found = []
        for index, (left, right) in enumerate(zip(before, after)):
            found.extend(paths_differing(left, right, f"{path}.{index}" if path else str(index)))
        return found
    return [] if before == after else [path]


# Locations the walker may rewrite (design §4.4); every other byte of a body must stay identical.
SCRUBBABLE_PATHS = re.compile(r"""^(?:
      system\.\d+\.text | system
    | messages\.\d+\.content
    | messages\.\d+\.content\.\d+\.(?:text|title|context|input(?:\..+)?|content|source)
    | messages\.\d+\.content\.\d+\.source\.(?:data|content(?:\..+)?)
    | messages\.\d+\.content\.\d+\.content\.\d+\.(?:text|content(?:\..+)?)
    | tools\.\d+\.(?:description|input_schema(?:\..+)?)
    | output_config\.format\.schema(?:\..+)?
)$""", re.VERBOSE)


class EgressFieldTableTest(unittest.TestCase):
    def setUp(self):
        self.session = Presentation(self)

    def test_every_scrubbable_field_is_scrubbed(self):
        request = planted_request()
        output = self.session.project(request)
        m = output["messages"]
        tool_result = m[5]["content"][0]["content"]
        scrubbed = [output["system"][0]["text"], output["system"][1]["text"], m[0]["content"], m[1]["content"],
                    m[2]["content"][0]["text"], m[2]["content"][2]["title"], m[2]["content"][2]["context"],
                    m[2]["content"][2]["source"]["data"], m[2]["content"][4]["source"]["content"][0]["text"],
                    m[2]["content"][5]["title"], m[2]["content"][5]["source"], m[2]["content"][5]["content"][0]["text"],
                    m[3]["content"][0]["text"], m[4]["content"][2]["text"], json.dumps(m[4]["content"][3]["input"]),
                    tool_result[0]["text"], tool_result[4]["text"], m[5]["content"][1]["content"],
                    m[5]["content"][2]["text"], m[6]["content"][0]["text"],
                    output["tools"][0]["description"], output["tools"][1]["description"],
                    output["tools"][2]["description"], json.dumps(output["tools"][0]["input_schema"]),
                    json.dumps(output["output_config"]["format"]["schema"])]
        for index, value in enumerate(scrubbed):
            with self.subTest(index=index, value=value[:80]):
                self.assertNotIn(DIGEST, value)
                self.assertNotIn(ARTIFACT, value)
                self.assertNotIn(LONG_OBSERVATION, value)
        self.assertIn("artifact:ref_", output["system"][1]["text"])
        self.assertIn("obs_1", m[4]["content"][3]["input"]["observation_id"])
        self.assertNotIn("digest_sha256", m[4]["content"][3]["input"])
        self.assertNotIn("frame_sha256", tool_result[0]["text"])
        # Claude Code image-source block: exact constant, block and its other keys kept, indices unchanged
        self.assertEqual(tool_result[2], {"type": "text", "text": "[Image]", "cache_control": {"type": "ephemeral"}})
        self.assertEqual(len(tool_result), len(request["messages"][5]["content"][0]["content"]))
        # roles and positions of role:"system" messages unchanged
        self.assertEqual([message["role"] for message in m], [message["role"] for message in request["messages"]])
        self.assertEqual(self.session.unknown, ["future_block"])
        self.assertNotIn(DIGEST, m[6]["content"][0]["text"])
        self.assertEqual(m[6]["content"][0]["data"], "keep-" + DIGEST)

    def test_every_protected_field_is_byte_identical(self):
        request = planted_request()
        output = self.session.project(request)
        m, r = output["messages"], request["messages"]
        protected = [
            (output["model"], request["model"]), (output["max_tokens"], request["max_tokens"]),
            (output["stream"], request["stream"]), (output["betas"], request["betas"]),
            (output["metadata"], request["metadata"]), (output["thinking"], request["thinking"]),
            (output["context_management"], request["context_management"]),
            (output["tool_choice"], request["tool_choice"]),
            (output["output_config"]["effort"], request["output_config"]["effort"]),
            (output["output_config"]["format"]["type"], request["output_config"]["format"]["type"]),
            (output["system"][1]["cache_control"], request["system"][1]["cache_control"]),
            (m[2]["content"][0]["cache_control"], r[2]["content"][0]["cache_control"]),
            (m[2]["content"][0]["citations"], r[2]["content"][0]["citations"]),
            (m[2]["content"][1], r[2]["content"][1]),                        # image block, bytes and media type
            (m[2]["content"][2]["citations"], r[2]["content"][2]["citations"]),
            (m[2]["content"][2]["source"]["media_type"], r[2]["content"][2]["source"]["media_type"]),
            (m[2]["content"][3], r[2]["content"][3]),                        # base64 PDF document
            (m[2]["content"][5]["citations"], r[2]["content"][5]["citations"]),
            (m[4]["content"][0], r[4]["content"][0]),                        # thinking + signature
            (m[4]["content"][1], r[4]["content"][1]),                        # redacted_thinking
            (m[4]["content"][3]["id"], r[4]["content"][3]["id"]), (m[4]["content"][3]["name"], r[4]["content"][3]["name"]),
            (m[4]["content"][3]["cache_control"], r[4]["content"][3]["cache_control"]),
            (m[5]["content"][0]["tool_use_id"], r[5]["content"][0]["tool_use_id"]),
            (m[5]["content"][0]["cache_control"], r[5]["content"][0]["cache_control"]),
            (m[5]["content"][0]["content"][1], r[5]["content"][0]["content"][1]),   # image inside tool_result
            (m[5]["content"][0]["content"][3], r[5]["content"][0]["content"][3]),   # tool_reference
            (m[5]["content"][1]["is_error"], r[5]["content"][1]["is_error"]),
            (m[5]["content"][1]["tool_use_id"], r[5]["content"][1]["tool_use_id"]),
            (m[6]["content"][0]["content"][0], r[6]["content"][0]["content"][0]),   # image nested in unknown block
        ]
        for tool, original in zip(output["tools"], request["tools"]):
            protected += [(tool.get(key), original.get(key)) for key in ("name", "type", "cache_control",
                                                                          "eager_input_streaming", "strict")]
        for index, (value, original) in enumerate(protected):
            with self.subTest(index=index):
                self.assertEqual(json.dumps(value, sort_keys=False), json.dumps(original, sort_keys=False))
        for path in paths_differing(request, output):
            with self.subTest(path=path):
                self.assertRegex(path, SCRUBBABLE_PATHS)
        self.assertEqual(request, planted_request(), "the input body must not be mutated")

    def test_projection_is_deterministic_and_idempotent(self):
        first = self.session.project(planted_request())
        second = self.session.project(planted_request())
        self.assertEqual(json.dumps(first), json.dumps(second))
        self.assertEqual(json.dumps(self.session.project(first)), json.dumps(first))
        other = ActorPresentation(self.session.store)
        replay = ANTHROPIC_WIRE.project_request(planted_request(), other, registered_tools=REGISTERED, tool_name_prefix=PREFIX)
        self.assertEqual(json.dumps(replay), json.dumps(first))

    def test_each_rejection_fires(self):
        def mutate(change):
            request = planted_request()
            change(request)
            return request
        user = lambda request: request["messages"][2]["content"]
        nested = lambda request: request["messages"][5]["content"][0]["content"]
        cases = [
            (lambda r: r["tools"].append({"type": "web_search_20260209", "name": "web_search"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append({"type": "code_execution_20260521", "name": "code_execution"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append({"type": "computer_toolset_20260801"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append({"type": "tool_search_tool_regex_20251119", "name": "s"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append({"type": "memory_20250818", "name": "memory"}), "sandbox_remote_model_tool_disabled"),
            (lambda r: r["tools"].append("Bash"), "sandbox_remote_model_tool_disabled"),
            (lambda r: r.update(mcp_servers=[{"type": "url", "url": "https://x", "name": "x"}]), "sandbox_remote_model_tool_disabled"),
            (lambda r: r.update(container="container_1"), "sandbox_remote_model_tool_disabled"),
            (lambda r: user(r).append({"type": "image", "source": {"type": "url", "url": "https://x/y.png"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: user(r).append({"type": "image", "source": {"type": "file", "file_id": "file_1"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: nested(r).append({"type": "image", "source": {"type": "url", "url": "https://x"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: user(r).append({"type": "document", "source": {"type": "url", "url": "https://x/a.pdf"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: user(r).append({"type": "document", "source": {"type": "file", "file_id": "file_1"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: r["messages"][6]["content"][0]["content"].append({"type": "image", "source": {"type": "url", "url": "x"}}), "sandbox_remote_media_url_disabled"),
            (lambda r: nested(r).append({"type": "tool_reference", "tool_name": PREFIX + "unregistered"}), "native_discovered_tool_not_registered"),
            (lambda r: nested(r).append({"type": "tool_reference", "tool_name": "robot_move"}), "native_discovered_tool_not_registered"),
        ]
        for block_type in ("server_tool_use", "web_search_tool_result", "web_fetch_tool_result", "code_execution_tool_result",
                           "mcp_tool_use", "mcp_tool_result", "container_upload", "bash_code_execution_tool_result",
                           "future_server_tool_result"):
            cases.append((lambda r, t=block_type: r["messages"][4]["content"].append({"type": t, "id": "srvtoolu_1"}),
                          "sandbox_remote_model_tool_disabled"))
        for index, (change, error) in enumerate(cases):
            with self.subTest(index=index, error=error), self.assertRaisesRegex(ValueError, error):
                self.session.project(mutate(change))
        with self.assertRaisesRegex(ValueError, "model_request_not_object"):
            self.session.project([])

    def test_image_source_rewrite_is_profile_and_position_specific(self):
        request = dict(messages=[dict(role="user", content=[
            dict(type="tool_result", tool_use_id="t", content=[dict(type="text", text=PRIVATE_IMAGE_BLOCK),
                                                               dict(type="text", text=PRIVATE_IMAGE_BLOCK + " extra"),
                                                               dict(type="text", text="[Image: source: a\nb]")]),
            dict(type="text", text=PRIVATE_IMAGE_BLOCK)])])
        output = self.session.project(request)
        nested = output["messages"][0]["content"][0]["content"]
        self.assertEqual(nested[0]["text"], "[Image]")
        self.assertNotEqual(nested[1]["text"], "[Image]")
        self.assertNotEqual(nested[2]["text"], "[Image]")
        # P1-2 review item 3: the marker is rewritten outside a tool_result too (claude_code profile).
        self.assertEqual(output["messages"][0]["content"][1]["text"], "[Image]")
        other = present_messages_request(request, self.session.presentation, registered_tools=REGISTERED,
                                         tool_name_prefix=PREFIX, client="opencode")
        self.assertNotEqual(other["messages"][0]["content"][0]["content"][0]["text"], "[Image]")
        self.assertEqual(self.session.project(output), output)

    def test_absent_and_minimal_fields(self):
        self.assertEqual(self.session.project({}), {})
        minimal = {"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "hi"}], "system": "plain"}
        self.assertEqual(self.session.project(minimal), minimal)
        stringy = {"system": "System " + PLANTED, "messages": [{"role": "user"}, "odd", {"role": "user", "content": 5}]}
        output = self.session.project(stringy)
        self.assertNotIn(DIGEST, output["system"])
        self.assertEqual(output["messages"], stringy["messages"])


class RealCapturedBodiesTest(unittest.TestCase):
    """Real request bodies of Claude Code 2.1.321 and opencode 1.19.5 (spike captures, sanitised)."""

    @classmethod
    def setUpClass(cls):
        cls.cases = json.loads(FIXTURE.read_text())["cases"]

    def setUp(self):
        self.session = Presentation(self)

    @staticmethod
    def plant(body):
        """Plant digests/artifacts/long ids into the prose locations Claude Code fills with host data."""
        body = deepcopy(body)
        if isinstance(body.get("system"), list) and body["system"]:
            body["system"][-1]["text"] += "\n" + PLANTED
        for message in body.get("messages", []):
            content = message.get("content")
            if isinstance(content, str):
                message["content"] = content + " " + PLANTED
                continue
            for block in content:
                if block.get("type") == "text" and not am.IMAGE_SOURCE_BLOCK.fullmatch(block["text"]):
                    block["text"] += " " + PLANTED
                elif block.get("type") == "tool_result":
                    if isinstance(block.get("content"), str):
                        block["content"] += " " + PLANTED
                    else:
                        for part in block.get("content", []):
                            if part.get("type") == "text" and not am.IMAGE_SOURCE_BLOCK.fullmatch(part["text"]):
                                part["text"] += " " + PLANTED
        return body

    def test_fixture_covers_the_observed_shapes(self):
        self.assertEqual(set(self.cases), {"claude_code_tool_results", "claude_code_thinking_parallel",
                                           "claude_code_post_compaction", "claude_code_count_tokens",
                                           "opencode_anthropic_session"})
        text = json.dumps(self.cases)
        self.assertIn('"role": "system"', text)
        self.assertIn('"type": "thinking"', text)
        self.assertIn('"eager_input_streaming": true', text)
        self.assertGreaterEqual(len(am.IMAGE_SOURCE_BLOCK.findall(text)), 4)
        for name, case in self.cases.items():
            with self.subTest(name=name):
                route = ANTHROPIC_WIRE.route(case["method"], case["path"])
                self.assertEqual(route.kind, "auxiliary" if "count_tokens" in name else "generation")

    def test_planted_values_never_survive_and_protocol_bytes_stay_identical(self):
        for name, case in self.cases.items():
            with self.subTest(name=name):
                original = case["body"]
                planted = self.plant(original)
                output = self.session.project(planted)
                serialized = json.dumps(output)
                for value in (DIGEST, ARTIFACT, LONG_OBSERVATION):
                    self.assertNotIn(value, serialized)
                for path in paths_differing(planted, output):
                    self.assertRegex(path, SCRUBBABLE_PATHS)
                self.assertNotIn("[Image: source:", serialized)
                self.assertEqual(serialized.count('"text": "[Image]"'),
                                 len(am.IMAGE_SOURCE_BLOCK.findall(json.dumps(original))))
                for key in ("model", "max_tokens", "stream", "metadata", "thinking", "context_management",
                            "tool_choice"):
                    self.assertEqual(output.get(key), original.get(key), key)
                self.assertEqual([t["name"] for t in output.get("tools", [])], [t["name"] for t in original.get("tools", [])])
                self.assertEqual(self._protected(output), self._protected(original))
                self.assertEqual(json.dumps(self.session.project(output)), serialized)
                self.assertEqual(self.session.unknown, [])

    def test_unplanted_bodies_change_only_scrubbable_prose(self):
        for name, case in self.cases.items():
            with self.subTest(name=name):
                output = self.session.project(case["body"])
                for path in paths_differing(case["body"], output):
                    self.assertRegex(path, SCRUBBABLE_PATHS)

    @staticmethod
    def _protected(body):
        """Every protocol-bearing block field of a body, in order."""
        found = []
        for message in body.get("messages", []):
            found.append(message.get("role"))
            content = message.get("content")
            for block in content if isinstance(content, list) else []:
                kind = block.get("type")
                found.append((kind, block.get("cache_control"), block.get("id"), block.get("name"),
                              block.get("tool_use_id"), block.get("is_error")))
                if kind in ("thinking", "redacted_thinking", "image"):
                    found.append(json.dumps(block))
                for part in block.get("content", []) if isinstance(block.get("content"), list) else []:
                    found.append((part.get("type"), part.get("cache_control")))
                    if part.get("type") == "image":
                        found.append(json.dumps(part))
        for tool in body.get("tools", []):
            found.append((tool.get("name"), tool.get("cache_control"), tool.get("eager_input_streaming")))
        for block in body.get("system", []) if isinstance(body.get("system"), list) else []:
            found.append(block.get("cache_control"))
        return found

# ==== tests/test_wire_openai_responses.py ====

WIRE = OpenAIResponsesWire()
GENERATION = WIRE.route("POST", "/responses")
COMPACT = WIRE.route("POST", "/responses/compact")


def sse_block(value):
    return b"data: " + json.dumps(value).encode()


def stream(*events):
    return b"".join(sse_block(event) + b"\n\n" for event in events)


def response(status, **fields):
    return dict(id="resp-1", model="served-model", status=status, output=[], **fields)


def image(seed):
    return dict(type="input_image", image_url="data:image/png;base64," + seed * 8)


def observation_text(observation_id, cameras):
    return dict(type="input_text", text=json.dumps(dict(bundle=dict(
        observation_id=observation_id, cameras=[dict(camera_id=camera) for camera in cameras]))))


class RouteTest(unittest.TestCase):
    def test_exact_codex_routes(self):
        self.assertEqual(GENERATION, Route("generation", "/responses", "/responses", False))
        self.assertEqual(COMPACT, Route("generation", "/responses/compact", "/responses/compact", True))
        for method, path in (("GET", "/responses"), ("POST", "/v1/responses"), ("POST", "/responses/"),
                             ("POST", "/responses?stream=1"), ("POST", "/responses/compact/"), ("POST", "/")):
            with self.subTest(method=method, path=path):
                self.assertIsNone(WIRE.route(method, path))

    def test_endpoint_lookup_falls_back_to_plain_generation(self):
        self.assertTrue(WIRE.route_for_endpoint("/responses/compact").compaction)
        self.assertEqual(WIRE.route_for_endpoint("/v1/other"), Route("generation", "/v1/other", "/v1/other"))
        self.assertFalse(WIRE.route_for_endpoint(None).compaction)

    def test_wire_constants_are_unchanged(self):
        self.assertEqual(WIRE.name, "openai_responses")
        self.assertEqual(dict(WIRE.auth_header_names), AUTH_HEADERS)
        self.assertEqual(AUTH_HEADERS, {"authorization": "Authorization", "chatgpt-account-id": "ChatGPT-Account-Id"})
        self.assertEqual(CLIENT_AUTH_HEADERS, {"authorization", "chatgpt-account-id", "cookie", "x-api-key", "api-key",
                                               "openai-organization", "openai-project"})
        self.assertEqual((WIRE.body_limit_bytes, WIRE.trim_target_bytes), (4_500_000, 2_400_000))
        self.assertTrue(WIRE.supports_service_tier)
        self.assertIs(wire_from_name("openai_responses").__class__, OpenAIResponsesWire)
        with self.assertRaisesRegex(ValueError, "unknown_model_wire:anthropic_messages"):
            wire_from_name("anthropic_messages")

    def test_request_fields(self):
        self.assertEqual(WIRE.thread_key({"thread-id": "t1"}), "t1")
        self.assertEqual(WIRE.thread_key({}), "")
        self.assertTrue(WIRE.is_streaming({"stream": True}))
        self.assertFalse(WIRE.is_streaming({"stream": "true"}))
        self.assertFalse(WIRE.is_streaming({}))
        self.assertEqual(WIRE.requested_model({"model": "m"}), "m")
        self.assertIsNone(WIRE.requested_model({}))
        self.assertEqual(WIRE.decode_body({}, b"raw"), b"raw")


class TerminalTest(unittest.TestCase):
    def test_terminal_types(self):
        for event_type in ("response.completed", "response.failed", "response.incomplete", "error"):
            with self.subTest(event_type=event_type):
                self.assertEqual(WIRE.terminal_sse_type(sse_block(dict(type=event_type))), event_type)
                self.assertTrue(model_proxy._terminal_sse_block(sse_block(dict(type=event_type))))
        for block in (sse_block(dict(type="response.created")), sse_block(dict(type="response.output_text.delta")),
                      b"data: [DONE]", b": comment", b"event: response.completed", b"data: {not json",
                      b'data: ["response.completed"]', b"data: \xff\xfe", b""):
            with self.subTest(block=block):
                self.assertIsNone(WIRE.terminal_sse_type(block))
                self.assertIsNone(model_proxy._terminal_sse_type(block))
                self.assertFalse(model_proxy._terminal_sse_block(block))

    def test_terminal_data_lines_join_and_event_lines_are_ignored(self):
        block = b'event: response.completed\ndata:{"type":\ndata:  "response.completed"}'
        self.assertEqual(WIRE.terminal_sse_type(block), "response.completed")
        self.assertEqual(WIRE.terminal_sse_type(b'event: x\r\ndata: {"type":"error"}'), "error")

    def test_keepalive_and_connection_reuse(self):
        self.assertEqual(WIRE.keepalive_frame(), b'event: ni_rua.keepalive\ndata: {"type":"ni_rua.keepalive"}\n\n')
        self.assertTrue(WIRE.terminal_allows_reuse("response.completed"))
        for terminal in ("response.failed", "response.incomplete", "error", None):
            self.assertFalse(WIRE.terminal_allows_reuse(terminal))
        self.assertTrue(WIRE.is_event_stream(b"  \ndata: x"))
        self.assertTrue(WIRE.is_event_stream(bytearray(b": ping")))
        self.assertFalse(WIRE.is_event_stream(b'{"status":"completed"}'))


class EvidenceTest(unittest.TestCase):
    def outcome(self, raw, route=GENERATION):
        complete, evidence, terminal = WIRE.response_evidence(raw, route)
        return WIRE.completion_outcome(complete, evidence), evidence, terminal

    def test_completed_stream_is_usable_with_served_model_and_raw_usage(self):
        usage = dict(input_tokens=5, output_tokens=2)
        outcome, evidence, terminal = self.outcome(stream(
            dict(type="response.output_text.delta", delta="x"),
            dict(type="response.completed", response=response("completed", usage=usage))))
        self.assertEqual(outcome, "usable")
        self.assertEqual(evidence, dict(response_id="resp-1", served_model="served-model", response_status="completed",
                                        usage=usage, usage_known=True))
        self.assertEqual(terminal["status"], "completed")

    def test_failed_and_incomplete_terminals(self):
        error = dict(code="context_length_exceeded", message="private provider text")
        outcome, evidence, _ = self.outcome(stream(dict(type="response.failed", response=response("failed", error=error))))
        self.assertEqual(outcome, "failed")
        self.assertEqual(evidence["response_error"]["classification"], "context_window_exceeded")
        self.assertNotIn("private", json.dumps(evidence))
        outcome, evidence, _ = self.outcome(stream(dict(type="response.incomplete", response=response("incomplete"))))
        self.assertEqual(outcome, "incomplete")
        self.assertNotIn("response_error", evidence)

    def test_error_event_is_a_failed_terminal_without_usage(self):
        outcome, evidence, terminal = self.outcome(stream(dict(type="error", code="server_error", message="private")))
        self.assertEqual(outcome, "failed")
        self.assertEqual(terminal, dict(status="failed", output=[], error=dict(type="error", code="server_error",
                                                                                message="private")))
        self.assertEqual(evidence["usage"], None)
        self.assertFalse(evidence["usage_known"])
        self.assertEqual(evidence["response_error"]["code"], "server_error")

    def test_missing_or_conflicting_terminal_raises(self):
        with self.assertRaises(EOFError):
            WIRE.response_evidence(stream(dict(type="response.created")), GENERATION)
        with self.assertRaisesRegex(ValueError, "conflict"):
            WIRE.response_evidence(stream(dict(type="response.completed", response=response("failed"))), GENERATION)
        with self.assertRaisesRegex(ValueError, "invalid response structure"):
            WIRE.response_evidence(stream(dict(type="response.completed", response=dict(status="completed"))), GENERATION)
        with self.assertRaises(ValueError):
            WIRE.response_evidence(b"data: {broken\n\n", GENERATION)

    def test_json_bodies_and_compaction_route(self):
        compacted = json.dumps(dict(object="response.compaction", output=[], usage={})).encode()
        outcome, evidence, _ = self.outcome(compacted, COMPACT)
        self.assertEqual(outcome, "usable")
        self.assertFalse(evidence["usage_known"])
        self.assertEqual(self.outcome(compacted, GENERATION)[0], "not_completed")
        self.assertEqual(self.outcome(json.dumps(response("in_progress")).encode())[0], "not_completed")
        self.assertEqual(self.outcome(json.dumps(response("completed")).encode())[0], "usable")
        self.assertEqual(ModelRequestProxy._response_evidence(compacted, "/responses/compact"),
                         WIRE.response_evidence(compacted, COMPACT))
        self.assertEqual(ModelRequestProxy._response_evidence(compacted, "/responses"),
                         WIRE.response_evidence(compacted, GENERATION))

    def test_failed_payload_drops_output_and_keeps_order(self):
        terminal = dict(id="r", status="failed", output=[dict(type="function_call")], error=dict(code="x"))
        self.assertEqual(WIRE.failed_payload(terminal, streaming=True),
                         b'event: response.failed\ndata: {"type": "response.failed", "response": '
                         b'{"id": "r", "status": "failed", "output": [], "error": {"code": "x"}}}\n\n')
        self.assertEqual(WIRE.failed_payload(dict(terminal, status="incomplete"), streaming=False),
                         b'{"id": "r", "status": "incomplete", "output": [], "error": {"code": "x"}}')
        self.assertEqual(terminal["output"], [dict(type="function_call")])


class ErrorTest(unittest.TestCase):
    def classify(self, code=None, type_=None, message=None):
        error = {key: value for key, value in (("code", code), ("type", type_), ("message", message)) if value is not None}
        return WIRE.failed_response_error(dict(error=error))

    def test_classification_table(self):
        table = {
            "context_length_exceeded": ("context_window_exceeded", False),
            "insufficient_quota": ("quota_exceeded", False), "credit_balance_exhausted": ("quota_exceeded", False),
            "invalid_api_key": ("authentication_error", False), "model_not_found": ("configuration_error", False),
            "unsupported_parameter": ("configuration_error", False), "invalid_prompt": ("invalid_prompt", False),
            "image_compression_failed": ("retryable", True), "server_is_overloaded": ("server_overloaded", True),
            "content_filter": ("refusal", False), "flex_unavailable": ("flex_unavailable", False),
            "rate_limit_exceeded": ("rate_limit", True), "slow_down": ("rate_limit", True),
            "server_error": ("retryable", True), "cyber_policy": ("cyber_policy", False),
        }
        for code, (classification, retryable) in table.items():
            with self.subTest(code=code):
                result = self.classify(code, message="secret provider message")
                self.assertEqual((result["classification"], result["retryable"]), (classification, retryable))
                self.assertEqual(result, ModelRequestProxy._failed_response_error(
                    dict(error=dict(code=code, message="secret provider message"))))
                self.assertNotIn("secret", json.dumps(result))
        self.assertEqual(self.classify(type_="invalid_request_error")["classification"], "configuration_error")
        self.assertEqual(self.classify(code="unknown", type_="overloaded")["classification"], "server_overloaded")

    def test_sanitising_and_retry_after(self):
        self.assertEqual(self.classify("rate_limit_exceeded", message="Please try again in 250ms.")["retry_after_s"], .25)
        self.assertEqual(self.classify("rate_limit_exceeded", message="try again in 3 seconds")["retry_after_s"], 3.0)
        self.assertIsNone(self.classify("rate_limit_exceeded", message="try again in 0s")["retry_after_s"])
        self.assertIsNone(self.classify("server_error", message="try again in 3s")["retry_after_s"])
        unsafe = self.classify("bad code with spaces")
        self.assertEqual((unsafe["code"], unsafe["code_state"]), (None, "redacted_or_invalid"))
        self.assertEqual(self.classify(type_="server_error")["code_state"], "unknown")
        for body in ({}, dict(error="text"), dict(error=dict(code=5)), dict(error=dict(resets_at="soon"))):
            with self.subTest(body=body):
                self.assertEqual(WIRE.failed_response_error(body)["classification"], "stream")
                self.assertTrue(WIRE.failed_response_error(body)["retryable"])

    def test_http_failure_classification_uses_pinned_policy(self):
        policy = upstream().joint
        def failure(status, body, headers=None):
            return WIRE.classify_http_failure(status, headers or {}, body, policy=policy)
        self.assertTrue(failure(503, dict(error=dict(code="server_error"))).retryable)
        self.assertFalse(failure(503, dict(error=dict(code="model_not_found", type="new_api_error"))).retryable)
        self.assertTrue(failure(400, dict(error=dict(code="image_compression_failed"))).retryable)
        self.assertFalse(failure(401, None).retryable)
        limited = failure(429, dict(error=dict(code="rate_limit_exceeded")), {"Retry-After": ".5"})
        self.assertTrue(limited.retryable)
        self.assertEqual(limited.retry_after_s, .5)
        self.assertIsInstance(failure(500, [1, 2]), policy.InfrastructureFailure)
        self.assertIsInstance(failure(400, dict(error=dict(code="content_filter"))), policy.CapabilityFailure)
        self.assertFalse(WIRE.is_streaming([1, 2]))
        self.assertIsNone(WIRE.requested_model([1, 2]))
        self.assertTrue(WIRE.image_rejected(dict(code="image_compression_failed")))
        self.assertFalse(WIRE.image_rejected(dict(classification="retryable")))

    def test_relay_errors_and_denials(self):
        evidence = dict(http_status=503, classification="retryable")
        streamed = WIRE.relay_error(streaming=True, evidence=evidence, manual_retry_allowed=True)
        self.assertEqual(streamed.body, (
            b'event: error\ndata: {"type": "error", "error": {"type": "upstream_error", "code": "model_request_failed", '
            b'"message": "Model request failed; provider-error retry policy exhausted or not applicable.", '
            b'"evidence": {"http_status": 503, "classification": "retryable"}, "manual_retry_allowed": true}}\n\n'))
        plain = WIRE.relay_error(streaming=False, evidence=evidence, manual_retry_allowed=False)
        self.assertEqual((plain.status, dict(plain.headers)), (503, {"Content-Type": "application/json"}))
        self.assertEqual(json.loads(plain.body), dict(error=dict(
            code="model_request_failed",
            message="Model request failed; provider-error retry policy exhausted or not applicable.",
            evidence=evidence, manual_retry_allowed=False)))
        self.assertEqual(WIRE.relay_error(streaming=False, evidence=None, manual_retry_allowed=False).status, 502)
        self.assertEqual(WIRE.denial("thread_cancelled"), WireReply(409, "model thread explicitly cancelled"))
        self.assertEqual(WIRE.denial("transport_blocked"), WireReply(409, "model transport blocked; manual review required"))
        self.assertEqual(WIRE.denial("budget"), WireReply(429, "episode model completion budget unavailable"))
        for kind in ("thread_cancelled", "transport_blocked", "budget"):
            self.assertIsNone(WIRE.denial(kind).body)
        with self.assertRaises(KeyError):
            WIRE.denial("unknown")
        self.assertEqual(WIRE.not_found(), WireReply(404))


class UsageTest(unittest.TestCase):
    def test_normalized_usage_matches_responses_accounting(self):
        samples = [
            dict(input_tokens=11, input_tokens_details=dict(cached_tokens=4, cache_write_tokens=2), output_tokens=3,
                 output_tokens_details=dict(reasoning_tokens=1), total_tokens=14),
            dict(input_tokens=5, output_tokens=2, cache_write_input_tokens=1),
            dict(input_tokens=-1, output_tokens="3", total_tokens=None),
            {}, None, "not usage",
        ]
        for usage in samples:
            with self.subTest(usage=usage):
                normalized = WIRE.normalize_usage(usage)
                self.assertEqual(tuple(normalized), TOKEN_FIELDS)
                accounted = account_responses_usage(usage)["tokens"]
                self.assertEqual(normalized, {key: accounted[key] for key in TOKEN_FIELDS})
        self.assertEqual(WIRE.normalize_usage(dict(input_tokens=5, output_tokens=2)),
                         dict(input_tokens=5, cached_input_tokens=None, cache_write_input_tokens=None,
                              output_tokens=2, reasoning_output_tokens=None, total_tokens=None))
        self.assertFalse(WIRE.context_edit_applied(dict(status="completed")))


class ImageTest(unittest.TestCase):
    def request(self):
        return dict(input=[
            dict(type="message", role="user", content=[dict(type="input_text", text="task"), image("a")]),
            dict(type="function_call", call_id="c1", name="robot_observe", arguments="{}"),
            dict(type="function_call_output", call_id="c1", output=[observation_text("obs_1", ["head", "wrist"]),
                                                                    image("b"), image("c")]),
            dict(type="function_call", call_id="c2", name="robot_observe", arguments="{}"),
            dict(type="custom_tool_call_output", call_id="c2", output=[observation_text("obs_2", ["head"]), image("b")]),
            dict(type="function_call_output", call_id="c3", output=[image("d"),
                 dict(type="input_image", image_url="https://example.org/x.png")]),
            dict(type="message", role="assistant", content=[dict(type="output_text", text="ok")]),
            dict(type="reasoning", content=[image("e")]),
        ])

    def test_locations_labels_and_step_boundaries(self):
        request = self.request()
        locations = WIRE.image_locations(request)
        self.assertEqual(len(locations), 5)
        labels = [label for _, _, label in WIRE.image_locations(request, include_labels=True)]
        self.assertEqual(labels, [(None, None), ("obs_1", "head"), ("obs_1", "wrist"), ("obs_2", "head"), (None, None)])
        found, boundaries = WIRE.step_image_boundaries(request)
        self.assertEqual(len(found), 5)
        self.assertEqual(boundaries, [1, 3, 5])
        self.assertEqual(ModelRequestProxy._step_image_boundaries(request), (found, boundaries))
        self.assertEqual(ModelRequestProxy._inline_image_locations(request, include_labels=True),
                         WIRE.image_locations(request, include_labels=True))
        self.assertEqual(ModelRequestProxy._part_observation_labels(request["input"][2]["output"]),
                         [(None, None), ("obs_1", "head"), ("obs_1", "wrist")])
        for value in (None, [], dict(input="text")):
            self.assertEqual(WIRE.image_locations(value), [])
            self.assertEqual(WIRE.step_image_boundaries(value), ([], []))

    def test_trim_placeholders_and_delegates(self):
        request = self.request()
        original = json.dumps(request)
        trimmed, count = WIRE.trim_image_range(request, 1, 4)
        self.assertEqual(count, 3)
        self.assertEqual(json.dumps(request), original)
        self.assertEqual(trimmed["input"][2]["output"][1], dict(type="input_text", text=(
            "[Earlier historical observation image omitted (observation_id=obs_1, camera=head) to stay within "
            "upstream gateway payload limit; call history_view_observation(observation_id='obs_1') if you need "
            "to inspect this frame.]")))
        self.assertEqual(len(WIRE.image_locations(trimmed)), 2)
        self.assertEqual(ModelRequestProxy._trim_image_range(request, 1, 4), (trimmed, 3))
        self.assertEqual(WIRE.trim_image_range(request, 3, 3), (request, 0))
        self.assertEqual(WIRE.trim_image_range(request, 0, 99)[1], 5)
        self.assertEqual(ModelRequestProxy._trim_prefix_images(request, 2), WIRE.trim_prefix_images(request, 2))
        self.assertEqual(WIRE.omitted_image_text("obs_9", None),
                         "[Earlier historical observation image omitted (observation_id=obs_9) to stay within upstream "
                         "gateway payload limit; call history_view_observation(observation_id='obs_9') if you need to "
                         "inspect this frame.]")
        self.assertEqual(WIRE.omitted_image_text(None, "head"),
                         "[Earlier historical observation image omitted to stay within upstream gateway payload limit.]")
        self.assertEqual(WIRE.placeholder_part("x"), dict(type="input_text", text="x"))
        self.assertEqual(WIRE.image_part_size(image("a")), len(image("a")["image_url"]))

    def test_trim_oldest_images_body(self):
        body = json.dumps(self.request(), ensure_ascii=False).encode()
        out = WIRE.trim_oldest_images(body, keep_latest=2)
        self.assertEqual(len(WIRE.image_locations(json.loads(out))), 2)
        self.assertNotIn(b", ", out.split(b'"text"')[0])
        self.assertEqual(ModelRequestProxy._trim_oldest_images(body, keep_latest=2), out)
        self.assertIs(WIRE.trim_oldest_images(body, keep_latest=5), body)
        self.assertEqual(WIRE.trim_oldest_images(b"not json", keep_latest=0), b"not json")

    def test_reuse_default_locator_and_custom_locator(self):
        request = self.request()
        default = reuse_exact_images(request)
        self.assertEqual(default[1], dict(images=5, replaced=1, saved_url_chars=len(image("b")["image_url"])))
        self.assertEqual(reuse_exact_images(request, locate=responses_image_slots), default)
        self.assertEqual(WIRE.reuse_exact_images(request), default)
        self.assertIn("identical pixels to image 2", default[0]["input"][4]["output"][1]["text"])

        def blocks(value):
            for message in value["messages"]:
                for index, part in enumerate(message["content"]):
                    if part.get("type") == "image":
                        source = part["source"]
                        yield message["content"], index, source["data"] if source.get("type") == "base64" else None
        messages = dict(messages=[dict(role="user", content=[
            dict(type="image", source=dict(type="url", url="https://example.org/x.png")),
            dict(type="image", source=dict(type="base64", media_type="image/png", data="AAAA")),
            dict(type="image", source=dict(type="base64", media_type="image/png", data="AAAA"))])])
        result, stats = reuse_exact_images(messages, locate=blocks, text_part=lambda text: dict(type="text", text=text))
        self.assertEqual(stats, dict(images=2, replaced=1, saved_url_chars=4))
        self.assertEqual(result["messages"][0]["content"][2]["type"], "text")
        self.assertIn("identical pixels to image 2", result["messages"][0]["content"][2]["text"])
        self.assertEqual(messages["messages"][0]["content"][2]["type"], "image")


class EmittedToolCallsTest(unittest.TestCase):
    """``emitted_tool_calls``: ``function_call`` items in ``output_index`` order, read-only."""

    @staticmethod
    def call(call_id, arguments, name="ni_rua_robot_move"):
        return dict(type="function_call", id="fc_" + call_id, call_id=call_id, name=name, arguments=arguments)

    def test_stream_reads_the_done_items_in_output_index_order(self):
        done = lambda index, item: {"type": "response.output_item.done", "output_index": index, "item": item}
        raw = stream(done(0, dict(type="reasoning", id="rs_1")), done(2, self.call("call_b", '{"x": 2}')),
                     done(1, self.call("call_a", '{"x": 1}')), done(3, self.call("call_c", "{broken")),
                     {"type": "response.completed", "response": response("completed")})
        self.assertEqual(WIRE.emitted_tool_calls(raw), [("call_a", "ni_rua_robot_move", {"x": 1}),
                                                        ("call_b", "ni_rua_robot_move", {"x": 2}),
                                                        ("call_c", "ni_rua_robot_move", None)])

    def test_stream_without_done_items_and_a_json_body_read_the_terminal_output(self):
        output = [dict(type="message", content=[]), self.call("call_a", '{"x": 1}'), self.call("call_b", "{}", "bash")]
        completed = dict(response("completed"), output=output)
        expected = [("call_a", "ni_rua_robot_move", {"x": 1}), ("call_b", "bash", {})]
        self.assertEqual(WIRE.emitted_tool_calls(stream({"type": "response.completed", "response": completed})), expected)
        self.assertEqual(WIRE.emitted_tool_calls(json.dumps(completed).encode()), expected)
        self.assertEqual(WIRE.emitted_tool_calls(json.dumps(response("completed")).encode()), [])
        with self.assertRaises(ValueError):
            WIRE.emitted_tool_calls(b"not json")


class BaseAdapterTest(unittest.TestCase):
    def test_base_defaults_and_unimplemented_methods(self):
        base = WireAdapter()
        self.assertEqual(base.emitted_tool_calls(b"anything"), [])
        self.assertEqual(base.decode_body({}, b"x"), b"x")
        self.assertEqual(base.thread_key({"thread-id": "t"}), "")
        self.assertIsNone(base.keepalive_frame())
        self.assertFalse(base.terminal_allows_reuse("response.completed"))
        self.assertFalse(base.image_rejected(dict(code="image_compression_failed")))
        self.assertFalse(base.context_edit_applied({}))
        self.assertEqual(base.not_found(), WireReply(404))
        calls = [lambda: base.route("POST", "/"), lambda: base.is_streaming({}), lambda: base.requested_model({}),
                 lambda: base.conversation_key({}), lambda: base.history_fingerprints({}),
                 lambda: base.project_request({}, None, registered_tools=frozenset(), tool_name_prefix=""),
                 lambda: base.image_locations({}), lambda: base.step_image_boundaries({}),
                 lambda: base.image_part_size({}), lambda: base.placeholder_part("x"), lambda: base.reuse_exact_images({}),
                 lambda: base.terminal_sse_type(b""), lambda: base.response_evidence(b"", None),
                 lambda: base.completion_outcome(True, {}), lambda: base.failed_response_error({}),
                 lambda: base.classify_http_failure(500, {}, None, policy=None),
                 lambda: base.failed_payload({}, streaming=True),
                 lambda: base.relay_error(streaming=True, evidence=None, manual_retry_allowed=False),
                 lambda: base.denial("budget"), lambda: base.normalize_usage({}),
                 lambda: base.build_evidence_request(model="m", base_instructions="", developer_instructions="",
                                                     input_items=[], output_schema={}, effort="low"),
                 lambda: base.evidence_answer({})]
        for index, call in enumerate(calls):
            with self.subTest(index=index), self.assertRaises(NotImplementedError):
                call()


class ProxyWiringTest(unittest.TestCase):
    def setUp(self):
        self.upstream = golden.Upstream()
        self.addCleanup(self.upstream.close)

    def test_constructor_keywords(self):
        proxy = ModelRequestProxy("http://127.0.0.1:1")
        self.assertIsInstance(proxy.wire, OpenAIResponsesWire)
        self.assertIsNone(proxy.default_thread_id)
        self.assertIsNone(proxy.on_context_discontinuity)
        self.assertIsNone(proxy.on_tool_calls)      # Codex: nothing is read from the answer, nothing called
        with self.assertRaisesRegex(ValueError, "model_proxy_tool_calls_requires_callable"):
            ModelRequestProxy("http://127.0.0.1:1", on_tool_calls="callback")
        custom = OpenAIResponsesWire()
        self.assertIs(ModelRequestProxy("http://127.0.0.1:1", wire=custom).wire, custom)
        with self.assertRaisesRegex(ValueError, "model_proxy_wire_requires_adapter"):
            ModelRequestProxy("http://127.0.0.1:1", wire="openai_responses")
        for value in ("", 5, "a\nb", "x" * 257):
            with self.subTest(value=value), self.assertRaisesRegex(ValueError, "model_proxy_invalid_default_thread_id"):
                ModelRequestProxy("http://127.0.0.1:1", default_thread_id=value)
        with self.assertRaisesRegex(ValueError, "model_proxy_context_discontinuity_requires_callable"):
            ModelRequestProxy("http://127.0.0.1:1", on_context_discontinuity="callback")
        self.assertEqual(ModelRequestProxy("http://127.0.0.1:1", default_thread_id="t",
                                           on_context_discontinuity=print).default_thread_id, "t")

    def test_auth_header_names_follow_the_wire(self):
        self.assertEqual(upstream_auth_headers_copy({"authorization": "Bearer a"}), {"Authorization": "Bearer a"})
        with self.assertRaisesRegex(ValueError, "invalid_upstream_auth_headers"):
            upstream_auth_headers_copy({"x-api-key": "k"})
        self.assertEqual(upstream_auth_headers_copy({"X-Api-Key": "k"}, {"x-api-key": "x-api-key"}), {"x-api-key": "k"})

        class KeyWire(OpenAIResponsesWire):
            auth_header_names = {"x-api-key": "x-api-key"}
        with ModelRequestProxy(self.upstream.url, wire=KeyWire(), upstream_auth_headers={"x-api-key": "host"}) as proxy:
            golden.call(proxy)
        headers = dict((key.lower(), value) for key, value in self.upstream.requests[0]["headers"])
        self.assertEqual(headers["x-api-key"], "host")
        self.assertNotIn("authorization", headers)
        self.assertNotIn("chatgpt-account-id", headers)

    def test_proxy_uses_wire_replies_for_denials(self):
        class StrictWire(OpenAIResponsesWire):
            def denial(self, kind):
                return WireReply(400, headers={"Content-Type": "application/json", "x-should-retry": "false"},
                                 body=json.dumps(dict(error=dict(type="invalid_request_error", denial=kind))).encode())
        with ModelRequestProxy(self.upstream.url, wire=StrictWire(), max_requests=1) as proxy:
            self.assertEqual(golden.call(proxy)["status"], 200)
            denied = golden.call(proxy, admitted=False)
        self.assertEqual(denied["status"], 400)
        self.assertIn(["x-should-retry", "false"], denied["headers"])
        self.assertEqual(json.loads(denied["body"]["text"]), dict(error=dict(type="invalid_request_error", denial="budget")))

    def test_default_thread_id_attributes_requests_without_a_registered_key(self):
        with ModelRequestProxy(self.upstream.url, default_thread_id="session-thread") as proxy:
            proxy.register_thread("session-thread", "actor", "req-session")
            proxy.register_thread("helper-thread", "helper", "req-helper")
            golden.call(proxy, headers=golden.CLIENT_HEADERS[:3])
            golden.call(proxy, extra_headers=(("thread-id", "unregistered"),))
            golden.call(proxy, extra_headers=(("thread-id", "helper-thread"),))
            proxy.cancel_thread("helper-thread")
            proxy.unregister_thread("helper-thread")
            cancelled = golden.call(proxy, extra_headers=(("thread-id", "helper-thread"),), admitted=False)
            started = [event for event in proxy.events if event["kind"] == "model_call_started"]
        self.assertEqual([(event["native_thread_id"], event["source"], event["caller_request_id"]) for event in started],
                         [("session-thread", "actor", "req-session"), ("session-thread", "actor", "req-session"),
                          ("helper-thread", "helper", "req-helper")])
        self.assertEqual(cancelled["status"], 409)

    def test_codex_default_keeps_header_attribution(self):
        with ModelRequestProxy(self.upstream.url) as proxy:
            golden.call(proxy, headers=golden.CLIENT_HEADERS[:3])
            started = next(event for event in proxy.events if event["kind"] == "model_call_started")
        self.assertEqual((started["native_thread_id"], started["source"], started["caller_request_id"]), ("", "actor", ""))

    def test_worker_payload_names_the_wire(self):
        payloads = []
        result = dict(status=200, headers={"Content-Type": "text/event-stream"},
                      body=base64.b64encode(golden.completed(1)[2]).decode(), terminal_seen=True)

        class FakeWorker:
            pid, returncode = 0, 0
            def __init__(self, *args, **kwargs):
                self.stdin = self
                self.stdout = iter([json.dumps(dict(result=result, incoming_bytes=1, first_upstream_byte_s=0.0)) + "\n"])
            def write(self, text):
                payloads.append(json.loads(text))
            def close(self): pass
            def poll(self): return 0
            def wait(self, timeout=None): return 0
            def communicate(self, *args, **kwargs): return "", ""
        with ModelRequestProxy(self.upstream.url) as proxy, \
                mock.patch("ni_rua.cortex.relay.server.subprocess.Popen", FakeWorker):
            self.assertEqual(golden.call(proxy)["status"], 200)
        self.assertEqual(payloads[0]["wire"], "openai_responses")
        self.assertEqual(set(payloads[0]), {"url", "body", "headers", "wire"})

    def run_worker(self, payload):
        output = io.StringIO()
        with mock.patch.object(sys, "stdin", io.StringIO(json.dumps(payload))), redirect_stdout(output):
            model_proxy._http_worker()
        return [json.loads(line) for line in output.getvalue().splitlines()]

    def test_http_worker_reads_wire_name_with_responses_default(self):
        self.upstream.script = [golden.completed(1), golden.completed(2)]
        payload = dict(url=self.upstream.url + "/responses", body=base64.b64encode(b"{}").decode(),
                       headers={"Content-Type": "application/json"})
        for extra in ({}, dict(wire="openai_responses")):
            with self.subTest(extra=extra):
                lines = self.run_worker(dict(payload, **extra))
                result = lines[-1]["result"]
                self.assertTrue(result["terminal_seen"])
                self.assertEqual(base64.b64decode(result["body"]), golden.completed(len(self.upstream.requests))[2])
                self.assertIn("model_response_terminal_received", [line.get("kind") for line in lines])
        with self.assertRaisesRegex(ValueError, "unknown_model_wire"):
            self.run_worker(dict(payload, wire="unknown"))
        self.assertEqual(len(self.upstream.requests), 2)

    def test_worker_script_imports_adapters_from_its_own_checkout(self):
        import subprocess
        from pathlib import Path
        script = Path(model_proxy.__file__).resolve()
        probe = ("import runpy, sys; sys.argv=['server.py']; "
                 f"namespace = runpy.run_path({str(script)!r}); "
                 "import ni_rua.cortex.session.wire as wire; print(wire.__file__)")
        completed = subprocess.run([sys.executable, "-c", probe], capture_output=True, text=True, timeout=60,
                                   cwd="/")
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(Path(completed.stdout.strip()).resolve(), script.parent.parent / "session" / "wire" / "__init__.py")


class EgressProjectionCacheAndSubagentHeaderTests(unittest.TestCase):
    """验证跨 Wire EgressProjectionCache（字符串/工具/条目 LRU）、确定性修复、OpenCode system_key=None 与 Claude 子链头识别。"""

    def setUp(self):
        self.session = Presentation(self)

    def test_is_subagent_request_and_opencode_continuity_system_key(self):
        claude_wire = AnthropicMessagesWire(client="claude_code")
        oc_anthropic = AnthropicMessagesWire(client="opencode")
        oc_responses = OpenAIResponsesWire(client="opencode")
        codex_wire = OpenAIResponsesWire(client="codex")

        self.assertFalse(WireAdapter().is_subagent_request({"x-claude-code-agent-id": "a1"}))
        self.assertFalse(codex_wire.is_subagent_request({"x-claude-code-agent-id": "a1"}))
        self.assertFalse(oc_responses.is_subagent_request({"x-claude-code-agent-id": "a1"}))
        self.assertFalse(oc_anthropic.is_subagent_request({"x-claude-code-agent-id": "a1"}))
        self.assertFalse(claude_wire.is_subagent_request({}))
        self.assertFalse(claude_wire.is_subagent_request({"X-Claude-Code-Agent-Id": "   "}))
        self.assertTrue(claude_wire.is_subagent_request({"X-Claude-Code-Agent-Id": "agent-fork-1"}))
        self.assertTrue(claude_wire.is_subagent_request({"x-claude-code-agent-id": "agent-2"}))

        anthropic_body = {
            "system": "SYS",
            "messages": [{"role": "user", "content": "hi"}],
            "tools": [{"name": "t", "input_schema": {"type": "object"}}],
        }
        responses_body = {
            "instructions": "SYS",
            "input": [{"role": "user", "content": "hi"}],
            "tools": [{"type": "function", "name": "t", "parameters": {"type": "object"}}],
        }
        self.assertIsNotNone(claude_wire.continuity_inputs(anthropic_body)["system_key"])
        self.assertIsNone(oc_anthropic.continuity_inputs(anthropic_body)["system_key"])
        self.assertIsNone(oc_responses.continuity_inputs(responses_body)["system_key"])

    def test_anthropic_string_and_tools_lru_hits_mixed_image_tool_results_and_guards_media(self):
        from ni_rua.cortex.session.egress_cache import EgressProjectionCache

        cache = EgressProjectionCache()
        heavy_json = json.dumps({
            "observation_id": LONG_OBSERVATION,
            "sha256": DIGEST,
            "points": list(range(200)),
            "note": "keep frame-1234abcd1234 stable",
        })
        large_image = "A" * (80 * 1024)
        schema = {
            "type": "object",
            "properties": {
                "target": {"type": "string", "description": "Use frame-1234abcd1234"},
            },
        }
        turn1 = {
            "model": "claude-opus-5-5",
            "max_tokens": 1024,
            "system": [{"type": "text", "text": "System prompt"}],
            "output_config": {"format": {"type": "json_schema", "schema": deepcopy(schema)}},
            "output_format": {"type": "json_schema", "schema": deepcopy(schema)},
            "tools": [{"name": PREFIX + "robot_observe", "description": "Observe scene", "input_schema": deepcopy(schema)}],
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "tool_result",
                            "tool_use_id": "toolu_1",
                            "cache_control": {"type": "ephemeral"},
                            "content": [
                                {"type": "text", "text": heavy_json},
                                {"type": "image", "source": {"type": "base64", "media_type": "image/webp", "data": large_image}},
                                {"type": "text", "text": PRIVATE_IMAGE_BLOCK},
                            ],
                        }
                    ],
                }
            ],
        }
        out1 = ANTHROPIC_WIRE.project_request(
            turn1,
            self.session.presentation,
            registered_tools=REGISTERED,
            tool_name_prefix=PREFIX,
            cache=cache,
        )
        self.assertEqual(cache.bucket_size("tools"), 1)
        # 大图 base64 绝不驻留字符串缓存
        self.assertLess(cache.text_bytes, len(large_image))

        # 后续轮次注册未知前缀裸词并移动 cache_control 滑窗，验证 schema 零漂移 + 图文混排文本 100% 命中字符串 LRU
        self.session.presentation.present({"frame_id": "frame-1234abcd1234"})
        turn2 = deepcopy(turn1)
        del turn2["messages"][0]["content"][0]["cache_control"]
        turn2["messages"].append({"role": "user", "content": [{"type": "text", "text": "next turn"}]})

        with mock.patch.object(
            self.session.presentation,
            "present_egress_text",
            wraps=self.session.presentation.present_egress_text,
        ) as spy_text, mock.patch.object(
            self.session.presentation,
            "present_egress_data",
            wraps=self.session.presentation.present_egress_data,
        ) as spy_data:
            out2 = ANTHROPIC_WIRE.project_request(
                turn2,
                self.session.presentation,
                registered_tools=REGISTERED,
                tool_name_prefix=PREFIX,
                cache=cache,
            )
            # 仅新增的 "next turn" 触发 1 次文本清洗；tools 命中工具缓存，不再重复清洗 tools schema
            self.assertEqual(spy_text.call_count, 1)
            # 仅 output_config / output_format 两处顶层 schema 调用 present_egress_data，tools 已命中缓存
            self.assertEqual(spy_data.call_count, 2)

        self.assertEqual(
            json.dumps(out1["tools"], sort_keys=False),
            json.dumps(out2["tools"], sort_keys=False),
        )
        self.assertEqual(
            json.dumps(out1["output_config"]["format"]["schema"], sort_keys=False),
            json.dumps(out2["output_config"]["format"]["schema"], sort_keys=False),
        )
        self.assertEqual(
            json.dumps(out1["output_format"]["schema"], sort_keys=False),
            json.dumps(out2["output_format"]["schema"], sort_keys=False),
        )
        self.assertEqual(
            out1["messages"][0]["content"][0]["content"][0]["text"],
            out2["messages"][0]["content"][0]["content"][0]["text"],
        )

        # 篡改返回 tools 副本不得污染缓存
        out2["tools"][0]["description"] = "MUTATED"
        out3 = ANTHROPIC_WIRE.project_request(
            turn2,
            self.session.presentation,
            registered_tools=REGISTERED,
            tool_name_prefix=PREFIX,
            cache=cache,
        )
        self.assertEqual(out3["tools"][0]["description"], "Observe scene")

        # 暖缓存下仍严格拒绝远端图片 URL，绝不绕过安全检查
        bad = deepcopy(turn2)
        bad["messages"][0]["content"][0]["content"][1] = {
            "type": "image",
            "source": {"type": "url", "url": "https://evil.invalid/a.png"},
        }
        with self.assertRaisesRegex(ValueError, "sandbox_remote_media_url_disabled"):
            ANTHROPIC_WIRE.project_request(
                bad,
                self.session.presentation,
                registered_tools=REGISTERED,
                tool_name_prefix=PREFIX,
                cache=cache,
            )

    def test_responses_cache_preserves_key_order_skips_large_media_fingerprint_and_enforces_lru(self):
        from ni_rua.cortex.session.egress_cache import EgressProjectionCache
        from ni_rua.cortex.session.wire.openai_responses import present_responses_request, request_item_fingerprint

        # 键顺序不同必须产生不同指纹（sort_keys=False）
        self.assertNotEqual(
            request_item_fingerprint({"a": 1, "b": 2}),
            request_item_fingerprint({"b": 2, "a": 1}),
        )

        # 大图条目（>64 KiB）跳过外层条目缓存，但内部 JSON 文本仍命中字符串 LRU
        cache = EgressProjectionCache()
        heavy_text = json.dumps({"note": "large-image-item frame-1234abcd1234", "sha256": DIGEST})
        large_url = "data:image/webp;base64," + ("QUJD" * (20 * 1024))
        req = {
            "model": "gpt-6.1-sol",
            "instructions": "System instructions",
            "tools": [
                {
                    "type": "function",
                    "name": "robot_observe",
                    "description": "Observe",
                    "parameters": {
                        "type": "object",
                        "properties": {"x": {"type": "string", "description": "Ref frame-1234abcd1234"}},
                    },
                }
            ],
            "input": [
                {
                    "type": "function_call_output",
                    "call_id": "c1",
                    "output": [
                        {"type": "input_text", "text": heavy_text},
                        {"type": "input_image", "image_url": large_url},
                    ],
                }
            ],
        }
        first = present_responses_request(req, self.session.presentation, deferred_tools=frozenset(), namespace=None, cache=cache)
        self.assertEqual(cache.bucket_size("items"), 0)
        self.assertEqual(cache.bucket_size("tools"), 1)

        self.session.presentation.present({"frame_id": "frame-1234abcd1234"})
        with mock.patch.object(
            self.session.presentation,
            "present_egress_text",
            wraps=self.session.presentation.present_egress_text,
        ) as spy_text:
            second = present_responses_request(req, self.session.presentation, deferred_tools=frozenset(), namespace=None, cache=cache)
            self.assertEqual(spy_text.call_count, 0)
        self.assertEqual(
            json.dumps(first, ensure_ascii=False, sort_keys=False),
            json.dumps(second, ensure_ascii=False, sort_keys=False),
        )

        # 字符串 LRU：热条目被访问后保留，冷条目优先淘汰；超单条字节上限跳过缓存
        small_cache = EgressProjectionCache(max_text_entries=2, max_text_bytes=40, max_single_text_bytes=20, max_item_entries=2)
        calls = []
        fn = lambda s: (calls.append(s), s.upper())[1]
        self.assertEqual(small_cache.cached_text("a" * 10, fn), "A" * 10)
        self.assertEqual(small_cache.cached_text("b" * 10, fn), "B" * 10)
        # 命中 "a"*10 使其成为最新热条目
        self.assertEqual(small_cache.cached_text("a" * 10, fn), "A" * 10)
        # 插入 "c"*10 触发条目数上限，淘汰较冷的 "b"*10 而保留 "a"*10
        self.assertEqual(small_cache.cached_text("c" * 10, fn), "C" * 10)
        self.assertEqual(small_cache.cached_text("a" * 10, fn), "A" * 10)
        self.assertEqual(calls, ["a" * 10, "b" * 10, "c" * 10])
        # 超单条上限不进缓存
        self.assertEqual(small_cache.cached_text("d" * 25, fn), "D" * 25)
        self.assertEqual(small_cache.cached_text("d" * 25, fn), "D" * 25)
        self.assertEqual(calls.count("d" * 25), 2)

if __name__ == "__main__":
    unittest.main()
