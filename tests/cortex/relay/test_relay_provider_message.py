"""Relay failure evidence keeps a bounded operator copy of the provider's error message (Claude Code and OpenCode
profiles; ``ClientProfile.record_provider_message``), never in client bytes; the Codex profile is unchanged. An
operator manual retry lifts the transport block after an explicit provider error (e.g. after a model switch)."""

NI_RUA_TEST_TIER = "fast"

import json
import unittest
from cortex.relay import test_model_proxy_anthropic as relay_fixture
from support.fake_model_servers import Failure, Step, text_step
from ni_rua.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire
from ni_rua.cortex.session.wire.base import PROVIDER_MESSAGE_LIMIT, provider_error_message

IMAGE_400 = {"type": "error", "error": {"type": "invalid_request_error", "message": "IMAGE_INPUT_UNSUPPORTED"}}


class ProviderMessageTests(unittest.TestCase):
    def test_shapes_and_bounds(self):
        self.assertEqual(provider_error_message(IMAGE_400), "IMAGE_INPUT_UNSUPPORTED")
        self.assertEqual(provider_error_message({"error": {"message": "model not found", "code": "x"}}),
                         "model not found")
        self.assertEqual(provider_error_message({"error": "plain text error"}), "plain text error")
        self.assertEqual(provider_error_message({"message": "top level"}), "top level")
        for body in (None, [], "text", {}, {"error": {"message": 5}}, {"error": {"message": " \n\t "}}):
            self.assertIsNone(provider_error_message(body), body)
        self.assertEqual(provider_error_message({"error": {"message": "line one\nline\ttwo\x00\x07end"}}),
                         "line one line two end")
        long = provider_error_message({"error": {"message": "word " * 200}})
        self.assertTrue(len(long) <= PROVIDER_MESSAGE_LIMIT + 1 and long.endswith("…"), long)

    def test_credential_like_runs_are_masked(self):
        message = ("invalid key sk-live-abcdefghijklmnop for Bearer eyJhbGciOiJIUzI1NiJ9.payload and "
                   "request 0123456789abcdef0123456789abcdef0123")
        masked = provider_error_message({"error": {"message": message}})
        for secret in ("sk-live-abcdefghijklmnop", "eyJhbGciOiJIUzI1NiJ9", "0123456789abcdef0123456789abcdef0123"):
            self.assertNotIn(secret, masked)
        self.assertEqual(masked.count("[redacted]"), 3, masked)
        self.assertTrue(masked.startswith("invalid key [redacted] for [redacted]"), masked)

    def test_profiles(self):
        codex = OpenAIResponsesWire().operator_response_error({"error": {"message": "model gone", "code": "x"}})
        self.assertNotIn("message", codex)                     # Codex evidence (and client bytes) unchanged
        for wire in (AnthropicMessagesWire(), AnthropicMessagesWire(client="opencode"),
                     OpenAIResponsesWire(client="opencode")):
            with self.subTest(wire=(wire.name, wire.client)):
                self.assertTrue(wire.record_provider_message)
                error = wire.operator_response_error(IMAGE_400)
                self.assertEqual({key: error[key] for key in error if key != "message"}, wire.failed_response_error(IMAGE_400))
                self.assertEqual(error["message"], "IMAGE_INPUT_UNSUPPORTED")
                self.assertNotIn("message", wire.operator_response_error({"error": {"type": "x"}}))


class RelayProviderMessageTests(unittest.TestCase):
    setUp = relay_fixture.AnthropicRelayTest.setUp
    server = relay_fixture.AnthropicRelayTest.server
    proxy = relay_fixture.AnthropicRelayTest.proxy
    call = relay_fixture.AnthropicRelayTest.call
    count = staticmethod(relay_fixture.AnthropicRelayTest.count)
    kinds = staticmethod(relay_fixture.AnthropicRelayTest.kinds)

    def test_http_400_message_is_operator_evidence_and_manual_retry_resumes(self):
        server = self.server([Step(failure=Failure(kind="http", status=400, error_type="invalid_request_error",
                                                   message="IMAGE_INPUT_UNSUPPORTED")),
                              text_step("after the switch")])
        proxy = self.proxy(server, max_requests=5)
        failed = self.call(proxy, relay_fixture.request_body("look at the image"))
        self.assertNotIn(b"IMAGE_INPUT_UNSUPPORTED", failed.body)          # client bytes stay neutral
        self.assertTrue(proxy.transport_blocked)
        evidence = proxy.blocked_error["evidence"]
        self.assertEqual((evidence["http_status"], evidence["type"], evidence["message"],
                          proxy.blocked_error["manual_retry_allowed"]),
                         (400, "invalid_request_error", "IMAGE_INPUT_UNSUPPORTED", True))
        blocked, = self.kinds(proxy, "model_transport_blocked")
        self.assertEqual(blocked["error"]["evidence"]["message"], "IMAGE_INPUT_UNSUPPORTED")
        self.assertNotIn("host-secret", json.dumps(blocked))
        denied = self.call(proxy, relay_fixture.request_body("again"), bookkeeping=None)
        self.assertEqual(denied.status, 403)
        self.assertEqual(len(server.generation_requests()), 1)
        proxy.manual_retry()
        resumed = self.call(proxy, relay_fixture.request_body("with an image-capable model", model="kimi-k2.6"))
        self.assertEqual(resumed.status, 200)
        self.assertIn(b"after the switch", resumed.body)
        self.assertFalse(proxy.transport_blocked)
        self.assertEqual(server.generation_requests()[-1].json["model"], "kimi-k2.6")

if __name__ == "__main__":
    unittest.main()
