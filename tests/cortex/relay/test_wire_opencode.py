"""OpenCode client profiles of the Anthropic Messages and OpenAI Responses wires (design §4.2, plan T9 / P3-4;
re-pinned to opencode 2.2.7 by OC2-E1, design 11 §5).

Inputs are real opencode 2.2.7 requests and measured client attempt counts (``tests/fixtures/opencode/
opencode_wire_bodies.json``, built by ``build_opencode_bodies.py`` from ``capture_opencode.py``) and the 2.2.7 sequences
of the continuity fixture (``tests/fixtures/continuity/continuity_fixture.json``: steer, interrupts, a foreground
subagent, auto-compaction). The relay is exercised for real against the fake model servers. OpenCode's client-side
retry rule is simulated from the 2.2.7 binary's provider-error classification, and the simulator is first checked
against every attempt count measured on the real binary. Then every relay denial and every failed delivery must cost
exactly one client attempt.
"""

NI_RUA_TEST_TIER = "component"

import base64
from email.message import Message
import json
from pathlib import Path
import re
import threading
import time
import unittest
from ni_rua.cortex.relay import continuity as cs
from ni_rua.cortex.relay.server import ModelRequestProxy
from ni_rua.cortex.session.wire import (AnthropicMessagesWire, OPENCODE_PROFILE, OpenAIResponsesWire, Route,
                                        present_messages_request, wire_from_name)
from ni_rua.cortex.session.wire import base as wire_base
from support.fake_model_servers import Failure, FakeAnthropicServer, FakeResponsesServer, Gate, Step, ToolCall
from cortex.relay.test_model_proxy_continuity_and_guard import RelayContinuityCase, fixture_body, load_fixture, post
from support.paths import FIXTURES_ROOT

FIXTURE = FIXTURES_ROOT / "opencode" / "opencode_wire_bodies.json"
DIGEST = "0123456789abcdef" * 4
ARTIFACT = "artifact:" + "a" * 64 + "-" + "b" * 16
LONG_OBSERVATION = "observation-" + "c" * 32
PLANTED = f"sha256={DIGEST} see {ARTIFACT} and {LONG_OBSERVATION}"
INSTRUCTIONS_FILE = "/Users/operator/runs/run-7/private/opencode/instructions/NI-RUA.md"
# opencode 2.2.7 provider-error classification (its AI layer's ``AI.Error`` reason classifier and ``isRetryable``, read
# in the binary), reduced to what decides a retry. A response header ``x-should-retry: true|false`` overrides it. Text
# patterns apply to the error message and the raw body; at an HTTP status >= 400 only the rate-limit pattern does (the
# 1.19.5 digit / "overloaded" / "connection" patterns at HTTP 400 are gone).
RATE_LIMIT_TEXT = re.compile(r"rate increased too quickly|rate[-_\s]?limit|too[_\s]?many[_\s]?requests", re.I)
# Error types/codes (lower case) that end the request without a retry: authentication, quota, content policy.
AUTH_CODES = frozenset({"authentication_error", "permission_error", "permission_denied", "api_key_invalid"})
QUOTA_CODES = frozenset({"insufficient_quota", "usage_not_included", "billing_error", "gousagelimiterror",
                         "budgetlimiterror", "freeusagelimiterror", "creditlimitexceeded"})
POLICY_CODES = frozenset({"content_filter", "content_moderation", "responsibleaipolicyviolation",
                          "content_policy_violation", "image_content_policy_violation", "refusal",
                          "datainspectionfailed", "moderation_blocked"})
# Invalid-request codes: not retried, unless a transient code or the rate-limit text also matches (checked first).
INVALID_CODES = frozenset({"model_not_found", "invalid_prompt", "invalid_request", "invalid_request_error",
                           "validationexception"})
TRANSIENT_CODES = frozenset({"api_error", "internal_error", "internalserverexception", "modelstreamerrorexception",
                             "overloaded_error", "server_error", "server_is_overloaded", "slow_down",
                             "serviceunavailableexception"})
MAX_RETRIES = 10       # measured: every retried form made 11 attempts in 77-92 s (capture_opencode.py attempts)
# Measured: an empty ``response.incomplete`` answer is continued twice (a ``synthetic`` "hit the output token limit"
# message each time), then the turn fails.
MAX_LENGTH_CONTINUATIONS = 2


def error_codes(error):
    """Lower-case type/code strings of a decoded provider error object (``{"type", "code", ...}``)."""
    if not isinstance(error, dict):
        return set()
    return {value.lower() for key in ("type", "code") for value in [error.get(key)] if isinstance(value, str)}


def classify(status, codes, text, should_retry=None):
    """``retry`` or ``fail`` for one provider error: HTTP ``status`` (None for an error inside a 200 stream), its
    type/code strings and its message + raw body text."""
    if should_retry in ("true", "false"):
        return "retry" if should_retry == "true" else "fail"
    if codes & (POLICY_CODES | QUOTA_CODES | AUTH_CODES) or status in (401, 402, 403, 413):
        return "fail"
    if status == 429 or any("rate_limit" in code or code == "too_many_requests" for code in codes) \
            or RATE_LIMIT_TEXT.search(text):
        return "retry"
    if status is not None:
        return "retry" if status in (408, 409) or status >= 500 else "fail"
    if codes & TRANSIENT_CODES or any("exhausted" in code or "unavailable" in code for code in codes):
        return "retry"
    return "fail" if codes & INVALID_CODES else "retry"   # any other stream error: unknown provider error, retried


def sse_events(body):
    events = []
    for block in body.replace(b"\r\n", b"\n").split(b"\n\n"):
        data = b"\n".join(line[5:].lstrip() for line in block.split(b"\n") if line.startswith(b"data:"))
        if data and data != b"[DONE]":
            events.append(json.loads(data))
    return events


def opencode_reaction(wire_name, status, body, headers=None):
    """What opencode 2.2.7 does with one answer: ``success``, ``fail`` (one attempt, error surfaced), ``retry``
    (backoff and re-send, at most ``MAX_RETRIES`` times) or ``continue`` (a ``response.incomplete`` answer: a
    ``synthetic`` continuation request, at most ``MAX_LENGTH_CONTINUATIONS`` times). A stream that ends without its
    terminal is an incomplete-stream error and retried like a 5xx (1.19.5 re-requested it in a tight loop)."""
    headers = {str(k).lower(): v for k, v in dict(headers or {}).items()}
    should_retry = headers.get("x-should-retry")
    if status != 200:
        try:
            error = json.loads(body).get("error")
        except (ValueError, AttributeError):
            error = None
        message = error.get("message") if isinstance(error, dict) else error if isinstance(error, str) else ""
        return classify(status, error_codes(error), f"{message} {body.decode(errors='replace')}", should_retry)
    events = [event for event in sse_events(body) if event.get("type") != "ping"]
    kinds = [event.get("type") for event in events]
    if wire_name == "anthropic_messages":
        if "error" in kinds:
            error = events[kinds.index("error")].get("error")
            return classify(None, error_codes(error), json.dumps(error), should_retry)
        return "success" if "message_stop" in kinds else "retry"
    for kind in ("response.failed", "error"):
        if kind in kinds:
            event = events[kinds.index(kind)]
            error = (event.get("response") or {}).get("error") if kind == "response.failed" else event
            return classify(None, error_codes(error), json.dumps(error), should_retry)
    if "response.incomplete" in kinds:
        return "continue"
    return "success" if "response.completed" in kinds else "retry"


def run_opencode_client(send, wire_name):
    """Drive ``send() -> (status, body)`` or ``(status, body, headers)`` with the simulated client rule; returns
    (attempts, final reaction)."""
    attempts, retries, continuations = 0, 0, 0
    while True:
        answer = send()
        attempts += 1
        reaction = opencode_reaction(wire_name, *answer)
        if reaction == "retry" and retries < MAX_RETRIES:
            retries += 1
            continue
        if reaction == "continue" and continuations < MAX_LENGTH_CONTINUATIONS:
            continuations += 1
            continue
        return attempts, reaction


def rehydrate_images(node):
    """The continuity fixture stores media as digests (``<sha256:... len=N>``); the egress scrub rightly rejects a
    Responses ``input_image`` whose URL is not ``data:``. Map each digest to a deterministic data URL (equal digests,
    equal URLs), so every continuity relation of the captured sequence is preserved."""
    if isinstance(node, dict):
        out = {k: rehydrate_images(v) for k, v in node.items()}
        url = out.get("image_url")
        if out.get("type") == "input_image" and isinstance(url, str) and not url.startswith("data:"):
            out["image_url"] = "data:image/png;base64," + base64.b64encode(url.encode()).decode()
        return out
    if isinstance(node, list):
        return [rehydrate_images(v) for v in node]
    return node


class SimulatedClientMatchesCaptureTest(unittest.TestCase):
    """The simulator reproduces every attempt count measured on the real 2.2.7 binary (fixture ``client_attempts``:
    each answer form served verbatim on every request of one session, all forms in parallel)."""

    @classmethod
    def setUpClass(cls):
        cls.measured = json.loads(FIXTURE.read_text())["client_attempts"]

    def test_measured_attempts(self):
        for wire_name, forms in self.measured.items():
            self.assertGreaterEqual(len(forms), 15)
            for form in forms:
                with self.subTest(wire=wire_name, form=form["form"]):
                    answer = (form["status"], form["body"].encode(), form["headers"])
                    attempts, _ = run_opencode_client(lambda answer=answer: answer, wire_name)
                    self.assertEqual(attempts, form["attempts"])
                    self.assertEqual((form["stopped"], form["outcome"]), ("idle", "failed"))
                    if attempts == MAX_RETRIES + 1:
                        self.assertGreater(form["span_s"], 60)  # backoff, never a tight loop

    def test_measured_relay_replies_are_the_current_relay_bytes(self):
        """The measured relay forms are exactly what the adapters send today, and each cost one attempt."""
        for wire_name, forms in self.measured.items():
            wire = wire_from_name(wire_name, client="opencode")
            measured = {form["form"]: form for form in forms}
            replies = {"relay_denial:" + kind: wire.denial(kind) for kind in wire_base.OPENCODE_DENIAL_MESSAGES}
            replies["relay_error"] = wire.relay_error(streaming=False, evidence=None, manual_retry_allowed=False)
            for name, reply in replies.items():
                with self.subTest(wire=wire_name, form=name):
                    form = measured[name]
                    self.assertEqual((form["status"], form["headers"], form["body"].encode()),
                                     (reply.status, dict(reply.headers), reply.body))
                    self.assertEqual((form["attempts"], form["error_type"]), (1, "provider.invalid-request"))


# ------------------------------------------------------------------------------------------------- adapters
class OpenCodeProfileUnitTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())["cases"]

    def wires(self):
        return (AnthropicMessagesWire(client="opencode"), OpenAIResponsesWire(client="opencode"))

    def test_profile_and_worker_round_trip(self):
        self.assertEqual(OPENCODE_PROFILE, wire_base.ClientProfile(early_stream=False, client_loop_guard=True,
                                                                   record_provider_message=True))
        for wire in self.wires():
            with self.subTest(wire=wire.name):
                self.assertEqual((wire.client, wire.early_stream, wire.client_loop_guard), ("opencode", False, True))
                self.assertIsNone(wire.keepalive_frame())
                self.assertIsNone(wire.failed_payload(dict(status="failed", output=[]), streaming=True))
                restored = wire_from_name(wire.name, client="opencode")
                self.assertEqual((type(restored), restored.client), (type(wire), "opencode"))
        codex = OpenAIResponsesWire()
        self.assertEqual((codex.client, codex.early_stream), ("codex", True))
        self.assertIsNotNone(codex.keepalive_frame())

    def test_routes(self):
        responses = OpenAIResponsesWire(client="opencode")
        self.assertEqual(responses.route("POST", "/v1/responses"), Route("generation", "/responses", "/responses"))
        self.assertEqual(responses.route("POST", "/responses"), Route("generation", "/responses", "/responses"))
        self.assertEqual(responses.route("POST", "/v1/responses/compact"),
                         Route("generation", "/responses/compact", "/responses/compact", compaction=True))
        self.assertEqual(responses.route("POST", "/v1/responses?x=1").upstream_path, "/responses?x=1")
        self.assertEqual(responses.route("POST", "/v1/responses?x=1").endpoint, "/responses")
        for method, path in (("GET", "/v1/responses"), ("POST", "/v1/chat/completions"), ("POST", "/v1/models")):
            self.assertIsNone(responses.route(method, path))
        # Codex keeps exact routes only.
        self.assertIsNone(OpenAIResponsesWire().route("POST", "/v1/responses"))
        anthropic = AnthropicMessagesWire(client="opencode")
        self.assertEqual(anthropic.route("POST", self.fixture["anthropic_messages"]["path"]).kind, "generation")
        self.assertEqual(self.fixture["openai_responses"]["path"], "/v1/responses")
        self.assertEqual(responses.route("POST", self.fixture["openai_responses"]["path"]).kind, "generation")

    def test_thread_key_from_captured_headers(self):
        for wire in self.wires():
            with self.subTest(wire=wire.name):
                headers = self.fixture[wire.name]["headers"]
                self.assertTrue(headers["x-opencode-session-id"].startswith("ses_"))
                self.assertEqual(wire.thread_key(headers), headers["x-opencode-session-id"])
                message = Message()
                message["X-OpenCode-Session-Id"] = "ses_case"
                self.assertEqual(wire.thread_key(message), "ses_case")
                self.assertEqual(wire.thread_key({}), "")
        # The Claude Code / Codex keys are not read on the opencode profile and vice versa.
        self.assertEqual(AnthropicMessagesWire(client="opencode").thread_key({"X-Claude-Code-Session-Id": "x"}), "")
        self.assertEqual(AnthropicMessagesWire().thread_key({"x-opencode-session-id": "x"}), "")
        message = Message()
        message["thread-id"] = "codex-thread"
        self.assertEqual(OpenAIResponsesWire(client="opencode").thread_key(message), "")
        self.assertEqual(OpenAIResponsesWire().thread_key(message), "codex-thread")

    def test_relay_generated_answers_are_neutral_single_attempt_400s(self):
        # 2.2.7 retries an HTTP 400 only for the rate-limit text; the other words and digits retried at 400 in 1.19.5
        # and stay excluded so the replies remain neutral for either rule.
        forbidden = ("rate limit", "overloaded", "timeout", "try again", "capacity", "server error", "connection",
                     "unavailable", "exhausted", "logical", "evidence")
        for wire in self.wires():
            replies = {kind: wire.denial(kind) for kind in wire_base.OPENCODE_DENIAL_MESSAGES}
            replies["relay_error"] = wire.relay_error(streaming=False, evidence=dict(http_status=503, code="abc123",
                                                                                     detail="logical-call-9"),
                                                      manual_retry_allowed=True)
            replies["relay_error_streaming_flag"] = wire.relay_error(streaming=True, evidence=None,
                                                                     manual_retry_allowed=False)
            for name, reply in replies.items():
                with self.subTest(wire=wire.name, reply=name):
                    self.assertEqual(reply.status, 400)
                    self.assertEqual(dict(reply.headers)["Content-Type"], "application/json")
                    self.assertIsNone(re.search(rb"\d", reply.body))
                    for word in forbidden:
                        self.assertNotIn(word, reply.body.decode().lower())
                    self.assertEqual(opencode_reaction(wire.name, reply.status, reply.body, reply.headers), "fail")
                    # Also without the Anthropic adapter's x-should-retry: false header (the Responses replies carry
                    # none): the classification alone makes each one attempt.
                    self.assertEqual(opencode_reaction(wire.name, reply.status, reply.body), "fail")
                    run = run_opencode_client(lambda reply=reply: (reply.status, reply.body, reply.headers), wire.name)
                    self.assertEqual(run, (1, "fail"))
            self.assertEqual(len({reply.body for reply in replies.values()}), len(wire_base.OPENCODE_DENIAL_MESSAGES) + 1)
            missing = wire.not_found()
            self.assertEqual(missing.status, 404)
            self.assertIsNone(re.search(rb"\d", missing.body))
        self.assertEqual(json.loads(AnthropicMessagesWire(client="opencode").denial("budget").body)["error"]["type"],
                         "invalid_request_error")
        self.assertEqual(json.loads(OpenAIResponsesWire(client="opencode").denial("budget").body),
                         {"error": {"message": "ni_rua_model_budget_spent", "type": "invalid_request_error",
                                    "param": None, "code": None}})
        # Codex answers are unchanged.
        self.assertEqual(OpenAIResponsesWire().denial("budget").status, 429)
        self.assertEqual(OpenAIResponsesWire().not_found().body, None)


class _Presentation:
    def __init__(self, case):
        import tempfile
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        from ni_rua.core.storage.store import Store
        directory = tempfile.TemporaryDirectory()
        case.addCleanup(directory.cleanup)
        store = Store(Path(directory.name) / "run", dict(run_id="relay", episode_id="episode",
                                                         env_instance_id="environment", env_generation=1))
        case.addCleanup(store.close)
        self.presentation = ActorPresentation(store)


class OpenCodeEgressTest(unittest.TestCase):
    """Egress projection of real opencode bodies on both wires (planted host data, protected protocol fields)."""

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())["cases"]

    def setUp(self):
        self.presentation = _Presentation(self).presentation

    def project(self, wire, body):
        return wire.project_request(body, self.presentation, registered_tools=frozenset({"fast", "image"}),
                                    tool_name_prefix="ni_rua_")

    @staticmethod
    def plant_responses(body):
        body = json.loads(json.dumps(body))
        body["instructions"] = body["instructions"].replace(
            "<env>", "Instructions from: " + INSTRUCTIONS_FILE + "\nRUN " + PLANTED + "\n<env>", 1)
        for item in body["input"]:
            if item.get("type") == "function_call_output":
                if isinstance(item["output"], str):
                    item["output"] += " " + PLANTED
                else:
                    item["output"][0]["text"] += " " + PLANTED
            elif item.get("type") == "function_call":
                item["arguments"] = json.dumps(dict(json.loads(item["arguments"]), note=PLANTED))
            elif isinstance(item.get("content"), list):
                item["content"][0]["text"] += " " + PLANTED
        return body

    def assert_clean(self, value):
        serialized = json.dumps(value)
        for planted in (DIGEST, ARTIFACT, LONG_OBSERVATION, INSTRUCTIONS_FILE):
            self.assertNotIn(planted, serialized)

    def test_responses_real_body(self):
        wire = OpenAIResponsesWire(client="opencode")
        original = self.fixture["openai_responses"]["last_body"]
        fields = ("prompt_cache_key", "include", "store", "max_output_tokens", "reasoning", "stream")
        for key in fields + ("instructions",):
            self.assertIn(key, original)
        # 2.2.7 sends the system text as ``instructions`` (1.19.5: a leading developer item) and no tool_choice.
        self.assertNotIn("tool_choice", original)
        self.assertEqual(original["input"][0]["role"], "user")
        self.assertIn("\nInstructions from: /run/", original["instructions"])
        planted = self.plant_responses(original)
        output = self.project(wire, planted)
        self.assert_clean(output)
        self.assertIn("\nInstructions:\nRUN ", output["instructions"])
        self.assertNotIn("Instructions from: ", output["instructions"])
        for key in ("model",) + fields:
            self.assertEqual(output[key], original[key], key)
        self.assertEqual([(t["type"], t["name"], t.get("strict")) for t in output["tools"]],
                         [(t["type"], t["name"], t.get("strict")) for t in original["tools"]])
        for before, after in zip(original["input"], output["input"]):
            self.assertEqual((after.get("type"), after.get("role"), after.get("call_id"), after.get("name")),
                             (before.get("type"), before.get("role"), before.get("call_id"), before.get("name")))
            if isinstance(before.get("output"), list):
                self.assertEqual([p for p in after["output"] if p["type"] == "input_image"],
                                 [p for p in before["output"] if p["type"] == "input_image"])
        self.assertEqual(self.project(wire, output), output)
        # The unplanted real body is unchanged apart from the instruction header and the canonical re-serialisation
        # (sorted keys) of JSON tool-output text: 2.2.7 renders an MCP isError result as the JSON text
        # {"error": {"type": "tool.execution", "message": ...}, "content": []}.
        projected = self.project(wire, original)["input"]
        self.assertEqual(len(projected), len(original["input"]))
        for before, after in zip(original["input"], projected):
            if before != after:
                self.assertEqual(before["type"], "function_call_output")
                self.assertEqual(json.loads(after["output"]), json.loads(before["output"]))
                self.assertIn('"tool.execution"', before["output"])

    def test_anthropic_real_body_instruction_header(self):
        original = self.fixture["anthropic_messages"]["last_body"]
        # 2.2.7 splits the system text into four blocks; the instruction file and <env> are in one of them.
        block = next(k for k, part in enumerate(original["system"]) if "<env>" in part["text"])
        self.assertIn("\nInstructions from: /run/", original["system"][block]["text"])
        body = json.loads(json.dumps(original))
        body["system"][block]["text"] = body["system"][block]["text"].replace(
            "<env>", "Instructions from: " + INSTRUCTIONS_FILE + "\nRUN " + PLANTED + "\n<env>", 1)
        output = self.project(AnthropicMessagesWire(client="opencode"), body)
        self.assert_clean(output)
        self.assertIn("\nInstructions:\nRUN ", output["system"][block]["text"])
        self.assertEqual([part.get("cache_control") for part in output["system"]],
                         [part.get("cache_control") for part in original["system"]])
        for key in ("max_tokens", "thinking", "output_config", "stream"):
            self.assertEqual(output[key], original[key], key)
        self.assertEqual([(t["name"], t.get("cache_control")) for t in output["tools"]],
                         [(t["name"], t.get("cache_control")) for t in original["tools"]])
        self.assertEqual(output["messages"], self.project(AnthropicMessagesWire(client="opencode"), original)["messages"])
        # Only the opencode profile rewrites the header; only the system text is touched.
        claude = present_messages_request(body, self.presentation, registered_tools=frozenset(),
                                          tool_name_prefix="mcp__ni_rua__", client="claude_code")
        self.assertIn(INSTRUCTIONS_FILE, claude["system"][block]["text"])
        user_header = dict(original, messages=[dict(role="user", content="Instructions from: " + INSTRUCTIONS_FILE)])
        self.assertIn(INSTRUCTIONS_FILE, json.dumps(self.project(AnthropicMessagesWire(client="opencode"), user_header)))

    def test_responses_continuity_inputs_separate_the_instructions(self):
        body = self.fixture["openai_responses"]["first_body"]
        wire = OpenAIResponsesWire(client="opencode")
        view = cs.openai_responses_history(body)
        self.assertEqual(view.system, (body["instructions"],))
        inputs = wire.continuity_inputs(body)
        self.assertFalse(inputs["helper"])
        self.assertEqual(len(inputs["fingerprints"]), len(body["input"]))


# ------------------------------------------------------------------------------------- relay x fake servers
class OpenCodeRelayTest(RelayContinuityCase):
    """The real relay with the opencode profiles against the fake servers."""

    WIRES = (("anthropic_messages", "/v1/messages", FakeAnthropicServer, ""),
             ("openai_responses", "/v1/responses", FakeResponsesServer, ""))

    @classmethod
    def setUpClass(cls):
        cls.fixture = json.loads(FIXTURE.read_text())["cases"]
        cls.continuity = {sequence["id"]: sequence for sequence in load_fixture()["sequences"]}

    def opencode_proxy(self, wire_name, server, **kwargs):
        wire = wire_from_name(wire_name, client="opencode")
        return self.proxy(server, wire, **kwargs)

    # Session headers of a 2.2.7 request (thread key, root session copies, a child's parent id).
    SESSION_HEADERS = ("x-opencode-session-id", "x-opencode-session", "x-session-affinity", "x-session-id",
                       "x-opencode-parent-session-id", "x-parent-session-id")

    def headers(self, wire_name, session="ses_fixture"):
        headers = tuple((k, v) for k, v in self.fixture[wire_name]["headers"].items()
                        if k.lower() not in self.SESSION_HEADERS)
        return headers + (("x-opencode-session-id", session),)

    def body(self, wire_name, **extra):
        return json.dumps(dict(self.fixture[wire_name]["first_body"], **extra)).encode()

    @staticmethod
    def answer(reply):
        """``post`` result ``(status, headers, body)`` as the simulator's ``(status, body, headers)``."""
        return reply[0], reply[2], reply[1]

    def test_success_is_delivered_whole_and_nothing_is_sent_before_validation(self):
        for wire_name, path, server_cls, _ in self.WIRES:
            with self.subTest(wire=wire_name):
                gate = Gate()
                server = self.server(server_cls, [Step(text="final answer", gate=gate)])
                proxy = self.opencode_proxy(wire_name, server)
                received = []
                worker = threading.Thread(target=lambda: received.append(post(proxy, path, self.body(wire_name),
                                                                              self.headers(wire_name))))
                worker.start()
                self.assertTrue(gate.wait_entered(5))
                time.sleep(0.4)
                self.assertEqual(received, [])  # no status line, no keepalive while the upstream answer is pending
                gate.release()
                worker.join(10)
                status, headers, body = received[0]
                self.assertEqual(status, 200)
                self.assertEqual(headers["content-length"], str(len(body)))
                self.assertNotIn(b"keepalive", body)
                self.assertNotIn(b"ping", body)
                self.assertEqual(opencode_reaction(wire_name, status, body), "success")
                self.assertEqual(proxy.request_count, 1)
                forwarded = server.generation_requests()[0]
                self.assertEqual(forwarded.headers["x-opencode-session-id"], "ses_fixture")
                completed = self.events(proxy, "model_call_completed")[0]
                self.assertEqual((completed["native_thread_id"], completed["call_purpose"]), ("ses_fixture", "main"))
                if wire_name == "openai_responses":
                    self.assertEqual(forwarded.route, "/responses")

    def test_denials_cost_exactly_one_client_attempt(self):
        for wire_name, path, server_cls, _ in self.WIRES:
            server = self.server(server_cls)
            for kind in ("budget", "transport_blocked", "thread_cancelled"):
                with self.subTest(wire=wire_name, denial=kind):
                    proxy = self.opencode_proxy(wire_name, server, max_requests=1)
                    if kind == "budget":
                        with proxy.lock:
                            proxy.request_count = 1
                    elif kind == "transport_blocked":
                        with proxy.lock:
                            proxy.blocked_error = dict(logical_call_id="x", error_type="Fixture",
                                                       manual_retry_allowed=True, evidence=None)
                    else:
                        proxy.register_thread("ses_fixture", "actor", "ses_fixture")
                        proxy.cancel_thread("ses_fixture")
                    attempts, reaction = run_opencode_client(
                        lambda proxy=proxy: self.answer(post(proxy, path, self.body(wire_name), self.headers(wire_name),
                                                             bookkeeping=None)), wire_name)
                    self.assertEqual((attempts, reaction), (1, "fail"))
            self.assertEqual(server.generation_requests(), [])

    def test_failed_deliveries_cost_exactly_one_client_attempt(self):
        anthropic_failures = [
            Step(failure=Failure("stream_error", message="mid stream failure")),  # overloaded_error event
            Step(failure=Failure("stream_error", 400, "bad", error_type="invalid_request_error")),
            Step(text="cut", failure=Failure("truncated")),
            Step(tool_calls=[ToolCall("ni_rua_fast", {"note": "x"})], failure=Failure("malformed_tool_json"),
                 stop_reason="max_tokens"),
            Step(failure=Failure("http", 529, "provider is busy")),
            Step(failure=Failure("http", 400, "provider 400 text", error_type="invalid_request_error")),
        ]
        responses_failures = [
            Step(failure=Failure("failed", error_type="server_error")),
            Step(text="partial", failure=Failure("incomplete")),
            Step(failure=Failure("stream_error", message="mid stream failure")),
            Step(text="cut", failure=Failure("truncated")),
            Step(failure=Failure("http", 503, "provider unavailable")),
        ]
        for (wire_name, path, server_cls, _), failures in zip(self.WIRES, (anthropic_failures, responses_failures)):
            for retries in (0, 2):
                for step in failures:
                    with self.subTest(wire=wire_name, failure=step.failure.kind, status=step.failure.status,
                                      retries=retries):
                        server = self.server(server_cls, default=step)
                        proxy = self.opencode_proxy(wire_name, server, max_retries=retries)
                        attempts, reaction = run_opencode_client(
                            lambda proxy=proxy: self.answer(post(proxy, path, self.body(wire_name),
                                                                 self.headers(wire_name))), wire_name)
                        self.assertEqual((attempts, reaction), (1, "fail"))
                        self.assertEqual(proxy.request_count, 0)
                        self.assertTrue(proxy.transport_blocked)

    def test_cut_messages_never_reach_the_client(self):
        """T0 for the deferred profile: an upstream answer cut after a tool_use start, mid argument JSON or before
        message_stop yields one neutral 400 and zero content bytes."""
        from cortex.relay.test_relay_review_followups import CutMessageNeverReleasedTest, RawUpstream
        for name, raw in CutMessageNeverReleasedTest.CUTS.items():
            with self.subTest(cut=name):
                upstream = RawUpstream([raw])
                self.addCleanup(upstream.close)
                wire = AnthropicMessagesWire(client="opencode")
                proxy = ModelRequestProxy(upstream.base_url, wire=wire, max_retries=0,
                                          upstream_auth_headers={"x-api-key": "k"}).start()
                self.addCleanup(proxy.close)
                status, headers, body = post(proxy, "/v1/messages", self.body("anthropic_messages"),
                                             self.headers("anthropic_messages"))
                self.assertEqual(status, 400)
                self.assertEqual(json.loads(body)["error"]["message"], "ni_rua_model_request_failed")
                for forbidden in (b"tool_use", b"toolu_cut", b"content_block", b"message_start", b"visible text"):
                    self.assertNotIn(forbidden, body)

    def test_captured_sessions_through_the_relay(self):
        """Every 2.2.7 sequence on a relay wire (basic, steer + interrupts, foreground subagent, three
        auto-compactions) through the real relay, wired as ``OpenCodeSession`` wires it (the session's thread
        registered as the actor thread, with and without ``default_thread_id``): one discontinuity per compaction, on
        the first request after each summary request; the summary requests and every child-session request
        (``x-opencode-parent-session-id``, own ``x-opencode-session-id``, the parent's exact system text) pass without
        one; child requests are ``subagent``, everything else ``main``; no helper call exists; every call is charged."""
        sequences = [s for s in self.continuity.values() if s["client"] == "opencode"
                     and s["wire"] in dict((w[0], w) for w in self.WIRES)]
        self.assertEqual(len(sequences), 8)
        paths = {wire_name: path for wire_name, path, _, _ in self.WIRES}
        subagent_rows = 0
        for sequence in sequences:
            for default in (True, False):
                with self.subTest(sequence=sequence["id"], default_thread_id=default):
                    self.reasons = []
                    server = self.server(dict((w[0], w[2]) for w in self.WIRES)[sequence["wire"]])
                    proxy = self.opencode_proxy(sequence["wire"], server, default_thread_id="s-t1" if default else None)
                    proxy.register_thread("s-t1", "actor", "s-t1")
                    headers = tuple((k, v) for k, v in self.headers(sequence["wire"])
                                    if k.lower() != "x-opencode-session-id")
                    replayed = dict(sequence, values=rehydrate_images(sequence["values"]))
                    statuses = self.replay(proxy, replayed, range(len(sequence["requests"])),
                                           path=paths[sequence["wire"]], thread_header="x-opencode-session-id",
                                           base_headers=headers)
                    self.assertEqual(set(statuses), {200})
                    labels = [request["label"] for request in sequence["requests"]]
                    lost = [k + 1 for k, label in enumerate(labels) if label == "context_lost"]
                    events = self.events(proxy, "model_context_discontinuity")
                    self.assertEqual([event["logical_call_index"] for event in events], lost)
                    self.assertEqual(self.reasons, [cs.HISTORY_SHORTENED] * labels.count("compaction_request"))
                    self.assertEqual({event["call_purpose"] for event in events}, {"main"} if lost else set())
                    purposes = {e["logical_call_index"]: e["call_purpose"]
                                for e in self.events(proxy, "model_call_completed")}
                    self.assertEqual(purposes, {k + 1: "subagent" if label == "subagent" else "main"
                                                for k, label in enumerate(labels)})
                    self.assertEqual(proxy.request_count, len(labels))  # subagent and summary calls are charged
                    subagent_rows += default and labels.count("subagent")
        self.assertEqual(subagent_rows, 6)

if __name__ == "__main__":
    unittest.main()
