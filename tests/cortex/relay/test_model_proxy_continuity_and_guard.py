"""Merged cortex tests:
- tests/test_model_proxy_continuity.py: ContinuityTracker call sites in ModelRequestProxy (design §4.3, P1-3a integration).
- tests/test_model_proxy_consume_guard.py: Relay consume guard (``ModelRequestProxy``, class docstring "Consume guard"; work package CONSUME-GUARD).
"""

NI_RUA_TEST_TIER = "component"

import base64
import http.client
import json
from pathlib import Path
import socket
import tempfile
import threading
import time
import unittest
from unittest import mock
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.core.storage.store import Store
from ni_rua.cortex.relay import continuity as cs
from ni_rua.cortex.relay.continuity import ContinuityTracker
from ni_rua.cortex.relay.server import ModelRequestProxy, PROVIDER_CONTEXT_EDIT
from ni_rua.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire
from support.fake_model_servers import Failure, FakeAnthropicServer, Step, ToolCall, sse
from ni_rua.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire, wire_from_name
from support.fake_model_servers import FakeAnthropicServer, FakeResponsesServer, Gate, Step, text_step
from support.paths import FIXTURES_ROOT

# ==== tests/test_model_proxy_continuity.py ====

FIXTURE = FIXTURES_ROOT / "continuity" / "continuity_fixture.json"
SESSION = "9ff8e771-aa08-4c17-b00d-032e1cb2011f"
PREFIX = "mcp__ni_rua__"
PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
CLAUDE_HEADERS = (("Content-Type", "application/json"), ("x-api-key", "ni-rua-local-relay"),
                  ("anthropic-version", "2023-06-01"))
COMPACTION_SEQUENCE = "claude_code/s5-compact/model-requests"


def load_fixture():
    return json.loads(FIXTURE.read_text())


def fixture_body(sequence, request, **extra):
    """The fixture request (system/messages/input/tools) as a complete generation body."""
    values = sequence["values"]
    body = {key: [values[i] for i in ref] if isinstance(ref, list) else values[ref]
            for key, ref in request["body"].items()}
    return dict(dict(model="fixture-model", stream=True), **body, **extra)


def post(proxy, path, data, headers, *, bookkeeping="model_call_finished"):
    """One client request read to EOF; waits for the relay's ``bookkeeping`` event (None: no wait).
    Returns (status, headers, body)."""
    with proxy.lock:
        before = sum(event["kind"] == bookkeeping for event in proxy.events)
    port = int(proxy.base_url.rsplit(":", 1)[1])
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=30)
    try:
        connection.putrequest("POST", path, skip_accept_encoding=True)
        for key, value in headers:
            connection.putheader(key, value)
        connection.putheader("Content-Length", str(len(data)))
        connection.endheaders(data)
        response = connection.getresponse()
        status, reply_headers, body = response.status, dict(response.getheaders()), response.read()
    finally:
        connection.close()
    deadline = time.monotonic() + 10
    while bookkeeping and time.monotonic() < deadline:
        with proxy.lock:
            if sum(event["kind"] == bookkeeping for event in proxy.events) > before:
                break
        time.sleep(0.005)
    return status, {k.lower(): v for k, v in reply_headers.items()}, body


class RelayContinuityCase(unittest.TestCase):
    """Shared relay/fake-server plumbing (no tests here)."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "run", dict(run_id="relay", episode_id="episode",
                                                              env_instance_id="environment", env_generation=1))
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.reasons = []

    def server(self, cls=FakeAnthropicServer, script=(), **kwargs):
        server = cls(script, **kwargs).start()
        self.addCleanup(server.close)
        return server

    def proxy(self, server, wire, *, upstream_suffix="", **kwargs):
        kwargs.setdefault("request_projection", lambda request: wire.project_request(
            request, self.presentation, registered_tools=frozenset(), tool_name_prefix=PREFIX))
        kwargs.setdefault("on_context_discontinuity", self.reasons.append)
        kwargs.setdefault("max_retries", 0)
        proxy = ModelRequestProxy(server.base_url + upstream_suffix, wire=wire,
                                  upstream_auth_headers={next(iter(wire.auth_header_names.values())): "host-secret"},
                                  **kwargs).start()
        self.addCleanup(proxy.close)
        patch = mock.patch.object(proxy.policy, "_retry_delay_seconds", return_value=0)
        patch.start()
        self.addCleanup(patch.stop)
        return proxy

    @staticmethod
    def events(proxy, kind):
        with proxy.lock:
            return [dict(event) for event in proxy.events if event["kind"] == kind]

    def replay(self, proxy, sequence, rows, *, path, thread_header, base_headers, session_prefix="s"):
        """Send fixture rows of one sequence in order; every answer is the fake's default text."""
        statuses = []
        for k in rows:
            request = sequence["requests"][k]
            headers = base_headers + ((thread_header, session_prefix + "-" + request["thread"]),)
            status, _, _ = post(proxy, path, json.dumps(fixture_body(sequence, request)).encode(), headers)
            statuses.append(status)
        return statuses


class ClaudeCodeContinuityTest(RelayContinuityCase):
    @classmethod
    def setUpClass(cls):
        cls.fixture = load_fixture()
        cls.sequences = {sequence["id"]: sequence for sequence in cls.fixture["sequences"]}

    def replay_claude(self, proxy, sequence, rows, session_prefix="s"):
        return self.replay(proxy, sequence, rows, path="/v1/messages?beta=true",
                           thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS,
                           session_prefix=session_prefix)

    def test_replayed_compaction_triggers_exactly_one_discontinuity(self):
        sequence = self.sequences[COMPACTION_SEQUENCE]
        labels = [request["label"] for request in sequence["requests"]]
        first, second = [k for k, label in enumerate(labels) if label == "compaction_request"]
        lost = labels.index("context_lost")
        self.assertEqual(lost, first + 1)
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire())
        statuses = self.replay_claude(proxy, sequence, range(second))  # one compaction, stop before the next
        self.assertEqual(set(statuses), {200})
        events = self.events(proxy, "model_context_discontinuity")
        self.assertEqual(len(events), 1, events)
        event = events[0]
        self.assertEqual(event["reason"], "history_shortened")
        self.assertEqual(event["logical_call_index"], lost + 1)
        self.assertEqual(event["native_thread_id"], "s-t1")
        self.assertEqual(event["call_purpose"], "main")
        self.assertEqual(self.reasons, ["history_shortened"])
        self.assertEqual(proxy.image_context_epoch, event["logical_call_id"])
        # Claude Code's own summary request carries tools and only appends: continuous, main.
        started = {e["logical_call_index"]: e for e in self.events(proxy, "model_call_completed")}
        self.assertEqual(started[first + 1]["call_purpose"], "main")
        self.assertEqual(proxy.request_count, second)
        # Digests are internal comparison keys: they never reach events.
        recorded = json.dumps(proxy.events)
        for key in ("fingerprints", "system_key", "image_count"):
            self.assertNotIn('"%s"' % key, recorded)

    def test_full_compaction_sequence_reports_each_compaction_once(self):
        sequence = self.sequences[COMPACTION_SEQUENCE]
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire())
        self.replay_claude(proxy, sequence, range(len(sequence["requests"])))
        lost = [k + 1 for k, request in enumerate(sequence["requests"]) if request["label"] == "context_lost"]
        events = self.events(proxy, "model_context_discontinuity")
        self.assertEqual([event["logical_call_index"] for event in events], lost)
        self.assertEqual(self.reasons, ["history_shortened"] * 2)
        self.assertEqual(proxy.image_context_epoch, events[-1]["logical_call_id"])

    def test_normal_multi_step_sessions_trigger_none(self):
        """Every Claude Code spike session without compaction, prune or undelivered answer (tool rounds, images,
        parallel calls, steering merges, title helper calls, cache-marker moves) replays without a discontinuity."""
        clean = [sequence for sequence in self.fixture["sequences"] if sequence["client"] == "claude_code"
                 and all(r["delivered"] and r["status"] == 200 and r["label"] in ("continuous", "helper")
                         for r in sequence["requests"])]
        self.assertGreaterEqual(len(clean), 40)
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire(), max_requests=180 * 4)
        total = helpers = 0
        for index, sequence in enumerate(clean):
            self.replay_claude(proxy, sequence, range(len(sequence["requests"])), session_prefix="seq%d" % index)
            total += len(sequence["requests"])
            helpers += sum(request["label"] == "helper" for request in sequence["requests"])
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        self.assertEqual(self.reasons, [])
        self.assertIsNone(proxy.image_context_epoch)
        completed = self.events(proxy, "model_call_completed")
        self.assertEqual(len(completed), total)
        self.assertEqual(sum(event["call_purpose"] == "helper" for event in completed), helpers)
        self.assertGreater(helpers, 0)

    def tool_session(self, steps):
        """A Claude-Code-shaped history after ``steps`` tool rounds, one tool-result image per round."""
        messages = [dict(role="user", content=[dict(type="text", text="observe the scene")])]
        for k in range(steps):
            call = "toolu_%02d" % k
            messages.append(dict(role="assistant", content=[dict(type="tool_use", id=call, name=PREFIX + "robot_observe",
                                                                input=dict(step=k))]))
            messages.append(dict(role="user", content=[dict(type="tool_result", tool_use_id=call, content=[
                dict(type="text", text=json.dumps(dict(observation_id="obs-%d" % k))),
                dict(type="image", source=dict(type="base64", media_type="image/png", data=PNG))])]))
        return dict(model="fixture-model", max_tokens=1024, stream=True, system="SYSTEM", messages=messages,
                    tools=[dict(name=PREFIX + "robot_observe", input_schema=dict(type="object"))])

    def test_relay_image_trim_does_not_trigger(self):
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire(), max_inline_images=24)
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        for steps in range(1, 33):
            status, _, _ = post(proxy, "/v1/messages", json.dumps(self.tool_session(steps)).encode(), headers)
            self.assertEqual(status, 200)
        self.assertTrue(self.events(proxy, "model_images_trimmed"))
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        self.assertEqual(self.reasons, [])
        # Negative control: the bodies the relay forwarded (trimmed) would have been flagged had the tracker read them.
        wire, tracker, flagged = AnthropicMessagesWire(), ContinuityTracker(), []
        for recorded in server.generation_requests():
            inputs = wire.continuity_inputs(recorded.json)
            flagged.append(tracker.observe("t", **inputs))
            tracker.commit("t", **inputs)
        self.assertTrue(any(flagged), flagged)

    def test_baseline_advances_only_on_usable_delivery(self):
        def body(*texts):
            messages = []
            for k, text in enumerate(texts):
                messages.append(dict(role="user" if k % 2 == 0 else "assistant", content=text))
            return json.dumps(dict(model="fixture-model", max_tokens=64, stream=True, messages=messages,
                                   tools=[dict(name="t", input_schema=dict(type="object"))])).encode()
        rejected = Step(failure=Failure("http", 400, "fixture-private rejection", error_type="invalid_request_error"))
        server = self.server(script=[Step(text="a1"), rejected, Step(text="a2")])
        proxy = self.proxy(server, AnthropicMessagesWire())
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        self.assertEqual(post(proxy, "/v1/messages", body("u1"), headers)[0], 200)
        # A rewritten history is reported at admission, but its failed answer never becomes the baseline.
        status, _, failed = post(proxy, "/v1/messages", body("u0", "x", "u1b"), headers)
        self.assertEqual(status, 200)  # early stream: the failure is one error event in the open stream
        self.assertIn(b"event: error", failed)
        self.assertTrue(proxy.transport_blocked)
        self.assertEqual(self.reasons, ["history_rewritten"])
        proxy.manual_retry()
        # Continuing the delivered history is continuous (it would be a rewrite against the failed request).
        self.assertEqual(post(proxy, "/v1/messages", body("u1", "a1", "u2"), headers)[0], 200)
        self.assertEqual(self.reasons, ["history_rewritten"])
        self.assertEqual(len(self.events(proxy, "model_context_discontinuity")), 1)

    def test_provider_context_edit_on_a_usable_answer_starts_an_epoch(self):
        class EditingServer(FakeAnthropicServer):
            def stream_parts(self, turn):
                head, body, terminal, error = super().stream_parts(turn)
                delta = {"type": "message_delta", "delta": {"stop_reason": turn.stop, "stop_sequence": None},
                         "usage": {"output_tokens": 1},
                         "context_management": {"applied_edits": [{"type": "clear_tool_uses_20250919"}]}}
                return head, body, [sse("message_delta", delta), terminal[-1]], error
        server = self.server(cls=EditingServer)
        proxy = self.proxy(server, AnthropicMessagesWire())
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        status, _, _ = post(proxy, "/v1/messages", json.dumps(self.tool_session(1)).encode(), headers)
        self.assertEqual(status, 200)
        events = self.events(proxy, "model_context_discontinuity")
        self.assertEqual([event["reason"] for event in events], [PROVIDER_CONTEXT_EDIT])
        self.assertEqual(self.reasons, [PROVIDER_CONTEXT_EDIT])
        self.assertEqual(proxy.image_context_epoch, events[0]["logical_call_id"])

    def test_callback_failure_is_recorded_and_never_fails_the_call(self):
        def explode(reason):
            raise RuntimeError("session callback bug")
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire(), on_context_discontinuity=explode)
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        self.assertEqual(post(proxy, "/v1/messages", json.dumps(self.tool_session(2)).encode(), headers)[0], 200)
        self.assertEqual(post(proxy, "/v1/messages", json.dumps(self.tool_session(1)).encode(), headers)[0], 200)
        failures = self.events(proxy, "model_context_discontinuity_callback_failed")
        self.assertEqual([(event["reason"], event["error_type"]) for event in failures],
                         [("history_shortened", "RuntimeError")])
        self.assertEqual(proxy.request_count, 2)
        self.assertFalse(proxy.transport_blocked)

    def test_discontinuity_resets_the_thread_trim_window_before_trimming(self):
        server = self.server()
        proxy = self.proxy(server, AnthropicMessagesWire(), max_inline_images=24)
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        seen = []
        original = proxy._prepare_responses_body

        def spy(state, *args, **kwargs):
            with proxy.lock:
                seen.append(proxy._image_trim_watermarks.get(SESSION))
            return original(state, *args, **kwargs)
        with mock.patch.object(proxy, "_prepare_responses_body", side_effect=spy):
            for steps in (30, 31, 3):
                post(proxy, "/v1/messages", json.dumps(self.tool_session(steps)).encode(), headers)
        self.assertIsNone(seen[0])
        self.assertIsNotNone(seen[1])
        self.assertIsNone(seen[2])  # reset by the discontinuity before the relay looked at the shortened body
        self.assertEqual(self.reasons, ["history_shortened"])


class SubagentChainRelayTest(RelayContinuityCase):
    """Subagent chains through the real relay, wired as the sessions wire it (``default_thread_id`` + actor thread)."""

    @classmethod
    def setUpClass(cls):
        cls.sequences = {sequence["id"]: sequence for sequence in load_fixture()["sequences"]}

    def actor_proxy(self, wire, *, default=True, **kwargs):
        server = self.server()
        proxy = self.proxy(server, wire, default_thread_id="s-t1" if default else None, **kwargs)
        proxy.register_thread("s-t1", "actor", "s-t1")
        return proxy

    def purposes(self, proxy, kind="model_call_completed"):
        return {event["logical_call_index"]: event.get("call_purpose") for event in self.events(proxy, kind)}

    def expected(self, sequence):
        """``(main context_lost call indexes, {call index: expected call_purpose})``."""
        lost, purposes = [], {}
        for k, request in enumerate(sequence["requests"]):
            purposes[k + 1] = ("helper" if request["label"] == "helper" else
                               "subagent" if request["chain"] == "subagent" else "main")
            if request["chain"] == "main" and request["label"] == "context_lost":
                lost.append(k + 1)
        return lost, purposes

    def test_claude_task_subagent_interleaved_with_the_main_compaction(self):
        from cortex.relay.test_context_signal import subagent_interleaving
        sequence = subagent_interleaving(self.sequences[COMPACTION_SEQUENCE],
                                         self.sequences["claude_code/s5-normal/model-requests"])
        proxy = self.actor_proxy(AnthropicMessagesWire())
        statuses = self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages?beta=true",
                               thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS)
        self.assertEqual(set(statuses), {200})
        lost, purposes = self.expected(sequence)
        events = self.events(proxy, "model_context_discontinuity")
        self.assertEqual([event["logical_call_index"] for event in events], lost)
        self.assertEqual(len(lost), 2)
        self.assertEqual({event["call_purpose"] for event in events}, {"main"})
        self.assertEqual(self.reasons, ["history_shortened"] * 2)
        self.assertEqual(proxy.image_context_epoch, events[-1]["logical_call_id"])
        self.assertEqual(self.purposes(proxy), purposes)
        self.assertEqual(sum(purpose == "subagent" for purpose in purposes.values()), 24)
        self.assertEqual(proxy.request_count, len(sequence["requests"]))  # subagent calls are charged too
        self.assertEqual({event["native_thread_id"] for event in self.events(proxy, "model_call_completed")}, {"s-t1"})
        # Which events carry call_purpose: every event after the admission, none before the body is decoded.
        sub = next(index for index, purpose in purposes.items() if purpose == "subagent")
        for kind in ("model_call_started", "model_request_body_received"):
            self.assertNotIn("call_purpose", self.events(proxy, kind)[sub - 1])
        for kind in ("model_request_started", "model_request_finished", "model_call_completed", "model_call_finished"):
            self.assertEqual(self.purposes(proxy, kind)[sub], "subagent", kind)

    def test_captured_subagent_sessions(self):
        """Every R4-0 sequence (two subagents in one session, foreground and background, interrupted task trees)."""
        probe = [s for s in self.sequences.values() if s.get("source") == "r4-0"]
        self.assertEqual(len(probe), 4)
        for sequence in probe:
            with self.subTest(sequence=sequence["id"]):
                self.reasons = []
                proxy = self.actor_proxy(AnthropicMessagesWire())
                statuses = self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages?beta=true",
                                       thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS)
                self.assertEqual(set(statuses), {200})
                self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
                self.assertEqual(self.reasons, [])
                expected = {k + 1: "subagent" if r["label"] == "subagent" else "main"
                            for k, r in enumerate(sequence["requests"])}
                self.assertEqual(self.purposes(proxy), expected)

    def test_claude_main_compaction_with_captured_subagent_requests(self):
        from cortex.relay.test_context_signal import subagent_interleaving
        probe = self.sequences["claude_code/r4-0/cc-agent/model-requests"]
        sub = dict(probe, requests=[r for r in probe["requests"] if r["label"] == "subagent"])
        sequence = subagent_interleaving(self.sequences[COMPACTION_SEQUENCE], sub, sub_system=None,
                                         task_prompt=False, start=4, burst=1)
        proxy = self.actor_proxy(AnthropicMessagesWire())
        self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages?beta=true",
                    thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS)
        lost, purposes = self.expected(sequence)
        self.assertEqual([e["logical_call_index"] for e in self.events(proxy, "model_context_discontinuity")], lost)
        self.assertEqual(self.reasons, ["history_shortened"] * 2)
        self.assertEqual(self.purposes(proxy), purposes)

    def test_claude_subagent_compaction_never_starts_an_epoch(self):
        from cortex.relay.test_context_signal import subagent_interleaving
        sequence = subagent_interleaving(self.sequences["claude_code/s5-normal/model-requests"],
                                         self.sequences[COMPACTION_SEQUENCE], start=1, burst=1)
        proxy = self.actor_proxy(AnthropicMessagesWire())
        self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages",
                    thread_header="X-Claude-Code-Session-Id", base_headers=CLAUDE_HEADERS)
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        self.assertEqual(self.reasons, [])
        self.assertIsNone(proxy.image_context_epoch)
        _, purposes = self.expected(sequence)
        self.assertEqual(self.purposes(proxy), purposes)
        self.assertEqual(sum(purpose == "subagent" for purpose in purposes.values()), 26)

    def test_opencode_child_session_is_a_subagent(self):
        """A child session (own x-opencode-session-id, the main session's system text) compacts its history while the
        main session compacts (the captured 2.2.7 compaction as both). The main thread is ``default_thread_id`` or
        only a registered actor thread."""
        from cortex.relay.test_context_signal import MAIN_SYSTEM, subagent_interleaving
        compact = self.sequences["opencode/compact-anthropic"]
        sequence = subagent_interleaving(compact, compact, sub_thread="t-child", sub_system=MAIN_SYSTEM)
        for default in (True, False):
            with self.subTest(default_thread_id=default):
                self.reasons = []
                proxy = self.actor_proxy(AnthropicMessagesWire(client="opencode"), default=default)
                statuses = self.replay(proxy, sequence, range(len(sequence["requests"])), path="/v1/messages",
                                       thread_header="x-opencode-session-id", base_headers=CLAUDE_HEADERS)
                self.assertEqual(set(statuses), {200})
                lost, purposes = self.expected(sequence)
                events = self.events(proxy, "model_context_discontinuity")
                self.assertEqual([event["logical_call_index"] for event in events], lost)
                self.assertEqual(len(lost), 3)
                self.assertEqual(self.reasons, ["history_shortened"] * 3)
                self.assertEqual(self.purposes(proxy), purposes)
                self.assertEqual(sum(purpose == "subagent" for purpose in purposes.values()), 42)
                # Attribution is unchanged: an unregistered child id falls back to default_thread_id when one is set.
                threads = {event["logical_call_index"]: event["native_thread_id"]
                           for event in self.events(proxy, "model_call_completed")}
                child = {threads[index] for index, purpose in purposes.items() if purpose == "subagent"}
                self.assertEqual(child, {"s-t1"} if default else {"s-t-child"})

    def test_provider_context_edit_on_a_subagent_answer_is_not_a_discontinuity(self):
        class EditingServer(FakeAnthropicServer):
            """Reports a server-side context edit on every answer to a request whose system text says EDIT."""
            def stream_parts(self, turn):
                head, body, terminal, error = super().stream_parts(turn)
                if "EDIT" not in json.dumps((turn.req.json or {}).get("system")):
                    return head, body, terminal, error
                delta = {"type": "message_delta", "delta": {"stop_reason": turn.stop, "stop_sequence": None},
                         "usage": {"output_tokens": 1},
                         "context_management": {"applied_edits": [{"type": "clear_tool_uses_20250919"}]}}
                return head, body, [sse("message_delta", delta), terminal[-1]], error
        server = self.server(cls=EditingServer)
        proxy = self.proxy(server, AnthropicMessagesWire(), default_thread_id=SESSION)
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        tools = [dict(name=PREFIX + "robot_observe", input_schema=dict(type="object"))]

        def body(system, *texts):
            messages = [dict(role="user" if k % 2 == 0 else "assistant", content=text) for k, text in enumerate(texts)]
            return json.dumps(dict(model="fixture-model", max_tokens=64, stream=True, system=system,
                                   messages=messages, tools=tools)).encode()
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN", "u1"), headers)[0], 200)
        self.assertEqual(post(proxy, "/v1/messages", body("SUBAGENT EDIT", "task"), headers)[0], 200)
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN", "u1", "done", "u2"), headers)[0], 200)
        self.assertEqual(list(self.purposes(proxy).values()), ["main", "subagent", "main"])
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        self.assertEqual(self.reasons, [])
        # The same edit on a main answer is one (the P1-3a behaviour).
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN EDIT", "u1", "done", "u2", "a", "u3"), headers)[0], 200)
        self.assertEqual([event["reason"] for event in self.events(proxy, "model_context_discontinuity")],
                         [cs.SYSTEM_CHANGED, PROVIDER_CONTEXT_EDIT])

    def test_claude_agent_id_header_isolates_cold_start_and_same_system_forks(self):
        proxy = self.actor_proxy(AnthropicMessagesWire(), default=True)
        tools = [dict(name=PREFIX + "robot_observe", input_schema=dict(type="object"))]

        def body(system, *texts):
            messages = [dict(role="user" if k % 2 == 0 else "assistant", content=text) for k, text in enumerate(texts)]
            return json.dumps(dict(model="fixture-model", max_tokens=64, stream=True, system=system,
                                   messages=messages, tools=tools)).encode()

        main_headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", "s-t1"),)
        fork_headers = main_headers + (("x-claude-code-agent-id", "agent-fork-1"),)

        # 1. 冷启动时带 x-claude-code-agent-id 的子链先到达，不得抢占主链基线
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN", "fork-first"), fork_headers)[0], 200)
        # 2. 主链首次请求正常建立基线且无虚假不连续
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN", "u1"), main_headers)[0], 200)
        # 3. 相同 system_key 的 fork 子链即便历史完全不同，仍被标为 subagent 且不污染主基线
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN", "fork-divergent"), fork_headers)[0], 200)
        # 4. 主链继续延伸保持连续
        self.assertEqual(post(proxy, "/v1/messages", body("MAIN", "u1", "a1", "u2"), main_headers)[0], 200)

        self.assertEqual(list(self.purposes(proxy).values()), ["subagent", "main", "subagent", "main"])
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        self.assertEqual(self.reasons, [])

    def test_opencode_system_key_none_avoids_sticky_blindness_on_compaction_with_system_change(self):
        for wire_name, path, server_cls in (
            ("anthropic_messages", "/v1/messages", FakeAnthropicServer),
            ("openai_responses", "/v1/responses", FakeResponsesServer),
        ):
            with self.subTest(wire=wire_name):
                self.reasons = []
                server = self.server(server_cls)
                wire = wire_from_name(wire_name, client="opencode")
                proxy = self.proxy(server, wire, default_thread_id="oc-main")
                proxy.register_thread("oc-main", "actor", "oc-main")
                headers = CLAUDE_HEADERS + (("x-opencode-session-id", "oc-main"),)

                def make_body(system_text, *texts):
                    if wire_name == "openai_responses":
                        items = [dict(role="user" if k % 2 == 0 else "assistant", content=text) for k, text in enumerate(texts)]
                        return json.dumps(dict(
                            model="fixture-model", stream=True, instructions=system_text, input=items,
                            tools=[dict(type="function", name="ni_rua_look", parameters=dict(type="object"))],
                        )).encode()
                    messages = [dict(role="user" if k % 2 == 0 else "assistant", content=text) for k, text in enumerate(texts)]
                    return json.dumps(dict(
                        model="fixture-model", max_tokens=64, stream=True, system=system_text, messages=messages,
                        tools=[dict(name="ni_rua_look", input_schema=dict(type="object"))],
                    )).encode()

                # 主会话两轮正常增长
                self.assertEqual(post(proxy, path, make_body("SYS-V1", "u1", "a1", "u2"), headers)[0], 200)
                # 压缩同时伴随 system 文本更新：因 system_key=None，必须识别为 main + history_shortened 并推进基线
                self.assertEqual(post(proxy, path, make_body("SYS-V2", "summary-1"), headers)[0], 200)
                # 新基线上继续增长再发生第二次压缩，必须再次检出（绝不粘滞失明）
                self.assertEqual(post(proxy, path, make_body("SYS-V2", "summary-1", "a2", "u3"), headers)[0], 200)
                self.assertEqual(post(proxy, path, make_body("SYS-V3", "summary-2"), headers)[0], 200)

                self.assertEqual(list(self.purposes(proxy).values()), ["main", "main", "main", "main"])
                self.assertEqual(self.reasons, [cs.HISTORY_SHORTENED, cs.HISTORY_SHORTENED])


class EmittedToolCallsRelayTest(RelayContinuityCase):
    """``on_tool_calls``: the relay names the tool calls of a usable actor completion, in emitted order, before any
    byte of that completion is released to the client."""
    BODY = dict(model="fixture-model", max_tokens=64, stream=True, system="SYSTEM",
                messages=[dict(role="user", content=[dict(type="text", text="go")])],
                tools=[dict(name=PREFIX + "robot_move", input_schema=dict(type="object"))])
    STEP = Step(tool_calls=[ToolCall(PREFIX + "robot_move", {"x": 1}, "toolu_a"), ToolCall("Bash", {"command": "ls"}, "toolu_b"),
                            ToolCall(PREFIX + "robot_move", {"x": 2}, "toolu_c")])
    CALLS = [("toolu_a", PREFIX + "robot_move", {"x": 1}), ("toolu_b", "Bash", {"command": "ls"}),
             ("toolu_c", PREFIX + "robot_move", {"x": 2})]

    def test_calls_are_announced_with_the_thread_key_before_the_answer_is_released(self):
        """The client's socket is read directly while the callback blocks: no byte of the answer has arrived (an
        early-stream profile has only its 200 head and keepalive frames). Announcing after the write fails here."""
        for client, header in (("claude_code", "X-Claude-Code-Session-Id"), ("opencode", "x-opencode-session-id")):
            for streaming in (True, False):
                with self.subTest(client=client, streaming=streaming):
                    entered, release, seen = threading.Event(), threading.Event(), []

                    def on_tool_calls(thread, calls):
                        seen.append((thread, calls))
                        entered.set()
                        release.wait(10)
                    self.addCleanup(release.set)
                    server = self.server(script=[self.STEP])
                    wire = AnthropicMessagesWire(client=client)
                    proxy = self.proxy(server, wire, on_tool_calls=on_tool_calls)
                    body = json.dumps(dict(self.BODY, stream=streaming)).encode()
                    head = "".join(f"{key}: {value}\r\n" for key, value in (
                        *CLAUDE_HEADERS, (header, SESSION), ("Host", "127.0.0.1"), ("Content-Length", len(body))))
                    connection = socket.create_connection(("127.0.0.1", int(proxy.base_url.rsplit(":", 1)[1])), 10)
                    self.addCleanup(connection.close)
                    connection.sendall(f"POST /v1/messages HTTP/1.1\r\n{head}\r\n".encode() + body)
                    self.assertTrue(entered.wait(10))
                    # Everything the relay wrote so far, read until the socket has been quiet for 0.3 s.
                    connection.settimeout(0.3)
                    early = b""
                    with self.assertRaises(socket.timeout):
                        while True:
                            chunk = connection.recv(65536)
                            self.assertTrue(chunk, "the relay closed the connection inside the callback")
                            early += chunk
                    if streaming and wire.early_stream:
                        _, _, frames = early.partition(b"\r\n\r\n")
                        self.assertTrue(early.startswith(b"HTTP/1.0 200") or early.startswith(b"HTTP/1.1 200"), early)
                        self.assertEqual(frames.replace(wire.keepalive_frame(), b""), b"", frames)
                    else:
                        self.assertEqual(early, b"")
                    self.assertEqual(self.events(proxy, "model_call_completed"), [])
                    release.set()
                    connection.settimeout(10)
                    rest = b""
                    while chunk := connection.recv(65536):
                        rest += chunk
                    self.assertIn(b" 200", (early + rest).split(b"\r\n", 1)[0])
                    self.assertEqual(seen, [(SESSION, self.CALLS)])
                    self.assertIn(b"toolu_c", rest)
                    self.assertNotIn(b"toolu_", early)

    def test_no_thread_key_uses_the_default_thread_and_an_answer_without_tool_calls_is_announced_empty(self):
        seen = []
        server = self.server(script=[self.STEP])
        proxy = self.proxy(server, AnthropicMessagesWire(), default_thread_id="thread-default",
                           on_tool_calls=lambda thread, calls: seen.append((thread, calls)))
        self.assertEqual(post(proxy, "/v1/messages", json.dumps(self.BODY).encode(), CLAUDE_HEADERS)[0], 200)
        self.assertEqual(post(proxy, "/v1/messages", json.dumps(self.BODY).encode(), CLAUDE_HEADERS)[0], 200)
        self.assertEqual(seen, [("thread-default", self.CALLS), ("thread-default", [])])

    def test_callback_failure_is_recorded_and_the_answer_is_delivered_unchanged(self):
        def explode(thread, calls):
            raise RuntimeError("session callback bug")
        plain_server = self.server(script=[self.STEP])
        plain = self.proxy(plain_server, AnthropicMessagesWire())
        failing_server = self.server(script=[self.STEP])
        failing = self.proxy(failing_server, AnthropicMessagesWire(), on_tool_calls=explode)
        headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        body = json.dumps(self.BODY).encode()
        expected, delivered = post(plain, "/v1/messages", body, headers), post(failing, "/v1/messages", body, headers)
        self.assertEqual((delivered[0], delivered[2]), (200, expected[2]))
        self.assertEqual([event["error_type"] for event in self.events(failing, "model_tool_call_order_unavailable")],
                         ["RuntimeError"])
        self.assertEqual(self.events(plain, "model_tool_call_order_unavailable"), [])
        self.assertEqual((failing.request_count, failing.transport_blocked), (1, False))

    def test_a_failed_answer_announces_nothing(self):
        seen = []
        server = self.server(script=[Step(failure=Failure("http", 400, "bad request",
                                                          error_type="invalid_request_error"))])
        proxy = self.proxy(server, AnthropicMessagesWire(), on_tool_calls=lambda *args: seen.append(args))
        _, _, raw = post(proxy, "/v1/messages", json.dumps(self.BODY).encode(), CLAUDE_HEADERS)
        self.assertNotIn(b"tool_use", raw)
        self.assertEqual((seen, proxy.request_count), ([], 0))

    def test_three_arg_on_tool_calls_receives_call_purpose_for_main_and_subagent(self):
        seen = []

        def on_tool_calls(thread, calls, purpose=None):
            seen.append((thread, calls, purpose))

        server = self.server(script=[self.STEP, self.STEP])
        proxy = self.proxy(server, AnthropicMessagesWire(), default_thread_id=SESSION, on_tool_calls=on_tool_calls)
        proxy.register_thread(SESSION, "actor", SESSION)
        main_headers = CLAUDE_HEADERS + (("X-Claude-Code-Session-Id", SESSION),)
        sub_headers = main_headers + (("x-claude-code-agent-id", "agent-sub-1"),)
        self.assertEqual(post(proxy, "/v1/messages", json.dumps(self.BODY).encode(), main_headers)[0], 200)
        self.assertEqual(post(proxy, "/v1/messages", json.dumps(self.BODY).encode(), sub_headers)[0], 200)
        self.assertEqual(seen, [(SESSION, self.CALLS, "main"), (SESSION, self.CALLS, "subagent")])


class CodexProfileTest(unittest.TestCase):
    def test_codex_profile_installs_no_tracker(self):
        self.assertIsNone(ModelRequestProxy("http://127.0.0.1:1").continuity)
        self.assertIsNone(ModelRequestProxy("http://127.0.0.1:1", wire=OpenAIResponsesWire(client="codex")).continuity)
        self.assertIsNone(ModelRequestProxy("http://127.0.0.1:1", classify_call_purpose=True).continuity)
        self.assertIsInstance(ModelRequestProxy("http://127.0.0.1:1", wire=AnthropicMessagesWire()).continuity,
                              ContinuityTracker)
        with self.assertRaisesRegex(ValueError, "model_proxy_classify_call_purpose_requires_boolean"):
            ModelRequestProxy("http://127.0.0.1:1", classify_call_purpose="yes")

    def test_codex_profile_classify_call_purpose_without_continuity_tracker(self):
        # Codex 不安装 ContinuityTracker，但在 classify_call_purpose=True 时按统一口径归属 main/helper/subagent，且零字节漂移
        server = FakeResponsesServer().start()
        self.addCleanup(server.close)
        proxy = ModelRequestProxy(
            server.base_url,
            wire=OpenAIResponsesWire(client="codex"),
            upstream_auth_headers={"Authorization": "Bearer host-secret"},
            max_retries=0,
            classify_call_purpose=True,
        ).start()
        self.addCleanup(proxy.close)
        self.assertIsNone(proxy.continuity)
        proxy.register_thread("thread-main", "actor", "thread-main")
        proxy.register_thread("thread-helper", "sol_visual_review", "req-helper-1")

        # 使用非紧凑 JSON 字节验证出站线字节零漂移
        main_raw = (
            b'{"model": "gpt-6.1-sol",  "stream": true, "tools": [{"type": "function", "name": "robot_observe"}], '
            b'"input": [{"role": "user", "content": "stack bowls"}]}'
        )
        helper_raw = json.dumps(dict(
            model="gpt-6.1-sol",
            stream=True,
            tools=[],
            input=[dict(role="user", content="review evidence")],
        )).encode()
        helper_with_tool_raw = json.dumps(dict(
            model="gpt-6.1-sol",
            stream=True,
            tools=[dict(type="function", name="native_stub")],
            input=[dict(role="user", content="review evidence")],
        )).encode()
        compact_raw = json.dumps(dict(
            model="gpt-6.1-sol",
            instructions="base",
            input=[dict(role="user", content="compact")],
        )).encode()
        subagent_raw = json.dumps(dict(
            model="gpt-6.1-sol",
            stream=True,
            tools=[dict(type="function", name="robot_observe")],
            input=[dict(role="user", content="subagent task")],
        )).encode()

        base_headers = (("Content-Type", "application/json"),)
        self.assertEqual(post(proxy, "/responses", main_raw, base_headers + (("thread-id", "thread-main"),))[0], 200)
        self.assertEqual(post(proxy, "/responses", helper_raw, base_headers + (("thread-id", "thread-helper"),))[0], 200)
        self.assertEqual(
            post(proxy, "/responses", helper_with_tool_raw, base_headers + (("thread-id", "thread-helper"),))[0], 200
        )
        self.assertEqual(
            post(proxy, "/responses/compact", compact_raw, base_headers + (("thread-id", "thread-main"),))[0], 200
        )
        self.assertEqual(post(proxy, "/responses", subagent_raw, base_headers + (("thread-id", "thread-sub"),))[0], 200)

        with proxy.lock:
            finished = [dict(e) for e in proxy.events if e["kind"] == "model_request_finished"]
            completed = [dict(e) for e in proxy.events if e["kind"] == "model_call_completed"]
            discontinuities = [dict(e) for e in proxy.events if e["kind"] == "model_context_discontinuity"]
        self.assertEqual([e.get("call_purpose") for e in finished], ["main", "helper", "helper", "helper", "subagent"])
        self.assertEqual([e.get("call_purpose") for e in completed], ["main", "helper", "helper", "helper", "subagent"])
        self.assertEqual(discontinuities, [])
        self.assertEqual(server.generation_requests()[0].body, main_raw)

    def test_adapter_continuity_inputs_delegate_to_context_signal(self):
        from ni_rua.cortex.relay import continuity as cs
        body = dict(system=[dict(type="text", text="S", cache_control=dict(type="ephemeral"))],
                    messages=[dict(role="user", content="hi"),
                              dict(role="user", content=[dict(type="image", source=dict(
                                  type="base64", media_type="image/png", data=base64.b64encode(b"x").decode()))])])
        wire = AnthropicMessagesWire()
        view = cs.anthropic_messages_history(body)
        self.assertEqual(wire.continuity_inputs(body), dict(
            helper=True, fingerprints=cs.history_fingerprints(view), image_count=1,
            system_key=cs.conversation_key(view)))
        self.assertEqual(wire.conversation_key(body), cs.conversation_key(view))
        self.assertEqual(wire.history_fingerprints(body), cs.history_fingerprints(view))
        self.assertEqual(wire.history_image_count(body), 1)
        self.assertFalse(wire.is_helper_request(dict(body, tools=[dict(name="t")])))
        responses = dict(instructions="I", input=[dict(role="developer", content="D"), dict(role="user", content="u")])
        rview = cs.openai_responses_history(responses)
        self.assertEqual(OpenAIResponsesWire().history_fingerprints(responses), cs.history_fingerprints(rview))
        self.assertEqual(OpenAIResponsesWire().conversation_key(responses), cs.conversation_key(rview))

# ==== tests/test_model_proxy_consume_guard.py ====

def _wire_opencode():
    from cortex.relay import test_wire_opencode
    return test_wire_opencode

K = ModelRequestProxy.CONSUME_GUARD_MAX_IDLE_COMPLETIONS
CONSUME_EVENT = "model_client_did_not_consume_completion"
OPENCODE_THREAD = "x-opencode-session-id"
CLAUDE_THREAD = "X-Claude-Code-Session-Id"
TOOLS = [dict(name="mcp__ni_rua__look", description="Read-only view of the scene.", input_schema=dict(type="object"))]
RESPONSES_TOOLS = [dict(type="function", name="ni_rua_look", description="Read-only view of the scene.",
                        parameters=dict(type="object"), strict=False)]


class NullUsageResponses(FakeResponsesServer):
    """Responses provider whose usage carries ``input_tokens: null`` (the T12 trigger)."""

    def _usage(self, u):
        usage = super()._usage(u)
        usage.update(input_tokens=None, total_tokens=None)
        return usage


class NullUsageAnthropic(FakeAnthropicServer):
    """Anthropic provider whose usage carries ``input_tokens: null`` (the T12 trigger)."""

    def _usage(self, u, output=None):
        usage = super()._usage(u, output)
        usage["input_tokens"] = None
        return usage


def delivered_responses_text(body):
    """Assistant text of a delivered Responses event stream (what OpenCode echoes back)."""
    for event in _wire_opencode().sse_events(body):
        if event.get("type") == "response.completed":
            return "".join(part.get("text", "") for item in event["response"]["output"] if item.get("type") == "message"
                           for part in item.get("content") or [])
    return ""


class GuardCase(RelayContinuityCase):
    """Relay plumbing shared by the consume-guard tests (no tests here)."""

    @classmethod
    def setUpClass(cls):
        cls.opencode = json.loads(_wire_opencode().FIXTURE.read_text())["cases"]

    def opencode_headers(self, wire_name, session="ses_fixture"):
        headers = tuple((k, v) for k, v in self.opencode[wire_name]["headers"].items()
                        if k.lower() not in (OPENCODE_THREAD, "x-session-affinity", "x-session-id"))
        return headers + ((OPENCODE_THREAD, session),)

    def send(self, proxy, path, body, headers):
        status, _, data = post(proxy, path, json.dumps(body).encode(), headers)
        return status, data

    def assert_consume_block(self, proxy, server, statuses):
        """The loop was refused right after K idle delivered completions: K + 1 forwarded and charged."""
        self.assertEqual(statuses, [200] * (K + 1) + [statuses[-1]])
        self.assertNotEqual(statuses[-1], 200)
        self.assertEqual(len(server.generation_requests()), K + 1)
        self.assertEqual((proxy.request_count, proxy.reserved_calls), (K + 1, 0))
        evidence = dict(classification="client_did_not_consume_completion", idle_completions=K,
                        max_idle_completions=K)
        self.assertEqual(proxy.blocked_error["error_type"], "ClientDidNotConsumeCompletion")
        self.assertTrue(proxy.blocked_error["manual_retry_allowed"])
        self.assertEqual(proxy.blocked_error["evidence"], evidence)
        event, = self.events(proxy, CONSUME_EVENT)
        self.assertEqual((event["logical_call_index"], event["call_purpose"], event["idle_completions"]),
                         (K + 2, "main", K))
        self.assertEqual(len(self.events(proxy, "model_transport_blocked")), 1)
        self.assertEqual(self.events(proxy, "model_client_retry_runaway"), [])
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])
        finished = {e["logical_call_index"]: e for e in self.events(proxy, "model_call_finished")}
        self.assertFalse(finished[K + 2]["delivered"])


class UnconsumedCompletionLoopTest(GuardCase):
    """The T12 repro: every answer is charged, the client re-requests without acting on it."""

    def test_opencode_responses_null_usage_loop_is_stopped(self):
        server = self.server(NullUsageResponses, default=text_step("NULL-USAGE"))
        proxy = self.proxy(server, wire_from_name("openai_responses", client="opencode"))
        headers = self.opencode_headers("openai_responses")
        body = dict(self.opencode["openai_responses"]["first_body"])
        statuses = []
        for _ in range(20):  # the real client went on until the budget (180) was spent
            status, data = self.send(proxy, "/v1/responses", body, headers)
            statuses.append(status)
            if status != 200:
                break
            # What opencode 1.19.5 sent next: the history plus the delivered text, no new input.
            echoed = dict(role="assistant", content=[dict(type="output_text", text=delivered_responses_text(data))])
            body = dict(body, input=body["input"] + [echoed])
        self.assert_consume_block(proxy, server, statuses)
        self.assertEqual(json.loads(data)["error"]["message"], "ni_rua_client_retry_runaway")
        self.assertEqual(_wire_opencode().opencode_reaction("openai_responses", status, data), "fail")  # one client attempt
        # Blocked: even a request with new input is refused before forwarding.
        follow_up = dict(body, input=body["input"] + [dict(role="user", content="next")])
        status, data = self.send(proxy, "/v1/responses", follow_up, headers)
        self.assertEqual((status, json.loads(data)["error"]["message"]), (400, "ni_rua_model_transport_blocked"))
        self.assertEqual(len(server.generation_requests()), K + 1)
        # The operator reviewed the loop: a manual retry starts a fresh count.
        proxy.manual_retry()
        self.assertEqual(proxy._idle_runs, {})
        self.assertEqual(self.send(proxy, "/v1/responses", follow_up, headers)[0], 200)
        self.assertEqual(proxy.request_count, K + 2)

    def test_opencode_anthropic_unchanged_history_loop_is_stopped(self):
        server = self.server(NullUsageAnthropic, default=text_step("NULL-USAGE"))
        proxy = self.proxy(server, wire_from_name("anthropic_messages", client="opencode"))
        headers = self.opencode_headers("anthropic_messages")
        body = dict(self.opencode["anthropic_messages"]["first_body"])
        statuses = []
        for _ in range(20):
            # What opencode 1.19.5 sent after each unparseable answer: the unchanged history (after a backoff).
            status, data = self.send(proxy, "/v1/messages", body, headers)
            statuses.append(status)
            if status != 200:
                break
        self.assert_consume_block(proxy, server, statuses)
        error = json.loads(data)["error"]
        self.assertEqual((error["type"], error["message"]), ("invalid_request_error", "ni_rua_client_retry_runaway"))
        self.assertEqual(_wire_opencode().opencode_reaction("anthropic_messages", status, data), "fail")

    def test_claude_code_profile_is_guarded_too(self):
        server = self.server(NullUsageAnthropic, default=text_step("NULL-USAGE"))
        proxy = self.proxy(server, AnthropicMessagesWire(client="claude_code"))
        headers = CLAUDE_HEADERS + ((CLAUDE_THREAD, "s-1"),)
        body = dict(model="fixture-model", max_tokens=1024, stream=True, system="SYSTEM", tools=TOOLS,
                    messages=[dict(role="user", content="hello")])
        statuses = []
        for _ in range(20):
            status, data = self.send(proxy, "/v1/messages?beta=true", body, headers)
            statuses.append(status)
            if status != 200:
                break
            echoed = dict(role="assistant", content=[dict(type="text", text="NULL-USAGE")])
            body = dict(body, messages=body["messages"] + [echoed])
        self.assert_consume_block(proxy, server, statuses)
        error = json.loads(data)["error"]
        self.assertEqual((status, error["type"], error["message"]),
                         (403, "permission_error", "ni_rua_client_retry_runaway"))

    def test_codex_profile_has_no_consume_guard(self):
        server = self.server(NullUsageResponses, default=text_step("NULL-USAGE"))
        proxy = ModelRequestProxy(server.base_url, max_retries=0).start()
        self.addCleanup(proxy.close)
        self.assertFalse(proxy.wire.client_loop_guard)
        self.assertIsNone(proxy.continuity)
        body = dict(model="fixture-model", stream=True, instructions="SYSTEM", tools=RESPONSES_TOOLS,
                    input=[dict(role="user", content="hello")])
        for _ in range(K + 4):
            status, data = self.send(proxy, "/responses", body, (("Content-Type", "application/json"),))
            self.assertEqual(status, 200)
            echoed = dict(role="assistant", content=[dict(type="output_text", text=delivered_responses_text(data))])
            body = dict(body, input=body["input"] + [echoed])
        self.assertEqual(proxy.request_count, K + 4)
        self.assertFalse(proxy.transport_blocked)
        self.assertEqual((proxy._idle_runs, self.events(proxy, CONSUME_EVENT)), ({}, []))


class ConsumeRuleTest(GuardCase):
    """What counts as idle, what ends a run, and what is never compared."""

    PATH = "/v1/messages"

    def setUp(self):
        super().setUp()
        self.fake = self.server(FakeAnthropicServer, default=text_step("answer"))

    def opencode_proxy(self, **kwargs):
        return self.proxy(self.fake, wire_from_name("anthropic_messages", client="opencode"), **kwargs)

    def call(self, proxy, messages, *, session="ses_main", tools=True, system="SYSTEM"):
        body = dict(model="fixture-model", max_tokens=1024, stream=True, system=system, messages=messages)
        if tools:
            body["tools"] = TOOLS
        return self.send(proxy, self.PATH, body, self.opencode_headers("anthropic_messages", session))[0]

    @staticmethod
    def answered(messages, text="answer"):
        return messages + [dict(role="assistant", content=[dict(type="text", text=text)])]

    def idle_run(self, proxy, messages, count, **kwargs):
        """``count`` idle re-requests (each echoes the previous answer, no new input); returns the history."""
        for _ in range(count):
            messages = self.answered(messages)
            self.assertEqual(self.call(proxy, messages, **kwargs), 200)
        return messages

    def assert_refused(self, proxy, messages, **kwargs):
        self.assertEqual(self.call(proxy, self.answered(messages), **kwargs), 400)
        self.assertEqual(proxy.blocked_error["error_type"], "ClientDidNotConsumeCompletion")

    def test_only_consecutive_idle_completions_count(self):
        proxy = self.opencode_proxy()
        messages = [dict(role="user", content="hello")]
        self.assertEqual(self.call(proxy, messages), 200)
        messages = self.idle_run(proxy, messages, K - 1)
        messages = self.answered(messages) + [dict(role="user", content="a new prompt")]
        self.assertEqual(self.call(proxy, messages), 200)          # client input: the run ends
        messages = self.idle_run(proxy, messages, K)               # K idle completions are tolerated
        self.assertFalse(proxy.transport_blocked)
        self.assert_refused(proxy, messages)                        # the next idle request is not
        self.assertEqual(proxy.request_count, 1 + (K - 1) + 1 + K)

    def test_client_input_of_every_kind_ends_a_run(self):
        hello = [dict(role="user", content=[dict(type="text", text="hello")])]
        call = dict(type="tool_use", id="toolu_1", name="mcp__ni_rua__look", input={})
        result = dict(type="tool_result", tool_use_id="toolu_1", content="scene: one cube")
        cases = {
            "user_message": lambda m: self.answered(m) + [dict(role="user", content="next")],
            "tool_result": lambda m: m + [dict(role="assistant", content=[call]),
                                          dict(role="user", content=[result])],
            # Claude Code merges a steer into the last user message (blocks appended to a client item).
            "merged_steer": lambda m: m[:-1] + [dict(m[-1], content=m[-1]["content"] + [
                dict(type="text", text="steer")])],
            "compaction": lambda m: [dict(role="user", content="summary of the conversation")],
        }
        for name, add_input in cases.items():
            with self.subTest(case=name):
                proxy = self.opencode_proxy()
                self.assertEqual(self.call(proxy, hello), 200)
                messages = self.idle_run(proxy, hello, K)
                if name == "merged_steer":
                    messages = messages + [dict(role="user", content=[dict(type="text", text="go on")])]
                    self.assertEqual(self.call(proxy, messages), 200)  # a client item: run ends, baseline here
                messages = add_input(messages)
                self.assertEqual(self.call(proxy, messages), 200)
                self.assertFalse(proxy.transport_blocked)
                messages = self.idle_run(proxy, messages, K)
                self.assert_refused(proxy, messages)

    def test_helper_and_subagent_requests_are_not_compared(self):
        proxy = self.opencode_proxy(default_thread_id="ses_main")
        hello = [dict(role="user", content="hello")]
        self.assertEqual(self.call(proxy, hello), 200)
        messages = self.idle_run(proxy, hello, K)
        for _ in range(K + 2):  # helper calls (no tools) with the same history: neither counted nor resetting
            self.assertEqual(self.call(proxy, self.answered(messages), tools=False), 200)
        child = [dict(role="user", content="child task")]  # an OpenCode child session: a subagent chain
        self.assertEqual(self.call(proxy, child, session="ses_child"), 200)
        self.idle_run(proxy, child, K + 2, session="ses_child")
        self.assertFalse(proxy.transport_blocked)
        self.assertEqual(set(proxy._idle_runs), {"ses_main"})
        self.assert_refused(proxy, messages)

    def test_an_undelivered_completion_never_counts(self):
        gate = Gate()
        self.fake = self.server(FakeAnthropicServer, [text_step("answer")] * K + [Step(text="never", gate=gate)],
                                default=text_step("answer"))
        self.addCleanup(gate.release)
        proxy = self.opencode_proxy()
        hello = [dict(role="user", content="hello")]
        self.assertEqual(self.call(proxy, hello), 200)
        messages = self.idle_run(proxy, hello, K - 1)
        held = []
        worker = threading.Thread(target=lambda: held.append(self.call(proxy, self.answered(messages))))
        worker.start()
        self.assertTrue(gate.wait_entered(10))
        proxy.cancel_pending("turn_interrupted")  # the idle request's answer is never delivered
        gate.release()
        worker.join(10)
        self.assertEqual(held, [400])
        self.assertFalse(proxy.transport_blocked)
        self.assertTrue(proxy.reopen_thread("ses_main"))
        self.assertEqual(proxy._idle_runs["ses_main"].idle, K - 1)
        messages = self.idle_run(proxy, messages, 1)  # the K-th idle completion is still tolerated
        self.assert_refused(proxy, messages)

    def test_responses_output_items(self):
        server = self.server(FakeResponsesServer, default=text_step("answer"))
        proxy = self.proxy(server, wire_from_name("openai_responses", client="opencode"))
        headers = self.opencode_headers("openai_responses")
        body = dict(model="fixture-model", stream=True, tools=RESPONSES_TOOLS,
                    input=[dict(role="developer", content="SYSTEM"), dict(role="user", content="hello")])

        def send(items):
            return self.send(proxy, "/v1/responses", dict(body, input=body["input"] + items), headers)[0]

        self.assertEqual(send([]), 200)
        reasoning = dict(type="reasoning", id="rs_1", summary=[], encrypted_content="opaque")
        call = dict(type="function_call", call_id="call_1", name="ni_rua_look", arguments="{}")
        output = dict(type="function_call_output", call_id="call_1", output="scene: one cube")
        tool_round = [reasoning, call, output]
        self.assertEqual(send(tool_round), 200)  # a tool result is client input
        idle = list(tool_round)
        for _ in range(K):  # model output only: reasoning, message and a call without its result
            idle += [reasoning, dict(role="assistant", content=[dict(type="output_text", text="answer")])]
            self.assertEqual(send(idle), 200)
        self.assertEqual(send(idle + [call]), 400)
        self.assertEqual(proxy.blocked_error["error_type"], "ClientDidNotConsumeCompletion")


class NormalSessionsNeverTripTest(GuardCase):
    """False-positive measurement: captured and synthetic normal sessions never produce an idle completion."""

    PROFILES = {
        ("claude_code", "anthropic_messages"): (FakeAnthropicServer, "/v1/messages?beta=true", CLAUDE_THREAD),
        ("opencode", "anthropic_messages"): (FakeAnthropicServer, "/v1/messages", OPENCODE_THREAD),
        ("opencode", "openai_responses"): (FakeResponsesServer, "/v1/responses", OPENCODE_THREAD),
    }

    def spy(self, proxy):
        """Record the consume-guard verdict of every delivered usable main completion."""
        verdicts = []
        original = proxy._commit_idle_run

        def commit(thread, call_purpose, fingerprints, idle):
            verdicts.append((call_purpose, idle))
            return original(thread, call_purpose, fingerprints, idle)

        patch = mock.patch.object(proxy, "_commit_idle_run", side_effect=commit)
        patch.start()
        self.addCleanup(patch.stop)
        return verdicts

    def test_captured_sessions_replay_without_an_idle_completion(self):
        """Every Claude Code and OpenCode sequence of the continuity fixture on a wire the relay serves: its delivered
        requests (the only ones that ever move the relay's state) in capture order, through the real relay with the
        client's own profile. Failed and aborted rows of the spike are left out, because their answers never reached
        the client; the relay answers every replayed row usably."""
        sequences = [s for s in load_fixture()["sequences"] if (s["client"], s["wire"]) in self.PROFILES]
        verdicts, replayed = [], 0
        for (client, wire_name), (server_cls, path, thread_header) in self.PROFILES.items():
            with self.subTest(client=client, wire=wire_name):
                server = self.server(server_cls)
                proxy = self.proxy(server, wire_from_name(wire_name, client=client), max_requests=4000)
                seen = self.spy(proxy)
                base = CLAUDE_HEADERS if client == "claude_code" else tuple(
                    (k, v) for k, v in self.opencode_headers(wire_name) if k.lower() != OPENCODE_THREAD)
                for index, sequence in enumerate(s for s in sequences if (s["client"], s["wire"]) == (client, wire_name)):
                    if wire_name == "openai_responses":
                        sequence = dict(sequence, values=_wire_opencode().rehydrate_images(sequence["values"]))
                    rows = [k for k, request in enumerate(sequence["requests"]) if request["delivered"]]
                    statuses = self.replay(proxy, sequence, rows, path=path, thread_header=thread_header,
                                           base_headers=base, session_prefix="seq%d" % index)
                    self.assertLessEqual(set(statuses), {200}, sequence["id"])  # some sequences deliver nothing
                    replayed += len(rows)
                self.assertFalse(proxy.transport_blocked)
                self.assertEqual(self.events(proxy, CONSUME_EVENT), [])
                verdicts += seen
        main = [idle for purpose, idle in verdicts if purpose == "main"]
        self.assertNotIn(True, main)
        # Measured on the checked-in fixture (OpenCode sequences: the 2.2.7 capture): 381 delivered rows, 370 main
        # completions, 293 of them compared (77 are the first main request of their thread), 0 idle.
        self.assertEqual((replayed, main.count(False), main.count(None)), (381, 293, 77))

    def test_synthetic_tool_loops_never_trip(self):
        """A conformance-style agent loop on every guarded profile: 3 x K tool rounds, a steer as a new user message,
        then a final answer and the next user turn."""
        for (client, wire_name), (server_cls, path, thread_header) in self.PROFILES.items():
            with self.subTest(client=client, wire=wire_name):
                server = self.server(server_cls, default=text_step("answer"))
                proxy = self.proxy(server, wire_from_name(wire_name, client=client))
                seen = self.spy(proxy)
                headers = (CLAUDE_HEADERS if client == "claude_code" else tuple(
                    (k, v) for k, v in self.opencode_headers(wire_name) if k.lower() != OPENCODE_THREAD)
                ) + ((thread_header, "loop"),)
                history = [dict(role="user", content="start")]
                for k in range(3 * K):
                    self.assertEqual(self.send(proxy, path, self.loop_body(wire_name, history), headers)[0], 200)
                    history = history + self.tool_round(wire_name, k)
                    if k == K:
                        history.append(dict(role="user", content="steer: go left"))
                for text in ("final", "next turn"):
                    self.assertEqual(self.send(proxy, path, self.loop_body(wire_name, history), headers)[0], 200)
                    history = history + self.answer(wire_name, text) + [dict(role="user", content=text)]
                self.assertEqual(self.send(proxy, path, self.loop_body(wire_name, history), headers)[0], 200)
                self.assertFalse(proxy.transport_blocked)
                self.assertEqual([idle for _, idle in seen], [None] + [False] * (3 * K + 2))

    @staticmethod
    def loop_body(wire_name, history):
        if wire_name == "openai_responses":
            return dict(model="fixture-model", stream=True, tools=RESPONSES_TOOLS,
                        input=[dict(role="developer", content="SYSTEM")] + history)
        return dict(model="fixture-model", max_tokens=1024, stream=True, system="SYSTEM", tools=TOOLS,
                    messages=history)

    @staticmethod
    def answer(wire_name, text):
        kind = "output_text" if wire_name == "openai_responses" else "text"
        return [dict(role="assistant", content=[dict(type=kind, text=text)])]

    @staticmethod
    def tool_round(wire_name, k):
        if wire_name == "openai_responses":
            return [dict(type="function_call", call_id="call_%d" % k, name="ni_rua_look", arguments="{}"),
                     dict(type="function_call_output", call_id="call_%d" % k, output="scene %d" % k)]
        call = "toolu_%02d" % k
        return [dict(role="assistant", content=[dict(type="tool_use", id=call, name="mcp__ni_rua__look", input={})]),
                dict(role="user", content=[dict(type="tool_result", tool_use_id=call, content="scene %d" % k)])]


class ModelOutputFlagsTest(unittest.TestCase):
    """``WireAdapter.history_model_output_flags``: one flag per conversation item, unknown items are client input."""

    def test_responses(self):
        request = dict(instructions="SYSTEM", input=[
            dict(role="developer", content="SYSTEM"), dict(role="user", content="hello"),
            dict(type="reasoning", summary=[]), dict(role="assistant", content="hi"),
            dict(type="function_call", call_id="c", name="n", arguments="{}"),
            dict(type="function_call_output", call_id="c", output="r"),
            dict(type="custom_tool_call", call_id="d", name="m", input="x"),
            dict(type="custom_tool_call_output", call_id="d", output="r"),
            dict(type="unknown_item"), dict(role="system", content="late system text")])
        for client in ("codex", "opencode"):
            wire = OpenAIResponsesWire(client=client)
            flags = wire.history_model_output_flags(request)
            self.assertEqual(flags, (False, True, True, True, False, True, False, False, False))
            self.assertEqual(len(flags), len(wire.history_fingerprints(request)))
        self.assertEqual(OpenAIResponsesWire().history_model_output_flags(dict(input="hello")), (False,))

    def test_anthropic_messages(self):
        request = dict(system="SYSTEM", messages=[
            dict(role="user", content="hello"), dict(role="assistant", content=[dict(type="text", text="hi")]),
            dict(role="system", content="environment"), "not a message"])
        for client in ("claude_code", "opencode"):
            wire = AnthropicMessagesWire(client=client)
            flags = wire.history_model_output_flags(request)
            self.assertEqual(flags, (False, True, False, False))
            self.assertEqual(len(flags), len(wire.history_fingerprints(request)))

    def test_documented_bound(self):
        # The constant's comment and the class docstring cite this value (an unparseable-answer loop costs K + 1 = 4
        # charged completions); change them together.
        self.assertEqual(K, 3)

if __name__ == "__main__":
    unittest.main()
