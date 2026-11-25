"""Merged cortex tests:
- tests/test_relay_bindings_evidence.py: Relay upstream bindings and per-wire evidence request builders (design §4.1 "Upstream bindings", §9.2-§9.4; P1-7).
- tests/test_relay_evidence_client.py: RelayEvidenceClient (design 04 §9.2-§9.6, P1-8): one fresh, streamed, tool-free evidence request through the run's
"""

NI_RUA_TEST_TIER = "component"

import base64
import http.client
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.core.storage.store import Store
from ni_rua.cortex.relay import server as model_proxy
from ni_rua.cortex.session.codex_evidence_agent import BASE_INSTRUCTIONS
from ni_rua.cortex.relay.server import ModelRequestProxy, UpstreamBinding
from ni_rua.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire
from ni_rua.cortex.session.wire import anthropic_messages as am
from ni_rua.cortex.session.wire import base as wire_base
from ni_rua.cortex.session.wire import openai_responses as orw
from support.fake_model_servers import Failure, FakeAnthropicServer, FakeResponsesServer, Step, ToolCall, text_step
import inspect
import io
import os
import threading
import time
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock, patch
from PIL import Image
from ni_rua.apps.evaluation import queue
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.core.config import AgentConfig, Config, ConfigError, EvidenceConfig
from ni_rua.cortex.agents import evidence as evidence_module
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.base import AgentCapabilities
from ni_rua.cortex.agents.evidence import (BACKEND_LABEL, CONTEXT_POLICY, RelayEvidenceClient, bind_evidence_upstream,
                                           resolve_upstream)
from ni_rua.cortex.session import visual_review
from ni_rua.cortex.agents.codex.session import CodexSession
from ni_rua.cortex.session.codex_evidence_agent import BASE_INSTRUCTIONS, CodexEvidenceAgent
from support.fake_agent import FakeAgentSession
from support.fake_model_servers import (Failure, FakeAnthropicServer, FakeResponsesServer, Step, Usage, text_step,
                                        tool_step)
from support.paths import FIXTURES_ROOT, REPO_ROOT, isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


# ==== tests/test_relay_bindings_evidence.py ====

CAPTURE = FIXTURES_ROOT / "evidence" / "codex_output_schema_request.json"
BIND_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
BIND_DATA_URL = "data:image/png;base64," + BIND_PNG
BIND_DIGEST = "0123456789abcdef" * 4
BIND_SCHEMA = {"type": "object", "additionalProperties": False, "required": ["assessment"],
          "properties": {"assessment": {"type": "string", "enum": ["supported", "uncertain"]}}}
BIND_ITEMS = [dict(type="text", text="evidence text"), dict(type="image", url=BIND_DATA_URL)]
BIND_DEVELOPER = "Use only the supplied evidence packet."
SESSION = "9ff8e771-aa08-4c17-b00d-032e1cb2011f"
CLAUDE_HEADERS = (("Content-Type", "application/json"), ("x-api-key", "ni-rua-local-relay"),
                  ("anthropic-version", "2023-06-01"), ("X-Claude-Code-Session-Id", SESSION))


def build(wire, **overrides):
    kwargs = dict(model="gpt-6.1-sol", base_instructions=BASE_INSTRUCTIONS, developer_instructions=BIND_DEVELOPER,
                  input_items=BIND_ITEMS, output_schema=BIND_SCHEMA, effort="high")
    kwargs.update(overrides)
    return wire.build_evidence_request(**kwargs)


class ResponsesEvidenceBuilderTest(unittest.TestCase):
    WIRE = OpenAIResponsesWire()

    def test_golden_body(self):
        body, headers = build(self.WIRE)
        self.assertEqual(headers, {})
        self.assertEqual(body, {
            "model": "gpt-6.1-sol", "instructions": BASE_INSTRUCTIONS,
            "input": [{"type": "message", "role": "developer", "content": [{"type": "input_text", "text": BIND_DEVELOPER}]},
                      {"type": "message", "role": "user", "content": [
                          {"type": "input_text", "text": "evidence text"},
                          {"type": "input_image", "image_url": BIND_DATA_URL}]}],
            "tools": [],
            "text": {"format": {"type": "json_schema", "name": "evidence_answer", "schema": BIND_SCHEMA, "strict": True}},
            "reasoning": {"effort": "high"}, "store": False, "stream": True})
        self.assertEqual(OpenAIResponsesWire(client="opencode").build_evidence_request(
            model="gpt-6.1-sol", base_instructions=BASE_INSTRUCTIONS, developer_instructions=BIND_DEVELOPER,
            input_items=BIND_ITEMS, output_schema=BIND_SCHEMA, effort="high"), (body, headers))

    def test_mirrors_the_pinned_codex_output_schema_request(self):
        """What the pinned Codex app-server (openai-codex 0.160.0) sent for one CodexEvidenceAgent turn with
        ``outputSchema`` against FakeResponsesServer (captured by fixtures/evidence/capture_codex_output_schema.py)."""
        captured = json.loads(CAPTURE.read_text())
        self.assertEqual(captured["codex_cli_version"], "0.160.0")
        body, _ = build(self.WIRE, output_schema=captured["output_schema"])
        codex_format, relay_format = captured["text"]["format"], body["text"]["format"]
        # The structured-output request the provider sees: same type, same strictness, same schema bytes.
        self.assertEqual(set(codex_format), set(relay_format))
        for key in ("type", "strict", "schema"):
            self.assertEqual(relay_format[key], codex_format[key], key)
        self.assertEqual(body["reasoning"]["effort"], captured["reasoning"]["effort"])
        self.assertEqual((body["store"], body["stream"]), (captured["store"], captured["stream"]))
        evidence_message = captured["input_items"][-1]
        self.assertEqual(evidence_message["role"], "user")
        self.assertEqual([part["type"] for part in body["input"][-1]["content"]],
                         [part["type"] for part in evidence_message["content"]])
        self.assertEqual(evidence_message["content"][1], {"type": "input_image", "image_url": "<data-url>"})
        # Recorded differences (Codex harness context and Codex-specific request fields, design §9.3/§9.6): the
        # relay request carries none of them.
        self.assertEqual(codex_format["name"], "codex_output_schema")
        self.assertEqual(relay_format["name"], "evidence_answer")
        self.assertEqual(captured["text"].get("verbosity"), "high")
        self.assertNotIn("verbosity", body["text"])
        self.assertEqual(captured["reasoning"].get("context"), "all_turns")
        self.assertEqual(set(captured["top_level_keys"]) - set(body),
                         {"client_metadata", "include", "parallel_tool_calls", "prompt_cache_key", "tool_choice"})
        self.assertEqual(set(body) - set(captured["top_level_keys"]), {"instructions", "tools"})
        self.assertFalse(captured["instructions_is_base_instructions"])  # Codex puts its own prompt there
        self.assertEqual(sum(item.get("role") == "developer" for item in captured["input_items"]), 5)

    def test_rejected_items(self):
        for items, error in (([], "evidence_input_item_unsupported"),
                             ([dict(type="audio", data="x")], "evidence_input_item_unsupported"),
                             ([dict(type="image", url="https://example.invalid/a.png")],
                              "evidence_image_requires_inline_data_url"),
                             ([dict(type="image", url="data:text/plain;base64,AAAA")],
                              "evidence_image_requires_inline_data_url")):
            with self.subTest(items=items), self.assertRaisesRegex(ValueError, error):
                build(self.WIRE, input_items=items)

    def test_answer_extraction(self):
        message = dict(type="message", role="assistant", content=[dict(type="output_text", text='{"assessment":'),
                                                                   dict(type="output_text", text='"uncertain"}')])
        reasoning = dict(type="reasoning", summary=[], encrypted_content="x")
        self.assertEqual(self.WIRE.evidence_answer(dict(status="completed", output=[reasoning, message])),
                         '{"assessment":"uncertain"}')
        cases = [(dict(status="incomplete", output=[message]), wire_base.EVIDENCE_UNVERIFIED),
                 (dict(status="failed", output=[]), wire_base.EVIDENCE_UNVERIFIED),
                 (dict(status="completed", output=[message, dict(type="function_call", name="x", arguments="{}")]),
                  wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (dict(status="completed", output=[dict(type="web_search_call")]), wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (dict(status="completed", output=[dict(message, content=[dict(type="refusal", refusal="no")])]),
                  wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (dict(status="completed", output=[reasoning]), wire_base.EVIDENCE_ONE_ANSWER),
                 (dict(status="completed", output=[message, message]), wire_base.EVIDENCE_ONE_ANSWER)]
        for terminal, error in cases:
            with self.subTest(terminal=terminal), self.assertRaisesRegex(ValueError, error):
                self.WIRE.evidence_answer(terminal)


class AnthropicEvidenceBuilderTest(unittest.TestCase):
    WIRE = AnthropicMessagesWire()

    def test_golden_body(self):
        body, headers = build(self.WIRE, model="claude-opus-5-5")
        self.assertEqual(headers, {"anthropic-version": "2023-06-01"})
        self.assertEqual(body, {
            "model": "claude-opus-5-5", "max_tokens": 16000,
            "system": [{"type": "text", "text": BASE_INSTRUCTIONS + "\n\n" + BIND_DEVELOPER}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "evidence text"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": BIND_PNG}}]}],
            "output_config": {"format": {"type": "json_schema", "schema": BIND_SCHEMA}, "effort": "high"},
            "stream": True})
        for absent in ("tools", "tool_choice", "thinking", "temperature", "top_p", "top_k", "output_format"):
            self.assertNotIn(absent, body)
        self.assertEqual(build(AnthropicMessagesWire(client="opencode"), model="claude-opus-5-5"), (body, headers))

    def stream(self, blocks, stop_reason):
        frames = [am.KEEPALIVE_FRAME, f"event: message_start\ndata: {json.dumps(dict(type='message_start', message=dict(id='msg_e', model='claude-served', content=[], stop_reason=None, usage=dict(input_tokens=9, output_tokens=1))))}\n\n".encode()]
        for index, (start, deltas) in enumerate(blocks):
            frames.append(f"event: content_block_start\ndata: {json.dumps(dict(type='content_block_start', index=index, content_block=start))}\n\n".encode())
            for delta in deltas:
                frames.append(f"event: content_block_delta\ndata: {json.dumps(dict(type='content_block_delta', index=index, delta=delta))}\n\n".encode())
            frames.append(f"event: content_block_stop\ndata: {json.dumps(dict(type='content_block_stop', index=index))}\n\n".encode())
        frames.append(f"event: message_delta\ndata: {json.dumps(dict(type='message_delta', delta=dict(stop_reason=stop_reason), usage=dict(output_tokens=4)))}\n\n".encode())
        frames.append(b'event: message_stop\ndata: {"type": "message_stop"}\n\n')
        return b"".join(frames)

    def answer(self, raw):
        _, _, terminal = self.WIRE.response_evidence(raw, self.WIRE.route("POST", "/v1/messages"))
        return self.WIRE.evidence_answer(terminal)

    def test_answer_extraction_from_stream_and_json(self):
        thinking = ({"type": "thinking", "thinking": ""}, [{"type": "signature_delta", "signature": "s"}])
        text = ({"type": "text", "text": ""}, [{"type": "text_delta", "text": '{"assessment":'},
                                               {"type": "text_delta", "text": '"supported"}'}])
        self.assertEqual(self.answer(self.stream([thinking, text], "end_turn")), '{"assessment":"supported"}')
        body = dict(type="message", id="msg_j", model="m", role="assistant", stop_reason="end_turn",
                    content=[dict(type="thinking", thinking="t", signature="s"), dict(type="text", text='{"a":1}')],
                    usage=dict(input_tokens=1, output_tokens=1))
        self.assertEqual(self.answer(json.dumps(body).encode()), '{"a":1}')
        tool = ({"type": "tool_use", "id": "toolu_1", "name": "submit_answer", "input": {}},
                [{"type": "input_json_delta", "partial_json": "{}"}])
        cases = [(self.stream([text], "refusal"), wire_base.EVIDENCE_UNVERIFIED),
                 (self.stream([text], "max_tokens"), wire_base.EVIDENCE_UNVERIFIED),
                 (self.stream([text], "pause_turn"), wire_base.EVIDENCE_UNVERIFIED),
                 (self.stream([text, tool], "end_turn"), wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (self.stream([({"type": "server_tool_use", "id": "s", "name": "web_search", "input": {}}, [])],
                              "end_turn"), wire_base.EVIDENCE_TOOL_ACTIVITY),
                 (self.stream([thinking], "end_turn"), wire_base.EVIDENCE_ONE_ANSWER)]
        for raw, error in cases:
            with self.subTest(error=error, raw=raw[-120:]), self.assertRaisesRegex(ValueError, error):
                self.answer(raw)
        with self.assertRaisesRegex(ValueError, wire_base.EVIDENCE_UNVERIFIED):
            self.WIRE.evidence_answer(dict(status="failed", error={}))


# ------------------------------------------------------------------------------------------------- bindings
class BindingsTest(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(Path(directory.name) / "run", dict(run_id="relay", episode_id="episode",
                                                              env_instance_id="environment", env_generation=1))
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.actor_server = FakeAnthropicServer().start()
        self.evidence_server = FakeResponsesServer(default=text_step('{"assessment":"uncertain"}')).start()
        self.addCleanup(self.actor_server.close)
        self.addCleanup(self.evidence_server.close)

    def proxy(self, **kwargs):
        actor_wire = AnthropicMessagesWire()
        kwargs.setdefault("max_retries", 0)
        kwargs.setdefault("request_projection", lambda request: actor_wire.project_request(
            request, self.presentation, registered_tools=frozenset(), tool_name_prefix="mcp__ni_rua__"))
        proxy = ModelRequestProxy(self.actor_server.base_url, wire=actor_wire,
                                  upstream_auth_headers={"x-api-key": "actor-secret"},
                                  default_thread_id=SESSION, **kwargs).start()
        self.addCleanup(proxy.close)
        patch = mock.patch.object(proxy.policy, "_retry_delay_seconds", return_value=0)
        patch.start()
        self.addCleanup(patch.stop)
        proxy.register_thread(SESSION, "actor", SESSION)
        url = proxy.add_upstream("evidence", wire=OpenAIResponsesWire(), upstream_url=self.evidence_server.base_url,
                                 upstream_auth_headers={"Authorization": "Bearer evidence-secret"},
                                 presentation=self.presentation)
        self.assertEqual(url, proxy.base_url + "/_ni_rua/u/evidence")
        return proxy

    @staticmethod
    def post(base_url, path, body, headers):
        host, port = base_url.split("//", 1)[1].split("/", 1)[0].split(":")
        prefix = "/" + base_url.split("//", 1)[1].split("/", 1)[1] if base_url.count("/") > 2 else ""
        data = json.dumps(body).encode()
        connection = http.client.HTTPConnection(host, int(port), timeout=30)
        try:
            connection.putrequest("POST", prefix + path, skip_accept_encoding=True)
            for key, value in headers:
                connection.putheader(key, value)
            connection.putheader("Content-Length", str(len(data)))
            connection.endheaders(data)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()

    def evidence_call(self, proxy, request_id="r1", items=BIND_ITEMS, headers=()):
        thread = "evidence-" + request_id
        proxy.register_thread(thread, "sol_visual_review", request_id)
        wire = proxy.upstream_binding("evidence").wire
        body, extra = build(wire, input_items=items)
        all_headers = (("Content-Type", "application/json"), ("X-Ni-Rua-Thread", thread),
                       ("X-Ni-Rua-Source", "sol_visual_review"), ("X-Ni-Rua-Request-Id", request_id),
                       *extra.items(), *headers)
        status, raw = self.post(proxy.upstream_base_url("evidence"), "/responses", body, all_headers)
        return status, raw, thread

    def events(self, proxy, kind):
        with proxy.lock:
            return [dict(event) for event in proxy.events if event["kind"] == kind]

    def test_evidence_request_through_a_binding(self):
        proxy = self.proxy()
        planted = [dict(type="text", text="fingerprint sha256=" + BIND_DIGEST), BIND_ITEMS[1]]
        status, raw, thread = self.evidence_call(proxy, items=planted,
                                                 headers=(("Authorization", "Bearer client-copy"),))
        self.assertEqual(status, 200)
        wire = proxy.upstream_binding("evidence").wire
        complete, evidence, terminal = wire.response_evidence(raw.replace(orw.KEEPALIVE_FRAME, b""),
                                                              wire.route("POST", "/responses"))
        self.assertEqual(wire.completion_outcome(complete, evidence), "usable")
        self.assertEqual(json.loads(wire.evidence_answer(terminal)), {"assessment": "uncertain"})
        forwarded = self.evidence_server.generation_requests()
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(self.actor_server.requests, [])
        headers = {k.lower(): v for k, v in forwarded[0].headers.items()}
        self.assertEqual(headers["authorization"], "Bearer evidence-secret")  # client copy replaced
        for private in ("x-api-key", "x-ni-rua-thread", "x-ni-rua-source", "x-ni-rua-request-id"):
            self.assertNotIn(private, headers)
        self.assertNotIn(BIND_DIGEST, forwarded[0].body.decode())  # binding egress projection with its presentation
        self.assertEqual(forwarded[0].json["text"]["format"]["schema"], BIND_SCHEMA)
        self.assertEqual(forwarded[0].json["input"][1]["content"][1], {"type": "input_image", "image_url": BIND_DATA_URL})
        completed = self.events(proxy, "model_call_completed")[0]
        self.assertEqual((completed["upstream"], completed["native_thread_id"], completed["source"],
                          completed["caller_request_id"], completed["endpoint"]),
                         ("evidence", thread, "sol_visual_review", "r1", "/responses"))
        self.assertNotIn("call_purpose", completed)  # no continuity tracking on helper bindings
        self.assertEqual(proxy.request_count, 1)
        self.assertEqual(self.events(proxy, "model_upstream_bound")[0]["credentials"], ["Authorization"])
        self.assertNotIn("evidence-secret", json.dumps(proxy.events))

    def test_shared_budget_and_actor_isolation(self):
        proxy = self.proxy(max_requests=2)
        actor_body = dict(model="claude-opus-5-5", max_tokens=64, stream=True,
                          messages=[dict(role="user", content="hi")], tools=[dict(name="t", input_schema={})])
        status, _ = self.post(proxy.base_url, "/v1/messages", actor_body,
                              CLAUDE_HEADERS + (("Authorization", "Bearer evidence-secret"),))
        self.assertEqual(status, 200)
        actor_headers = {k.lower(): v for k, v in self.actor_server.generation_requests()[0].headers.items()}
        self.assertEqual(actor_headers["x-api-key"], "actor-secret")
        self.assertNotIn("authorization", actor_headers)  # never another binding's credential
        self.assertEqual(self.evidence_call(proxy)[0], 200)
        self.assertEqual(proxy.request_count, 2)
        status, raw, _ = self.evidence_call(proxy, request_id="r2")
        self.assertEqual(status, 429)  # the evidence binding's own (Codex) budget denial
        self.assertEqual(self.post(proxy.base_url, "/v1/messages", actor_body, CLAUDE_HEADERS)[0], 403)
        self.assertEqual(len(self.evidence_server.generation_requests()), 1)
        actor_events = [e for e in self.events(proxy, "model_call_completed") if "upstream" not in e]
        self.assertEqual([e["call_purpose"] for e in actor_events], ["main"])

    def test_blocked_transport_is_shared(self):
        self.evidence_server.default = Step(failure=Failure("http", 400, "provider-private rejection",
                                                            error_type="invalid_request_error"))
        proxy = self.proxy()
        status, raw, _ = self.evidence_call(proxy)
        self.assertEqual(status, 200)  # Codex profile: early stream, the failure is one error event
        self.assertIn(b"event: error", raw)
        self.assertTrue(proxy.transport_blocked)
        self.assertEqual(proxy.blocked_error["evidence"]["http_status"], 400)
        actor_body = dict(model="m", max_tokens=8, stream=True, messages=[dict(role="user", content="x")])
        self.assertEqual(self.post(proxy.base_url, "/v1/messages", actor_body, CLAUDE_HEADERS)[0], 403)
        self.assertEqual(self.actor_server.requests, [])

    def test_cancelled_evidence_thread_is_denied(self):
        proxy = self.proxy()
        proxy.register_thread("evidence-r9", "sol_visual_review", "r9")
        proxy.cancel_thread("evidence-r9")
        body, _ = build(OpenAIResponsesWire())
        status, _ = self.post(proxy.upstream_base_url("evidence"), "/responses", body,
                              (("Content-Type", "application/json"), ("X-Ni-Rua-Thread", "evidence-r9")))
        self.assertEqual(status, 409)
        self.assertEqual(self.evidence_server.requests, [])

    def test_self_briefing_on_the_actor_binding_is_a_helper(self):
        """Self-briefing uses the actor binding (§9.4): no tools, so the tracker attributes it as a helper."""
        proxy = self.proxy()
        proxy.register_thread("evidence-brief", "self_briefing", "brief")
        body, extra = build(AnthropicMessagesWire(), model="claude-opus-5-5")
        self.actor_server.default = text_step('{"assessment":"supported"}')
        status, raw = self.post(proxy.base_url, "/v1/messages", body,
                                (("Content-Type", "application/json"), ("X-Ni-Rua-Thread", "evidence-brief"),
                                 *extra.items()))
        self.assertEqual(status, 200)
        completed = self.events(proxy, "model_call_completed")[0]
        self.assertEqual((completed["native_thread_id"], completed["source"], completed["call_purpose"]),
                         ("evidence-brief", "self_briefing", "helper"))
        self.assertNotIn("upstream", completed)
        forwarded = self.actor_server.generation_requests()[0]
        self.assertNotIn("x-ni-rua-thread", {k.lower() for k in forwarded.headers.keys()})
        self.assertEqual(forwarded.json["output_config"]["format"]["type"], "json_schema")
        wire = AnthropicMessagesWire()
        _, _, terminal = wire.response_evidence(raw.replace(am.KEEPALIVE_FRAME, b""), wire.route("POST", "/v1/messages"))
        self.assertEqual(json.loads(wire.evidence_answer(terminal)), {"assessment": "supported"})

    def test_routing_validation_and_worker_subprocess(self):
        proxy = self.proxy()
        status, _ = self.post(proxy.base_url, "/_ni_rua/u/missing/responses", dict(model="m"), ())
        self.assertEqual(status, 404)
        status, _ = self.post(proxy.base_url, "/_ni_rua/u/evidence/v1/messages", dict(model="m"), ())
        self.assertEqual(status, 404)  # the binding's own routes only (Codex Responses: exact paths)
        for name in ("actor", "Evidence", "", "a" * 40, "with/slash"):
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "model_proxy_invalid_upstream_name"):
                proxy.add_upstream(name, wire=OpenAIResponsesWire(), upstream_url="http://127.0.0.1:1",
                                   upstream_auth_headers=None, presentation=self.presentation)
        with self.assertRaisesRegex(ValueError, "model_proxy_upstream_already_bound"):
            proxy.add_upstream("evidence", wire=OpenAIResponsesWire(), upstream_url="http://127.0.0.1:1",
                               upstream_auth_headers=None, presentation=self.presentation)
        with self.assertRaisesRegex(ValueError, "invalid_upstream_auth_headers"):
            proxy.add_upstream("other", wire=OpenAIResponsesWire(), upstream_url="http://127.0.0.1:1",
                               upstream_auth_headers={"x-api-key": "k"}, presentation=self.presentation)
        with self.assertRaisesRegex(ValueError, "model_proxy_requires_http_base_url"):
            proxy.add_upstream("other", wire=OpenAIResponsesWire(), upstream_url="http://u:p@127.0.0.1:1",
                               upstream_auth_headers=None, presentation=self.presentation)
        with self.assertRaisesRegex(ValueError, "model_proxy_upstream_requires_presentation"):
            proxy.add_upstream("other", wire=OpenAIResponsesWire(), upstream_url="http://127.0.0.1:1",
                               upstream_auth_headers=None, presentation=None)
        with self.assertRaises(KeyError):
            proxy.upstream_base_url("missing")
        self.assertEqual(proxy.upstream_base_url("actor"), proxy.base_url)
        self.assertIsInstance(proxy.upstream_binding("actor"), UpstreamBinding)
        # The killable --http-worker subprocess path rebuilds the binding's wire from its payload.
        with mock.patch.object(model_proxy.subprocess, "Popen", side_effect=subprocess.Popen) as spawned:
            status, raw, _ = self.evidence_call(proxy, request_id="worker")
        self.assertEqual(status, 200)
        self.assertTrue(spawned.called)
        self.assertIn(b"response.completed", raw)
        self.assertEqual(self.evidence_server.generation_requests()[-1].headers["Authorization"],
                         "Bearer evidence-secret")

    def test_helper_binding_skips_image_trim(self):
        proxy = self.proxy(max_inline_images=24)
        items = [dict(type="text", text="frames")] + [dict(type="image", url=BIND_DATA_URL)] * 40
        status, _, _ = self.evidence_call(proxy, items=items)
        self.assertEqual(status, 200)
        self.assertEqual(self.events(proxy, "model_images_trimmed"), [])
        forwarded = self.evidence_server.generation_requests()[0].json
        self.assertEqual(sum(part["type"] == "input_image" for part in forwarded["input"][1]["content"]), 40)

# ==== tests/test_relay_evidence_client.py ====

PNG = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
DATA_URL = "data:image/png;base64," + PNG
DIGEST = "0123456789abcdef" * 4
SCHEMA = {"type": "object", "additionalProperties": False, "required": ["assessment"],
          "properties": {"assessment": {"type": "string", "enum": ["supported", "uncertain"]}}}
ITEMS = [dict(type="text", text="evidence text"), dict(type="image", url=DATA_URL)]
DEVELOPER = "Use only the supplied evidence packet."
CLAUDE_SESSION = "9ff8e771-aa08-4c17-b00d-032e1cb2011f"
ANSWER = '{"assessment":"uncertain"}'
IDENTITY = dict(run_id="relay-evidence", episode_id="episode", env_instance_id="environment", env_generation=1)


def jpeg(color="red"):
    buffer = io.BytesIO()
    Image.new("RGB", (32, 24), color).save(buffer, "JPEG", quality=90)
    return buffer.getvalue()


class RelayCase(unittest.TestCase):
    """A run Store plus helpers to start a relay in front of a fake server."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = Store(self.root / "run", dict(IDENTITY))
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)

    def server(self, cls, *script, default=None):
        server = cls(list(script), default=default or text_step(ANSWER)).start()
        self.addCleanup(server.close)
        return server

    def relay(self, server, wire, *, auth=None, **kwargs):
        kwargs.setdefault("max_retries", 0)
        kwargs.setdefault("request_projection", lambda request: wire.project_request(
            request, self.presentation, registered_tools=frozenset(), tool_name_prefix=""))
        if isinstance(wire, AnthropicMessagesWire) and wire.client == "claude_code":
            kwargs.setdefault("default_thread_id", CLAUDE_SESSION)
        proxy = ModelRequestProxy(server.base_url, wire=wire, upstream_auth_headers=auth, **kwargs).start()
        self.addCleanup(proxy.close)
        retry = mock.patch.object(proxy.policy, "_retry_delay_seconds", return_value=0)
        retry.start()
        self.addCleanup(retry.stop)
        return proxy

    def session(self, proxy):
        return SimpleNamespace(store=self.store, model_proxy=proxy)

    @staticmethod
    def events(proxy, kind):
        with proxy.lock:
            return [dict(event) for event in proxy.events if event["kind"] == kind]

    def store_events(self, kind):
        return [event["payload"] for event in self.store.events(limit=10000) if event["type"] == kind]


# ------------------------------------------------------------------------------------------- request bodies per wire
class RequestBodyTests(RelayCase):
    def test_responses_actor_binding(self):
        server = self.server(FakeResponsesServer, text_step(ANSWER, usage=Usage(
            input_tokens=30, output_tokens=7, cache_read_input_tokens=4, reasoning_tokens=2)))
        proxy = self.relay(server, OpenAIResponsesWire(), auth={"Authorization": "Bearer host-secret"})
        client = RelayEvidenceClient(self.session(proxy))
        planted = [dict(type="text", text="evidence text fingerprint sha256=" + DIGEST), ITEMS[1]]
        result = client.review_evidence(input_items=planted, output_schema=SCHEMA, request_id="r1")
        forwarded, = server.generation_requests()
        presented = ActorPresentation(self.store)
        expected, _ = OpenAIResponsesWire().build_evidence_request(
            model="gpt-6.1-sol", base_instructions=BASE_INSTRUCTIONS, developer_instructions=DEVELOPER,
            input_items=presented.present(planted), output_schema=presented.present_data(SCHEMA), effort="high")
        self.assertEqual(forwarded.json, expected)
        self.assertEqual(forwarded.route, "/responses")
        self.assertNotIn(DIGEST, forwarded.body.decode())  # presented before the request and re-scrubbed at egress
        self.assertEqual((forwarded.json["tools"], forwarded.json["stream"], forwarded.json["store"]), ([], True, False))
        self.assertEqual(forwarded.json["text"]["format"]["name"], "evidence_answer")
        headers = {key.lower(): value for key, value in forwarded.headers.items()}
        self.assertEqual(headers["authorization"], "Bearer host-secret")
        for private in ("x-ni-rua-thread", "x-ni-rua-source", "x-ni-rua-request-id"):
            self.assertNotIn(private, headers)
        tokens = dict(input_tokens=34, cached_input_tokens=4, cache_write_input_tokens=None, output_tokens=7,
                      reasoning_output_tokens=2, total_tokens=41)
        self.assertEqual(result, dict(
            model="gpt-6.1-sol", status="completed", answer={"assessment": "uncertain"},
            usage=dict(total=tokens, last=tokens, model_context_window=None), thread_id="evidence-r1",
            turn_id="resp_fake_0", served_model="gpt-6.1-sol", context_policy=CONTEXT_POLICY, backend=BACKEND_LABEL,
            wire_api="openai_responses", upstream="actor",
            model_verification="requested_model_recorded_served_model_reported",
            runtime_policy="single_stateless_request_no_session_no_memory_v1",
            tool_policy="no_tools_declared_reject_any_tool_output_v1"))
        completed, = self.events(proxy, "model_call_completed")
        self.assertEqual((completed["native_thread_id"], completed["source"], completed["caller_request_id"],
                          completed["endpoint"]), ("evidence-r1", "sol_visual_review", "r1", "/responses"))
        self.assertNotIn("upstream", completed)
        self.assertEqual(proxy.request_count, 1)
        self.assertEqual([event["native_thread_id"] for event in self.events(proxy, "model_thread_unregistered")],
                         ["evidence-r1"])
        self.assertNotIn("evidence-r1", proxy.thread_owners)

    def test_anthropic_actor_binding_helper_call(self):
        helper_dev = "Summarize the workspace layout in one sentence."
        helper_schema = {
            "type": "object",
            "properties": {"notes": {"type": "string"}},
            "required": ["notes"],
            "additionalProperties": False,
        }
        server = self.server(FakeAnthropicServer, text_step('{"notes":"Grasp the cup first."}', usage=Usage(
            input_tokens=20, output_tokens=6, cache_read_input_tokens=3, cache_creation_input_tokens=2)))
        proxy = self.relay(server, AnthropicMessagesWire(), auth={"x-api-key": "host-key"})
        proxy.register_thread(CLAUDE_SESSION, "actor", CLAUDE_SESSION)
        client = RelayEvidenceClient(self.session(proxy), model="claude-opus-5-5",
                                     base_instructions=BASE_INSTRUCTIONS,
                                     developer_instructions=helper_dev, source="scene_helper")
        result = client.review_evidence(input_items=ITEMS, output_schema=helper_schema, request_id="brief")
        forwarded, = server.generation_requests()
        self.assertEqual(forwarded.route, "/v1/messages")
        self.assertEqual(forwarded.json, {
            "model": "claude-opus-5-5", "max_tokens": 16000,
            "system": [{"type": "text", "text": BASE_INSTRUCTIONS + "\n\n" + helper_dev}],
            "messages": [{"role": "user", "content": [
                {"type": "text", "text": "evidence text"},
                {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": PNG}}]}],
            "output_config": {"format": {"type": "json_schema", "schema": helper_schema}, "effort": "high"},
            "stream": True})
        headers = {key.lower(): value for key, value in forwarded.headers.items()}
        self.assertEqual((headers["x-api-key"], headers["anthropic-version"]), ("host-key", "2023-06-01"))
        self.assertNotIn("x-ni-rua-thread", headers)
        self.assertEqual((result["model"], result["answer"], result["wire_api"], result["thread_id"]),
                         ("claude-opus-5-5", {"notes": "Grasp the cup first."}, "anthropic_messages", "evidence-brief"))
        self.assertEqual(result["usage"]["total"], dict(input_tokens=25, cached_input_tokens=3,
                                                         cache_write_input_tokens=2, output_tokens=6,
                                                         reasoning_output_tokens=None, total_tokens=31))
        completed, = self.events(proxy, "model_call_completed")
        # Not the actor's thread (the default thread key) and attributed as a no-tools helper by the tracker.
        self.assertEqual((completed["native_thread_id"], completed["source"], completed["call_purpose"]),
                         ("evidence-brief", "scene_helper", "helper"))
        self.assertEqual(self.events(proxy, "model_context_discontinuity"), [])

    def test_opencode_profiles_deferred_delivery(self):
        for server_class, wire, path in ((FakeResponsesServer, OpenAIResponsesWire(client="opencode"), "/responses"),
                                         (FakeAnthropicServer, AnthropicMessagesWire(client="opencode"), "/v1/messages")):
            with self.subTest(wire=wire.name):
                server = self.server(server_class)
                proxy = self.relay(server, wire, max_requests=1)
                client = RelayEvidenceClient(self.session(proxy), model="m")
                result = client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="oc1")
                self.assertEqual(result["answer"], {"assessment": "uncertain"})
                self.assertEqual(server.generation_requests()[0].route, path)
                # Budget spent: the deferred profile's denial (HTTP 400) is an ordinary unverified failure.
                with self.assertRaisesRegex(ValueError, "^sol_agent_unverified_or_incomplete_turn$"):
                    client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="oc2")
                self.assertEqual(len(server.generation_requests()), 1)
                unverified = self.store_events("visual.agent_turn_unverified")[-1]
                self.assertEqual((unverified["request_id"], unverified["http_status"], unverified["wire_api"]),
                                 ("oc2", 400, wire.name))


# ------------------------------------------------------------------------------------------------- bindings and budget
class BindingTests(RelayCase):
    def config(self, server, **upstream):
        return Config.from_dict({"agent": {"backend": "claude_code", "evidence": {"upstream": {
            "wire_api": "openai_responses", "base_url": server.base_url, **upstream}}}})

    def test_visual_review_uses_the_evidence_binding_and_shares_the_budget(self):
        actor_server = self.server(FakeAnthropicServer, default=text_step("actor answer"))
        evidence_server = self.server(FakeResponsesServer)
        proxy = self.relay(actor_server, AnthropicMessagesWire(), auth={"x-api-key": "actor-secret"}, max_requests=2)
        proxy.register_thread(CLAUDE_SESSION, "actor", CLAUDE_SESSION)
        session = self.session(proxy)
        config = self.config(evidence_server, api_key_env="EVIDENCE_KEY")
        name = bind_evidence_upstream(session, config, environ={"EVIDENCE_KEY": "evidence-secret"})
        self.assertEqual(name, "evidence")
        self.assertEqual(bind_evidence_upstream(session, config, environ={}), "evidence")  # idempotent
        self.assertEqual(len(self.events(proxy, "model_upstream_bound")), 1)
        client = RelayEvidenceClient(session, upstream=name)
        result = client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="v1")
        self.assertEqual((result["upstream"], result["wire_api"], result["model"]),
                         ("evidence", "openai_responses", "gpt-6.1-sol"))
        self.assertEqual(actor_server.requests, [])
        forwarded, = evidence_server.generation_requests()
        self.assertEqual(forwarded.headers["Authorization"], "Bearer evidence-secret")
        self.assertNotIn("x-api-key", {key.lower() for key in forwarded.headers.keys()})
        completed, = self.events(proxy, "model_call_completed")
        self.assertEqual((completed["upstream"], completed["source"]), ("evidence", "sol_visual_review"))
        # One actor-binding self-briefing call spends the last unit of the shared 2-call budget.
        briefer = RelayEvidenceClient(session, model="claude-opus-5-5", source="self_briefing")
        actor_server.default = text_step('{"assessment":"supported"}')
        self.assertEqual(briefer.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="b1")["answer"],
                         {"assessment": "supported"})
        self.assertEqual(proxy.request_count, 2)
        with self.assertRaisesRegex(ValueError, "^sol_agent_unverified_or_incomplete_turn$"):
            client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="v2")
        self.assertEqual(len(evidence_server.generation_requests()), 1)
        unverified = self.store_events("visual.agent_turn_unverified")[-1]
        self.assertEqual((unverified["request_id"], unverified["http_status"], unverified["upstream"]),
                         ("v2", 429, "evidence"))
        self.assertTrue(self.events(proxy, "model_budget_exhausted"))
        self.assertNotIn("evidence-secret", json.dumps(proxy.events))

    def test_unbound_upstream_and_missing_relay(self):
        server = self.server(FakeResponsesServer)
        proxy = self.relay(server, OpenAIResponsesWire())
        with self.assertRaisesRegex(ValueError, "^evidence_upstream_not_bound:evidence$"):
            RelayEvidenceClient(self.session(proxy), upstream="evidence").review_evidence(
                input_items=ITEMS, output_schema=SCHEMA, request_id="x")
        with self.assertRaisesRegex(RuntimeError, "evidence_client_requires_started_model_relay"):
            RelayEvidenceClient(SimpleNamespace(store=self.store, model_proxy=None)).review_evidence(
                input_items=ITEMS, output_schema=SCHEMA, request_id="x")
        self.assertEqual(bind_evidence_upstream(self.session(proxy), Config()), "actor")
        self.assertEqual(server.requests, [])

    def test_resolve_upstream(self):
        upstream = Config.from_dict({"agent": {"backend": "opencode", "evidence": {"upstream": {
            "wire_api": "anthropic_messages", "base_url": "https://example.invalid", "api_key_env": "K",
            "auth_scheme": "x-api-key", "provider": "lab"}}}}).agent.evidence.upstream
        wire, url, headers, provider = resolve_upstream(upstream, environ={"K": "secret"})
        self.assertIsInstance(wire, AnthropicMessagesWire)
        self.assertEqual((url, headers, provider), ("https://example.invalid", {"x-api-key": "secret"}, "lab"))
        with self.assertRaisesRegex(ConfigError, "^evidence_upstream_api_key_missing:K"):
            resolve_upstream(upstream, environ={})
        home = self.root / "codex-home"
        home.mkdir()
        (home / "config.toml").write_text('model_provider = "openai"\n')
        (home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "sk-fixture"}))
        codex_home = Config.from_dict({"agent": {"backend": "opencode", "evidence": {"upstream": {
            "source": "codex_home"}}}}).agent.evidence.upstream
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("OPENAI_BASE_URL", None)
            wire, url, headers, provider = resolve_upstream(codex_home, codex_home=str(home))
        self.assertIsInstance(wire, OpenAIResponsesWire)
        self.assertEqual((url, headers, provider), ("https://api.openai.com/v1",
                                                    {"Authorization": "Bearer sk-fixture"}, "openai"))
        with self.assertRaisesRegex(ConfigError, "evidence_upstream_codex_home_unreadable"):
            resolve_upstream(codex_home, codex_home=str(self.root / "missing"))


# ------------------------------------------------------------------------------------------- cancellation and gates
class CancellationTests(RelayCase):
    def test_cancel_while_the_upstream_holds_the_answer(self):
        held = Step(text=ANSWER, gate=True)
        server = self.server(FakeResponsesServer, held, text_step(ANSWER))
        self.addCleanup(held.gate.release)
        proxy = self.relay(server, OpenAIResponsesWire())
        client = RelayEvidenceClient(self.session(proxy))
        cancel, outcome = threading.Event(), {}

        def call():
            try:
                client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="held",
                                       cancel_event=cancel)
            except BaseException as error:  # recorded for the assertion below
                outcome["error"] = error
        worker = threading.Thread(target=call, daemon=True)
        worker.start()
        self.assertTrue(held.gate.wait_entered(10))
        started = time.monotonic()
        cancel.set()
        worker.join(10)
        self.assertFalse(worker.is_alive())
        self.assertLess(time.monotonic() - started, 5)
        self.assertIsInstance(outcome.get("error"), TimeoutError)
        self.assertEqual(str(outcome["error"]), "visual_review_program_cancelled")
        self.assertEqual(proxy.cancelled_threads.get("evidence-held"), "helper_turn_interrupted")
        self.assertFalse(proxy.transport_blocked)  # an explicit helper cancellation never blocks the actor
        self.assertEqual(proxy.request_count, 0)
        held.gate.release()
        # The relay keeps serving the next helper call.
        result = client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="after")
        self.assertEqual(result["answer"], {"assessment": "uncertain"})
        self.assertEqual(proxy.request_count, 1)

    def test_already_cancelled_sends_nothing(self):
        server = self.server(FakeResponsesServer)
        proxy = self.relay(server, OpenAIResponsesWire())
        cancel = threading.Event()
        cancel.set()
        with self.assertRaisesRegex(TimeoutError, "^visual_review_program_cancelled$"):
            RelayEvidenceClient(self.session(proxy)).review_evidence(
                input_items=ITEMS, output_schema=SCHEMA, request_id="early", cancel_event=cancel)
        self.assertEqual((server.requests, self.events(proxy, "model_thread_registered")), ([], []))

    def test_recovery_gate_refuses_the_cooldown_resend(self):
        server = self.server(FakeResponsesServer, Step(text=ANSWER, failure=Failure("truncated")))
        recovery = Config().model.transport_recovery.to_dict()
        proxy = self.relay(server, OpenAIResponsesWire(), transport_recovery=recovery,
                           recovery_gate=lambda: "physical_action_unconfirmed")
        with self.assertRaisesRegex(ValueError, "^sol_agent_unverified_or_incomplete_turn$"):
            RelayEvidenceClient(self.session(proxy)).review_evidence(
                input_items=ITEMS, output_schema=SCHEMA, request_id="gate")
        denied, = self.events(proxy, "model_transport_recovery_denied")
        self.assertEqual((denied["reason"], denied["native_thread_id"]), ("physical_action_unconfirmed", "evidence-gate"))
        self.assertEqual(len(server.generation_requests()), 1)


# ------------------------------------------------------------------------------- failure strings, egress and restore
class FailureStringTests(RelayCase):
    STRINGS = ("sol_agent_invalid_request_id", "visual_review_program_cancelled",
               "sol_agent_unverified_or_incomplete_turn", "sol_agent_unexpected_tool_or_context_item",
               "sol_agent_requires_one_structured_final_answer")

    def test_strings_are_the_codex_evidence_agent_strings(self):
        source = inspect.getsource(CodexEvidenceAgent)
        for text in self.STRINGS:
            self.assertIn('"' + text + '"', source, text)
        self.assertEqual((wire_base.EVIDENCE_UNVERIFIED, wire_base.EVIDENCE_TOOL_ACTIVITY, wire_base.EVIDENCE_ONE_ANSWER),
                         self.STRINGS[2:])
        self.assertEqual((evidence_module.INVALID_REQUEST_ID, evidence_module.CANCELLED), self.STRINGS[:2])

    def test_invalid_request_ids(self):
        client = RelayEvidenceClient(SimpleNamespace(store=self.store, model_proxy=None))
        for request_id in ("", "a/b", "x" * 129, None, 3):
            with self.subTest(request_id=request_id), self.assertRaisesRegex(ValueError, "^sol_agent_invalid_request_id$"):
                client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id=request_id)

    def test_failure_paths_per_wire(self):
        cases = (
            ("tool", lambda: tool_step("robot_move", {"dx": 1}), wire_base.EVIDENCE_TOOL_ACTIVITY),
            ("thinking_only", lambda: Step(thinking="only thoughts"), wire_base.EVIDENCE_ONE_ANSWER),
            ("max_tokens", lambda: Step(text=ANSWER, stop_reason="max_tokens"), wire_base.EVIDENCE_UNVERIFIED),
            ("refusal", lambda: Step(text=ANSWER, stop_reason="refusal"), wire_base.EVIDENCE_UNVERIFIED),
            ("provider_400", lambda: Step(failure=Failure("http", 400, "provider-private text",
                                                          error_type="invalid_request_error")),
             wire_base.EVIDENCE_UNVERIFIED),
            ("truncated", lambda: Step(text=ANSWER, failure=Failure("truncated")), wire_base.EVIDENCE_UNVERIFIED),
            ("not_json", lambda: text_step("not json"), "Expecting value"),
        )
        for server_class, make_wire in ((FakeResponsesServer, OpenAIResponsesWire),
                                        (FakeAnthropicServer, AnthropicMessagesWire)):
            for label, step, error in cases:
                if label == "refusal" and server_class is FakeResponsesServer:
                    continue  # a Responses refusal arrives as a message part (covered by the P1-7 extractor tests)
                with self.subTest(wire=make_wire.name, case=label):
                    server = self.server(server_class, step())
                    proxy = self.relay(server, make_wire())
                    with self.assertRaisesRegex(ValueError, error):
                        RelayEvidenceClient(self.session(proxy), model="m").review_evidence(
                            input_items=ITEMS, output_schema=SCHEMA, request_id="f-" + label)
                    self.assertEqual(proxy.thread_owners.get("f-" + label), None)
                    self.assertNotIn("provider-private text", json.dumps(self.store_events("visual.agent_turn_unverified")))

    def test_aliases_are_presented_and_restored(self):
        artifact = self.store.put_artifact(b"fixture", "rgb", "image/png")
        alias = ActorPresentation(self.store).present_text("see " + artifact).split()[-1]
        self.assertTrue(alias.startswith("artifact:ref_"))
        server = self.server(FakeResponsesServer, text_step(json.dumps({"assessment": "supported", "ref": alias})))
        proxy = self.relay(server, OpenAIResponsesWire())
        schema = {**SCHEMA, "properties": {**SCHEMA["properties"], "ref": {"type": "string"}}}
        result = RelayEvidenceClient(self.session(proxy)).review_evidence(
            input_items=[dict(type="text", text="frame " + artifact)], output_schema=schema, request_id="alias")
        body = server.generation_requests()[0].body.decode()
        self.assertNotIn(artifact, body)
        self.assertIn(alias, body)
        self.assertEqual(result["answer"], {"assessment": "supported", "ref": artifact})


# ------------------------------------------------------------------------------------------- Codex evidence client
class CodexSelectionTests(RelayCase):
    def codex(self, document):
        config = Config.from_dict({**document, "model": {"codex_home": str(self.root / "codex-home")}})
        session = CodexSession(self.root / "run", self.store, lambda params: {}, config=config)
        self.addCleanup(session.close)
        return session

    def test_native_by_default_relay_on_request(self):
        native = self.codex({})
        self.assertIsInstance(native.evidence_agent(), CodexEvidenceAgent)
        self.assertEqual(native.provenance()["evidence_client"], "codex_agent")
        relay = self.codex({"agent": {"evidence": {"client": "relay"}}})
        helper = relay.evidence_agent(model="gpt-6-astra", source="self_briefing")
        self.assertIsInstance(helper, RelayEvidenceClient)
        self.assertEqual((helper.session, helper.model, helper.source, helper.upstream), (relay, "gpt-6-astra",
                                                                                        "self_briefing", "actor"))
        self.assertEqual(relay.provenance()["evidence_client"], "relay")

    def test_codex_relay_through_the_codex_relay_shape(self):
        """Codex with client="relay": the request rides the actor (Responses, codex profile) binding."""
        server = self.server(FakeResponsesServer)
        proxy = self.relay(server, OpenAIResponsesWire())
        session = self.codex({"agent": {"evidence": {"client": "relay"}}})
        session.model_proxy = proxy
        result = session.evidence_agent().review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="c1")
        self.assertEqual((result["backend"], result["upstream"]), (BACKEND_LABEL, "actor"))
        session.model_proxy = None  # owned by the test


# ------------------------------------------------------------------------------------------ VisualReviewer via relay
class VisualReviewerTests(RelayCase):
    def test_review_through_the_relay_client_records_its_backend(self):
        answer = dict(visibility_after="visible", motion="displaced", after_center_px=[10, 10],
                      retained_with_gripper="uncertain", evidence="Cup moved right.")
        server = self.server(FakeResponsesServer, text_step(json.dumps(answer)))
        proxy = self.relay(server, OpenAIResponsesWire())
        reviewer = visual_review.VisualReviewer(RelayEvidenceClient(self.session(proxy)))
        self.assertEqual((reviewer.backend_label, reviewer.context_policy), (BACKEND_LABEL, CONTEXT_POLICY))
        rgb = self.store.put_artifact(jpeg(), "rgb", "image/jpeg")
        sources = [dict(rgb_ref=rgb, width=32, height=24), dict(rgb_ref=rgb, width=32, height=24)]
        with patch.object(visual_review, "comparison_inputs", return_value=(None, sources)), \
                patch.object(visual_review, "saved_observation_state", return_value={"sim_step": 1}):
            result = reviewer.review(self.store, self.store.identity, before_observation_id="b",
                                     after_observation_id="a", camera_id="head", roi=[1, 1, 4, 4], target="cup")
        self.assertEqual(result["answer"], answer)
        reviewed, = self.store_events("visual.model_reviewed")
        self.assertEqual((reviewed["backend"], reviewed["context_policy"], reviewed["agent_thread_id"]),
                         (BACKEND_LABEL, CONTEXT_POLICY, "evidence-" + reviewed["request_id"]))
        _, path = self.store.artifact(reviewed["request_ref"])
        request = json.loads(path.read_text())
        self.assertEqual((request["backend"], request["context_policy"]), (BACKEND_LABEL, CONTEXT_POLICY))
        forwarded, = server.generation_requests()
        self.assertEqual(sum(part["type"] == "input_image" for part in forwarded.json["input"][1]["content"]), 2)
        self.assertEqual(forwarded.json["text"]["format"]["schema"], visual_review.ANSWER_SCHEMA)

    def test_default_labels_for_clients_without_them(self):
        reviewer = visual_review.VisualReviewer(SimpleNamespace(model=visual_review.MODEL))
        self.assertEqual((reviewer.backend_label, reviewer.context_policy),
                         ("official_codex_app_server_sdk", "fresh_ephemeral_thread_selected_evidence_only_v1"))


# ------------------------------------------------------------------------------------------- gateway (Workbench)
class FakeBackendWorkbenchTests(unittest.IsolatedAsyncioTestCase):
    """Non-Codex FakeAgentSession run whose evidence helpers use the relay client (design §9, P2 acceptance item)."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)

    def relay_for(self, server, wire, store):
        presentation = ActorPresentation(store)
        proxy = ModelRequestProxy(server.base_url, wire=wire, max_retries=0, default_thread_id=CLAUDE_SESSION,
                                  request_projection=lambda request: wire.project_request(
                                      request, presentation, registered_tools=frozenset(),
                                      tool_name_prefix="mcp__ni_rua__")).start()
        self.addCleanup(proxy.close)
        return proxy

    async def start(self, config, server, *, benchmark):
        created = []

        def create(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
            session = FakeAgentSession(run_dir, store, tool_handler, capabilities=AgentCapabilities(),
                                       model="claude-opus-5-5", provider="anthropic", price_card=price_card)
            session.model_proxy = self.relay_for(server, AnthropicMessagesWire(), store)
            created.append(session)
            return session
        robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
        with patch("ni_rua.apps.gateway.workbench.create_session", side_effect=create):
            workbench = Workbench(self.root / "run", robot=robot, config=config)
        session, = created
        store = workbench.store
        cameras = [dict(camera_id="head", rgb_ref=store.put_artifact(jpeg("green"), "rgb", "image/jpeg"))]
        state = {"frame_id": "env_origin", "sim_step": 0, "step_domain": "adapter_control_step", "arms": {}}
        initial = {**store.identity, "schema_version": "0.1", "world_kind": "observed", "observation_id": "initial",
                   "sim_step": 0, "step_domain": "adapter_control_step",
                   "robot_state_ref": store.put_artifact(json.dumps(state).encode(), "robot_state", "application/json"),
                   "task_instruction": "Fixture native instruction.", "cameras": cameras,
                   "observation_profile": workbench.harness.observation_profile}
        harness = SimpleNamespace(
            accepting=True, active_action=None, states={}, action_count=0, latest=initial, home_check=None,
            capabilities={}, episode_done=threading.Event(), episode_terminal=None,
            action_budget=lambda: dict(native_steps_used=0, native_steps_limit=None, native_steps_remaining=None),
            native_action_count=0, started_at=time.monotonic(), last_native_progress_at=time.monotonic(),
            scene=SimpleNamespace(capabilities=lambda: {}), start=Mock())
        for field in ("tool_registry", "config", "disabled_tools", "observation_profile", "motion_review",
                      "enable_wait", "vision_compare", "visual_reviewer", "action_policy", "programs"):
            setattr(harness, field, getattr(workbench.harness, field))
        harness.request_stop = lambda: setattr(harness, "accepting", False)
        workbench.harness = harness
        with patch("ni_rua.apps.gateway.workbench.Terminal") as terminal:
            await workbench.start("Smoke instruction.", benchmark=benchmark)
        terminal.assert_not_called()
        self.addAsyncCleanup(workbench.close)
        return workbench, session

    async def test_visual_reviewer_uses_actor_binding_on_benchmark_start_and_self_briefing_is_retired(self):
        server = FakeAnthropicServer([text_step('{"assessment":"supported"}')]).start()
        self.addCleanup(server.close)
        config = Config().with_overrides({"run": {"task": "pour_balls_into_vase"},
                                          "features": {"visual_review_model": "gpt-6.1-sol"}})
        workbench, session = await self.start(config, server, benchmark=True)
        self.assertEqual(server.generation_requests(), [])
        self.assertEqual([call[0] for call in session.calls[:2]], ["start", "run_turn"])
        manifest = json.loads((workbench.root / "manifest.json").read_text())
        self.assertEqual(manifest["evidence_client"], "relay")
        self.assertNotIn("self_briefing", manifest)
        client = workbench.harness.visual_reviewer.client
        self.assertIsInstance(client, RelayEvidenceClient)
        self.assertEqual((client.upstream, client.model, client.source),
                         ("actor", "gpt-6.1-sol", "sol_visual_review"))
        result = client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="wb-actor")
        self.assertEqual((result["answer"], result["upstream"]), ({"assessment": "supported"}, "actor"))
        forwarded, = server.generation_requests()
        self.assertNotIn("tools", forwarded.json)
        self.assertEqual(forwarded.json["model"], "gpt-6.1-sol")
        completed, = [event for event in session.model_proxy.events if event["kind"] == "model_call_completed"]
        self.assertEqual((completed["source"], completed["call_purpose"]), ("sol_visual_review", "helper"))

    async def test_visual_reviewer_uses_the_evidence_binding(self):
        actor = FakeAnthropicServer().start()
        reviewer_server = FakeResponsesServer([text_step('{"assessment":"supported"}')]).start()
        self.addCleanup(actor.close)
        self.addCleanup(reviewer_server.close)
        config = Config.from_dict({"features": {"visual_review_model": "gpt-6.1-sol"},
                                   "agent": {"evidence": {"client": "relay", "upstream": {
                                       "wire_api": "openai_responses", "base_url": reviewer_server.base_url}}}})
        workbench, session = await self.start(config, actor, benchmark=False)
        client = workbench.harness.visual_reviewer.client
        self.assertIsInstance(client, RelayEvidenceClient)
        self.assertEqual((client.upstream, client.model, client.source), ("evidence", "gpt-6.1-sol", "sol_visual_review"))
        manifest = json.loads((workbench.root / "manifest.json").read_text())
        self.assertEqual(manifest["evidence_client"], "relay")
        result = client.review_evidence(input_items=ITEMS, output_schema=SCHEMA, request_id="wb")
        self.assertEqual((result["answer"], result["upstream"]), ({"assessment": "supported"}, "evidence"))
        self.assertEqual((len(actor.generation_requests()), len(reviewer_server.generation_requests())), (0, 1))


class CodexWorkbenchManifestTests(unittest.IsolatedAsyncioTestCase):
    async def test_codex_visual_review_records_the_native_client(self):
        with tempfile.TemporaryDirectory() as temporary:
            robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
            workbench = Workbench(Path(temporary) / "run", robot=robot, visual_review_model="gpt-6.1-sol",
                                  config=Config.from_dict({}))
            from support.agent_fixtures import CODEX_MANIFEST_KEYS_BEFORE, fixture_harness, terminal_double
            workbench.harness = fixture_harness(workbench)
            workbench.session = SimpleNamespace(  # duck-typed Codex session without protocol members
                thread_id="fixture-thread", turn_id=None, state="completed", turn_done=threading.Event(),
                model="fixture", provider="fixture", effort="high", price_card=None, cli_executable=Path("/fixture/codex"),
                command_prefix=[], actor_dir=workbench.root, socket_path=workbench.root / "fixture.sock",
                env={}, process=SimpleNamespace(pid=1), start=Mock(return_value="fixture-thread"),
                run_turn=Mock(), close=Mock(), model_proxy=None, store=workbench.store)
            with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal_double()), \
                    patch("ni_rua.cortex.session.codex_evidence_agent.CodexEvidenceAgent") as native:
                native.return_value.model = "gpt-6.1-sol"
                await workbench.start("Smoke instruction.")
            try:
                native.assert_called_once_with(workbench.session)
                manifest = json.loads((workbench.root / "manifest.json").read_text())
                self.assertEqual(set(manifest) - CODEX_MANIFEST_KEYS_BEFORE, {"agent_backend", "evidence_client"})
                self.assertEqual((manifest["evidence_client"], manifest["agent_backend"]["evidence_client"]),
                                 ("codex_agent", "codex_agent"))
            finally:
                await workbench.close()


# --------------------------------------------------------------------------------------- config, gating, queue
class ConfigAndGatingTests(unittest.TestCase):
    def test_evidence_config_defaults_and_omission(self):
        self.assertEqual(AgentConfig().evidence, EvidenceConfig())
        self.assertEqual((EvidenceConfig().client, EvidenceConfig().upstream), ("native", None))
        self.assertNotIn("agent", Config().to_dict())
        exported = Config.from_dict({"agent": {"backend": "claude_code"}}).to_dict()
        self.assertNotIn("evidence", exported["agent"])  # agent snapshots of earlier checkouts stay identical
        relay = Config.from_dict({"agent": {"evidence": {"client": "relay"}}})
        self.assertEqual(relay.to_dict()["agent"]["evidence"], {"client": "relay", "upstream": None})
        self.assertEqual(Config.from_dict(relay.to_dict()), relay)
        bound = Config.from_dict({"agent": {"backend": "claude_code",
                                            "evidence": {"upstream": {"source": "codex_home"}}}})
        self.assertEqual(Config.from_dict(json.loads(json.dumps(bound.to_dict()))), bound)
        self.assertEqual(bound.agent.evidence.upstream.source, "codex_home")

    def test_evidence_config_validation(self):
        for document, message in (
                ({"evidence": {"client": "auto"}}, "invalid_config_choice:EvidenceConfig.client"),
                ({"evidence": {"upstream": {"source": "codex_home"}}}, "evidence_upstream_requires_relay_client"),
                ({"backend": "opencode", "evidence": {"upstream": {"wire_api": "openai_chat",
                                                                   "base_url": "http://127.0.0.1:1"}}},
                 "evidence_upstream_wire_not_implemented:openai_chat"),
                ({"backend": "opencode", "evidence": {"upstream": {"wire_api": "openai_responses"}}},
                 "evidence_upstream_requires_wire_api_and_base_url"),
                ({"backend": "opencode", "evidence": {"upstream": {"source": "codex_home",
                                                                   "wire_api": "anthropic_messages"}}},
                 "evidence_upstream_codex_home_requires_openai_responses"),
                ({"evidence": {"upstream": 3}}, "config_object_required:config.agent.evidence.upstream"),
                ({"evidence": {"upstream": {"api_key": "secret"}}}, "unknown_config_keys")):
            with self.subTest(document=document), self.assertRaisesRegex(ConfigError, message):
                Config.from_dict({"agent": document})

    def test_effective_client(self):
        self.assertEqual(factory.evidence_client(Config()), "codex_agent")
        self.assertEqual(factory.evidence_client(SimpleNamespace()), "codex_agent")  # no agent section: Codex native
        self.assertEqual(factory.evidence_client(Config.from_dict({"agent": {"evidence": {"client": "relay"}}})), "relay")
        for kind in ("claude_code", "opencode", "fake_agent"):
            self.assertEqual(factory.evidence_client(Config(), kind), "relay")
        for kind in ("claude_code", "opencode"):
            self.assertEqual(factory.evidence_client(Config.from_dict({"agent": {"backend": kind}})), "relay")

    def test_backend_requirements_evidence_rules(self):
        def code(document):
            with self.assertRaises(ConfigError) as raised:
                factory.backend_requirements(Config.from_dict(document))
            return str(raised.exception).split(":")[0]
        review = {"visual_review_model": "gpt-6.1-sol"}
        # The interim refusal is gone: evidence features pass on to the backend's own checks.
        self.assertEqual(code({"agent": {"backend": "opencode"}, "features": review}), "opencode_upstream_incomplete")
        with mock.patch.object(factory, "claude_code_runtime_problem", return_value=None):
            self.assertIsNone(factory.backend_requirements(Config.from_dict(
                {"agent": {"backend": "claude_code"}, "features": {"visual_review_model": None}})))
            self.assertEqual(code({"agent": {"backend": "claude_code"}, "features": review}),
                             "evidence_upstream_required_for_visual_review_model")
            self.assertIsNone(factory.backend_requirements(Config.from_dict(
                {"agent": {"backend": "claude_code", "evidence": {"upstream": {"source": "codex_home"}}},
                 "features": review})))
        self.assertEqual(code({"agent": {"backend": "opencode", "opencode": {"upstream": {
            "wire_api": "openai_responses", "base_url": "http://127.0.0.1:1", "api_key_env": "K"}}}, "features": review}),
            "agent_backend_unavailable")
        self.assertIsNone(factory.backend_requirements(Config.from_dict({"features": review})))

    def test_queue_projection_fails_closed_for_old_checkouts(self):
        config_dir = REPO_ROOT / "ni_rua/core/config"
        source = "\n".join(p.read_text() for p in sorted(config_dir.glob("*.py")))
        old = source.replace("    evidence: EvidenceConfig = field(default_factory=EvidenceConfig)\n", "")
        self.assertNotEqual(old, source)
        with tempfile.TemporaryDirectory() as directory:
            checkout = Path(directory)
            (checkout / "ni_rua/apps/gateway").mkdir(parents=True)
            (checkout / "ni_rua/core").mkdir(parents=True)
            (checkout / "ni_rua/apps/gateway/cli.py").write_text("parser.add_argument('--config')\nConfig.load()\n")
            (checkout / "ni_rua/core/config.py").write_text(old)
            schema = queue.runtime_config_schema(checkout)
        default_evidence = Config.from_dict({"agent": {"backend": "opencode"}}).to_dict()
        self.assertEqual(queue.project_config_for_runtimes(default_evidence, [schema]), default_evidence)
        relay = Config.from_dict({"agent": {"evidence": {"client": "relay"}}}).to_dict()
        with self.assertRaisesRegex(ValueError, "unified_config_incompatible:agent.evidence"):
            queue.project_config_for_runtimes(relay, [schema])
        current = queue.runtime_config_schema(REPO_ROOT)
        self.assertEqual(queue.project_config_for_runtimes(relay, [current]), relay)

    def test_queue_warns_when_cases_use_different_evidence_clients(self):
        codex = Config.from_dict({"features": {"visual_review_model": "gpt-6.1-sol"}})
        claude = codex.with_overrides({"agent": {"backend": "claude_code"}})
        quiet_claude = claude.with_overrides({"features": {"visual_review_model": None}})
        relay_codex = codex.with_overrides({"agent": {"evidence": {"client": "relay"}}})
        self.assertEqual(queue.evidence_path_warnings([codex, codex]), [])
        self.assertEqual(queue.evidence_path_warnings([codex, quiet_claude]), [])  # no evidence feature on one arm
        self.assertEqual(queue.evidence_path_warnings([relay_codex, claude]), [])  # the fairness rule followed
        warning, = queue.evidence_path_warnings([codex, claude, relay_codex])
        self.assertEqual((warning["kind"], warning["evidence_clients"], warning["agent_backends"]),
                         ("comparison_evidence_path_mismatch", {"codex_agent": [0], "relay": [1, 2]},
                          ["claude_code", "codex"]))

if __name__ == "__main__":
    unittest.main()
