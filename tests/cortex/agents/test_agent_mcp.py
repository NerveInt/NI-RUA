"""Merged cortex tests:
- tests/test_agent_mcp_server.py: NiRuaMcpServer + encode_mcp_result (design 04 §3.2-§3.4, §5.3; loopback only, no binaries, no model).
- tests/test_agent_mcp_session.py: Shared MCP-backend session skeleton (plan 07 §1; design 04 §2.2, §3.3, §3.7, §5.5): McpToolBridge + McpSessionMixin.
"""

NI_RUA_TEST_TIER = "component"

import base64
import hashlib
import http.client
import json
import socket
import tempfile
import threading
import time
import unittest
from types import SimpleNamespace
from support.fake_mcp_client import FakeMcpClient, parse_sse
from ni_rua.apps.evaluation.native_evidence import load_native_evidence
from ni_rua.core.config import Config
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents.dispatch import ToolDispatcher
from ni_rua.cortex.agents.mcp_server import (DEFAULT_RESULT_SIZE_HINT_CHARS, SUPPORTED_PROTOCOL_VERSIONS,
                                             NiRuaMcpServer, encode_mcp_result, mcp_tool_catalog,
                                             portable_input_schema)
from unittest import mock
from pathlib import Path
from support.fake_mcp_client import FakeMcpClient
from ni_rua.cortex.agents.base import AgentRejected, AgentSession
from ni_rua.cortex.agents.mcp_session import GENERATION_COUNTING, McpSessionMixin, McpToolBridge
from ni_rua.cortex.agents.transcript import TranscriptWriter
from ni_rua.cortex.session.wire import AnthropicMessagesWire

# ==== tests/test_agent_mcp_server.py ====

IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
TIMEOUT = 5
PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(256)) * 4
PNG_B64 = base64.b64encode(PNG).decode()
PLANTED_HASH = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
TOOLS = [
    {"type": "function", "name": "look", "description": "Read-only view. build sha256: " + PLANTED_HASH,
     "inputSchema": SCHEMA},
    {"type": "function", "name": "move", "description": "Stateful motion.", "inputSchema": SCHEMA,
     "deferLoading": True},
    {"type": "function", "name": "mixed", "description": "Request dependent.", "inputSchema": SCHEMA},
    {"type": "function", "name": "plain", "description": "No hint.", "inputSchema": SCHEMA},
]
HINTS = {"look": True, "move": False, "mixed": "request_dependent"}


def text_result(text, success=True):
    return {"success": success, "contentItems": [{"type": "inputText", "text": text}]}


def wait_until(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class FakeCallTool:
    """Stands in for the session's §3.3 closure: records calls, optional gates, scripted results."""

    def __init__(self):
        self.calls, self.started, self.finished = [], {}, {}
        self.gates, self.results, self.errors = {}, {}, {}
        self.lock = threading.Lock()
        self.threads = []          # ``native_thread_id`` of each call, in call order

    def gate(self, name):
        self.gates[name] = threading.Event()
        return self.gates[name]

    def __call__(self, name, arguments, *, call_id, native_call_id, rpc_id, native_thread_id=None):
        with self.lock:
            self.calls.append(dict(name=name, arguments=arguments, call_id=call_id, native_call_id=native_call_id,
                                   rpc_id=rpc_id))
            self.threads.append(native_thread_id)
            self.started.setdefault(name, threading.Event()).set()
        try:
            if name in self.gates:
                self.gates[name].wait(TIMEOUT * 2)
            if name in self.errors:
                raise self.errors[name]
            return self.results.get(name, text_result("ok:" + name))
        finally:
            with self.lock:
                self.finished.setdefault(name, threading.Event()).set()


class ServerCase(unittest.TestCase):
    server_kwargs = {}

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.store = Store(self.tmp.name, IDENTITY)
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.fake = FakeCallTool()
        self.failures = []
        self.server = self.make_server(**self.server_kwargs)
        self.client = FakeMcpClient(self.server.url, token=self.server.token, protocol_version="2025-11-25")

    def make_server(self, **kwargs):
        options = dict(tools=TOOLS, presentation=self.presentation, call_tool=self.fake,
                       raw_log=lambda direction, message: self.store.raw_agent(direction, message, channel="mcp"),
                       on_delivery_failure=lambda call_id, reason: self.failures.append((call_id, reason)),
                       parallel_hints=HINTS, keepalive_s=0.05, peer_check_s=0.02)
        options.update(kwargs)
        server = NiRuaMcpServer(**options).start()
        self.addCleanup(server.close)
        return server

    def raw_records(self):
        path = self.store.root / "agent-raw.jsonl"
        return [json.loads(line) for line in path.read_text().splitlines()] if path.exists() else []


# ---------------------------------------------------------------------------------------------------------------------
class HandshakeTest(ServerCase):
    def test_server_version_reads_installed_distribution(self):
        from ni_rua.cortex.agents import mcp_server

        def installed(names):
            def lookup(distribution):
                if distribution in names:
                    return names[distribution]
                raise mcp_server.PackageNotFoundError(distribution)
            return lookup
        for names, expected in (({"ni-rua-harness": "0.1.0"}, "0.1.0"), ({}, "unknown")):
            with self.subTest(names=names), mock.patch.object(mcp_server, "package_version", side_effect=installed(names)):
                self.assertEqual(mcp_server._server_version(), expected)

    def test_claude_code_opening_sequence(self):
        probe = self.client.discover()
        self.assertEqual(probe.status, 200)
        self.assertEqual(probe.messages[0], {"jsonrpc": "2.0", "id": "server-discover-probe-1",
                                             "error": {"code": -32601, "message": "Method not found"}})
        init = self.client.initialize()
        self.assertEqual(init["protocolVersion"], "2025-11-25")
        self.assertEqual(init["capabilities"], {"tools": {"listChanged": False}})
        self.assertEqual(init["serverInfo"]["name"], "ni_rua")
        self.assertIsInstance(init["serverInfo"]["version"], str)
        self.assertNotIn("instructions", init)
        self.assertEqual(self.client.initialized().status, 202)
        stream = self.client.open_standalone_stream()
        self.assertEqual((stream.status, stream.headers.get("Allow")), (405, "POST"))
        self.assertEqual(self.client.http("DELETE").status, 405)
        self.assertEqual(self.client.ping(), {})
        self.assertEqual(len(self.client.list_tools()), len(TOOLS))

    def test_protocol_negotiation(self):
        for offered in SUPPORTED_PROTOCOL_VERSIONS:
            client = FakeMcpClient(self.server.url, token=self.server.token, protocol_version=offered)
            self.assertEqual(client.initialize()["protocolVersion"], offered)
        for offered in ("2099-01-01", None):
            client = FakeMcpClient(self.server.url, token=self.server.token, protocol_version=offered)
            self.assertEqual(client.initialize()["protocolVersion"], "2025-11-25")

    def test_unknown_method_parse_error_batch_and_notifications(self):
        response = self.client.request("resources/list", {})
        self.assertEqual(response.messages[0]["error"]["code"], -32601)
        self.assertEqual(self.client.post(b"{not json").status, 400)
        self.assertEqual(self.client.last_response.messages[0]["error"]["code"], -32700)
        batch = self.client.post(json.dumps([{"jsonrpc": "2.0", "id": 1, "method": "ping"}]).encode())
        self.assertEqual((batch.status, batch.messages[0]["error"]["code"]), (400, -32600))
        self.assertEqual(self.client.notify("notifications/roots/list_changed").status, 202)
        self.assertEqual(self.client.post({"jsonrpc": "2.0", "id": 4, "result": {}}).status, 202)
        self.assertEqual(self.client.request("ping", request_id=True).status, 400)


class CatalogTest(ServerCase):
    def test_list_tools_scrub_annotations_and_meta(self):
        tools = {tool["name"]: tool for tool in self.client.list_tools()}
        self.assertEqual(list(tools), ["look", "move", "mixed", "plain"])
        for name, tool in tools.items():
            self.assertEqual(set(tool), {"name", "description", "inputSchema", "annotations", "_meta"})
            self.assertEqual(tool["_meta"], {"anthropic/maxResultSizeChars": DEFAULT_RESULT_SIZE_HINT_CHARS})
        self.assertEqual({n: t["annotations"]["readOnlyHint"] for n, t in tools.items()},
                         {"look": True, "move": False, "mixed": False, "plain": False})
        self.assertNotIn(PLANTED_HASH, json.dumps(tools))
        self.assertEqual(tools["look"]["description"], self.presentation.present_text(TOOLS[0]["description"]))
        self.assertEqual(tools["move"]["inputSchema"], self.presentation.present_data(SCHEMA))
        self.assertEqual(self.server.read_only_tools, ["look"])

    def test_top_level_combinators_are_removed_and_nested_ones_stay(self):
        nested = {"type": "object", "anyOf": [{"required": ["a"]}], "properties": {"a": {"type": "string"}}}
        schema = {"type": "object", "properties": {"op": {"type": "string"}, "n": nested}, "required": [],
                  "allOf": [{"if": {"required": ["op"]}}], "anyOf": [{"required": ["op"]}], "oneOf": [{}],
                  "if": {"required": ["op"]}, "then": {"required": ["n"]}, "else": {}, "additionalProperties": False}
        tool = {"name": "t", "description": "d", "inputSchema": schema}
        published = mcp_tool_catalog([tool], self.presentation)[0]["inputSchema"]
        self.assertEqual(published, {"type": "object", "properties": {"op": {"type": "string"}, "n": nested},
                                     "required": [], "additionalProperties": False})
        self.assertIn("allOf", schema)   # the registry definition is not mutated

    def test_task_plan_is_published_without_top_level_allof_but_rules_stay_in_prose(self):
        from ni_rua.runtime.tools.builtins import create_registry
        # the prose catalogue (full_v1) and the default one (layered_v3, typed notation) both state the rules in text
        for profile, marker in (("full_v1", "Object rules:"), (Config().features.tool_catalog_profile, "-- Rules:")):
            with self.subTest(profile=profile):
                config = Config().with_overrides({"features": {"tool_catalog_profile": profile}})
                runtime = SimpleNamespace(config=config, disabled_tools=config.features.disabled_tools,
                                          observation_profile=config.features.observation_profile,
                                          motion_review=config.features.motion_review, enable_wait=False,
                                          vision_compare=config.features.vision_compare, visual_reviewer=None,
                                          action_policy=None, capabilities=None, programs=True)
                tools = create_registry(runtime=runtime).catalog(runtime, role="actor")
                plan = next(tool for tool in tools if tool["name"] == "task_plan")
                self.assertIn("allOf", plan["inputSchema"])
                published = next(tool for tool in mcp_tool_catalog(tools, self.presentation) if tool["name"] == "task_plan")
                self.assertFalse({"anyOf", "oneOf", "allOf", "if", "then", "else"} & set(published["inputSchema"]))
                self.assertEqual(published["description"], self.presentation.present_text(plan["description"]))
                self.assertIn(marker, published["description"])

    def test_custom_size_hint_and_invalid_definitions(self):
        catalog = mcp_tool_catalog(TOOLS[:1], self.presentation, result_size_hint_chars=123)
        self.assertEqual(catalog[0]["_meta"], {"anthropic/maxResultSizeChars": 123})
        self.assertFalse(catalog[0]["annotations"]["readOnlyHint"])
        for bad in ([{"name": "x"}], [TOOLS[0], TOOLS[0]]):
            with self.assertRaises(ValueError):
                mcp_tool_catalog(bad, self.presentation)
        for size in (0, -1, 1.5, True):
            with self.assertRaises(ValueError):
                mcp_tool_catalog(TOOLS, self.presentation, result_size_hint_chars=size)

    def test_registry_catalog_matches_codex_dynamic_tools_presentation(self):
        from ni_rua.cortex.agents.codex.session import DYNAMIC_TOOL_NAMESPACE, _native_dynamic_tools
        from ni_rua.cortex.session.wire.openai_responses import present_responses_request
        from ni_rua.runtime.tools.builtins import create_registry
        config = Config()
        runtime = SimpleNamespace(config=config, disabled_tools=config.features.disabled_tools,
                                  observation_profile=config.features.observation_profile,
                                  motion_review=config.features.motion_review, enable_wait=False,
                                  vision_compare=config.features.vision_compare, visual_reviewer=None,
                                  action_policy=None, capabilities=None, programs=True)
        registry = create_registry(runtime=runtime)
        tools = registry.catalog(runtime, role="actor")
        manifest = registry.manifest(runtime)
        hints = {tool["name"]: tool["parallel"] for tool in manifest["tools"]}
        catalog = mcp_tool_catalog(tools, self.presentation, parallel_hints=hints)
        deferred = {tool["name"] for tool in tools if tool.get("deferLoading")}
        codex = present_responses_request(dict(tools=_native_dynamic_tools(tools)), self.presentation,
                                          deferred_tools=deferred, namespace=DYNAMIC_TOOL_NAMESPACE)["tools"]
        flat = {tool["name"]: tool for entry in codex for tool in (entry.get("tools") or [entry])}
        self.assertEqual({tool["name"] for tool in catalog}, set(flat))
        for tool in catalog:
            self.assertEqual(tool["description"], flat[tool["name"]]["description"])
            # the MCP projection differs from the Codex schema only by the removed top-level combinators
            self.assertEqual(tool["inputSchema"], portable_input_schema(flat[tool["name"]]["inputSchema"]))
            self.assertEqual(tool["annotations"]["readOnlyHint"], hints[tool["name"]] is True)
        self.assertEqual({t["name"] for t in catalog if t["annotations"]["readOnlyHint"]},
                         {name for name, value in hints.items() if value is True})


class CallTest(ServerCase):
    def test_client_session_key_reaches_call_tool_only_when_it_is_a_usable_string(self):
        """``_meta["ai.opencode/sessionID"]`` (OpenCode) -> ``native_thread_id``; anything else -> None."""
        for meta in ({"ai.opencode/sessionID": "ses_1", "progressToken": 2}, None, {"ai.opencode/sessionID": ""},
                     {"ai.opencode/sessionID": 7}, {"ai.opencode/sessionID": "x" * 257},
                     {"claudecode/toolUseId": "toolu_1"}):
            self.client.call_tool("look", {}, meta=meta)
        self.assertEqual(self.fake.threads, ["ses_1", None, None, None, None, None])

    def test_json_call_with_native_id(self):
        result = self.client.call_tool("look", {"x": 1}, meta={"claudecode/toolUseId": "toolu_1"},
                                       accept_sse=False)
        self.assertFalse(result.is_sse)
        self.assertEqual(result.result, {"content": [{"type": "text", "text": "ok:look"}], "isError": False})
        self.assertEqual(self.fake.calls, [dict(name="look", arguments={"x": 1}, call_id="toolu_1",
                                                native_call_id="toolu_1", rpc_id=result.request_id)])
        self.assertEqual(self.failures, [])

    def test_sse_call_with_progress_and_generated_call_id(self):
        self.fake.gate("move")
        timer = threading.Timer(0.4, self.fake.gates["move"].set)
        timer.start()
        self.addCleanup(timer.cancel)
        result = self.client.call_tool("move", {}, progress_token="tok-7")
        self.assertTrue(result.is_sse)
        self.assertGreaterEqual(len(result.progress), 3)
        self.assertEqual([p["progress"] for p in result.progress], list(range(1, len(result.progress) + 1)))
        self.assertEqual({p["progressToken"] for p in result.progress}, {"tok-7"})
        self.assertTrue(all(not frame["comments"] for frame in result.response.frames))
        self.assertEqual(result.response.frames[-1]["event"], "message")
        self.assertEqual(result.result["content"][0]["text"], "ok:move")
        call = self.fake.calls[0]
        self.assertIsNone(call["native_call_id"])
        self.assertTrue(call["call_id"].startswith("call-"))

    def test_sse_without_progress_token_uses_comments(self):
        self.fake.gate("move")
        timer = threading.Timer(0.3, self.fake.gates["move"].set)
        timer.start()
        self.addCleanup(timer.cancel)
        result = self.client.call_tool("move", {})
        self.assertEqual(result.progress, [])
        self.assertGreaterEqual(sum(frame["comments"] == ["keepalive"] for frame in result.response.frames), 2)
        self.assertEqual(result.result["content"][0]["text"], "ok:move")

    def test_call_tool_exception_becomes_presented_error_result(self):
        self.fake.errors["move"] = ValueError("native_tool_call_stale_connection")
        result = self.client.call_tool("move", {"x": 2}, progress_token=1)
        self.assertTrue(result.result["isError"])
        payload = json.loads(result.result["content"][0]["text"])
        self.assertEqual(payload["error"], "native_tool_call_stale_connection")
        self.assertEqual(payload["tool"], "move")
        self.assertEqual(self.failures, [])

    def test_invalid_params_are_not_dispatched(self):
        for params in ({"name": "look", "arguments": [1]}, {"arguments": {}}, {"name": 3}):
            response = self.client.request("tools/call", params)
            self.assertEqual(response.messages[-1]["error"]["code"], -32602)
        self.assertEqual(self.fake.calls, [])

    def test_missing_arguments_default_to_empty_object(self):
        response = self.client.request("tools/call", {"name": "look"}, accept="application/json")
        self.assertFalse(response.messages[0]["result"]["isError"])
        self.assertEqual(self.fake.calls[0]["arguments"], {})

    def test_duplicate_in_flight_request_id_is_refused(self):
        self.fake.gate("move")
        pending = self.client.start_call("move", {}, request_id=42, progress_token=1)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        second = self.client.call_tool("look", {}, request_id=42, accept_sse=False)
        self.assertEqual(second.error["code"], -32600)
        self.fake.gates["move"].set()
        self.assertFalse(pending.wait(TIMEOUT).result["isError"])
        self.assertEqual([call["name"] for call in self.fake.calls], ["move"])

    def test_multi_client_same_request_id_and_cancel_are_isolated(self):
        # 外部无 bearer 多客户端场景：不同客户端并发使用相同 JSON-RPC id 不冲突，且取消通知绝不跨客户端掐断原连接
        server = self.make_server(require_bearer=False)
        client_a = FakeMcpClient(server.url)
        client_b = FakeMcpClient(server.url)
        client_a.handshake()
        client_b.handshake()
        self.assertTrue(client_a.session_id and client_b.session_id)
        self.assertNotEqual(client_a.session_id, client_b.session_id)

        self.fake.gate("move")
        pending_a = client_a.start_call("move", {}, request_id=7, progress_token=1)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        # client_b 用相同 id=7 并发调用 look 必须成功，不得报 duplicate in-flight request id
        res_b = client_b.call_tool("look", {}, request_id=7, accept_sse=False)
        self.assertIsNone(res_b.error)
        self.assertFalse(res_b.result["isError"])
        # client_b 发送针对 id=7 的取消通知，以及无 session 头客户端发取消通知，均不得影响 client_a 的在途调用
        self.assertEqual(client_b.cancel(7).status, 202)
        unscoped = FakeMcpClient(server.url)
        self.assertEqual(unscoped.cancel(7).status, 202)
        # 无 session 头客户端同样允许并发使用 id=7（即使两个无头请求携带相同 _meta toolUseId 也不得冲突）
        unscoped_a = FakeMcpClient(server.url)
        unscoped_b = FakeMcpClient(server.url)
        self.fake.gate("look")
        pending_u1 = unscoped_a.start_call(
            "look", {}, request_id=7, meta={"claudecode/toolUseId": "dup-tool-use-1"}, progress_token=1
        )
        self.assertTrue(self.fake.started.setdefault("look", threading.Event()).wait(TIMEOUT))
        pending_u2 = unscoped_b.start_call(
            "look", {}, request_id=7, meta={"claudecode/toolUseId": "dup-tool-use-1"}, progress_token=2
        )
        self.fake.gates["look"].set()
        res_u1 = pending_u1.wait(TIMEOUT)
        res_u2 = pending_u2.wait(TIMEOUT)
        self.assertIsNone(res_u1.error)
        self.assertIsNone(res_u2.error)
        self.assertFalse(res_u1.result["isError"])
        self.assertFalse(res_u2.result["isError"])

        self.fake.gates["move"].set()
        done_a = pending_a.wait(TIMEOUT)
        self.assertIsNone(done_a.error)
        self.assertFalse(done_a.result["isError"])
        self.assertEqual(self.failures, [])

    def test_encoding_failure_is_a_jsonrpc_error_and_a_delivery_failure(self):
        self.fake.results["look"] = {"success": True, "contentItems": [
            {"type": "inputText", "text": "see"}, {"type": "inputImage", "imageUrl": "https://example.invalid/a.png"}]}
        result = self.client.call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_9"}, progress_token=1)
        self.assertEqual(result.error["code"], -32603)
        self.assertTrue(result.error["message"].startswith("invalid_tool_image_url"))
        self.assertNotIn(b"example.invalid", result.response.raw)
        self.assertEqual(self.failures, [("toolu_9", "tool_reply_encoding_failed")])


class KeepaliveCadenceTest(ServerCase):
    server_kwargs = dict(keepalive_s=0.2, peer_check_s=0.02)

    def test_long_call_headers_immediately_then_progress_every_keepalive(self):
        self.fake.gate("move")
        body = json.dumps({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                           "params": {"name": "move", "arguments": {}, "_meta": {"progressToken": 5}}})
        conn = http.client.HTTPConnection("127.0.0.1", self.server.server.server_port, timeout=TIMEOUT)
        self.addCleanup(conn.close)
        started = time.monotonic()
        conn.request("POST", "/mcp", body=body, headers={**self.server.headers, "Content-Type": "application/json",
                                                         "Accept": "application/json, text/event-stream"})
        response = conn.getresponse()  # headers arrive while the call is still gated
        self.assertFalse(self.fake.finished.get("move", threading.Event()).is_set())
        self.assertEqual(response.getheader("Content-Type"), "text/event-stream")
        arrivals = []
        while len(arrivals) < 5:
            line = response.fp.readline()
            if line.startswith(b"data:"):
                arrivals.append(time.monotonic())
                self.assertEqual(json.loads(line[5:])["method"], "notifications/progress")
        self.fake.gates["move"].set()
        rest = response.read()
        self.assertIn(b'"ok:move"', rest)
        gaps = [later - earlier for earlier, later in zip(arrivals, arrivals[1:])]
        self.assertGreater(arrivals[0] - started, 0.1)
        for gap in gaps:
            self.assertGreater(gap, 0.1)
            self.assertLess(gap, 1.0)


class CancelAndDisconnectTest(ServerCase):
    def test_cancel_after_answer_is_ignored(self):
        result = self.client.call_tool_then_cancel("look", {}, progress_token=3)
        self.assertEqual(self.client.last_response.status, 202)
        self.assertFalse(result.result["isError"])
        time.sleep(0.1)
        self.assertEqual(self.failures, [])
        records = [r["message"] for r in self.raw_records()]
        sent = next(i for i, m in enumerate(records) if m.get("id") == result.request_id and "result" in m)
        cancel = next(i for i, m in enumerate(records) if m.get("method") == "notifications/cancelled")
        self.assertLess(sent, cancel)

    def test_in_flight_cancel_abandons_without_aborting(self):
        self.fake.gate("move")
        pending = self.client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_c"}, progress_token=9)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.assertEqual(self.client.cancel(pending.request_id, "AbortError: interrupt").status, 202)
        self.assertEqual(self.failures, [("toolu_c", "tool_reply_delivery_uncertain")])
        outcome = pending.wait(TIMEOUT)  # the stream is closed without a result
        self.assertIsNone(outcome.message)
        self.assertFalse(self.fake.finished.get("move", threading.Event()).is_set())  # never aborted
        self.fake.gates["move"].set()
        self.assertTrue(self.fake.finished.setdefault("move", threading.Event()).wait(TIMEOUT))
        time.sleep(0.1)
        self.assertEqual(len(self.failures), 1)
        self.assertFalse(any(r["direction"] == "sent" and r["message"].get("id") == pending.request_id
                             for r in self.raw_records()))

    def open_call(self, request_id, name, *, sse=True, meta=None):
        """Raw socket ``tools/call`` so the test decides exactly when the peer goes away."""
        body = json.dumps({"jsonrpc": "2.0", "id": request_id, "method": "tools/call",
                           "params": {"name": name, "arguments": {}, "_meta": {"progressToken": 1, **(meta or {})}}})
        port = self.server.server.server_port
        sock = socket.create_connection(("127.0.0.1", port), timeout=TIMEOUT)
        self.addCleanup(sock.close)
        accept = "application/json, text/event-stream" if sse else "application/json"
        sock.sendall((f"POST /mcp HTTP/1.1\r\nHost: 127.0.0.1:{port}\r\n"
                      f"Authorization: {self.server.headers['Authorization']}\r\nContent-Type: application/json\r\n"
                      f"Accept: {accept}\r\nContent-Length: {len(body)}\r\n\r\n{body}").encode())
        return sock

    @staticmethod
    def hang_up(sock):
        sock.shutdown(socket.SHUT_RDWR)
        sock.close()

    def test_peer_close_during_sse_call(self):
        self.fake.gate("move")
        sock = self.open_call(11, "move", meta={"claudecode/toolUseId": "toolu_d"})
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.hang_up(sock)
        self.assertTrue(wait_until(lambda: self.failures))
        self.assertEqual(self.failures, [("toolu_d", "tool_reply_delivery_uncertain")])
        self.fake.gates["move"].set()
        self.assertTrue(self.fake.finished.setdefault("move", threading.Event()).wait(TIMEOUT))
        time.sleep(0.1)
        self.assertEqual(len(self.failures), 1)
        self.assertFalse(any(r["direction"] == "sent" and r["message"].get("id") == 11 for r in self.raw_records()))

    def test_peer_close_before_dispatch_is_never_executed_nor_reported(self):
        logged, release = threading.Event(), threading.Event()

        def raw_log(direction, message):
            self.store.raw_agent(direction, message, channel="mcp")
            if message.get("method") == "tools/call":
                logged.set()
                release.wait(TIMEOUT)  # the peer hangs up while the server is still recording the call

        server = self.make_server(raw_log=raw_log)
        self.server = server
        sock = self.open_call(12, "move")
        self.assertTrue(logged.wait(TIMEOUT))
        self.hang_up(sock)
        self.assertTrue(wait_until(lambda: not server.inflight))
        release.set()
        time.sleep(0.2)
        self.assertEqual((self.fake.calls, self.failures), ([], []))

    def test_peer_close_during_json_call(self):
        self.fake.gate("move")
        self.client.call_tool_and_disconnect("move", {}, accept_sse=False, wait_for="sent")
        self.assertTrue(wait_until(lambda: self.failures))
        self.assertEqual(len(self.failures), 1)
        call_id, reason = self.failures[0]
        self.assertEqual(reason, "tool_reply_delivery_uncertain")
        self.assertTrue(call_id.startswith("call-"))
        self.fake.gates["move"].set()

    def test_peer_close_found_at_final_write(self):
        server = self.make_server(peer_check_s=60, keepalive_s=60)  # only the pre-write check can see it
        client = FakeMcpClient(server.url, token=server.token)
        self.fake.gate("move")
        client.call_tool_and_disconnect("move", {}, progress_token=1)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        time.sleep(0.1)
        self.assertEqual(self.failures, [])
        self.fake.gates["move"].set()
        self.assertTrue(wait_until(lambda: self.failures))
        self.assertEqual(self.failures, [(self.fake.calls[0]["call_id"], "tool_reply_delivery_uncertain")])

    def test_write_error_on_final_write(self):
        server = self.make_server(peer_check_s=60, keepalive_s=60)
        server._peer_gone = lambda call: False  # force the write itself to discover the closed peer
        big = base64.b64encode(bytes(8 * 1024 * 1024)).decode()
        self.fake.results["move"] = {"success": True, "contentItems": [
            {"type": "inputImage", "imageUrl": "data:image/png;base64," + big}]}
        self.fake.gate("move")
        self.server = server
        sock = self.open_call(13, "move", meta={"claudecode/toolUseId": "toolu_w"})
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.hang_up(sock)
        time.sleep(0.05)
        self.fake.gates["move"].set()
        self.assertTrue(wait_until(lambda: self.failures))
        self.assertEqual(self.failures, [("toolu_w", "tool_reply_delivery_uncertain")])
        self.assertFalse(any(r["direction"] == "sent" and r["message"].get("id") == 13 for r in self.raw_records()))

    def test_close_abandons_in_flight_calls_and_is_idempotent(self):
        self.fake.gate("move")
        pending = self.client.start_call("move", {}, progress_token=1)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.server.close()
        self.server.close()
        self.assertEqual(self.failures, [(self.fake.calls[0]["call_id"], "tool_reply_delivery_uncertain")])
        self.assertIsNone(pending.wait(TIMEOUT).message)
        self.assertFalse(self.server.keepalive_thread.is_alive())
        with self.assertRaises(OSError):
            socket.create_connection(("127.0.0.1", self.server.server.server_port), timeout=1).close()
        self.fake.gates["move"].set()


class CallbackFailureTest(ServerCase):
    def test_unrecorded_call_is_refused_before_dispatch(self):
        def raw_log(direction, message):
            if message.get("method") == "tools/call":
                raise OSError("disk full")
            self.store.raw_agent(direction, message, channel="mcp")

        server = self.make_server(raw_log=raw_log)
        client = FakeMcpClient(server.url, token=server.token)
        result = client.call_tool("move", {}, progress_token=1)
        self.assertEqual(result.error["code"], -32603)
        self.assertEqual((self.fake.calls, self.failures), ([], []))
        self.assertEqual(server.errors, [("raw_log", "OSError('disk full')")])
        self.assertEqual(client.ping(), {})

    def test_raising_delivery_callback_does_not_stop_the_server(self):
        def broken(call_id, reason):
            raise RuntimeError("boom")

        server = self.make_server(on_delivery_failure=broken)
        client = FakeMcpClient(server.url, token=server.token)
        self.fake.gate("move")
        pending = client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_x"}, progress_token=1)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        client.cancel(pending.request_id)
        self.assertEqual(server.errors, [("toolu_x", "RuntimeError('boom')")])
        self.fake.gates["move"].set()
        self.assertFalse(client.call_tool("look", {}, progress_token=2).result["isError"])
        self.assertTrue(server.keepalive_thread.is_alive())


class AccessTest(ServerCase):
    def test_bearer_and_host(self):
        for token in ("wrong", None):
            response = FakeMcpClient(self.server.url, token=token).call_tool("look", {})
            self.assertEqual((response.status, response.message), (401, None))
        port = self.server.server.server_port
        for host in ("evil.example:%d" % port, "127.0.0.1:1", "10.0.0.1:%d" % port):
            client = FakeMcpClient(self.server.url, token=self.server.token, headers={"Host": host})
            self.assertEqual(client.call_tool("look", {}).status, 403)
            self.assertEqual(client.http("GET").status, 403)
        self.assertEqual(FakeMcpClient(self.server.url, token="wrong").http("GET").status, 401)
        self.assertEqual(self.fake.calls, [])
        self.assertEqual(self.raw_records(), [])
        local = FakeMcpClient(self.server.url, token=self.server.token, headers={"Host": "localhost:%d" % port})
        self.assertFalse(local.call_tool("look", {}).result["isError"])
        other = FakeMcpClient(self.server.url.replace("/mcp", "/other"), token=self.server.token)
        self.assertEqual(other.request("ping").status, 404)

    def test_request_size_cap(self):
        server = self.make_server(max_request_bytes=512)
        client = FakeMcpClient(server.url, token=server.token)
        self.assertEqual(client.call_tool("look", {"x": "a" * 1024}).status, 413)
        self.assertEqual(client.call_tool("look", {"x": "a" * 100}).status, 200)
        chunked = client.post(b"", headers={"Transfer-Encoding": "chunked"})
        self.assertEqual(chunked.status, 411)
        self.assertEqual([call["arguments"]["x"] for call in self.fake.calls], ["a" * 100])

    def test_constructor_validation(self):
        base = dict(tools=TOOLS, presentation=self.presentation, call_tool=self.fake, raw_log=print,
                    on_delivery_failure=print)
        for bad in (dict(keepalive_s=0), dict(peer_check_s=-1), dict(io_timeout_s=True),
                    dict(max_request_bytes=0), dict(server_name="a b"), dict(result_size_hint_chars=0)):
            with self.assertRaises(ValueError):
                NiRuaMcpServer(**base, **bad)
        with self.assertRaises(ValueError):
            self.server.start()


# ---------------------------------------------------------------------------------------------------------------------
class EncodeTest(unittest.TestCase):
    def test_encode_table_order_and_byte_identity(self):
        audio = base64.b64encode(b"RIFF....WAVE").decode()
        response = {"success": True, "contentItems": [
            {"type": "inputText", "text": "camera follows"},
            {"type": "inputImage", "imageUrl": "data:image/png;base64," + PNG_B64},
            {"type": "inputText", "text": "and audio"},
            {"type": "inputAudio", "audioUrl": "data:audio/wav;base64," + audio},
            {"type": "inputImage", "imageUrl": "data:image/webp;foo=bar;base64," + PNG_B64}]}
        encoded = encode_mcp_result(response)
        self.assertEqual(encoded, {"isError": False, "content": [
            {"type": "text", "text": "camera follows"},
            {"type": "image", "data": PNG_B64, "mimeType": "image/png"},
            {"type": "text", "text": "and audio"},
            {"type": "audio", "data": audio, "mimeType": "audio/wav"},
            {"type": "image", "data": PNG_B64, "mimeType": "image/webp"}]})
        self.assertEqual(base64.b64decode(encoded["content"][1]["data"]), PNG)
        self.assertEqual(response["contentItems"][1]["imageUrl"], "data:image/png;base64," + PNG_B64)  # not mutated

    def test_is_error_only_without_media(self):
        self.assertTrue(encode_mcp_result(text_result("failed", success=False))["isError"])
        self.assertFalse(encode_mcp_result(text_result("fine"))["isError"])
        with_image = {"success": False, "contentItems": [
            {"type": "inputText", "text": "failed, evidence follows"},
            {"type": "inputImage", "imageUrl": "data:image/jpeg;base64," + PNG_B64}]}
        encoded = encode_mcp_result(with_image)
        self.assertFalse(encoded["isError"])
        self.assertEqual([part["type"] for part in encoded["content"]], ["text", "image"])
        with_audio = {"success": False, "contentItems": [{"type": "inputAudio", "audioUrl": "data:audio/wav;base64,AA=="}]}
        self.assertFalse(encode_mcp_result(with_audio)["isError"])

    def test_rejections(self):
        cases = [
            ({"type": "inputImage", "imageUrl": "https://example.invalid/x.png"}, "invalid_tool_image_url"),
            ({"type": "inputImage", "imageUrl": "file:///tmp/x.png"}, "invalid_tool_image_url"),
            ({"type": "inputImage", "imageUrl": "data:image/png,rawbytes"}, "invalid_tool_image_url"),
            ({"type": "inputImage", "imageUrl": "data:;base64,AA=="}, "invalid_tool_image_url"),
            ({"type": "inputAudio", "audioUrl": "https://example.invalid/a.wav"}, "invalid_tool_audio_url"),
            ({"type": "inputVideo", "videoUrl": "data:video/mp4;base64,AA=="}, "invalid_native_content_item"),
            ({"type": "inputText"}, "invalid_native_content_item"),
            ("text", "invalid_native_content_item"),
        ]
        for item, code in cases:
            with self.subTest(item=item):
                with self.assertRaises(ValueError) as caught:
                    encode_mcp_result({"success": True, "contentItems": [item]})
                self.assertEqual(str(caught.exception), code)
        for envelope in (None, {"success": 1, "contentItems": []}, {"success": True}, {"success": True, "contentItems": {}}):
            with self.assertRaises(ValueError) as caught:
                encode_mcp_result(envelope)
            self.assertEqual(str(caught.exception), "invalid_tool_result")


# ---------------------------------------------------------------------------------------------------------------------
class RawLogRoundTripTest(ServerCase):
    def test_log_round_trips_through_agent_raw_evidence(self):
        self.fake.results["look"] = {"success": True, "contentItems": [
            {"type": "inputText", "text": "view"}, {"type": "inputImage", "imageUrl": "data:image/png;base64," + PNG_B64}]}
        self.client.handshake(discover_probe=True)
        self.client.list_tools()
        native = self.client.call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_img"}, progress_token=1)
        generated = self.client.call_tool_then_cancel("plain", {}, progress_token=2)
        self.fake.gate("move")
        pending = self.client.start_call("move", {}, progress_token=3)
        self.assertTrue(self.fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        self.client.cancel(pending.request_id)
        self.fake.gates["move"].set()
        pending.wait(TIMEOUT)
        self.assertTrue(self.fake.finished.setdefault("move", threading.Event()).wait(TIMEOUT))
        records = self.raw_records()
        self.assertTrue(all(record["channel"] == "mcp" for record in records))
        calls = [r for r in records if r["message"].get("method") == "tools/call"]
        self.assertTrue(all(r["direction"] == "received" for r in calls))
        ids = {call["rpc_id"]: call["call_id"] for call in self.fake.calls}
        self.assertEqual([r["message"]["params"]["_meta"]["item_id"] for r in calls],
                         ["toolu_img", ids[generated.request_id], ids[pending.request_id]])
        self.assertEqual(calls[0]["message"]["params"]["_meta"]["claudecode/toolUseId"], "toolu_img")
        responses = [r for r in records if r["direction"] == "sent"]
        self.assertTrue(all("id" in r["message"] for r in responses))
        received_methods = [r["message"].get("method") for r in records if r["direction"] == "received"]
        self.assertEqual(received_methods[:4], ["server/discover", "initialize", "notifications/initialized",
                                                "tools/list"])
        self.assertEqual(received_methods.count("notifications/cancelled"), 2)
        self.assertFalse(any(r["message"].get("method") == "notifications/progress" for r in records))
        evidence = load_native_evidence(self.tmp.name, dict(thread_id="thr", agent_backend=dict(kind="claude_code")),
                                        self.store)
        self.assertEqual(evidence.tool_item_ids, {"toolu_img", ids[generated.request_id]})
        self.assertEqual(evidence.sent_image_sha256, {hashlib.sha256(PNG).hexdigest()})
        self.assertEqual(native.result["content"][1]["data"], PNG_B64)


class DispatcherOrderingTest(ServerCase):
    """Concurrent MCP calls through the session-style closure keep the OrderedBarrier semantics."""

    def setUp(self):
        super().setUp()
        self.events, self.lock = [], threading.Lock()
        self.submitted = {}
        self.gates = {name: threading.Event() for name in ("A", "D")}
        self.rendezvous = threading.Barrier(2, timeout=TIMEOUT)

        def handler(params):
            name = params["arguments"]["tag"]
            with self.lock:
                self.events.append(("start", name))
            if name in self.gates:
                self.gates[name].wait(TIMEOUT)
            if name in ("B", "C"):
                self.rendezvous.wait()  # completes only if B and C overlap
            with self.lock:
                self.events.append(("end", name))
            return text_result(name)

        def parallel_safe(tool, arguments):
            return tool == "look"

        self.dispatcher = ToolDispatcher(handler, parallel_safe, max_workers=8)
        self.addCleanup(self.dispatcher.shutdown, wait=False)

        def call_tool(name, arguments, *, call_id, native_call_id, rpc_id, native_thread_id=None):
            future = self.dispatcher.submit(dict(tool=name, arguments=arguments, callId=call_id))
            self.submitted[arguments["tag"]].set()
            return future.result()

        self.server.call_tool = call_tool

    def start(self, tool, tag):
        self.submitted[tag] = threading.Event()
        pending = self.client.start_call(tool, {"tag": tag}, progress_token=tag)
        self.assertTrue(self.submitted[tag].wait(TIMEOUT))
        return pending

    def test_parallel_calls_overlap_and_ordered_calls_wait(self):
        a = self.start("move", "A")
        b = self.start("look", "B")
        c = self.start("look", "C")
        d = self.start("move", "D")
        time.sleep(0.1)
        self.assertEqual(self.events, [("start", "A")])  # B, C wait for the earlier ordered call
        self.gates["A"].set()
        for pending, tag in ((b, "B"), (c, "C")):
            self.assertEqual(pending.wait(TIMEOUT).result["content"][0]["text"], tag)
        self.assertTrue(wait_until(lambda: ("start", "D") in self.events))
        self.assertFalse(d.done())
        self.gates["D"].set()
        self.assertEqual(d.wait(TIMEOUT).result["content"][0]["text"], "D")
        self.assertEqual(a.wait(TIMEOUT).result["content"][0]["text"], "A")
        order = [event for event in self.events]
        self.assertLess(order.index(("end", "A")), order.index(("start", "B")))
        self.assertLess(order.index(("end", "A")), order.index(("start", "C")))
        self.assertLess(max(order.index(("end", "B")), order.index(("end", "C"))), order.index(("start", "D")))
        self.assertEqual(self.failures, [])

# ==== tests/test_agent_mcp_session.py ====

MCP_SESS_IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
MCP_SESS_PLANTED_HASH = "9f86d081884c7d659a2feaa0c55ad015a3bf4f1b2b0b822cd15d6c15b0f00a08"
MCP_SESS_SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
MCP_SESS_TOOLS = [{"type": "function", "name": "look", "description": "Read-only view.", "inputSchema": MCP_SESS_SCHEMA},
         {"type": "function", "name": "move", "description": "Stateful motion.", "inputSchema": MCP_SESS_SCHEMA}]


def mcp_sess_text_result(text, success=True):
    return {"success": success, "contentItems": [{"type": "inputText", "text": text}]}


def mcp_sess_wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class StubSession(McpSessionMixin, AgentSession):
    """Minimal MCP-backend session over the mixin: real Store/transcript/presentation, no native process."""
    backend_kind = "stub_backend"
    counting = GENERATION_COUNTING

    def __init__(self, root, store, handler, *, parallel_safe=None):
        self.lock = threading.RLock()
        self.root, self.store, self.private = root, store, root / "private"
        self.private.mkdir(parents=True, exist_ok=True)
        self.presentation = ActorPresentation(store)
        self.tool_handler, self.tool_parallel_safe = handler, parallel_safe
        self.state, self.thread_id, self.turn_id = "idle", "thread-1", None
        self.turn_done, self.turn_result = threading.Event(), None
        self.transcript = TranscriptWriter(store, backend=self.backend_kind, thread_id=self.thread_id)
        self.closing, self.close_report = False, None
        self.model, self.provider, self.price_card = "model-a", "provider-a", None
        self.config, self.wire = Config(), AnthropicMessagesWire(client="claude_code")
        self.model_proxy, self.model_limit_reached, self.bridge = None, threading.Event(), None
        self.last_activity_at = 0.0
        self.resets, self.disconnects, self.order = [], [], []
        self.on_context_reset = self.resets.append
        self.on_disconnect = lambda: self.disconnects.append(self.state)
        self._identity = None
        self._teardown_confirmed = True

    def _stop_backend(self):
        self.order.append("backend")


class RecordingHandler:
    def __init__(self, results=None, error=None, gate=None):
        self.calls, self.results, self.error, self.gate = [], results or {}, error, gate

    def __call__(self, params):
        self.calls.append(params)
        if self.gate is not None:
            self.gate.wait(5)
        if self.error is not None:
            raise self.error
        return self.results.get(params["tool"], mcp_sess_text_result("ok " + params["tool"]))


class Case(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name) / "run"
        self.store = Store(self.root, MCP_SESS_IDENTITY)
        self.addCleanup(self.store.close)

    def session(self, handler=None, **kwargs):
        self.handler = handler or RecordingHandler()
        return StubSession(self.root, self.store, self.handler, **kwargs)

    def bridge(self, session, hints=None):
        bridge = McpToolBridge(session, MCP_SESS_TOOLS, parallel_hints=hints).start()
        session.bridge = bridge
        self.addCleanup(bridge.join)
        self.addCleanup(bridge.stop_accepting)
        client = FakeMcpClient(bridge.url, headers=bridge.headers)
        client.handshake()
        return bridge, client

    def events(self, kind):
        return self.store.events_by_type([kind], limit=1000)


class EmittedOrderBridgeTests(Case):
    """Real ``NiRuaMcpServer`` + ``McpToolBridge`` + ``ToolDispatcher``: the ``tools/call`` requests of one model step
    arrive inverted over HTTP and still run in the order the model emitted them (the relay's announcement)."""
    OPENCODE_STEP = [(None, "ni_rua_move", {"x": 1}), (None, "bash", {"command": "ls"}),
                     (None, "ni_rua_move", {"x": 2}), (None, "ni_rua_unknown_tool", {}), (None, "ni_rua_move", {"x": 3})]

    def running_bridge(self, session, *, prefix, wait_s=5.0, hints=None):
        session.tool_name_prefix = prefix
        bridge = McpToolBridge(session, MCP_SESS_TOOLS, parallel_hints=hints, order_wait_s=wait_s).start()
        session.bridge = bridge
        self.addCleanup(bridge.join)
        self.addCleanup(bridge.stop_accepting)
        client = FakeMcpClient(bridge.url, headers=bridge.headers)
        client.handshake()
        session._begin_turn()
        return bridge, client

    def send_one_by_one(self, bridge, client, calls):
        """Send ``(tool, arguments, meta)`` in this order, each only once the previous one is held by the bridge or
        has run (so the arrival order is exactly this order); returns the results in the same order."""
        pending = []
        for tool, arguments, meta in calls:
            held, ran = bridge.order.held, len(self.handler.calls)
            pending.append(client.start_call(tool, arguments, meta=meta))
            mcp_sess_wait_until(lambda: bridge.order.held > held or len(self.handler.calls) > ran)
        return [call.wait(10) for call in pending]

    def executed(self):
        return [(params["tool"], params["arguments"]["x"]) for params in self.handler.calls]

    def test_inverted_arrival_without_call_ids_runs_in_emitted_order(self):
        """OpenCode: no call id, the session id in ``_meta``; arrival 3, 2, 1."""
        session = self.session()
        bridge, client = self.running_bridge(session, prefix="ni_rua_")
        session._model_tool_calls("ses_1", self.OPENCODE_STEP)
        session._model_tool_calls("ses_child", [(None, "ni_rua_move", {"x": 3})])    # a newer step of another session
        meta = {"ai.opencode/sessionID": "ses_1"}
        results = self.send_one_by_one(bridge, client, [("move", {"x": x}, meta) for x in (3, 2, 1)])
        self.assertEqual(self.executed(), [("move", 1), ("move", 2), ("move", 3)])
        self.assertTrue(all(result.result["isError"] is False for result in results))
        self.assertEqual(self.events("agent.tool_order_wait_expired"), [])
        self.assertEqual(bridge.order.held, 0)

    def test_inverted_arrival_with_native_call_ids_runs_in_emitted_order(self):
        """Claude Code: the model's ``tool_use`` id in ``_meta``; no session id."""
        session = self.session()
        bridge, client = self.running_bridge(session, prefix="mcp__ni_rua__")
        session._model_tool_calls("thread-1", [(f"toolu_{x}", "mcp__ni_rua__move", {"x": x}) for x in (1, 2, 3)])
        self.send_one_by_one(bridge, client, [("move", {"x": x}, {"claudecode/toolUseId": f"toolu_{x}"})
                                              for x in (2, 3, 1)])
        self.assertEqual(self.executed(), [("move", 1), ("move", 2), ("move", 3)])
        self.assertEqual([params["callId"] for params in self.handler.calls], ["toolu_1", "toolu_2", "toolu_3"])

    def test_parallel_safe_calls_still_overlap_and_the_ordered_call_follows_them(self):
        together = threading.Barrier(2)

        class Handler(RecordingHandler):
            def __call__(self, params):
                if params["tool"] == "look":
                    together.wait(5)
                return super().__call__(params)
        session = self.session(Handler(), parallel_safe=lambda tool, arguments: tool == "look")
        bridge, client = self.running_bridge(session, prefix="ni_rua_", hints={"look": True})
        session._model_tool_calls("ses_1", [(None, "ni_rua_look", {"x": 1}), (None, "ni_rua_look", {"x": 2}),
                                            (None, "ni_rua_move", {"x": 3})])
        meta = {"ai.opencode/sessionID": "ses_1"}
        self.send_one_by_one(bridge, client, [("move", {"x": 3}, meta), ("look", {"x": 2}, meta),
                                              ("look", {"x": 1}, meta)])
        self.assertFalse(together.broken, "the parallel-safe calls did not overlap")
        self.assertEqual((sorted(self.executed()[:2]), self.executed()[2:]), ([("look", 1), ("look", 2)], [("move", 3)]))

    def unmatched(self):
        return [(event["payload"], event["correlation"]["tool_call_id"])
                for event in self.events("agent.tool_order_unmatched")]

    def test_a_call_without_an_exact_place_runs_in_arrival_order_and_records_why(self):
        """Every reason: nothing is held or guessed, the call runs on arrival, and the reason is an observer event
        correlated with the call (a correlation that stopped working must not be silent)."""
        session = self.session()
        bridge, client = self.running_bridge(session, prefix="ni_rua_")
        meta = {"ai.opencode/sessionID": "ses_1"}
        # 1. no step known: nothing was announced (inverted arrival stays inverted).
        self.send_one_by_one(bridge, client, [("move", {"x": x}, meta) for x in (3, 2, 1)])
        self.assertEqual(self.executed(), [("move", 3), ("move", 2), ("move", 1)])
        # 2. session-key mismatch: the relay announced the step under another key than the client sends.
        session._model_tool_calls("thread-default", [(None, "ni_rua_move", {"x": x}) for x in (4, 5)])
        self.send_one_by_one(bridge, client, [("move", {"x": 5}, meta), ("move", {"x": 4}, meta)])
        # 3. arguments differ: the client rewrote them; no match by tool name. 4. the exact call still has its place.
        session._model_tool_calls("ses_1", [(None, "ni_rua_move", {"x": 6}), (None, "ni_rua_move", {"x": 7})])
        self.send_one_by_one(bridge, client, [("move", {"x": 7, "rewritten": True}, meta), ("move", {"x": 6}, meta)])
        # 5. not in the step, 6. no session key and no call id, 7. a call id the model never emitted.
        client.call_tool("look", {"x": 8}, meta=meta)
        client.call_tool("move", {"x": 9})
        client.call_tool("move", {"x": 10}, meta={"claudecode/toolUseId": "toolu_unknown"})
        self.assertEqual(self.executed(), [("move", 3), ("move", 2), ("move", 1), ("move", 5), ("move", 4), ("move", 7),
                                           ("move", 6), ("look", 8), ("move", 9), ("move", 10)])
        by_x = {params["arguments"]["x"]: params["callId"] for params in self.handler.calls}
        expected = [(x, "move", "no_step_known") for x in (3, 2, 1)]
        expected += [(5, "move", "session_key_mismatch"), (4, "move", "session_key_mismatch"),
                     (7, "move", "arguments_differ"), (8, "look", "not_in_step"), (9, "move", "no_session_key"),
                     (10, "move", "call_id_unknown")]
        self.assertEqual(self.unmatched(), [(dict(tool=tool, reason=reason), by_x[x]) for x, tool, reason in expected])
        self.assertEqual((self.events("agent.tool_order_wait_expired"), bridge.order.held), ([], 0))

    def test_a_matched_call_records_no_unmatched_event_and_a_refused_call_records_none(self):
        session = self.session()
        bridge, client = self.running_bridge(session, prefix="ni_rua_")
        session._model_tool_calls("ses_1", [(None, "ni_rua_move", {"x": 1})])
        client.call_tool("move", {"x": 1}, meta={"ai.opencode/sessionID": "ses_1"})
        session._finish_turn("completed")
        client.call_tool("move", {"x": 2})                    # stale and unmatched: refused, nothing recorded
        self.assertEqual((self.executed(), self.unmatched()), ([("move", 1)], []))

    def test_a_held_call_whose_turn_ended_or_changed_is_refused_as_stale(self):
        """Review 30 M1: the hold sits between admission and dispatch; a call released after its turn ended, after a
        new turn began or while the session closes must not run (and must not carry the old turn id)."""
        real_monotonic = time.monotonic
        endings = (("ended", lambda s: s._finish_turn("interrupted")),
                   ("new_turn", lambda s: (s._finish_turn("interrupted"), s._begin_turn())),
                   ("closing", lambda s: setattr(s, "closing", True)))
        for label, end in endings:
            with self.subTest(label):
                offset = [0.0]
                with mock.patch("ni_rua.cortex.agents.dispatch.time.monotonic",
                                side_effect=lambda: real_monotonic() + offset[0]):
                    session = self.session()
                    bridge, client = self.running_bridge(session, prefix="ni_rua_", wait_s=2.0)
                    session._model_tool_calls("ses_1", [(None, "ni_rua_move", {"x": 1}), (None, "ni_rua_move", {"x": 2})])
                    pending = client.start_call("move", {"x": 2}, meta={"ai.opencode/sessionID": "ses_1"})
                    self.assertTrue(mcp_sess_wait_until(lambda: bridge.order.held == 1))
                    end(session)                              # during the hold; move(1) is never sent
                    self.assertEqual(bridge.order.held, 1)
                    with bridge.order._condition:
                        offset[0] += 3.0
                        bridge.order._condition.notify_all()
                    result = pending.wait(10)
                self.assertTrue(result.result["isError"])
                self.assertIn("native_tool_call_stale_connection", result.result["content"][0]["text"])
                self.assertEqual(self.handler.calls, [])
                self.assertEqual([row for row in self.store.states("codex_item") if row.get("type") == "dynamicToolCall"], [])
                self.assertEqual(bridge.order.held, 0)
                session.closing = False
                bridge.stop_accepting()

    def test_transcript_rows_start_in_execution_order(self):
        session = self.session()
        bridge, client = self.running_bridge(session, prefix="ni_rua_")
        session._model_tool_calls("ses_1", [(None, "ni_rua_move", {"x": x}) for x in (1, 2)])
        started = []
        tool_started = session.transcript.tool_started
        session.transcript.tool_started = lambda call_id, name, arguments, **kw: (
            started.append(arguments["x"]), tool_started(call_id, name, arguments, **kw))[1]
        meta = {"ai.opencode/sessionID": "ses_1"}
        self.send_one_by_one(bridge, client, [("move", {"x": 2}, meta), ("move", {"x": 1}, meta)])
        self.assertEqual((started, self.executed()), ([1, 2], [("move", 1), ("move", 2)]))

    def test_an_earlier_call_that_never_arrives_is_waited_for_once_and_reported(self):
        session = self.session()
        bridge, client = self.running_bridge(session, prefix="ni_rua_", wait_s=0.05)
        session._model_tool_calls("ses_1", [(None, "ni_rua_move", {"x": x}) for x in (1, 2, 3)])
        meta = {"ai.opencode/sessionID": "ses_1"}
        started = time.monotonic()
        results = self.send_one_by_one(bridge, client, [("move", {"x": 3}, meta), ("move", {"x": 2}, meta)])
        elapsed = time.monotonic() - started
        self.assertEqual(self.executed(), [("move", 2), ("move", 3)])
        self.assertTrue(all(result.result["isError"] is False for result in results))     # typed results, nothing refused
        self.assertTrue(0.05 <= elapsed < 0.5 + 2, elapsed)
        expired, = self.events("agent.tool_order_wait_expired")
        payload = dict(expired["payload"])
        waited = payload.pop("waited_s")
        self.assertEqual(payload, dict(tool="move", position=payload["position"], step_size=3, matched_by="arguments",
                                       missing_positions=[0]))
        self.assertIn(payload["position"], (1, 2))
        self.assertGreaterEqual(waited, 0.05)
        call_id = next(params["callId"] for params in self.handler.calls
                       if params["arguments"]["x"] == payload["position"] + 1)
        self.assertEqual(expired["correlation"]["tool_call_id"], call_id)
        # The missing call arrives after all: it runs on arrival.
        client.call_tool("move", {"x": 1}, meta=meta)
        self.assertEqual(self.executed()[-1], ("move", 1))
        self.assertEqual(len(self.events("agent.tool_order_wait_expired")), 1)

    def test_a_refused_earlier_call_does_not_hold_the_later_one(self):
        session = self.session()
        bridge, client = self.running_bridge(session, prefix="ni_rua_")
        session._model_tool_calls("ses_1", [(None, "ni_rua_move", {"x": 1}), (None, "ni_rua_move", {"x": 2})])
        meta = {"ai.opencode/sessionID": "ses_1"}
        turn = session.turn_id
        session.turn_id = None                               # stale: the first call is refused before dispatch
        refused = client.call_tool("move", {"x": 1}, meta=meta)
        self.assertIn("native_tool_call_stale_connection", refused.result["content"][0]["text"])
        session.turn_id = turn
        started = time.monotonic()
        client.call_tool("move", {"x": 2}, meta=meta)
        self.assertLess(time.monotonic() - started, 2)
        self.assertEqual(self.executed(), [("move", 2)])
        self.assertEqual(self.events("agent.tool_order_wait_expired"), [])

    def test_relay_hook_is_the_session_method_and_ignores_a_missing_bridge(self):
        session = self.session()
        session._model_tool_calls("ses_1", self.OPENCODE_STEP)       # before the bridge exists: nothing to do
        proxy = session._start_relay("http://127.0.0.1:9", upstream_auth_headers={"x-api-key": "k"},
                                     request_projection=lambda request: request, default_thread_id="thread-1",
                                     image_settings=dict(reuse_exact_images=False, trim_oversized_requests=False,
                                                         trim_max_inline_images=0))
        self.addCleanup(proxy.close)
        self.assertEqual(proxy.on_tool_calls, session._model_tool_calls)


class BridgeTests(Case):
    def running(self, session):
        session._begin_turn()
        return session.turn_id

    def test_native_call_id_reaches_the_handler_and_names_the_transcript_row(self):
        session = self.session(RecordingHandler(results={"look": mcp_sess_text_result("seen sha256 " + MCP_SESS_PLANTED_HASH)}))
        bridge, client = self.bridge(session, hints={"look": True})
        turn = self.running(session)
        result = client.call_tool("look", {"x": 1}, meta={"claudecode/toolUseId": "toolu_01"})
        self.assertFalse(result.result["isError"])
        params, = self.handler.calls
        self.assertEqual(params, dict(threadId="thread-1", turnId=turn, callId="toolu_01", tool="look",
                                      arguments={"x": 1}, callIdSource="native_tool_use_id",
                                      contextScope="isolated"))
        row = self.store.transcript_item("toolu_01")
        self.assertEqual((row["type"], row["status"], row["success"], row["turn_id"], row["native_id"]),
                         ("dynamicToolCall", "completed", True, turn, "toolu_01"))
        # The second presentation scrub removed the planted digest from the delivered text and the row.
        self.assertNotIn(MCP_SESS_PLANTED_HASH, result.result["content"][0]["text"])
        self.assertNotIn(MCP_SESS_PLANTED_HASH, row["contentItems"][0]["text"])
        self.assertEqual(bridge.read_only_tools, ["look"])
        self.assertGreater(session.last_activity_at, 0)

    def test_call_without_native_id_gets_a_host_call_id(self):
        session = self.session()
        _, client = self.bridge(session)
        self.running(session)
        client.call_tool("move", {"x": 2})
        params, = self.handler.calls
        self.assertTrue(params["callId"].startswith("call-"))
        self.assertEqual(params["callIdSource"], "mcp_request")
        self.assertEqual(self.store.transcript_item(params["callId"])["tool"], "move")

    def test_stale_connection_is_refused_before_dispatch(self):
        for label, mutate in (("closing", lambda s: setattr(s, "closing", True)),
                              ("disconnected", lambda s: setattr(s, "state", "disconnected")),
                              ("closed", lambda s: setattr(s, "state", "closed")),
                              ("not_started", lambda s: setattr(s, "state", "not_started")),
                              ("replay", lambda s: setattr(s, "state", "replay")),
                              ("no_turn", lambda s: setattr(s, "turn_id", None)),
                              ("no_thread", lambda s: setattr(s, "thread_id", None)),
                              # Review A1: after its turn's terminal a call has no turn that owns it.
                              ("completed", lambda s: s._finish_turn("completed")),
                              ("interrupted", lambda s: s._finish_turn("interrupted")),
                              ("idle", lambda s: setattr(s, "state", "idle"))):
            with self.subTest(label):
                session = self.session()
                _, client = self.bridge(session)
                self.running(session)
                mutate(session)
                result = client.call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_" + label})
                self.assertTrue(result.result["isError"])
                self.assertIn("native_tool_call_stale_connection", result.result["content"][0]["text"])
                self.assertEqual(self.handler.calls, [])
                self.assertIsNone(self.store.transcript_item("toolu_" + label))

    def test_handler_exception_is_an_error_result_and_a_failed_row(self):
        session = self.session(RecordingHandler(error=ValueError("tool_exploded")))
        _, client = self.bridge(session)
        self.running(session)
        result = client.call_tool("move", {}, meta={"claudecode/toolUseId": "toolu_err"})
        self.assertTrue(result.result["isError"])
        self.assertIn("tool_exploded", result.result["content"][0]["text"])
        row = self.store.transcript_item("toolu_err")
        self.assertEqual((row["status"], row["success"]), ("failed", False))

    def test_uncertain_delivery_changes_the_epoch_and_reports_a_context_reset(self):
        gate = threading.Event()
        session = self.session(RecordingHandler(gate=gate))
        _, client = self.bridge(session)
        self.running(session)
        pending = client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_cancel"})
        self.assertTrue(mcp_sess_wait_until(lambda: self.handler.calls))
        client.cancel(pending.request_id, reason="AbortError: remote-cancel")
        self.assertTrue(mcp_sess_wait_until(lambda: session.resets))
        gate.set()
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((epoch["source"], epoch["tool_call_id"], epoch["thread_id"]),
                         ("tool_reply_delivery_uncertain", "toolu_cancel", "thread-1"))
        self.assertTrue(epoch["marker"].startswith("image-delivery-failure-"))
        self.assertEqual(session.resets, ["tool_reply_delivery_uncertain"])
        unknown, = self.events("agent.tool_reply_unknown")
        self.assertEqual((unknown["payload"], unknown["correlation"]["tool_call_id"]),
                         ({"reason": "tool_reply_delivery_uncertain"}, "toolu_cancel"))

    def test_delivery_failure_never_raises(self):
        session = self.session()
        bridge = McpToolBridge(session, MCP_SESS_TOOLS)
        self.addCleanup(bridge.join)
        session.store = None  # every write fails
        bridge._delivery_failure("call-x", "tool_reply_delivery_uncertain")
        self.assertEqual(session.resets, ["tool_reply_delivery_uncertain"])
        self.assertTrue(any(entry[0].startswith("delivery_failure_invalidate_context:") for entry in bridge.server.errors))

    def test_post_execution_transcript_or_presentation_failure_never_claims_no_motion(self):
        receipt_json = json.dumps({"status": "completed", "executed_steps": 12})
        session = self.session(RecordingHandler(results={"move": mcp_sess_text_result(receipt_json)}))
        bridge, client = self.bridge(session)
        self.running(session)
        session.transcript.tool_completed = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("store_closed"))
        delivered = client.call_tool("move", {"x": 1}, meta={"claudecode/toolUseId": "toolu_receipt"})
        self.assertFalse(delivered.result["isError"])
        self.assertEqual(json.loads(delivered.result["content"][0]["text"]), {"status": "completed", "executed_steps": 12})
        self.assertTrue(any(entry[0] == "tool_completed:toolu_receipt" for entry in bridge.server.errors))

        session2 = self.session(RecordingHandler(results={"move": mcp_sess_text_result(receipt_json)}))
        bridge2, client2 = self.bridge(session2)
        self.running(session2)
        session2.presentation.present = lambda value: (_ for _ in ()).throw(RuntimeError("alias_store_failed"))
        failed_present = client2.call_tool("move", {"x": 2}, meta={"claudecode/toolUseId": "toolu_pres_fail"},
                                           accept_sse=False)
        self.assertEqual(failed_present.response.messages[0]["error"]["code"], -32603)
        self.assertIn("invalid_tool_result", failed_present.response.messages[0]["error"]["message"])
        self.assertTrue(mcp_sess_wait_until(lambda: session2.resets == ["tool_reply_encoding_failed"]))

    def test_internal_tool_bug_logs_observer_traceback_and_scrubs_host_paths(self):
        session = self.session(RecordingHandler(error=TypeError("bad operand in /Users/secret/private/worker.py:42")))
        _, client = self.bridge(session)
        self.running(session)
        result = client.call_tool("move", {"x": 1}, meta={"claudecode/toolUseId": "toolu_bug"})
        self.assertTrue(result.result["isError"])
        payload = json.loads(result.result["content"][0]["text"])
        self.assertEqual(payload["code"], "internal_runtime_error")
        self.assertEqual(payload["error_category"], "internal_runtime_error")
        self.assertFalse(payload["retryable"])
        self.assertIsNone(payload["minimal_example"])
        self.assertNotIn("/Users/secret", json.dumps(payload))
        internal_events = self.events("agent.tool_internal_error")
        self.assertEqual(len(internal_events), 1)
        self.assertEqual(internal_events[0]["payload"]["error_type"], "TypeError")
        self.assertTrue(internal_events[0]["payload"]["traceback_ref"].startswith("artifact:"))

        session_to = self.session(RecordingHandler(error=TimeoutError("robot status unknown: deadline /tmp/sock")))
        _, client_to = self.bridge(session_to)
        self.running(session_to)
        timeout_res = client_to.call_tool("move", {"x": 1}, meta={"claudecode/toolUseId": "toolu_to"})
        timeout_payload = json.loads(timeout_res.result["content"][0]["text"])
        self.assertEqual(timeout_payload["code"], "execution_status_unknown")
        self.assertEqual(timeout_payload["error_category"], "unknown_execution")
        self.assertFalse(timeout_payload["retryable"])
        self.assertNotIn("/tmp/sock", json.dumps(timeout_payload))

    def test_short_request_body_disconnect_is_dropped_cleanly(self):
        session = self.session()
        bridge, client = self.bridge(session)
        self.running(session)
        parsed = http.client.urlsplit(bridge.url)
        with socket.create_connection((parsed.hostname, parsed.port), timeout=2) as sock:
            raw_req = (
                f"POST {parsed.path} HTTP/1.1\r\n"
                f"Host: {parsed.hostname}:{parsed.port}\r\n"
                f"Authorization: {bridge.headers['Authorization']}\r\n"
                "Content-Type: application/json\r\n"
                "Content-Length: 120\r\n\r\n"
                '{"jsonrpc":"2.0"'
            ).encode()
            sock.sendall(raw_req)
        self.assertEqual(self.handler.calls, [])
        ok = client.call_tool("look", {"x": 1})
        self.assertFalse(ok.result["isError"])

    def test_join_waits_for_the_handler_threads_transcript_writes(self):
        # Close joins the tool workers, then the gateway closes the Store: a handler thread still writing its
        # transcript row after join returned would use a closed sqlite connection (seen as a segfault).
        gate = threading.Event()
        session = self.session(RecordingHandler(gate=gate))
        bridge, client = self.bridge(session)
        self.running(session)
        original, written = session.transcript.tool_completed, []
        completion_started, allow_completion = threading.Event(), threading.Event()
        self.addCleanup(allow_completion.set)

        def slow_completion(*args, **kwargs):
            completion_started.set()
            allow_completion.wait(5)
            written.append(original(*args, **kwargs))
        session.transcript.tool_completed = slow_completion
        pending = client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_join"})
        self.assertTrue(mcp_sess_wait_until(lambda: self.handler.calls))
        gate.set()
        self.assertTrue(completion_started.wait(2))
        join_done = threading.Event()
        join_thread = threading.Thread(target=lambda: (bridge.join(), join_done.set()))
        join_thread.start()
        self.assertFalse(join_done.wait(0.02))
        allow_completion.set()
        join_thread.join(2)
        self.assertTrue(join_done.is_set())
        self.assertEqual(written, ["toolu_join"])
        pending.wait()
        blocker = threading.Event()
        self.addCleanup(blocker.set)
        session.transcript.tool_completed = lambda *args, **kwargs: blocker.wait(5)
        with mock.patch.object(bridge, "_active", 1):
            with self.assertRaisesRegex(RuntimeError, "^tool_handlers_still_running$"):
                bridge.join(timeout_s=0.02)

    def test_parallel_safe_calls_overlap_and_ordered_calls_wait(self):
        gate, active, peak = threading.Event(), [], []
        lock = threading.Lock()

        def handler(params):
            with lock:
                active.append(params["tool"])
                peak.append(len(active))
            if params["tool"] == "look":
                gate.wait(5)
            with lock:
                active.remove(params["tool"])
            return mcp_sess_text_result("ok")

        session = self.session(handler, parallel_safe=lambda tool, arguments: tool == "look")
        _, client = self.bridge(session)
        self.running(session)
        first = client.start_call("look", {}, meta={"claudecode/toolUseId": "toolu_a"})
        second = client.start_call("look", {}, meta={"claudecode/toolUseId": "toolu_b"})
        self.assertTrue(mcp_sess_wait_until(lambda: max(peak or [0]) == 2))
        ordered = client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_c"})
        self.assertTrue(mcp_sess_wait_until(lambda: self.store.transcript_item("toolu_c") is not None))
        self.assertIsNone(self.store.transcript_item("toolu_c").get("success"))
        gate.set()
        for call in (first, second, ordered):
            self.assertFalse(call.wait().result["isError"])
        self.assertEqual(max(peak), 2)


class TurnHelperTests(Case):
    def test_begin_enters_running_with_a_host_turn_id(self):
        session = self.session()
        previous = session._begin_turn()
        self.assertEqual(previous[0], "idle")
        self.assertEqual(session.state, "running")
        self.assertTrue(session.turn_id.startswith("turn-"))
        self.assertEqual(session.transcript.turn_id, session.turn_id)
        self.assertEqual(session.active_turn_model, "model-a")
        self.assertFalse(session.turn_done.is_set())
        started, = self.events("agent.turn_started")
        self.assertEqual(started["correlation"]["turn_id"], session.turn_id)

    def test_begin_refuses_while_running_and_when_not_connected(self):
        session = self.session()
        session._begin_turn()
        with self.assertRaisesRegex(ValueError, "^turn_already_active$"):
            session._begin_turn()
        for state in ("disconnected", "closed", "unknown", "not_started", "replay"):
            session.state = state
            with self.subTest(state), self.assertRaisesRegex(ValueError, "^session_not_connected$"):
                session._begin_turn()

    def test_store_failure_in_begin_changes_nothing(self):
        # Review (b)1: the turn record is written first; a refusal keeps turn_done set and the previous result.
        session = self.session()
        session._begin_turn()
        session._finish_turn("completed")
        kept = (session.state, session.turn_id, session.turn_result, session.transcript.turn_id)
        original = session.transcript.event
        session.transcript.event = lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("database is locked"))
        with self.assertRaisesRegex(RuntimeError, "database is locked"):
            session._begin_turn()
        session.transcript.event = original
        self.assertEqual((session.state, session.turn_id, session.turn_result, session.transcript.turn_id), kept)
        self.assertTrue(session.turn_done.is_set())

    def test_adopted_turn_origin(self):
        session = self.session()
        session._begin_turn()
        session._finish_turn("completed")
        session._begin_turn(origin="native_wakeup")   # no admissibility check: the native runtime already runs it
        self.assertEqual((session.state, session.turn_origin), ("running", "native_wakeup"))
        self.assertEqual([event["payload"]["origin"] for event in self.events("agent.turn_started")],
                         ["host", "native_wakeup"])
        with self.assertRaisesRegex(ValueError, "^invalid_turn_origin$"):
            session._begin_turn(origin="other")

    def test_reject_restores_the_previous_turn(self):
        session = self.session()
        session.turn_id, session.turn_result, session.state = "turn-old", {"status": "completed"}, "completed"
        session.transcript.turn_id = "turn-old"
        session.active_turn_model = "model-old"
        previous = session._begin_turn()
        rejected = session.turn_id
        with self.assertRaises(AgentRejected) as raised:
            session._reject_turn(previous, "not_connected", "backend_not_connected")
        self.assertEqual((raised.exception.code, raised.exception.message, raised.exception.busy),
                         ("not_connected", "backend_not_connected", False))
        self.assertEqual((session.state, session.turn_id, session.turn_result, session.active_turn_model,
                          session.transcript.turn_id),
                         ("completed", "turn-old", {"status": "completed"}, "model-old", "turn-old"))
        self.assertTrue(session.turn_done.is_set())
        event, = self.events("agent.turn_start_rejected")
        self.assertEqual(event["correlation"]["turn_id"], rejected)
        self.assertEqual(event["payload"]["code"], "not_connected")

    def test_finish_stamps_the_final_answer_once(self):
        session = self.session()
        session._begin_turn()
        session.transcript.assistant_text("thinking aloud", message_id="m1")
        session.transcript.tool_started("call-1", "look", {}, message_id="m1")
        session.transcript.assistant_text("FINAL", message_id="m2")
        self.assertTrue(session._finish_turn("completed"))
        self.assertFalse(session._finish_turn("failed", "late"))
        self.assertEqual(session.turn_result, dict(id=session.turn_id, status="completed", error=None))
        self.assertEqual(session.state, "completed")
        self.assertTrue(session.turn_done.is_set())
        self.assertEqual(session.completed_final_answer(), "FINAL")
        completed, = self.events("agent.turn_completed")
        self.assertEqual(completed["payload"], {"status": "completed", "error": None})

    def test_finish_with_error_keeps_a_host_message(self):
        session = self.session()
        session._begin_turn()
        session._finish_turn("failed", "claude_code_turn_failed:api_error:403")
        self.assertEqual(session.turn_result["error"], {"message": "claude_code_turn_failed:api_error:403"})
        self.assertIsNone(session.completed_final_answer())

    def test_disconnected_is_reported_once_with_a_synthetic_failed_turn(self):
        session = self.session()
        session._begin_turn()
        session._disconnected(ConnectionResetError("pipe"))
        session._disconnected(OSError("again"))
        self.assertEqual(session.state, "disconnected")
        self.assertEqual(session.disconnects, ["disconnected"])
        self.assertEqual(session.turn_result["status"], "failed")
        self.assertEqual(session.turn_result["error"]["message"], "upstream_connection_lost: ConnectionResetError")
        self.assertTrue(session.turn_done.is_set())
        lost, = self.events("agent.connection_lost")
        self.assertEqual(lost["payload"]["reason"], "ConnectionResetError")
        self.assertFalse(lost["payload"]["history_complete"])
        # A terminal after the disconnect changes nothing.
        self.assertFalse(session._finish_turn("completed"))
        with self.assertRaisesRegex(ValueError, "session_not_connected"):
            session._begin_turn()

    def test_disconnected_while_idle_keeps_the_previous_turn_result(self):
        session = self.session()
        session._begin_turn()
        session._finish_turn("completed")
        kept = session.turn_result
        session._disconnected(EOFError("stream_closed"))
        self.assertIs(session.turn_result, kept)
        self.assertEqual(session.disconnects, ["disconnected"])

    def test_disconnected_is_ignored_while_closing(self):
        session = self.session()
        session.closing = True
        session._disconnected(EOFError())
        self.assertEqual((session.state, session.disconnects), ("idle", []))

    def test_disconnect_listener_failure_still_sets_turn_done(self):
        session = self.session()
        session._begin_turn()
        session.on_disconnect = lambda: (_ for _ in ()).throw(RuntimeError("listener"))
        with self.assertRaises(RuntimeError):
            session._disconnected(EOFError())
        self.assertTrue(session.turn_done.is_set())

    def test_context_invalidation_and_relay_discontinuity(self):
        session = self.session()
        session._invalidate_context("agent_resume_after_unclean_exit")
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual(epoch["source"], "agent_resume_after_unclean_exit")
        session._relay_discontinuity("history_rewritten")
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((epoch["source"], epoch["signal"]), ("relay_context_discontinuity", "history_rewritten"))
        self.assertEqual(session.resets, ["agent_resume_after_unclean_exit", "relay_context_discontinuity"])
        compaction, = self.events("agent.compaction")
        self.assertEqual(compaction["payload"], {"signal": "history_rewritten", "source": "relay"})
        session.on_context_reset = lambda reason: (_ for _ in ()).throw(RuntimeError("listener"))
        session._context_reset("ignored")  # never raises

        # Store failure in _invalidate_context / _relay_discontinuity still resets in-memory Harness context.
        failing_session = self.session()
        failing_session.store = SimpleNamespace(
            set_state=lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("sqlite_busy"))
        )
        with self.assertRaisesRegex(RuntimeError, "sqlite_busy"):
            failing_session._relay_discontinuity("images_removed")
        self.assertEqual(failing_session.resets, ["relay_context_discontinuity"])


class FakeProxy:
    def __init__(self, order):
        self.order, self.request_count, self.attempt_count = order, 0, 0
        self.transport_blocked, self.blocked_error = False, None
        self.transport_recovery_summary = dict(policy=None, rounds=0)

    def cancel_pending(self, reason):
        self.order.append(("cancel_pending", reason))

    def reopen_thread(self, thread_id):
        self.order.append(("reopen_thread", thread_id))
        return True

    def close(self):
        self.order.append("relay_close")


class RelayAndCloseTests(Case):
    def test_close_order_and_clean_shutdown(self):
        session = self.session()
        proxy = session.model_proxy = FakeProxy(session.order)
        bridge = McpToolBridge(session, MCP_SESS_TOOLS).start()
        session.bridge = bridge
        original_stop, original_join = bridge.stop_accepting, bridge.join
        bridge.stop_accepting = lambda: (session.order.append("mcp_close"), original_stop())
        bridge.join = lambda: (session.order.append("tool_join"), original_join())
        session._identity = dict(backend="stub_backend", thread_id="thread-1", clean_shutdown=False)
        report = session.close()
        self.assertEqual(session.order, [("cancel_pending", "session_closed"), "backend", "relay_close", "mcp_close",
                                         "tool_join"])
        self.assertIs(proxy, session.model_proxy)
        self.assertTrue(report["clean_shutdown"])
        self.assertEqual(self.store.read_state("runtime", "agent")["clean_shutdown"], True)
        self.assertEqual((session.state, session.turn_done.is_set()), ("closed", True))
        self.assertIs(session.close(), report)  # idempotent
        self.assertEqual(len(self.events("agent.session_closed")), 1)

    def test_close_during_a_turn_is_not_clean_and_collects_step_errors(self):
        session = self.session()
        session._identity = dict(backend="stub_backend", clean_shutdown=False)
        session._begin_turn()

        def broken():
            raise RuntimeError("backend_stuck")
        session._stop_backend = broken
        report = session.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertEqual(report["steps"]["backend"], "failed")
        self.assertIn("backend_stuck", report["errors"][0]["error"])
        self.assertFalse(self.store.read_state("runtime", "agent")["clean_shutdown"])
        self.assertTrue(session.turn_done.is_set())
        # Tool calls after close are stale.
        with self.assertRaisesRegex(ValueError, "native_tool_call_stale_connection"):
            McpToolBridge(session, MCP_SESS_TOOLS)._call_tool("look", {}, call_id="c", native_call_id=None, rpc_id=1)

    def test_native_turn_in_progress_at_close_is_not_clean(self):
        session = self.session()
        session._native_active = lambda: True   # e.g. a native wake-up the host has not adopted
        report = session.close()
        self.assertFalse(report["clean_shutdown"])
        self.assertTrue(report["native_active_at_close"])

    def test_unconfirmed_teardown_is_not_clean(self):
        session = self.session()
        session._teardown_confirmed = False
        self.assertFalse(session.close()["clean_shutdown"])

    def test_model_events_account_relay_usage_of_this_thread_only(self):
        session = self.session()
        session._begin_turn()
        proxy = session.model_proxy = FakeProxy([])
        session._relay_used, session._relay_attempts, session._attempt_usage = 3, 4, {}
        usage = {"input_tokens": 100, "output_tokens": 7, "cache_read_input_tokens": 50,
                 "cache_creation_input_tokens": 0}
        proxy.request_count, proxy.attempt_count = 1, 1
        session._model_event(dict(kind="model_request_finished", logical_call_id="t:1", usage=usage, usage_known=True))
        session._model_event(dict(kind="model_call_completed", logical_call_id="t:1", native_thread_id="thread-1"))
        session._model_event(dict(kind="model_request_finished", logical_call_id="t:2", usage=usage, usage_known=True))
        session._model_event(dict(kind="model_call_completed", logical_call_id="t:2", native_thread_id="other"))
        requests = self.store.read_state("runtime", "model_requests")
        self.assertEqual((requests["completed_calls"], requests["http_attempt_count"], requests["counting"],
                          requests["limit"]), (4, 5, GENERATION_COUNTING, 180))
        relay = self.store.read_state("runtime", "model_usage_relay")
        self.assertEqual(relay["calls"], 1)
        self.assertEqual(relay["total"]["output_tokens"], 7)
        accounting = self.store.read_state("runtime", "token_accounting")
        self.assertEqual(accounting["source"], "relay.anthropic_messages.usage.sum")
        # A delivered call without usage makes the totals unknown, never zero.
        session._model_event(dict(kind="model_request_finished", logical_call_id="t:3", usage=None, usage_known=False))
        session._model_event(dict(kind="model_call_completed", logical_call_id="t:3", native_thread_id="thread-1"))
        relay = self.store.read_state("runtime", "model_usage_relay")
        # The Anthropic wire never reports reasoning tokens separately, so every call has an unknown counter.
        self.assertEqual((relay["calls"], relay["unknown_calls"]), (2, 2))
        self.assertIsNone(relay["total"]["output_tokens"])
        self.assertIsNone(relay["total"]["reasoning_output_tokens"])
        self.assertEqual(len(self.events("model.model_call_completed")), 3)

    def test_reopen_relay_thread_lifts_the_interrupt_tombstone(self):
        session = self.session()
        self.assertFalse(session._reopen_relay_thread())  # no relay yet
        session.model_proxy = FakeProxy(session.order)
        self.assertTrue(session._reopen_relay_thread())
        self.assertEqual(session.order, [("reopen_thread", "thread-1")])

    def test_budget_resume_rules(self):
        session = self.session()
        self.assertEqual(session._relay_budget(), (0, 0, session.config.model.transport_recovery.to_dict()))
        self.store.set_state("runtime", "model_requests", dict(counting=GENERATION_COUNTING, completed_calls=12,
                                                               http_attempt_count=15,
                                                               transport_recovery=dict(policy=None)))
        self.assertEqual(session._relay_budget(), (12, 15, None))
        self.store.set_state("runtime", "model_requests", dict(counting="other", completed_calls=1))
        with self.assertRaisesRegex(ValueError, "^resume_model_budget_semantics_changed"):
            session._relay_budget()
        self.store.set_state("runtime", "model_requests", dict(counting=GENERATION_COUNTING, completed_calls=180))
        with self.assertRaisesRegex(ValueError, "^episode_model_request_limit_exhausted$"):
            session._relay_budget()

    def test_start_relay_builds_the_budgeted_proxy(self):
        session = self.session()
        self.store.set_state("runtime", "model_requests", dict(counting=GENERATION_COUNTING, completed_calls=170,
                                                               transport_recovery=dict(policy=None)))
        settings = dict(reuse_exact_images=False, trim_oversized_requests=True, trim_max_inline_images=0)
        proxy = session._start_relay("http://127.0.0.1:9", upstream_auth_headers={"x-api-key": "host-secret"},
                                     request_projection=lambda request: request, default_thread_id="thread-1",
                                     image_settings=settings)
        self.addCleanup(proxy.close)
        self.assertIs(session.model_proxy, proxy)
        self.assertEqual((proxy.max_requests, proxy.default_thread_id, proxy.wire.name), (10, "thread-1",
                                                                                          "anthropic_messages"))
        self.assertTrue(proxy.base_url.startswith("http://127.0.0.1:"))
        self.assertEqual(self.store.read_state("runtime", "model_image_policy")["oversized_trim"], True)


class DeltaLeaseAndSettlementTests(Case):
    """验证 MCP 单主写回租约、四条件合取主链判据、排队取消零字节不播种新纪元及并发兄弟写回竞态隔离。"""

    def _real_harness_session(self, *, context_delivery="delta", parallel_safe=None):
        from ni_rua.runtime.harness import Harness
        from support.fixture_robot import FixtureRobot

        robot = FixtureRobot(self.root)
        harness = Harness(self.store, robot)
        harness.start()
        harness.thread_id = "thread-1"
        harness.context_epoch_volatile = context_delivery == "full"
        session = self.session(
            harness.handle_tool,
            parallel_safe=parallel_safe if parallel_safe is not None else harness.tool_registry.parallel_safe,
        )
        session.context_delivery = context_delivery
        session.tool_name_prefix = "mcp__ni_rua__"
        session.on_context_reset = harness.model_context_reset
        return harness, session

    def _start_bridge(self, session, tools=None, *, hints=None, wait_s=5.0):
        catalog = tools if tools is not None else [
            {"type": "function", "name": "robot_capabilities", "description": "Read capabilities.",
             "inputSchema": {"type": "object", "properties": {"detail": {"type": "string"}}}},
            {"type": "function", "name": "robot_observe", "description": "Observe scene.",
             "inputSchema": {"type": "object", "properties": {}}},
        ]
        bridge = McpToolBridge(session, catalog, parallel_hints=hints, order_wait_s=wait_s).start()
        session.bridge = bridge
        self.addCleanup(bridge.join)
        self.addCleanup(bridge.stop_accepting)
        client = FakeMcpClient(bridge.url, headers=bridge.headers)
        client.handshake()
        session._begin_turn()
        return bridge, client

    def test_on_call_settled_reports_delivery_truth_across_outcomes(self):
        settled = []
        fake = FakeCallTool()
        server = NiRuaMcpServer(
            tools=TOOLS,
            presentation=ActorPresentation(self.store),
            call_tool=fake,
            raw_log=lambda direction, message: self.store.raw_agent(direction, message, channel="mcp"),
            on_delivery_failure=lambda call_id, reason: None,
            on_call_settled=lambda call_id, delivered: settled.append((call_id, delivered)),
            keepalive_s=0.05,
            peer_check_s=0.02,
        ).start()
        self.addCleanup(server.close)
        client = FakeMcpClient(server.url, token=server.token)

        # 1) 正常送达 -> delivered=True
        res = client.call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_ok"}, accept_sse=False)
        self.assertFalse(res.result["isError"])
        self.assertTrue(mcp_sess_wait_until(lambda: ("toolu_ok", True) in settled))

        # 2) 编码失败 -> delivered=False
        fake.results["look"] = {"success": True, "contentItems": [
            {"type": "inputImage", "imageUrl": "https://example.invalid/bad.png"}]}
        err = client.call_tool("look", {}, meta={"claudecode/toolUseId": "toolu_enc"}, accept_sse=False)
        self.assertEqual(err.error["code"], -32603)
        self.assertTrue(mcp_sess_wait_until(lambda: ("toolu_enc", False) in settled))

        # 3) 在途取消 -> delivered=False
        gate = fake.gate("move")
        self.addCleanup(gate.set)
        pending = client.start_call("move", {}, meta={"claudecode/toolUseId": "toolu_abn"}, progress_token=1)
        self.assertTrue(fake.started.setdefault("move", threading.Event()).wait(TIMEOUT))
        client.cancel(pending.request_id)
        gate.set()
        pending.wait(TIMEOUT)
        self.assertTrue(mcp_sess_wait_until(lambda: ("toolu_abn", False) in settled))

    def test_conjunctive_caller_is_main_and_ticket_purpose(self):
        harness, session = self._real_harness_session(context_delivery="delta")
        recorded = []
        real_handle = harness.handle_tool

        def spy(params):
            recorded.append(dict(params))
            return real_handle(params)

        session.tool_handler = spy
        session.harness = harness
        bridge, client = self._start_bridge(session, hints={"robot_capabilities": True})

        # 主链且匹配 thread -> contextScope="main" + contextEpoch
        session.on_tool_calls("thread-1", [("toolu_main", "mcp__ni_rua__robot_capabilities", {"detail": "summary"})], "main")
        r_main = client.call_tool("robot_capabilities", {"detail": "summary"},
                                  meta={"claudecode/toolUseId": "toolu_main"}, accept_sse=False)
        self.assertFalse(r_main.result["isError"])
        self.assertEqual(recorded[-1]["contextScope"], "main")
        self.assertEqual(recorded[-1]["contextEpoch"], harness.decision_context.model_context_epoch())
        self.assertEqual(recorded[-1]["threadId"], "thread-1")

        # 非主链 purpose="subagent" -> isolated 且保留主 threadId
        session.on_tool_calls("thread-1", [("toolu_sub", "mcp__ni_rua__robot_capabilities", {"detail": "summary"})], "subagent")
        client.call_tool("robot_capabilities", {"detail": "summary"},
                         meta={"claudecode/toolUseId": "toolu_sub"}, accept_sse=False)
        self.assertEqual(recorded[-1]["contextScope"], "isolated")
        self.assertNotIn("contextEpoch", recorded[-1])
        self.assertEqual(recorded[-1]["threadId"], "thread-1")

        # purpose=None -> isolated
        session.on_tool_calls("thread-1", [("toolu_none", "mcp__ni_rua__robot_capabilities", {"detail": "summary"})], None)
        client.call_tool("robot_capabilities", {"detail": "summary"},
                         meta={"claudecode/toolUseId": "toolu_none"}, accept_sse=False)
        self.assertEqual(recorded[-1]["contextScope"], "isolated")

        # OpenCode 风格 native_thread_id 不等于主 session.thread_id -> isolated
        session.on_tool_calls("ses_child", [(None, "mcp__ni_rua__robot_capabilities", {"detail": "summary"})], "main")
        client.call_tool("robot_capabilities", {"detail": "summary"},
                         meta={"ai.opencode/sessionID": "ses_child"}, accept_sse=False)
        self.assertEqual(recorded[-1]["contextScope"], "isolated")

        # 未匹配 step（ticket is None）-> isolated
        client.call_tool("robot_capabilities", {"detail": "summary"},
                         meta={"claudecode/toolUseId": "toolu_unmatched"}, accept_sse=False)
        self.assertEqual(recorded[-1]["contextScope"], "isolated")

        # context_delivery="full" 即使主链也一律 isolated
        session.context_delivery = "full"
        session.on_tool_calls("thread-1", [("toolu_full", "mcp__ni_rua__robot_capabilities", {"detail": "summary"})], "main")
        client.call_tool("robot_capabilities", {"detail": "summary"},
                         meta={"claudecode/toolUseId": "toolu_full"}, accept_sse=False)
        self.assertEqual(recorded[-1]["contextScope"], "isolated")

    def test_queued_cancel_zero_bytes_never_seeds_new_epoch(self):
        """双 threading.Event 确定性复现：排队等待前置调用期间被取消，释放后不得在已翻转的新纪元 E2 播种基线。"""
        harness, session = self._real_harness_session(context_delivery="delta")
        bridge, client = self._start_bridge(session, hints={"robot_capabilities": True})

        # 宣告同步 2 个调用：toolu_first 未到达前，toolu_queued 会在 EmittedOrder.wait 阻塞
        session.on_tool_calls(
            "thread-1",
            [
                ("toolu_first", "mcp__ni_rua__robot_capabilities", {"detail": "summary"}),
                ("toolu_queued", "mcp__ni_rua__robot_capabilities", {"detail": "summary"}),
            ],
            "main",
        )
        queued_entered = threading.Event()
        queued_done = threading.Event()
        orig_handle = harness.handle_tool

        def wrapped_handle(params):
            if params["callId"] == "toolu_queued":
                queued_entered.set()
            try:
                return orig_handle(params)
            finally:
                if params["callId"] == "toolu_queued":
                    queued_done.set()

        session.tool_handler = wrapped_handle
        bridge.dispatcher.handler = wrapped_handle
        session.harness = harness

        epoch_before = harness.decision_context.model_context_epoch()
        pending_queued = client.start_call(
            "robot_capabilities", {"detail": "summary"},
            meta={"claudecode/toolUseId": "toolu_queued"}, progress_token=1,
        )
        self.assertTrue(mcp_sess_wait_until(lambda: bridge.order.held == 1))

        # 客户端取消排队中的 toolu_queued -> _abandon 先翻转 image_context_epoch 至 E2
        self.assertEqual(client.cancel(pending_queued.request_id).status, 202)
        self.assertTrue(mcp_sess_wait_until(lambda: harness.decision_context.model_context_epoch() != epoch_before))
        epoch_e2 = harness.decision_context.model_context_epoch()

        # 通过发送 toolu_first 放行排队的 toolu_queued（实际中前置调用先到或超时释放均会唤醒）
        first_res = client.call_tool(
            "robot_capabilities", {"detail": "summary"},
            meta={"claudecode/toolUseId": "toolu_first"}, accept_sse=False,
        )
        self.assertFalse(first_res.result["isError"])
        self.assertEqual(json.loads(first_res.result["content"][0]["text"])["capabilities_delivery"]["status"], "inline")
        self.assertTrue(queued_done.wait(TIMEOUT))
        pending_queued.wait(TIMEOUT)

        # 再次清空 E2 到 E3 并模拟“仅被取消的调用单独进入写回”：绝不能播种 E3
        session._invalidate_context("test_reset_e3")
        epoch_e3 = harness.decision_context.model_context_epoch()
        self.assertNotEqual(epoch_e3, epoch_e2)

        # 新步骤：toolu_a 进入 handler 后阻塞，被取消后翻转至 E4，验证迟到 toolu_a 不会污染 E4
        a_entered = threading.Event()
        a_release = threading.Event()
        self.addCleanup(a_release.set)

        def gated_handle(params):
            if params["callId"] == "toolu_a":
                a_entered.set()
                a_release.wait(TIMEOUT)
            return orig_handle(params)

        session.tool_handler = gated_handle
        bridge.dispatcher.handler = gated_handle

        session.on_tool_calls("thread-1", [("toolu_a", "mcp__ni_rua__robot_capabilities", {"detail": "summary"})], "main")
        pending_a = client.start_call(
            "robot_capabilities", {"detail": "summary"},
            meta={"claudecode/toolUseId": "toolu_a"}, progress_token=2,
        )
        self.assertTrue(a_entered.wait(TIMEOUT))
        self.assertEqual(client.cancel(pending_a.request_id).status, 202)
        self.assertTrue(mcp_sess_wait_until(lambda: harness.decision_context.model_context_epoch() != epoch_e3))
        a_release.set()
        pending_a.wait(TIMEOUT)
        self.assertTrue(mcp_sess_wait_until(lambda: bridge._lease_call_id is None))

        # 取消后首个真正送达主链的调用必须收到完整 inline，绝不能收到 unchanged
        session.on_tool_calls("thread-1", [("toolu_after", "mcp__ni_rua__robot_capabilities", {"detail": "summary"})], "main")
        after = client.call_tool(
            "robot_capabilities", {"detail": "summary"},
            meta={"claudecode/toolUseId": "toolu_after"}, accept_sse=False,
        )
        self.assertEqual(json.loads(after.result["content"][0]["text"])["capabilities_delivery"]["status"], "inline")

    def test_concurrent_readonly_siblings_writeback_race_isolated_without_blocking(self):
        """双 threading.Event 确定性复现并发兄弟 A/B 写回竞态：A 持租约在写回前被取消，B 不阻塞且必须返回完整 inline。"""
        harness, session = self._real_harness_session(context_delivery="delta")
        a_entered = threading.Event()
        a_release = threading.Event()
        self.addCleanup(a_release.set)
        orig_handle = harness.handle_tool

        def race_handle(params):
            result = orig_handle(params)
            if params["callId"] == "toolu_sib_a":
                # A 已在 Harness 内完成 project() 播种基线，但在 HTTP 写回客户端前卡住
                a_entered.set()
                a_release.wait(TIMEOUT)
            return result

        session.tool_handler = race_handle
        session.harness = harness
        bridge, client = self._start_bridge(session, hints={"robot_capabilities": True})

        session.on_tool_calls(
            "thread-1",
            [
                ("toolu_sib_a", "mcp__ni_rua__robot_capabilities", {"detail": "summary"}),
                ("toolu_sib_b", "mcp__ni_rua__robot_capabilities", {"detail": "summary"}),
            ],
            "main",
        )
        pending_a = client.start_call(
            "robot_capabilities", {"detail": "summary"},
            meta={"claudecode/toolUseId": "toolu_sib_a"}, progress_token=1,
        )
        self.assertTrue(a_entered.wait(TIMEOUT))
        self.assertEqual(bridge._lease_call_id, "toolu_sib_a")

        # 并发兄弟 B 到达：不阻塞等待 A，直接降级 isolated 返回完整 inline（严禁引用未送达的 A）
        res_b = client.call_tool(
            "robot_capabilities", {"detail": "summary"},
            meta={"claudecode/toolUseId": "toolu_sib_b"}, accept_sse=False,
        )
        self.assertFalse(res_b.result["isError"])
        payload_b = json.loads(res_b.result["content"][0]["text"])
        self.assertEqual(payload_b["capabilities_delivery"]["status"], "inline")

        # 此时取消 A（0 字节到达客户端）并放行 A 退出
        self.assertEqual(client.cancel(pending_a.request_id).status, 202)
        a_release.set()
        pending_a.wait(TIMEOUT)
        self.assertTrue(mcp_sess_wait_until(lambda: bridge._lease_call_id is None))

        # 取消后的新主链调用 C 拿租约返回 inline，顺序后续 D 复用基线返回 unchanged
        session.on_tool_calls("thread-1", [("toolu_c", "mcp__ni_rua__robot_capabilities", {"detail": "summary"})], "main")
        res_c = client.call_tool(
            "robot_capabilities", {"detail": "summary"},
            meta={"claudecode/toolUseId": "toolu_c"}, accept_sse=False,
        )
        self.assertEqual(json.loads(res_c.result["content"][0]["text"])["capabilities_delivery"]["status"], "inline")

        session.on_tool_calls("thread-1", [("toolu_d", "mcp__ni_rua__robot_capabilities", {"detail": "summary"})], "main")
        res_d = client.call_tool(
            "robot_capabilities", {"detail": "summary"},
            meta={"claudecode/toolUseId": "toolu_d"}, accept_sse=False,
        )
        self.assertEqual(json.loads(res_d.result["content"][0]["text"])["capabilities_delivery"]["status"], "unchanged")

if __name__ == "__main__":
    unittest.main()
