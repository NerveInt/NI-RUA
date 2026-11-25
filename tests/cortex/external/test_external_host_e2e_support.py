"""CPU tests of the host end-to-end support (``support.host_e2e``, the scene's slow mode and the fake model's Codex
namespace): loopback only, no host binary, no gateway process, no model provider, no arm."""

NI_RUA_TEST_TIER = "component"

import base64
import http.client
import json
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from support import host_e2e, so101_scene
from support.fake_mcp_client import FakeMcpClient
from support.fake_model_servers import FakeResponsesServer, Step, ToolCall
from support.host_e2e import (ConformanceModel, ScriptCall, codex_projection_problems, mcp_exchanges, request_view,
                              script_of)

PNG = b"\x89PNG\r\n\x1a\n" + bytes(range(64))


def post(base_url, path, body):
    """One POST of a JSON body; returns the parsed JSON answer or the raw SSE text."""
    connection = http.client.HTTPConnection("127.0.0.1", int(base_url.rsplit(":", 1)[1]), timeout=10)
    connection.request("POST", path, json.dumps(body), {"Content-Type": "application/json"})
    response = connection.getresponse()
    data = response.read().decode()
    connection.close()
    return json.loads(data) if response.getheader("Content-Type", "").startswith("application/json") else data


class NamespaceToolCallTests(unittest.TestCase):
    def test_responses_function_call_carries_the_namespace(self):
        step = Step(tool_calls=[ToolCall("robot_observe", {}, "call_1", namespace="mcp__ni_rua"),
                                ToolCall("exec_command", {"cmd": "ls"}, "call_2")])
        with FakeResponsesServer([step, step]) as server:
            body = post(server.base_url, "/v1/responses", {"model": "m", "stream": False})
            streamed = post(server.base_url, "/v1/responses", {"model": "m", "stream": True})
        first, second = body["output"]
        self.assertEqual((first["name"], first["namespace"]), ("robot_observe", "mcp__ni_rua"))
        self.assertNotIn("namespace", second)
        added = [json.loads(line[6:]) for line in streamed.splitlines() if line.startswith("data: ")]
        items = [event["item"] for event in added if event["type"] == "response.output_item.added"]
        self.assertEqual([item.get("namespace") for item in items], ["mcp__ni_rua", None])


class HostTargetPathTests(unittest.TestCase):
    def test_user_binary_falls_back_to_path_and_the_override_wins(self):
        with tempfile.TemporaryDirectory() as directory:
            binary = Path(directory) / "codex"
            binary.write_text("#!/bin/sh\n")
            binary.chmod(0o755)
            target = host_e2e.HostTarget("t", "codex", "NI_RUA_TEST_NO_SUCH_VARIABLE", Path(directory) / "missing",
                                         "0.169.0", False, None, False, "codex")
            with mock.patch.dict("os.environ", {"PATH": directory}):
                self.assertEqual(target.path, binary)
            with mock.patch.dict("os.environ", {"PATH": directory, "NI_RUA_TEST_NO_SUCH_VARIABLE": "/nonexistent"}):
                self.assertIsNone(target.path)         # an explicit override never falls back
            with mock.patch.dict("os.environ", {"PATH": "/nonexistent"}):
                self.assertIsNone(target.path)
        self.assertEqual(host_e2e.TARGETS["codex-user"].named_version, "0.169.0")

    def test_user_binary_skips_active_env_prefix_on_path_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            env_root = Path(directory) / "env"
            env_bin = env_root / "bin"
            user_bin = Path(directory) / "user-bin"
            env_bin.mkdir(parents=True)
            user_bin.mkdir(parents=True)
            for folder in (env_bin, user_bin):
                exe = folder / "codex"
                exe.write_text("#!/bin/sh\n")
                exe.chmod(0o755)
            target = host_e2e.HostTarget(
                "codex-user", "codex", "NI_RUA_TEST_NO_SUCH_VARIABLE",
                (Path(directory) / "missing-a", Path(directory) / "missing-b"),
                "0.169.0", False, None, False, "codex",
            )
            with mock.patch.object(host_e2e.sys, "prefix", str(env_root)), \
                    mock.patch.dict("os.environ", {"PATH": f"{env_bin}:{user_bin}"}):
                self.assertEqual(target.path, user_bin / "codex")


class RequestViewTests(unittest.TestCase):
    def test_anthropic(self):
        body = {"system": [{"type": "text", "text": "sys"}], "tools": [{"name": "mcp__x__a", "input_schema": {}}],
                "messages": [
                    {"role": "user", "content": "go NI-RUA-E2E-SCRIPT=main"},
                    {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_e2e_main_0_0",
                                                       "name": "mcp__x__a", "input": {"k": 1}}]},
                    {"role": "user", "content": [
                        {"type": "tool_result", "tool_use_id": "toolu_e2e_main_0_0", "content": [
                            {"type": "text", "text": "one"}, {"type": "image", "source": {
                                "type": "base64", "media_type": "image/png", "data": base64.b64encode(PNG).decode()}},
                            {"type": "text", "text": "two"}]},
                        {"type": "text", "text": "<task-notification>done</task-notification>"}]}]}
        view = request_view("anthropic", body)
        self.assertEqual((list(view.tools), view.system, script_of(view)), (["mcp__x__a"], ["sys"], "main"))
        self.assertEqual(view.calls, [dict(id="toolu_e2e_main_0_0", name="mcp__x__a", namespace=None,
                                           arguments={"k": 1})])
        result = view.results["toolu_e2e_main_0_0"]
        self.assertEqual((result.texts, [image.data for image in result.images], result.is_error),
                         (["one", "two"], [PNG], False))
        self.assertEqual(view.last_user_text, "<task-notification>done</task-notification>")

    def test_responses(self):
        image = "data:image/webp;base64," + base64.b64encode(PNG).decode()
        body = {"instructions": "base", "tools": [
                    {"type": "function", "name": "exec_command"},
                    {"type": "namespace", "name": "mcp__ni_rua", "description": "instr", "tools": []}],
                "input": [
                    {"type": "message", "role": "user",
                     "content": [{"type": "input_text", "text": "NI-RUA-E2E-SCRIPT=w"}]},
                    {"type": "function_call", "call_id": "call_e2e_w_0_0", "name": "run_begin",
                     "namespace": "mcp__ni_rua", "arguments": "{}"},
                    {"type": "function_call_output", "call_id": "call_e2e_w_0_0", "output": [
                        {"type": "input_text", "text": "Wall time: 0 seconds"},
                        {"type": "input_text", "text": "brief"}, {"type": "input_image", "image_url": image}]},
                    {"type": "function_call_output", "call_id": "other", "output": "plain"}]}
        view = request_view("responses", body)
        self.assertEqual((list(view.tools), list(view.namespaces), view.system), (["exec_command"], ["mcp__ni_rua"],
                                                                                  ["base"]))
        self.assertEqual(view.calls[0]["namespace"], "mcp__ni_rua")
        self.assertEqual(view.results["call_e2e_w_0_0"].texts, ["Wall time: 0 seconds", "brief"])
        self.assertEqual(view.results["call_e2e_w_0_0"].images[0].data, PNG)
        self.assertEqual(view.results["other"].texts, ["plain"])
        self.assertEqual(script_of(view), "w")


class FakeHost:
    wire = "anthropic"

    def tool_call(self, call, call_id):
        return ToolCall(call.tool if call.native else "mcp__x__" + call.tool, call.arguments, call_id)


class ConformanceModelTests(unittest.TestCase):
    def setUp(self):
        self.model = ConformanceModel(FakeHost()).start()
        self.addCleanup(self.model.close)
        self.seen = []

        def reactive(conversation):
            self.seen.append(conversation.result(0))
            return [ScriptCall("b", {"from": conversation.result(0).texts[0]}), ScriptCall("Bash", {}, native=True)]
        self.model.add("main", [[ScriptCall("a", {"x": 1})], reactive])

    def generate(self, messages, tools=True):
        body = {"model": "m", "messages": messages, "stream": False}
        if tools:
            body["tools"] = [{"name": "mcp__x__a", "input_schema": {}}]
        return post(self.model.base_url, "/v1/messages", body)["content"]

    def test_script_advances_by_its_own_call_ids(self):
        prompt = {"role": "user", "content": self.model.prompt("main", "go")}
        self.assertEqual(self.generate([prompt], tools=False), [{"type": "text", "text": "ok"}])   # side request
        first, = self.generate([prompt])
        self.assertEqual((first["id"], first["name"], first["input"]), ("toolu_e2e_main_0_0", "mcp__x__a", {"x": 1}))
        history = [prompt, {"role": "assistant", "content": [first]},
                   {"role": "user", "content": [{"type": "tool_result", "tool_use_id": first["id"],
                                                 "content": [{"type": "text", "text": "R0"}]}]}]
        second = self.generate(history)
        self.assertEqual([(block["id"], block["name"], block["input"]) for block in second],
                         [("toolu_e2e_main_1_0", "mcp__x__b", {"from": "R0"}), ("toolu_e2e_main_1_1", "Bash", {})])
        self.assertEqual(self.seen[0].texts, ["R0"])
        self.assertFalse(self.model.done["main"].is_set())
        history += [{"role": "assistant", "content": second}, {"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": block["id"], "content": "ok"} for block in second]}]
        self.assertEqual(self.generate(history), [{"type": "text", "text": "NI-RUA-E2E-DONE main"}])
        self.assertTrue(self.model.done["main"].is_set())
        self.assertEqual([entry["kind"] for entry in self.model.served], ["side", "tools", "tools", "text"])

    def test_unknown_script_and_foreign_ids_are_side_requests(self):
        prompt = {"role": "user", "content": "no marker"}
        self.assertEqual(self.generate([prompt]), [{"type": "text", "text": "ok"}])
        foreign = [{"role": "user", "content": self.model.prompt("main", "go")}, {"role": "assistant", "content": [
            {"type": "tool_use", "id": "toolu_e2e_other_0_0", "name": "mcp__x__a", "input": {}}]}]
        self.assertEqual(self.generate(foreign)[0]["id"], "toolu_e2e_main_0_0")


class CodexProjectionTests(unittest.TestCase):
    PROPERTIES = {
        "mode": {"type": "string", "enum": ["a", "b"], "description": "Mode.", "default": "a"},
        "count": {"type": "integer", "minimum": 1, "maximum": 5},
        "arms": {"type": "array", "maxItems": 1, "items": {"enum": ["right"]}},
        "request": {"type": "object", "description": "Query.", "oneOf": [{"type": "object"}]},
        "memo": {"type": "object", "description": "Free object."}}
    PUBLISHED = {"type": "object", "additionalProperties": False, "examples": [{}], "required": ["mode"],
                 "properties": PROPERTIES}

    def delivered(self):
        return {"type": "object", "additionalProperties": False, "required": ["mode"], "properties": {
            "mode": {"type": "string", "enum": ["a", "b"]}, "count": {"type": "integer"},
            "arms": {"type": "array", "items": {"type": "string", "enum": ["right"]}}, "request": {},
            "memo": {"type": "object", "description": "Free object.", "properties": {}}}}

    def test_known_projection_passes(self):
        self.assertEqual(codex_projection_problems(self.PUBLISHED, self.delivered()), [])

    def test_focus_3d_anyof_object_injection_passes(self):
        published = {
            "type": "object",
            "properties": {
                "focus_3d": {
                    "type": "object",
                    "additionalProperties": False,
                    "anyOf": [{"required": ["target_m"]}, {"required": ["object_id"]}],
                    "properties": {
                        "target_m": {"type": "array", "items": {"type": "number"}},
                        "object_id": {"type": "string"},
                    },
                },
            },
        }
        delivered = {
            "type": "object",
            "properties": {
                "focus_3d": {
                    "type": "object",
                    "additionalProperties": False,
                    "anyOf": [
                        {"type": "object", "properties": {}, "required": ["target_m"]},
                        {"type": "object", "properties": {}, "required": ["object_id"]},
                    ],
                    "properties": {
                        "target_m": {"type": "array", "items": {"type": "number"}},
                        "object_id": {"type": "string"},
                    },
                },
            },
        }
        self.assertEqual(codex_projection_problems(published, delivered), [])

    def test_unexplained_differences_are_reported(self):
        delivered = self.delivered()
        del delivered["properties"]["count"]
        delivered["properties"]["mode"]["enum"] = ["a"]
        delivered["properties"]["memo"]["format"] = "x"
        problems = codex_projection_problems(self.PUBLISHED, delivered)
        self.assertEqual(len(problems), 3, problems)
        self.assertIn("/properties: properties dropped: count", problems)
        dropped_type = self.delivered()
        del dropped_type["properties"]["mode"]["type"]
        self.assertEqual(codex_projection_problems(self.PUBLISHED, dropped_type),
                         ["/properties/mode/type: dropped by the host (not a known Codex projection keyword)"])
        wrong_type = self.delivered()
        wrong_type["properties"]["arms"]["items"]["type"] = "integer"
        self.assertEqual(codex_projection_problems(self.PUBLISHED, wrong_type),
                         ["/properties/arms/items/type: added by the host"])


class McpExchangeTests(unittest.TestCase):
    def test_pairs_requests_with_responses_and_skips_notifications(self):
        def record(direction, message, second):
            return dict(direction=direction, channel="mcp", message=message,
                        received_at=f"2026-11-15T00:00:{second:02d}.000000Z")
        records = [record("received", {"id": 1, "method": "initialize"}, 0),
                   record("received", {"method": "notifications/initialized"}, 1),
                   record("received", {"id": 2, "method": "tools/call", "params": {"name": "a"}}, 2),
                   record("received", {"id": 2, "method": "tools/call", "params": {"name": "b"}}, 3),
                   record("sent", {"id": 1, "result": {"instructions": "i"}}, 4),
                   record("sent", {"id": 2, "result": {"content": []}}, 9),
                   record("received", {"id": 3, "method": "tools/call", "params": {"name": "c"}}, 10)]
        initialize, first, second, third = mcp_exchanges(records)
        self.assertEqual((initialize.result, first.tool, first.duration_s, second.response, third.response),
                         ({"instructions": "i"}, "a", 7.0, None, None))


class SceneHoldTests(unittest.TestCase):
    """The scene's test-only ``execute_hold_s`` delays the answer to an execute after the adapter ran the motion (so
    the plan lifetime is not spent waiting); a stop ends the hold."""

    def backend(self, hold):
        backend = so101_scene.SceneBackend.__new__(so101_scene.SceneBackend)
        backend.lock, backend.failure = threading.Lock(), None
        backend.scene = so101_scene.scene_document("red", execute_hold_s=hold)
        calls = []

        class Adapter:
            def execute(self, payload, report, cancel):
                calls.append(time.monotonic())
                return {"status": "completed"}
        backend.adapter = Adapter()
        return backend, calls

    def test_hold_delays_the_answer_after_the_motion(self):
        backend, calls = self.backend(0.3)
        started = time.monotonic()
        self.assertEqual(backend._request("execute", {"timeout_s": 1}, cancel=threading.Event()),
                         {"status": "completed"})
        self.assertLess(calls[0] - started, 0.1)                 # the motion ran at once
        self.assertGreaterEqual(time.monotonic() - started, 0.3)  # the answer came after the hold

    def test_a_stop_ends_the_hold(self):
        backend, calls = self.backend(30)
        cancel = threading.Event()
        threading.Timer(0.1, cancel.set).start()
        started = time.monotonic()
        self.assertEqual(backend._request("execute", {"timeout_s": 1}, cancel=cancel), {"status": "completed"})
        self.assertLess(time.monotonic() - started, 5)

    def test_default_is_off_and_bad_values_are_refused(self):
        self.assertEqual(so101_scene.scene_document("red")["execute_hold_s"], 0.0)
        for bad in (-1, True, "5", float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                so101_scene.scene_document("red", execute_hold_s=bad)


class HttpLogTests(unittest.TestCase):
    def test_every_endpoint_request_is_logged_with_its_status_and_headers(self):
        from ni_rua.core.presentation.actor_presentation import ActorPresentation
        from ni_rua.core.storage.store import Store
        from ni_rua.cortex.agents import mcp_server
        directory = self.enterContext(tempfile.TemporaryDirectory())
        with mock.patch.object(mcp_server, "BaseHTTPRequestHandler", mcp_server.BaseHTTPRequestHandler):
            log = Path(directory) / "http.jsonl"
            host_e2e.install_http_log(log)
            store = Store(directory, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
            self.addCleanup(store.close)
            server = mcp_server.NiRuaMcpServer(
                tools=[{"type": "function", "name": "look", "description": "d", "inputSchema": {"type": "object"}}],
                presentation=ActorPresentation(store), call_tool=lambda *a, **k: None,
                raw_log=lambda *a: None, on_delivery_failure=lambda *a: None, require_bearer=False).start()
            self.addCleanup(server.close)
            FakeMcpClient(server.url).handshake()
            refused = FakeMcpClient(server.url, headers={"Origin": "http://evil.example"})
            with self.assertRaises(Exception):
                refused.initialize()
            records = [json.loads(line) for line in log.read_text().splitlines()]
        self.assertEqual([(record["method"], record["status"]) for record in records],
                         [("POST", 200), ("POST", 202), ("POST", 403)])
        self.assertEqual(records[2]["headers"]["Origin"], "http://evil.example")
        self.assertNotIn("Origin", records[0]["headers"])


PLACEHOLDER_KEY = "sk-placeholder-not-a-real-key-0123456789abcdef"


def write_profile(directory):
    """A two-wire provider profile with a placeholder key (never a real credential)."""
    path = Path(directory) / "placeholder.local.json"
    path.write_text(json.dumps({"schema": "ni_rua.provider_profile.v1", "name": "placeholder",
                                "credential": {"api_key": PLACEHOLDER_KEY},
                                "wires": {"anthropic_messages": {"base_url": "https://llm.example/anthropic/",
                                                                 "auth_scheme": "x-api-key"},
                                          "openai_responses": {"base_url": "https://llm.example/openai/v1",
                                                               "auth_scheme": "bearer"}}}))
    return path


class UpstreamTests(unittest.TestCase):
    """Live mode reuses the host drivers; the scripted path must stay byte-identical and a real key must reach only the
    host process (never the installer)."""

    def host(self, key, directory):
        target = host_e2e.TARGETS[key]
        return host_e2e.HOST_CLASSES[target.host](target, host_e2e.HostHome(directory, target.host),
                                                  Path(directory) / "log")

    def installed(self, host):
        """``(installer env, opencode.json document or None)`` of ``host.install`` with the installer mocked."""
        calls = []
        with mock.patch.object(host_e2e.HostHome, "installer",
                               lambda home, *arguments, env=None, timeout=180: calls.append(env) or (0, "")):
            host.install(8790)
        path = getattr(host, "config_path", None)
        return calls[0], (json.loads(path.read_text()) if path is not None else None)

    def test_the_scripted_model_path_is_unchanged(self):
        fake = mock.Mock(base_url="http://127.0.0.1:9")
        with tempfile.TemporaryDirectory() as directory:
            claude = self.host("claude-pinned", Path(directory) / "c")
            claude.attach(fake)
            env = claude.env()
            self.assertEqual((env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_API_KEY"]), (fake.base_url, host_e2e.FAKE_KEY))
            self.assertNotIn("ANTHROPIC_DEFAULT_SONNET_MODEL", env)
            self.assertEqual(self.installed(claude)[0]["ANTHROPIC_API_KEY"], host_e2e.FAKE_KEY)
            codex = self.host("codex-pinned", Path(directory) / "x")
            codex.attach(fake)
            self.assertEqual(codex.overrides(), [
                "-c", 'model_provider="ni_rua_e2e"', "-c", 'model="fake-model"', "-c",
                'model_providers.ni_rua_e2e.name="fake"', "-c",
                'model_providers.ni_rua_e2e.base_url="http://127.0.0.1:9/v1"', "-c",
                'model_providers.ni_rua_e2e.wire_api="responses"', "-c",
                'model_providers.ni_rua_e2e.env_key="NI_RUA_E2E_FAKE_KEY"'])
            self.assertEqual(codex.env()["NI_RUA_E2E_FAKE_KEY"], host_e2e.FAKE_KEY)
            opencode = self.host("opencode-pinned", Path(directory) / "o")
            opencode.attach(fake)
            _, document = self.installed(opencode)
            self.assertEqual(document, {"model": "fake/fake-model", "providers": {"fake": {
                "name": "fake", "package": "@opencode/ai/providers/anthropic-compatible",
                "settings": {"baseURL": "http://127.0.0.1:9/v1", "apiKey": host_e2e.FAKE_KEY},
                "models": {"fake-model": {"name": "fake-model", "capabilities": {
                    "tools": True, "input": ["text", "image"], "output": ["text"]},
                    "limit": {"context": 1000000, "output": 32000}}}}}})

    def test_a_live_key_reaches_only_the_host_process(self):
        with tempfile.TemporaryDirectory() as directory:
            profile = write_profile(directory)
            upstream = host_e2e.live_upstream(profile, "claude", "model-a")
            self.assertNotIn(PLACEHOLDER_KEY, repr(upstream))
            self.assertEqual(upstream.secrets(), [PLACEHOLDER_KEY.encode(), PLACEHOLDER_KEY[-20:].encode()])
            claude = self.host("claude-pinned", Path(directory) / "c")
            claude.use_upstream(upstream)
            env = claude.env()
            self.assertEqual((env["ANTHROPIC_BASE_URL"], env["ANTHROPIC_API_KEY"]),
                             ("https://llm.example/anthropic", PLACEHOLDER_KEY))
            self.assertEqual(env["ANTHROPIC_DEFAULT_SONNET_MODEL"], "model-a")
            installer_env, _ = self.installed(claude)
            self.assertNotIn(PLACEHOLDER_KEY, json.dumps(installer_env))

            codex = self.host("codex-pinned", Path(directory) / "x")
            codex.use_upstream(host_e2e.live_upstream(profile, "codex", "model-b"))
            self.assertIn('model_providers.ni_rua_live.base_url="https://llm.example/openai/v1"', codex.overrides())
            self.assertIn('model_providers.ni_rua_live.env_key="NI_RUA_E2E_LIVE_KEY"', codex.overrides())
            self.assertNotIn(PLACEHOLDER_KEY, json.dumps(codex.overrides()))
            self.assertEqual(codex.env()["NI_RUA_E2E_LIVE_KEY"], PLACEHOLDER_KEY)
            self.assertNotIn(PLACEHOLDER_KEY, json.dumps(self.installed(codex)[0]))

            opencode = self.host("opencode-pinned", Path(directory) / "o")
            opencode.use_upstream(host_e2e.live_upstream(profile, "opencode", "model-c"))
            installer_env, document = self.installed(opencode)
            self.assertNotIn(PLACEHOLDER_KEY, json.dumps(installer_env))
            self.assertEqual(opencode.config_path.stat().st_mode & 0o777, 0o600)
            provider = document["providers"]["live"]
            self.assertEqual((document["model"], provider["package"], provider["settings"]),
                             ("live/model-c", "@opencode/ai/providers/anthropic-compatible",
                              {"baseURL": "https://llm.example/anthropic/v1", "apiKey": PLACEHOLDER_KEY}))
            responses = host_e2e.live_upstream(profile, "opencode", "model-c", "openai_responses")
            opencode.use_upstream(responses)
            _, document = self.installed(opencode)
            self.assertEqual(document["providers"]["live"]["settings"]["baseURL"], "https://llm.example/openai/v1")

    def test_a_wire_the_host_cannot_speak_is_refused(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "codex cannot use the anthropic_messages wire"):
                host_e2e.live_upstream(write_profile(directory), "codex", "m", "anthropic_messages")

    def test_profile_reference(self):
        self.assertEqual(host_e2e.live_profile_path("opencode-console"),
                         host_e2e.REPOSITORY / "configs" / "profiles" / "opencode-console.local.json")
        self.assertEqual(host_e2e.live_profile_path("/x/y/p.local.json"), Path("/x/y/p.local.json"))


class LiveFactsTests(unittest.TestCase):
    """The per-host transcript readers of live mode, on the shapes the pinned binaries write."""

    def test_claude_stream(self):
        prefix = host_e2e.ClaudeHost.prefix
        messages = [
            {"type": "system", "subtype": "init", "model": "m-1", "permissionMode": "default"},
            {"type": "assistant", "message": {"id": "msg_1", "model": "m-1-served", "content": [
                {"type": "text", "text": "Starting."}, {"type": "tool_use", "id": "t1", "name": "Bash"}]}},
            {"type": "assistant", "message": {"id": "msg_1", "model": "m-1-served", "content": [
                {"type": "tool_use", "id": "t2", "name": prefix + "robot_observe"}]}},
            {"type": "user", "message": {"content": [
                {"type": "tool_result", "tool_use_id": "t1", "is_error": True, "content": "Permission denied\nmore"},
                {"type": "tool_result", "tool_use_id": "t2", "content": [{"type": "text", "text": "{}"}]}]}},
            {"type": "assistant", "message": {"id": "msg_2", "model": "m-1-served", "content": [
                {"type": "text", "text": "CUBE_COLOUR: red"}]}},
            {"type": "result", "subtype": "success", "is_error": False, "result": "done", "num_turns": 2,
             "usage": {"input_tokens": 5}, "total_cost_usd": 0.01,
             "permission_denials": [{"tool_name": "Bash", "tool_use_id": "t1"}]}]
        facts = host_e2e.claude_live_facts(messages)
        self.assertEqual(facts["calls"], [dict(tool="Bash", error="Permission denied"),
                                          dict(tool="ni_rua:robot_observe", error=None)])
        self.assertEqual((facts["requests"], facts["served_models"], facts["configured_model"],
                          facts["permission_mode"], facts["final_text"], facts["permission_denials"]),
                         (2, ["m-1-served"], "m-1", "default", "done", ["Bash"]))
        self.assertEqual(facts["texts"], ["Starting.", "CUBE_COLOUR: red"])
        self.assertEqual(facts["usage"]["total_cost_usd"], 0.01)
        self.assertEqual(facts["errors"], ["Bash: Permission denied"])

    def test_codex_events_and_rollout(self):
        events = [
            {"type": "item.completed", "item": {"id": "i0", "type": "error", "message": "metadata warning"}},
            {"type": "item.started", "item": {"id": "i1", "type": "mcp_tool_call", "server": "ni_rua",
                                              "tool": "run_begin", "status": "in_progress"}},
            {"type": "item.started", "item": {"id": "i2", "type": "command_execution", "command": "ls"}},
            {"type": "item.completed", "item": {"id": "i2", "type": "command_execution", "status": "failed"}},
            {"type": "item.completed", "item": {"id": "i1", "type": "mcp_tool_call", "server": "ni_rua",
                                                "tool": "run_begin", "status": "completed",
                                                "result": {"content": [{"type": "text", "text": "ok"}]}}},
            {"type": "item.completed", "item": {"id": "i3", "type": "agent_message", "text": "CUBE_COLOUR: blue"}},
            {"type": "turn.completed", "usage": {"input_tokens": 9}}]
        rollout = [{"type": "turn_context", "payload": {"model": "gpt-x", "approval_policy": "never",
                                                         "sandbox_policy": {"type": "read-only"}}},
                   {"type": "token_usage_record", "payload": {"response_id": "r1"}},
                   {"type": "token_usage_record", "payload": {"response_id": "r2"}},
                   {"type": "event_msg", "payload": {"type": "token_count"}}]
        facts = host_e2e.codex_live_facts(events, rollout)
        self.assertEqual(facts["calls"], [dict(tool="ni_rua:run_begin", error=None),
                                          dict(tool="command_execution", error="failed")])
        self.assertEqual((facts["requests"], facts["configured_model"], facts["approval_policy"], facts["sandbox"],
                          facts["usage"], facts["final_text"], facts["served_models"]),
                         (2, "gpt-x", "never", "read-only", {"input_tokens": 9}, "CUBE_COLOUR: blue", None))
        self.assertEqual(facts["errors"], ["metadata warning", "command_execution: failed"])

    def test_opencode_timeline(self):
        messages = [
            {"type": "user", "text": "go"},
            {"type": "assistant", "model": {"id": "m-2", "providerID": "live"}, "cost": 0.5,
             "tokens": {"input": 10, "output": 2, "reasoning": 1, "cache": {"read": 4, "write": 3}},
             "content": [{"type": "tool", "name": "ni_rua_run_begin", "state": {"status": "completed"}},
                         {"type": "tool", "name": "shell", "state": {"status": "error", "error": "declined"}}]},
            {"type": "assistant", "model": {"id": "m-2"}, "cost": 0.25, "tokens": {"input": 1, "output": 1},
             "content": [{"type": "text", "text": "CUBE_COLOUR: green"}]},
            {"type": "idle", "outcome": "succeeded"}]
        facts = host_e2e.opencode_live_facts(messages)
        self.assertEqual(facts["calls"], [dict(tool="ni_rua:run_begin", error=None), dict(tool="shell", error="declined")])
        self.assertEqual((facts["requests"], facts["configured_model"], facts["final_text"]), (2, "m-2",
                                                                                             "CUBE_COLOUR: green"))
        self.assertEqual(facts["usage"], dict(input=11, output=3, reasoning=1, cache_read=4, cache_write=3, cost=0.75))
        self.assertEqual(host_e2e.opencode_live_facts([])["usage"], None)


class LiveColourLineTests(unittest.TestCase):
    def test_prose_about_the_cube_colour_is_not_an_answer(self):
        # A real summary (Claude Code, live session 2026-11-15): the answer line, then prose naming "cube colour".
        summary = ("CUBE_COLOUR: red\n\nMoved the end effector 5 mm up. The object_table service was unavailable, so "
                   "I identified the cube colour from the camera images only.")
        self.assertEqual(host_e2e.LIVE_COLOUR_LINE.findall(summary), ["red"])
        self.assertEqual(host_e2e.LIVE_COLOUR_LINE.findall("**CUBE_COLOUR:** Blue"), ["Blue"])
        self.assertEqual(host_e2e.LIVE_COLOUR_LINE.findall("The cube colour is red."), [])


class ScrubTests(unittest.TestCase):
    def test_scrub_overwrites_the_key_and_its_suffix_in_place(self):
        key = PLACEHOLDER_KEY.encode()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "a").mkdir()
            (root / "a" / "db").write_bytes(b"\x00head" + key + b"tail\x00")
            (root / "suffix.json").write_bytes(b'{"approved": ["' + key[-20:] + b'"]}')
            (root / "clean.txt").write_text("nothing here")
            needles = [key, key[-20:]]
            self.assertEqual(host_e2e.files_containing(root, needles), [root / "a" / "db", root / "suffix.json"])
            self.assertEqual(sorted(host_e2e.scrub_files(root, needles)), [root / "a" / "db", root / "suffix.json"])
            self.assertEqual(host_e2e.files_containing(root, needles), [])
            self.assertEqual((root / "a" / "db").read_bytes(), b"\x00head" + b"*" * len(key) + b"tail\x00")

if __name__ == "__main__":
    unittest.main()
