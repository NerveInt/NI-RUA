"""``OpenCodeClient`` (OpenCode 2.2.7 ``/api/*``) against the scripted fake server, over TCP and over a unix socket,
plus one real-binary pass of the same calls (``NI_RUA_OPENCODE_EXECUTABLE`` at the pin, skipped elsewhere).

Shapes asserted here are the ones captured from the pinned binary (probes ``s01``..``s17`` and ``b01``..``b07`` under
``/tmp/ni-rua-agent-harness/oc2/runs``); ``CAPTURED_FRAMES`` are real ``/api/event`` frames (local paths replaced).
"""

NI_RUA_TEST_TIER = "component"

import base64
import json
import os
import shutil
import socket
import subprocess
import tempfile
import threading
import time
import unittest
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from unittest import mock
from urllib.parse import parse_qs
from ni_rua.cortex.agents.opencode import client as oc
from support.fake_opencode import PIN, PASSWORD, FakeOpenCodeServer, reasoning_item, text_item, tool_item
from support.requirements import OPENCODE_EXECUTABLE_ENV, requires_opencode

CAPTURED_FRAMES = [json.loads(line) for line in r"""
{"id":"evt_1d1010c1d001D39fWLeojIlAQl","type":"server.connected","data":{}}
{"id":"evt_1d1010fda001MMplPZuZIEI5OR","created":1794507870170,"type":"session.created","location":{"directory":"/tmp/actor"},"data":{"sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo","slug":"crisp-cactus","version":"2.2.7","projectID":"82d60bc7263af5f7bde5181964ad66ceb2765775","location":{"directory":"/tmp/actor"},"subpath":"","title":"ni-rua-actor"},"durable":{"aggregateID":"ses_e2efef026ffeyA4sKEzv97gIRo","seq":0,"version":1}}
{"id":"evt_1d1010fde002C7KnpkFNEnjON7","created":1794507870174,"type":"session.inbox.enqueued","location":{"directory":"/tmp/actor"},"data":{"inboxID":"msg_1d1010fde001UuDUnm7J0WyXkq","sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo","item":{"type":"user","payload":{"text":"Take one observation then answer."},"delivery":"steer"}},"durable":{"aggregateID":"ses_e2efef026ffeyA4sKEzv97gIRo","seq":1,"version":1}}
{"id":"evt_1d1010fdf001Hv65B8B56Btcr7","created":1794507870175,"type":"session.execution.started","data":{"sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo"},"durable":{"aggregateID":"ses_e2efef026ffeyA4sKEzv97gIRo","seq":2,"version":1}}
{"id":"evt_1d1010ff1001zV2ZFsrWhYJX4j","created":1794507870193,"type":"session.step.started","location":{"directory":"/tmp/actor"},"data":{"sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo","agent":"build","model":{"id":"probe-model","providerID":"relay"},"assistantMessageID":"msg_1d1010fec001LCguNHMaaiq3Mg","started":1794507870190},"durable":{"aggregateID":"ses_e2efef026ffeyA4sKEzv97gIRo","seq":5,"version":1}}
{"id":"evt_1d1010ff4002woCoY2FswlPfEt","created":1794507870196,"type":"session.reasoning.ended","location":{"directory":"/tmp/actor"},"data":{"sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo","assistantMessageID":"msg_1d1010fec001LCguNHMaaiq3Mg","ordinal":0,"text":"think-1","native":{"signature":"fake-signature-0001"}},"durable":{"aggregateID":"ses_e2efef026ffeyA4sKEzv97gIRo","seq":7,"version":1}}
{"id":"evt_1d1010ff50016e6egzC12YSDRm","created":1794507870197,"type":"session.text.delta","location":{"directory":"/tmp/actor"},"data":{"sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo","assistantMessageID":"msg_1d1010fec001LCguNHMaaiq3Mg","ordinal":0,"delta":"Looking."}}
{"id":"evt_1d1010ff7002KbFZ4Q7zJRozbJ","created":1794507870199,"type":"session.tool.called","location":{"directory":"/tmp/actor"},"data":{"sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo","assistantMessageID":"msg_1d1010fec001LCguNHMaaiq3Mg","id":"call_fake_0_0","input":{"camera":"front"},"executed":false},"durable":{"aggregateID":"ses_e2efef026ffeyA4sKEzv97gIRo","seq":12,"version":1}}
{"id":"evt_1d1011004001OvQWoFWU15krjL","created":1794507870212,"type":"session.tool.success","location":{"directory":"/tmp/actor"},"data":{"sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo","assistantMessageID":"msg_1d1010fec001LCguNHMaaiq3Mg","id":"call_fake_0_0","content":[{"type":"text","text":"observation ok MARKER-TOOL-TEXT"},{"type":"file","uri":"data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAAAwCAIAAAAuKetIAAAAY0lEQVR4nO3PQQkAMAzAwCqpfymTNRF7HINABFzm7H7dcEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUNaEEDWtCAFjSgBQ1oQQNa0IAWNKAFDWhBA1rQgBY0oAUN+O4BAc1sgRZy0QIAAAAASUVORK5CYII=","mime":"image/png"}],"metadata":{"truncated":false},"executed":false},"durable":{"aggregateID":"ses_e2efef026ffeyA4sKEzv97gIRo","seq":14,"version":2}}
{"id":"evt_1d1011006001MxClS5gogVblz3","created":1794507870214,"type":"session.step.ended","location":{"directory":"/tmp/actor"},"data":{"sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo","assistantMessageID":"msg_1d1010fec001LCguNHMaaiq3Mg","finish":"tool-calls","rawFinish":"tool_use","tokens":{"input":10,"output":5,"reasoning":0,"cache":{"read":0,"write":0}},"cost":0},"durable":{"aggregateID":"ses_e2efef026ffeyA4sKEzv97gIRo","seq":15,"version":1}}
{"id":"evt_1d1011012001ypdByZHg4L3CkW","created":1794507870226,"type":"session.execution.succeeded","data":{"sessionID":"ses_e2efef026ffeyA4sKEzv97gIRo"},"durable":{"aggregateID":"ses_e2efef026ffeyA4sKEzv97gIRo","seq":21,"version":1}}
{"id":"evt_1d129f188002YqV0LD0d8IJLBU","created":1794510549384,"type":"permission.asked","location":{"directory":"/tmp/actor"},"data":{"id":"per_1d129f188001sReRGevZ89ndNC","sessionID":"ses_e2ed60ff0ffeEWVD5HkvstRL75","action":"read","resources":[".env"],"save":["*"],"source":{"type":"tool","messageID":"msg_1d129f177001TNjWtVLP2zZ8MG","id":"call_fake_0_0"}}}
{"id":"evt_1d129f1c0001rcoMk6JqDFOD9T","created":1794510549440,"type":"permission.replied","location":{"directory":"/tmp/actor"},"data":{"sessionID":"ses_e2ed60ff0ffeEWVD5HkvstRL75","requestID":"per_1d129f188001sReRGevZ89ndNC","reply":"reject"}}
{"id":"evt_1d129f1c30028Qb5rf47OKH8rd","created":1794510549443,"type":"session.execution.interrupted","data":{"sessionID":"ses_e2ed60ff0ffeEWVD5HkvstRL75","reason":"shutdown"},"durable":{"aggregateID":"ses_e2ed60ff0ffeEWVD5HkvstRL75","seq":12,"version":1}}
""".strip().splitlines()]


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def sse(frame) -> bytes:
    """One SSE frame exactly as the 2.2.7 server writes it (compact JSON, ``data:`` line, blank line)."""
    return b"data: " + json.dumps(frame, separators=(",", ":")).encode() + b"\n\n"


class ClientBase(unittest.TestCase):
    unix = False

    def setUp(self):
        self.socket_path = None
        if self.unix:
            directory = tempfile.TemporaryDirectory(prefix="oc2b-", dir="/tmp")
            self.addCleanup(directory.cleanup)
            self.socket_path = os.path.join(directory.name, "s.sock")
        self.server = FakeOpenCodeServer(unix_socket=self.socket_path).start()
        self.addCleanup(self.server.close)
        self.client = self.make_client()
        self.addCleanup(self.client.close_streams)

    def make_client(self, password=PASSWORD, **keywords):
        keywords = {"timeout_s": 1.0, "stream_timeout_s": 3.0, **keywords}
        if self.unix:   # base_url names a port nothing listens on: every byte travels through the unix socket
            return oc.OpenCodeClient("http://127.0.0.1:1", password, unix_socket=self.socket_path, **keywords)
        return oc.OpenCodeClient(self.server.url, password, **keywords)

    def last(self, path_suffix):
        return [item for item in self.server.requests if item["path"].endswith(path_suffix)][-1]


# --------------------------------------------------------------------------------------------------- routes
class RouteTests(ClientBase):
    def test_health_is_api_info_none_while_starting_and_basic_auth(self):
        self.server.starting_polls = 2
        self.assertIsNone(self.client.health())
        self.assertIsNone(self.client.health(timeout=0.5))
        info = self.client.health()
        self.assertEqual(info["version"], PIN)
        self.assertEqual(set(info), {"version", "pid", "hostname", "urls", "paths", "capabilities"})
        request = self.last("/api/info")
        self.assertEqual(request["method"], "GET")
        self.assertEqual(request["headers"]["Authorization"],
                         "Basic " + base64.b64encode(f"opencode:{PASSWORD}".encode()).decode())
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            self.make_client("wrong").health()
        self.assertEqual((ctx.exception.status, ctx.exception.tag), (401, "UnauthorizedError"))
        self.assertTrue(ctx.exception.definite_rejection)

    def test_session_create_info_exists(self):
        sid = self.client.create_session("ni-rua-run-1", model=oc.model_ref("relay", "fake-model", "high"))
        self.assertTrue(sid.startswith("ses"))
        self.assertEqual(self.last("/api/session")["body"], {
            "title": "ni-rua-run-1", "model": {"id": "fake-model", "providerID": "relay", "variant": "high"}})
        info = self.client.session_info(sid)
        self.assertEqual((info["id"], info["title"]), (sid, "ni-rua-run-1"))
        self.assertEqual(info["model"], {"id": "fake-model", "providerID": "relay", "variant": "high"})
        plain = self.client.create_session("plain")
        self.assertEqual(self.last("/api/session")["body"], {"title": "plain"})
        self.assertNotIn("model", self.client.session_info(plain))
        self.assertTrue(self.client.session_exists(sid))
        self.assertFalse(self.client.session_exists("ses_missing"))
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            self.client.session_info("ses_missing")
        self.assertEqual((ctx.exception.status, ctx.exception.tag), (404, "SessionNotFoundError"))

    def test_prompt_receipt_steer_inbox_status_and_interrupt(self):
        sid = self.client.create_session("t")
        receipt = self.client.prompt(sid, "hello")
        self.assertEqual(self.last("/prompt")["body"], {"text": "hello"})
        self.assertEqual(set(receipt), {"id", "sessionID", "time", "type", "payload", "delivery"})
        self.assertEqual((receipt["type"], receipt["payload"], receipt["delivery"]),
                         ("user", {"text": "hello"}, "steer"))
        self.assertEqual(self.client.status(), {sid: {"type": "running"}})
        self.assertEqual(self.client.messages(sid), [{"id": receipt["id"], "time": mock.ANY, "text": "hello",
                                                      "type": "user"}])
        steer = self.client.prompt(sid, "steer now", delivery="steer")
        self.assertEqual(self.last("/prompt")["body"], {"text": "steer now", "delivery": "steer"})
        self.assertEqual([item["id"] for item in self.client.inbox(sid)], [steer["id"]])
        self.assertTrue(self.client.interrupt(sid))
        self.assertIsNone(self.last("/interrupt")["body"])
        self.assertEqual(self.client.status(), {})
        timeline = self.client.messages(sid)
        self.assertEqual([m["type"] for m in timeline], ["user", "assistant", "idle"])
        self.assertEqual(timeline[1]["error"], {"type": "aborted", "message": "Step interrupted"})
        self.assertEqual(timeline[2]["outcome"], "interrupted")
        self.assertFalse(self.client.interrupt(sid))        # idle no-op
        self.server.finish_turn(sid, "late")                 # parked steer is delivered by the next execution
        self.assertIn(steer["id"], [m["id"] for m in self.client.messages(sid)])

    def test_queue_delivery_waits_for_the_end_of_the_execution(self):
        sid = self.client.create_session("t")
        self.client.prompt(sid, "first")
        queued = self.client.prompt(sid, "later", delivery="queue")
        self.assertEqual(self.client.inbox(sid)[0]["delivery"], "queue")
        self.server.finish_turn(sid, "A1")
        self.assertEqual(self.client.inbox(sid), [])
        self.assertEqual(self.client.status(), {sid: {"type": "running"}})
        self.assertEqual(self.client.messages(sid)[-1]["id"], queued["id"])

    def test_prompt_forwards_inline_image_file_attachments(self):
        sid = self.client.create_session("t")
        uri = "data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
        receipt = self.client.prompt(sid, "Inspect overview", files=[{"uri": uri, "name": "overview.png"}])
        self.assertTrue(receipt["id"].startswith("msg_"))
        self.assertEqual(
            self.last("/prompt")["body"],
            {"text": "Inspect overview", "files": [{"uri": uri, "name": "overview.png"}]},
        )

    def test_switch_model_writes_model_switched(self):
        sid = self.client.create_session("t", model=oc.model_ref("relay", "fake-model", "low"))
        self.assertIsNone(self.client.switch_model(sid, oc.model_ref("relay", "other-model", "high")))
        self.assertEqual(self.last("/model")["body"],
                         {"model": {"id": "other-model", "providerID": "relay", "variant": "high"}})
        self.client.switch_model(sid, oc.model_ref("relay", "other-model"))
        switched = [m for m in self.client.messages(sid) if m["type"] == "model-switched"]
        self.assertEqual([m["model"] for m in switched], [{"id": "other-model", "providerID": "relay", "variant": "high"},
                                                          {"id": "other-model", "providerID": "relay"}])
        self.assertEqual(switched[0]["previous"], {"id": "fake-model", "providerID": "relay", "variant": "low"})
        self.assertEqual(self.client.session_info(sid)["model"]["variant"], "default")

    def test_routes_mcp_models_reload(self):
        self.assertEqual(self.client.routes(), sorted(self.server.routes))
        self.assertEqual(self.last("/openapi.json")["method"], "GET")
        self.server.mcp_pending_polls = 1
        self.assertEqual(self.client.mcp_status(), {"ni_rua": {"status": "pending"}})
        self.assertEqual(self.client.mcp_status(), {"ni_rua": {"status": "connected"}})
        self.server.mcp_servers = [{"name": "ni_rua", "status": {"status": "failed", "error": "boom"}}]
        self.assertEqual(self.client.mcp_status()["ni_rua"], {"status": "failed", "error": "boom"})
        models = self.client.models()
        self.assertEqual([(m["providerID"], m["id"]) for m in models], [("relay", "fake-model")])
        self.assertEqual(set(models[0]["limit"]), {"context", "output"})
        reloaded = []
        self.server.reload_hook = lambda: reloaded.append(True)
        self.assertIsNone(self.client.reload(timeout_s=2.0))
        self.assertEqual((reloaded, self.last("/api/location/reload")["method"]), ([True], "POST"))

    def test_permissions_and_reject_reply(self):
        sid = self.client.create_session("t")
        self.client.prompt(sid, "read env")
        tool = tool_item("read", {"path": ".env"}, call_id="call_1")
        message = self.server.assistant(sid, [tool])
        request_id = self.server.ask_permission(sid, tool_message_id=message["id"], tool_call_id="call_1")
        pending = self.client.permissions(sid)
        self.assertEqual([(item["id"], item["action"], item["resources"]) for item in pending],
                         [(request_id, "read", [".env"])])
        self.assertIsNone(self.client.permission_reply(sid, request_id, "reject"))
        self.assertEqual(self.last("/reply")["body"], {"decision": "reject"})
        self.assertEqual(self.client.permissions(sid), [])
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            self.client.permission_reply(sid, request_id, "reject")
        self.assertEqual((ctx.exception.status, ctx.exception.tag), (404, "PermissionNotFoundError"))
        # 2.2.7: a reject ends the execution without an `idle` message.
        self.assertEqual(self.client.status(), {})
        timeline = self.client.messages(sid)
        self.assertEqual([m["type"] for m in timeline], ["user", "assistant"])
        self.assertEqual(timeline[1]["content"][0]["state"]["error"],
                         {"type": "aborted", "message": "The user declined this tool call"})

    def test_argument_validation_sends_nothing(self):
        sid = self.client.create_session("t")
        before = len(self.server.requests)
        for call in (lambda: self.client.prompt(sid, ""), lambda: self.client.prompt(sid, "x", delivery="later"),
                     lambda: self.client.prompt(sid, "x", files="not-a-list"),
                     lambda: self.client.prompt(sid, "x", files=[{"name": "missing-uri"}]),
                     lambda: self.client.create_session("  "), lambda: self.client.messages("../x"),
                     lambda: self.client.messages(sid, page_size=0), lambda: self.client.messages(sid, page_size=201),
                     lambda: self.client.permission_reply(sid, "x", "reject"),
                     lambda: self.client.permission_reply(sid, "per_1", "deny"),
                     lambda: self.client.switch_model(sid, {"id": "m"}),
                     lambda: self.client.switch_model(sid, {"id": "m", "providerID": "relay", "extra": 1}),
                     lambda: self.client.create_session("t", model={"id": "", "providerID": "relay"}),
                     lambda: oc.model_ref("relay", "m", ""), lambda: self.client.events(types="a").__next__(),
                     lambda: self.client.interrupt("not-a-session")):
            with self.assertRaises(ValueError):
                call()
        self.assertEqual(len(self.server.requests), before)

    def test_prompt_files_validation_and_no_retry_on_transport_error(self):
        """校验 OpenCodeClient.prompt files 附件格式、非法 URI 零请求拒绝以及未知提交不重试不降级。"""
        sid = self.client.create_session("t")
        uri = "data:image/png;base64,iVBORw0KGgo="
        receipt = self.client.prompt(sid, "with image", files=[uri])
        self.assertTrue(receipt["id"].startswith("msg_"))
        self.assertEqual(self.last("/prompt")["body"], {"text": "with image", "files": [{"uri": uri}]})

        # 非法 URI 或含换行符：零 HTTP 请求直接抛出 ValueError
        before = len(self.server.requests)
        for bad in (
            ["https://evil.invalid/a.png"],
            ["data:image/png;base64,AAAA\nBBBB"],
            ["data:image/png,plain"],
            [123],
            "data:image/png;base64,AAAA",
        ):
            with self.assertRaises(ValueError):
                self.client.prompt(sid, "bad", files=bad)
        self.assertEqual(len(self.server.requests), before)

        # 5xx 属于未知提交，绝不自动降级或重发（严格只发 1 次 HTTP 请求）
        before_500 = len(self.server.requests)
        with mock.patch.object(self.client, "_request", side_effect=oc.OpenCodeHttpError(500, {"message": "boom"}, method="POST", path="/prompt")) as req:
            with self.assertRaises(oc.OpenCodeHttpError) as ctx:
                self.client.prompt(sid, "server-500", files=[uri])
            self.assertFalse(ctx.exception.definite_rejection)
            req.assert_called_once()
        self.assertEqual(len(self.server.requests), before_500)

    def test_base_url_validation(self):
        for url in ("https://x", "http://u:p@127.0.0.1:1", "http://127.0.0.1:1/p", "127.0.0.1"):
            with self.assertRaises(ValueError):
                oc.OpenCodeClient(url, "p")

    def test_malformed_answers_are_transport_errors(self):
        sid = self.client.create_session("t")
        cases = ((lambda: self.client.prompt(sid, "x"), {"data": {"type": "user"}}),
                 (self.client.status, {"data": []}),
                 (lambda: self.client.interrupt(sid), {"interrupted": "yes"}),
                 (self.client.mcp_status, {"data": [{"name": "x"}]}),
                 (lambda: self.client.inbox(sid), {"nodata": []}),
                 (lambda: self.client.create_session("t"), {"data": {"id": "nope"}}),
                 (self.client.health, ["not", "an", "object"]))
        for index, (call, answer) in enumerate(cases):
            with self.subTest(case=index), mock.patch.object(self.client, "_request", return_value=answer):
                with self.assertRaises(oc.OpenCodeTransportError):
                    call()

    def test_non_json_success_is_a_transport_error(self):
        # A 1.x route on a 2.2.7 server answers 200 with the web UI's HTML (probe b01: `GET /global/health`).
        class Html(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *args):
                pass

            def do_GET(self):
                body = b"<!doctype html><title>OpenCode</title>"
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
        httpd = ThreadingHTTPServer(("127.0.0.1", 0), Html)
        threading.Thread(target=httpd.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.addCleanup(httpd.server_close)
        self.addCleanup(httpd.shutdown)
        with self.assertRaises(oc.OpenCodeTransportError) as ctx:
            oc.OpenCodeClient(f"http://127.0.0.1:{httpd.server_address[1]}", "p", timeout_s=2).health()
        self.assertEqual(ctx.exception.reason, "response_not_json")


# ------------------------------------------------------------------------------------------------ pagination
class PaginationTests(ClientBase):
    def fill(self, count):
        sid = self.server.add_session("t")
        for index in range(count):
            self.server.append(sid, {"type": "user", "text": f"m{index}"})
        return sid

    def message_requests(self):
        return [parse_qs(item["query"]) for item in self.server.requests if item["path"].endswith("/message")]

    def test_follows_cursor_next_without_order_and_stops_at_a_short_page(self):
        sid = self.fill(7)
        self.assertEqual([m["text"] for m in self.client.messages(sid, page_size=2)], [f"m{i}" for i in range(7)])
        queries = self.message_requests()
        self.assertEqual(len(queries), 4)
        self.assertEqual(queries[0], {"limit": ["2"], "order": ["asc"]})
        for query in queries[1:]:
            self.assertEqual(set(query), {"limit", "cursor"})   # `order` with `cursor` is HTTP 400
        self.assertEqual(json.loads(base64.urlsafe_b64decode(queries[1]["cursor"][0] + "==")),
                         {"id": self.server.messages_of(sid)[1]["id"], "order": "asc", "direction": "next"})

    def test_exact_multiple_ends_with_an_empty_page_and_default_page_is_200(self):
        sid = self.fill(4)
        self.assertEqual(len(self.client.messages(sid, page_size=2)), 4)
        self.assertEqual(len(self.message_requests()), 3)
        self.client.messages(sid)
        self.assertEqual(self.message_requests()[-1], {"limit": ["200"], "order": ["asc"]})

    def test_timeline_order_is_kept_when_ids_are_not_sorted(self):
        # Real 2.2.7 capture (b04): a background subagent's report carries an id allocated before the parent's
        # `idle`, yet is listed after it; cursors are positional, so every page size returns the same timeline.
        sid = self.server.add_session("t")
        for message_id, kind in (("msg_0001", "user"), ("msg_0002", "assistant"), ("msg_0004", "idle"),
                                 ("msg_0003", "synthetic"), ("msg_0005", "assistant"), ("msg_0006", "idle")):
            self.server.append(sid, {"id": message_id, "type": kind})
        expected = ["msg_0001", "msg_0002", "msg_0004", "msg_0003", "msg_0005", "msg_0006"]
        for size in (1, 2, 3, 200):
            self.assertEqual([m["id"] for m in self.client.messages(sid, page_size=size)], expected)

    def test_fake_answers_cursor_errors_like_the_real_server(self):
        sid = self.fill(3)
        path = f"/api/session/{sid}/message?limit=1&order=asc"
        cursor = self.client._request("GET", path)["cursor"]["next"]
        for query, tag in ((f"limit=1&order=asc&cursor={cursor}", "InvalidCursorError"),
                           ("limit=1&cursor=garbage", "InvalidCursorError"), ("limit=201", "InvalidRequestError")):
            with self.assertRaises(oc.OpenCodeHttpError) as ctx:
                self.client._request("GET", f"/api/session/{sid}/message?{query}")
            self.assertEqual((ctx.exception.status, ctx.exception.tag), (400, tag))
        newest_first = self.client._request("GET", f"/api/session/{sid}/message")["data"]
        self.assertEqual([m["text"] for m in newest_first], ["m2", "m1", "m0"])     # default order: desc

    def test_repeated_cursor_is_a_transport_error(self):
        page = {"data": [{"type": "user"}], "cursor": {"previous": "p", "next": "same"}}
        with mock.patch.object(self.client, "_request", return_value=page):
            with self.assertRaises(oc.OpenCodeTransportError) as ctx:
                self.client.messages("ses_x", page_size=1)
        self.assertEqual(ctx.exception.reason, "messages_cursor_loop")
        with mock.patch.object(self.client, "_request", return_value={"data": {}, "cursor": {}}):
            with self.assertRaises(oc.OpenCodeTransportError):
                self.client.messages("ses_x")


# ------------------------------------------------------------------------------------------------- admission
class AdmissionTests(ClientBase):
    def test_4xx_is_definite_rejection_and_nothing_is_admitted(self):
        sid = self.client.create_session("t")
        for status, tag in ((400, "InvalidRequestError"), (404, "SessionNotFoundError"), (409, "ConflictError")):
            self.server.prompt_script.append(status)
            with self.assertRaises(oc.OpenCodeHttpError) as ctx:
                self.client.prompt(sid, "x")
            self.assertTrue(oc.is_definite_rejection(ctx.exception))
            self.assertEqual((ctx.exception.status, ctx.exception.tag), (status, tag))
        self.assertEqual((self.server.messages_of(sid), self.server.inbox_of(sid)), ([], []))

    def test_unknown_session_is_a_404_rejection(self):
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            self.client.prompt("ses_nope", "x")
        self.assertEqual((ctx.exception.status, ctx.exception.tag), (404, "SessionNotFoundError"))
        self.assertTrue(ctx.exception.definite_rejection)

    def test_5xx_is_not_definite(self):
        sid = self.client.create_session("t")
        for status in (500, 503):
            self.server.prompt_script.append(status)
            with self.assertRaises(oc.OpenCodeHttpError) as ctx:
                self.client.prompt(sid, "x")
            self.assertFalse(oc.is_definite_rejection(ctx.exception))

    def test_timeout_is_transport_unknown_and_not_retried(self):
        sid = self.client.create_session("t")
        self.server.hang_s = 2.0
        self.server.prompt_script.append("hang")
        with self.assertRaises(oc.OpenCodeTransportError) as ctx:
            self.client.prompt(sid, "x")
        self.assertIsInstance(ctx.exception, OSError)
        self.assertTrue(ctx.exception.admission_unknown)
        self.assertFalse(oc.is_definite_rejection(ctx.exception))
        self.assertEqual(self.server.request_paths("POST").count(f"/api/session/{sid}/prompt"), 1)

    def test_dropped_connection_after_admission_is_transport_unknown(self):
        sid = self.client.create_session("t")
        self.server.prompt_script.append("drop")
        with self.assertRaises(oc.OpenCodeTransportError):
            self.client.prompt(sid, "x")
        self.assertEqual([m["type"] for m in self.server.messages_of(sid)], ["user"])   # admitted, unacknowledged

    def test_receipt_without_message_id_is_unknown_admission(self):
        with mock.patch.object(self.client, "_request", return_value={"data": {"id": "nope"}}):
            with self.assertRaises(oc.OpenCodeTransportError) as ctx:
                self.client.prompt("ses_x", "x")
        self.assertEqual(ctx.exception.reason, "prompt_receipt_invalid")

    def test_interrupt_failures(self):
        sid = self.client.create_session("t")
        self.client.prompt(sid, "x")
        self.server.script("session.interrupt", 500, "drop")
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            self.client.interrupt(sid)
        self.assertFalse(ctx.exception.definite_rejection)
        self.assertEqual(self.client.status(), {sid: {"type": "running"}})
        with self.assertRaises(oc.OpenCodeTransportError):
            self.client.interrupt(sid, timeout_s=1.0)
        self.assertEqual(self.client.status(), {})            # it took effect, the answer was lost

    def test_connection_refused(self):
        dead = oc.OpenCodeClient("http://127.0.0.1:1", "p", timeout_s=0.5)
        with self.assertRaises(oc.OpenCodeTransportError):
            dead.health()

# ---------------------------------------------------------------------------------------------------- events
class EventTests(ClientBase):
    def collect(self, count, client=None, **keywords):
        frames, errors = [], []
        client = client or self.client

        def run():
            try:
                for frame in client.events(**keywords):
                    frames.append(frame)
                    if len(frames) >= count:
                        return
            except Exception as error:   # noqa: BLE001 - recorded for the assertion
                errors.append(error)
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return frames, errors, thread

    def test_captured_real_frames_roundtrip(self):
        frames, errors, thread = self.collect(len(CAPTURED_FRAMES))
        self.assertTrue(self.server.wait_subscribers())
        for frame in CAPTURED_FRAMES[1:]:
            self.server.send_raw_event(sse(frame))       # one chunk per frame, as the real server writes them
        thread.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(frames[0]["type"], "server.connected")
        self.assertEqual(frames[1:], CAPTURED_FRAMES[1:])

    def test_heartbeat_comments_unknown_fields_and_invalid_frames_skipped(self):
        frames, _errors, thread = self.collect(2)
        self.assertTrue(self.server.wait_subscribers())
        self.server.comment()                                  # ": heartbeat"
        self.server.send_raw_event(b"data: {not json\n\n")
        self.server.send_raw_event(b"data: [1, 2]\n\n")
        self.server.send_raw_event(b"event: x\nid: 7\nretry: 10\n\n")
        self.server.push("session.status", {"sessionID": "ses_a"})
        thread.join(5)
        self.assertEqual([f["type"] for f in frames], ["server.connected", "session.status"])
        self.assertEqual(self.client.invalid_event_frames, 2)

    def test_frames_split_across_chunks_and_several_frames_in_one_chunk(self):
        first, second, third = (sse(frame) for frame in CAPTURED_FRAMES[1:4])
        frames, errors, thread = self.collect(4)
        self.assertTrue(self.server.wait_subscribers())
        self.server.send_raw_event(first, chunks=[1, 3, 6, 40, len(first) - 1])    # inside "data:", JSON, "\n\n"
        self.server.send_raw_event(second + b": heartbeat\n\n" + third)
        thread.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(frames[1:], CAPTURED_FRAMES[1:4])

    def test_multiline_data_and_crlf_lines(self):
        frames, _errors, thread = self.collect(2)
        self.assertTrue(self.server.wait_subscribers())
        self.server.send_raw_event(b'data: {"type": "a",\r\ndata:"data": {}}\r\n\r\n')
        thread.join(5)
        self.assertEqual(frames[1], {"type": "a", "data": {}})

    def test_large_frame_spanning_many_reads(self):
        image = "data:image/png;base64," + base64.b64encode(os.urandom(3 << 20)).decode()
        big = {"id": "evt_big", "type": "session.tool.success", "data": {"content": [{"type": "file", "uri": image}]}}
        frames, errors, thread = self.collect(2)
        self.assertTrue(self.server.wait_subscribers())
        self.server.send_raw_event(sse(big))
        thread.join(10)
        self.assertEqual(errors, [])
        self.assertEqual(frames[1], big)

    def test_oversized_frames_are_skipped_without_breaking_the_stream(self):
        with mock.patch.object(oc, "_MAX_EVENT_BYTES", 1000), mock.patch.object(oc, "_READ_SIZE", 256):
            frames, _errors, thread = self.collect(2)
            self.assertTrue(self.server.wait_subscribers())
            self.server.send_raw_event(b"data: " + b"x" * 5000 + b"\n\n")             # one line > cap
            self.server.send_raw_event(b"data: " + b"y" * 600 + b"\ndata: " + b"z" * 600 + b"\n\n")  # lines sum > cap
            self.server.push("session.status", {"sessionID": "ses_after"})
            thread.join(5)
        self.assertEqual(frames[1]["data"], {"sessionID": "ses_after"})
        self.assertEqual(self.client.invalid_event_frames, 2)

    def test_types_filter_is_a_repeated_query_parameter(self):
        frames, _errors, thread = self.collect(2, types=["session.execution.started", "permission.asked"])
        self.assertTrue(self.server.wait_subscribers())
        self.server.push("session.status", {"sessionID": "ses_a"})
        self.server.push("permission.asked", {"id": "per_1"})
        thread.join(5)
        self.assertEqual([f["type"] for f in frames], ["server.connected", "permission.asked"])
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            next(self.client.events(types=["no.such.event"]))
        self.assertEqual((ctx.exception.status, ctx.exception.tag), (400, "InvalidRequestError"))

    def test_dropped_and_cleanly_ended_streams_raise_stream_ended(self):
        for end in (self.server.drop_event_streams, self.server.end_event_streams):
            _frames, errors, thread = self.collect(100)
            self.assertTrue(self.server.wait_subscribers())
            end()
            thread.join(5)
            self.assertEqual(len(errors), 1)
            self.assertIsInstance(errors[0], oc.OpenCodeStreamEnded)

    def test_silence_beyond_stream_timeout_is_a_transport_error(self):
        quiet = self.make_client(stream_timeout_s=0.4)
        _frames, errors, thread = self.collect(100, client=quiet)
        thread.join(5)
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], oc.OpenCodeTransportError)
        self.assertNotIsInstance(errors[0], oc.OpenCodeStreamEnded)

    def test_heartbeats_keep_a_quiet_stream_alive(self):
        self.server.heartbeat_s = 0.1
        quiet = self.make_client(stream_timeout_s=0.5)
        frames, errors, thread = self.collect(2, client=quiet)
        self.assertTrue(self.server.wait_subscribers())
        time.sleep(1.2)
        self.server.push("session.status", {"sessionID": "ses_late"})
        thread.join(5)
        self.assertEqual(errors, [])
        self.assertEqual(frames[1]["data"], {"sessionID": "ses_late"})

    def test_event_stream_reconnects_reports_gaps_and_loses_events(self):
        stop = threading.Event()
        frames, gaps = [], []
        thread = threading.Thread(target=lambda: [frames.append(f) for f in self.client.event_stream(
            stop, on_gap=gaps.append, initial_backoff_s=0.05, max_backoff_s=0.1)], daemon=True)
        thread.start()
        self.assertTrue(self.server.wait_subscribers())
        self.server.push("session.status", {"sessionID": "ses_a"})
        self.assertTrue(wait_until(lambda: any(f["data"].get("sessionID") == "ses_a" for f in frames)))
        self.server.drop_event_streams()
        self.server.push("session.status", {"sessionID": "ses_lost"})   # emitted while disconnected: gone
        self.assertTrue(wait_until(lambda: gaps and self.server.wait_subscribers(1, 0.05)))
        self.server.push("session.status", {"sessionID": "ses_b"})
        self.assertTrue(wait_until(lambda: any(f["data"].get("sessionID") == "ses_b" for f in frames)))
        stop.set()
        self.client.close_streams()
        thread.join(5)
        ids = [f["data"].get("sessionID") for f in frames]
        self.assertNotIn("ses_lost", ids)
        self.assertEqual([f["type"] for f in frames].count("server.connected"), 2)
        self.assertIsInstance(gaps[0], oc.OpenCodeStreamEnded)

    def test_event_stream_http_errors_are_gaps(self):
        self.server.script("event.subscribe", 503)
        for client, expected in ((self.client, (503, "ServiceUnavailableError")),
                                 (self.make_client("wrong"), (401, "UnauthorizedError"))):
            stop, gaps = threading.Event(), []

            def on_gap(error):
                gaps.append(error)
                stop.set()
            self.assertEqual(list(client.event_stream(stop, on_gap=on_gap, initial_backoff_s=0.01)), [])
            self.assertEqual([(gap.status, gap.tag) for gap in gaps], [expected])

    def test_close_streams_unblocks_reader(self):
        _frames, errors, thread = self.collect(100)
        self.assertTrue(self.server.wait_subscribers())
        self.client.close_streams()
        thread.join(5)
        self.assertFalse(thread.is_alive())
        self.assertEqual(len(errors), 1)
        self.assertIsInstance(errors[0], oc.OpenCodeTransportError)


# ----------------------------------------------------------------------------------------------- fake fidelity
# Session event types of one real 2.2.7 turn (probe s01: reasoning + text + one tool call, then the final text),
# without `session.instructions.updated` (instruction digests; the fake does not emit it).
REAL_TURN_EVENTS = [
    "session.created", "session.inbox.enqueued", "session.execution.started", "session.inbox.delivered",
    "session.step.started", "session.reasoning.started", "session.reasoning.delta", "session.reasoning.ended",
    "session.text.started", "session.text.delta", "session.text.ended", "session.tool.input.started",
    "session.tool.input.ended", "session.tool.called", "session.step.streamed", "session.tool.success",
    "session.step.ended", "session.usage.updated", "session.step.started", "session.text.started",
    "session.text.delta", "session.text.ended", "session.step.streamed", "session.step.ended",
    "session.usage.updated", "session.execution.succeeded"]


class FakeFidelityTests(ClientBase):
    def test_a_driven_turn_emits_the_real_event_sequence_and_timeline(self):
        stop, frames = threading.Event(), []
        reader = threading.Thread(target=lambda: [frames.append(f) for f in self.client.event_stream(stop)],
                                  daemon=True)
        reader.start()
        self.addCleanup(lambda: (stop.set(), self.client.close_streams(), reader.join(5)))
        self.assertTrue(self.server.wait_subscribers())
        sid = self.client.create_session("ni-rua-actor")
        receipt = self.client.prompt(sid, "Take one observation then answer.")
        step = self.server.assistant(sid, [reasoning_item("think-1"), text_item("Looking."),
                                           tool_item("ni_rua_observe", {"camera": "front"}, call_id="call_fake_0_0")])
        self.server.finish_tool(sid, step, "call_fake_0_0", content=[{"type": "text", "text": "observation ok"}])
        self.server.complete(sid, step, finish="tool-calls")
        self.server.finish_turn(sid, "FINAL-ANSWER-1")
        self.assertTrue(wait_until(lambda: any(f["type"] == "session.execution.succeeded" for f in frames)))
        self.assertEqual([f["type"] for f in frames if f["type"].startswith("session.")], REAL_TURN_EVENTS)
        for frame in frames[1:]:
            self.assertEqual(frame["data"]["sessionID"], sid)
            self.assertTrue(frame["id"].startswith("evt_"))
        timeline = self.client.messages(sid)
        self.assertEqual([m["type"] for m in timeline], ["user", "assistant", "assistant", "idle"])
        self.assertEqual(timeline[0]["id"], receipt["id"])
        self.assertEqual((timeline[1]["finish"], timeline[1]["rawFinish"]), ("tool-calls", "tool_use"))
        self.assertEqual([item["type"] for item in timeline[1]["content"]], ["reasoning", "text", "tool"])
        self.assertEqual(timeline[1]["content"][2]["state"]["status"], "completed")
        self.assertEqual((timeline[2]["content"], timeline[2]["finish"]), ([text_item("FINAL-ANSWER-1")], "stop"))
        succeeded = next(f for f in frames if f["type"] == "session.execution.succeeded")
        self.assertEqual(succeeded["id"][4:], timeline[3]["id"][4:])     # evt_X ends the execution, msg_X is idle

    def test_background_report_starts_an_idle_parent(self):
        parent = self.client.create_session("ni-rua-actor")
        self.client.prompt(parent, "spawn")
        child = self.server.add_child(parent, title="bg work")
        self.assertEqual(self.client.session_info(child)["parentID"], parent)
        self.server.finish_turn(parent, "MAIN-FINAL-BEFORE-CHILD-DONE")
        self.assertEqual(self.client.status(), {})
        self.server.report_to_parent(child, "CHILD-RESULT", description="bg work")
        self.assertEqual(self.client.status(), {parent: {"type": "running"}})
        report = self.client.messages(parent)[-1]
        self.assertEqual((report["type"], report["description"], report["metadata"]["childID"]),
                         ("synthetic", "bg work", child))
        self.assertIn("CHILD-RESULT", report["text"])

    def test_compaction_message_shares_the_started_event_id(self):
        sid = self.client.create_session("t")
        stop, frames = threading.Event(), []
        reader = threading.Thread(target=lambda: [frames.append(f) for f in self.client.event_stream(stop)],
                                  daemon=True)
        reader.start()
        self.addCleanup(lambda: (stop.set(), self.client.close_streams(), reader.join(5)))
        self.assertTrue(self.server.wait_subscribers())
        message = self.server.compact(sid)
        self.assertTrue(wait_until(lambda: any(f["type"] == "session.compaction.ended" for f in frames)))
        started = next(f for f in frames if f["type"] == "session.compaction.started")
        self.assertEqual(started["id"][4:], message["id"][4:])
        self.assertEqual(self.client.messages(sid)[-1]["type"], "compaction")


# ------------------------------------------------------------------------------------------------ unix socket
class UnixSocketRouteTests(RouteTests):
    """Every route test again, with the client reaching the fake only through its unix socket listener (what the
    sandbox's inbound socket gives the host)."""
    unix = True


class UnixSocketEventTests(EventTests):
    """Every SSE test again over the unix socket: chunked framing, heartbeats, reconnects."""
    unix = True

    def test_tcp_is_the_default_transport(self):
        self.assertIsNone(oc.OpenCodeClient(self.server.url, PASSWORD).unix_socket)
        self.assertEqual(oc.OpenCodeClient(self.server.url, PASSWORD).health()["version"], PIN)

    def test_missing_socket_is_a_transport_error(self):
        client = oc.OpenCodeClient("http://127.0.0.1:1", PASSWORD, timeout_s=1, unix_socket=self.socket_path + ".x")
        with self.assertRaises(oc.OpenCodeTransportError):
            client.health()


class RouteListTests(unittest.TestCase):
    def test_routes_from_openapi(self):
        doc = {"paths": {"/a": {"get": {}, "post": {}, "parameters": []}, "/b": {"delete": {}}}}
        self.assertEqual(oc.routes_from_openapi(doc), ["DELETE /b", "GET /a", "POST /a"])
        with self.assertRaises(ValueError):
            oc.routes_from_openapi({})

    def test_model_ref(self):
        self.assertEqual(oc.model_ref("relay", "m"), {"id": "m", "providerID": "relay"})
        self.assertEqual(oc.model_ref("relay", "m", "high"), {"id": "m", "providerID": "relay", "variant": "high"})


# ----------------------------------------------------------------------------------------------- real binary
class _UnixBridge:
    """A unix-socket listener that forwards every connection to a TCP port (the sandbox's inbound forwarder)."""

    def __init__(self, path, port):
        self.path, self.port = path, port
        self.listener = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.listener.bind(path)
        self.listener.listen(8)
        threading.Thread(target=self._accept, daemon=True).start()

    def close(self):
        self.listener.close()

    def _accept(self):
        while True:
            try:
                downstream, _ = self.listener.accept()
            except OSError:
                return
            upstream = socket.create_connection(("127.0.0.1", self.port))
            for source, target in ((downstream, upstream), (upstream, downstream)):
                threading.Thread(target=self._pump, args=(source, target), daemon=True).start()

    @staticmethod
    def _pump(source, target):
        try:
            while True:
                data = source.recv(65536)
                if not data:
                    break
                target.sendall(data)
        except OSError:
            pass
        finally:
            for sock in (source, target):
                try:
                    sock.shutdown(socket.SHUT_RDWR)
                except OSError:
                    pass


@requires_opencode
class RealServerTest(unittest.TestCase):
    """The same client calls against the pinned 2.2.7 binary (``opencode serve`` in an isolated temporary home, a
    fake Anthropic-compatible model, no MCP server, no network).  Checks the shapes the fake reproduces: 503-or-200
    readiness, receipt ids, cursor paging, the inbox, interrupt no-op, model switch, the permission reject ending
    the execution without ``idle``, typed 4xx errors, and SSE frames over TCP and through a unix-socket forwarder."""

    def setUp(self):
        from support.fake_model_servers import FakeAnthropicServer, text_step, tool_step
        root = Path(tempfile.mkdtemp(prefix="oc2b-real-", dir="/tmp"))
        self.addCleanup(shutil.rmtree, root, True)
        for sub in ("home", "xdg/config", "xdg/data", "xdg/cache", "xdg/state", "tmp", "cfgdir", "actor"):
            (root / sub).mkdir(parents=True)
        (root / "actor" / ".env").write_text("SECRET=1\n")
        self.model = FakeAnthropicServer([tool_step("read", {"path": ".env"}), text_step("SECOND-ANSWER")]).start()
        self.addCleanup(self.model.close)
        config = {"update": "disable", "share": "disabled", "snapshots": False, "model": "relay/probe-model",
                  "permissions": [{"action": "webfetch", "resource": "*", "effect": "deny"}],
                  "policies": [{"action": "provider.use", "resource": "*", "effect": "deny"},
                               {"action": "provider.use", "resource": "relay", "effect": "allow"}],
                  "providers": {"relay": {"name": "relay", "package": "@opencode/ai/providers/anthropic-compatible",
                                          "settings": {"baseURL": self.model.base_url + "/v1", "apiKey": "x",
                                                       "headerTimeout": False, "chunkTimeout": False},
                                          "models": {"probe-model": {"name": "probe-model", "capabilities": {
                                              "tools": True, "input": ["text"], "output": ["text"]},
                                              "limit": {"context": 200000, "output": 32000}}}}}}
        probe = socket.socket()
        probe.bind(("127.0.0.1", 0))
        self.port = probe.getsockname()[1]
        probe.close()
        env = {"PATH": "/usr/bin:/bin", "HOME": str(root / "home"), "TMPDIR": str(root / "tmp"),
               "XDG_CONFIG_HOME": str(root / "xdg/config"), "XDG_DATA_HOME": str(root / "xdg/data"),
               "XDG_CACHE_HOME": str(root / "xdg/cache"), "XDG_STATE_HOME": str(root / "xdg/state"),
               "OPENCODE_CONFIG_CONTENT": json.dumps(config), "OPENCODE_CONFIG_DIR": str(root / "cfgdir"),
               "OPENCODE_PASSWORD": "real-pass", "OPENCODE_DISABLE_PROJECT_CONFIG": "1",
               "OPENCODE_DISABLE_CLAUDE_CODE": "1", "OPENCODE_DISABLE_AUTOUPDATE": "1",
               "OPENCODE_DISABLE_MODELS_FETCH": "1"}
        self.process = subprocess.Popen([os.environ[OPENCODE_EXECUTABLE_ENV], "serve", "--hostname", "127.0.0.1",
                                         "--port", str(self.port)], env=env, cwd=root / "actor",
                                        stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL,
                                        stderr=subprocess.DEVNULL, start_new_session=True)
        self.addCleanup(self._stop)
        self.client = oc.OpenCodeClient(f"http://127.0.0.1:{self.port}", "real-pass", timeout_s=10)
        self.bridge = _UnixBridge(str(root / "in.sock"), self.port)
        self.addCleanup(self.bridge.close)

    def _stop(self):
        if self.process.poll() is None:
            os.killpg(self.process.pid, 15)
            try:
                self.process.wait(10)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, 9)

    def wait_inactive(self, sid, timeout=20):
        self.assertTrue(wait_until(lambda: sid not in self.client.status(), timeout))

    def test_client_against_the_real_server(self):
        deadline = time.monotonic() + 30
        info = None
        while info is None and time.monotonic() < deadline:
            try:
                info = self.client.health(timeout=1.0)
            except oc.OpenCodeTransportError:
                time.sleep(0.05)
        self.assertEqual(info["version"], PIN)
        routes = self.client.routes()
        self.assertEqual(len(routes), 170)
        self.assertIn("POST /api/session/{sessionID}/prompt", routes)
        self.assertEqual(self.client.mcp_status(), {})
        self.assertEqual([(m["providerID"], m["id"]) for m in self.client.models()], [("relay", "probe-model")])
        events = oc.OpenCodeClient(f"http://127.0.0.1:{self.port}", "real-pass", unix_socket=self.bridge.path)
        stop, frames = threading.Event(), []
        reader = threading.Thread(target=lambda: [frames.append(f) for f in events.event_stream(stop)], daemon=True)
        reader.start()
        self.addCleanup(lambda: (stop.set(), events.close_streams(), reader.join(5)))
        self.assertTrue(wait_until(lambda: frames, 10))
        sid = self.client.create_session("ni-rua-client-test", model=oc.model_ref("relay", "probe-model"))
        self.assertEqual(self.client.session_info(sid)["model"],
                         {"id": "probe-model", "providerID": "relay", "variant": "default"})

        first = self.client.prompt(sid, "read the env file")
        self.assertTrue(wait_until(lambda: self.client.permissions(sid), 20))
        request = self.client.permissions(sid)[0]
        self.assertEqual((request["action"], request["resources"]), ("read", [".env"]))
        self.client.permission_reply(sid, request["id"], "reject")
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            self.client.permission_reply(sid, request["id"], "reject")
        self.assertEqual((ctx.exception.status, ctx.exception.tag), (404, "PermissionNotFoundError"))
        self.wait_inactive(sid)
        timeline = self.client.messages(sid)
        self.assertEqual([m["type"] for m in timeline], ["user", "assistant"])          # no `idle` after a reject
        self.assertEqual(timeline[0]["id"], first["id"])
        self.assertEqual(timeline[1]["content"][0]["state"]["error"]["message"], "The user declined this tool call")

        second = self.client.prompt(sid, "answer now")
        self.assertTrue(wait_until(lambda: self.client.messages(sid)[-1]["type"] == "idle", 20))
        timeline = self.client.messages(sid)
        self.assertEqual([m["type"] for m in timeline[2:]], ["user", "assistant", "idle"])
        self.assertEqual(timeline[2]["id"], second["id"])
        self.assertEqual(timeline[3]["content"], [text_item("SECOND-ANSWER")])
        self.assertEqual(timeline[4]["outcome"], "succeeded")
        for size in (1, 2, 3):
            self.assertEqual(self.client.messages(sid, page_size=size), timeline)
        self.assertEqual(self.client.inbox(sid), [])
        self.assertFalse(self.client.interrupt(sid))
        self.client.switch_model(sid, oc.model_ref("relay", "probe-model", "high"))
        self.assertEqual(self.client.messages(sid)[-1]["type"], "model-switched")
        self.assertEqual(self.client.session_info(sid)["model"]["variant"], "high")
        for call in (lambda: self.client.session_info("ses_doesnotexist00"),
                     lambda: self.client.prompt("ses_doesnotexist00", "x")):
            with self.assertRaises(oc.OpenCodeHttpError) as ctx:
                call()
            self.assertEqual((ctx.exception.status, ctx.exception.tag), (404, "SessionNotFoundError"))
        with self.assertRaises(oc.OpenCodeHttpError) as ctx:
            oc.OpenCodeClient(f"http://127.0.0.1:{self.port}", "wrong").health()
        self.assertEqual((ctx.exception.status, ctx.exception.tag), (401, "UnauthorizedError"))

        self.assertTrue(wait_until(lambda: any(f["type"] == "session.model.selected" for f in frames), 5))
        kinds = [f["type"] for f in frames]
        self.assertEqual(kinds[0], "server.connected")
        for kind in ("session.created", "session.inbox.enqueued", "permission.asked", "permission.replied",
                     "session.execution.interrupted", "session.execution.succeeded"):
            self.assertIn(kind, kinds)
        interrupted = next(f for f in frames if f["type"] == "session.execution.interrupted")
        self.assertEqual(interrupted["data"], {"sessionID": sid, "reason": "shutdown"})
        self.assertEqual(events.invalid_event_frames, 0)

if __name__ == "__main__":
    unittest.main()
