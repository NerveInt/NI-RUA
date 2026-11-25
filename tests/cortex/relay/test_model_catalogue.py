"""Merged cortex tests:
- tests/test_model_catalogue.py: Provider model catalogue (``ni_rua.cortex.agents.model_catalogue``) against a loopback fake listing server, and
- tests/test_model_catalogue_reasoning.py: Model catalogue reasoning claims, typed failures, native Codex metadata and the per-(profile, wire) cache
"""

NI_RUA_TEST_TIER = "component"

import json
import tempfile
import threading
import time
import unittest
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlsplit
from support.fake_claude_sdk import requires_sdk
from ni_rua.cortex.agents import model_catalogue as mc
from urllib.parse import urlsplit

# ==== tests/test_model_catalogue.py ====

KEY = "sk-test-catalogue-key"


def gateway_item(identifier, *, protocols=("COMPLETIONS", "MESSAGES", "RESPONSES"), tools=True, image=True,
                 context=256000, output=131072, name=None):
    return {"id": identifier, "object": "model", "name": name or identifier.upper(), "protocols": list(protocols),
            "effective_price": {"input_price": "1.0"},
            "capabilities": {"tools": tools, "image_input": image, "context_window": context,
                             "max_output_tokens": output, "reasoning": True}}


class Listing:
    """Loopback listing server. ``pages``: ``{query string: (status, body)}``; ``""`` is the first page.
    ``accept``: credential header values that authorise (None = any). ``delay_s``: answer delay."""

    def __init__(self, pages, *, accept=None, delay_s=0.0, path="/v1/models"):
        from support.fake_model_servers import is_host_port_probe

        self.pages, self.accept, self.delay_s, self.path, self.requests = pages, accept, delay_s, path, []
        self.port_probes = []
        listing = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                parts = urlsplit(self.path)
                # 过滤共享宿主机对新开回环端口的 GET / 探测，避免污染请求计数
                if listing.path != "/" and is_host_port_probe(self.command, self.path, b"", self.headers):
                    listing.port_probes.append(dict(path=parts.path, query=parse_qs(parts.query),
                                                    headers={k.lower(): v for k, v in self.headers.items()}))
                    data = json.dumps({"error": "not found"}).encode()
                    self.send_response(404)
                    self.send_header("Content-Type", "application/json")
                    self.send_header("Content-Length", str(len(data)))
                    self.end_headers()
                    try:
                        self.wfile.write(data)
                    except (BrokenPipeError, ConnectionResetError):
                        pass
                    return
                listing.requests.append(dict(path=parts.path, query=parse_qs(parts.query),
                                             headers={k.lower(): v for k, v in self.headers.items()}))
                time.sleep(listing.delay_s)
                credential = self.headers.get("Authorization") or self.headers.get("x-api-key")
                if parts.path != listing.path:
                    status, body = 404, {"error": "not found"}
                elif listing.accept is not None and credential not in listing.accept:
                    status, body = 401, {"error": {"code": "AUTHENTICATION_REQUIRED"}}
                else:
                    status, body = listing.pages.get(parts.query, (404, {"error": "no page"}))
                data = body if isinstance(body, bytes) else json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class CatalogueCase(unittest.TestCase):
    def listing(self, pages, **options):
        server = Listing(pages, **options)
        self.addCleanup(server.close)
        return server


class ParseAndFilterTests(unittest.TestCase):
    def test_gateway_shape_and_usable_filter(self):
        document = {"object": "list", "next_cursor": None, "data": [
            gateway_item("kimi-k2.6", name="Kimi K2.6"), gateway_item("glm-5.3", image=False),
            gateway_item("neohorse", tools=False), gateway_item("wan-video", protocols=("NATIVE",), tools=False),
            gateway_item("chat-only", protocols=("COMPLETIONS",)), {"id": "bare"}, {"name": "no id"}, "junk"]}
        models, more = mc.parse_models_page(document)
        self.assertIsNone(more)
        self.assertEqual([model.id for model in models], ["kimi-k2.6", "glm-5.3", "neohorse", "wan-video",
                                                          "chat-only", "bare"])
        kimi = models[0]
        self.assertEqual((kimi.name, kimi.protocols, kimi.tools, kimi.image_input, kimi.context_window,
                          kimi.max_output_tokens),
                         ("Kimi K2.6", ("COMPLETIONS", "MESSAGES", "RESPONSES"), True, True, 256000, 131072))
        self.assertEqual(models[5], mc.CatalogueModel(id="bare"))
        # protocols must include the wire's protocol when listed; tools false is never usable; unlisted = usable.
        self.assertEqual([model.id for model in mc.usable_models(models, "anthropic_messages")],
                         ["kimi-k2.6", "glm-5.3", "bare"])
        self.assertEqual([model.id for model in mc.usable_models(models, "openai_responses")],
                         ["kimi-k2.6", "glm-5.3", "bare"])
        self.assertEqual([model.id for model in mc.usable_models(models + models, "openai_chat")],
                         ["kimi-k2.6", "glm-5.3", "chat-only", "bare"])

    def test_anthropic_shape_and_cursor_queries(self):
        models, more = mc.parse_models_page({"data": [{"id": "claude-x", "display_name": "Claude X", "type": "model"}],
                                             "has_more": True, "last_id": "claude-x"})
        self.assertEqual((models[0].name, models[0].protocols, more), ("Claude X", None, {"after_id": "claude-x"}))
        self.assertEqual(mc.parse_models_page({"data": [], "next_cursor": "c2"})[1], {"cursor": "c2"})
        self.assertIsNone(mc.parse_models_page({"data": [], "has_more": False, "last_id": "x"})[1])
        for bad in ({}, {"data": {}}, []):
            with self.assertRaisesRegex(ValueError, "invalid_shape"):
                mc.parse_models_page(bad)
        # bogus capability values are "not reported", never coerced
        model, = mc.parse_models_page({"data": [{"id": "m", "capabilities": {
            "tools": "yes", "image_input": 0, "context_window": True, "max_output_tokens": -5}}]})[0]
        self.assertEqual((model.tools, model.image_input, model.context_window, model.max_output_tokens),
                         (None, None, None, None))

    def test_models_url_follows_the_relay_base_convention(self):
        self.assertEqual(mc.models_url("https://gw.example/gateway/", "anthropic_messages"),
                         "https://gw.example/gateway/v1/models")
        self.assertEqual(mc.models_url("https://gw.example/gateway/v1", "openai_responses"),
                         "https://gw.example/gateway/v1/models")


class ModelsDevShapeTests(unittest.TestCase):
    """The models.dev-style listing (OpenCode Console): nested tool flag, image media types, limits under ``limit``."""

    def test_nested_capabilities_modalities_and_limit(self):
        from ni_rua.cortex.agents import model_catalogue as mc
        models, _ = mc.parse_models_page({"object": "list", "data": [
            {"id": "kimi-k3", "name": "Kimi K3",
             "modalities": {"input": ["text", "image/png", "video/mp4"], "output": ["text"]},
             "capabilities": {"tools": {"supported": True}, "reasoning": {"supported": True}},
             "limit": {"context": 1048576, "output": 131072}},
            {"id": "big-pickle", "modalities": {"input": ["text"], "output": ["text"]},
             "capabilities": {"tools": {"supported": False}}, "limit": {"context": 200000}}]})
        kimi, pickle = models
        self.assertEqual((kimi.tools, kimi.image_input, kimi.context_window, kimi.max_output_tokens),
                         (True, True, 1048576, 131072))
        self.assertEqual((pickle.tools, pickle.image_input, pickle.context_window, pickle.max_output_tokens),
                         (False, False, 200000, None))
        self.assertEqual([m.id for m in mc.usable_models(models, "anthropic_messages")], ["kimi-k3"])

    def test_request_names_a_client_agent_unless_given(self):
        from ni_rua.cortex.agents import model_catalogue as mc
        seen = []

        class Response:
            def __init__(self): self.body = b'{"data": []}'
            def read(self, n): return self.body
            def __enter__(self): return self
            def __exit__(self, *a): return False

        def opener(request, timeout):
            seen.append(request.get_header("User-agent"))
            return Response()
        mc._get_json("https://example.invalid/v1/models", {"Authorization": "Bearer x"}, 1.0, opener)
        mc._get_json("https://example.invalid/v1/models", {"User-Agent": "custom/1"}, 1.0, opener)
        self.assertEqual(seen, [mc.USER_AGENT, "custom/1"])


class FetchTests(CatalogueCase):
    def test_single_page_with_bearer_records_limits_and_image_flag_without_key(self):
        server = self.listing({"": (200, {"object": "list", "next_cursor": None, "data": [
            gateway_item("kimi-k2.5", output=16384), gateway_item("glm-5.3", image=False),
            gateway_item("wan", protocols=("NATIVE",), tools=False)]})}, accept={"Bearer " + KEY})
        catalogue = mc.fetch_upstream_catalogue(server.base, {"Authorization": "Bearer " + KEY},
                                                wire_api="anthropic_messages")
        self.assertEqual((catalogue.status, catalogue.reason, catalogue.pages, catalogue.auth),
                         ("ok", None, 1, "configured"))
        request, = server.requests
        self.assertEqual((request["path"], request["headers"].get("anthropic-version")), ("/v1/models", "2023-06-01"))
        record = catalogue.record("anthropic_messages")
        self.assertEqual((record["listed"], record["usable"]), (3, 2))
        self.assertEqual([(item["id"], item["max_output_tokens"], item["image_input"]) for item in record["models"]],
                         [("kimi-k2.5", 16384, True), ("glm-5.3", 131072, False)])
        self.assertNotIn(KEY, json.dumps(record))

    def test_x_api_key_is_retried_once_as_bearer(self):
        server = self.listing({"": (200, {"data": [gateway_item("kimi-k2.6")]})}, accept={"Bearer " + KEY})
        catalogue = mc.fetch_upstream_catalogue(server.base, {"x-api-key": KEY}, wire_api="anthropic_messages")
        self.assertEqual((catalogue.status, catalogue.auth, len(server.requests)), ("ok", "bearer_fallback", 2))
        self.assertEqual(server.requests[1]["headers"].get("authorization"), "Bearer " + KEY)
        self.assertNotIn("x-api-key", server.requests[1]["headers"])
        # a refused bearer credential is not retried in another form
        refused = mc.fetch_upstream_catalogue(server.base, {"Authorization": "Bearer wrong"},
                                              wire_api="anthropic_messages")
        self.assertEqual((refused.status, refused.reason, refused.models), ("unavailable", "http_401", ()))

    def test_next_cursor_and_after_id_pagination(self):
        server = self.listing({"": (200, {"data": [gateway_item("a")], "next_cursor": "p2"}),
                               "cursor=p2": (200, {"data": [gateway_item("b")], "next_cursor": "p3"}),
                               "cursor=p3": (200, {"data": [gateway_item("c")], "next_cursor": None})})
        catalogue = mc.fetch_model_catalogue(server.base + "/v1/models", {})
        self.assertEqual((catalogue.status, catalogue.pages, [model.id for model in catalogue.models]),
                         ("ok", 3, ["a", "b", "c"]))
        anthropic = self.listing({"": (200, {"data": [{"id": "x"}], "has_more": True, "last_id": "x"}),
                                  "after_id=x": (200, {"data": [{"id": "y"}], "has_more": False, "last_id": "y"})})
        catalogue = mc.fetch_model_catalogue(anthropic.base + "/v1/models", {"x-api-key": KEY})
        self.assertEqual(([model.id for model in catalogue.models], catalogue.status), (["x", "y"], "ok"))

    def test_page_cap_repeated_cursor_and_late_failure_truncate(self):
        loop = self.listing({"": (200, {"data": [gateway_item("a")], "next_cursor": "p2"}),
                             "cursor=p2": (200, {"data": [gateway_item("b")], "next_cursor": "p2"})})
        repeated = mc.fetch_model_catalogue(loop.base + "/v1/models", {})
        self.assertEqual((repeated.status, repeated.reason, repeated.pages), ("truncated", "cursor_repeated", 2))
        capped = mc.fetch_model_catalogue(loop.base + "/v1/models", {}, max_pages=1)
        self.assertEqual((capped.status, capped.reason, [m.id for m in capped.models]), ("truncated", "page_cap", ["a"]))
        broken = self.listing({"": (200, {"data": [gateway_item("a")], "next_cursor": "p2"}),
                               "cursor=p2": (500, {"error": "boom"})})
        late = mc.fetch_model_catalogue(broken.base + "/v1/models", {})
        self.assertEqual((late.status, late.reason, [m.id for m in late.models]), ("truncated", "http_500", ["a"]))

    def test_failures_are_a_status_never_an_exception(self):
        cases = [(self.listing({"": (500, {"error": "boom"})}), "http_500"),
                 (self.listing({"": (200, b"<html>gateway</html>")}), "invalid_json"),
                 (self.listing({"": (200, {"models": []})}), "invalid_shape")]
        for server, reason in cases:
            catalogue = mc.fetch_model_catalogue(server.base + "/v1/models", {"Authorization": "Bearer " + KEY})
            self.assertEqual((catalogue.status, catalogue.reason, catalogue.models), ("unavailable", reason, ()))
        slow = self.listing({"": (200, {"data": []})}, delay_s=1.0)
        started = time.monotonic()
        catalogue = mc.fetch_model_catalogue(slow.base + "/v1/models", {}, timeout_s=0.2)
        self.assertEqual((catalogue.status, catalogue.reason), ("unavailable", "timeout"))
        self.assertLess(time.monotonic() - started, 0.9)
        closed = mc.fetch_model_catalogue("http://127.0.0.1:9/v1/models", {}, timeout_s=0.5)
        self.assertEqual(closed.status, "unavailable")


@requires_sdk
class ClaudeSessionCatalogueTests(CatalogueCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()

    def session(self, upstream, name="run"):
        from ni_rua.core.config import Config
        from ni_rua.core.storage.store import Store
        from ni_rua.cortex.agents.claude_code.session import ClaudeCodeSession
        run = self.root / name
        run.mkdir()
        store = Store(run, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(store.close)
        config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": upstream}},
                                   "model": {"name": "kimi-k2.6"},
                                   "features": {"self_briefing": False, "visual_review_model": None}})
        return ClaudeCodeSession(run, store, lambda params: None, config=config), store

    def settings_upstream(self, base):
        path = self.root / "claude.local.json"
        path.write_text(json.dumps({"env": {"ANTHROPIC_BASE_URL": base, "ANTHROPIC_AUTH_TOKEN": KEY}}))
        return {"source": "claude_settings", "settings": str(path)}

    def wait_catalogue(self, session):
        deadline = time.monotonic() + 5
        while session._catalogue is None and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIsNotNone(session._catalogue)

    def test_background_listing_feeds_model_options_and_is_recorded(self):
        server = self.listing({"": (200, {"data": [gateway_item("kimi-k2.6", name="Kimi K2.6"),
                                                    gateway_item("glm-5.3", name="GLM-5.3", image=False),
                                                    gateway_item("wan", protocols=("NATIVE",), tools=False)]})},
                              accept={"Bearer " + KEY})
        session, store = self.session(self.settings_upstream(server.base))
        self.assertIsNone(session.model_options())           # not read yet: the console uses its default list
        session._start_model_catalogue(*session._upstream())
        self.wait_catalogue(session)
        self.assertEqual(session.model_options(), [dict(value="kimi-k2.6", name="Kimi K2.6", image_input=True),
                                                   dict(value="glm-5.3", name="GLM-5.3", image_input=False)])
        deadline = time.monotonic() + 5
        while not store.events_by_type(["agent.model_catalogue"]) and time.monotonic() < deadline:
            time.sleep(0.01)
        event, = store.events_by_type(["agent.model_catalogue"])
        self.assertEqual((event["payload"]["status"], event["payload"]["usable"]), ("ok", 2))
        self.assertNotIn(KEY, json.dumps(event))

    def test_configured_model_without_image_input_warns_once_never_refuses(self):
        import contextlib
        import io
        server = self.listing({"": (200, {"data": [gateway_item("kimi-k2.6", image=False)]})})
        session, store = self.session(self.settings_upstream(server.base))
        stderr = io.StringIO()
        with contextlib.redirect_stderr(stderr):
            session._start_model_catalogue(*session._upstream())
            self.wait_catalogue(session)
            deadline = time.monotonic() + 5
            while not store.events_by_type(["agent.model_image_input_unsupported"]) and time.monotonic() < deadline:
                time.sleep(0.01)
            time.sleep(0.05)
        warning, = store.events_by_type(["agent.model_image_input_unsupported"])
        self.assertEqual(warning["payload"], dict(model="kimi-k2.6", source="provider_model_catalogue"))
        self.assertEqual(stderr.getvalue().count("without image input"), 1)
        self.assertEqual((session.model, session.model_options()[0]["image_input"]), ("kimi-k2.6", False))

    def test_failed_listing_gives_no_suggestions_and_other_sources_do_not_read(self):
        server = self.listing({"": (500, {"error": "boom"})})
        session, store = self.session(self.settings_upstream(server.base))
        session._start_model_catalogue(*session._upstream())
        self.wait_catalogue(session)
        self.assertEqual((session._catalogue.status, session.model_options()), ("unavailable", None))
        explicit, _store = self.session({"source": "explicit", "base_url": server.base,
                                         "api_key_env": "UNUSED_TEST_KEY_ENV"}, name="explicit")
        explicit._start_model_catalogue(server.base, {"x-api-key": KEY})
        time.sleep(0.1)
        self.assertIsNone(explicit._catalogue)
        self.assertEqual(len(server.requests), 1)

# ==== tests/test_model_catalogue_reasoning.py ====

REASONING_KEY = "sk-test-wpb-catalogue-key-0123456789"

# One models.dev-shaped page (OpenCode Console shape, as listed live on 2026-11-14).
MODELS_DEV_PAGE = {"object": "list", "data": [
    {"id": "kimi-k3", "name": "Kimi K3", "modalities": {"input": ["text", "image/png"], "output": ["text"]},
     "capabilities": {"tools": {"supported": True},
                      "reasoning": {"supported": True, "toggle": True, "effort": ["low", "high", "max"]},
                      "effort_updates": False},
     "limit": {"context": 1048576, "output": 131072}},
    {"id": "qwen3.8-flash", "capabilities": {"tools": {"supported": True}, "reasoning": {
        "supported": True, "toggle": True, "effort": ["low", "medium", "xhigh"], "budget": {}}}},
    {"id": "gemini-4-hydrogen", "capabilities": {"reasoning": {
        "supported": True, "default": {"type": "effort", "value": "high"},
        "effort": ["low", "medium", "HIGH", "high"]}}},
    {"id": "big-pickle", "capabilities": {"tools": {"supported": True}, "reasoning": {"supported": True}}},
    {"id": "plain", "capabilities": {"tools": {"supported": True}}},
]}


class ReasoningListing:
    """Loopback ``GET <path>`` server answering ``body`` with ``status``; counts requests; ``delay_s`` per answer."""

    def __init__(self, body, *, status=200, path="/v1/models", delay_s=0.0):
        self.body, self.status, self.path, self.delay_s, self.requests = body, status, path, delay_s, []
        listing = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                listing.requests.append(dict(path=urlsplit(self.path).path,
                                             headers={k.lower(): v for k, v in self.headers.items()}))
                time.sleep(listing.delay_s)
                status, body = ((listing.status, listing.body) if urlsplit(self.path).path == listing.path
                                else (404, {"error": "not found"}))
                data = json.dumps(body).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                try:
                    self.wfile.write(data)
                except (BrokenPipeError, ConnectionResetError):
                    pass

        self.server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.base = f"http://127.0.0.1:{self.server.server_address[1]}"

    def close(self):
        self.server.shutdown()
        self.server.server_close()


class ReasoningParseTests(unittest.TestCase):
    def test_models_dev_reasoning_effort_list_is_read(self):
        models, _ = mc.parse_models_page(MODELS_DEV_PAGE)
        kimi, qwen, gemini, pickle, plain = models
        self.assertEqual((kimi.reasoning, kimi.reasoning_efforts, kimi.reasoning_toggle, kimi.effort_updates,
                          kimi.reasoning_budget, kimi.reasoning_default_effort),
                         (True, ("low", "high", "max"), True, False, None, None))
        self.assertEqual((qwen.reasoning_efforts, qwen.reasoning_budget), (("low", "medium", "xhigh"), True))
        # lower-cased, duplicates dropped, listing order kept; the declared default is read
        self.assertEqual((gemini.reasoning_efforts, gemini.reasoning_default_effort),
                         (("low", "medium", "high"), "high"))
        # reasoning reported without an effort list / not reported at all -> efforts None (not "unsupported")
        self.assertEqual((pickle.reasoning, pickle.reasoning_efforts), (True, None))
        self.assertEqual((plain.reasoning, plain.reasoning_efforts, plain.reasoning_toggle), (None, None, None))

    def test_other_shapes_report_no_efforts(self):
        gateway, = mc.parse_models_page({"object": "list", "next_cursor": None, "data": [
            {"id": "kimi-k3", "protocols": ["MESSAGES"], "capabilities": {"tools": True, "reasoning": True}}]})[0]
        anthropic, = mc.parse_models_page({"data": [{"id": "claude-x", "display_name": "Claude X"}],
                                           "has_more": False})[0]
        off, = mc.parse_models_page({"data": [{"id": "m", "capabilities": {"reasoning": False}}]})[0]
        self.assertEqual((gateway.reasoning, gateway.reasoning_efforts), (True, None))
        self.assertEqual(anthropic, mc.CatalogueModel(id="claude-x", name="Claude X"))
        self.assertEqual((off.reasoning, off.reasoning_efforts), (False, None))

    def test_tolerant_values(self):
        levels, bogus, empty = mc.parse_models_page({"data": [
            {"id": "a", "supported_reasoning_levels": [{"effort": "low"}, {"effort": "Max"}, 3, {"x": 1}]},
            {"id": "b", "capabilities": {"reasoning": {"supported": "yes", "toggle": 1, "effort": "high",
                                                       "budget": True, "default": {"type": "tokens", "value": 9}},
                                         "effort_updates": "no"}},
            {"id": "c", "capabilities": {"reasoning": {"supported": True, "effort": []}}}]})[0]
        self.assertEqual((levels.reasoning, levels.reasoning_efforts), (True, ("low", "max")))
        self.assertEqual((bogus.reasoning, bogus.reasoning_efforts, bogus.reasoning_toggle, bogus.reasoning_budget,
                          bogus.reasoning_default_effort, bogus.effort_updates), (None,) * 6)
        self.assertEqual((empty.reasoning, empty.reasoning_efforts), (True, ()))

    def test_to_dict_round_trip(self):
        models, _ = mc.parse_models_page(MODELS_DEV_PAGE)
        for model in models:
            data = model.to_dict()
            json.dumps(data)
            self.assertEqual(mc.CatalogueModel.from_dict(data), model)
        self.assertEqual(models[0].to_dict()["reasoning_efforts"], ["low", "high", "max"])
        self.assertEqual(mc.CatalogueModel.from_dict({"id": "x", "unknown": 1}), mc.CatalogueModel(id="x"))
        with self.assertRaisesRegex(ValueError, "invalid_model"):
            mc.CatalogueModel.from_dict({"name": "no id"})


class FailureKindTests(unittest.TestCase):
    def test_kinds(self):
        cases = {("ok", None): None, ("truncated", "page_cap"): "incomplete", ("not_read", None): "not_read",
                 ("unavailable", "http_401"): "key_refused", ("unavailable", "http_403"): "key_refused",
                 ("unavailable", "http_404"): "not_found", ("unavailable", "http_429"): "rate_limited",
                 ("unavailable", "http_400"): "request_refused", ("unavailable", "http_502"): "provider_error",
                 ("unavailable", "timeout"): "timeout", ("unavailable", "URLError"): "network",
                 ("unavailable", "invalid_json"): "invalid_listing",
                 ("unavailable", "invalid_shape"): "invalid_listing",
                 ("unavailable", "credential_unavailable:profile_missing_key"): "credential_unavailable"}
        for (status, reason), kind in cases.items():
            self.assertEqual(mc.failure_kind(status, reason), kind, (status, reason))
            self.assertTrue(kind is None or kind in mc.FAILURE_KINDS)
        self.assertEqual(mc.ModelCatalogue("unavailable", "http_401").failure, "key_refused")
        self.assertIsNone(mc.ModelCatalogue("ok").failure)


class NativeCodexTests(unittest.TestCase):
    def test_known_and_unknown_ids(self):
        sol = mc.native_codex_model("gpt-6.1-sol")
        self.assertEqual((sol.name, sol.image_input, sol.context_window, sol.reasoning, sol.reasoning_default_effort),
                         ("GPT-6.1-Sol", True, 272000, True, "low"))
        self.assertEqual(sol.reasoning_efforts, ("low", "medium", "high", "xhigh", "max", "ultra"))
        self.assertIsNone(mc.native_codex_model("kimi-k3"))
        self.assertIsNone(mc.native_codex_model(None))
        entry = mc.native_codex_entry("gpt-6.1-sol")
        entry["slug"] = "mutated"
        self.assertEqual(mc.native_codex_entry("gpt-6.1-sol")["slug"], "gpt-6.1-sol")   # a copy

    def test_merge_fills_only_unreported_fields(self):
        listed = mc.CatalogueModel(id="gpt-6.1-sol", context_window=1050000, reasoning_efforts=("low", "high"))
        merged = mc.merge_model(listed, mc.native_codex_model("gpt-6.1-sol"))
        self.assertEqual((merged.id, merged.context_window, merged.reasoning_efforts, merged.image_input),
                         ("gpt-6.1-sol", 1050000, ("low", "high"), True))
        self.assertIs(mc.merge_model(listed, None), listed)
        self.assertEqual(mc.merge_model(None, listed), listed)


class ListingUrlTests(unittest.TestCase):
    def test_explicit_listing_url_replaces_the_derived_one(self):
        server = ReasoningListing({"data": [{"id": "m"}]}, path="/custom/models")
        self.addCleanup(server.close)
        catalogue = mc.fetch_upstream_catalogue(server.base + "/v1", {"Authorization": "Bearer " + REASONING_KEY},
                                                wire_api="openai_responses", listing_url=server.base + "/custom/models")
        self.assertEqual((catalogue.status, [m.id for m in catalogue.models]), ("ok", ["m"]))
        self.assertEqual(server.requests[0]["path"], "/custom/models")


class Clock:
    def __init__(self, now=1_800_000_000.0):
        self.now = now

    def __call__(self):
        return self.now


class CatalogueCacheTests(unittest.TestCase):
    def setUp(self):
        self.server = ReasoningListing(MODELS_DEV_PAGE)
        self.addCleanup(self.server.close)
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.directory = Path(self.tmp.name) / "profiles" / mc.CACHE_DIR_NAME
        self.clock = Clock()
        self.calls = 0

    def credential(self):
        self.calls += 1
        return self.server.base, {"x-api-key": REASONING_KEY}

    def cache(self, **options):
        return mc.CatalogueCache(self.directory, clock=self.clock, timeout_s=2.0, **options)

    def read(self, cache, **options):
        options.setdefault("profile_mtime", 1)
        return cache.read("console", "anthropic_messages", credential=self.credential, **options)

    def test_second_read_within_ttl_makes_no_request_and_refresh_does(self):
        cache = self.cache()
        first = self.read(cache)
        self.assertEqual((first.status, first.origin, first.stale, first.age_s, len(first.models)),
                         ("ok", "network", False, 0.0, 5))
        self.assertEqual(first.models[0].reasoning_efforts, ("low", "high", "max"))
        self.assertEqual(first.fetched_at, "2027-01-15T08:00:00Z")
        self.clock.now += 599
        second = self.read(cache)
        self.assertEqual((second.origin, second.age_s, len(self.server.requests), self.calls), ("memory", 599.0, 1, 1))
        third = self.read(cache, refresh=True)
        self.assertEqual((third.origin, len(self.server.requests), self.calls), ("network", 2, 2))
        self.clock.now += 600   # TTL over: read again
        self.assertEqual((self.read(cache).origin, len(self.server.requests)), ("network", 3))
        # the anthropic wire gets the version header; the request carried the credential, the view does not
        self.assertEqual(self.server.requests[0]["headers"].get("anthropic-version"), mc.ANTHROPIC_VERSION)
        self.assertNotIn(REASONING_KEY, json.dumps(third.to_dict()))

    def test_changed_profile_mtime_invalidates(self):
        cache = self.cache()
        self.read(cache, profile_mtime=1)
        view = self.read(cache, profile_mtime=2)
        self.assertEqual((view.origin, len(self.server.requests)), ("network", 2))
        self.assertEqual((self.read(cache, profile_mtime=2).origin, len(self.server.requests)), ("memory", 2))

    def test_failed_refresh_returns_the_stale_list(self):
        cache = self.cache()
        self.read(cache)
        self.clock.now += 3 * 3600
        self.server.status, self.server.body = 401, {"error": {"code": "AUTHENTICATION_REQUIRED"}}
        view = self.read(cache, refresh=True)
        self.assertEqual((view.status, view.reason, view.failure, view.stale, len(view.models)),
                         ("unavailable", "http_401", "key_refused", True, 5))
        self.assertEqual((view.age_s, view.fetched_at, view.checked_at),
                         (10800.0, "2027-01-15T08:00:00Z", "2027-01-15T11:00:00Z"))
        # within the TTL the failure is answered from the cache, still with the stale list (the refused x-api-key
        # read was retried once as Bearer: two requests)
        requests = len(self.server.requests)
        self.assertEqual(requests, 3)
        again = self.read(cache)
        self.assertEqual((again.origin, again.stale, again.status, len(self.server.requests)),
                         ("memory", True, "unavailable", requests))
        # the provider answers again: fresh list, not stale
        self.server.status, self.server.body = 200, MODELS_DEV_PAGE
        self.assertEqual(self.read(cache, refresh=True).stale, False)

    def test_failure_without_earlier_list_or_after_profile_change_has_no_models(self):
        cache = self.cache()
        self.read(cache, profile_mtime=1)
        self.server.status = 500
        view = self.read(cache, profile_mtime=2)   # another profile version: the old list is not shown
        self.assertEqual((view.status, view.failure, view.stale, view.models, view.age_s, view.fetched_at),
                         ("unavailable", "provider_error", False, (), None, None))

    def test_concurrent_reads_share_one_request(self):
        self.server.delay_s = 0.4
        cache = self.cache()
        views, barrier = [], threading.Barrier(4)

        def worker():
            barrier.wait()
            views.append(self.read(cache))
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertEqual((len(views), len(self.server.requests), self.calls), (4, 1, 1))
        self.assertEqual(sorted(view.origin for view in views), ["network", "shared", "shared", "shared"])
        self.assertTrue(all(len(view.models) == 5 for view in views))

    def test_peek_reports_loading_then_the_cached_view(self):
        self.server.delay_s = 0.4
        cache = self.cache()
        self.assertIsNone(cache.peek("console", "anthropic_messages", profile_mtime=1))
        thread = threading.Thread(target=self.read, args=(cache,))
        thread.start()
        deadline = time.monotonic() + 5
        while not self.server.requests and time.monotonic() < deadline:
            time.sleep(0.01)
        loading = cache.peek("console", "anthropic_messages", profile_mtime=1)
        thread.join(10)
        self.assertEqual((loading.status, loading.refreshing, loading.failure), ("loading", True, None))
        done = cache.peek("console", "anthropic_messages", profile_mtime=1)
        self.assertEqual((done.status, done.refreshing, len(done.models), len(self.server.requests)),
                         ("ok", False, 5, 1))
        self.assertIsNone(cache.peek("console", "anthropic_messages", profile_mtime=2))

    def test_disk_copy_survives_a_restart_and_holds_no_secret(self):
        self.read(self.cache())
        path = self.directory / "console.anthropic_messages.json"
        text = path.read_text()
        self.assertNotIn(REASONING_KEY, text)
        self.assertNotIn(self.server.base, text)     # no URL either
        document = json.loads(text)
        self.assertEqual((document["schema"], document["id"], document["wire_api"], document["profile_mtime"],
                          document["status"]), (mc.CACHE_SCHEMA, "console", "anthropic_messages", 1, "ok"))
        self.assertEqual((self.directory / ".gitignore").read_text(), "*\n")
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        restarted = self.cache()
        self.clock.now += 60
        view = self.read(restarted)
        self.assertEqual((view.origin, view.age_s, len(view.models), len(self.server.requests)), ("disk", 60.0, 5, 1))
        self.assertEqual(view.models, self.read(restarted).models)
        # a corrupt file is ignored (read again), never an error
        path.write_text("{not json")
        self.assertEqual(self.read(self.cache()).origin, "network")

    def test_invalidate_forgets_memory_and_disk(self):
        cache = self.cache()
        self.read(cache)
        cache.invalidate("console")
        self.assertFalse((self.directory / "console.anthropic_messages.json").exists())
        self.assertEqual(self.read(cache).origin, "network")

    def test_credential_problems_are_statuses(self):
        cache = self.cache()

        def refused():
            raise ValueError(f"profile_missing_key:/x/console.local.json: {REASONING_KEY}")
        view = cache.read("console", "openai_responses", credential=refused, profile_mtime=1)
        self.assertEqual((view.status, view.reason, view.failure, view.models),
                         ("unavailable", "credential_unavailable:profile_missing_key", "credential_unavailable", ()))
        self.assertNotIn(REASONING_KEY, json.dumps(view.to_dict()))

        def odd():
            raise RuntimeError(REASONING_KEY)
        self.assertEqual(cache.read("other", "openai_responses", credential=odd, refresh=True).reason,
                         "credential_unavailable:RuntimeError")
        none = cache.read("codexhome", "openai_responses", credential=lambda: None)
        self.assertEqual((none.status, none.failure), ("not_read", "not_read"))
        self.assertEqual(self.server.requests, [])

    def test_memory_only_cache_and_key_validation(self):
        cache = mc.CatalogueCache(clock=self.clock)
        self.read(cache)
        self.assertEqual(self.read(cache).origin, "memory")
        self.assertFalse(self.directory.exists())
        for profile_id, wire in (("Bad", "anthropic_messages"), ("../x", "anthropic_messages"),
                                 ("ok", "carrier_pigeon"), (None, "openai_responses")):
            with self.assertRaisesRegex(ValueError, "catalogue_cache_key_invalid"):
                cache.read(profile_id, wire, credential=self.credential)

if __name__ == "__main__":
    unittest.main()
