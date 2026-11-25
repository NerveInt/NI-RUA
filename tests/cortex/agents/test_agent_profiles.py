"""Merged cortex tests:
- tests/test_account_control.py: Offline account relay and actual native Codex bootstrap; no model/GPU.
- tests/test_agent_profile_sessions.py: Agent sessions through a NI-RUA provider profile (design 16 WP-C): the Claude Code and OpenCode sessions take their
- tests/test_agent_profiles.py: Provider profiles (design 16 WP-A, section 13 Q1): loading, the key boundary, writing, config, launch gate, import.
- tests/test_provider_compat.py: Per-upstream provider-compatibility opt-ins (``UpstreamConfig.tool_result_images`` and
"""

from __future__ import annotations

NI_RUA_TEST_TIER = "component"

import base64
from datetime import datetime, timezone
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import selectors
import ssl
import subprocess
import tempfile
import threading
import time
import tomllib
import unittest
from unittest import mock
from ni_rua.cortex.session.account_control import AccountControlRelay, ACCOUNT_HOST, ACCOUNT_PATH
from support.fake_model_servers import is_host_port_probe
from ni_rua.cortex.agents.codex.session import CodexSession
from ni_rua.core.storage.store import Store
from urllib.parse import urlsplit
from cortex.opencode import test_opencode_session_unit as oc_unit
from support.fake_claude_sdk import SDK_AVAILABLE, FakeClaudeSDK, requires_sdk
from support.fake_model_servers import FakeAnthropicServer, FakeResponsesServer
from ni_rua.core.config import Config
from ni_rua.cortex.agents import profiles
from ni_rua.cortex.agents.base import AgentSession
from ni_rua.cortex.agents.model_catalogue import CACHE_DIR_NAME, CatalogueCache
from ni_rua.cortex.agents.model_listing import CodexModelListingMixin, shared_catalogue_cache
import contextlib
import io
import pickle
import shutil
import stat
from dataclasses import asdict
from unittest.mock import patch
from ni_rua.core.config import (AgentConfig, Config, ConfigError, DEFAULT_PROFILES_DIR, UpstreamConfig)
from ni_rua.cortex.agents import factory, profiles
from ni_rua.cortex.agents.legacy_profile_import import plan_import, write_import
from ni_rua.cortex.agents.profiles import (ProfileInfo, apply_profile, check_selection, compatibility, credential,
                                           expand_profile_references, forwarded_env, list_profiles, load_profile,
                                           matrix_cell, public_view, remove_profile,
                                           resolve_profile_upstream, resolve_wire, save_profile, save_profile_key)
import copy
from types import SimpleNamespace
from cortex.relay import test_model_proxy_anthropic as relay_fixture
from support.paths import REPO_ROOT
from support.fake_mcp_client import FakeMcpClient
from support.fake_model_servers import text_step
from ni_rua.core.config import Config, ConfigError
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.cortex.agents.mcp_server import (FOLDED_SCHEMA_NOTE, NiRuaMcpServer, fold_catalog_schemas,
                                             fold_schema_depth, json_depth, mcp_tool_catalog)
from ni_rua.cortex.relay.server import ModelRequestProxy
from ni_rua.cortex.session.wire import AnthropicMessagesWire, OpenAIResponsesWire
from ni_rua.cortex.session.wire.base import MOVED_IMAGE_NOTE, tool_result_image_label

# ==== tests/test_account_control.py ====

HTTPSConnection = http.client.HTTPSConnection
ACCOUNT_ID = "00000000-0000-4000-8000-000000000001"


class AccountControlTest(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.requests, self.port_probes = [], []
        self.response_status, self.response_headers = 200, {}
        self.response_body = json.dumps({"accounts": [{"id": ACCOUNT_ID,
            "workspace_backend_origin": "NO_CONSTRAINT", "account_routing_override": "NO_CONSTRAINT"}]}).encode()
        self.response_delay = 0
        owner = self

        class Upstream(BaseHTTPRequestHandler):
            def log_message(self, *_): pass
            def do_GET(self):
                if is_host_port_probe(self.command, self.path):
                    owner.port_probes.append((self.path, dict(self.headers)))   # not account control's request
                    self.send_response(404); self.send_header("Content-Length", "0"); self.end_headers()
                    return
                owner.requests.append((self.path, dict(self.headers)))
                time.sleep(owner.response_delay)
                self.send_response(owner.response_status)
                self.send_header("Content-Length", str(len(owner.response_body)))
                for key, value in owner.response_headers.items(): self.send_header(key, value)
                self.end_headers()
                try: self.wfile.write(owner.response_body)
                except (BrokenPipeError, ConnectionResetError): pass

        self.upstream = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
        self.upstream.daemon_threads = True
        self.upstream_thread = threading.Thread(target=self.upstream.serve_forever,
            kwargs={"poll_interval": .01}, daemon=True)
        self.upstream_thread.start()

        def connection(host, *, timeout):
            self.assertEqual(host, ACCOUNT_HOST, "the request cannot choose its upstream")
            return http.client.HTTPConnection("127.0.0.1", self.upstream.server_port, timeout=timeout)
        self.connector = mock.patch("ni_rua.cortex.session.account_control.HTTPSConnection", side_effect=connection)
        self.connector.start()
        self.relay = AccountControlRelay(self.root / "private", self.root / "public-ca.pem").start()
        self.context = ssl.create_default_context(cafile=self.relay.ca_path)

    def tearDown(self):
        self.relay.close()
        self.connector.stop()
        self.upstream.shutdown()
        self.upstream.server_close()
        self.upstream_thread.join(1)
        self.temporary.cleanup()

    def request(self, path=ACCOUNT_PATH, method="GET", headers=None, body=None):
        connection = HTTPSConnection("127.0.0.1", self.relay.server.server_port,
                                     timeout=3, context=self.context)
        try:
            connection.request(method, path, body=body,
                               headers={"Authorization": "Bearer fixture-not-a-credential", **(headers or {})})
            response = connection.getresponse()
            return response.status, response.read(), dict(response.getheaders())
        finally:
            connection.close()

    def test_exact_upstream_response_with_verified_local_tls_and_minimal_headers(self):
        status, body, headers = self.request(headers={"Cookie": "must-not-forward", "X-Target": "https://other.invalid"})
        self.assertEqual((status, body), (200, self.response_body))
        self.assertEqual(headers["Cache-Control"], "no-store")
        self.assertEqual(self.requests[0][0], ACCOUNT_PATH)
        self.assertEqual(self.requests[0][1]["Authorization"], "Bearer fixture-not-a-credential")
        self.assertNotIn("Cookie", self.requests[0][1])
        self.assertNotIn("X-Target", self.requests[0][1])
        self.assertEqual(self.relay.request_count, 1)
        self.assertEqual(self.relay.last_status, 200)

    def test_rejects_other_destinations_paths_methods_and_request_bodies(self):
        paths = ["/responses", "/responses/compact", "/backend-api/wham/usage",
                 ACCOUNT_PATH + "?next=https://other.invalid", ACCOUNT_PATH + "/",
                 "/backend-api/wham/../wham/accounts/check", "/backend-api/wham/accounts/%63heck",
                 "https://other.invalid" + ACCOUNT_PATH]
        for path in paths:
            with self.subTest(path=path): self.assertEqual(self.request(path)[0], 404)
        for method in ["POST", "PUT", "PATCH", "DELETE", "CONNECT", "OPTIONS"]:
            with self.subTest(method=method): self.assertEqual(self.request(method=method)[0], 405)
        self.assertEqual(self.request(body=b"unexpected-body")[0], 400)
        self.assertEqual(self.request(headers={"Authorization": ""})[0], 401)
        self.assertFalse(self.requests)
        self.assertEqual(self.relay.request_count, 0)

    def test_host_credentials_override_client_and_allow_no_client_authorization(self):
        host = {'authorization': 'Bearer fixture-host-secret', 'chatgpt-account-id': 'fixture-host-account'}
        self.relay.close()
        self.relay = AccountControlRelay(self.root / 'private', self.root / 'public-ca.pem',
                                         upstream_auth_headers=host).start()
        self.context = ssl.create_default_context(cafile=self.relay.ca_path)
        host['authorization'] = 'Bearer changed-after-construction'
        for supplied in ({'Authorization': ''}, {'Authorization': 'not-bearer'},
                         {'aUtHoRiZaTiOn': 'Bearer fixture-client-secret',
                          'cHaTgPt-AcCoUnT-Id': 'fixture-client-account'}):
            status, body, _ = self.request(headers=supplied)
            self.assertEqual((status, body), (200, self.response_body))
            sent = {key.lower(): value for key, value in self.requests[-1][1].items()}
            self.assertEqual(sent['authorization'], 'Bearer fixture-host-secret')
            self.assertEqual(sent['chatgpt-account-id'], 'fixture-host-account')
            self.assertNotIn(b'fixture-host-secret', body)
        self.assertEqual(self.relay.request_count, 3)

    def test_empty_host_auth_never_falls_back_to_client_credentials(self):
        self.relay.close()
        self.relay = AccountControlRelay(self.root / 'private', self.root / 'public-ca.pem',
                                         upstream_auth_headers={}).start()
        self.context = ssl.create_default_context(cafile=self.relay.ca_path)
        self.assertEqual(self.request()[0], 401)
        self.assertEqual(self.requests, [])

    def test_invalid_host_auth_validation_does_not_expose_secret(self):
        with self.assertRaises(ValueError) as raised:
            AccountControlRelay(self.root / 'unused', self.root / 'unused.pem',
                                upstream_auth_headers={'Authorization': 'fixture-secret\nInjected: yes'})
        self.assertEqual(str(raised.exception), 'invalid_upstream_auth_headers')

    def test_redirects_errors_oversized_and_slow_responses_are_bounded(self):
        self.response_status = 302
        self.response_headers = {"Location": "https://other.invalid/credential-sink"}
        self.response_body = b"fixture-private-upstream-detail"
        status, body, headers = self.request()
        self.assertEqual(status, 502)
        self.assertNotIn("Location", headers)
        self.assertNotIn(b"fixture-private", body)
        self.assertEqual(len(self.requests), 1, "a redirect is never followed")
        self.response_status = 401
        self.assertEqual(self.request()[0], 401, "Codex retains its native unauthorized semantics")
        self.response_status, self.response_headers = 200, {}
        self.relay.max_response_bytes = 32
        self.response_body = b'{"accounts":[],"padding":"' + b"x" * 100 + b'"}'
        self.assertEqual(self.request()[0], 502)
        self.relay.max_response_bytes = 1024
        self.response_body, self.response_delay = b'{"accounts":[]}', .25
        self.relay.timeout_s = .05
        started = time.monotonic()
        self.assertEqual(self.request()[0], 502)
        self.assertLess(time.monotonic() - started, .5)

    def profile(self):
        profile = self.root / "profile"
        profile.mkdir()
        (profile / "config.toml").write_text('model_provider = "openai"\n')
        claims = {"email": "fixture@example.invalid", "https://api.openai.com/auth": {
            "chatgpt_account_id": ACCOUNT_ID, "chatgpt_user_id": "fixture-user", "chatgpt_plan_type": "plus"}}
        encode = lambda value: base64.urlsafe_b64encode(json.dumps(value).encode()).decode().rstrip("=")
        token = encode({"alg": "none"}) + "." + encode(claims) + ".synthetic-signature"
        (profile / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {
            "id_token": token, "access_token": "fixture-not-a-credential", "refresh_token": "fixture-refresh",
            "account_id": ACCOUNT_ID}, "last_refresh": datetime.now(timezone.utc).isoformat()}))
        return profile

    def test_actual_sandbox_codex_uses_placeholder_and_host_relay_keeps_chatgpt_credentials(self):
        profile = self.profile()
        store = Store(self.root / "run", dict(run_id="account-fixture", episode_id="none",
                                             env_instance_id="none", env_generation=0))
        from ni_rua.core.config import Config
        session = CodexSession(store.root, store, lambda params: None,
                               config=Config.from_dict({"model": {"codex_home": str(profile)}}))
        process = selector = None
        try:
            # Authentication is frozen at construction; a later host environment
            # change cannot switch the selected provider or credentials.
            with mock.patch.dict(os.environ, {"CODEX_HOME": str(self.root / "unused-home")}): session._prepare()
            config = tomllib.loads((session.codex_home / "config.toml").read_text())
            self.assertFalse((session.private / "isolation.json").exists())
            self.assertTrue(config["model_providers"][config["model_provider"]]["base_url"].startswith("http://127.0.0.1:"))
            self.assertEqual(config["chatgpt_base_url"], session.account_control.base_url)
            self.assertEqual(session.env["CODEX_CA_CERTIFICATE"], str(session.account_control.ca_path))
            self.assertEqual(session.env["CODEX_HOME"], str(session.codex_home))
            self.assertEqual(config["sandbox_mode"], "danger-full-access")
            self.assertEqual(config["web_search"], "disabled")
            for feature in ("shell_tool", "unified_exec"):
                self.assertTrue(config["features"][feature], feature)
            self.assertFalse(config["features"]["network_proxy"])
            self.assertEqual(json.loads((session.codex_home / 'auth.json').read_text()),
                             {'OPENAI_API_KEY': 'ni-rua-local-relay'})
            stderr_path = session.private / "fixture-stderr.log"
            with stderr_path.open("wb") as stderr:
                # Match CodexSession.start with the fixture's frozen environment.
                process = subprocess.Popen(session.command[:-1] + ["stdio://"], stdin=subprocess.PIPE,
                    stdout=subprocess.PIPE, stderr=stderr, env=session.env, cwd=session.actor_dir)
                session.process = process
                selector = selectors.DefaultSelector()
                selector.register(process.stdout, selectors.EVENT_READ)
                pending = b""

                def send(value):
                    process.stdin.write((json.dumps(value) + "\n").encode()); process.stdin.flush()

                def receive(identifier):
                    nonlocal pending
                    deadline = time.monotonic() + 10
                    while time.monotonic() < deadline:
                        self.assertTrue(selector.select(deadline - time.monotonic()), "native RPC timed out")
                        chunk = os.read(process.stdout.fileno(), 65536)
                        self.assertTrue(chunk, "native app-server exited: " +
                                        stderr_path.read_text(errors="replace"))
                        pending += chunk
                        while b"\n" in pending:
                            line, pending = pending.split(b"\n", 1)
                            value = json.loads(line)
                            if value.get("id") == identifier: return value
                    self.fail("native RPC did not respond")

                send({"id": 1, "method": "initialize", "params": {"clientInfo": {
                    "name": "offline-account-fixture", "title": "Offline account fixture", "version": "1"},
                    "capabilities": {"experimentalApi": True}}})
                self.assertIn("result", receive(1)); send({"method": "initialized"})
                send({"id": 2, "method": "account/read", "params": {"refreshToken": False}})
                account = receive(2)
                self.assertIn("result", account)
                self.assertEqual(account["result"]["account"]["type"], "apiKey")
                self.assertEqual(session.model_proxy.request_count, 0)
                self.assertEqual(session.model_proxy.attempt_count, 0)
                self.assertEqual(session.account_control.request_count, 0)
                connection = HTTPSConnection('127.0.0.1', session.account_control.server.server_port,
                    timeout=3, context=ssl.create_default_context(cafile=session.account_control.ca_path))
                try:
                    connection.request('GET', ACCOUNT_PATH)
                    response = connection.getresponse()
                    self.assertEqual((response.status, response.read()), (200, self.response_body))
                finally:
                    connection.close()
                sent = {key.lower(): value for key, value in self.requests[-1][1].items()}
                self.assertEqual(sent['authorization'], 'Bearer fixture-not-a-credential')
                self.assertEqual(sent['chatgpt-account-id'], ACCOUNT_ID)
        finally:
            if selector: selector.close()
            if process:
                process.stdin.close(); process.stdout.close()
            session.close()
            store.close()

# ==== tests/test_agent_profile_sessions.py ====

if SDK_AVAILABLE:
    from ni_rua.cortex.agents.claude_code import options as claude_options
    from ni_rua.cortex.agents.claude_code.session import ClaudeCodeSession

SESS_FAKE_KEY = "sk-test-PROFILE-SESSION-0123456789abcdefKEY"
PLACEHOLDER = "ni-rua-local-relay"
IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
TOOLS = [{"type": "function", "name": "look", "description": "Read-only view.", "inputSchema": SCHEMA}]
QUIET = {"self_briefing": False, "visual_review_model": None}
FORWARDED = {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0", "CLAUDE_CODE_MAX_OUTPUT_TOKENS": "16000"}


def listing(*items):
    """A models.dev-shaped listing page (OpenCode Console's shape)."""
    return {"object": "list", "data": [dict(item) for item in items]}


def model_item(model_id, name=None, *, image=True, context=None, output=None):
    item = {"id": model_id, "name": name or model_id, "capabilities": {"tools": {"supported": True}},
            "modalities": {"input": ["text", "image"] if image else ["text"]}}
    if context or output:
        item["limit"] = {"context": context, "output": output}
    return item


def wait_until(predicate, timeout=5.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def relay_request(session, wire, *, model="profile-model", session_header=None):
    """POST one generation request to the session's relay as the agent would: with the relay placeholder credential
    only. Returns ``(status, body)``."""
    url = urlsplit(session.model_proxy.base_url)
    if wire == "anthropic_messages":
        path = "/v1/messages"
        body = {"model": model, "max_tokens": 8, "messages": [{"role": "user", "content": "hi"}]}
        headers = {"x-api-key": PLACEHOLDER, "anthropic-version": "2023-06-01"}
    else:
        path = "/v1/responses"
        body = {"model": model, "input": [{"role": "user", "content": "hi"}], "stream": False}
        headers = {"Authorization": "Bearer " + PLACEHOLDER}
    if session_header:
        headers["x-opencode-session-id"] = session_header
    connection = http.client.HTTPConnection(url.hostname, url.port, timeout=20)
    try:
        connection.request("POST", path, body=json.dumps(body).encode(),
                           headers={"Content-Type": "application/json", **headers})
        response = connection.getresponse()
        return response.status, response.read()
    finally:
        connection.close()


class ProfileFiles:
    """Mixin: a temporary profiles directory and run directory; profile and config writers; the key scan."""

    def make_dirs(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.work = Path(directory.name).resolve()
        self.profiles_dir = self.work / "profiles"
        self.profiles_dir.mkdir()
        local = mock.patch("ni_rua.core.config.LOCAL_CONFIG_PATH", self.work / "absent.local.json")
        local.start()
        self.addCleanup(local.stop)

    def write_profile(self, profile_id, wires, *, key=SESS_FAKE_KEY, env=None, compat=None, display=None, name=None):
        document = {"schema": profiles.SCHEMA, "name": name or f"Gateway {profile_id}", "wires": wires}
        if key:
            document["credential"] = {"api_key": key}
        if compat:
            document["compat"] = compat
        backends = {}
        if env:
            backends["claude_code"] = {"env": env}
        if display:
            backends["opencode"] = {"thinking_display": display}
        if backends:
            document["backends"] = backends
        path = self.profiles_dir / f"{profile_id}.local.json"
        path.write_text(json.dumps(document))
        return path

    def load(self, document):
        """``Config.load`` of a run configuration file holding ``document`` (profile references expanded)."""
        path = self.work / "ni-rua.json"
        path.write_text(json.dumps(document))
        return Config.load(path, environ={})

    def assert_no_key(self, *texts):
        for text in texts:
            self.assertNotIn(SESS_FAKE_KEY, text)

    def assert_key_absent_from(self, *directories):
        """No file below ``directories`` holds the key (store database and WAL, logs, generated configs, caches)."""
        for directory in directories:
            for path in Path(directory).rglob("*"):
                if path.is_file() and not path.is_symlink():
                    self.assertNotIn(SESS_FAKE_KEY.encode(), path.read_bytes(), str(path))


# ---------------------------------------------------------------------------------------------- Claude Code


@requires_sdk
class ClaudeCodeProfileTests(ProfileFiles, unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.make_dirs()
        self.server = FakeAnthropicServer([]).start()
        self.addCleanup(self.server.close)
        self.server.local_routes[("GET", "/v1/models")] = (200, listing(
            model_item("profile-model", "Profile Model"), model_item("text-only", image=False)))
        self.runtime = claude_options.ClaudeRuntime(sdk_version=claude_options.SDK_VERSION,
                                                    cli_version=claude_options.CLI_VERSION,
                                                    executable="/usr/bin/true", sha256="0" * 64)
        self.runs = 0

    def profile(self, **extra):
        return self.write_profile("gw", {
            "anthropic_messages": {"base_url": self.server.base_url, "auth_scheme": "bearer", **extra},
            "openai_responses": {"base_url": self.server.base_url + "/v1"}},
            env=FORWARDED, compat={"tool_result_images": "user_blocks", "tool_schema_max_depth": 13})

    def config(self, path):
        return self.load({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "profile", "settings": str(path)}}}, "model": {"name": "profile-model", "effort": "high"},
            "features": QUIET})

    def session(self, config, cache=None):
        self.runs += 1
        run = self.work / f"run{self.runs}"
        run.mkdir()
        store = Store(run, IDENTITY)
        self.addCleanup(store.close)
        session = ClaudeCodeSession(run, store, lambda params: None, config=config,
                                    client_factory=FakeClaudeSDK().factory, runtime=self.runtime)
        if cache is not None:
            session.catalogue_cache = cache
        session.tool_parallel_hints = {"look": True}
        session.submit_timeout_s = session.interrupt_timeout_s = 2.0
        self.addCleanup(session.close)
        return session, store, run

    def test_profile_key_goes_upstream_placeholder_to_the_agent_and_variables_are_forwarded(self):
        config = self.config(self.profile())
        upstream = config.agent.claude_code.upstream
        self.assertEqual((upstream.source, upstream.provider, upstream.wire_api, upstream.base_url, upstream.api_key_env,
                          upstream.tool_result_images, upstream.tool_schema_max_depth),
                         ("profile", "gw", "anthropic_messages", self.server.base_url, None, "user_blocks", 13))
        session, store, run = self.session(config)
        session.start(TOOLS, "Be careful.")
        # The agent sees the placeholder only; the profile's variables reach its environment, names only recorded.
        env = session.launch.env
        self.assertEqual(env["ANTHROPIC_API_KEY"], claude_options.PLACEHOLDER_API_KEY)
        self.assertEqual({name: env[name] for name in FORWARDED}, FORWARDED)
        self.assertEqual(session.launch.native_settings["settings_env"], sorted(FORWARDED))
        self.assert_no_key(json.dumps(env), session.launch.launcher.read_text())
        # The relay forwards the agent's request with the profile's credential instead of the placeholder.
        status, _body = relay_request(session, "anthropic_messages")
        self.assertEqual(status, 200)
        forwarded, = self.server.generation_requests()
        self.assertEqual(forwarded.header("Authorization"), "Bearer " + SESS_FAKE_KEY)
        self.assertIsNone(forwarded.header("x-api-key"))
        # Compat opt-ins of the profile reach the relay and the MCP catalogue.
        self.assertEqual(session.model_proxy.tool_result_images, "user_blocks")
        provenance = session.provenance()
        self.assertEqual((provenance["provider_compat"]["tool_result_images"],
                          provenance["provider_compat"]["tool_schema_max_depth"]), ("user_blocks", 13))
        self.assertEqual(provenance["upstream_provider"], "gw")
        self.assertEqual(provenance["upstream_profile"], dict(id="gw", name="Gateway gw", wire_api="anthropic_messages",
                                                             wires=["anthropic_messages", "openai_responses"]))
        self.assert_no_key(json.dumps(provenance), json.dumps(store.read_state("runtime", "agent")))
        session.close()
        self.assert_key_absent_from(run, self.profiles_dir / CACHE_DIR_NAME)

    def test_listing_comes_through_the_profile_cache_at_the_models_url_and_is_shared(self):
        self.server.local_routes[("GET", "/catalog/models")] = (200, listing(
            model_item("profile-model", "Profile Model"), model_item("other-model", "Other")))
        config = self.config(self.profile(models_url=self.server.base_url + "/catalog/models"))
        first, store, _run = self.session(config)
        first.start(TOOLS, "Be careful.")
        self.assertTrue(first._catalogue_done.wait(5))
        self.assertEqual(first.model_options(), [dict(value="profile-model", name="Profile Model", image_input=True),
                                                 dict(value="other-model", name="Other", image_input=True)])
        listings = [request for request in self.server.requests if request.method == "GET"]
        self.assertEqual([request.route for request in listings], ["/catalog/models"])
        self.assertEqual(listings[0].header("Authorization"), "Bearer " + SESS_FAKE_KEY)
        self.assertTrue(wait_until(lambda: store.events_by_type(["agent.model_catalogue"])))
        event, = store.events_by_type(["agent.model_catalogue"])
        self.assertEqual({key: event["payload"][key] for key in ("status", "source", "profile", "origin", "stale")},
                         dict(status="ok", source="profile", profile="gw", origin="network", stale=False))
        self.assert_no_key(json.dumps(event))
        cached = json.loads((self.profiles_dir / CACHE_DIR_NAME / "gw.anthropic_messages.json").read_text())
        self.assertEqual([item["id"] for item in cached["models"]], ["profile-model", "other-model"])
        # A second session of the same process (e.g. a handoff successor) is served from the shared cache.
        second, store2, _run2 = self.session(config)
        second.start(TOOLS, "Be careful.")
        self.assertTrue(second._catalogue_done.wait(5))
        self.assertEqual(len([request for request in self.server.requests if request.method == "GET"]), 1)
        self.assertTrue(wait_until(lambda: store2.events_by_type(["agent.model_catalogue"])))
        self.assertEqual(store2.events_by_type(["agent.model_catalogue"])[0]["payload"]["origin"], "memory")
        self.assertIs(shared_catalogue_cache(self.profiles_dir / CACHE_DIR_NAME),
                      shared_catalogue_cache(str(self.profiles_dir) + "/./" + CACHE_DIR_NAME))

    def test_failed_listing_is_a_status_and_an_injected_cache_is_used(self):
        self.server.local_routes[("GET", "/v1/models")] = (500, {"error": "boom"})
        cache = CatalogueCache(directory=None)
        session, store, _run = self.session(self.config(self.profile()), cache=cache)
        session.start(TOOLS, "Be careful.")
        self.assertTrue(session._catalogue_done.wait(5))
        self.assertEqual((session._catalogue.status, session._catalogue.reason, session.model_options()),
                         ("unavailable", "http_500", None))
        self.assertEqual(cache.peek("gw", "anthropic_messages", profile_mtime=os.stat(
            self.profiles_dir / "gw.local.json").st_mtime_ns).reason, "http_500")
        self.assertFalse((self.profiles_dir / CACHE_DIR_NAME).exists())     # memory-only cache: nothing on disk

    def test_start_refuses_a_stale_or_keyless_profile_before_anything_reaches_the_provider(self):
        path = self.profile()
        config = self.config(path)
        document = json.loads(path.read_text())
        document["wires"]["anthropic_messages"]["base_url"] = "http://127.0.0.1:9/elsewhere"
        path.write_text(json.dumps(document))
        session, _store, _run = self.session(config)
        with self.assertRaisesRegex(ValueError, r"^profile_upstream_stale:.*gw\.local\.json") as raised:
            session.start(TOOLS, "Be careful.")
        self.assert_no_key(str(raised.exception))
        keyless = self.write_profile("nokey", {"anthropic_messages": {"base_url": self.server.base_url}}, key=None)
        session, _store, _run = self.session(self.config(keyless))
        with self.assertRaisesRegex(ValueError, r"^profile_missing_key:.*nokey\.local\.json"):
            session.start(TOOLS, "Be careful.")
        self.assertEqual(self.server.requests, [])

    def test_forwarded_variables_per_source_and_provenance_before_start(self):
        session, _store, _run = self.session(self.config(self.profile()))
        self.assertEqual(session._settings_env(), FORWARDED)
        self.assertEqual(session.provenance()["upstream_profile"], dict(
            id="gw", name="Gateway gw", wire_api="anthropic_messages", wires=["anthropic_messages", "openai_responses"]))
        explicit = Config.from_dict({"agent": {"backend": "claude_code"}, "model": {"name": "m"}, "features": QUIET})
        legacy, _store, _run = self.session(explicit)
        self.assertEqual(legacy._settings_env(), {})
        self.assertIsNone(legacy.upstream_profile_block())
        self.assertNotIn("upstream_profile", legacy.provenance())


# ---------------------------------------------------------------------------------------------- OpenCode


class OpenCodeProfileTests(ProfileFiles, oc_unit.SessionCase):
    maxDiff = None

    def setUp(self):
        super().setUp()
        self.make_dirs()
        self.anthropic = FakeAnthropicServer([]).start()
        self.addCleanup(self.anthropic.close)
        self.responses = FakeResponsesServer([]).start()
        self.addCleanup(self.responses.close)
        listing_page = listing(model_item(oc_unit.MODEL, "Test Model", context=123456, output=7777),
                               model_item("other-model", "Other"))
        self.anthropic.local_routes[("GET", "/models")] = (200, listing_page)
        self.path = self.write_profile("oc", {
            "anthropic_messages": {"base_url": self.anthropic.base_url, "auth_scheme": "x-api-key",
                                   "models_url": self.anthropic.base_url + "/models"},
            "openai_responses": {"base_url": self.responses.base_url + "/v1", "auth_scheme": "bearer"}},
            compat={"tool_result_images": "user_blocks", "tool_schema_max_depth": 13}, display="omitted")

    def profile_config(self, wire_api, **opencode):
        return self.load({"agent": {"backend": "opencode", "opencode": {
            "executable": self.executable, "upstream": {"source": "profile", "settings": str(self.path),
                                                        "wire_api": wire_api}, **opencode}},
            "model": {"name": oc_unit.MODEL, "effort": "high"}, "features": QUIET})

    def test_messages_wire_profile_key_upstream_thinking_display_compat_and_listing_limits(self):
        config = self.profile_config("anthropic_messages")
        self.assertEqual(config.agent.opencode.thinking_display, "omitted")     # the profile's default
        session = self.make(config)
        session.start(oc_unit.TOOLS, "Be careful.")
        generated = self.generated(session)
        relay = generated["providers"]["relay"]
        self.assertEqual(relay["settings"]["apiKey"], PLACEHOLDER)                # the agent holds the placeholder
        declared = relay["models"][oc_unit.MODEL]
        self.assertEqual(declared["limit"], {"context": 123456, "output": 7777})  # from the profile's listing
        self.assertEqual({variant["body"]["thinking"].get("display") for variant in declared["variants"]}, {"omitted"})
        status, _body = relay_request(session, "anthropic_messages", model=oc_unit.MODEL,
                                      session_header=session.thread_id)
        self.assertEqual(status, 200)
        forwarded, = self.anthropic.generation_requests()
        self.assertEqual(forwarded.header("x-api-key"), SESS_FAKE_KEY)
        self.assertIsNone(forwarded.header("Authorization"))
        listings = [request for request in self.anthropic.requests if request.method == "GET"]
        self.assertEqual([request.route for request in listings], ["/models"])
        self.assertEqual(session.model_proxy.tool_result_images, "user_blocks")
        provenance = session.provenance()
        self.assertEqual((provenance["provider_compat"]["tool_result_images"],
                          provenance["provider_compat"]["tool_schema_max_depth"], provenance["upstream_provider"]),
                         ("user_blocks", 13, "oc"))
        self.assertEqual(provenance["upstream_profile"], dict(id="oc", name="Gateway oc", wire_api="anthropic_messages",
                                                             wires=["anthropic_messages", "openai_responses"]))
        self.assertEqual(session.model_options()[0], dict(value=oc_unit.MODEL, name="Test Model", image_input=True))
        self.assert_no_key(json.dumps(provenance), json.dumps(generated))
        session.close()
        self.assert_key_absent_from(self.root, self.profiles_dir / CACHE_DIR_NAME)

    def test_responses_wire_of_the_same_profile_sends_its_bearer(self):
        config = self.profile_config("openai_responses")
        self.assertIsNone(config.agent.opencode.thinking_display)                 # messages-only default
        session = self.make(config)
        session.start(oc_unit.TOOLS, "Be careful.")
        self.assertEqual((session.wire_api, session.provider), ("openai_responses", "oc"))
        status, _body = relay_request(session, "openai_responses", model=oc_unit.MODEL,
                                      session_header=session.thread_id)
        self.assertEqual(status, 200)
        forwarded, = self.responses.generation_requests()
        self.assertEqual(forwarded.route, "/v1/responses")
        self.assertEqual(forwarded.header("Authorization"), "Bearer " + SESS_FAKE_KEY)
        self.assertEqual(session.provenance()["upstream_profile"]["wire_api"], "openai_responses")
        session.close()
        self.assert_key_absent_from(self.root)

    def test_explicit_thinking_display_of_the_run_wins_and_legacy_source_is_unchanged(self):
        config = self.profile_config("anthropic_messages", thinking_display="summarized")
        session = self.make(config)
        session.start(oc_unit.TOOLS, "Be careful.")
        variants = self.generated(session)["providers"]["relay"]["models"][oc_unit.MODEL]["variants"]
        self.assertEqual({variant["body"]["thinking"].get("display") for variant in variants}, {"summarized"})
        legacy = self.make(self.config())
        self.assertIsNone(legacy.upstream_profile_block())
        self.assertEqual(type(legacy).model_catalogue_sources, ("explicit", "opencode_config", "profile"))


# ---------------------------------------------------------------------------------------------- Codex


class CodexListingStub(CodexModelListingMixin, AgentSession):
    """The listing half of a Codex session (what ``CodexSession`` gains by inheriting the mixin)."""
    backend_kind = "codex"

    def __init__(self, config, store):
        self.config, self.store = config, store


class CodexModelOptionsTests(ProfileFiles, unittest.TestCase):
    maxDiff = None

    def setUp(self):
        self.make_dirs()
        self.server = FakeResponsesServer([]).start()
        self.addCleanup(self.server.close)
        self.server.local_routes[("GET", "/v1/models")] = (200, {"object": "list", "data": [
            {"id": "gpt-6.1-sol", "object": "model"}, {"id": "vendor-model", "object": "model", "name": "Vendor"}]})
        run = self.work / "run"
        run.mkdir()
        self.run_dir = run
        self.store = Store(run, IDENTITY)
        self.addCleanup(self.store.close)

    def codex_home(self, *, chatgpt=False):
        home = self.work / ("chatgpt-home" if chatgpt else "api-home")
        home.mkdir()
        if chatgpt:
            (home / "config.toml").write_text('model = "gpt-6.1-sol"\n')
            (home / "auth.json").write_text(json.dumps({"auth_mode": "chatgpt", "tokens": {
                "access_token": SESS_FAKE_KEY, "account_id": "acct"}}))
        else:
            (home / "config.toml").write_text(
                f'model_provider = "vendor"\n[model_providers.vendor]\nname = "Vendor"\n'
                f'base_url = "{self.server.base_url}/v1"\nwire_api = "responses"\nrequires_openai_auth = true\n')
            (home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": SESS_FAKE_KEY}))
        return home

    def wait(self, session, thread):
        self.assertIsNotNone(thread)
        thread.join(5)
        self.assertTrue(session._catalogue_done.is_set())

    def test_profile_listing_is_read_through_the_cache_and_merged_with_the_native_catalogue(self):
        path = self.write_profile("vendor", {"openai_responses": {"base_url": self.server.base_url + "/v1"}})
        config = self.load({"agent": {"codex_upstream": {"source": "profile", "settings": str(path)}},
                            "model": {"name": "gpt-6.1-sol"}})
        session = CodexListingStub(config, self.store)
        self.assertEqual(session._catalogue_source(), "profile")
        self.wait(session, session._start_model_catalogue(self.server.base_url + "/v1",
                                                          {"Authorization": "Bearer " + SESS_FAKE_KEY}))
        self.assertEqual(session.model_options(), [dict(value="gpt-6.1-sol", name="GPT-6.1-Sol", image_input=True),
                                                   dict(value="vendor-model", name="Vendor", image_input=None)])
        listing_request, = self.server.requests
        self.assertEqual((listing_request.route, listing_request.header("Authorization")),
                         ("/v1/models", "Bearer " + SESS_FAKE_KEY))
        self.assertTrue((self.profiles_dir / CACHE_DIR_NAME / "vendor.openai_responses.json").is_file())
        self.assertEqual(session.upstream_profile_block(), dict(id="vendor", name="Gateway vendor",
                                                                wire_api="openai_responses", wires=["openai_responses"]))
        self.assertIsNone(session._start_model_catalogue("http://127.0.0.1:9", {}))   # once per session
        self.assert_key_absent_from(self.run_dir, self.profiles_dir / CACHE_DIR_NAME)

    def test_codex_home_api_provider_listing_is_read_directly(self):
        config = Config.from_dict({"model": {"name": "gpt-6.1-sol", "codex_home": str(self.codex_home())}})
        session = CodexListingStub(config, self.store)
        self.assertEqual(session._catalogue_source(), "codex_home")
        self.assertIsNone(session.upstream_profile_block())
        self.wait(session, session._start_model_catalogue(self.server.base_url + "/v1",
                                                          {"Authorization": "Bearer " + SESS_FAKE_KEY}))
        self.assertEqual([item["value"] for item in session.model_options()], ["gpt-6.1-sol", "vendor-model"])
        self.assertFalse((self.profiles_dir / CACHE_DIR_NAME).exists())
        self.assertTrue(wait_until(lambda: self.store.events_by_type(["agent.model_catalogue"])))
        event, = self.store.events_by_type(["agent.model_catalogue"])
        self.assertNotIn("source", event["payload"])                               # legacy event shape
        self.assert_key_absent_from(self.run_dir)

    def test_chatgpt_login_reads_nothing_and_offers_the_pinned_native_catalogue(self):
        config = Config.from_dict({"model": {"name": "gpt-6.1-sol",
                                             "codex_home": str(self.codex_home(chatgpt=True))}})
        session = CodexListingStub(config, self.store)
        self.assertIsNone(session.model_options())                                 # nothing known before start
        self.assertIsNone(session._start_model_catalogue("https://chatgpt.example/backend-api/codex",
                                                         {"Authorization": "Bearer " + SESS_FAKE_KEY}))
        options = session.model_options()
        self.assertEqual(options[0], dict(value="gpt-6.1-sol", name="GPT-6.1-Sol", image_input=True))
        values = [item["value"] for item in options]
        self.assertIn("gpt-6-astra", values)
        self.assertNotIn("codex-auto-review", values)                              # visibility "hide"
        self.assertEqual(self.server.requests, [])

    def test_failed_listing_gives_no_suggestions(self):
        self.server.local_routes[("GET", "/v1/models")] = (401, {"error": {"message": "bad key"}})
        config = Config.from_dict({"model": {"name": "gpt-6.1-sol", "codex_home": str(self.codex_home())}})
        session = CodexListingStub(config, self.store)
        self.wait(session, session._start_model_catalogue(self.server.base_url + "/v1",
                                                          {"Authorization": "Bearer " + SESS_FAKE_KEY}))
        self.assertEqual((session._catalogue.status, session._catalogue.reason, session.model_options()),
                         ("unavailable", "http_401", None))

    def test_codex_session_accepts_the_mixin(self):
        """``CodexSession`` carries the mixin (``class CodexSession(CodexModelListingMixin, AgentSession)``): its members
        win over ``AgentSession``'s and clash with nothing ``CodexSession`` sets."""
        from ni_rua.cortex.agents.codex.session import CodexSession
        config = Config.from_dict({"model": {"name": "gpt-6.1-sol", "codex_home": str(self.codex_home())}})
        self.assertTrue(issubclass(CodexSession, CodexModelListingMixin))
        session = CodexSession(self.run_dir, self.store, lambda params: None, config=config)
        self.assertIsNone(session.model_options())
        self.wait(session, session._start_model_catalogue(self.server.base_url + "/v1",
                                                          {"Authorization": "Bearer " + SESS_FAKE_KEY}))
        self.assertEqual([item["value"] for item in session.model_options()], ["gpt-6.1-sol", "vendor-model"])
        self.assertEqual(session.wire_api, "openai_responses")


class SharedCacheTests(unittest.TestCase):
    def test_one_instance_per_directory_across_threads(self):
        with tempfile.TemporaryDirectory() as directory:
            results = []
            threads = [threading.Thread(target=lambda: results.append(shared_catalogue_cache(Path(directory) / "c")))
                       for _ in range(8)]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            self.assertEqual(len({id(item) for item in results}), 1)
            self.assertIsNot(results[0], shared_catalogue_cache(Path(directory) / "other"))

# ==== tests/test_agent_profiles.py ====

REPOSITORY = REPO_ROOT
FAKE_KEY = "sk-test-FAKE-0123456789abcdefKEY"
OTHER_KEY = "sk-test-OTHER-9876543210zyxwvuKEY"


def document(**changes):
    """A valid two-wire profile document with FAKE_KEY; ``changes`` replace top-level members (None removes)."""
    value = {
        "schema": "ni_rua.provider_profile.v1",
        "name": "Example Gateway",
        "credential": {"api_key": FAKE_KEY},
        "wires": {"anthropic_messages": {"base_url": "https://gw.example/gateway/", "auth_scheme": "bearer"},
                  "openai_responses": {"base_url": "https://gw.example/gateway/v1", "auth_scheme": "bearer"}},
        "compat": {"tool_result_images": "user_blocks", "tool_schema_max_depth": 13},
        "backends": {"claude_code": {"env": {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}},
                     "opencode": {"thinking_display": "omitted"}},
        "defaults": {"model": "example-model", "effort": "high"},
    }
    for key, item in changes.items():
        if item is None:
            value.pop(key, None)
        else:
            value[key] = item
    return value


def write(directory, profile_id, value, *, raw=None):
    path = Path(directory) / f"{profile_id}.local.json"
    path.write_text(raw if raw is not None else json.dumps(value))
    return path


def no_runtime_probe():
    """Patch the launch gate's runtime probes (no CLI is run)."""
    stack = contextlib.ExitStack()
    stack.enter_context(patch.object(factory, "claude_code_runtime_problem", return_value=None))
    stack.enter_context(patch.object(factory, "opencode_runtime_problem", return_value=None))
    return stack


class TemporaryDirectoryTest(unittest.TestCase):
    def setUp(self):
        self.directory = Path(tempfile.mkdtemp(prefix="ni-rua-profiles-test-")).resolve()
        self.addCleanup(shutil.rmtree, self.directory, True)

    def assertNoKey(self, *texts):
        for text in texts:
            self.assertNotIn(FAKE_KEY, text)
            self.assertNotIn(FAKE_KEY[8:20], text)


class LoadTests(TemporaryDirectoryTest):
    def test_inline_profile_yields_wires_base_urls_and_headers(self):
        profile = load_profile(write(self.directory, "example", document()))
        self.assertEqual((profile.id, profile.name, profile.key_present), ("example", "Example Gateway", True))
        self.assertEqual(credential(profile, "anthropic_messages"),
                         ("https://gw.example/gateway", {"Authorization": "Bearer " + FAKE_KEY}))
        self.assertEqual(credential(profile, "openai_responses"),
                         ("https://gw.example/gateway/v1", {"Authorization": "Bearer " + FAKE_KEY}))
        self.assertEqual(forwarded_env(profile, "claude_code"), {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
        self.assertEqual(forwarded_env(profile, "opencode"), {})
        self.assertEqual((profile.tool_result_images, profile.tool_schema_max_depth, profile.opencode_thinking_display),
                         ("user_blocks", 13, "omitted"))
        self.assertEqual((profile.default_model, profile.default_effort), ("example-model", "high"))
        minimal = load_profile(write(self.directory, "minimal", {
            "schema": "ni_rua.provider_profile.v1", "credential": {"api_key": FAKE_KEY},
            "wires": {"openai_responses": {"base_url": "http://127.0.0.1:9/v1"}}}))
        self.assertEqual((minimal.name, minimal.wires["openai_responses"].auth_scheme, minimal.tool_result_images,
                          minimal.tool_schema_max_depth, minimal.claude_code_env), ("minimal", "bearer", "inline",
                                                                                   None, {}))

    def test_every_refusal_names_the_file_and_a_code_and_never_the_key(self):
        cases = [
            ({"schema": "v0"}, "schema"),
            ({FAKE_KEY: 1}, "unknown_keys"),
            ({"credential": {"format": "claude_settings", "file": "../claude.local.json"}}, "reference_unsupported"),
            ({"credential": {"api_key": FAKE_KEY + " tail"}}, "invalid_key"),
            ({"credential": {"api_key": FAKE_KEY, "token": FAKE_KEY}}, "unknown_keys"),
            ({"credential": FAKE_KEY}, "invalid_credential"),
            ({"name": FAKE_KEY + "\n"}, "invalid_name"),
            ({"name": "x" * 61}, "invalid_name"),
            ({"wires": {}}, "no_wires"),
            ({"wires": {FAKE_KEY: {"base_url": "https://h"}}}, "unknown_wire"),
            ({"wires": {"openai_chat": {"base_url": "https://h"}}}, "unknown_wire"),
            ({"wires": {"anthropic_messages": FAKE_KEY}}, "invalid_wires"),
            ({"wires": {"anthropic_messages": {"base_url": FAKE_KEY}}}, "invalid_base_url"),
            ({"wires": {"anthropic_messages": {"base_url": f"https://h/?key={FAKE_KEY}"}}}, "invalid_base_url"),
            ({"wires": {"anthropic_messages": {"base_url": f"https://u:{FAKE_KEY}@h"}}}, "invalid_base_url"),
            ({"wires": {"anthropic_messages": {"base_url": "https://h", "auth_scheme": FAKE_KEY}}},
             "invalid_auth_scheme"),
            ({"wires": {"anthropic_messages": {"base_url": "https://h", "models_url": "ftp://h"}}},
             "invalid_models_url"),
            ({"compat": {"tool_result_images": FAKE_KEY}}, "invalid_compat"),
            ({"compat": {"tool_schema_max_depth": 2}}, "invalid_compat"),
            ({"compat": {"tool_schema_max_depth": True}}, "invalid_compat"),
            ({"backends": {"codex": {}}}, "unknown_keys"),
            ({"backends": {"claude_code": {"env": {"ANTHROPIC_AUTH_TOKEN": FAKE_KEY}}}}, "env_reserved"),
            ({"backends": {"claude_code": {"env": {"ANTHROPIC_BASE_URL": "https://h"}}}}, "env_reserved"),
            ({"backends": {"claude_code": {"env": {"CLAUDE_CONFIG_DIR": "/x"}}}}, "env_reserved"),
            ({"backends": {"claude_code": {"env": {"1BAD": FAKE_KEY}}}}, "env_invalid_name"),
            ({"backends": {"claude_code": {"env": {"OK_NAME": 1}}}}, "env_not_string"),
            ({"backends": {"opencode": {"thinking_display": FAKE_KEY}}}, "invalid_thinking_display"),
            ({"backends": {"opencode": {"thinking_display": "summarized"}},
              "wires": {"openai_responses": {"base_url": "https://h/v1"}}},
             "thinking_display_requires_anthropic_messages"),
            ({"defaults": {"model": FAKE_KEY + " x"}}, "invalid_defaults"),
            ({"defaults": {"effort": FAKE_KEY}}, "invalid_defaults"),
        ]
        for changes, code in cases:
            with self.subTest(code=code, changes=list(changes)):
                path = write(self.directory, "case", document(**changes))
                with self.assertRaises(ValueError) as caught:
                    load_profile(path)
                message = str(caught.exception)
                self.assertTrue(message.startswith(f"profile_{code}:{path}: "), message)
                self.assertNoKey(message, repr(caught.exception))
                info = list_profiles(self.directory)[0]
                self.assertEqual((info.valid, info.error["code"]), (False, "profile_" + code))
                self.assertNoKey(json.dumps(info.to_dict()))

    def test_unreadable_file_and_bad_file_name_do_not_chain_the_text(self):
        path = write(self.directory, "broken", None, raw='{"credential": {"api_key": "%s"}, oops' % FAKE_KEY)
        with self.assertRaises(ValueError) as caught:
            load_profile(path)
        self.assertTrue(str(caught.exception).startswith(f"profile_unreadable:{path}: JSONDecodeError"))
        self.assertIsNone(caught.exception.__context__)   # the parser error (its .doc holds the text) is dropped
        self.assertIsNone(caught.exception.__cause__)
        self.assertNoKey(str(caught.exception))
        info = list_profiles(self.directory)[0]
        self.assertEqual((info.id, info.valid, info.error["code"], info.key_present),
                         ("broken", False, "profile_unreadable", False))
        bad = self.directory / "Bad Name.local.json"
        bad.write_text(json.dumps(document()))
        with self.assertRaisesRegex(ValueError, "^profile_invalid_file_name:"):
            load_profile(bad)
        with self.assertRaisesRegex(ValueError, "^profile_invalid_file_name:"):
            load_profile(self.directory / "example.json")

    def test_key_is_hidden_from_repr_serialisation_and_public_view(self):
        profile = load_profile(write(self.directory, "example", document()))
        self.assertNoKey(repr(profile), str(profile), json.dumps(public_view(profile).to_dict()))
        with self.assertRaises(TypeError):
            json.dumps(asdict(profile))   # the key holder refuses serialisation instead of leaking
        with self.assertRaises(TypeError):
            pickle.dumps(profile)
        upstream = resolve_profile_upstream(UpstreamConfig(**profiles.upstream_fields(profile, "claude_code",
                                                                                       "anthropic_messages")),
                                            "claude_code")
        self.assertEqual(upstream.headers, {"Authorization": "Bearer " + FAKE_KEY})
        self.assertNoKey(repr(upstream))
        info = public_view(profile).to_dict()
        self.assertEqual(info["wires"]["anthropic_messages"],
                         {"base_url": "https://gw.example/gateway", "host": "gw.example", "auth_scheme": "bearer",
                          "models_url": None})
        self.assertEqual((info["key_present"], info["valid"], info["error"]), (True, True, None))
        self.assertEqual(info["backends"]["claude_code"]["env"], {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})

    def test_a_profile_without_key_loads_and_refuses_credential(self):
        profile = load_profile(write(self.directory, "nokey", document(credential=None)))
        self.assertFalse(profile.key_present)
        self.assertFalse(public_view(profile).key_present)
        with self.assertRaisesRegex(ValueError, "^profile_missing_key:"):
            credential(profile, "anthropic_messages")
        with self.assertRaisesRegex(ValueError, "^profile_wire_not_served:"):
            credential(load_profile(write(self.directory, "one", document(
                wires={"openai_responses": {"base_url": "https://h/v1"}}, backends=None))), "anthropic_messages")

    def test_discovery_lists_invalid_files_and_skips_symlinks_and_other_names(self):
        write(self.directory, "b-valid", document())
        write(self.directory, "a-invalid", document(schema="nope"))
        (self.directory / "notes.json").write_text("{}")
        (self.directory / "c.local.json.removed-20260101T000000Z").write_text(json.dumps(document()))
        (self.directory / "link.local.json").symlink_to(self.directory / "b-valid.local.json")
        (self.directory / "dir.local.json").mkdir()
        infos = list_profiles(self.directory)
        self.assertEqual([(info.id, info.valid) for info in infos], [("a-invalid", False), ("b-valid", True)])
        self.assertTrue(infos[0].key_present)
        self.assertEqual(list_profiles(self.directory / "missing"), [])
        self.assertIsInstance(infos[1].mtime_ns, int)


class WriteTests(TemporaryDirectoryTest):
    def test_save_is_atomic_private_and_canonical(self):
        target = self.directory / "profiles"
        info = save_profile(target, "gw", document())
        path = target / "gw.local.json"
        self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        self.assertEqual(stat.S_IMODE(target.stat().st_mode), 0o700)
        self.assertIsInstance(info, ProfileInfo)
        self.assertTrue(info.key_present)
        self.assertNoKey(json.dumps(info.to_dict()))
        stored = json.loads(path.read_text())
        self.assertEqual(stored["credential"], {"api_key": FAKE_KEY})
        self.assertEqual(stored["wires"]["anthropic_messages"]["base_url"], "https://gw.example/gateway")
        self.assertEqual(load_profile(path).wires, load_profile(write(self.directory, "same", document())).wires)
        self.assertEqual([item.name for item in target.iterdir()], ["gw.local.json"])   # no temp file left

    def test_keep_key_preserves_the_stored_key_and_false_drops_it(self):
        save_profile(self.directory, "gw", document())
        edited = document(name="Renamed", credential={"api_key": ""})
        self.assertTrue(save_profile(self.directory, "gw", edited).key_present)
        self.assertEqual(credential(load_profile(self.directory / "gw.local.json"), "anthropic_messages")[1],
                         {"Authorization": "Bearer " + FAKE_KEY})
        self.assertTrue(save_profile(self.directory, "gw", document(credential=None)).key_present)
        self.assertEqual(load_profile(self.directory / "gw.local.json").name, "Example Gateway")
        replaced = save_profile(self.directory, "gw", document(credential={"api_key": OTHER_KEY}))
        self.assertTrue(replaced.key_present)
        self.assertIn(OTHER_KEY, (self.directory / "gw.local.json").read_text())
        self.assertFalse(save_profile(self.directory, "gw", document(credential=None), keep_key=False).key_present)
        self.assertNotIn("credential", json.loads((self.directory / "gw.local.json").read_text()))
        self.assertFalse(save_profile(self.directory, "fresh", document(credential=None)).key_present)
        (self.directory / "broken.local.json").write_text("{not json")
        with self.assertRaisesRegex(ValueError, "^profile_key_not_kept:"):
            save_profile(self.directory, "broken", document(credential=None))

    def test_save_validates_before_writing_and_never_echoes_the_key(self):
        save_profile(self.directory, "gw", document())
        before = (self.directory / "gw.local.json").read_bytes()
        with self.assertRaises(ValueError) as caught:
            save_profile(self.directory, "gw", document(wires={"anthropic_messages": {"base_url": FAKE_KEY}}))
        self.assertNoKey(str(caught.exception))
        self.assertEqual((self.directory / "gw.local.json").read_bytes(), before)
        for bad_id in ("", "UPPER", "../escape", "a" * 41, "-lead"):
            with self.subTest(bad_id=bad_id), self.assertRaisesRegex(ValueError, "^profile_invalid_id:"):
                save_profile(self.directory, bad_id, document())

    def test_failed_rename_leaves_the_original_and_no_temp_file(self):
        save_profile(self.directory, "gw", document())
        before = (self.directory / "gw.local.json").read_bytes()
        with patch.object(profiles.os, "replace", side_effect=OSError("disk full")), self.assertRaises(OSError):
            save_profile(self.directory, "gw", document(name="Changed"))
        self.assertEqual((self.directory / "gw.local.json").read_bytes(), before)
        self.assertEqual(sorted(item.name for item in self.directory.iterdir()), ["gw.local.json"])

    def test_save_key_sets_only_the_key_and_returns_no_key(self):
        save_profile(self.directory, "gw", document(credential=None), keep_key=False)
        info = save_profile_key(self.directory, "gw", "  " + OTHER_KEY + "  ")
        self.assertTrue(info.key_present)
        self.assertNotIn(OTHER_KEY, json.dumps(info.to_dict()))
        profile = load_profile(self.directory / "gw.local.json")
        self.assertEqual(credential(profile, "openai_responses")[1], {"Authorization": "Bearer " + OTHER_KEY})
        self.assertEqual(profile.name, "Example Gateway")
        self.assertEqual(stat.S_IMODE((self.directory / "gw.local.json").stat().st_mode), 0o600)
        with self.assertRaises(ValueError) as caught:
            save_profile_key(self.directory, "gw", FAKE_KEY + " with space")
        self.assertTrue(str(caught.exception).startswith("profile_invalid_key:"))
        self.assertNoKey(str(caught.exception))
        with self.assertRaisesRegex(ValueError, "^profile_invalid_key:"):
            save_profile_key(self.directory, "gw", "")
        with self.assertRaisesRegex(ValueError, "^profile_not_found:"):
            save_profile_key(self.directory, "missing", FAKE_KEY)

    def test_remove_renames_and_refuses_a_profile_in_use(self):
        save_profile(self.directory, "gw", document())
        path = self.directory / "gw.local.json"
        for in_use in (["gw"], [str(path)]):
            with self.subTest(in_use=in_use), self.assertRaisesRegex(ValueError, "^profile_in_use:"):
                remove_profile(self.directory, "gw", in_use=in_use)
        target = remove_profile(self.directory, "gw", in_use=["other"])
        self.assertFalse(path.exists())
        self.assertRegex(target.name, r"^gw\.local\.json\.removed-\d{8}T\d{6}Z$")
        self.assertIn(FAKE_KEY, target.read_text())   # retired, never unlinked
        self.assertEqual(list_profiles(self.directory), [])
        with self.assertRaisesRegex(ValueError, "^profile_not_found:"):
            remove_profile(self.directory, "gw")


class ConfigTests(TemporaryDirectoryTest):
    # Agent sections of to_dict() recorded at b341d57 (before this package): unchanged for existing configurations.
    BASELINE_AGENT = {
        "codex_evidence_relay": ({"agent": {"evidence": {"client": "relay"}}},
                                 ["backend", "claude_code", "opencode", "evidence"]),
        "claude_explicit": ({"agent": {"backend": "claude_code"}}, ["backend", "claude_code", "opencode"]),
    }

    def test_snapshots_of_existing_configurations_are_unchanged(self):
        self.assertNotIn("agent", Config().to_dict())
        self.assertEqual(AgentConfig().to_dict().keys(), {"backend", "claude_code", "opencode"})
        for name, (value, keys) in self.BASELINE_AGENT.items():
            with self.subTest(name=name):
                agent = Config.from_dict(value).to_dict()["agent"]
                self.assertEqual(list(agent), keys)
                self.assertEqual(list(agent["claude_code"]["upstream"]),
                                 ["source", "settings", "wire_api", "base_url", "api_key_env", "auth_scheme",
                                  "provider", "tool_result_images", "tool_schema_max_depth"])
                self.assertEqual(Config.from_dict(Config.from_dict(value).to_dict()), Config.from_dict(value))
        self.assertEqual(Config.from_dict({"agent": {"backend": "claude_code"}}).to_dict()["agent"]["claude_code"][
            "upstream"], {"source": "explicit", "settings": None, "wire_api": "anthropic_messages",
                          "base_url": "https://api.anthropic.com", "api_key_env": "ANTHROPIC_API_KEY",
                          "auth_scheme": "x-api-key", "provider": "anthropic", "tool_result_images": "inline",
                          "tool_schema_max_depth": None})

    def test_profile_source_codex_upstream_and_profiles_dir_validation(self):
        with self.assertRaisesRegex(ConfigError, "^upstream_profile_requires_settings"):
            UpstreamConfig(source="profile")
        self.assertEqual(UpstreamConfig(source="profile", settings="/p/x.local.json").source, "profile")
        for upstream, message in (
                ({"source": "explicit", "base_url": "https://h/v1"}, "^codex_upstream_source_unsupported:explicit"),
                ({"source": "codex_home"}, "^codex_upstream_source_unsupported:codex_home"),
                ({"source": "profile", "settings": "/p/x.local.json", "wire_api": "anthropic_messages"},
                 "^codex_upstream_requires_openai_responses"),
                ({"source": "profile", "settings": "/p/x.local.json", "tool_result_images": "user_blocks"},
                 "^codex_upstream_tool_options_unsupported"),
                ({"source": "profile"}, "upstream_profile_requires_settings")):
            with self.subTest(upstream=upstream), self.assertRaisesRegex(ConfigError, message):
                Config.from_dict({"agent": {"codex_upstream": upstream}})
        config = Config.from_dict({"agent": {"codex_upstream": {"source": "profile", "settings": "/p/x.local.json"},
                                             "profiles_dir": "/p"}})
        exported = config.to_dict()["agent"]
        self.assertEqual(exported["codex_upstream"]["settings"], "/p/x.local.json")
        self.assertEqual(exported["profiles_dir"], "/p")
        self.assertEqual(Config.from_dict(config.to_dict()), config)
        self.assertEqual(factory.actor_upstream(config, "codex"), config.agent.codex_upstream)
        self.assertIsNone(factory.actor_upstream(Config(), "codex"))
        self.assertEqual(profiles.profiles_directory(Config()), DEFAULT_PROFILES_DIR)
        self.assertEqual(profiles.profiles_directory(config), Path("/p"))
        self.assertEqual(DEFAULT_PROFILES_DIR, REPOSITORY / "configs" / "profiles")

    def test_load_resolves_profiles_dir_and_expands_a_hand_written_reference(self):
        (self.directory / "profiles").mkdir()
        write(self.directory / "profiles", "gw", document())
        config_path = self.directory / "run.json"
        config_path.write_text(json.dumps({"agent": {
            "backend": "claude_code", "profiles_dir": "profiles",
            "claude_code": {"upstream": {"source": "profile", "settings": "profiles/gw.local.json"}},
            "opencode": {"upstream": {"source": "profile", "settings": "profiles/gw.local.json",
                                      "wire_api": "anthropic_messages"}}}}))
        config = Config.load(config_path, environ={})
        self.assertEqual(config.agent.profiles_dir, str(self.directory / "profiles"))
        claude = config.agent.claude_code.upstream
        self.assertEqual((claude.source, claude.settings, claude.wire_api, claude.base_url, claude.api_key_env,
                          claude.auth_scheme, claude.provider, claude.tool_result_images, claude.tool_schema_max_depth),
                         ("profile", str(self.directory / "profiles/gw.local.json"), "anthropic_messages",
                          "https://gw.example/gateway", None, "bearer", "gw", "user_blocks", 13))
        self.assertEqual(config.agent.opencode.thinking_display, "omitted")   # the profile default
        self.assertNoKey(json.dumps(config.to_dict()))
        with no_runtime_probe():
            self.assertIsNone(factory.backend_requirements(config))
        # An explicit thinking_display of the run wins over the profile's default.
        explicit = json.loads(config_path.read_text())
        explicit["agent"]["opencode"]["thinking_display"] = "summarized"
        config_path.write_text(json.dumps(explicit))
        self.assertEqual(Config.load(config_path, environ={}).agent.opencode.thinking_display, "summarized")

    def test_load_leaves_a_broken_reference_for_the_gate_to_refuse(self):
        config_path = self.directory / "run.json"
        config_path.write_text(json.dumps({"agent": {"backend": "claude_code", "claude_code": {
            "upstream": {"source": "profile", "settings": "missing.local.json"}}}}))
        config = Config.load(config_path, environ={})
        self.assertEqual(config.agent.claude_code.upstream.source, "profile")
        with no_runtime_probe(), self.assertRaisesRegex(ConfigError, "^profile_unreadable:.*missing.local.json"):
            factory.backend_requirements(config)
        with self.assertRaisesRegex(ConfigError, "^profile_unreadable:"):
            expand_profile_references(config)

    def test_legacy_sources_still_load_and_gate(self):
        settings = self.directory / "claude.local.json"
        settings.write_text(json.dumps({"env": {"ANTHROPIC_AUTH_TOKEN": FAKE_KEY,
                                                "ANTHROPIC_BASE_URL": "https://legacy.example"}}))
        config = Config.from_dict({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "claude_settings", "settings": str(settings)}}}})
        self.assertIs(expand_profile_references(config), config)
        with no_runtime_probe():
            self.assertIsNone(factory.backend_requirements(config))
        self.assertNoKey(json.dumps(config.to_dict()))


class GateTests(TemporaryDirectoryTest):
    def setUp(self):
        super().setUp()
        self.both = load_profile(write(self.directory, "both", document()))
        self.messages = load_profile(write(self.directory, "messages", document(
            wires={"anthropic_messages": {"base_url": "https://m.example", "auth_scheme": "x-api-key"}})))
        self.responses = load_profile(write(self.directory, "responses", document(
            wires={"openai_responses": {"base_url": "https://r.example/v1"}}, backends=None)))
        self.nokey = load_profile(write(self.directory, "nokey", document(credential=None)))
        self.base = Config.from_dict({"agent": {"opencode": {"executable": "/opt/opencode"}}})

    def test_backend_wires_table(self):
        self.assertEqual(factory.BACKEND_WIRES, {"codex": ("openai_responses",),
                                                 "claude_code": ("anthropic_messages",),
                                                 "opencode": ("anthropic_messages", "openai_responses")})
        self.assertEqual(factory.OPENCODE_WIRES, factory.BACKEND_WIRES["opencode"])
        self.assertEqual(profiles.PROFILE_WIRES, ("anthropic_messages", "openai_responses"))

    def test_apply_profile_passes_the_gate_for_every_compatible_pair(self):
        pairs = [("claude_code", self.both, None, "anthropic_messages"),
                 ("claude_code", self.messages, None, "anthropic_messages"),
                 ("opencode", self.both, "anthropic_messages", "anthropic_messages"),
                 ("opencode", self.both, "openai_responses", "openai_responses"),
                 ("opencode", self.responses, None, "openai_responses"),
                 ("codex", self.both, None, "openai_responses"),
                 ("codex", self.responses, None, "openai_responses")]
        for backend, profile, wire, expected in pairs:
            with self.subTest(backend=backend, profile=profile.id, wire=wire), no_runtime_probe():
                effort = "high" if backend != "opencode" or expected == "anthropic_messages" else "medium"
                candidate = apply_profile(self.base, backend, profile, wire, model="m-1", effort=effort)
                self.assertIsNone(factory.backend_requirements(candidate))
                upstream = factory.actor_upstream(candidate, backend)
                self.assertEqual((upstream.source, upstream.wire_api, upstream.provider, upstream.api_key_env),
                                 ("profile", expected, profile.id, None))
                self.assertEqual(upstream.base_url, profile.wires[expected].base_url)
                self.assertEqual((candidate.agent.backend, candidate.model.name, candidate.model.effort),
                                 (backend, "m-1", effort))
                exported = json.dumps(candidate.to_dict())
                self.assertNoKey(exported)
                self.assertEqual(Config.from_dict(candidate.to_dict()), candidate)
                result = check_selection(self.base, backend, profile, wire, model="m-1", effort=effort)
                self.assertEqual((result.ok, result.reason, result.wire_api), (True, None, expected))
                self.assertEqual(result.to_dict(), {"ok": True, "reason": None, "message": None,
                                                    "wire_api": expected})
                resolved = resolve_profile_upstream(upstream, backend)
                self.assertEqual((resolved.wire_api, resolved.base_url),
                                 (expected, profile.wires[expected].base_url))
        codex = apply_profile(self.base, "codex", self.both)
        self.assertEqual((codex.agent.codex_upstream.tool_result_images, codex.agent.codex_upstream.tool_schema_max_depth,
                          codex.agent.codex_upstream.auth_scheme), ("inline", None, "bearer"))
        claude = apply_profile(self.base, "claude_code", self.both).agent.claude_code.upstream
        self.assertEqual((claude.tool_result_images, claude.tool_schema_max_depth), ("user_blocks", 13))

    def test_opencode_thinking_display_follows_the_profile_on_messages_only(self):
        self.assertEqual(apply_profile(self.base, "opencode", self.both, "anthropic_messages")
                         .agent.opencode.thinking_display, "omitted")
        self.assertIsNone(apply_profile(self.base, "opencode", self.both, "openai_responses", effort="medium")
                          .agent.opencode.thinking_display)
        self.assertEqual(apply_profile(self.base, "opencode", self.both, "anthropic_messages",
                                       thinking_display="summarized").agent.opencode.thinking_display, "summarized")
        with self.assertRaisesRegex(ConfigError, "^opencode_thinking_display_requires_anthropic_messages"):
            apply_profile(self.base, "opencode", self.both, "openai_responses", effort="medium",
                          thinking_display="summarized")
        with self.assertRaisesRegex(ValueError, "^thinking_display_requires_opencode"):
            apply_profile(self.base, "claude_code", self.both, thinking_display="omitted")

    def test_incompatible_pairs_give_the_documented_reasons(self):
        with no_runtime_probe():
            cases = [
                (("claude_code", self.responses, None), "no_common_wire", "profile_no_common_wire"),
                (("codex", self.messages, None), "no_common_wire", "profile_no_common_wire"),
                (("opencode", self.both, None), "wire_required", "profile_wire_ambiguous"),
                (("opencode", self.messages, "openai_responses"), "wire_not_in_profile", "profile_wire_not_served"),
                (("claude_code", self.both, "openai_responses"), "wire_unsupported", "profile_wire_unsupported"),
            ]
            for (backend, profile, wire), reason, code in cases:
                with self.subTest(backend=backend, profile=profile.id, wire=wire):
                    result = check_selection(self.base, backend, profile, wire)
                    self.assertEqual((result.ok, result.reason, result.config), (False, reason, None))
                    self.assertTrue(result.message.startswith(f"{code}:{profile.path}: "), result.message)
                    with self.assertRaisesRegex(ValueError, f"^{code}:"):
                        apply_profile(self.base, backend, profile, wire)
            missing = check_selection(self.base, "claude_code", self.nokey)
            self.assertEqual((missing.ok, missing.reason), (False, "profile_missing_key"))
            config = apply_profile(self.base, "claude_code", self.nokey)
            with self.assertRaisesRegex(ConfigError, "^profile_missing_key:"):
                factory.backend_requirements(config)
            effort = check_selection(self.base, "opencode", self.responses, effort="max")
            self.assertEqual(effort.reason, "gate:agent_effort_unsupported")
            self.assertIn("openai_responses wire", effort.message)
            invalid = check_selection(self.base, "claude_code", self.directory / "absent.local.json")
            self.assertEqual(invalid.reason, "profile_invalid:profile_unreadable")
        with patch.object(factory, "claude_code_runtime_problem", return_value="claude_code_sdk_missing:ImportError"):
            unavailable = check_selection(self.base, "claude_code", self.both)
        self.assertEqual(unavailable.reason, "backend_unavailable")
        self.assertIn("claude_code_sdk_missing", unavailable.message)
        for result in (missing, effort, invalid, unavailable):
            self.assertNoKey(json.dumps(result.to_dict()))

    def test_gate_refuses_a_configuration_that_no_longer_matches_its_profile(self):
        candidate = apply_profile(self.base, "claude_code", self.messages)
        changed = document(wires={"anthropic_messages": {"base_url": "https://moved.example",
                                                         "auth_scheme": "x-api-key"}},
                           compat={"tool_result_images": "inline"})
        write(self.directory, "messages", changed)
        with no_runtime_probe(), self.assertRaises(ConfigError) as caught:
            factory.backend_requirements(candidate)
        message = str(caught.exception)
        self.assertTrue(message.startswith(f"profile_upstream_stale:{self.messages.path}: "), message)
        self.assertIn("base_url is 'https://m.example' here, 'https://moved.example' in the profile", message)
        self.assertIn("tool_result_images is 'user_blocks' here, 'inline' in the profile", message)
        self.assertNoKey(message)
        # A reference built without Config.load / apply_profile keeps the backend defaults: refused, not guessed.
        partial = self.base.with_overrides({"agent": {"backend": "claude_code", "claude_code": {"upstream": {
            "source": "profile", "settings": str(self.both.path)}}}})
        with no_runtime_probe(), self.assertRaisesRegex(ConfigError, "^profile_upstream_stale:.*api_key_env"):
            factory.backend_requirements(partial)
        expanded = expand_profile_references(partial)
        with no_runtime_probe():
            self.assertIsNone(factory.backend_requirements(expanded))
        codex = self.base.with_overrides({"agent": {"codex_upstream": {"source": "profile",
                                                                       "settings": str(self.messages.path)}}})
        with self.assertRaisesRegex(ConfigError, "^profile_no_common_wire:"):
            factory.backend_requirements(codex)

    def test_compatibility_rows_and_matrix_cells(self):
        self.assertEqual(compatibility("opencode", self.messages), [
            {"wire": "anthropic_messages", "compatible": True, "reason": None, "message": None},
            {"wire": "openai_responses", "compatible": False, "reason": "wire_not_in_profile",
             "message": "the profile serves anthropic_messages; opencode on openai_responses needs that wire"}])
        self.assertEqual(matrix_cell("opencode", self.both)["wires"], ["anthropic_messages", "openai_responses"])
        self.assertEqual(matrix_cell("claude_code", self.responses)["reason"], "no_common_wire")
        self.assertEqual(matrix_cell("claude_code", self.nokey)["reason"], "profile_missing_key")
        write(self.directory, "zz-broken", document(schema="v0"))
        broken = [info for info in list_profiles(self.directory) if info.id == "zz-broken"][0]
        cell = matrix_cell("codex", broken)
        self.assertEqual((cell["compatible"], cell["reason"]), (False, "profile_invalid:profile_schema"))
        self.assertEqual(resolve_wire(self.both, "opencode", "openai_responses"), "openai_responses")


def _claude_file(path, key, base_url=None, *, scheme="bearer", env=None):
    values = {"ANTHROPIC_AUTH_TOKEN" if scheme == "bearer" else "ANTHROPIC_API_KEY": key, **(env or {})}
    if base_url:
        values["ANTHROPIC_BASE_URL"] = base_url
    path.write_text(json.dumps({"env": values}))
    return path


def _codex_home(root, name, *, config, auth):
    home = root / name
    home.mkdir(parents=True)
    (home / "config.toml").write_text(config)
    if auth is not None:
        (home / "auth.json").write_text(json.dumps(auth))
    return home


class ImportTests(TemporaryDirectoryTest):
    KEY_A, KEY_B, KEY_C, KEY_D, KEY_E = (f"sk-test-import-{letter * 6}-0123456789" for letter in "ABCDE")

    def setUp(self):
        super().setUp()
        configs = self.configs = self.directory / "configs"
        configs.mkdir()
        _claude_file(configs / "claude.alpha.local.json", self.KEY_A, "https://alpha.example/gw/",
                     env={"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
        (configs / "opencode.local.json").write_text(json.dumps({"providers": {"alpha": {   # V2, same key as above
            "package": "aisdk:@ai-sdk/anthropic", "name": "Alpha Gateway",
            "settings": {"baseURL": "https://alpha.example/gw/v1", "apiKey": self.KEY_A}}}}))
        (configs / "opencode.beta.local.json").write_text(json.dumps({"provider": {"beta": {   # V1
            "npm": "@ai-sdk/openai", "name": "Beta",
            "options": {"baseURL": "https://beta.example/v1", "apiKey": self.KEY_B}}}}))
        _claude_file(configs / "claude.gamma.local.json", self.KEY_C, scheme="x-api-key")
        _claude_file(configs / "claude.epsilon.local.json", self.KEY_E, "https://e1.example")
        (configs / "opencode.epsilon.local.json").write_text(json.dumps({"providers": {"epsilon": {
            "package": "@ai-sdk/anthropic", "settings": {"baseURL": "https://e2.example/v1", "apiKey": self.KEY_E}}}}))
        (configs / "opencode.nokey.local.json").write_text(json.dumps({"providers": {"nokey": {
            "package": "@ai-sdk/anthropic", "settings": {"baseURL": "https://n.example/v1",
                                                         "apiKey": "{env:SOME_KEY}"}}}}))
        (configs / "ni-rua.local.json").write_text(json.dumps({"agent": {
            "claude_code": {"upstream": {"source": "claude_settings", "settings": "claude.alpha.local.json",
                                         "tool_result_images": "user_blocks", "tool_schema_max_depth": 13}},
            "opencode": {"upstream": {"source": "opencode_config", "settings": "opencode.local.json",
                                      "provider": "alpha", "wire_api": "anthropic_messages",
                                      "tool_result_images": "user_blocks", "tool_schema_max_depth": 13}}}}))
        (configs / "so101.local.json").write_text(json.dumps({"not": "a key file"}))
        homes = self.homes = self.directory / "providers"
        _codex_home(homes, "alpha-codex", config='model_provider = "alphacodex"\n[model_providers.alphacodex]\n'
                    'name = "Alpha via Codex"\nbase_url = "https://alpha.example/gw/v1"\nwire_api = "responses"\n'
                    'requires_openai_auth = true\n', auth={"OPENAI_API_KEY": self.KEY_A})
        _codex_home(homes, "so101-delta", config='model_provider = "delta"\n[model_providers.delta]\nname = "Delta"\n'
                    'base_url = "https://delta.example/v1"\nwire_api = "responses"\nrequires_openai_auth = true\n',
                    auth={"OPENAI_API_KEY": self.KEY_D})
        _codex_home(homes, "chatgpt", config='model = "x"\n', auth={"auth_mode": "chatgpt", "tokens": {
            "access_token": "tok-" + self.KEY_D, "account_id": "acct"}})
        _codex_home(homes, "envkey", config='model_provider = "envp"\n[model_providers.envp]\n'
                    'base_url = "https://env.example/v1"\nenv_key = "ENVP_KEY"\n', auth=None)
        _codex_home(homes, "noauth", config='model_provider = "nauth"\n[model_providers.nauth]\n'
                    'base_url = "https://noauth.example/v1"\nrequires_openai_auth = true\n', auth={})
        self.target = self.directory / "profiles"
        self.originals = {path: (path.read_bytes(), path.stat().st_mtime_ns)
                          for path in [*configs.iterdir(), *homes.rglob("*")] if path.is_file()}

    def plan(self):
        with patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-must-not-be-imported", "ENVP_KEY": "sk-env-envp"}):
            return plan_import(source_dir=self.configs, codex_homes=self.homes, profiles_dir=self.target)

    def keys(self):
        return (self.KEY_A, self.KEY_B, self.KEY_C, self.KEY_D, self.KEY_E, "sk-env-must-not-be-imported",
                "sk-env-envp")

    def test_dry_run_plan_merges_one_key_into_one_profile_and_writes_nothing(self):
        entries = self.plan()
        creates = {entry.profile_id: entry for entry in entries if entry.action == "create"}
        self.assertEqual(sorted(creates), ["alpha", "beta", "delta", "epsilon", "epsilon-claude", "gamma"])
        alpha = creates["alpha"].document
        self.assertEqual(alpha["name"], "Alpha Gateway")
        self.assertEqual(alpha["wires"], {
            "anthropic_messages": {"base_url": "https://alpha.example/gw", "auth_scheme": "bearer"},
            "openai_responses": {"base_url": "https://alpha.example/gw/v1", "auth_scheme": "bearer"}})
        self.assertEqual(alpha["compat"], {"tool_result_images": "user_blocks", "tool_schema_max_depth": 13})
        self.assertEqual(alpha["backends"], {"claude_code": {"env": {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}}})
        self.assertEqual(len(creates["alpha"].sources), 3)
        self.assertTrue(any("x-api-key because its SDK fixes the header" in note for note in creates["alpha"].notes))
        self.assertEqual(creates["beta"].document["wires"],
                         {"openai_responses": {"base_url": "https://beta.example/v1", "auth_scheme": "bearer"}})
        self.assertEqual(creates["gamma"].document["wires"],
                         {"anthropic_messages": {"base_url": "https://api.anthropic.com", "auth_scheme": "x-api-key"}})
        self.assertEqual(creates["delta"].document["name"], "Delta")
        self.assertTrue(any("not merged: anthropic_messages is declared with different base URLs" in note
                            for note in creates["epsilon"].notes))
        skipped = {source for entry in entries if entry.action == "skip" for source in entry.sources}
        self.assertEqual({Path(label.split(" (")[0]).name for label in skipped},
                         {"chatgpt", "envkey", "noauth", "opencode.nokey.local.json"})
        reasons = " ".join(entry.reason for entry in entries if entry.action == "skip")
        for code in ("codex_home_chatgpt_login", "codex_home_key_in_environment", "codex_home_missing_key",
                     "opencode_config_key_substitution_unsupported"):
            self.assertIn(code, reasons)
        self.assertFalse(self.target.exists())
        text = "\n".join(line for entry in entries for line in entry.summary_lines())
        for key in self.keys():
            self.assertNotIn(key, text)
        self.assertNotIn("so101.local.json", text)

    def test_cli_dry_run_prints_ids_wires_and_urls_but_no_key(self):
        output = io.StringIO()
        with contextlib.redirect_stdout(output), patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env-must-not-be-imported"}):
            self.assertEqual(profiles.main(["import", "--dry-run", "--source-dir", str(self.configs), "--codex-homes",
                                            str(self.homes), "--profiles-dir", str(self.target)]), 0)
        text = output.getvalue()
        self.assertIn("create alpha.local.json", text)
        self.assertIn("wire openai_responses: base_url=https://alpha.example/gw/v1  auth_scheme=bearer", text)
        self.assertIn("6 profile(s) to create, 4 skipped", text)
        for key in self.keys():
            self.assertNotIn(key, text)
        self.assertFalse(self.target.exists())

    def test_import_round_trips_every_format_and_leaves_the_sources_untouched(self):
        from ni_rua.cortex.agents.claude_code.options import read_claude_settings_upstream
        from ni_rua.cortex.agents.opencode.upstream_file import read_opencode_config_upstream
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            profiles.main(["import", "--source-dir", str(self.configs), "--codex-homes", str(self.homes),
                           "--profiles-dir", str(self.target)])
        for key in self.keys():
            self.assertNotIn(key, output.getvalue())
        written = sorted(path.name for path in self.target.iterdir())
        self.assertEqual(written, sorted(f"{name}.local.json" for name in
                                         ("alpha", "beta", "delta", "epsilon", "epsilon-claude", "gamma")))
        for path in self.target.iterdir():
            self.assertEqual(stat.S_IMODE(path.stat().st_mode), 0o600)
        loaded = {info.id: load_profile(info.path) for info in list_profiles(self.target)}
        self.assertTrue(all(info.valid and info.key_present for info in list_profiles(self.target)))
        # Same wire, base URL and key as the official readers; the header follows the profile's scheme.
        self.assertEqual(credential(loaded["gamma"], "anthropic_messages"),
                         read_claude_settings_upstream(self.configs / "claude.gamma.local.json"))
        self.assertEqual(credential(loaded["epsilon-claude"], "anthropic_messages"),
                         read_claude_settings_upstream(self.configs / "claude.epsilon.local.json"))
        beta = read_opencode_config_upstream(self.configs / "opencode.beta.local.json")
        self.assertEqual(credential(loaded["beta"], "openai_responses"), (beta.base_url, beta.headers))
        epsilon = read_opencode_config_upstream(self.configs / "opencode.epsilon.local.json")
        self.assertEqual(credential(loaded["epsilon"], "anthropic_messages"), (epsilon.base_url, epsilon.headers))
        self.assertEqual(credential(loaded["alpha"], "anthropic_messages"),
                         read_claude_settings_upstream(self.configs / "claude.alpha.local.json"))
        opencode_alpha = read_opencode_config_upstream(self.configs / "opencode.local.json")
        self.assertEqual(credential(loaded["alpha"], "anthropic_messages")[0], opencode_alpha.base_url)
        self.assertEqual(credential(loaded["alpha"], "openai_responses"),
                         ("https://alpha.example/gw/v1", {"Authorization": "Bearer " + self.KEY_A}))
        self.assertEqual(credential(loaded["delta"], "openai_responses"),
                         ("https://delta.example/v1", {"Authorization": "Bearer " + self.KEY_D}))
        self.assertEqual(forwarded_env(loaded["alpha"], "claude_code"), {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"})
        with no_runtime_probe():
            for backend, profile_id in (("claude_code", "alpha"), ("opencode", "beta"), ("codex", "delta"),
                                        ("codex", "alpha")):
                with self.subTest(backend=backend, profile=profile_id):
                    config = apply_profile(Config.from_dict({"agent": {"opencode": {"executable": "/opt/oc"}}}),
                                           backend, loaded[profile_id], "openai_responses"
                                           if backend != "claude_code" else None,
                                           effort="high")
                    self.assertIsNone(factory.backend_requirements(config))
        for path, (content, mtime) in self.originals.items():
            self.assertEqual((path.read_bytes(), path.stat().st_mtime_ns), (content, mtime), path)
        # A second import never overwrites: every profile already exists.
        before = {path.name: path.read_bytes() for path in self.target.iterdir()}
        entries = self.plan()
        self.assertEqual([entry.action for entry in entries if entry.profile_id], ["skip"] * 6)
        self.assertTrue(all("profile_exists" in entry.reason for entry in entries if entry.profile_id))
        self.assertEqual(write_import(entries, self.target), [])
        self.assertEqual({path.name: path.read_bytes() for path in self.target.iterdir()}, before)


class ImportListingTests(TemporaryDirectoryTest):
    def test_known_provider_gets_its_full_listing_url_written_into_the_profile(self):
        (self.directory / "opencode.console.local.json").write_text(json.dumps({"providers": {"opencode-console": {
            "package": "aisdk:@ai-sdk/anthropic", "name": "OpenCode Console",
            "settings": {"baseURL": "https://opencode.ai/inference/anthropic/v1", "apiKey": FAKE_KEY}}}}))
        entries = plan_import(source_dir=self.directory, codex_homes=None, profiles_dir=self.directory / "profiles")
        self.assertEqual([entry.action for entry in entries], ["create"])
        self.assertEqual(entries[0].document["wires"], {"anthropic_messages": {
            "base_url": "https://opencode.ai/inference/anthropic", "auth_scheme": "x-api-key",
            "models_url": "https://opencode.ai/inference/v1/models"}})
        self.assertIn("models_url=https://opencode.ai/inference/v1/models", "\n".join(entries[0].summary_lines()))
        write_import(entries, self.directory / "profiles")
        profile = load_profile(self.directory / "profiles" / "opencode-console.local.json")
        self.assertEqual(profile.wires["anthropic_messages"].models_url, "https://opencode.ai/inference/v1/models")
        self.assertEqual(public_view(profile).wires["anthropic_messages"]["models_url"],
                         "https://opencode.ai/inference/v1/models")


class RepositoryTests(unittest.TestCase):
    def test_git_ignores_profiles_and_local_key_files(self):
        if shutil.which("git") is None or not (REPOSITORY / ".git").exists():
            self.skipTest("not a git checkout")
        for path in ("configs/profiles/x.local.json", "configs/any.local.json", "configs/profiles/.cache/x.json",
                     "configs/claude.provider.local.json", "configs/opencode.console.local.json"):
            with self.subTest(path=path):
                result = subprocess.run(["git", "check-ignore", "-q", "--no-index", path], cwd=REPOSITORY)
                self.assertEqual(result.returncode, 0, path)
        self.assertEqual(subprocess.run(["git", "check-ignore", "-q", "--no-index", "configs/profile.example.json"],
                                        cwd=REPOSITORY).returncode, 1)

    def test_example_profile_is_valid_and_holds_a_placeholder(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "example.local.json"
            shutil.copy(REPOSITORY / "configs" / "profile.example.json", path)
            profile = load_profile(path)
        self.assertEqual(credential(profile, "anthropic_messages")[1],
                         {"Authorization": "Bearer sk-REPLACE-WITH-YOUR-KEY"})
        self.assertEqual(set(profile.wires), {"anthropic_messages", "openai_responses"})

    def test_field_help_documents_every_member(self):
        documented = {name for name, _ in profiles.FIELD_HELP}
        for name in ("schema", "name", "credential.api_key", "wires.<wire>.base_url", "wires.<wire>.auth_scheme",
                     "wires.<wire>.models_url", "compat.tool_result_images", "compat.tool_schema_max_depth",
                     "backends.claude_code.env", "backends.opencode.thinking_display", "defaults.model",
                     "defaults.effort"):
            self.assertIn(name, documented)
            self.assertIn(f"``{name}``", profiles.__doc__)

class AuthSchemeWireTests(unittest.TestCase):
    """A profile wire whose credential header the relay adapter does not forward is incompatible up front (gate and
    matrix), instead of passing the gate and failing at the first request."""

    def test_x_api_key_on_responses_is_refused_and_bearer_accepted(self):
        from ni_rua.cortex.agents import profiles as P
        self.assertIsNone(P._auth_scheme_problem("openai_responses", "bearer"))
        self.assertIn("not x-api-key", P._auth_scheme_problem("openai_responses", "x-api-key"))
        self.assertIsNone(P._auth_scheme_problem("anthropic_messages", "x-api-key"))
        self.assertIsNone(P._auth_scheme_problem("anthropic_messages", "bearer"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gw.local.json"
            path.write_text(json.dumps({"schema": "ni_rua.provider_profile.v1", "name": "GW",
                                        "credential": {"api_key": "sk-test-not-real"},
                                        "wires": {"openai_responses": {"base_url": "https://gw.example/v1",
                                                                        "auth_scheme": "x-api-key"}}}))
            profile = P.load_profile(path)
            cell = P.matrix_cell("codex", profile)
            self.assertFalse(cell["compatible"])
            self.assertEqual(cell["reason"], "profile_auth_scheme_unsupported")
            self.assertNotIn("sk-test-not-real", json.dumps(cell))

# ==== tests/test_provider_compat.py ====

PNG_A = "iVBORw0KGgoAAAANSUhEUgAAAAQAAAADCAIAAAA7ljmRAAAAEElEQVR4nGM4IScHRww4OQD1xwwxB+utmQAAAABJRU5ErkJggg=="
PNG_B = PNG_A[:-8] + "QUFBQUFB"


def anthropic_image(data):
    return {"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": data}}


def canonical(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


class AnthropicMoveTests(unittest.TestCase):
    wire = AnthropicMessagesWire()

    def request(self):
        return {"model": "kimi-k2.6", "max_tokens": 64, "messages": [
            {"role": "user", "content": "look twice"},
            {"role": "assistant", "content": [{"type": "tool_use", "id": "toolu_1", "name": "look", "input": {}},
                                              {"type": "tool_use", "id": "toolu_2", "name": "look", "input": {}}]},
            {"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": "toolu_1",
                 "content": [{"type": "text", "text": "wrist"}, anthropic_image(PNG_A), anthropic_image(PNG_B)]},
                {"type": "tool_result", "tool_use_id": "toolu_2", "content": [anthropic_image(PNG_B)]},
                {"type": "text", "text": "operator note"}]},
            {"role": "assistant", "content": [{"type": "text", "text": "done"}]}]}

    def test_images_move_after_the_last_tool_result_with_labels(self):
        original = self.request()
        before = copy.deepcopy(original)
        moved, count = self.wire.move_tool_result_images(original)
        self.assertEqual(original, before)                       # input never mutated
        self.assertEqual(count, 3)
        content = moved["messages"][2]["content"]
        self.assertEqual(content[0], {"type": "tool_result", "tool_use_id": "toolu_1",
                                      "content": [{"type": "text", "text": "wrist"}]})
        self.assertEqual(content[1], {"type": "tool_result", "tool_use_id": "toolu_2",
                                      "content": [{"type": "text", "text": MOVED_IMAGE_NOTE}]})
        self.assertEqual(content[2:8], [
            {"type": "text", "text": tool_result_image_label(1, "toolu_1")}, anthropic_image(PNG_A),
            {"type": "text", "text": tool_result_image_label(2, "toolu_1")}, anthropic_image(PNG_B),
            {"type": "text", "text": tool_result_image_label(1, "toolu_2")}, anthropic_image(PNG_B)])
        self.assertEqual(content[8], {"type": "text", "text": "operator note"})
        self.assertEqual(tool_result_image_label(1, "toolu_2"), "[image 1 of tool result toolu_2]")
        self.assertEqual(moved["messages"][:2] + moved["messages"][3:], original["messages"][:2] + original["messages"][3:])
        # identical image blocks, same order as in the tool results
        images = [canonical(part) for part in content if part.get("type") == "image"]
        self.assertEqual(images, [canonical(anthropic_image(PNG_A)), canonical(anthropic_image(PNG_B)),
                                  canonical(anthropic_image(PNG_B))])

    def test_deterministic_and_no_op_without_tool_result_images(self):
        first = canonical(self.wire.move_tool_result_images(self.request())[0])
        self.assertEqual(first, canonical(self.wire.move_tool_result_images(self.request())[0]))
        moved = self.wire.move_tool_result_images(self.request())[0]
        self.assertEqual(self.wire.move_tool_result_images(moved), (moved, 0))      # idempotent
        plain = {"messages": [{"role": "user", "content": [anthropic_image(PNG_A), {"type": "text", "text": "x"}]},
                              {"role": "user", "content": [{"type": "tool_result", "tool_use_id": "t", "content": "s"}]}]}
        self.assertIs(self.wire.move_tool_result_images(plain)[0], plain)
        for request in ({}, {"messages": "x"}, []):
            self.assertEqual(self.wire.move_tool_result_images(request), (request, 0))


class ResponsesMoveTests(unittest.TestCase):
    wire = OpenAIResponsesWire(client="opencode")

    @staticmethod
    def image(data):
        return {"type": "input_image", "image_url": "data:image/png;base64," + data}

    def request(self):
        text = lambda value: {"type": "input_text", "text": value}  # noqa: E731
        return {"model": "kimi-k2.6", "input": [
            {"role": "user", "content": [text("look")]},
            {"type": "function_call", "call_id": "call_1", "name": "look", "arguments": "{}"},
            {"type": "function_call", "call_id": "call_2", "name": "look", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_1", "output": [text("wrist"), self.image(PNG_A)]},
            {"type": "function_call_output", "call_id": "call_2", "output": [self.image(PNG_B)]},
            {"type": "function_call", "call_id": "call_3", "name": "look", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_3", "output": [text("a"), text("b")]},
            {"type": "function_call", "call_id": "call_4", "name": "file", "arguments": "{}"},
            {"type": "function_call_output", "call_id": "call_4",
             "output": [text("pdf"), {"type": "input_file", "file_data": "x"}, self.image(PNG_A)]},
            {"type": "function_call_output", "call_id": "call_5", "output": "plain string"}]}

    def test_outputs_become_text_and_images_follow_each_output_run(self):
        original = self.request()
        before = copy.deepcopy(original)
        moved, count = self.wire.move_tool_result_images(original)
        self.assertEqual(original, before)
        self.assertEqual(count, 3)
        items = moved["input"]
        self.assertEqual([item.get("type", "message") for item in items], [
            "message", "function_call", "function_call", "function_call_output", "function_call_output", "message",
            "function_call", "function_call_output", "function_call", "function_call_output", "function_call_output",
            "message"])
        self.assertEqual((items[3]["output"], items[4]["output"], items[7]["output"]), ("wrist", MOVED_IMAGE_NOTE, "a\nb"))
        self.assertEqual(items[5], {"type": "message", "role": "user", "content": [
            {"type": "input_text", "text": tool_result_image_label(1, "call_1")}, self.image(PNG_A),
            {"type": "input_text", "text": tool_result_image_label(1, "call_2")}, self.image(PNG_B)]})
        self.assertEqual(items[9]["output"], [{"type": "input_text", "text": "pdf"},
                                              {"type": "input_file", "file_data": "x"}])   # other part types kept
        self.assertEqual(items[10]["output"], "plain string")
        self.assertEqual(items[11]["content"], [{"type": "input_text", "text": tool_result_image_label(1, "call_4")},
                                                self.image(PNG_A)])
        self.assertEqual(canonical(moved), canonical(self.wire.move_tool_result_images(self.request())[0]))
        plain = {"input": [{"type": "function_call_output", "call_id": "c", "output": "s"}]}
        self.assertIs(self.wire.move_tool_result_images(plain)[0], plain)


class RelayPlacementTests(unittest.TestCase):
    setUp = relay_fixture.AnthropicRelayTest.setUp
    server = relay_fixture.AnthropicRelayTest.server
    proxy = relay_fixture.AnthropicRelayTest.proxy
    call = relay_fixture.AnthropicRelayTest.call
    count = staticmethod(relay_fixture.AnthropicRelayTest.count)
    kinds = staticmethod(relay_fixture.AnthropicRelayTest.kinds)

    def body(self):
        return relay_fixture.request_body(messages=AnthropicMoveTests().request()["messages"][:3])

    def test_user_blocks_moves_on_the_forwarded_body_and_counts_in_events(self):
        server = self.server([text_step("ok")])
        proxy = self.proxy(server, tool_result_images="user_blocks")
        self.assertEqual(self.call(proxy, self.body()).status, 200)
        forwarded = server.generation_requests()[-1].json
        self.assertEqual(forwarded["messages"][2]["content"][2], {"type": "text", "text": "[image 1 of tool result toolu_1]"})
        self.assertEqual(forwarded["messages"][2]["content"][3], anthropic_image(PNG_A))
        started, = self.kinds(proxy, "model_request_started")
        self.assertEqual(started["tool_result_images_moved"], 3)

    def test_inline_default_forwards_the_body_unchanged(self):
        server = self.server([text_step("ok")])
        proxy = self.proxy(server)
        self.assertEqual(proxy.tool_result_images, "inline")
        body = self.body()
        self.assertEqual(self.call(proxy, body).status, 200)
        self.assertEqual(server.generation_requests()[-1].json["messages"], body["messages"])
        self.assertNotIn("tool_result_images_moved", self.kinds(proxy, "model_request_started")[0])

    def test_invalid_placement_refused(self):
        with self.assertRaisesRegex(ValueError, "model_proxy_invalid_tool_result_images"):
            ModelRequestProxy("http://127.0.0.1:9", wire=AnthropicMessagesWire(), tool_result_images="inside")


def nested(levels, leaf=None):
    node = leaf or {"type": "number", "description": "x in m"}
    for index in range(levels):
        node = {"type": "object", "description": f"level {index}", "properties": {f"p{index}": node}, "required": [f"p{index}"]}
    return node


class FoldingTests(unittest.TestCase):
    def test_json_depth_counts_non_empty_containers(self):
        self.assertEqual([json_depth(value) for value in (1, {}, [], {"type": "number"}, {"enum": []},
                                                           {"enum": ["a"]}, nested(1))], [0, 0, 0, 1, 1, 2, 3])

    def test_deepest_first_minimal_folding_keeps_type_and_full_contract_as_text(self):
        schema = {"type": "object", "properties": {"deep": nested(4), "shallow": {"type": "string", "enum": ["a"]}},
                  "required": ["deep"]}
        self.assertEqual(json_depth(schema), 11)
        before = copy.deepcopy(schema)
        folded, pointers = fold_schema_depth(schema, 9)
        self.assertEqual(schema, before)
        self.assertEqual(pointers, ["/properties/deep/properties/p3/properties/p2/properties/p1"])
        self.assertLessEqual(json_depth(folded), 9)
        node = folded["properties"]["deep"]["properties"]["p3"]["properties"]["p2"]["properties"]["p1"]
        self.assertEqual(node["type"], "object")
        self.assertNotIn("properties", node)
        original = before["properties"]["deep"]["properties"]["p3"]["properties"]["p2"]["properties"]["p1"]
        prefix, _, rest = node["description"].partition(FOLDED_SCHEMA_NOTE)
        self.assertEqual(prefix, "level 0 ")
        self.assertEqual(json.loads(rest), {key: value for key, value in original.items() if key != "description"})
        self.assertEqual(folded["properties"]["shallow"], schema["properties"]["shallow"])   # untouched sibling
        self.assertIs(fold_schema_depth(schema, 11)[0], schema)                               # fits: unchanged

    def test_combinators_items_and_pointer_escaping(self):
        schema = {"type": "object", "properties": {"a/b": {"oneOf": [{"type": "array", "items": nested(3)}]}}}
        folded, pointers = fold_schema_depth(schema, 6)
        self.assertEqual(pointers, ["/properties/a~1b/oneOf/0/items"])
        self.assertLessEqual(json_depth(folded), 6)
        self.assertEqual(folded["properties"]["a/b"]["oneOf"][0]["items"]["type"], "object")
        whole, pointers = fold_schema_depth({"type": "object", "enum": [[[[1]]]]}, 3)
        self.assertEqual((pointers, whole["type"]), ([""], "object"))
        with self.assertRaisesRegex(ValueError, "invalid_tool_schema_max_depth"):
            fold_schema_depth(schema, 2)

    def test_catalog_folding_reports_tools_and_pointers(self):
        catalog = [{"name": "deep", "inputSchema": {"type": "object", "properties": {"v": nested(3)}}},
                   {"name": "flat", "inputSchema": {"type": "object", "properties": {"x": {"type": "number"}}}}]
        self.assertEqual(fold_catalog_schemas(catalog, None), (catalog, {}))
        folded, report = fold_catalog_schemas(catalog, 6)
        self.assertEqual(list(report), ["deep"])
        self.assertIs(folded[1], catalog[1])


class RegistryFoldingTests(unittest.TestCase):
    """The real actor catalogue: folded at a strict gateway depth limit, the registry still rejects a bad argument inside
    the folded region with the precise error."""

    def setUp(self):
        from ni_rua.runtime.tools.builtins import create_registry
        config = Config()
        self.runtime = SimpleNamespace(config=config, disabled_tools=config.features.disabled_tools,
                                       observation_profile=config.features.observation_profile,
                                       motion_review=config.features.motion_review, enable_wait=False,
                                       vision_compare=config.features.vision_compare, visual_reviewer=None,
                                       action_policy=None, capabilities=None, programs=True)
        self.registry = create_registry(runtime=self.runtime)
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(directory.name, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(self.store.close)

    def test_real_catalogue_fits_and_validation_uses_the_full_schema(self):
        tools = self.registry.catalog(self.runtime, role="actor")
        published = mcp_tool_catalog(tools, ActorPresentation(self.store))
        folded, report = fold_catalog_schemas(published, 13)
        self.assertTrue(all(json_depth(tool["inputSchema"]) <= 13 for tool in folded))
        self.assertIn("geometry_query", report)
        self.assertTrue(all(pointer.endswith("/waypoints/items") for pointer in report["geometry_query"]), report)
        full = next(tool for tool in tools if tool["name"] == "geometry_query")["inputSchema"]
        self.assertGreater(json_depth(full), 13)
        branch = next(index for index, item in enumerate(full["properties"]["request"]["oneOf"])
                      if "/oneOf/%d/" % index in report["geometry_query"][0])
        schema = full["properties"]["request"]["oneOf"][branch]
        mode = schema["properties"]["action"]["properties"]["mode"]["const"]
        pose = {"position_m": [0.1, 0.2, 0.3], "quaternion_xyzw": [0, 0, 0, 1]}

        def arguments(left):
            return {"request": {"kind": schema["properties"]["kind"]["const"], "camera_id": "fixed",
                                "observation_id": "obs-1", "action": {"frame_id": "base", "mode": mode,
                                                                      "path": {"waypoints": [{"left": left}]}}}}
        from ni_rua.core.tool_errors import ToolInputError
        self.registry.validate("geometry_query", arguments(pose), role="actor")       # the full contract accepts it
        # A two-element position: structurally fine for the folded (published) waypoint node, wrong for the full
        # contract, which the registry still enforces with the precise error.
        bad = arguments(dict(pose, position_m=[0.1, 0.2]))
        from ni_rua.runtime.tools.registry import validator_for
        published = next(tool for tool in folded if tool["name"] == "geometry_query")["inputSchema"]
        self.assertIsNone(next(validator_for(published).iter_errors(bad), None))      # the folded node accepts it
        with self.assertRaises(ToolInputError) as raised:
            self.registry.validate("geometry_query", bad, role="actor")
        expected = next(validator_for(full).iter_errors(bad))                          # exactly the full-schema error
        self.assertEqual((raised.exception.field_path, raised.exception.code),
                         ("/" + "/".join(map(str, expected.absolute_path)), "invalid_tool_arguments"))


class McpPublicationTests(unittest.TestCase):
    def test_server_publishes_folded_schemas_and_passes_arguments_unchanged(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        store = Store(directory.name, dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(store.close)
        calls = []

        def call_tool(name, arguments, **_):
            calls.append((name, arguments))
            return {"success": True, "contentItems": [{"type": "inputText", "text": "ok"}]}
        tool = {"name": "deep", "description": "d", "inputSchema": {"type": "object", "properties": {"v": nested(3)}}}
        server = NiRuaMcpServer(tools=[tool], presentation=ActorPresentation(store), call_tool=call_tool,
                                 raw_log=lambda *args: None, on_delivery_failure=lambda *args: None,
                                 schema_max_depth=6).start()
        self.addCleanup(server.close)
        client = FakeMcpClient(server.url, token=server.token)
        client.handshake()
        listed, = client.list_tools()
        self.assertLessEqual(json_depth(listed["inputSchema"]), 6)
        self.assertEqual(server.folded_schemas, {"deep": ["/properties/v/properties/p2"]})
        arguments = {"v": {"p2": {"p1": {"p0": "not a number"}}}}
        client.call_tool("deep", arguments)
        self.assertEqual(calls, [("deep", arguments)])


class CompatConfigTests(unittest.TestCase):
    def test_defaults_choices_and_bounds(self):
        upstream = Config().agent.opencode.upstream
        self.assertEqual((upstream.tool_result_images, upstream.tool_schema_max_depth), ("inline", None))
        for bad, code in (({"tool_result_images": "moved"}, "invalid_config_choice"),
                          ({"tool_schema_max_depth": 2}, "config_below_minimum"),
                          ({"tool_schema_max_depth": "13"}, "invalid_config_type")):
            with self.assertRaisesRegex(ConfigError, "^" + code):
                Config.from_dict({"agent": {"claude_code": {"upstream": bad}}})
        with self.assertRaisesRegex(ConfigError, "^evidence_upstream_tool_options_unsupported"):
            Config.from_dict({"agent": {"backend": "claude_code", "evidence": {"client": "relay", "upstream": {
                "wire_api": "openai_responses", "base_url": "https://x.example/v1", "tool_result_images": "user_blocks"}}}})
        default = Config().to_dict()
        self.assertNotIn("agent", default)          # default agent section omitted: Codex snapshots unchanged


class OpenCodeSessionTests(oc_unit.SessionCase):
    def test_options_reach_relay_bridge_and_provenance(self):
        upstream = {"wire_api": "anthropic_messages", "base_url": "http://127.0.0.1:9", "api_key_env": oc_unit.KEY_ENV,
                    "auth_scheme": "x-api-key", "provider": "fake", "tool_result_images": "user_blocks",
                    "tool_schema_max_depth": 5}
        session = self.make(self.config(upstream=upstream))
        deep = {"type": "object", "properties": {"x": nested(2)}}
        tools = [dict(oc_unit.TOOLS[0], inputSchema=deep), oc_unit.TOOLS[1]]
        session.start(tools, "Be careful.")
        self.assertEqual(session.model_proxy.tool_result_images, "user_blocks")
        block = session.provenance()["provider_compat"]
        self.assertEqual((block["tool_result_images"], block["tool_schema_max_depth"], block["schema_depth_measure"]),
                         ("user_blocks", 5, "json_nonempty_container_depth_v1"))
        self.assertEqual(block["folded_schemas"], {"look": ["/properties/x/properties/p1"]})
        listed = self.mcp(session).list_tools()
        self.assertTrue(all(json_depth(tool["inputSchema"]) <= 5 for tool in listed))
        self.assertEqual(listed[1]["inputSchema"], oc_unit.TOOLS[1]["inputSchema"])

    def test_default_provenance(self):
        session = self.started()
        self.assertEqual(session.provenance()["provider_compat"], dict(
            tool_result_images="inline", tool_schema_max_depth=None, schema_depth_measure=None, folded_schemas={}))


if __name__ == "__main__":
    unittest.main()
