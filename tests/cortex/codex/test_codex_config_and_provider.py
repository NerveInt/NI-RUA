"""Merged cortex tests:
- tests/test_codex_native_preferences.py: Fixed harness Codex preferences, verified without model or GPU calls.
- tests/test_codex_profile_upstream.py: Codex through a provider profile (design 16 WP-C2): ``agent.codex_upstream`` instead of ``model.codex_home``.
- tests/test_codex_provider.py: Provider routing and native configuration checks; no model or GPU calls.
"""

NI_RUA_TEST_TIER = "component"

import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import os
from pathlib import Path
import sqlite3
import subprocess
import tempfile
import threading
import tomllib
import unittest
from unittest import mock
from openai_codex.generated.v2_all import ConfigReadResponse, ExperimentalFeatureListResponse
from ni_rua.cortex.agents.codex.session import CodexSession, MODEL_PROXY_PROVIDER, CODEX_PROTOCOL_VERSION
from ni_rua.core.config import Config
from ni_rua.core.storage.store import Store
from support.host_capabilities import bubblewrap_namespaces, require
import re
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.profiles import apply_profile
from ni_rua.cortex.agents.codex.session import CodexSession
from support.fake_model_servers import FakeResponsesServer, text_step
import sys
from codex_cli_bin import bundled_codex_path
from ni_rua.cortex.agents.codex.session import CodexSession, MODEL_PROXY_PROVIDER, ACTOR_ISOLATION, _model_provider, _upstream_auth
from ni_rua.core.contracts import digest
from support.paths import REPO_ROOT

# ==== tests/test_codex_native_preferences.py ====

class CodexNativePreferencesTest(unittest.TestCase):
    enabled_features = (
        "goals", "hooks", "memories", "mentions_v2", "multi_agent",
        "prevent_idle_sleep", "guardian_approval", "multi_agent_v2", "context_management",
        "shell_tool", "unified_exec",
    )
    disabled_features = (
        "apps", "plugins", "browser_use", "browser_use_external",
        "computer_use", "artifact", "image_generation", "network_proxy",
    )

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.source_config = self.source / "config.toml"
        self.source_config.write_text('model = "different-local-model"\n'
            'model_reasoning_effort = "low"\nmodel_verbosity = "low"\n'
            'service_tier = "fast"\nsuppress_unstable_features_warning = false\n'
            'model_context_window = 9999\nweb_search = "live"\n'
            'approval_policy = "on-request"\nmodel_provider = "fixture"\n'
            '[model_providers.fixture]\nbase_url = "https://fixture.invalid/v1"\n'
            '[features]\nfast_mode = true\napps = true\nplugins = true\n'
            'shell_tool = true\nmulti_agent = false\nmemories = false\n'
            'goals = false\nhooks = false\n'
            '[features.context_management]\nexperimental_mode = false\n'
            '[mcp_servers.fixture]\ncommand = "fixture-tool"\nargs = ["--flag", "two words"]\n'
            'env_vars = ["NI_RUA_NATIVE_TOOL_FIXTURE", "HOME", "CODEX_HOME", "TMPDIR"]\n'
            '[mcp_servers.fixture.env]\nMULTILINE = "line one\\nline two"\n'
            '[plugins."fixture@local"]\nenabled = true\n'
            '[marketplaces.local]\nsource_type = "local"\nsource = "/synthetic/path"\n'
            '[skills]\nconfig = [{path = "/synthetic/skill", enabled = true}]\n')
        self.store = Store(self.root / "run", dict(run_id="native-preferences", episode_id="episode",
            env_instance_id="environment", env_generation=1))
        self.addCleanup(self.store.close)
        config = Config.from_dict({"features": {"sandbox_enabled": True},
                                   "model": {"name": "gpt-6-astra", "effort": "high",
                                             "codex_home": str(self.source)}})
        self.session = CodexSession(self.store.root, self.store, lambda _: {}, config=config)
        self.addCleanup(self.session.close)
        relay = mock.MagicMock(base_url="http://127.0.0.1:54321")
        relay.start.return_value = relay
        self.proxy = mock.patch("ni_rua.cortex.agents.codex.session.ModelRequestProxy", return_value=relay)
        self.proxy.start()
        self.addCleanup(self.proxy.stop)

    def generated(self):
        return tomllib.loads((self.session.codex_home / "config.toml").read_text())

    def assert_fixed_preferences(self, config):
        # 默认 model_verbosity 为 high，忽略宿主配置中的同名低详略度偏好
        self.assertEqual(config["model_verbosity"], "high")
        self.assertEqual(config["service_tier"], "default")
        self.assertTrue(config["suppress_unstable_features_warning"])
        self.assertFalse(config["features"]["fast_mode"])
        for feature in self.enabled_features:
            if feature != "context_management":
                self.assertIs(config["features"][feature], True, feature)
        for feature in self.disabled_features:
            self.assertIs(config["features"][feature], False, feature)
        self.assertEqual(config["features"]["context_management"], {"experimental_mode": True})

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_fixed_preferences_ignore_local_values_and_preserve_actor_configuration(self):
        original = self.source_config.read_bytes()
        self.session._prepare()
        config = self.generated()
        self.assert_fixed_preferences(config)
        self.assertEqual(config["model"], "gpt-6-astra")
        self.assertEqual(config["model_reasoning_effort"], "high")
        self.assertEqual(config["model_provider"], MODEL_PROXY_PROVIDER)
        for key in ('mcp_servers', 'plugins', 'marketplaces', 'skills'):
            self.assertNotIn(key, config)
        self.assertNotIn("model_context_window", config)
        self.assertEqual(config["web_search"], "disabled")
        self.assertEqual(config["approval_policy"], "never")
        self.assertEqual(config["features"]["code_mode"], {"enabled": False})
        self.assertFalse(config["features"]["code_mode_host"])
        catalog_path = Path(config['model_catalog_json'])
        self.assertEqual(catalog_path.parent, self.session.codex_home)
        catalog = json.loads(catalog_path.read_text())
        original_catalog = json.loads((REPO_ROOT /
                                      'ni_rua/cortex/session/codex_models.json').read_text())
        self.assertEqual(catalog, {**original_catalog, 'models': [
            {**model, 'tool_mode': 'direct', 'supports_search_tool': False}
            for model in original_catalog['models']]})
        self.assertIn(config['model'], [model['slug'] for model in catalog['models']])
        self.assertEqual(self.source_config.read_bytes(), original)
        # The actual bundled binary parses the private actor config, without inference.
        result = subprocess.run(self.session.command[:-5] + [str(self.session.cli_executable), "features", "list"],
            env=self.session.env, cwd=self.session.actor_dir,
            capture_output=True, text=True, timeout=10)
        self.assertEqual(result.returncode, 0, result.stderr)
        for feature in self.enabled_features:
            self.assertEqual(next(line.split()[-1] for line in result.stdout.splitlines()
                                  if line.startswith(feature + " ")), "true")
        for feature in ("fast_mode", "code_mode", "code_mode_host", *self.disabled_features):
            self.assertEqual(next(line.split()[-1] for line in result.stdout.splitlines()
                                  if line.startswith(feature + " ")), "false")

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_host_extensions_credentials_and_environment_are_not_copied(self):
        credentials = self.source / '.credentials.json'
        credentials.write_text('{"fixture-mcp": {"access_token": "synthetic-only"}}')
        for relative in ('skills/example/SKILL.md', 'plugins/cache/example/plugin.json'):
            path = self.source / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text('fixture extension')
        (self.source / 'memories').mkdir()
        (self.source / 'memories' / 'memory_summary.md').write_text('host private memory')
        with mock.patch.dict(os.environ, {'NI_RUA_NATIVE_TOOL_FIXTURE': 'fixture value'}):
            self.session._prepare()
        self.assertNotIn('NI_RUA_NATIVE_TOOL_FIXTURE', self.session.env)
        self.assertEqual(self.session.env['HOME'], str(self.session.actor_dir))
        self.assertEqual(self.session.env['CODEX_HOME'], str(self.session.codex_home))
        self.assertEqual(self.session.env['TMPDIR'], '/tmp')
        copied_credentials = self.session.codex_home / '.credentials.json'
        self.assertFalse(copied_credentials.exists())
        self.assertIn('synthetic-only', credentials.read_text())
        for relative in ('skills/example/SKILL.md', 'plugins/cache/example/plugin.json'):
            copied = self.session.codex_home / relative
            self.assertFalse(copied.exists())
            self.assertEqual((self.source / relative).read_text(), 'fixture extension')
        self.assertFalse((self.session.codex_home / 'memories' / 'memory_summary.md').exists())
        self.assertEqual(json.loads((self.session.codex_home / 'auth.json').read_text()),
                         {'OPENAI_API_KEY': 'ni-rua-local-relay'})

    def test_dependency_mismatch_rejects_start_before_stopping_owned_processes(self):
        for field in ('CODEX_SDK_VERSION', 'CODEX_CLI_VERSION'):
            with self.subTest(field=field), mock.patch('ni_rua.cortex.agents.codex.session.' + field, '0.0.0'), \
                    mock.patch.object(self.session, '_stop_previous_owned_processes') as stop:
                with self.assertRaisesRegex(RuntimeError, 'codex_version_mismatch'):
                    self.session.start([], 'autonomous task', resume_thread_id='existing-thread')
                stop.assert_not_called()

    def test_sandbox_startup_failure_does_not_fall_back_to_host_execution(self):
        with mock.patch('ni_rua.cortex.agents.codex.sandbox.CodexSandbox.prepare', side_effect=RuntimeError('sandbox unavailable')), \
                mock.patch('ni_rua.cortex.agents.codex.session.subprocess.Popen') as process:
            with self.assertRaisesRegex(RuntimeError, 'sandbox unavailable'):
                self.session.start([], 'fixture')
            process.assert_not_called()
            self.assertIsNone(self.session.model_proxy)
            self.assertEqual(self.session.state, 'closed')

    def test_start_rolls_back_spawned_process_and_client_on_post_spawn_failure(self):
        fake_proc = mock.MagicMock()
        fake_proc.pid = 99991
        fake_proc.poll.return_value = None
        fake_pool = mock.MagicMock()
        fake_client = mock.MagicMock(
            _proc=None,
            _reader_thread=None,
            _stderr_thread=None,
            server_pool=fake_pool,
        )
        fake_client.initialize.side_effect = RuntimeError("init_boom")
        fake_proxy = mock.MagicMock()

        def fake_prepare(_expected):
            self.session.private.mkdir(parents=True, exist_ok=True)
            self.session.actor_dir.mkdir(parents=True, exist_ok=True)
            self.session.socket_path = self.root / "ready.sock"
            self.session.command = ["fixture-app-server"]
            self.session.env = {}
            self.session.cli_executable = Path("/fixture/codex")
            self.session.model_proxy = fake_proxy
            return "fixture"

        with mock.patch.object(self.session, "_stop_previous_owned_processes"), \
                mock.patch.object(self.session, "_prepare", side_effect=fake_prepare), \
                mock.patch("ni_rua.cortex.agents.codex.session.subprocess.Popen", return_value=fake_proc), \
                mock.patch.object(Path, "is_socket", return_value=True), \
                mock.patch("ni_rua.cortex.agents.codex.session.RecordedClient", return_value=fake_client):
            with self.assertRaisesRegex(RuntimeError, "init_boom"):
                self.session.start([], "fixture")
        fake_proc.terminate.assert_called_once()
        fake_client.close.assert_called_once()
        fake_pool.shutdown.assert_called_once_with(wait=True, cancel_futures=True)
        fake_proxy.close.assert_called_once()
        self.assertIsNone(self.session.client)
        self.assertTrue(self.session.log.closed)
        self.assertEqual(self.session.state, "closed")

    def test_malformed_auth_json_scrubs_credentials_and_sandbox_repair_closes_fd_on_fstat_error(self):
        from ni_rua.core.config import ConfigError, UpstreamConfig
        from ni_rua.cortex.agents.codex.sandbox import CodexSandbox
        from ni_rua.cortex.agents.codex.session import _chatgpt_login, _upstream_auth
        from ni_rua.cortex.agents.evidence import resolve_upstream
        from ni_rua.cortex.session import sandbox_pypi
        import http.client

        bad_home = self.root / "bad-auth-home"
        bad_home.mkdir()
        (bad_home / "config.toml").write_text('model = "gpt-6.1-sol"\n')
        secret = "sk-live-secret-codex-auth-token-999"
        (bad_home / "auth.json").write_text('{"OPENAI_API_KEY": "' + secret + '" broken')
        for fn in (lambda: _chatgpt_login(bad_home), lambda: _upstream_auth({"requires_openai_auth": True}, bad_home)):
            with self.assertRaisesRegex(ValueError, "codex_auth_unreadable") as caught:
                fn()
            self.assertIsNone(caught.exception.__context__)
            self.assertIsNone(caught.exception.__cause__)
            self.assertNotIn(secret, str(caught.exception))
        with self.assertRaisesRegex(ConfigError, "evidence_upstream_codex_home_unreadable") as ev_err:
            resolve_upstream(UpstreamConfig(source="codex_home"), codex_home=str(bad_home), environ={})
        self.assertIsNone(ev_err.exception.__context__)
        self.assertNotIn(secret, str(ev_err.exception))

        # CodexSandbox.repair_socket_alias closes native_fd if os.fstat raises OSError.
        ipc = self.root / "ipc-fd"
        ipc.mkdir()
        daemon = ipc / "daemon"
        daemon.mkdir()
        sock_alias = ipc / "codex.sock"
        os.symlink(f"/tmp/codex-daemon-{os.getuid()}/" + ("a" * 64), sock_alias)
        box = object.__new__(CodexSandbox)
        box.ipc = ipc
        box.daemon_directory_fd = os.open(daemon, os.O_RDONLY)
        box.native_socket_fd = None
        closed_fds = []
        try:
            real_fstat = os.fstat
            with mock.patch("ni_rua.cortex.agents.codex.sandbox.os.open", return_value=777), \
                    mock.patch("ni_rua.cortex.agents.codex.sandbox.os.close", side_effect=closed_fds.append), \
                    mock.patch.object(os, "O_PATH", 1, create=True), \
                    mock.patch("ni_rua.cortex.agents.codex.sandbox.os.fstat",
                               side_effect=lambda fd: (_ for _ in ()).throw(PermissionError("fstat denied"))
                               if fd == 777 else real_fstat(fd)):
                with self.assertRaises(PermissionError):
                    box.repair_socket_alias(sock_alias)
            self.assertEqual(closed_fds, [777])
        finally:
            os.close(box.daemon_directory_fd)

        # sandbox_pypi.https_request closes tls/connection on OSError and HTTPException and falls back across IPs.
        tls1, tls2 = mock.MagicMock(), mock.MagicMock()
        raw1, raw2 = mock.MagicMock(), mock.MagicMock()
        conn1, conn2 = mock.MagicMock(), mock.MagicMock()
        conn1.getresponse.side_effect = http.client.BadStatusLine("corrupt")
        ok_resp = mock.MagicMock(status=200)
        conn2.getresponse.return_value = ok_resp
        fake_ctx = mock.MagicMock()
        fake_ctx.wrap_socket.side_effect = [tls1, tls2]
        with mock.patch.object(sandbox_pypi.socket, "getaddrinfo",
                               return_value=[(0, 0, 0, "", ("1.1.1.1", 443)), (0, 0, 0, "", ("8.8.8.8", 443))]), \
                mock.patch.object(sandbox_pypi.socket, "socket", side_effect=[raw1, raw2]), \
                mock.patch.object(sandbox_pypi.ssl, "create_default_context", return_value=fake_ctx), \
                mock.patch.object(sandbox_pypi.http.client, "HTTPSConnection", side_effect=[conn1, conn2]):
            got_conn, got_resp = sandbox_pypi.https_request("pypi.org", "/simple/pkg/", {})
        self.assertIs(got_conn, conn2)
        self.assertIs(got_resp, ok_resp)
        conn1.close.assert_called_once()
        tls1.close.assert_called_once()
        raw1.close.assert_called_once()
        tls2.close.assert_not_called()

    def test_protocol_version_matches_declared_dependencies(self):
        root = REPO_ROOT
        dependencies = tomllib.loads((root / 'pyproject.toml').read_text())['project']['dependencies']
        self.assertIn('openai-codex==' + CODEX_PROTOCOL_VERSION, dependencies)
        pins = (root / 'requirements.lock.txt').read_text().splitlines()
        self.assertIn('openai-codex==' + CODEX_PROTOCOL_VERSION, pins)
        self.assertIn('openai-codex-cli-bin==' + CODEX_PROTOCOL_VERSION, pins)
        self.assertIn('codex_models.json', tomllib.loads((root / 'pyproject.toml').read_text())
                      ['tool']['setuptools']['package-data']['ni_rua.cortex.session'])

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_fixed_preferences_also_apply_when_source_omits_them_or_run_is_resumed(self):
        self.source_config.write_text('model_provider = "fixture"\n'
            '[model_providers.fixture]\nbase_url = "https://fixture.invalid/v1"\n')
        self.session._prepare()
        self.assert_fixed_preferences(self.generated())
        self.session._prepare((self.session.model, "fixture"))
        self.assert_fixed_preferences(self.generated())

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_native_memories_can_be_disabled_in_private_run_without_changing_host(self):
        original = self.source_config.read_bytes()
        self.session.config = self.session.config.with_overrides({'model': {'native_memories': False}})
        self.session._prepare()
        generated = self.generated()
        self.assertFalse(generated['features']['memories'])
        self.assertTrue(generated['features']['multi_agent'])
        self.assertTrue(generated['features']['hooks'])
        self.assertEqual(self.source_config.read_bytes(), original)
        self.session._prepare((self.session.model, "fixture"))
        self.assertFalse(self.generated()['features']['memories'])

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_external_model_and_effort_reach_native_config_and_turn(self):
        for model, effort in [("gpt-6.1-sol", "xhigh"), ("gpt-6-luna", "low")]:
            with self.subTest(model=model, effort=effort):
                session = CodexSession(self.root / model, self.store, lambda _: {},
                    model=model, effort=effort, config=self.session.config)
                self.addCleanup(session.close)
                session._prepare()
                config = tomllib.loads((session.codex_home / "config.toml").read_text())
                self.assertEqual(config["model"], model)
                self.assertEqual(config["model_reasoning_effort"], effort)
                # Native memory side threads must not fall back to a built-in model the provider may not serve.
                self.assertEqual(config["memories"], dict(extract_model=model, consolidation_model=model))
                client = mock.MagicMock()
                session.client = client
                session.thread_id = "fixture-thread"
                session.run_turn("fixture prompt")
                client.turn_start.assert_called_once_with("fixture-thread",
                    [{"type": "text", "text": "fixture prompt"}], {"model": model, "effort": effort})
                session.client = None

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_real_app_server_initializes_thread_with_requested_features(self):
        # Use real runtime/relays, a closed localhost fixture upstream, and no turns.
        # This catches feature startup failures that `features list` cannot detect.
        self.proxy.stop()
        self.source_config.write_text('model_provider = "fixture"\n'
            '[model_providers.fixture]\nbase_url = "http://127.0.0.1:1/v1"\n')
        session = CodexSession(self.root / "startup", self.store, lambda _: {},
            model="gpt-6.1-sol", effort="medium", config=self.session.config)
        def close_runtime():
            bridge = session.client._proc if session.client else None
            session.close()
            if bridge:
                for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                    if stream and not stream.closed:
                        stream.close()
        self.addCleanup(close_runtime)
        self.assertTrue(session.start([], "Offline feature startup validation."))
        self.assertEqual(session.state, "idle")
        self.assertIsNone(session.process.poll())
        response = session.client.request("config/read", {"cwd": str(session.actor_dir)},
                                          response_model=ConfigReadResponse)
        config = response.config.model_dump(mode="json")
        self.assert_fixed_preferences(config)
        self.assertEqual(config["model"], "gpt-6.1-sol")
        self.assertEqual(config["model_reasoning_effort"], "medium")
        response = session.client.request("experimentalFeature/list", {"limit": 200},
                                          response_model=ExperimentalFeatureListResponse)
        available = {feature.name: feature.enabled for feature in response.data}
        for feature in ("hooks", "memories", "multi_agent", "multi_agent_v2"):
            self.assertTrue(available[feature], feature)
        for feature in ('shell_tool', 'unified_exec'):
            self.assertTrue(available[feature], feature)
        self.assertFalse(available['network_proxy'])
        self.assertEqual(session.model_proxy.request_count, 0)
        self.assertEqual(session.model_proxy.attempt_count, 0)
        self.assertFalse((session.private / "isolation.json").exists())
        self.assertEqual(self.store.read_state('runtime', 'codex')['isolation'],
                         'bubblewrap_namespaces_package_egress_v1')


class CodexNativeMemoryProtocolTest(unittest.TestCase):
    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_native_shell_and_memory_consolidation_execute_in_private_fixture(self):
        self._check_native_shell_and_memory(sandbox_enabled=True)

    def test_native_shell_memory_and_terminal_execute_without_sandbox(self):
        with mock.patch('ni_rua.cortex.agents.codex.sandbox.CodexSandbox.prepare', side_effect=AssertionError('sandbox used')):
            self._check_native_shell_and_memory(sandbox_enabled=False)

    def _check_native_shell_and_memory(self, *, sandbox_enabled):
        import shlex
        import time

        requests = []
        memory_output_received = threading.Event()
        actor_marker = 'NI_RUA_NATIVE_SHELL_READ_WRITE_OK'
        memory_marker = 'NI_RUA_NATIVE_MEMORY_READ_WRITE_OK'
        commands = {}

        class ResponsesHandler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_):
                pass

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                memory = payload.get('client_metadata', {}).get('x-openai-subagent') == 'memory_consolidation'
                role = 'memory' if memory else 'actor'
                call_id = 'fixture-' + role + '-exec'
                outputs = [item for item in payload['input'] if
                           item.get('type') == 'function_call_output' and item.get('call_id') == call_id]
                if outputs:
                    item = dict(type='message', id='fixture-' + role + '-finish', role='assistant',
                        status='completed', content=[dict(type='output_text', text='fixture complete', annotations=[])])
                else:
                    item = dict(type='function_call', id='fixture-' + role + '-call', call_id=call_id,
                        name='exec_command', namespace='functions', status='completed',
                        arguments=json.dumps(dict(cmd=commands[role], yield_time_ms=1000, max_output_tokens=1000)))
                events = [
                    ('response.created', {'response': {'id': 'fixture-response-' + str(len(requests)), 'status': 'in_progress', 'output': []}}),
                    ('response.output_item.added', {'output_index': 0, 'item': item}),
                    ('response.output_item.done', {'output_index': 0, 'item': item}),
                    ('response.completed', {'response': {'id': 'fixture-response-' + str(len(requests)), 'status': 'completed',
                        'output': [item], 'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}}}),
                ]
                body = ''.join('event: ' + event + '\ndata: ' + json.dumps({'type': event, **value})
                               + '\n\n' for event, value in events).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.send_header('Connection', 'close')
                self.end_headers()
                self.wfile.write(body)
                if memory and outputs:
                    memory_output_received.set()

        server = ThreadingHTTPServer(('127.0.0.1', 0), ResponsesHandler)
        threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory(prefix='ni-rua-native-memory-') as directory:
            root = Path(directory)
            source = root / 'source'
            source.mkdir()
            (source / 'config.toml').write_text('model_provider="fixture"\n'
                '[model_providers.fixture]\nname="Loopback fixture"\n'
                'base_url="http://127.0.0.1:' + str(server.server_port) + '/v1"\n'
                'wire_api="responses"\nrequires_openai_auth=false\nsupports_websockets=false\n')
            store = Store(root / 'run', dict(run_id='native-memory', episode_id='episode',
                env_instance_id='synthetic', env_generation=1))
            with mock.patch('ni_rua.core.config.LOCAL_CONFIG_PATH', root / 'absent.local.json'):
                config = Config.load(environ={}, overrides={'model': {'codex_home': str(source)},
                    'features': {'sandbox_enabled': sandbox_enabled}})
            self.assertTrue(config.model.native_memories)
            session = CodexSession(store.root, store, lambda _: {}, config=config)
            memories = session.codex_home / 'memories'
            actor_file = session.actor_dir / 'synthetic-shell.txt'
            commands['actor'] = 'python3 -c ' + shlex.quote(
                f"from pathlib import Path; p=Path({str(actor_file)!r}); "
                f"p.write_text({actor_marker!r}); print(p.read_text())")
            commands['memory'] = 'python3 -c ' + shlex.quote(
                f"from pathlib import Path; p=Path({str(memories)!r}); "
                "print((p/'raw_memories.md').read_text()); "
                "print((p/'phase2_workspace_diff.md').read_text()); "
                "(p/'MEMORY.md').write_text('# Memory\\n\\nNo reusable memories yet.\\n'); "
                "(p/'memory_summary.md').write_text('v1\\n\\n## General Tips\\n\\nNo reusable memories yet.\\n'); "
                f"print({memory_marker!r})")
            jobs = []
            terminal_status = None
            try:
                session.start([], 'Loopback diagnostic for a synthetic task.')
                self.assertEqual(session.sandbox is not None, sandbox_enabled)
                self.assertEqual(store.read_state('runtime', 'codex')['isolation'], session.isolation)
                deadline = time.monotonic() + 25
                session.run_turn('Use exec_command to write and read the synthetic fixture, then finish.')
                self.assertTrue(session.turn_done.wait(max(0, deadline - time.monotonic())))
                self.assertEqual(session.state, 'completed')
                self.assertTrue(memory_output_received.wait(max(0, deadline - time.monotonic())),
                                'Native phase2 did not return an exec_command result')
                database = session.codex_home / 'memories_1.sqlite'
                while time.monotonic() < deadline:
                    if database.exists():
                        with sqlite3.connect(database.as_uri() + '?mode=ro', uri=True) as db:
                            jobs = db.execute('SELECT kind,status,last_error FROM jobs').fetchall()
                    if any(kind == 'memory_consolidate_global' and status in ('done', 'error')
                           for kind, status, _ in jobs):
                        break
                    time.sleep(.05)
                self.assertIn(('memory_consolidate_global', 'done', None), jobs)
                for role, marker in [('actor', actor_marker), ('memory', memory_marker)]:
                    replies = [item for request in requests for item in request['input'] if
                        item.get('type') == 'function_call_output' and item.get('call_id') == 'fixture-' + role + '-exec']
                    self.assertTrue(replies, role)
                    self.assertIn(marker, json.dumps(replies))
                    self.assertIn('Process exited with code 0', json.dumps(replies))
                self.assertEqual(actor_file.read_text(), actor_marker)
                self.assertTrue((memories / 'MEMORY.md').read_text().strip())
                self.assertTrue((memories / 'memory_summary.md').read_text().startswith('v1\n'))
                memory = [request for request in requests if
                    request.get('client_metadata', {}).get('x-openai-subagent') == 'memory_consolidation']
                actor = [request for request in requests if
                    request.get('client_metadata', {}).get('session_id') == session.thread_id]
                self.assertEqual(len(actor), 2)
                self.assertEqual(len(memory), 2)
                self.assertTrue(all(request['client_metadata']['session_id'] != session.thread_id for request in memory))
                self.assertFalse((source / 'auth.json').exists())
                # The real read-only remote TUI shares this completed native
                # thread through a second namespace without another model turn.
                import asyncio
                from ni_rua.cortex.session.terminal import Terminal

                async def attach_terminal():
                    terminal = Terminal(session.root, session.actor_dir,
                        command_prefix=session.command_prefix + [str(session.cli_executable),
                            '-c', 'model=' + json.dumps(session.model),
                            '-c', 'model_reasoning_effort=' + json.dumps(session.effort)], env=session.env)
                    count = len(requests)
                    try:
                        await terminal.start('unix://' + str(session.socket_path), session.thread_id)
                        deadline = time.monotonic() + 10
                        while time.monotonic() < deadline:
                            data, _ = terminal.read()
                            if b'fixture complete' in data or terminal.process.poll() is not None:
                                break
                            await asyncio.sleep(.05)
                        status = terminal.status()
                        self.assertFalse(status['bootstrap_failed'], data.decode(errors='replace'))
                        self.assertIsNone(status['exit_code'], data.decode(errors='replace'))
                        self.assertIn(b'fixture complete', data)
                        self.assertEqual(len(requests), count)
                        return status
                    finally:
                        await terminal.close()

                terminal_status = asyncio.run(attach_terminal())
                if not sandbox_enabled:
                    thread_id = session.thread_id
                    bridge = session.client._proc
                    session.close()
                    for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                        if stream and not stream.closed:
                            stream.close()
                    self.assertEqual(session.start([], 'Loopback diagnostic for a synthetic task.',
                                                   resume_thread_id=thread_id), thread_id)
                    self.assertEqual(session.isolation, 'disabled')
            finally:
                if evidence := os.environ.get('NI_RUA_CODEX_MEMORY_TEST_EVIDENCE'):
                    path = Path(evidence)
                    path.mkdir(parents=True, exist_ok=True)
                    (path / 'requests.json').write_text(json.dumps(requests, indent=2) + '\n')
                    (path / 'summary.json').write_text(json.dumps(dict(
                        runtime_version=CODEX_PROTOCOL_VERSION, actor_thread_id=session.thread_id,
                        requests=[dict(model=request['model'], reasoning=request.get('reasoning'),
                            role=request.get('client_metadata', {}).get('x-openai-subagent', 'actor'),
                            thread_id=request.get('client_metadata', {}).get('session_id')) for request in requests],
                        jobs=jobs, native_memories=True, terminal_status=terminal_status,
                        external_model_calls=0, host_credentials_used=False,
                        files={name: (memories / name).read_text() for name in ('MEMORY.md', 'memory_summary.md')
                               if (memories / name).exists()}), indent=2) + '\n')
                    if (session.private / 'app-server.log').exists():
                        (path / 'app-server.log').write_text((session.private / 'app-server.log').read_text())
                    if (session.root / 'terminal.bin').exists():
                        (path / 'terminal.bin').write_bytes((session.root / 'terminal.bin').read_bytes())
                bridge = session.client._proc if session.client else None
                session.close()
                if bridge:
                    for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                        if stream and not stream.closed:
                            stream.close()
                store.close()

# ==== tests/test_codex_profile_upstream.py ====

PROFILE_KEY = "sk-c2-PROFILE-0123456789abcdefKEY"
HOME_KEY = "sk-c2-CODEXHOME-9876543210zyxwvuKEY"
RELAY_URL = "http://127.0.0.1:54321"
PLACEHOLDER = {"OPENAI_API_KEY": "ni-rua-local-relay"}

# config.toml generated at 0d4acf7 (before WP-C2) for the Codex home of ``write_codex_home`` with
# model gpt-6.1-sol / effort high / sandbox off and the relay at RELAY_URL; <RUN> = the run directory.
# One intended change since: ``project_root_markers = []`` (information boundary, no host AGENTS.md; see
# ``CodexHostRepositoryBoundary`` in tests/test_agent_prompt_parity.py).
CODEX_HOME_GOLDEN = """\
model = "gpt-6.1-sol"
model_provider = "ni_rua_model_proxy"
model_catalog_json = "<RUN>/private/codex-home/models.json"
model_reasoning_effort = "high"
approval_policy = "never"
sandbox_mode = "danger-full-access"
web_search = "disabled"
model_verbosity = "high"
service_tier = "default"
suppress_unstable_features_warning = true
project_root_markers = []
tool_output_token_limit = 10000
[features]
code_mode_host = false
shell_tool = true
unified_exec = true
browser_use = false
browser_use_external = false
computer_use = false
apps = false
plugins = false
artifact = false
image_generation = false
goals = true
hooks = true
memories = true
network_proxy = false
mentions_v2 = true
multi_agent = true
prevent_idle_sleep = true
guardian_approval = true
multi_agent_v2 = true
fast_mode = false
context_management.experimental_mode = true
[features.code_mode]
enabled = false
[memories]
extract_model = "gpt-6.1-sol"
consolidation_model = "gpt-6.1-sol"
[model_providers.ni_rua_model_proxy]
name = "fixture"
requires_openai_auth = true
base_url = "http://127.0.0.1:54321"
wire_api = "responses"
supports_websockets = false
request_max_retries = 0
stream_max_retries = 0
stream_idle_timeout_ms = 9223372036854775807
"""


def write_codex_home(directory):
    """A Codex home in the user's provider shape: custom Responses provider + API key in auth.json."""
    home = Path(directory) / "codex-home-source"
    home.mkdir()
    (home / "config.toml").write_text('model_provider = "fixture"\n\n[model_providers.fixture]\nname = "Fixture"\n'
                                      'base_url = "https://fixture.invalid/v1"\nwire_api = "responses"\n'
                                      'requires_openai_auth = true\n')
    (home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": HOME_KEY}))
    return home


def write_profile(directory, profile_id, *, base_url="https://profile.invalid/v1", auth_scheme="bearer",
                  key=PROFILE_KEY, wires=None):
    document = {"schema": "ni_rua.provider_profile.v1", "name": "C2 Fixture Gateway",
                "wires": wires or {"openai_responses": {"base_url": base_url, "auth_scheme": auth_scheme}}}
    if key is not None:
        document["credential"] = {"api_key": key}
    path = Path(directory) / f"{profile_id}.local.json"
    path.write_text(json.dumps(document))
    path.chmod(0o600)
    return path


def files_containing(root, needle):
    """Every file under ``root`` whose bytes contain ``needle``."""
    data = needle.encode()
    return [str(path) for path in Path(root).rglob("*") if path.is_file() and data in path.read_bytes()]


class _SessionFixture(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()
        self.home = write_codex_home(self.root)
        self.profiles = self.root / "profiles"
        self.profiles.mkdir(mode=0o700)
        self.relay = mock.MagicMock(base_url=RELAY_URL)
        self.relay.start.return_value = self.relay
        for target, value in (("ModelRequestProxy", dict(return_value=self.relay)),
                              ("snapshot_actor_source", dict(return_value=self.root / "actor-source"))):
            patcher = mock.patch("ni_rua.cortex.agents.codex.session." + target, **value)
            setattr(self, target, patcher.start())
            self.addCleanup(patcher.stop)

    def base_config(self, codex_home=None, **model):
        return Config.from_dict({"model": {"name": "gpt-6.1-sol", "effort": "high",
                                           "codex_home": str(codex_home or self.home), **model},
                                 "features": {"sandbox_enabled": False}})

    def session(self, config, name="run"):
        store = Store(self.root / name, dict(run_id=name, episode_id="e", env_instance_id="i", env_generation=1))
        self.addCleanup(store.close)
        session = CodexSession(store.root, store, lambda _: {}, config=config)
        self.addCleanup(session.close)
        return session

    def generated(self, session):
        return (session.codex_home / "config.toml").read_text().replace(str(session.root), "<RUN>")


class CodexHomeUnchangedTest(_SessionFixture):
    def test_codex_home_start_is_byte_identical_to_the_golden(self):
        config = self.base_config()
        self.assertIsNone(factory.actor_upstream(config, "codex"))
        session = self.session(config)
        self.assertEqual(session._prepare(), "fixture")
        self.assertEqual(self.generated(session), CODEX_HOME_GOLDEN)
        self.assertEqual(json.loads((session.codex_home / "auth.json").read_text()), PLACEHOLDER)
        self.assertEqual(self.ModelRequestProxy.call_args.args, ("https://fixture.invalid/v1",))
        self.assertEqual(self.ModelRequestProxy.call_args.kwargs["upstream_auth_headers"],
                         {"Authorization": "Bearer " + HOME_KEY})
        self.assertIs(self.ModelRequestProxy.call_args.kwargs["classify_call_purpose"], True)
        self.assertIsNone(session.upstream_profile)
        self.assertEqual(session.provenance()["context_delivery"], "delta")
        self.assertNotIn("upstream_profile", session.provenance())
        self.assertNotIn("codex_upstream", config.to_dict().get("agent", {}))

    def test_budgeted_action_reply_states_the_catalogue_limit_and_fits_replies_to_the_same_number(self):
        from ni_rua.cortex.agents.base import AgentSession
        from ni_rua.cortex.agents.codex.session import codex_tool_output_budget_bytes, codex_tool_output_token_limit

        self.assertIs(self.base_config().features.budgeted_action_reply, True)  # the default
        session = self.session(self.base_config())
        self.assertEqual(session._prepare(), "fixture")
        limit = codex_tool_output_token_limit("gpt-6.1-sol")
        written = json.loads((session.codex_home / "models.json").read_text())
        self.assertEqual(next(model["truncation_policy"] for model in written["models"]
                              if model["slug"] == "gpt-6.1-sol"), {"mode": "tokens", "limit": limit})
        # The limit the Harness fits replies to is stated in the native configuration, before the first table ...
        line = f"tool_output_token_limit = {limit}\n"
        self.assertIn(line + "[features]\n", CODEX_HOME_GOLDEN)
        self.assertEqual(self.generated(session), CODEX_HOME_GOLDEN)
        self.assertEqual(session.native_config["tool_output_token_limit"], limit)
        # ... and that line is the only difference from a run with the switch off.
        off = self.session(self.base_config().with_overrides({"features": {"budgeted_action_reply": False}}), "off")
        self.assertEqual(off._prepare(), "fixture")
        self.assertEqual(self.generated(off), CODEX_HOME_GOLDEN.replace(line, "", 1))
        self.assertNotIn("tool_output_token_limit", off.native_config)
        # 1.2 x the limit, one token per started 4 UTF-8 bytes.
        self.assertEqual(session.tool_output_text_budget_bytes(), limit * 12 // 10 * 4)
        self.assertEqual(session.tool_output_text_budget_bytes(), codex_tool_output_budget_bytes("gpt-6.1-sol"))
        # 验证 gpt-6.2-sol 与 gpt-6.2-astra 目录命中、上下文窗口与 10000 token / 48000 B 截断预算契约
        by_slug = {model["slug"]: model for model in written["models"]}
        for slug, expected_ctx in (("gpt-6.2-sol", 500000), ("gpt-6.2-astra", 272000)):
            self.assertIn(slug, by_slug)
            self.assertEqual(by_slug[slug]["context_window"], expected_ctx)
            self.assertEqual(by_slug[slug]["max_context_window"], 872000)
            self.assertEqual(by_slug[slug]["truncation_policy"], {"mode": "tokens", "limit": 10000})
            self.assertEqual(codex_tool_output_token_limit(slug), 10000)
            self.assertEqual(codex_tool_output_budget_bytes(slug), 48000)
        sol62 = self.session(self.base_config(name="gpt-6.2-sol"), "sol62")
        self.assertEqual(sol62._prepare(), "fixture")
        self.assertEqual(sol62.native_config["tool_output_token_limit"], 10000)
        self.assertEqual(sol62.tool_output_text_budget_bytes(), 48000)
        self.assertIsNone(codex_tool_output_budget_bytes("model-not-in-the-catalogue"))
        self.assertIsNone(AgentSession.tool_output_text_budget_bytes(session))


class CodexProfileUpstreamTest(_SessionFixture):
    def profile_config(self, path, **model):
        # model.codex_home names a directory that does not exist: the profile path must never read it.
        return apply_profile(self.base_config(codex_home=self.root / "absent-codex-home", **model), "codex", path)

    def test_profile_feeds_the_relay_and_the_private_home_differs_only_in_the_provider_label(self):
        path = write_profile(self.profiles, "c2-gateway")
        config = self.profile_config(path)
        upstream = factory.actor_upstream(config, "codex")
        self.assertEqual((upstream.source, upstream.wire_api, upstream.provider, upstream.base_url),
                         ("profile", "openai_responses", "c2-gateway", "https://profile.invalid/v1"))
        self.assertIsNone(factory.backend_requirements(config))
        session = self.session(config)
        self.assertEqual(session._prepare(), "c2-gateway")
        self.assertEqual(self.generated(session),
                         CODEX_HOME_GOLDEN.replace('name = "fixture"', 'name = "c2-gateway"'))
        self.assertEqual(json.loads((session.codex_home / "auth.json").read_text()), PLACEHOLDER)
        self.assertEqual(self.ModelRequestProxy.call_args.args, ("https://profile.invalid/v1",))
        self.assertEqual(self.ModelRequestProxy.call_args.kwargs["upstream_auth_headers"],
                         {"Authorization": "Bearer " + PROFILE_KEY})
        self.assertIsNone(session.account_control)
        self.assertEqual(session.upstream_profile,
                         {"id": "c2-gateway", "name": "C2 Fixture Gateway", "wire_api": "openai_responses"})
        provenance = session.provenance()
        self.assertEqual(provenance["upstream_profile"], session.upstream_profile)
        self.assertEqual((provenance["wire_api"], provenance["kind"]), ("openai_responses", "codex"))
        for text in (json.dumps(provenance), json.dumps(config.to_dict()), repr(session.native_config)):
            self.assertNotIn(PROFILE_KEY, text)
        self.assertEqual(files_containing(session.root, PROFILE_KEY), [])
        self.assertFalse((self.root / "absent-codex-home").exists())

    def test_provider_label_drives_resume_identity(self):
        config = self.profile_config(write_profile(self.profiles, "c2-gateway"))
        session = self.session(config)
        with self.assertRaisesRegex(ValueError, "^resume_usage_identity_conflict"):
            session._prepare(expected_identity=("gpt-6.1-sol", "fixture"))
        self.assertEqual(session._prepare(expected_identity=("gpt-6.1-sol", "c2-gateway")), "c2-gateway")

    def test_refusals_name_the_profile_never_the_key_and_start_no_relay(self):
        good = write_profile(self.profiles, "good")
        nokey = write_profile(self.profiles, "nokey", key=None)
        apikey = write_profile(self.profiles, "apikey", auth_scheme="x-api-key")
        messages = write_profile(self.profiles, "messages", wires={
            "anthropic_messages": {"base_url": "https://profile.invalid", "auth_scheme": "x-api-key"}})
        base = self.base_config(codex_home=self.root / "absent-codex-home")

        def reference(path):
            # A reference not expanded by Config.load / apply_profile (wire, base URL and label unset).
            return base.with_overrides({"agent": {"codex_upstream": {"source": "profile", "settings": str(path)}}})
        cases = [("nokey", self.profile_config(nokey), "^profile_missing_key:" + re.escape(str(nokey))),
                 ("apikey", self.profile_config(apikey), "^profile_auth_scheme_unsupported:" + re.escape(str(apikey))),
                 ("stale", reference(good), "^profile_upstream_stale:" + re.escape(str(good))),
                 ("messages", reference(messages), "^profile_no_common_wire:" + re.escape(str(messages)))]
        for name, config, pattern in cases:
            with self.subTest(name):
                session = self.session(config, name)
                with self.assertRaisesRegex(ValueError, pattern) as raised:
                    session._prepare()
                self.assertNotIn(PROFILE_KEY, str(raised.exception))
                self.assertFalse((session.codex_home / "config.toml").exists())
                self.assertEqual(files_containing(session.root, PROFILE_KEY), [])
        self.ModelRequestProxy.assert_not_called()


class NativeCodexThroughProfileTest(unittest.TestCase):
    """The pinned Codex app-server (no sandbox) runs one turn through the relay to a fake Responses server."""

    def test_turn_reaches_the_fake_responses_server_with_the_profile_bearer(self):
        server = FakeResponsesServer([text_step("profile route ok")]).start()
        self.addCleanup(server.close)
        temporary = tempfile.TemporaryDirectory(prefix="ni-rua-codex-profile-")
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name).resolve()
        path = write_profile(root, "c2-loopback", base_url=server.base_url + "/v1")
        config = apply_profile(Config.from_dict({
            "model": {"name": "gpt-6.1-sol", "effort": "low", "codex_home": str(root / "absent-codex-home"),
                      "native_memories": False},
            "features": {"sandbox_enabled": False}}), "codex", path)
        store = Store(root / "run", dict(run_id="codex-profile", episode_id="e", env_instance_id="i",
                                         env_generation=1))
        self.addCleanup(store.close)
        session = CodexSession(store.root, store, lambda _: {}, config=config)

        def close_runtime():
            bridge = session.client._proc if session.client else None
            session.close()
            if bridge:
                for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                    if stream and not stream.closed:
                        stream.close()
        self.addCleanup(close_runtime)
        with mock.patch("ni_rua.cortex.agents.codex.sandbox.CodexSandbox.prepare",
                        side_effect=AssertionError("sandbox used")):
            session.start([], "Loopback check of the provider-profile route.")
        self.assertEqual(session.provider, "c2-loopback")
        session.run_turn("Reply with: profile route ok")
        self.assertTrue(session.turn_done.wait(60), "the native turn did not end")
        self.assertEqual(session.state, "completed", session.turn_result)
        generation = server.generation_requests()
        self.assertTrue(generation)
        for request in generation:
            self.assertEqual(request.header("Authorization"), "Bearer " + PROFILE_KEY)
            # Codex's own client identity reaches the provider (some gateways refuse library user agents).
            self.assertFalse((request.header("User-Agent") or "").lower().startswith("python-urllib"))
        self.assertEqual(server.last_json()["model"], "gpt-6.1-sol")
        # The relay's own counter is synchronous; its Store record (served models included) is written by its
        # callback thread and can trail the finished turn, so wait for it to settle first.
        self.assertEqual(session.model_proxy.request_count, len(generation))
        self.assertTrue(session.model_proxy.settle(), "the relay's records did not settle")
        requests = store.read_state("runtime", "model_requests")
        self.assertEqual((requests["completed_calls"], requests["transport_blocked"]), (len(generation), False))
        self.assertEqual(requests["served_models"], {"gpt-6.1-sol": len(generation)})
        self.assertEqual((store.read_state("runtime", "codex")["provider"],
                          store.read_state("runtime", "agent")["provider"]), ("c2-loopback", "c2-loopback"))
        self.assertEqual(json.loads((session.codex_home / "auth.json").read_text()), PLACEHOLDER)
        self.assertEqual(tomllib.loads((session.codex_home / "config.toml").read_text())
                         ["model_providers"]["ni_rua_model_proxy"]["name"], "c2-loopback")
        # Operator service-tier probes and select_model use the same relay upstream and credential.
        probe = session.model_proxy.probe_service_tier("gpt-6.1-sol", "priority", timeout_s=30)
        self.assertIs(probe["supported"], True, probe)
        self.assertEqual(server.requests[-1].header("Authorization"), "Bearer " + PROFILE_KEY)
        self.assertEqual(session.model_proxy.select_model("gpt-6.1-sol")["priority"], probe)
        self.assertEqual(files_containing(store.root, PROFILE_KEY), [])

# ==== tests/test_codex_provider.py ====

class CodexProviderTest(unittest.TestCase):
    def setUp(self):
        self.temporary = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(self.temporary)
        self.source = self.root / "source"
        self.source.mkdir()
        config = self.root / "ni-rua.json"
        config.write_text(json.dumps({"version": 1, "features": {"sandbox_enabled": True},
                                      "model": {"codex_home": str(self.source)}}))
        self.env = mock.patch.dict(os.environ, {"CODEX_HOME": str(self.source),
            "NI_RUA_CONFIG": str(config),
            "PATH": str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")})
        self.env.start()
        self.addCleanup(self.env.stop)
        self.base_env = mock.patch.dict(os.environ)
        self.base_env.start()
        self.addCleanup(self.base_env.stop)
        os.environ.pop("OPENAI_BASE_URL", None)

    def auth(self, value):
        (self.source / "auth.json").write_text(json.dumps(value))

    def test_builtin_auth_selects_chatgpt_or_api_upstream(self):
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        name, provider, url = _model_provider({}, self.source)
        self.assertEqual(name, "openai")
        self.assertTrue(provider["requires_openai_auth"])
        self.assertEqual(url, "https://chatgpt.com/backend-api/codex")
        self.auth({"OPENAI_API_KEY": "synthetic"})
        self.assertEqual(_model_provider({}, self.source)[2], "https://api.openai.com/v1")
        self.auth({"tokens": {"access_token": "synthetic"}})
        self.assertEqual(_model_provider({}, self.source)[2], "https://chatgpt.com/backend-api/codex")

    def test_explicit_routes_and_custom_provider_auth_are_preserved(self):
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        config = {"openai_base_url": "https://example.invalid/v1"}
        self.assertEqual(_model_provider(config, self.source)[2], config["openai_base_url"])
        with mock.patch.dict(os.environ, {"OPENAI_BASE_URL": "https://env.invalid/v1"}):
            self.assertEqual(_model_provider({}, self.source)[2], "https://env.invalid/v1")
            self.assertEqual(_model_provider(config, self.source)[2], config["openai_base_url"])
        config = {"model_provider": "fixture", "model_providers": {"fixture": {
            "base_url": "https://fixture.invalid/v1", "env_key": "FIXTURE_KEY"}}}
        name, provider, url = _model_provider(config, self.source)
        self.assertEqual((name, url), ("fixture", "https://fixture.invalid/v1"))
        self.assertEqual(provider["env_key"], "FIXTURE_KEY")
        self.assertNotIn("requires_openai_auth", provider)

    def test_relay_auth_is_resolved_without_exposing_secrets_to_native_config(self):
        self.auth({"OPENAI_API_KEY": "host-only-key"})
        self.assertEqual(_upstream_auth({"requires_openai_auth": True}, self.source),
                         {"Authorization": "Bearer host-only-key"})
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "host-token", "account_id": "host-account"}})
        self.assertEqual(_upstream_auth({"requires_openai_auth": True}, self.source),
                         {"Authorization": "Bearer host-token", "ChatGPT-Account-Id": "host-account"})
        with mock.patch.dict(os.environ, {"FIXTURE_KEY": "host-env-secret"}):
            self.assertEqual(_upstream_auth({"env_key": "FIXTURE_KEY"}, self.source),
                             {"Authorization": "Bearer host-env-secret"})

    def test_resume_rejects_actor_symlink_before_host_file_writes(self):
        (self.source / "config.toml").write_text('model_provider="fixture"\n[model_providers.fixture]\nbase_url="http://127.0.0.1:1/v1"\n')
        store = Store(self.root / "symlink-run", dict(run_id="symlink", episode_id="e", env_instance_id="i", env_generation=1))
        session = CodexSession(store.root, store, lambda _: {})
        try:
            session.codex_home.mkdir(parents=True)
            target = self.root / "host-file"
            target.write_text("host-only-sentinel")
            (session.codex_home / "models.json").symlink_to(target)
            with self.assertRaisesRegex(ValueError, "sandbox_home_symlink"):
                session._prepare()
            self.assertEqual(target.read_text(), "host-only-sentinel")
        finally:
            session.close()
            store.close()

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_verified_sandbox_home_can_resume_with_native_generated_skills(self):
        (self.source / "config.toml").write_text('model_provider="fixture"\n[model_providers.fixture]\nbase_url="http://127.0.0.1:1/v1"\n')
        store = Store(self.root / "resume-run", dict(run_id="resume", episode_id="e", env_instance_id="i", env_generation=1))
        session = CodexSession(store.root, store, lambda _: {})
        try:
            skills = session.codex_home / "skills" / ".system"
            skills.mkdir(parents=True)
            with self.assertRaisesRegex(ValueError, "sandbox_home_contains_host_extensions"):
                session._prepare()
            store.set_state("runtime", "codex", {"isolation": ACTOR_ISOLATION})
            session._prepare()
            self.assertEqual(json.loads((session.codex_home / "auth.json").read_text()),
                             {"OPENAI_API_KEY": "ni-rua-local-relay"})
        finally:
            session.close()
            store.close()

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_image_reuse_uses_frozen_settings_not_changed_environment(self):
        (self.source / "config.toml").write_text('model = "gpt-6-astra"\n')
        self.auth({"OPENAI_API_KEY": "synthetic"})
        store = Store(self.root / "run", dict(run_id="synthetic-image-policy", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        with mock.patch.dict(os.environ, {"NI_RUA_IMAGE_SETTINGS": '{"reuse_exact_images":false}'}):
            session = CodexSession(self.root / "run", store, lambda value: {})
        relay = mock.MagicMock(base_url="http://127.0.0.1:54321")
        relay.start.return_value = relay
        try:
            with mock.patch.dict(os.environ, {"NI_RUA_IMAGE_SETTINGS": '{"reuse_exact_images":true}'}), \
                 mock.patch("ni_rua.cortex.agents.codex.session.ModelRequestProxy", return_value=relay) as factory:
                session._prepare()
                self.assertIsNone(session.account_control, "API key mode does not need ChatGPT account discovery")
                self.assertFalse(factory.call_args.kwargs["image_reuse"])
                self.assertTrue(factory.call_args.kwargs["image_trim"])
                self.assertEqual(factory.call_args.kwargs["request_projection"], session._present_model_request)
                self.assertEqual(factory.call_args.kwargs["request_context"], session._task_decision_context)
                self.assertFalse(store.read_state("runtime", "model_image_policy")["enabled"])
        finally:
            session.close()
            store.close()

    def test_custom_chatgpt_backend_is_not_silently_rerouted(self):
        (self.source / "config.toml").write_text('chatgpt_base_url = "https://region.example.invalid/backend-api"\n')
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        store = Store(self.root / "run", dict(run_id="synthetic-regional-provider", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        session = CodexSession(store.root, store, lambda value: {})
        try:
            with mock.patch("ni_rua.cortex.agents.codex.session.ModelRequestProxy") as model, \
                 mock.patch("ni_rua.cortex.agents.codex.session.AccountControlRelay") as account:
                with self.assertRaisesRegex(ValueError, "account_control_requires_default_chatgpt_backend"):
                    session._prepare()
                model.assert_not_called()
                account.assert_not_called()
        finally:
            session.close()
            store.close()

    def test_account_relay_startup_failure_closes_the_started_model_proxy(self):
        (self.source / "config.toml").write_text('model_provider = "openai"\n')
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        store = Store(self.root / "run", dict(run_id="synthetic-relay-startup", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        session = CodexSession(store.root, store, lambda value: {})
        model, account = mock.MagicMock(), mock.MagicMock()
        model.start.return_value = model
        account.start.side_effect = RuntimeError("fixture-certificate-failure")
        try:
            with mock.patch("ni_rua.cortex.agents.codex.session.ModelRequestProxy", return_value=model), \
                 mock.patch("ni_rua.cortex.agents.codex.session.AccountControlRelay", return_value=account):
                with self.assertRaisesRegex(RuntimeError, "fixture-certificate-failure"):
                    session._prepare()
            model.close.assert_called_once()
            account.close.assert_called_once()
            self.assertIsNone(session.model_proxy)
            self.assertIsNone(session.account_control)
        finally:
            session.close()
            store.close()

    @require(bubblewrap_namespaces, "starts the native Codex actor inside its bubblewrap namespace sandbox")
    def test_generated_native_config_uses_private_id_but_preserves_source_identity(self):
        (self.source / "config.toml").write_text('model = "gpt-6-astra"\n')
        self.auth({"auth_mode": "chatgpt", "tokens": {"access_token": "synthetic"}})
        store = Store(self.root / "run", dict(run_id="synthetic-provider", episode_id="episode",
                      env_instance_id="environment", env_generation=1))
        session = CodexSession(self.root / "run", store, lambda value: {})
        relay = mock.MagicMock(base_url="http://127.0.0.1:54321")
        relay.start.return_value = relay
        account = mock.MagicMock(base_url="https://127.0.0.1:54322/backend-api",
                                 ca_path=session.codex_home / "account-control-ca.pem")
        account.server.server_port = 54322
        account.start.return_value = account
        wrapper = self.root / "codex"
        wrapper.write_text("#!/bin/sh\nexit 99\n")
        wrapper.chmod(0o755)
        try:
            with mock.patch("ni_rua.cortex.agents.codex.session.ModelRequestProxy", return_value=relay) as factory, \
                 mock.patch("ni_rua.cortex.agents.codex.session.AccountControlRelay", return_value=account), \
                 mock.patch.dict(os.environ, {"PATH": str(self.root)}):
                self.assertEqual(session._prepare(), "openai")
                self.assertEqual(factory.call_args.args[0], "https://chatgpt.com/backend-api/codex")
                self.assertTrue(factory.call_args.kwargs["image_reuse"])
                self.assertTrue(store.read_state("runtime", "image_settings")["reuse_exact_images"])
                policy = store.read_state("runtime", "model_image_policy")
                self.assertEqual(policy, dict(enabled=True, policy="exact_image_part_current_request_v1",
                                              compaction="unchanged", oversized_trim=True, max_inline_images=60))
                self.assertEqual(factory.call_args.kwargs["max_inline_images"], 60)
                self.assertEqual(factory.call_args.kwargs["transport_recovery"]["max_rounds"], 10)
                self.assertEqual(factory.call_args.kwargs["recovery_gate"](), "recovery_gate_unavailable")
            native = bundled_codex_path().resolve()
            self.assertEqual(session.cli_executable, native)
            self.assertEqual(session.command[-5:], [str(native), "--no-daemon", "app-server", "--listen",
                                                    f"unix://{session.socket_path}"])
            self.assertIn("--unshare-net", session.command)
            self.assertNotIn(str(wrapper), session.command)
            config = tomllib.loads((session.codex_home / "config.toml").read_text())
            self.assertEqual(config["model_provider"], MODEL_PROXY_PROVIDER)
            self.assertNotIn("openai", config["model_providers"])
            self.assertTrue(config["model_providers"][MODEL_PROXY_PROVIDER]["requires_openai_auth"])
            provider = config["model_providers"][MODEL_PROXY_PROVIDER]
            self.assertEqual(provider["request_max_retries"], 0)
            self.assertEqual(provider["stream_max_retries"], 0)
            self.assertEqual(provider["stream_idle_timeout_ms"], 2**63-1)
            self.assertEqual(config["chatgpt_base_url"], account.base_url)
            self.assertFalse((session.private / "isolation.json").exists())
            self.assertEqual(session.env["CODEX_CA_CERTIFICATE"], str(account.ca_path))
            self.assertEqual(session.env["CODEX_HOME"], str(session.codex_home))
            self.assertEqual(config["sandbox_mode"], "danger-full-access")
            self.assertEqual(config["web_search"], "disabled")
            self.assertFalse(config["features"]["network_proxy"])
            self.assertNotIn("synthetic", (session.codex_home / "auth.json").read_text())
            for feature in ("shell_tool", "unified_exec"):
                self.assertTrue(config["features"][feature], feature)
            with self.assertRaisesRegex(ValueError, "resume_usage_identity_conflict"):
                session._prepare((session.model, "different-provider"))
            if native.is_file():
                # features list parses the real pinned native config, without inference.
                result = subprocess.run([native, "features", "list"], env=session.env,
                    cwd=session.actor_dir, capture_output=True, text=True, timeout=10)
                self.assertEqual(result.returncode, 0, result.stderr)
            store.set_state("runtime", "codex", {"provider": "openai", "tools_digest": digest([]),
                                                  "isolation": ACTOR_ISOLATION})
            with mock.patch.object(session, "_stop_previous_owned_processes"), \
                 mock.patch.object(session, "_prepare", side_effect=RuntimeError("identity-checked")) as prepare:
                with self.assertRaisesRegex(RuntimeError, "identity-checked"):
                    session.start([], "synthetic", resume_thread_id="synthetic-thread")
                prepare.assert_called_once_with((session.model, "openai"))
        finally:
            session.close()
            store.close()

if __name__ == "__main__":
    unittest.main()
