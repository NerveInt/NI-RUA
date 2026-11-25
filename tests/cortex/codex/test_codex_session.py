"""Merged cortex tests:
- tests/test_codex.py: No-model Unix WebSocket/stdio regression, using the active Python environment.
- tests/test_codex_deferred_tools.py: Real CodexSession -> private relay -> loopback, including native discovery/resume.
- tests/test_codex_image_transport.py: Real isolated app-server -> loopback Responses, with no provider/robot calls.
- tests/test_codex_parallel_dispatch.py: Codex transport takes tool concurrency from the registry contract wired by Workbench (offline; no model/GPU).
"""

NI_RUA_TEST_TIER = "component"

import asyncio
import json
from pathlib import Path
import sys
import tempfile
import unittest
from aiohttp import web, WSMsgType
from ni_rua.cortex.agents.codex.session import CodexSession
from ni_rua.core.contracts import canonical
from ni_rua.core.storage.store import Store
from copy import deepcopy
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import threading
from unittest import mock
from ni_rua.cortex.agents.codex.session import CodexSession, DYNAMIC_TOOL_NAMESPACE
from ni_rua.core.config import Config
from ni_rua.runtime.harness import Harness
from ni_rua.core.tool_output import ToolOutput
from ni_rua.runtime.tools.registry import ToolRegistry, ToolSpec
from support.fixture_robot import FixtureRobot
import base64
import hashlib
import io
import os
import time
from PIL import Image
from codex_cli_bin import bundled_codex_path
from openai_codex.client import CodexConfig
from openai_codex.generated.v2_all import ConfigReadResponse
from ni_rua.core.presentation.actor_output import ActorOutput
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.cortex.agents.codex.session import RecordedClient, write_direct_model_catalog
from ni_rua.runtime.tools.definitions import (DEPTH_REQUEST_DESCRIPTIONS,
    GEOMETRY_REQUEST_DESCRIPTIONS, PUBLIC_EXAMPLES, public_definitions)
from types import SimpleNamespace
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.cortex.agents.codex.session import CodexSession, RecordedClient
from ni_rua.runtime.tools.builtins import create_registry
from ni_rua.runtime.tools.registry import ToolRegistry

# ==== tests/test_codex.py ====

class CodexCheck(unittest.IsolatedAsyncioTestCase):
    async def test_large_jsonl_echo_eof_and_notification_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            socket = Path(temporary) / "fixture-app-server.sock"

            async def echo(request):
                connection = web.WebSocketResponse(max_msg_size=2 * 1024 * 1024)
                await connection.prepare(request)
                async for message in connection:
                    if message.type == WSMsgType.TEXT:
                        await connection.send_str(message.data)
                return connection

            app = web.Application()
            app.router.add_get("/", echo)
            runner = web.AppRunner(app)
            await runner.setup()
            process = None
            try:
                await web.UnixSite(runner, str(socket)).start()
                process = await asyncio.create_subprocess_exec(
                    sys.executable, "-m", "ni_rua.cortex.agents.codex.bridge", "proxy", str(socket),
                    stdin=asyncio.subprocess.PIPE, stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE, limit=2 * 1024 * 1024,
                )
                payload = dict(jsonrpc="2.0", id=1, result=dict(data_origin="synthetic_fixture", payload="x" * (128 * 1024)))
                line = (canonical(payload) + "\n").encode()
                process.stdin.write(line)
                await process.stdin.drain()
                echoed = await asyncio.wait_for(process.stdout.readline(), timeout=10)
                self.assertEqual(echoed, line)
                self.assertEqual(json.loads(echoed), payload)
                process.stdin.close()
                await process.stdin.wait_closed()
                self.assertEqual(await asyncio.wait_for(process.wait(), timeout=5), 0)
                self.assertEqual(await process.stderr.read(), b"")
            finally:
                if process is not None and process.returncode is None:
                    process.kill()
                    await process.wait()
                await runner.cleanup()

            identity = dict(run_id="synthetic-codex", episode_id="fixture-episode", env_instance_id="fixture-env", env_generation=1)
            store = Store(Path(temporary) / "store", identity)
            try:
                # No app-server/model launch: exercise only raw notification indexing.
                session = CodexSession(temporary, store, lambda params: None)
                session.thread_id, session.turn_id, session.state = "owned-thread", "current-turn", "running"
                session._notification(dict(method="future/fixtureNotification", params=None))
                for thread, turn in [("other-thread", "other-turn"), ("owned-thread", "old-turn")]:
                    session._notification(dict(method="turn/completed", params=dict(threadId=thread, turn=dict(id=turn, status="completed"))))
                    self.assertEqual(session.state, "running")
                    self.assertFalse(session.turn_done.is_set())
                session._notification(dict(method="turn/completed", params=dict(threadId="owned-thread", turn=dict(id="current-turn", status="completed"))))
                self.assertEqual(session.state, "completed")
                self.assertTrue(session.turn_done.is_set())
                self.assertEqual(len(store.events()), 4)
                native_result = session.turn_result
                session.state = "running"
                session.turn_done.clear()
                session.close()
                self.assertEqual(session.state, "closed")
                self.assertTrue(session.turn_done.is_set())
                self.assertEqual(session.turn_result, native_result)

                session.state, session.thread_id, session.turn_id = "running", "owned-thread", "current-turn"

                class CompletedTurnClient:
                    def turn_interrupt(self, _thread_id, _turn_id):
                        from ni_rua.cortex.agents.codex.session import JsonRpcError
                        raise JsonRpcError(-32600, "no active turn to interrupt")

                session.client = CompletedTurnClient()
                session.interrupt()
            finally:
                store.close()

class CodexSteerTests(unittest.TestCase):
    def test_steer_targets_only_the_active_turn(self):
        import tempfile
        from types import SimpleNamespace
        identity = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary) / "store", identity)
            try:
                session = CodexSession(temporary, store, lambda params: None)
                calls = []

                class SteerClient:
                    def turn_steer(self, thread_id, expected_turn_id, items):
                        calls.append((thread_id, expected_turn_id, items))
                        return SimpleNamespace(turn_id=expected_turn_id)

                session.client = SteerClient()
                session.thread_id, session.turn_id, session.state = "owned-thread", "current-turn", "completed"
                with self.assertRaisesRegex(ValueError, "no_active_turn_to_steer"):
                    session.steer("wait")
                session.state = "running"
                self.assertEqual(session.steer("先看橙色方块"), "current-turn")
                self.assertEqual(calls, [("owned-thread", "current-turn", [{"type": "text", "text": "先看橙色方块"}])])
                self.assertEqual(session.state, "running")  # steering never ends or restarts the turn
                self.assertIn("codex.turn_steered", [event["type"] for event in store.events()])
            finally:
                store.close()

# ==== tests/test_codex_deferred_tools.py ====

class NativeDeferredToolsTests(unittest.TestCase):
    def test_discovery_calls_validation_disabled_tools_and_resume_share_native_path(self):
        self._native_lifecycle(deferred=True)

    def test_legacy_flat_catalog_resumes_without_namespace_and_rejects_changed_catalog(self):
        self._native_lifecycle(deferred=False)

    def _native_lifecycle(self, *, deferred):
        requests, dispatched = [], []
        # 先注册临时目录清理（LIFO 最后执行），避免后台代理线程仍在写 blobs 时提前 rmtree
        temporary_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temporary_dir.cleanup)
        if True:
            root = Path(temporary_dir.name)
            store = Store(root / 'run', dict(run_id='native-discovery', episode_id='episode',
                env_instance_id='synthetic', env_generation=1))
            self.addCleanup(store.close)
            robot = FixtureRobot(root)
            harness = Harness(store, robot)
            description = ('Read a deterministic fixture value. Input field guide: value is an integer; '
                'minimum 0, maximum 10. ' + 'Complete deferred field help. ' * 200)
            def read(context, args, correlation):
                dispatched.append(deepcopy(args))
                return ToolOutput({'value': args['value']}).response()
            definition = dict(type='function', name='fixture_read', description=description,
                inputSchema=dict(type='object', properties={'value': dict(type='integer', minimum=0,
                    maximum=10, description='Integer fixture value from 0 through 10.')},
                    required=['value'], additionalProperties=False))
            if deferred:
                definition['deferLoading'] = True
            result_schema = {'data': dict(type='object', properties={'value': {'type': 'integer'}},
                                         required=['value'], additionalProperties=False)}
            harness.tool_registry = ToolRegistry([
                ToolSpec(definition, read, output_schemas=result_schema),
                ToolSpec(dict(type='function', name='fixture_eager', description='Eager fixture sentinel.',
                    inputSchema={'type': 'object'}), read, output_schemas=result_schema),
                ToolSpec(dict(type='function', name='fixture_program_only', description='Program-only sentinel.',
                    inputSchema={'type': 'object'}, deferLoading=True), read,
                    roles=frozenset({'program'}), output_schemas=result_schema)])
            catalog = harness.tool_registry.catalog(harness)
            self.assertNotIn('fixture_program_only', [tool['name'] for tool in catalog])
            search_count = int(deferred)

            class Handler(BaseHTTPRequestHandler):
                def log_message(self, *_): pass
                def do_POST(self):
                    request = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                    requests.append(request)
                    index = len(requests)
                    if deferred and index == 1:
                        item = dict(type='tool_search_call', id='search-item', call_id='search-call',
                            execution='client', arguments={'query': 'fixture_read', 'limit': 1}, status='completed')
                    elif index - search_count in (1, 2, 3, 5):
                        phase = index - search_count
                        harness.disabled_tools = {'fixture_read'} if phase == 3 else set()
                        item = dict(type='function_call', id=f'call-item-{index}', call_id=f'call-{index}',
                            name='fixture_read', arguments=json.dumps({'value': {1: 7, 2: 11, 3: 8, 5: 9}[phase]}),
                            status='completed')
                        if deferred:
                            item['namespace'] = DYNAMIC_TOOL_NAMESPACE
                        elif phase == 5:
                            item['namespace'] = 'functions'
                    else:
                        harness.disabled_tools = set()
                        item = dict(type='message', id=f'message-{index}', role='assistant', status='completed',
                            content=[dict(type='output_text', text='Fixture complete.', annotations=[])])
                    response = dict(id=f'response-{index}', model=request['model'], status='completed', output=[item],
                        usage=dict(input_tokens=10, output_tokens=2, total_tokens=12))
                    events = [dict(type='response.created', response={**response, 'status': 'in_progress', 'output': []}),
                        dict(type='response.output_item.added', output_index=0, item=item),
                        dict(type='response.output_item.done', output_index=0, item=item),
                        dict(type='response.completed', response=response)]
                    payload = ''.join('data: ' + json.dumps(event) + '\n\n' for event in events).encode()
                    self.send_response(200)
                    self.send_header('Content-Type', 'text/event-stream')
                    self.send_header('Content-Length', str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)

            server = ThreadingHTTPServer(('127.0.0.1', 0), Handler)
            threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True).start()
            self.addCleanup(server.server_close)
            self.addCleanup(server.shutdown)
            source = root / 'source'
            source.mkdir()
            (source / 'config.toml').write_text('model_provider="fixture"\n[model_providers.fixture]\n'
                f'base_url="http://127.0.0.1:{server.server_port}/v1"\n')
            config = Config.from_dict(dict(model=dict(codex_home=str(source), native_memories=False)))

            def session_for_run():
                session = CodexSession(store.root, store, harness.handle_tool, config=config)
                if deferred:
                    session.supports_search_tool = True
                self.addCleanup(close_session, session)
                return session

            def close_session(session):
                bridge = session.client._proc if session.client else None
                session.close()
                if bridge:
                    for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                        if stream and not stream.closed:
                            stream.close()

            session = session_for_run()
            harness.thread_id = session.start(catalog, 'Exercise only the deterministic fixture tools.')
            session.run_turn('Use the fixture and finish.')
            self.assertTrue(session.turn_done.wait(15), list(session.store.events())[-3:])
            self.assertEqual(session.state, 'completed')
            self.assertEqual(len(requests), 4 + search_count)
            self.assertEqual(dispatched, [{'value': 7}])
            self.assertEqual(session.model_proxy.request_count, len(requests))
            self.assertEqual(robot.executions, 0)
            first_tools = [tool for item in requests[0]['input'] if item.get('type') == 'additional_tools'
                           for tool in item['tools']]
            first_functions = first_tools + [tool for namespace in first_tools for tool in namespace.get('tools', [])]
            self.assertTrue(any(tool.get('name') == 'fixture_eager' for tool in first_functions))
            self.assertEqual(any(tool.get('type') == 'tool_search' for tool in first_tools), deferred)
            self.assertTrue(any(tool.get('name') == 'exec_command' for namespace in first_tools
                                for tool in namespace.get('tools', [])))
            self.assertTrue(any(part.get('text', '').startswith('You are Codex, an agent based on GPT-6.')
                for item in requests[0]['input'] for part in item.get('content', [])))
            if deferred:
                self.assertNotIn(description, json.dumps(requests[0]))
                found = next(item for item in requests[1]['input'] if item.get('type') == 'tool_search_output')
                self.assertEqual(found['execution'], 'client')
                self.assertEqual(found['tools'][0]['name'], DYNAMIC_TOOL_NAMESPACE)
                self.assertEqual(found['tools'][0]['tools'][0]['description'], description)
            outputs = [item for item in requests[-1]['input'] if item.get('type') == 'function_call_output']
            self.assertEqual(len(outputs), 3)
            self.assertIn('"value":7', json.dumps(outputs[0], separators=(',', ':')).replace('\\', ''))
            self.assertIn('maximum', json.dumps(outputs[1]))
            self.assertIn('tool_disabled_by_config:fixture_read', json.dumps(outputs[2]))
            thread_id = session.thread_id
            close_session(session)

            resumed = session_for_run()
            changed = deepcopy(catalog)
            changed[0]['deferLoading'] = not deferred
            with mock.patch.object(resumed, '_prepare') as prepare, \
                    mock.patch.object(resumed, '_stop_previous_owned_processes') as stop:
                with self.assertRaisesRegex(ValueError, 'resume_tool_registry_conflict'):
                    resumed.start(changed, 'Changed publication.', resume_thread_id=thread_id)
                prepare.assert_not_called()
                stop.assert_not_called()
            harness.thread_id = resumed.start(catalog, 'Exercise only the deterministic fixture tools.',
                                              resume_thread_id=thread_id)
            self.assertEqual(harness.thread_id, thread_id)
            resumed.run_turn('Continue with the previously declared fixture.')
            self.assertTrue(resumed.turn_done.wait(15))
            self.assertEqual(resumed.state, 'completed')
            self.assertEqual(dispatched, [{'value': 7}, {'value': 9}])
            self.assertEqual(len(requests), 6 + search_count)
            self.assertEqual(resumed.model_proxy.request_count, 2)
            self.assertEqual(store.read_state('runtime', 'model_requests')['completed_calls'], len(requests))
            self.assertEqual(len(store.records('tool_call')), 4)
            self.assertNotIn('Program-only sentinel.', json.dumps(requests))
            self.assertEqual(robot.executions, 0)

# ==== tests/test_codex_image_transport.py ====

class CodexImageTransportTests(unittest.TestCase):
    def test_native_dynamic_image_and_text_reach_actual_responses_request(self):
        requests, calls, notifications, endpoints = [], [], [], []
        # Inspect every schema in this transport fixture; discovery has its own
        # full CodexSession lifecycle test.
        public_tools = [dict(tool, deferLoading=False) for tool in public_definitions()]
        evidence = os.environ.get('NI_RUA_CODEX_IMAGE_TEST_EVIDENCE')
        evidence_dir = Path(evidence) if evidence else None
        if evidence_dir:
            evidence_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
        buffer = io.BytesIO()
        Image.new('RGB', (9, 7), (204, 51, 17)).save(buffer, format='PNG')
        image_bytes = buffer.getvalue()
        image_sha = hashlib.sha256(image_bytes).hexdigest()
        image_url = 'data:image/png;base64,' + base64.b64encode(image_bytes).decode()
        log = 'begin-native-log|' + 'diagnostic ' * 60 + '|end-native-log'
        proof_url = 'https://example.invalid/source-proof'
        metadata_log = 'fixture progress\n' + json.dumps(dict(hash=image_sha, source_url=proof_url,
            measured_position_m=[0.125, 0.0, 1.25], source_kind='measurement')) + '\nfixture end\n'
        result = dict(job_id='fixture-job', status='succeeded', stdout_text=log,
                      stderr_text=metadata_log, hash=image_sha, source_url=proof_url,
                      nested=dict(contentDigest=image_sha, checksums=[image_sha], sourceUrl=proof_url),
                      version=image_sha)
        image_part = dict(type='inputImage', imageUrl=image_url, hash=image_sha, source_url=proof_url)

        class ResponsesHandler(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'

            def log_message(self, *_):
                pass

            def do_GET(self):
                # 忽略共享开发机 IDE 端口转发守护进程发送的 Go-http-client 根路径探测
                if self.path == '/' and self.headers.get('User-Agent', '').startswith('Go-http-client/'):
                    self.send_response(204)
                    self.end_headers()
                    return
                # A provider/model discovery request cannot leave loopback.
                endpoints.append(dict(method='GET', path=self.path))
                body = json.dumps({'data': []}).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

            def do_POST(self):
                payload = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                requests.append(payload)
                endpoints.append(dict(method='POST', path=self.path))
                if evidence_dir:
                    (evidence_dir / ('request-' + str(len(requests)) + '.json')).write_text(json.dumps(payload))
                if len(requests) == 1:
                    item = dict(type='function_call', id='fc-native-image', call_id='native-image-call',
                                name='image_fixture', arguments='{}', status='completed')
                else:
                    item = dict(type='message', id='msg-fixture-finish', role='assistant',
                                status='completed', content=[dict(type='output_text', text='fixture complete', annotations=[])])
                events = [
                    ('response.created', {'response': {'id': 'resp-fixture-' + str(len(requests)),
                         'status': 'in_progress', 'output': []}}),
                    ('response.output_item.added', {'output_index': 0, 'item': item}),
                    ('response.output_item.done', {'output_index': 0, 'item': item}),
                    ('response.completed', {'response': {'id': 'resp-fixture-' + str(len(requests)),
                         'status': 'completed', 'output': [item],
                         'usage': {'input_tokens': 10, 'output_tokens': 2, 'total_tokens': 12}}}),
                ]
                body = ''.join('event: ' + event + '\ndata: ' + json.dumps({'type': event, **value})
                               + '\n\n' for event, value in events).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        server = ThreadingHTTPServer(('127.0.0.1', 0), ResponsesHandler)
        thread = threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.01}, daemon=True)
        thread.start()
        self.addCleanup(server.server_close)
        self.addCleanup(server.shutdown)
        with tempfile.TemporaryDirectory(prefix='ni-rua-native-image-') as directory:
            root = Path(directory)
            codex_home = root / 'codex-home'
            codex_home.mkdir()
            catalog = write_direct_model_catalog(codex_home / 'models.json')
            base_url = 'http://127.0.0.1:' + str(server.server_address[1]) + '/v1'
            (codex_home / 'config.toml').write_text(
                'model = "gpt-6.1-sol"\nmodel_provider = "fixture"\n'
                'model_catalog_json = ' + json.dumps(str(catalog)) + '\n'
                'approval_policy = "never"\nsandbox_mode = "danger-full-access"\n'
                'web_search = "disabled"\n'
                '[model_providers.fixture]\nname = "Loopback fixture"\n'
                'base_url = ' + json.dumps(base_url) + '\nwire_api = "responses"\n'
                'requires_openai_auth = false\nsupports_websockets = false\n'
                '[features]\ncode_mode_host = false\nshell_tool = true\nunified_exec = true\n'
                'memories = false\napps = false\nplugins = false\n'
                '[features.code_mode]\nenabled = false\n')
            store = Store(root / 'run', dict(run_id='native-image-transport', episode_id='episode',
                env_instance_id='synthetic', env_generation=1))
            presenter = ActorOutput(store, ActorPresentation(store))

            def tool_handler(params):
                calls.append(params)
                self.assertEqual(params['tool'], 'image_fixture')
                return presenter.present(ToolOutput(result, extra_content=(image_part,)).response(),
                                         tool=params['tool'])

            # The SDK normally copies os.environ. Start a completely separate
            # app-server without host auth, proxy, or orchestrator context.
            client = RecordedClient(CodexConfig(cwd=str(root), launch_args_override=(
                '/usr/bin/env', '-i', 'PATH=/usr/bin:/bin', 'HOME=' + str(root),
                'CODEX_HOME=' + str(codex_home), 'TMPDIR=' + str(root),
                str(bundled_codex_path()), '--no-daemon', 'app-server', '--listen', 'stdio://')),
                store, tool_handler, notifications.append)
            try:
                client.start()
                client.initialize()
                effective = client.request('config/read', {'includeLayers': True}, response_model=ConfigReadResponse)
                features = effective.model_dump(by_alias=True)['config']['features']
                self.assertFalse(features['code_mode']['enabled'])
                self.assertFalse(features['code_mode_host'])
                started = client.thread_start(dict(model='gpt-6.1-sol', modelProvider='fixture',
                    cwd=str(root), approvalPolicy='never', sandbox='danger-full-access',
                    dynamicTools=public_tools + [dict(type='function', name='image_fixture', deferLoading=False,
                        description='Return deterministic fixture image and diagnostic log.',
                        inputSchema=dict(type='object', properties={}, additionalProperties=False))]))
                turn = client.turn_start(started.thread.id, 'Call image_fixture, then finish.')
                deadline = time.monotonic() + 20
                while time.monotonic() < deadline:
                    terminal = next((message for message in notifications
                        if message.get('method') == 'turn/completed'
                        and message.get('params', {}).get('turn', {}).get('id') == turn.turn.id), None)
                    if terminal:
                        break
                    time.sleep(.01)
                else:
                    self.fail('isolated native turn did not finish; stderr: ' + '\n'.join(client._stderr_lines)[-2500:])
                self.assertEqual(terminal['params']['turn']['status'], 'completed', terminal)
                self.assertEqual(len(calls), 1)
                self.assertEqual(len(requests), 2)
                self.assertEqual(endpoints, [dict(method='POST', path='/v1/responses')] * 2)
                self.assertEqual(requests[1]['model'], 'gpt-6.1-sol')
                tool_specs = requests[0].get('tools', []) + [tool
                    for item in requests[0]['input'] if item.get('type') == 'additional_tools'
                    for tool in item.get('tools', [])]
                flat_specs = [spec for spec in tool_specs if spec.get('type') == 'function']
                flat_specs += [tool for spec in tool_specs for tool in spec.get('tools', [])
                               if tool.get('type') == 'function']
                native_tools = {spec['name']: spec for spec in flat_specs}
                names = sorted({spec['name'] for spec in tool_specs if spec.get('name')} | set(native_tools))
                description_coverage = {}
                for definition in public_tools:
                    name = definition['name']
                    self.assertIn(name, native_tools)
                    self.assertEqual(native_tools[name]['description'], definition['description'], name)
                    for example in PUBLIC_EXAMPLES.get(name, []):
                        self.assertIn(canonical(example), native_tools[name]['description'], name)
                    def schema_nodes(value, path=()):
                        if isinstance(value, dict):
                            yield path, value
                            for key, child in value.items():
                                if key not in {'description', 'examples', 'default', 'enum', 'const', 'if', 'not'}:
                                    yield from schema_nodes(child, path + (key,))
                        elif isinstance(value, list):
                            for index, child in enumerate(value):
                                yield from schema_nodes(child, path + (index,))
                    local_nodes = dict(schema_nodes(definition['inputSchema']))
                    local_help = {path: node['description'] for path, node in local_nodes.items()
                                  if 'description' in node}
                    native_help = {path: node['description'] for path, node in
                                   schema_nodes(native_tools[name]['parameters']) if 'description' in node}
                    for path, help_text in local_help.items():
                        self.assertTrue(native_help.get(path) == help_text or
                            help_text in native_tools[name]['description'], (name, path))
                    lost_limits = ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
                                   'maxItems', 'minLength', 'maxLength', 'pattern', 'uniqueItems')
                    limit_nodes = [node for node in local_nodes.values() if any(key in node for key in lost_limits)]
                    # Constraint prose is checked by description tests. The
                    # exact equality above proves its complete native delivery.
                    description_coverage[name] = dict(
                        local_schema_field_guides=len(local_help),
                        native_schema_field_guides=sum(native_help.get(path) == help_text
                            for path, help_text in local_help.items()),
                        native_visible_distinct_schema_descriptions=len(set(local_help.values())),
                        native_visible_limit_nodes=len(limit_nodes),
                        native_schema_bytes=len(canonical(native_tools[name]['parameters']).encode()),
                        top_description_bytes=len(native_tools[name]['description'].encode()),
                        top_description_equal=True)
                self.assertEqual(native_tools['robot_action']['parameters']['properties']['stages']
                    ['items']['properties']['angles_deg']['items']['type'], 'number')
                request_guide_counts = {}
                for name, fields in (('geometry_query', GEOMETRY_REQUEST_DESCRIPTIONS),
                                     ('depth_query', DEPTH_REQUEST_DESCRIPTIONS)):
                    for field, help_text in fields.items():
                        self.assertIn(help_text, native_tools[name]['description'], (name, field))
                    request_guide_counts[name] = len(fields)
                output = next(x for x in requests[1]['input']
                    if x.get('type') == 'function_call_output' and x.get('call_id') == 'native-image-call')
                self.assertIsInstance(output['output'], list)
                images = [part for part in output['output'] if part.get('type') == 'input_image']
                self.assertEqual(len(images), 1)
                self.assertEqual(images[0]['image_url'], image_url)
                self.assertIn(images[0].get('detail'), (None, 'high'))
                received = base64.b64decode(images[0]['image_url'].split(',', 1)[1], validate=True)
                self.assertEqual(received, image_bytes)
                self.assertEqual(hashlib.sha256(received).hexdigest(), image_sha)
                text = '\n'.join(part['text'] for part in output['output'] if part['type'] == 'input_text')
                self.assertNotIn('data:image', text)
                self.assertIn(log, text)
                self.assertNotIn(image_sha, text)
                self.assertNotIn(proof_url, text)
                shown_result = json.loads(next(part['text'] for part in output['output']
                    if part['type'] == 'input_text'))
                self.assertEqual(shown_result, dict(job_id='fixture-job', status='succeeded', stdout_text=log,
                    stderr_text='fixture progress\n' + canonical(dict(measured_position_m=[0.125, 0.0, 1.25],
                        source_kind='measurement')) + '\nfixture end\n', nested={}))
                self.assertEqual(result['hash'], image_sha)
                self.assertEqual(result['stderr_text'], metadata_log)
                self.assertIn('image_fixture', names)
                self.assertNotIn('exec', names)
                if evidence_dir:
                    (evidence_dir / 'summary.json').write_text(json.dumps(dict(
                        request_count=len(requests), endpoints=endpoints, model=requests[1]['model'],
                        tool_names=names, image_count=len(images), image_sha256=image_sha,
                        public_tool_count=len(public_tools), public_description_coverage=description_coverage,
                        request_field_guides=request_guide_counts,
                        native_detail=images[0].get('detail'), source_bytes_equal=received == image_bytes,
                        stdout_equal=shown_result['stdout_text'] == log,
                        wire_metadata_removed=image_sha not in text and proof_url not in text,
                        jsonl_measurements_preserved=shown_result['stderr_text'].splitlines()[1]
                            == canonical(dict(measured_position_m=[0.125, 0.0, 1.25], source_kind='measurement')),
                        feature_flags=features, app_server_args=list(client._proc.args),
                        request_body_sha256=[hashlib.sha256(json.dumps(x).encode()).hexdigest() for x in requests],
                        external_model_or_robot_calls=0, catalog_discovery_requests=0,
                        turn_status=terminal['params']['turn']['status']), indent=2))
            finally:
                process = client._proc
                client.close()
                if process:
                    for stream in (process.stdout, process.stderr):
                        if stream:
                            stream.close()
                client.server_pool.shutdown(wait=True)
                store.close()

# ==== tests/test_codex_parallel_dispatch.py ====

TIMEOUT = 5


def tool_message(identifier, tool, arguments):
    return dict(id=identifier, method='item/tool/call', params=dict(
        threadId='thread-owned', turnId='turn-owned', callId='call-%s' % identifier, tool=tool, arguments=arguments))


class RecordedClientDispatchTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.store = Store(self.root, dict(run_id='parallel-dispatch', episode_id='ep', env_instance_id='env',
                                           env_generation=1))
        self.addCleanup(self.store.close)
        self.lock = threading.Lock()
        self.running, self.peak, self.order = 0, 0, []

    def client(self, handler_gate, **kwargs):
        def tool(params):
            with self.lock:
                self.running += 1
                self.peak = max(self.peak, self.running)
                self.order.append(('start', params['tool']))
            try:
                handler_gate(params)
                return {'success': True, 'contentItems': [{'type': 'inputText', 'text': params['tool']}]}
            finally:
                with self.lock:
                    self.running -= 1
                    self.order.append(('end', params['tool']))
        client = RecordedClient(CodexConfig(cwd=str(self.root)), self.store, tool, lambda message: None, **kwargs)
        self.addCleanup(client.server_pool.shutdown, wait=True, cancel_futures=True)
        self.written = []
        client._write_message = self.written.append
        return client

    def test_default_orders_every_tool_call_even_former_allowlist_reads(self):
        client = self.client(lambda params: time.sleep(0.03))
        self.assertIsNone(client.parallel_safe)
        futures = [client._submit_request(tool_message(index, 'robot_information', {})) for index in range(3)]
        for future in futures:
            future.result(TIMEOUT)
        self.assertEqual(self.peak, 1)
        self.assertEqual([reply['id'] for reply in self.written], [0, 1, 2])

    def test_registry_classifier_overlaps_reads_and_orders_state_changes(self):
        registry = create_registry()
        seen = []

        def classify(tool, arguments):
            seen.append((tool, arguments))
            return registry.parallel_safe(tool, arguments)
        rendezvous = threading.Barrier(2, timeout=TIMEOUT)

        def gate(params):
            if params['tool'] == 'robot_information':
                rendezvous.wait()
        client = self.client(gate, parallel_safe=classify)
        reads = [client._submit_request(tool_message('read-%d' % index, 'robot_information', {'section': 'default'}))
                 for index in range(2)]
        plan = client._submit_request(tool_message('plan', 'task_plan', {'operation': 'update', 'plan': 'x'}))
        for future in (*reads, plan):
            future.result(TIMEOUT)
        self.assertEqual(seen[-1], ('task_plan', {'operation': 'update', 'plan': 'x'}))
        self.assertEqual(self.peak, 2)
        plan_start = self.order.index(('start', 'task_plan'))
        self.assertEqual([event for event in self.order[:plan_start] if event[0] == 'end'],
                         [('end', 'robot_information')] * 2)
        self.assertEqual({reply['id'] for reply in self.written}, {'read-0', 'read-1', 'plan'})

    def test_classifier_failure_never_stops_the_reader(self):
        def classify(tool, arguments):
            raise RuntimeError('fixture classifier failure')
        client = self.client(lambda params: None, parallel_safe=classify)
        messages = iter([tool_message(1, 'robot_information', {}), dict(id=9, result={'still': 'reading'})])
        routed = []

        def read():
            try:
                return next(messages)
            except StopIteration:
                raise EOFError('fixture transport ended')
        client._read_message = read
        client._router = SimpleNamespace(route_response=routed.append, fail_all=lambda error: None)
        client._reader_loop()
        self.assertEqual(routed, [dict(id=9, result={'still': 'reading'})])
        client.server_pool.shutdown(wait=True)
        self.assertEqual([reply['id'] for reply in self.written], [1])

    def test_non_tool_requests_bypass_the_tool_barrier(self):
        release = threading.Event()
        replies = []
        client = self.client(lambda params: release.wait(TIMEOUT))
        client.request_handler = lambda message, **kwargs: replies.append((message['id'], kwargs)) or {}
        blocked = client._submit_request(tool_message('tool', 'task_plan', {'operation': 'update'}))
        other = client._submit_request(dict(id='approval', method='item/fileChange/requestApproval', params={}))
        other.result(TIMEOUT)
        self.assertFalse(blocked.done())
        release.set()
        blocked.result(TIMEOUT)
        self.assertEqual(replies[0], ('approval', {}))


class CodexSessionWiringTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)

    def start_with_fake_client(self, session):
        session.private.mkdir(parents=True, exist_ok=True)
        session.socket_path = self.root / 'fixture.sock'
        session.socket_path.touch()
        session.sandbox = mock.Mock(host_socket_path='/proc/fixture/fd/pinned')
        session.sandbox.repair_socket_alias.return_value = True
        session.command, session.env, session.cli_executable = ['fixture-only'], {}, Path('/fixture/codex')
        session.model_proxy = mock.Mock()
        client = mock.MagicMock()
        client.initialize.return_value.model_dump.return_value = {}
        client.thread_start.return_value = SimpleNamespace(thread=SimpleNamespace(id='fixture-thread'))
        with mock.patch.object(session, '_prepare', return_value='fixture'), \
                mock.patch.object(session, '_stop_previous_owned_processes'), \
                mock.patch('ni_rua.cortex.agents.codex.session.subprocess.Popen',
                           return_value=SimpleNamespace(pid=1, poll=lambda: 0)), \
                mock.patch('ni_rua.cortex.agents.codex.session.RecordedClient', return_value=client) as factory:
            session.start([], 'fixture instructions')
        self.addCleanup(session.close)
        return factory

    def test_unwired_session_orders_every_call(self):
        store = Store(self.root / 'run', dict(run_id='unwired', episode_id='ep', env_instance_id='env', env_generation=1))
        self.addCleanup(store.close)
        session = CodexSession(self.root / 'run', store, lambda params: {},
                               config=Config.from_dict({'model': {'codex_home': str(self.root / 'codex-home')}}))
        self.assertIsNone(CodexSession.tool_parallel_safe)
        factory = self.start_with_fake_client(session)
        self.assertIsNone(factory.call_args.kwargs['parallel_safe'])
        self.assertIs(factory.call_args.kwargs['route_notifications'], False)

    def test_workbench_wires_registry_classifier_into_the_native_client(self):
        robot = FixtureRobot(self.root)
        robot.close = lambda: None
        config = Config.from_dict({'features': {'grasp_memory_dir': str(self.root / 'memory')},
                                   'model': {'codex_home': str(self.root / 'codex-home')}})
        workbench = Workbench(self.root / 'workbench', robot=robot, config=config,
                              observation_profile='rgb_proprio_v1', motion_review=False)
        self.addCleanup(workbench.store.close)
        self.addCleanup(workbench.run_lock.close)
        if workbench.grasp_memory_store:
            self.addCleanup(workbench.grasp_memory_store.close)
        wired = workbench.session.tool_parallel_safe
        self.assertIs(wired.__self__, workbench.harness.tool_registry)
        self.assertIs(wired.__func__, ToolRegistry.parallel_safe)
        factory = self.start_with_fake_client(workbench.session)
        passed = factory.call_args.kwargs['parallel_safe']
        self.assertEqual(passed, workbench.harness.tool_registry.parallel_safe)
        self.assertTrue(passed('robot_information', {}))
        self.assertFalse(passed('robot_action', {'mode': 'execute'}))
        self.assertFalse(passed('robot_observe', {}))


if __name__ == "__main__":
    unittest.main()
