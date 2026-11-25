"""Self-tests for the shared fixtures in ``tests/support`` (fake Claude SDK, fake model servers, fake MCP client)."""
import asyncio
import base64
import gzip
import http.client
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import inspect
import json
import select
import socket
import threading
import time
import unittest

from support.fake_claude_sdk import (SDK_AVAILABLE, FakeClaudeSDK, FakeClaudeSDKClient, assistant_message,
                                     echo_message, init_message, load_capture, message_from_capture, requires_sdk,
                                     result_message, state_message, tool_result_message, tool_use, turn_messages)
from support.fake_mcp_client import FakeMcpClient, McpError, parse_sse
from support.fake_model_servers import (FakeAnthropicServer, FakeChatServer, FakeResponsesServer, Failure, Gate, Step,
                                        Usage, text_step, tool_step)
from support.paths import FIXTURES_ROOT

NI_RUA_TEST_TIER = "fast"
FIXTURES = FIXTURES_ROOT / "claude_sdk"


class Loop:
    """A private event loop on a thread, like the session's ``claude-loop``."""

    def __init__(self):
        self.loop = asyncio.new_event_loop()
        self.thread = threading.Thread(target=self.loop.run_forever, daemon=True)
        self.thread.start()

    def run(self, coro, timeout=5):
        return asyncio.run_coroutine_threadsafe(coro, self.loop).result(timeout)

    def submit(self, coro):
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def close(self):
        self.loop.call_soon_threadsafe(self.loop.stop)
        self.thread.join(5)
        self.loop.close()


async def drain(client, into):
    async for message in client.receive_messages():
        into.append(message)


@requires_sdk
class FakeClientCheck(unittest.TestCase):
    def setUp(self):
        self.loop = Loop()
        self.fake = FakeClaudeSDK()
        self.client = self.fake.factory(None)

    def tearDown(self):
        self.loop.close()

    def connect(self):
        self.loop.run(self.client.connect())

    def test_unconnected_calls_raise_like_the_sdk(self):
        from claude_agent_sdk._errors import CLIConnectionError
        for call in (self.client.query("x"), self.client.interrupt(), self.client.set_model("m")):
            with self.assertRaisesRegex(CLIConnectionError, "Not connected"):
                self.loop.run(call)
        with self.assertRaisesRegex(CLIConnectionError, "Not connected"):
            self.loop.run(drain(self.client, []))

    def test_stream_in_order_then_clean_end(self):
        self.connect()
        received = []
        reader = self.loop.submit(drain(self.client, received))
        self.client.emit(*turn_messages("hello", session_id="s1"))
        self.client.end()
        reader.result(5)
        self.assertEqual([type(m).__name__ for m in received],
                         ["SystemMessage", "SystemMessage", "AssistantMessage", "ResultMessage", "SystemMessage"])
        self.assertEqual(received[-1].data["state"], "idle")

    def test_failure_raised_once_and_ends_stream(self):
        from claude_agent_sdk._errors import ProcessError
        self.connect()
        received = []
        reader = self.loop.submit(drain(self.client, received))
        self.client.emit(assistant_message("a"))
        self.client.kill_process(-9)
        with self.assertRaises(ProcessError):
            reader.result(5)
        self.assertEqual(len(received), 1)
        self.assertEqual((self.client.process.returncode,), (-9,))

    def test_second_concurrent_consumer_is_a_test_error(self):
        self.connect()
        first = self.loop.submit(drain(self.client, []))
        self.client.wait_for_call("connect")
        for _ in range(200):
            if self.client._consumer_active:
                break
            threading.Event().wait(0.01)
        with self.assertRaisesRegex(RuntimeError, "fake_second_receive_consumer"):
            self.loop.run(drain(self.client, []))
        self.client.end()
        first.result(5)

    def test_disconnect_is_idempotent_ends_stream_and_blocks_later_calls(self):
        from claude_agent_sdk._errors import CLIConnectionError
        self.connect()
        reader = self.loop.submit(drain(self.client, []))
        self.loop.run(self.client.disconnect())
        self.loop.run(self.client.disconnect())
        reader.result(5)
        with self.assertRaises(CLIConnectionError):
            self.loop.run(self.client.query("late"))
        self.assertEqual(self.client.call_names(), ["connect", "disconnect", "disconnect"])

    def test_query_records_and_runs_handlers_that_emit(self):
        self.connect()
        received = []
        reader = self.loop.submit(drain(self.client, received))
        self.client.on_query.append(lambda client, text: client.emit(echo_message(text)))
        self.loop.run(self.client.query("go", "sid"))
        self.assertEqual(self.client.queries(), ["go"])
        self.assertEqual(self.client.calls[1], ("query", "go", "sid"))
        self.client.end()
        reader.result(5)
        self.assertEqual(received[0].content, "go")

    def test_fault_injection_and_gates(self):
        self.connect()
        self.client.query_error = lambda index: OSError("write failed") if index == 2 else None
        self.loop.run(self.client.query("one"))
        with self.assertRaisesRegex(OSError, "write failed"):
            self.loop.run(self.client.query("two"))
        self.loop.run(self.client.query("three"))
        self.client.hold("interrupt")
        pending = self.loop.submit(self.client.interrupt())
        self.assertTrue(self.client.wait_for_call("interrupt"))
        threading.Event().wait(0.05)
        self.assertFalse(pending.done())
        self.client.release("interrupt")
        pending.result(5)
        self.client.set_model_error = RuntimeError("Control request timeout: set_model")
        with self.assertRaisesRegex(RuntimeError, "timeout"):
            self.loop.run(self.client.set_model("m2"))

    def test_failing_connect_disconnects_before_raising(self):
        self.client.connect_error = RuntimeError("boom")
        with self.assertRaises(RuntimeError):
            self.loop.run(self.client.connect())
        self.assertEqual(self.client.call_names(), ["connect", "disconnect"])
        with self.assertRaisesRegex(RuntimeError, "fake_client_not_connected"):
            self.client.emit(assistant_message("x"))

    def test_emit_from_other_thread_and_init_on_connect(self):
        self.fake.mcp_tools = ("robot_observe",)
        self.fake.emit_init_on_connect()
        from claude_agent_sdk import ClaudeAgentOptions
        client = self.fake.factory(ClaudeAgentOptions(tools=["Bash"], session_id="sid-1", model="mm"))
        self.loop.run(client.connect())
        received = []
        reader = self.loop.submit(drain(client, received))
        thread = threading.Thread(target=lambda: (client.emit(assistant_message("t")), client.end()))
        thread.start()
        thread.join()
        reader.result(5)
        init = received[0]
        self.assertEqual((init.subtype, init.data["session_id"], init.data["model"], init.data["tools"]),
                         ("init", "sid-1", "mm", ["Bash", "mcp__ni_rua__robot_observe"]))
        self.assertIs(self.fake.client, client)

    def test_process_double_has_signal_semantics(self):
        process = self.client.process
        self.assertIsNone(process.returncode)
        process.terminate()
        process.kill()
        self.assertEqual((process.signals, process.returncode), (["TERM", "KILL"], -15))

    def test_public_surface_matches_the_installed_sdk_client(self):
        from claude_agent_sdk import ClaudeSDKClient
        for name in ("connect", "receive_messages", "query", "interrupt", "set_model", "disconnect"):
            real, fake = inspect.signature(getattr(ClaudeSDKClient, name)), inspect.signature(
                getattr(FakeClaudeSDKClient, name))
            self.assertEqual(list(real.parameters), list(fake.parameters), name)
            self.assertEqual(inspect.iscoroutinefunction(getattr(ClaudeSDKClient, name)) or
                             inspect.isasyncgenfunction(getattr(ClaudeSDKClient, name)),
                             inspect.iscoroutinefunction(getattr(FakeClaudeSDKClient, name)) or
                             inspect.isasyncgenfunction(getattr(FakeClaudeSDKClient, name)), name)


@requires_sdk
class BuildersAndCaptures(unittest.TestCase):
    def test_builders_make_real_sdk_dataclasses(self):
        import claude_agent_sdk as sdk
        messages = [init_message("s", tools=["Bash"]), state_message("running"),
                    assistant_message("hi", tool_use("t1", "Bash", command="ls")), tool_result_message("t1", "out"),
                    result_message(), echo_message("x")]
        self.assertEqual([type(m) for m in messages],
                         [sdk.SystemMessage, sdk.SystemMessage, sdk.AssistantMessage, sdk.UserMessage,
                          sdk.ResultMessage, sdk.UserMessage])
        self.assertEqual(messages[2].content[1].input, {"command": "ls"})
        self.assertEqual(messages[3].content[0].tool_use_id, "t1")

    def test_every_fixture_loads_into_sdk_objects(self):
        import claude_agent_sdk as sdk
        files = sorted(FIXTURES.glob("*.jsonl"))
        self.assertGreaterEqual(len(files), 7)
        for path in files:
            messages = load_capture(path)
            self.assertTrue(messages, path.name)
            self.assertTrue(all(isinstance(m, (sdk.SystemMessage, sdk.AssistantMessage, sdk.UserMessage,
                                               sdk.ResultMessage)) for m in messages), path.name)
        steer = load_capture(FIXTURES / "s3-during-tool-state-replay.jsonl")
        self.assertEqual([type(m).__name__ for m in steer][:3], ["SystemMessage", "SystemMessage", "UserMessage"])
        self.assertEqual(steer[-2].terminal_reason, "completed")
        self.assertEqual(steer[-1].data["state"], "idle")

    def test_message_from_capture_converts_blocks(self):
        message = message_from_capture(dict(_type="AssistantMessage", model="m", message_id="mid", usage={},
                                            content=[dict(_type="ToolUseBlock", id="i", name="Bash", input={})]))
        self.assertEqual(type(message.content[0]).__name__, "ToolUseBlock")


class Availability(unittest.TestCase):
    def test_sdk_is_installed_in_the_project_environment(self):
        self.assertTrue(SDK_AVAILABLE)


PNG = b'\x89PNG\r\n\x1a\n' + bytes(range(32))
PNG_B64 = base64.b64encode(PNG).decode()


def post(server, path, body, headers=None, raw=None):
    conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
    data = raw if raw is not None else json.dumps(body).encode()
    conn.request('POST', path, body=data, headers={'Content-Type': 'application/json', **(headers or {})})
    resp = conn.getresponse()
    out = resp.status, resp.headers, resp.read()
    conn.close()
    return out


def events(raw):
    """SSE frames as (event, parsed data); data may be the string [DONE]."""
    out = []
    for frame in parse_sse(raw):
        if frame['data'] is None:
            continue
        data = frame['data'] if frame['data'] == '[DONE]' else json.loads(frame['data'])
        out.append((frame['event'], data))
    return out


class AnthropicTests(unittest.TestCase):
    def test_streaming_thinking_text_tool(self):
        step = Step(thinking='hmm', text='hello world, long text', stop_reason='tool_use', keepalives=1,
                    tool_calls=[('read_file', {'path': '/a/b.txt', 'n': 3})],
                    usage=Usage(7, 9, cache_read_input_tokens=100, cache_creation_input_tokens=5))
        with FakeAnthropicServer([step]) as server:
            status, headers, raw = post(server, '/v1/messages?beta=true', {'model': 'm1', 'stream': True,
                                                                          'messages': []})
        self.assertEqual(status, 200)
        self.assertIn('text/event-stream', headers['Content-Type'])
        evs = events(raw)
        kinds = [e for e, _ in evs]
        self.assertEqual(kinds[:2], ['message_start', 'ping'])
        self.assertEqual(kinds[-2:], ['message_delta', 'message_stop'])
        start = evs[0][1]['message']
        self.assertEqual((start['model'], start['usage']['cache_read_input_tokens'],
                          start['usage']['cache_creation_input_tokens']), ('m1', 100, 5))
        self.assertTrue(start['id'])
        starts = [d['content_block']['type'] for e, d in evs if e == 'content_block_start']
        self.assertEqual(starts, ['thinking', 'text', 'tool_use'])
        text = ''.join(d['delta']['text'] for e, d in evs if e == 'content_block_delta'
                       and d['delta']['type'] == 'text_delta')
        self.assertEqual(text, 'hello world, long text')
        sig = [d['delta']['signature'] for e, d in evs if e == 'content_block_delta'
               and d['delta']['type'] == 'signature_delta']
        self.assertEqual(sig, ['fake-signature-0001'])
        fragments = [d['delta']['partial_json'] for e, d in evs if e == 'content_block_delta'
                     and d['delta']['type'] == 'input_json_delta']
        self.assertGreater(len(fragments), 1)
        self.assertEqual(json.loads(''.join(fragments)), {'path': '/a/b.txt', 'n': 3})
        delta = evs[-2][1]
        self.assertEqual((delta['delta']['stop_reason'], delta['usage']['output_tokens']), ('tool_use', 9))
        for e, d in evs:
            if e.startswith('content_block'):
                self.assertIn('index', d)

    def test_non_streaming_and_routes(self):
        with FakeAnthropicServer([tool_step('x', {'a': 1}, id='toolu_1', text='t')], count_tokens_value=42) as server:
            status, _, raw = post(server, '/v1/messages', {'model': 'm', 'stream': False})
            body = json.loads(raw)
            self.assertEqual(status, 200)
            self.assertEqual([b['type'] for b in body['content']], ['text', 'tool_use'])
            self.assertEqual(body['content'][1], {'type': 'tool_use', 'id': 'toolu_1', 'name': 'x', 'input': {'a': 1}})
            self.assertEqual(body['stop_reason'], 'tool_use')
            self.assertEqual(json.loads(post(server, '/v1/messages/count_tokens', {})[2]), {'input_tokens': 42})
            conn = http.client.HTTPConnection('127.0.0.1', server.port)
            conn.request('GET', '/v1/models')
            self.assertEqual(conn.getresponse().status, 404)
            self.assertEqual([r.path for r in server.requests],
                             ['/v1/messages', '/v1/messages/count_tokens', '/v1/models'])
            self.assertEqual(len(server.generation_requests()), 1)

    def test_host_port_probe_is_answered_but_not_counted(self):
        # Port-watching software on a shared host sends a body-less GET / to every new loopback listener.
        from support.fake_model_servers import is_host_port_probe
        self.assertTrue(is_host_port_probe('GET', '/'))
        for method, path, body in (('GET', '/v1/models', b''), ('POST', '/', b''), ('GET', '/', b'x'),
                                   ('GET', '/?x=1', b'')):
            self.assertFalse(is_host_port_probe(method, path, body), (method, path, body))
        with FakeAnthropicServer() as server:
            for host in ('127.0.0.1', 'localhost'):
                conn = http.client.HTTPConnection('127.0.0.1', server.port, timeout=5)
                conn.request('GET', '/', headers={'Host': f'{host}:{server.port}', 'User-Agent': 'Go-http-client/1.1'})
                self.assertEqual(conn.getresponse().status, 404)
                conn.close()
            post(server, '/v1/messages', {'model': 'm', 'stream': False})
            self.assertEqual([r.path for r in server.requests], ['/v1/messages'])
            self.assertEqual([(r.method, r.path) for r in server.port_probes], [('GET', '/'), ('GET', '/')])

    def test_query_and_case_insensitive_headers(self):
        with FakeAnthropicServer() as server:
            post(server, '/v1/messages?beta=true', {'stream': False}, {'X-Api-Key': 'k'})
            req = server.requests[0]
            self.assertEqual(req.path, '/v1/messages?beta=true')
            self.assertEqual((req.route, req.query), ('/v1/messages', {'beta': ['true']}))
            self.assertEqual((req.headers['x-api-key'], req.header('X-API-KEY')), ('k', 'k'))

    def test_gzip_body_decoded(self):
        payload = {'model': 'm', 'stream': False, 'messages': [{'role': 'user', 'content': 'hi'}]}
        with FakeAnthropicServer() as server:
            post(server, '/v1/messages', None, {'Content-Encoding': 'gzip'}, raw=gzip.compress(
                json.dumps(payload).encode()))
            req = server.requests[0]
            self.assertEqual(req.json, payload)
            self.assertEqual(json.loads(req.body), payload)
            self.assertNotEqual(req.raw_body, req.body)
            self.assertEqual(server.last_json(), payload)

    def test_scripted_errors(self):
        cases = [(400, 'invalid_request_error'), (401, 'authentication_error'), (403, 'permission_error'),
                 (404, 'not_found_error'), (413, 'request_too_large'), (429, 'rate_limit_error'),
                 (500, 'api_error'), (529, 'overloaded_error')]
        script = [Step(failure=Failure(status=s, message=f'm{s}', retry_after=7)) for s, _ in cases]
        with FakeAnthropicServer(script) as server:
            for status, kind in cases:
                code, headers, raw = post(server, '/v1/messages', {'stream': True})
                self.assertEqual(code, status)
                self.assertEqual(json.loads(raw), {'type': 'error', 'error': {'type': kind, 'message': f'm{status}'}})
                if status == 429:
                    self.assertEqual(headers['retry-after'], '7')

    def test_stream_failures(self):
        script = [Step(text='abcdefghijklmnopqrstuvwxyz0123456789', failure=Failure('stream_error')),
                  Step(text='partial', failure=Failure('truncated')),
                  Step(tool_calls=[('t', {'k': 'v'})], failure=Failure('malformed_tool_json'))]
        with FakeAnthropicServer(script) as server:
            evs = events(post(server, '/v1/messages', {'stream': True})[2])
            self.assertEqual(evs[-1][0], 'error')
            self.assertEqual(evs[-1][1]['error']['type'], 'overloaded_error')
            self.assertNotIn('message_stop', [e for e, _ in evs])
            evs = events(post(server, '/v1/messages', {'stream': True})[2])
            self.assertNotIn('message_stop', [e for e, _ in evs])
            self.assertNotIn('message_delta', [e for e, _ in evs])
            evs = events(post(server, '/v1/messages', {'stream': True})[2])
            args = ''.join(d['delta']['partial_json'] for e, d in evs if e == 'content_block_delta')
            with self.assertRaises(ValueError):
                json.loads(args)
            self.assertEqual(evs[-1][0], 'message_stop')

    def test_gate_blocks_until_released(self):
        gate = Gate()
        with FakeAnthropicServer([Step(text='late', gate=gate)]) as server:
            result = {}
            thread = threading.Thread(target=lambda: result.update(r=post(server, '/v1/messages',
                                                                         {'stream': False})))
            thread.start()
            self.assertTrue(gate.wait_entered())
            self.assertEqual(len(server.wait_for_requests(1)), 1)
            time.sleep(0.05)
            self.assertNotIn('r', result)
            gate.release()
            thread.join(5)
            self.assertEqual(json.loads(result['r'][2])['content'][0]['text'], 'late')

    def test_exhausted_script_default_and_callable(self):
        def react(req):
            return text_step('saw ' + req.json['messages'][0]['content'])

        with FakeAnthropicServer([react]) as server:
            first = json.loads(post(server, '/v1/messages', {'messages': [{'content': 'x'}]})[2])
            self.assertEqual(first['content'][0]['text'], 'saw x')
            for _ in range(2):
                again = json.loads(post(server, '/v1/messages', {'stream': False})[2])
                self.assertEqual(again['content'][0]['text'], 'done')
            self.assertEqual(len(server.generation_requests()), 3)
            with self.assertRaises(TimeoutError):
                server.wait_for_requests(4, timeout=0.05)

    def test_images_and_tool_results(self):
        body = {'messages': [
            {'role': 'user', 'content': 'go'},
            {'role': 'assistant', 'content': [{'type': 'tool_use', 'id': 'toolu_9', 'name': 'obs', 'input': {}}]},
            {'role': 'user', 'content': [{'type': 'tool_result', 'tool_use_id': 'toolu_9', 'content': [
                {'type': 'text', 'text': 'frame'},
                {'type': 'image', 'source': {'type': 'base64', 'media_type': 'image/png', 'data': PNG_B64}}]}]}]}
        with FakeAnthropicServer() as server:
            post(server, '/v1/messages', body)
            image, = server.images()
            self.assertEqual((image.media_type, image.data), ('image/png', PNG))
            result, = server.find_tool_results()
            self.assertEqual((result.call_id, result.text, result.is_error), ('toolu_9', 'frame', False))
            self.assertEqual(result.images[0].sha256, image.sha256)


class ChatTests(unittest.TestCase):
    def test_streaming_with_usage(self):
        step = Step(text='hello there, friend', tool_calls=[('f', {'q': 'x' * 30}, 'call_1')],
                    usage=Usage(5, 6, cache_read_input_tokens=3, reasoning_tokens=2))
        with FakeChatServer([step]) as server:
            _, _, raw = post(server, '/chat/completions', {'model': 'm', 'stream': True,
                                                          'stream_options': {'include_usage': True}})
        evs = [d for _, d in events(raw)]
        self.assertEqual(evs[-1], '[DONE]')
        usage = evs[-2]
        self.assertEqual(usage['choices'], [])
        self.assertEqual(usage['usage']['prompt_tokens_details']['cached_tokens'], 3)
        self.assertEqual((usage['usage']['prompt_tokens'], usage['usage']['total_tokens']), (8, 14))
        self.assertEqual(usage['usage']['completion_tokens_details']['reasoning_tokens'], 2)
        chunks = evs[:-2]
        self.assertEqual(chunks[-1]['choices'][0]['finish_reason'], 'tool_calls')
        self.assertEqual(chunks[0]['model'], 'm')
        text = ''.join(c['choices'][0]['delta'].get('content') or '' for c in chunks)
        self.assertEqual(text, 'hello there, friend')
        calls = [tc for c in chunks for tc in c['choices'][0]['delta'].get('tool_calls', [])]
        self.assertEqual((calls[0]['id'], calls[0]['type'], calls[0]['function']['name']), ('call_1', 'function', 'f'))
        self.assertEqual(json.loads(''.join(c['function']['arguments'] for c in calls)), {'q': 'x' * 30})

    def test_no_usage_chunk_without_option_and_non_streaming(self):
        with FakeChatServer([text_step('a'), tool_step('g', {'z': 1}, id='c9')]) as server:
            evs = [d for _, d in events(post(server, '/v1/chat/completions', {'stream': True})[2])]
            self.assertEqual(evs[-1], '[DONE]')
            self.assertTrue(all(c['choices'] for c in evs[:-1]))
            body = json.loads(post(server, '/v1/chat/completions', {})[2])
            message = body['choices'][0]['message']
            self.assertEqual(body['choices'][0]['finish_reason'], 'tool_calls')
            self.assertEqual(message['tool_calls'][0], {'id': 'c9', 'type': 'function',
                                                        'function': {'name': 'g', 'arguments': '{"z": 1}'}})

    def test_errors_and_truncation(self):
        script = [Step(failure=Failure(status=429, message='slow', retry_after=2)),
                  Step(text='abcdefghijklmnopqrstuvwxyz', failure=Failure('stream_error', message='boom')),
                  Step(text='abc', failure=Failure('truncated'))]
        with FakeChatServer(script) as server:
            code, headers, raw = post(server, '/v1/chat/completions', {'stream': True})
            self.assertEqual((code, headers['retry-after']), (429, '2'))
            self.assertEqual(json.loads(raw)['error']['type'], 'rate_limit_error')
            evs = [d for _, d in events(post(server, '/v1/chat/completions', {'stream': True})[2])]
            self.assertEqual(evs[-1]['error']['message'], 'boom')
            evs = [d for _, d in events(post(server, '/v1/chat/completions', {'stream': True})[2])]
            self.assertNotIn('[DONE]', evs)

    def test_tool_message_images(self):
        body = {'messages': [{'role': 'user', 'content': 'x'},
                             {'role': 'tool', 'tool_call_id': 'c1', 'content': [
                                 {'type': 'text', 'text': 'shot'},
                                 {'type': 'image_url', 'image_url': {'url': 'data:image/png;base64,' + PNG_B64}}]},
                             {'role': 'tool', 'tool_call_id': 'c2', 'content': 'plain'}]}
        with FakeChatServer() as server:
            post(server, '/v1/chat/completions', body)
            first, second = server.find_tool_results()
            self.assertEqual((first.call_id, first.text, second.text), ('c1', 'shot', 'plain'))
            self.assertEqual(server.images()[0].data, PNG)


class ResponsesTests(unittest.TestCase):
    def test_streaming(self):
        step = Step(thinking='t', text='some answer text', tool_calls=[('shell', {'cmd': 'ls -la /tmp'}, 'call_7')],
                    usage=Usage(4, 6, cache_read_input_tokens=2, reasoning_tokens=1))
        with FakeResponsesServer([step]) as server:
            _, _, raw = post(server, '/v1/responses', {'model': 'm', 'stream': True})
        evs = events(raw)
        kinds = [e for e, _ in evs]
        self.assertEqual(kinds[0], 'response.created')
        self.assertEqual(kinds[-1], 'response.completed')
        self.assertEqual([d['type'] for _, d in evs], kinds)
        self.assertLess(kinds.index('response.output_item.added'), kinds.index('response.output_text.delta'))
        self.assertLess(kinds.index('response.output_text.delta'), kinds.index('response.function_call_arguments.delta'))
        text = ''.join(d['delta'] for e, d in evs if e == 'response.output_text.delta')
        self.assertEqual(text, 'some answer text')
        args = ''.join(d['delta'] for e, d in evs if e == 'response.function_call_arguments.delta')
        self.assertEqual(json.loads(args), {'cmd': 'ls -la /tmp'})
        done, = [d for e, d in evs if e == 'response.function_call_arguments.done']
        self.assertEqual(json.loads(done['arguments']), {'cmd': 'ls -la /tmp'})
        response = evs[-1][1]['response']
        self.assertEqual((response['status'], response['model']), ('completed', 'm'))
        self.assertTrue(response['id'])
        self.assertEqual(response['usage']['input_tokens_details']['cached_tokens'], 2)
        self.assertEqual(response['usage']['output_tokens_details']['reasoning_tokens'], 1)
        call, = [i for i in response['output'] if i['type'] == 'function_call']
        self.assertEqual((call['call_id'], call['name']), ('call_7', 'shell'))

    def test_non_streaming_terminals_and_compact(self):
        script = [text_step('hi'), Step(text='cut', stop_reason='max_tokens'),
                  Step(failure=Failure('failed', message='nope')), Step(text='x', failure=Failure('incomplete'))]
        with FakeResponsesServer(script) as server:
            self.assertEqual(json.loads(post(server, '/responses', {})[2])['status'], 'completed')
            self.assertEqual(json.loads(post(server, '/responses', {})[2])['status'], 'incomplete')
            failed = events(post(server, '/responses', {'stream': True})[2])
            self.assertEqual(failed[-1][0], 'response.failed')
            self.assertEqual(failed[-1][1]['response']['output'], [])
            self.assertEqual(events(post(server, '/v1/responses', {'stream': True})[2])[-1][0],
                             'response.incomplete')
            compact = json.loads(post(server, '/responses/compact', {})[2])
            self.assertEqual(compact['object'], 'response.compaction')
            self.assertEqual(len(server.generation_requests()), 4)
            self.assertEqual(len(server.requests), 5)

    def test_errors_and_stream_error(self):
        script = [Step(failure=Failure(status=529, message='busy')), Step(text='abcdefghijklmnop' * 3,
                                                                          failure=Failure('stream_error'))]
        with FakeResponsesServer(script) as server:
            code, _, raw = post(server, '/responses', {'stream': True})
            self.assertEqual(code, 529)
            self.assertEqual(json.loads(raw)['error']['message'], 'busy')
            evs = events(post(server, '/responses', {'stream': True})[2])
            self.assertEqual(evs[-1][0], 'error')
            self.assertNotIn('response.completed', [e for e, _ in evs])

    def test_function_call_output_and_images(self):
        body = {'input': [{'type': 'function_call_output', 'call_id': 'call_1', 'output': 'text out'},
                          {'type': 'function_call_output', 'call_id': 'call_2', 'output': [
                              {'type': 'input_text', 'text': 'see'},
                              {'type': 'input_image', 'image_url': 'data:image/png;base64,' + PNG_B64}]},
                          {'type': 'message', 'role': 'user', 'content': [
                              {'type': 'input_image', 'image_url': 'https://example.invalid/x.png'}]}]}
        with FakeResponsesServer() as server:
            post(server, '/responses', body)
            one, two = server.find_tool_results()
            self.assertEqual((one.call_id, one.text, two.text), ('call_1', 'text out', 'see'))
            self.assertEqual([i.data for i in server.images()], [PNG])
            self.assertEqual(two.images[0].media_type, 'image/png')


class McpStub:
    """Tiny MCP server: initialize / tools/list / tools/call (``echo`` JSON or SSE with progress, ``block`` waits)."""

    def __init__(self, token='tok'):
        self.token, self.received = token, []
        self.in_call, self.release, self.disconnected = threading.Event(), threading.Event(), threading.Event()
        stub = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *args):
                pass

            def do_GET(self):
                self.send_response(405)
                self.send_header('Content-Length', '0')
                self.end_headers()

            do_DELETE = do_GET

            def do_POST(self):
                body = json.loads(self.rfile.read(int(self.headers['Content-Length'])))
                stub.received.append((dict(self.headers), body))
                if self.headers.get('Authorization') != f'Bearer {stub.token}':
                    return self.reply(401, {'error': 'unauthorized'})
                method, rid = body.get('method'), body.get('id')
                if rid is None:
                    self.send_response(202)
                    self.send_header('Content-Length', '0')
                    return self.end_headers()
                if method == 'initialize':
                    result = {'protocolVersion': body['params']['protocolVersion'],
                              'capabilities': {'tools': {'listChanged': False}},
                              'serverInfo': {'name': 'stub', 'version': '1'}}
                elif method == 'ping':
                    result = {}
                elif method == 'tools/list':
                    result = {'tools': [{'name': 'echo', 'description': 'd', 'inputSchema': {'type': 'object'}}]}
                elif method == 'tools/call':
                    return self.call(rid, body['params'])
                else:
                    return self.reply(200, {'jsonrpc': '2.0', 'id': rid,
                                            'error': {'code': -32601, 'message': 'no such method'}})
                self.reply(200, {'jsonrpc': '2.0', 'id': rid, 'result': result})

            def reply(self, status, message):
                data = json.dumps(message).encode()
                self.send_response(status)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def call(self, rid, params):
                result = {'content': [{'type': 'text', 'text': json.dumps(params['arguments'])}], 'isError': False}
                final = {'jsonrpc': '2.0', 'id': rid, 'result': result}
                if 'text/event-stream' not in self.headers.get('Accept', ''):
                    return self.reply(200, final)
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Connection', 'close')
                self.end_headers()
                token = (params.get('_meta') or {}).get('progressToken')
                if token is not None:
                    note = {'jsonrpc': '2.0', 'method': 'notifications/progress',
                            'params': {'progressToken': token, 'progress': 1}}
                    self.wfile.write(b': keepalive\n\n' + f'event: message\ndata: {json.dumps(note)}\n\n'.encode())
                self.wfile.flush()
                if params['name'] == 'block':
                    stub.in_call.set()
                    while not stub.release.is_set():
                        readable, _, _ = select.select([self.connection], [], [], 0.01)
                        try:
                            gone = bool(readable) and not self.connection.recv(1, socket.MSG_PEEK)
                        except OSError:  # RST: the client closed with unread data
                            gone = True
                        if gone:
                            stub.disconnected.set()
                            return
                self.wfile.write(f'event: message\ndata: {json.dumps(final)}\n\n'.encode())
                self.close_connection = True

        class Httpd(ThreadingHTTPServer):
            daemon_threads = True

        self.httpd = Httpd(('127.0.0.1', 0), Handler)
        threading.Thread(target=self.httpd.serve_forever, kwargs={'poll_interval': 0.01}, daemon=True).start()
        self.url = f'http://127.0.0.1:{self.httpd.server_address[1]}/mcp'

    def close(self):
        self.release.set()
        self.httpd.shutdown()
        self.httpd.server_close()


class McpClientTests(unittest.TestCase):
    def setUp(self):
        self.stub = McpStub()
        self.addCleanup(self.stub.close)
        self.client = FakeMcpClient(self.stub.url, token='tok')

    def test_handshake_ping_list(self):
        init = self.client.handshake()
        self.assertEqual(init['protocolVersion'], '2025-06-18')
        first = self.stub.received[0][1]
        self.assertEqual(first['params']['clientInfo'], {'name': 'fake-mcp-client', 'version': '0.0.0'})
        self.assertEqual(self.stub.received[1][1], {'jsonrpc': '2.0', 'method': 'notifications/initialized'})
        self.assertEqual(self.client.last_response.status, 202)
        self.assertEqual(self.client.ping(), {})
        self.assertEqual(self.client.list_tools()[0]['name'], 'echo')
        self.assertEqual(self.stub.received[0][0]['Authorization'], 'Bearer tok')
        self.assertEqual(self.client.http('GET').status, 405)
        with self.assertRaises(McpError) as ctx:
            self.client._call('nope')
        self.assertEqual(ctx.exception.error['code'], -32601)

    def test_bad_token(self):
        res = FakeMcpClient(self.stub.url, token='wrong').call_tool('echo', {})
        self.assertEqual((res.status, res.result, res.message), (401, None, None))

    def test_call_json_and_sse(self):
        res = self.client.call_tool('echo', {'a': 1}, accept_sse=False)
        self.assertFalse(res.is_sse)
        self.assertEqual(json.loads(res.result['content'][0]['text']), {'a': 1})
        self.assertEqual(res.progress, [])
        self.assertEqual(self.stub.received[-1][0]['Accept'], 'application/json')
        res = self.client.call_tool('echo', {'b': 2}, progress_token='p1', meta={'claudecode/toolUseId': 'toolu_1'})
        self.assertTrue(res.is_sse)
        self.assertEqual(res.progress, [{'progressToken': 'p1', 'progress': 1}])
        self.assertEqual(json.loads(res.result['content'][0]['text']), {'b': 2})
        self.assertEqual(res.response.frames[0]['comments'], ['keepalive'])
        self.assertEqual(self.stub.received[-1][1]['params']['_meta'],
                         {'claudecode/toolUseId': 'toolu_1', 'progressToken': 'p1'})
        self.assertIn(b'notifications/progress', res.response.raw)
        self.assertEqual(self.client.exchanges[-1].request_headers['Accept'], 'application/json, text/event-stream')

    def test_cancel_while_running(self):
        pending = self.client.start_call('block', {'k': 1}, progress_token=5)
        self.assertTrue(self.stub.in_call.wait(5))
        ack = self.client.cancel(pending.request_id, 'user')
        self.assertEqual(ack.status, 202)
        self.assertEqual(self.stub.received[-1][1], {'jsonrpc': '2.0', 'method': 'notifications/cancelled',
                                                     'params': {'requestId': pending.request_id, 'reason': 'user'}})
        self.assertFalse(pending.done())
        self.stub.release.set()
        res = pending.wait(5)
        self.assertEqual(res.request_id, pending.request_id)
        self.assertFalse(res.result['isError'])

    def test_disconnect_mid_call(self):
        request_id = self.client.call_tool_and_disconnect('block', {}, progress_token='p')
        self.assertTrue(self.stub.in_call.wait(5))
        self.assertTrue(self.stub.disconnected.wait(5))
        self.assertEqual(self.stub.received[-1][1]['id'], request_id)


if __name__ == '__main__':
    unittest.main()
