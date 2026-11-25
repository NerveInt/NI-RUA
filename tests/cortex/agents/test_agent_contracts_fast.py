"""Merged cortex tests:
- tests/test_agent_config.py: agent.* configuration: defaults, omission, round trip, validation and the --agent-backend flag.
- tests/test_agent_dispatch.py: OrderedBarrier / ToolDispatcher ordering semantics and EmittedOrder (the model's emitted order of one step's
- tests/test_conformance_deny_proxy.py: ``agent_conformance.DenyAllProxy``: only a probe-shaped request (review 29 should-fix 4) is kept out of ``attempts``.
- tests/test_cortex_layering.py: AST layering boundaries and Stage 5 cortex/relay/agent-registry invariants.
- tests/test_operation_context.py: Retirement of legacy structured operation_context/operation_candidates and read-only archive inspection.
- tests/test_public_task_rules.py: Public documentation delivery and frozen provenance, without a simulator.
- tests/test_reasoning_options.py: Generated reasoning controls (``ni_rua.cortex.agents.reasoning_options``; design 16 §7.1, WP-B).
- tests/test_recovery_guidance.py: Recovery advice neither invents a budget nor reads task/evaluator truth.
- tests/test_system_runtime.py: Runtime grants exclude system-installed private applications and sources.
- tests/test_terminal.py: Run with python -m unittest discover -s tests -p test_terminal.py.
"""

from __future__ import annotations

NI_RUA_TEST_TIER = "fast"

import ast
import importlib
import importlib.util
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import AsyncMock, patch
from ni_rua.core.config import (AgentConfig, ClaudeCodeConfig, Config, ConfigError, OpenCodeConfig,
                                UpstreamConfig, cli_overrides, namespace_values)
from concurrent.futures import CancelledError, ThreadPoolExecutor
import threading
import time
from ni_rua.cortex.agents.dispatch import (UNMATCHED_REASONS, EmittedOrder, OrderedBarrier, ToolDispatcher,
                                           is_parallel_call)
import http.client
from support.agent_conformance import DenyAllProxy
from copy import deepcopy
from unittest.mock import Mock
from ni_rua.core.storage.store import Store
from ni_rua.core.tool_errors import error_payload
from ni_rua.cortex.context.task_notes import default_notes_for, resolve_task_notes, version_for
from ni_rua.cortex.context.task_planning import (
    LEGACY_PLAN_SCHEMA,
    PLAN_SCHEMA,
    PLANNING_VERSION,
    TaskPlanning,
    _plan_schema_for_error,
    scheduling_failure_advisory,
)
import hashlib
import gzip
from unittest.mock import patch
from ni_rua.cortex.context.public_task_rules import actor_task_rules_text, public_task_rules
from scripts.extract_robodojo_task_rules import CATALOG, SNAPSHOT, extract_sections, rebuild
from support.agent_fixtures import fixture_workbench
from unittest import mock
from ni_rua.core.config import ModelConfig
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.base import AgentCapabilities, model_effort_choices
from ni_rua.cortex.agents.claude_code.session import ClaudeCodeSession
from ni_rua.cortex.agents.model_catalogue import CatalogueModel, parse_models_page
from ni_rua.cortex.agents.opencode import config as opencode_config
from ni_rua.cortex.agents.reasoning_options import SERVICE_TIERS, backend_traits, reasoning_options
from types import SimpleNamespace
from ni_rua.cortex.supervision.recovery_guidance import NEXT_STEP_INSTRUCTIONS, recovery_guidance
from ni_rua.cortex.session.system_runtime import (runtime_library_mounts,
                                     stdlib_runtime_entries, system_runtime_mounts)
from support.host_capabilities import linux_runtime_tools, require
import asyncio
import errno
import os
import pty
import select
import tty
from ni_rua.cortex.session.terminal import Terminal, terminal_bootstrap_status
from support.paths import isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


# ==== tests/test_agent_config.py ====

class AgentConfigTests(unittest.TestCase):
    def test_default_is_codex_and_omitted_from_snapshots(self):
        config = Config()
        self.assertEqual(config.agent, AgentConfig())
        self.assertEqual(config.agent.backend, 'codex')
        self.assertNotIn('agent', config.to_dict())
        self.assertEqual(Config.from_dict({}).to_dict().keys(), config.to_dict().keys())
        self.assertEqual(Config().to_dict(), Config(agent=AgentConfig()).to_dict())
        self.assertEqual(Config.from_dict(config.to_dict()).agent, AgentConfig())

    def test_claude_code_defaults_to_the_anthropic_upstream(self):
        upstream = Config().agent.claude_code.upstream
        self.assertEqual((upstream.wire_api, upstream.base_url, upstream.api_key_env, upstream.auth_scheme, upstream.provider),
                         ('anthropic_messages', 'https://api.anthropic.com', 'ANTHROPIC_API_KEY', 'x-api-key', 'anthropic'))
        self.assertIsNone(Config().agent.claude_code.cli_path)
        # null = resolved per declared model at start (catalogue, else 200000 / 32000; opencode.config.resolve_limits)
        self.assertIsNone(Config().agent.opencode.context_limit_tokens)
        self.assertIsNone(Config().agent.opencode.output_limit_tokens)
        self.assertIsNone(Config().agent.opencode.thinking_display)

    def test_partial_agent_document_inherits_backend_defaults(self):
        config = Config.from_dict({'agent': {'backend': 'claude_code',
                                             'claude_code': {'upstream': {'api_key_env': 'MY_KEY'}}}})
        upstream = config.agent.claude_code.upstream
        self.assertEqual(config.agent.backend, 'claude_code')
        self.assertEqual((upstream.api_key_env, upstream.base_url, upstream.auth_scheme), ('MY_KEY', 'https://api.anthropic.com', 'x-api-key'))

    def test_non_default_round_trip(self):
        document = {'agent': {'backend': 'opencode', 'claude_code': {'subagents': True, 'auto_compact_window_tokens': 200000,
                              'context_delivery': 'full'},
                    'opencode': {'executable': '/opt/opencode/bin/opencode', 'context_limit_tokens': 400000,
                                 'context_delivery': 'full',
                                 'upstream': {'wire_api': 'openai_chat', 'base_url': 'http://127.0.0.1:9000/v1',
                                              'api_key_env': 'UPSTREAM_KEY', 'auth_scheme': 'bearer', 'provider': 'local'}},
                    'external': {'context_delivery': 'full'}}}
        config = Config.from_dict(document)
        exported = config.to_dict()
        self.assertEqual(exported['agent']['opencode']['executable'], '/opt/opencode/bin/opencode')
        self.assertEqual(exported['agent']['claude_code']['context_delivery'], 'full')
        self.assertEqual(exported['agent']['opencode']['context_delivery'], 'full')
        self.assertEqual(exported['agent']['external']['context_delivery'], 'full')
        self.assertEqual(Config.from_dict(exported), config)
        self.assertEqual(Config.from_dict(json.loads(json.dumps(exported))).to_dict(), exported)
        self.assertEqual(config.with_overrides({'agent': {'backend': 'codex'}}).agent.opencode, config.agent.opencode)

    def test_relative_paths_resolve_next_to_the_config_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'c.json'
            path.write_text(json.dumps({'agent': {'claude_code': {'cli_path': 'bin/claude'}}}))
            config = Config.load(path, environ={})
            self.assertEqual(config.agent.claude_code.cli_path, str((Path(directory) / 'bin/claude').resolve()))

    def test_validation_errors(self):
        for agent, message in (
                ({'backend': 'gemini'}, 'invalid_config_choice:AgentConfig.backend'),
                ({'claude_code': {'context_delivery': 'both'}}, 'invalid_config_choice:ClaudeCodeConfig.context_delivery'),
                ({'opencode': {'context_delivery': 'both'}}, 'invalid_config_choice:OpenCodeConfig.context_delivery'),
                ({'external': {'context_delivery': 'both'}}, 'invalid_config_choice:ExternalConfig.context_delivery'),
                ({'claude_code': {'auto_compact_window_tokens': 99999}}, 'config_below_minimum'),
                ({'claude_code': {'auto_compact_window_tokens': 1000001}}, 'config_above_maximum'),
                ({'opencode': {'context_limit_tokens': 0}}, 'config_below_minimum'),
                ({'opencode': {'upstream': {'wire_api': 'grpc'}}}, 'invalid_config_choice:UpstreamConfig.wire_api'),
                ({'opencode': {'upstream': {'auth_scheme': 'basic'}}}, 'invalid_config_choice:UpstreamConfig.auth_scheme'),
                ({'opencode': {'upstream': {'source': 'env'}}}, 'invalid_config_choice:UpstreamConfig.source'),
                ({'claude_code': {'upstream': {'base_url': 'https://user:pw@api.example.com'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'base_url': 'https://api.example.com/v1?key=abc'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'base_url': 'https://api.example.com/#frag'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'base_url': 'ftp://api.example.com'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'base_url': 'api.example.com'}}}, 'invalid_upstream_base_url'),
                ({'claude_code': {'upstream': {'api_key_env': 'sk-ant-abc123 secret'}}}, 'invalid_upstream_api_key_env'),
                ({'claude_code': {'upstream': {'api_key_env': 'sk-ant-abc'}}}, 'invalid_upstream_api_key_env'),
                ({'claude_code': {'subagents': 'yes'}}, 'invalid_config_type'),
                ({'unknown_key': 1}, 'unknown_config_keys'),
                ({'claude_code': {'upstream': {'api_key': 'secret'}}}, 'unknown_config_keys')):
            with self.subTest(agent=agent), self.assertRaisesRegex(ConfigError, message):
                Config.from_dict({'agent': agent})

    def test_opencode_limits_and_thinking_display(self):
        messages = {'wire_api': 'anthropic_messages', 'base_url': 'http://127.0.0.1:9', 'api_key_env': 'K'}
        config = Config.from_dict({'agent': {'opencode': {'context_limit_tokens': 1000000, 'output_limit_tokens': 7777,
                                                          'thinking_display': 'summarized', 'upstream': messages}}})
        opencode = config.agent.opencode
        self.assertEqual((opencode.context_limit_tokens, opencode.output_limit_tokens, opencode.thinking_display),
                         (1000000, 7777, 'summarized'))
        self.assertEqual(Config.from_dict(config.to_dict()), config)
        self.assertIsNone(Config.from_dict({'agent': {'opencode': {'context_limit_tokens': None}}})
                          .agent.opencode.context_limit_tokens)
        self.assertEqual(OpenCodeConfig(upstream=UpstreamConfig(**messages), thinking_display='omitted').thinking_display,
                         'omitted')
        for opencode, message in (
                ({'output_limit_tokens': 0}, 'config_below_minimum:OpenCodeConfig.output_limit_tokens'),
                ({'output_limit_tokens': 1.5}, 'invalid_config_type:OpenCodeConfig.output_limit_tokens'),
                ({'thinking_display': 'full', 'upstream': messages},
                 'invalid_config_choice:OpenCodeConfig.thinking_display'),
                # display is an Anthropic Messages field: refused on every other (or an unset) wire, never ignored
                ({'thinking_display': 'summarized'}, '^opencode_thinking_display_requires_anthropic_messages'),
                ({'thinking_display': 'summarized', 'upstream': {'source': 'codex_home'}},
                 '^opencode_thinking_display_requires_anthropic_messages'),
                ({'thinking_display': 'omitted', 'upstream': dict(messages, wire_api='openai_responses')},
                 '^opencode_thinking_display_requires_anthropic_messages')):
            with self.subTest(opencode=opencode), self.assertRaisesRegex(ConfigError, message):
                Config.from_dict({'agent': {'opencode': opencode}})

    def test_section_constructors_validate_directly(self):
        with self.assertRaises(ConfigError):
            UpstreamConfig(base_url='https://host/path?token=1')
        self.assertEqual(UpstreamConfig(base_url='http://127.0.0.1:8080/v1').api_key_env, None)
        self.assertIsInstance(ClaudeCodeConfig().upstream, UpstreamConfig)
        self.assertEqual(OpenCodeConfig().upstream, UpstreamConfig())

    def test_cli_flag_maps_to_agent_backend_only(self):
        self.assertEqual(cli_overrides({'agent_backend': 'claude_code'}), {'agent': {'backend': 'claude_code'}})
        config = Config.load(None, environ={}, overrides=cli_overrides({'agent_backend': 'opencode'}))
        self.assertEqual(config.agent.backend, 'opencode')
        self.assertEqual(namespace_values(config)['agent_backend'], 'opencode')
        self.assertEqual(namespace_values(Config())['agent_backend'], 'codex')

    def test_gateway_parser_accepts_the_flag_and_rejects_other_values(self):
        from ni_rua.apps.gateway import cli as gateway
        with patch.object(gateway, 'serve', new_callable=AsyncMock) as serve, \
                patch.object(sys, 'argv', ['gateway', '--agent-backend', 'claude_code']):
            gateway.main()
        self.assertEqual(serve.call_args.args[0].agent_backend, 'claude_code')
        self.assertEqual(serve.call_args.args[0].resolved_config.agent.backend, 'claude_code')
        with patch.object(gateway, 'serve', new_callable=AsyncMock) as serve, \
                patch.object(sys, 'argv', ['gateway']):
            gateway.main()
        self.assertEqual(serve.call_args.args[0].agent_backend, 'codex')
        with patch.object(sys, 'argv', ['gateway', '--agent-backend', 'gemini']), self.assertRaises(SystemExit):
            gateway.main()

# ==== tests/test_agent_dispatch.py ====

TIMEOUT = 5


class Recorder:
    """Records start/end order and the peak number of concurrently running calls."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = []
        self.running = 0
        self.peak = 0

    def call(self, name, gate=None, error=None):
        with self.lock:
            self.events.append(('start', name))
            self.running += 1
            self.peak = max(self.peak, self.running)
        try:
            if gate is not None:
                gate()
            if error is not None:
                raise error
            return name
        finally:
            with self.lock:
                self.running -= 1
                self.events.append(('end', name))

    def index(self, kind, name):
        return self.events.index((kind, name))


class OrderedBarrierTest(unittest.TestCase):
    def setUp(self):
        self.pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix='barrier-test')
        self.addCleanup(self.pool.shutdown, wait=True, cancel_futures=True)
        self.barrier = OrderedBarrier()
        self.recorder = Recorder()

    def schedule(self, parallel, name, **kwargs):
        return self.barrier.schedule(self.pool, parallel, self.recorder.call, name, kwargs.get('gate'),
                                     kwargs.get('error'))

    def test_two_parallel_calls_overlap(self):
        rendezvous = threading.Barrier(2, timeout=TIMEOUT)
        first = self.schedule(True, 'read-1', gate=rendezvous.wait)
        second = self.schedule(True, 'read-2', gate=rendezvous.wait)
        # Each call blocks until the other is running: completion proves overlap.
        self.assertEqual([first.result(TIMEOUT), second.result(TIMEOUT)], ['read-1', 'read-2'])
        self.assertEqual(self.recorder.peak, 2)

    def test_ordered_call_waits_for_every_earlier_parallel_call(self):
        release = threading.Event()
        both_running = threading.Barrier(3, timeout=TIMEOUT)

        def hold():
            both_running.wait()
            release.wait(TIMEOUT)
        reads = [self.schedule(True, 'read-1', gate=hold), self.schedule(True, 'read-2', gate=hold)]
        write = self.schedule(False, 'write')
        both_running.wait()
        time.sleep(0.1)
        self.assertNotIn(('start', 'write'), self.recorder.events)
        release.set()
        self.assertEqual(write.result(TIMEOUT), 'write')
        for read in reads:
            read.result(TIMEOUT)
        start = self.recorder.index('start', 'write')
        self.assertLess(self.recorder.index('end', 'read-1'), start)
        self.assertLess(self.recorder.index('end', 'read-2'), start)

    def test_parallel_call_waits_for_previous_ordered_call(self):
        release = threading.Event()
        started = threading.Event()

        def hold():
            started.set()
            release.wait(TIMEOUT)
        write = self.schedule(False, 'write', gate=hold)
        read = self.schedule(True, 'read')
        self.assertTrue(started.wait(TIMEOUT))
        time.sleep(0.1)
        self.assertNotIn(('start', 'read'), self.recorder.events)
        release.set()
        read.result(TIMEOUT)
        write.result(TIMEOUT)
        self.assertLess(self.recorder.index('end', 'write'), self.recorder.index('start', 'read'))

    def test_arrival_order_is_execution_order_for_ordered_calls(self):
        # Earlier calls sleep longer; without the barrier later calls would finish first on 8 workers.
        names = ['call-%d' % index for index in range(8)]
        futures = [self.schedule(False, name, gate=lambda delay=(8 - index) * 0.01: time.sleep(delay))
                   for index, name in enumerate(names)]
        self.assertEqual([future.result(TIMEOUT) for future in futures], names)
        self.assertEqual(self.recorder.events, [event for name in names for event in (('start', name), ('end', name))])
        self.assertEqual(self.recorder.peak, 1)

    def test_mixed_sequence_respects_both_barriers(self):
        plan = [(False, 'o1'), (True, 'p1'), (True, 'p2'), (False, 'o2'), (True, 'p3'), (False, 'o3')]
        futures = [self.schedule(parallel, name, gate=lambda: time.sleep(0.02)) for parallel, name in plan]
        for future in futures:
            future.result(TIMEOUT)
        before = self.recorder.index
        self.assertLess(before('end', 'o1'), before('start', 'p1'))
        self.assertLess(before('end', 'o1'), before('start', 'p2'))
        self.assertLess(max(before('end', 'p1'), before('end', 'p2')), before('start', 'o2'))
        self.assertLess(before('end', 'o2'), before('start', 'p3'))
        self.assertLess(before('end', 'p3'), before('start', 'o3'))

    def test_exceptions_propagate_without_breaking_the_chain(self):
        failed_write = self.schedule(False, 'write-1', error=ValueError('fixture write failure'))
        failed_read = self.schedule(True, 'read-1', error=RuntimeError('fixture read failure'))
        read = self.schedule(True, 'read-2')
        write = self.schedule(False, 'write-2')
        with self.assertRaisesRegex(ValueError, 'fixture write failure'):
            failed_write.result(TIMEOUT)
        with self.assertRaisesRegex(RuntimeError, 'fixture read failure'):
            failed_read.result(TIMEOUT)
        self.assertEqual(read.result(TIMEOUT), 'read-2')
        self.assertEqual(write.result(TIMEOUT), 'write-2')
        self.assertLess(self.recorder.index('end', 'read-1'), self.recorder.index('start', 'write-2'))

    def test_non_true_classification_is_ordered(self):
        rendezvous = threading.Barrier(2, timeout=0.3)
        first = self.schedule(1, 'call-1', gate=rendezvous.wait)
        second = self.schedule('yes', 'call-2', gate=rendezvous.wait)
        # Ordered calls never overlap, so neither can pass the two-party rendezvous.
        for future in (first, second):
            with self.assertRaises(threading.BrokenBarrierError):
                future.result(TIMEOUT)
        self.assertEqual(self.recorder.peak, 1)

    def test_submit_error_propagates_to_the_caller(self):
        self.pool.shutdown(wait=True)
        with self.assertRaises(RuntimeError):
            self.schedule(False, 'late')
        with self.assertRaises(RuntimeError):
            self.schedule(True, 'late-read')


class IsParallelCallTest(unittest.TestCase):
    def test_only_an_explicit_true_verdict_is_parallel(self):
        seen = []

        def classify(tool, arguments):
            seen.append((tool, arguments))
            return tool == 'reader'
        self.assertTrue(is_parallel_call(classify, {'tool': 'reader', 'arguments': {'a': 1}}))
        self.assertFalse(is_parallel_call(classify, {'tool': 'writer', 'arguments': {}}))
        self.assertEqual(seen, [('reader', {'a': 1}), ('writer', {})])
        self.assertFalse(is_parallel_call(None, {'tool': 'reader', 'arguments': {}}))
        for params in (None, [], 'reader'):
            self.assertFalse(is_parallel_call(classify, params))
        self.assertFalse(is_parallel_call(lambda tool, arguments: 1, {'tool': 'reader'}))
        self.assertFalse(is_parallel_call(lambda tool, arguments: 1 / 0, {'tool': 'reader'}))


class ToolDispatcherTest(unittest.TestCase):
    def test_handler_runs_on_named_workers_with_registry_classification(self):
        rendezvous = threading.Barrier(2, timeout=TIMEOUT)
        threads = []

        def handler(params):
            threads.append(threading.current_thread().name)
            if params['tool'] == 'reader':
                rendezvous.wait()
            return dict(tool=params['tool'], arguments=params['arguments'])
        dispatcher = ToolDispatcher(handler, lambda tool, arguments: tool == 'reader', thread_name_prefix='fixture-tools')
        try:
            futures = [dispatcher.submit(dict(tool='reader', arguments={'n': index})) for index in range(2)]
            futures.append(dispatcher.submit(dict(tool='writer', arguments={})))
            self.assertEqual([future.result(TIMEOUT) for future in futures],
                             [dict(tool='reader', arguments={'n': 0}), dict(tool='reader', arguments={'n': 1}),
                              dict(tool='writer', arguments={})])
        finally:
            dispatcher.shutdown()
        self.assertTrue(all(name.startswith('fixture-tools') for name in threads))
        self.assertNotIn(threading.current_thread().name, threads)

    def test_without_classifier_every_call_is_ordered(self):
        recorder = Recorder()
        dispatcher = ToolDispatcher(lambda params: recorder.call(params['tool'], lambda: time.sleep(0.02)), None)
        try:
            futures = [dispatcher.submit(dict(tool='call-%d' % index, arguments={})) for index in range(4)]
            self.assertEqual([future.result(TIMEOUT) for future in futures], ['call-%d' % index for index in range(4)])
        finally:
            dispatcher.shutdown()
        self.assertEqual(recorder.peak, 1)

    def test_classifier_failure_is_ordered_and_handler_error_reaches_the_future(self):
        def classify(tool, arguments):
            raise RuntimeError('fixture classifier failure')

        def handler(params):
            if params['tool'] == 'broken':
                raise ValueError('fixture handler failure')
            return params['tool']
        dispatcher = ToolDispatcher(handler, classify, max_workers=2)
        try:
            broken = dispatcher.submit(dict(tool='broken', arguments={}))
            later = dispatcher.submit(dict(tool='later', arguments={}))
            with self.assertRaisesRegex(ValueError, 'fixture handler failure'):
                broken.result(TIMEOUT)
            self.assertEqual(later.result(TIMEOUT), 'later')
        finally:
            dispatcher.shutdown()

    def test_shutdown_cancels_queued_calls_and_rejects_new_ones(self):
        started, release = threading.Event(), threading.Event()

        def handler(params):
            if params['tool'] == 'running':
                started.set()
                release.wait(TIMEOUT)
            return params['tool']
        dispatcher = ToolDispatcher(handler, None, max_workers=1)
        running = dispatcher.submit(dict(tool='running', arguments={}))
        queued = dispatcher.submit(dict(tool='queued', arguments={}))
        self.assertTrue(started.wait(TIMEOUT))
        dispatcher.shutdown(wait=False, cancel_futures=True)
        self.assertTrue(queued.cancelled())
        with self.assertRaises(CancelledError):
            queued.result(TIMEOUT)
        with self.assertRaises(RuntimeError):
            dispatcher.submit(dict(tool='late', arguments={}))
        release.set()
        self.assertEqual(running.result(TIMEOUT), 'running')
        dispatcher.shutdown(wait=True)


STEP = [("toolu_a", "move", {"x": 1}), ("toolu_b", "move", {"x": 2}), ("toolu_c", "move", {"x": 3})]


class EmittedOrderTests(unittest.TestCase):
    """Calls of one announced model step are released in emitted order whatever order they arrive in."""

    def run_calls(self, order, arrivals, *, dispatcher=None):
        """Start one thread per ``(label, claim kwargs)`` in ``arrivals`` order, each started only after the previous
        one is held by ``order`` or has finished (so the arrival order is exactly ``arrivals``); returns the labels
        in the order their calls were released, and the tickets by label."""
        released, tickets, lock = [], {}, threading.Lock()

        def call(label, tool, arguments, identity):
            ticket, _ = order.claim(tool, arguments, **identity)
            order.wait(ticket)
            with lock:
                released.append(label)
                tickets[label] = ticket
            order.resolve(ticket)
        threads = []
        for label, tool, arguments, identity in arrivals:
            held, done = order.held, len(released)
            thread = threading.Thread(target=call, args=(label, tool, arguments, identity), daemon=True)
            thread.start()
            threads.append(thread)
            deadline = time.monotonic() + TIMEOUT
            while order.held == held and len(released) == done and time.monotonic() < deadline:
                time.sleep(0.001)
        for thread in threads:
            thread.join(TIMEOUT)
            self.assertFalse(thread.is_alive())
        return released, tickets

    def test_inverted_arrival_is_released_in_emitted_order_by_native_call_id(self):
        order = EmittedOrder(wait_s=TIMEOUT)
        order.announce("thread", STEP)
        released, tickets = self.run_calls(order, [
            ("c", "move", {"x": 3}, dict(native_call_id="toolu_c")),
            ("b", "move", {"x": 2}, dict(native_call_id="toolu_b")),
            ("a", "move", {"x": 1}, dict(native_call_id="toolu_a"))])
        self.assertEqual(released, ["a", "b", "c"])
        self.assertEqual([(tickets[k].position, tickets[k].matched_by, tickets[k].expired) for k in "abc"],
                         [(0, "native_call_id", False), (1, "native_call_id", False), (2, "native_call_id", False)])
        self.assertEqual(order.held, 0)

    def test_inverted_arrival_without_ids_is_correlated_by_thread_tool_and_arguments(self):
        order = EmittedOrder(wait_s=TIMEOUT)
        order.announce("other-session", [(None, "move", {"x": 3})])      # another session's newer step: not ours
        order.announce("ses_1", [(None, tool, arguments) for _, tool, arguments in STEP])
        order.announce("other-session", [(None, "move", {"x": 2})])
        released, tickets = self.run_calls(order, [
            ("c", "move", {"x": 3}, dict(thread_id="ses_1")), ("b", "move", {"x": 2.0}, dict(thread_id="ses_1")),
            ("a", "move", {"x": 1}, dict(thread_id="ses_1"))])
        self.assertEqual(released, ["a", "b", "c"])
        self.assertEqual({ticket.matched_by for ticket in tickets.values()}, {"arguments"})
        self.assertEqual([tickets[k].position for k in "abc"], [0, 1, 2])

    def test_equal_calls_are_interchangeable_and_changed_arguments_have_no_place(self):
        order = EmittedOrder(wait_s=TIMEOUT)
        order.announce("ses_1", [(None, "move", {"x": 1}), (None, "move", {"x": 1}), (None, "look", {})])
        first, _ = order.claim("move", {"x": 1}, thread_id="ses_1")
        # The client rewrote the arguments: no guess by tool name, the call is unmatched and its place stays open.
        self.assertEqual(order.claim("move", {"x": 1, "added_by_client": True}, thread_id="ses_1"),
                         (None, "arguments_differ"))
        second, _ = order.claim("move", {"x": 1}, thread_id="ses_1")
        self.assertEqual([(first.position, first.matched_by), (second.position, second.matched_by)],
                         [(0, "arguments"), (1, "arguments")])
        self.assertEqual(order.claim("move", {"x": 1}, thread_id="ses_1"), (None, "not_in_step"))   # both taken
        self.assertEqual(order.claim("look", {}, thread_id="ses_1")[0].position, 2)

    def test_true_is_not_one(self):
        order = EmittedOrder()
        order.announce("t", [(None, "move", {"x": True}), (None, "move", {"x": 1})])
        self.assertEqual(order.claim("move", {"x": 1}, thread_id="t")[0].position, 1)

    def test_every_reason_a_call_has_no_place(self):
        order = EmittedOrder()
        self.assertEqual(order.claim("move", {"x": 1}, thread_id="ses_1"), (None, "no_step_known"))
        order.announce("ses_1", [])                                   # a response without host tool calls
        self.assertEqual(order.claim("move", {"x": 1}, native_call_id="toolu_a"), (None, "no_step_known"))
        order.announce("ses_1", [("toolu_a", "move", {"x": 1})])
        self.assertEqual(set(UNMATCHED_REASONS), {
            order.claim("move", {"x": 1}, native_call_id="toolu_z", thread_id="ses_1")[1],   # an id never emitted
            order.claim("move", {"x": 1})[1],                                                # no id, no session key
            order.claim("move", {"x": 1}, thread_id="ses_2")[1],                             # another client session
            order.claim("move", {"x": 9}, thread_id="ses_1")[1],                             # other arguments
            order.claim("look", {}, thread_id="ses_1")[1],                                   # a tool not emitted
            "no_step_known"})
        self.assertEqual([order.claim("move", {"x": 1}, native_call_id="toolu_z", thread_id="ses_1")[1],
                          order.claim("move", {"x": 1})[1], order.claim("move", {"x": 1}, thread_id="ses_2")[1],
                          order.claim("move", {"x": 9}, thread_id="ses_1")[1], order.claim("look", {}, thread_id="ses_1")[1]],
                         ["call_id_unknown", "no_session_key", "session_key_mismatch", "arguments_differ", "not_in_step"])
        self.assertIsNone(order.wait(None))
        order.resolve(None)
        ticket, reason = order.claim("move", {"x": 1}, native_call_id="toolu_a")
        self.assertEqual((ticket.position, ticket.step_size, reason), (0, 1, None))
        self.assertEqual(order.claim("move", {"x": 1}, native_call_id="toolu_a"), (None, "call_id_unknown"))  # replayed

    def test_a_refused_earlier_call_releases_the_later_one(self):
        order = EmittedOrder(wait_s=TIMEOUT)
        order.announce("t", STEP[:2])
        later, _ = order.claim("move", {"x": 2}, native_call_id="toolu_b")
        earlier, _ = order.claim("move", {"x": 1}, native_call_id="toolu_a")
        order.resolve(earlier)                                        # refused before dispatch
        started = time.monotonic()
        order.wait(later)
        self.assertLess(time.monotonic() - started, 1)
        self.assertFalse(later.expired)

    def test_a_call_that_never_arrives_is_written_off_once_and_the_rest_keeps_its_order(self):
        order = EmittedOrder(wait_s=0.2)
        order.announce("t", STEP)                                     # "a" is never sent
        started = time.monotonic()
        released, tickets = self.run_calls(order, [("c", "move", {"x": 3}, dict(native_call_id="toolu_c")),
                                                   ("b", "move", {"x": 2}, dict(native_call_id="toolu_b"))])
        elapsed = time.monotonic() - started
        self.assertEqual(released, ["b", "c"])
        self.assertTrue(0.2 <= elapsed < 0.4 + 1, elapsed)             # one bound, not one per held call
        self.assertEqual(sorted(tuple(ticket.skipped) for ticket in tickets.values()), [(), (0,)])
        self.assertTrue(any(ticket.expired and ticket.skipped == [0] for ticket in tickets.values()))
        # The written-off call arrives after all: it is dispatched on arrival, nothing waits.
        late, _ = order.claim("move", {"x": 1}, native_call_id="toolu_a")
        self.assertEqual((late.position, order.wait(late).expired, late.waited_s < 0.1), (0, False, True))

    def test_an_arrived_call_that_is_never_handed_over_is_overtaken_after_the_second_bound(self):
        order = EmittedOrder(wait_s=0.1)
        order.announce("t", STEP[:2])
        stuck, _ = order.claim("move", {"x": 1}, native_call_id="toolu_a")   # arrived, never resolved
        later = order.wait(order.claim("move", {"x": 2}, native_call_id="toolu_b")[0])
        self.assertEqual((later.expired, later.skipped, 0.2 <= later.waited_s < 1.2), (True, [], True))
        order.resolve(stuck)

    def test_old_steps_are_dropped(self):
        order = EmittedOrder(max_steps=2)
        for index in range(3):
            order.announce("t", [(f"toolu_{index}", "move", {"x": index})])
        self.assertEqual(order.claim("move", {"x": 0}, native_call_id="toolu_0"), (None, "call_id_unknown"))
        self.assertIsNotNone(order.claim("move", {"x": 1}, native_call_id="toolu_1")[0])

    def test_invalid_bounds_are_refused(self):
        for wait_s in (0, -1, True, "5", None):
            with self.assertRaisesRegex(ValueError, "invalid_emitted_order_wait_s"):
                EmittedOrder(wait_s=wait_s)
        for max_steps in (0, 1.5, True):
            with self.assertRaisesRegex(ValueError, "invalid_emitted_order_max_steps"):
                EmittedOrder(max_steps=max_steps)

    def test_with_the_dispatcher_parallel_safe_calls_overlap_and_ordered_calls_follow_emitted_order(self):
        """look(1), look(2), move(3), move(4) emitted; arrival 4, 3, 2, 1: the looks overlap (rendezvous), the moves
        run after both, 3 before 4."""
        ran, lock, together = [], threading.Lock(), threading.Barrier(2)

        def handler(params):
            if params["tool"] == "look":
                together.wait(TIMEOUT)
            with lock:
                ran.append(params["arguments"]["x"])
            return params["arguments"]["x"]
        dispatcher = ToolDispatcher(handler, lambda tool, arguments: tool == "look")
        self.addCleanup(dispatcher.shutdown)
        order = EmittedOrder(wait_s=TIMEOUT)
        step = [(None, "look", {"x": 1}), (None, "look", {"x": 2}), (None, "move", {"x": 3}), (None, "move", {"x": 4})]
        order.announce("t", step)
        results = {}

        def call(tool, arguments):
            ticket, _ = order.claim(tool, arguments, thread_id="t")
            order.wait(ticket)
            try:
                future = dispatcher.submit(dict(tool=tool, arguments=arguments))
            finally:
                order.resolve(ticket)
            results[arguments["x"]] = future.result(TIMEOUT)
        threads = []
        for _, tool, arguments in reversed(step):
            held = order.held
            threads.append(threading.Thread(target=call, args=(tool, arguments), daemon=True))
            threads[-1].start()
            deadline = time.monotonic() + TIMEOUT
            while order.held == held and arguments["x"] != 1 and time.monotonic() < deadline:
                time.sleep(0.001)
        for thread in threads:
            thread.join(TIMEOUT)
        self.assertFalse(together.broken)
        self.assertEqual((sorted(ran[:2]), ran[2:], results), ([1, 2], [3, 4], {1: 1, 2: 2, 3: 3, 4: 4}))

# ==== tests/test_conformance_deny_proxy.py ====

class DenyAllProxyTests(unittest.TestCase):
    def test_only_a_probe_shaped_get_root_is_direct(self):
        proxy = DenyAllProxy().start()
        self.addCleanup(proxy.close)
        port = int(proxy.url.rsplit(":", 1)[1])
        host = f"127.0.0.1:{port}"
        requests = [("GET", "/", {"User-Agent": "Go-http-client/1.1", "Accept-Encoding": "gzip"}, None),   # the probe
                    ("GET", "/", {"Authorization": "Bearer k"}, None), ("GET", "/", {"x-api-key": "k"}, None),
                    ("GET", "/v1/models", {}, None), ("GET", "/?x=1", {}, None), ("HEAD", "/", {}, None),
                    ("POST", "/v1/messages", {"Content-Type": "application/json"}, b"{}"),
                    ("GET", "http://example.com/", {}, None), ("CONNECT", "example.com:443", {}, None)]
        for method, target, headers, body in requests:
            connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
            connection.request(method, target, body=body, headers=headers)
            self.assertEqual(connection.getresponse().status, 403)
            connection.close()
        self.assertEqual(proxy.direct, [("GET", "/", host)])
        self.assertEqual([(method, target) for method, target, _ in proxy.attempts],
                         [(method, target) for method, target, _, _ in requests[1:]])

# ==== tests/test_cortex_layering.py ====

from support.paths import REPO_ROOT
CORTEX_ROOT = REPO_ROOT / "ni_rua" / "cortex"
FORBIDDEN_CORTEX_TARGETS = frozenset({"runtime", "apps"})
FORBIDDEN_CORE_TARGETS = frozenset({"robot", "perception", "cortex", "runtime", "apps"})


def _resolve_relative_targets(file_path: Path, node: ast.ImportFrom) -> list[tuple[str, ...]]:
    rel_parts = file_path.relative_to(REPO_ROOT).parts
    pkg_parts = rel_parts[:-1]
    up = node.level - 1
    if up > len(pkg_parts):
        return []
    base = pkg_parts[: len(pkg_parts) - up]
    if node.module:
        return [base + tuple(node.module.split("."))]
    return [base + (alias.name,) for alias in node.names]


def _upward_imports(tree: ast.AST, file_path: Path, forbidden: frozenset[str]) -> list[tuple[int, str]]:
    """Return `(lineno, description)` for any module-level, nested, or dynamic upward import in `file_path`."""
    violations: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                parts = alias.name.split(".")
                if len(parts) >= 2 and parts[0] == "ni_rua" and parts[1] in forbidden:
                    violations.append((node.lineno, f"import {alias.name}"))
        elif isinstance(node, ast.ImportFrom):
            if node.level >= 1:
                targets = _resolve_relative_targets(file_path, node)
            else:
                module_parts = tuple(node.module.split(".")) if node.module else ()
                targets = (
                    [("ni_rua", alias.name) for alias in node.names]
                    if module_parts == ("ni_rua",)
                    else [module_parts]
                )
            for parts in targets:
                if len(parts) >= 2 and parts[0] == "ni_rua" and parts[1] in forbidden:
                    violations.append((node.lineno, f"from {'.'.join(parts)}"))
        elif isinstance(node, ast.Call):
            fn = node.func
            is_dynamic = (
                (isinstance(fn, ast.Name) and fn.id == "__import__")
                or (isinstance(fn, ast.Attribute) and fn.attr == "import_module")
            )
            if is_dynamic and node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                parts = node.args[0].value.split(".")
                if len(parts) >= 2 and parts[0] == "ni_rua" and parts[1] in forbidden:
                    violations.append((node.lineno, f"dynamic import {node.args[0].value}"))
    return violations


class CortexLayerBoundaryTests(unittest.TestCase):
    def test_ni_rua_cortex_has_zero_upward_imports_module_and_nested(self):
        py_files = sorted(CORTEX_ROOT.rglob("*.py"))
        self.assertTrue(py_files)
        violations: list[str] = []
        for path in py_files:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for lineno, desc in _upward_imports(tree, path, FORBIDDEN_CORTEX_TARGETS):
                violations.append(f"{path.relative_to(REPO_ROOT)}:{lineno}: {desc}")
        self.assertEqual(violations, [])

    def test_core_roboprobe_source_leaf_has_zero_upward_imports(self):
        leaf = REPO_ROOT / "ni_rua" / "core" / "roboprobe_source.py"
        self.assertTrue(leaf.is_file())
        tree = ast.parse(leaf.read_text(encoding="utf-8"), filename=str(leaf))
        self.assertEqual(_upward_imports(tree, leaf, FORBIDDEN_CORE_TARGETS), [])
        mod = importlib.import_module("ni_rua.core.roboprobe_source")
        self.assertTrue(callable(mod.upstream))
        self.assertTrue(callable(mod.joint_policy))

    def test_plugin_contract_has_zero_upward_imports_and_canonical_constants(self):
        path = CORTEX_ROOT / "agents" / "plugin_contract.py"
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        self.assertEqual(_upward_imports(tree, path, FORBIDDEN_CORTEX_TARGETS), [])
        from ni_rua.cortex.agents import plugin_contract

        self.assertEqual(plugin_contract.SCHEMA, "ni_rua.agent_plugin.manifest.v1")
        manifest = plugin_contract.build_manifest(version="0.2.0", tool_names=["robot_observe", "run_begin"])
        self.assertEqual(manifest["tool_names"], ["robot_observe", "run_begin"])
        self.assertEqual(manifest["default_port"], plugin_contract.default_port())

    def test_service_tiers_lives_canonically_in_cortex_and_reexported_in_console(self):
        from ni_rua.apps.gateway.extensions import operator_console
        from ni_rua.cortex.agents import reasoning_options

        self.assertIs(operator_console.SERVICE_TIERS, reasoning_options.SERVICE_TIERS)


class CortexUnificationAndRelayTests(unittest.TestCase):
    def test_backend_registry_is_explicit_and_complete(self):
        from ni_rua.cortex.agents.backend import AgentBackend, BackendTraits
        from ni_rua.cortex.agents.registry import BACKENDS_REGISTRY, get_backend

        self.assertEqual(BACKENDS_REGISTRY.hosted_kinds(), ("codex", "claude_code", "opencode"))
        for kind in ("codex", "claude_code", "opencode", "external"):
            entry = get_backend(kind)
            self.assertIsInstance(entry, AgentBackend)
            self.assertIsInstance(entry.traits, BackendTraits)
        sections = BACKENDS_REGISTRY.profile_sections()
        self.assertEqual(set(sections), {"claude_code", "opencode"})

    def test_agent_session_hook_descriptors_preserve_new_contract_and_external_turn_done(self):
        from ni_rua.cortex.agents.backend import SessionHooks, bind_session_hooks
        from ni_rua.cortex.agents.base import AgentSession
        from ni_rua.cortex.agents.external.session import ExternalSession

        self.assertNotIn("__init__", AgentSession.__dict__)
        session = AgentSession.__new__(AgentSession)
        self.assertEqual(vars(session), {})
        self.assertIsNone(session.on_disconnect)

        ext = ExternalSession.__new__(ExternalSession)
        ext.turn_done = threading.Event()
        ext.turn_done.set()
        cb = lambda: None
        bind_session_hooks(ext, SessionHooks(on_disconnect=cb))
        self.assertIs(ext.on_disconnect, cb)
        self.assertTrue(ext.turn_done.is_set())

    def test_relay_modules_and_wire_adapter_identity_preserved(self):
        from ni_rua.cortex.agents.codex import sandbox as codex_sandbox_new, session as codex_new
        from ni_rua.cortex.relay import ContinuityTracker, GuardMixin, ModelRequestProxy, PayloadMixin
        from ni_rua.cortex.relay import continuity, server as relay_server
        from ni_rua.cortex.session import agent_sandbox
        from ni_rua.cortex.session.wire import OpenAIResponsesWire, WireAdapter

        self.assertTrue(issubclass(ModelRequestProxy, GuardMixin))
        self.assertTrue(issubclass(ModelRequestProxy, PayloadMixin))
        self.assertIs(ModelRequestProxy, relay_server.ModelRequestProxy)
        self.assertIs(continuity.ContinuityTracker, ContinuityTracker)
        self.assertTrue(hasattr(codex_new, "CodexSession"))
        self.assertTrue(issubclass(codex_sandbox_new.CodexSandbox, agent_sandbox.AgentSandbox))
        self.assertIs(codex_sandbox_new._UnixForwarder, agent_sandbox._UnixForwarder)
        self.assertIs(relay_server.WireAdapter, WireAdapter)
        self.assertIs(relay_server.OpenAIResponsesWire, OpenAIResponsesWire)
        self.assertEqual(Path(relay_server.__file__).resolve(), REPO_ROOT / "ni_rua" / "cortex" / "relay" / "server.py")
        for retired in (
            "ni_rua.cortex._compat",
            "ni_rua.cortex.session.codex",
            "ni_rua.cortex.session.codex_sandbox",
            "ni_rua.cortex.session.model_proxy",
        ):
            self.assertIsNone(importlib.util.find_spec(retired), retired)

# ==== tests/test_operation_context.py ====

class RetiredOperationContextAndArchiveReplayTest(unittest.TestCase):
    def setUp(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        self.identity = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
        self.store = Store(Path(temp) / "store", self.identity)
        self.addCleanup(self.store.close)
        self.observation = dict(self.identity, observation_id="obs_1", sim_step=10)
        self.store.save("observation", "obs_1", self.observation)

    def test_legacy_modules_are_retired_and_compat_kwargs_preserved(self):
        self.assertIsNone(importlib.util.find_spec("ni_rua.cortex.context.legacy_task_planning"))
        self.assertIsNone(importlib.util.find_spec("ni_rua.cortex.context.operation_candidates"))

        notes = resolve_task_notes(default_enabled=True, task_scheduling=True, operation_candidates=False)
        self.assertEqual(notes["config"]["operation_candidates"], False)
        self.assertEqual(notes["config"]["task_scheduling"], True)
        self.assertEqual(default_notes_for(operation_candidates=True), default_notes_for(operation_candidates=False))
        self.assertEqual(version_for(task_scheduling=False, operation_candidates=True), "operation-notes-v14")

        with self.assertRaisesRegex(ValueError, "invalid_operation_planning_flags"):
            TaskPlanning(self.store, operation_candidates="bad")  # type: ignore[arg-type]

    def test_archived_structured_plan_reads_cleanly_while_live_dict_submit_is_rejected(self):
        archived_plan = dict(
            objective="Transfer the bottle onto the coaster.",
            stages=[
                dict(
                    id="deposit",
                    goal="Place onto visible support.",
                    advance_when="Bottle separates stably.",
                    operation_context=dict(
                        object_label="bottle",
                        state="retained",
                        release_observation_id="obs_0",
                        evidence=dict(
                            observation_id="obs_1",
                            sources=["rgb", "proprioception"],
                            basis="Visible separation.",
                            uncertainty="Back face occluded.",
                        ),
                    ),
                ),
                dict(
                    id="verify",
                    goal="Confirm upright retention.",
                    advance_when="No topple observed.",
                    depends_on=["deposit"],
                ),
            ],
            constraints=[],
            uncertainties=["Exact CoM uncertain."],
            completion_criteria=["Bottle rests upright on coaster."],
        )
        archived_record = dict(
            plan_id="task-plan-archived-1",
            version="actor_task_plan_v9",
            identity=dict(self.identity),
            revision=1,
            previous_plan_id=None,
            observation_id="obs_1",
            plan=deepcopy(archived_plan),
            source_kind="actor_judgment",
            tool_choices="advisory",
        )
        self.store.save("task_plan", archived_record["plan_id"], archived_record)
        self.store.set_state("runtime", "task_plan", archived_record)

        owner = TaskPlanning(self.store, required=True, operation_candidates=False)
        self.assertEqual(owner.current(self.identity), archived_record)
        self.assertEqual(
            owner.call(dict(operation="read"), self.identity, self.observation, {}),
            dict(required=True, plan=archived_record, tool_choices="advisory"),
        )
        ctx = owner.decision_context(self.identity)
        self.assertEqual(ctx["plan"], archived_plan)
        self.assertEqual(ctx["ledger"], [])

        advisory = scheduling_failure_advisory(dict(status="failed", reason="tracking_error"), archived_record)
        self.assertEqual(advisory["dependency_free_plan_stages_to_reassess"], ["deposit"])
        self.assertEqual(advisory["plan_observation_id"], "obs_1")

        # Live dict submissions are rejected; freeform updates replace the archived plan cleanly.
        with self.assertRaisesRegex(ValueError, "invalid_task_plan"):
            owner.call(dict(operation="submit", plan=archived_plan, ledger=[]), self.identity, self.observation, {})
        with self.assertRaisesRegex(ValueError, "task_plan_attention_requires_freeform_plan"):
            owner.call(dict(operation="submit", plan=archived_plan, attention=[]), self.identity, self.observation, {})

        updated = owner.call(
            dict(operation="update", plan="Grasp bottle neck, place on coaster, and check upright stability."),
            self.identity,
            self.observation,
            {},
        )
        self.assertEqual(updated, dict(plan_id="P2", status="saved"))
        current = owner.current(self.identity)
        self.assertEqual(current["version"], PLANNING_VERSION)
        self.assertEqual(current["revision"], 2)
        self.assertEqual(current["previous_plan_id"], "task-plan-archived-1")

    def test_plan_schema_for_error_preserves_dict_and_freeform_diagnostics(self):
        self.assertIs(_plan_schema_for_error({}), LEGACY_PLAN_SCHEMA)
        self.assertIs(_plan_schema_for_error("prose"), PLAN_SCHEMA)
        diag = error_payload(
            ValueError("invalid_task_plan: provide objective"),
            tool="task_plan",
            arguments=dict(plan=dict(objective="x", stages=[], constraints=[], uncertainties=[], completion_criteria=["ok"])),
        )
        self.assertEqual(diag["code"], "invalid_task_plan")
        self.assertEqual(diag["field_path"], "/plan/stages")

# ==== tests/test_public_task_rules.py ====

class PublicRulesTest(unittest.TestCase):
    def test_all_task_sections_match_frozen_official_html(self):
        catalog = json.loads(CATALOG.read_text())
        with gzip.open(SNAPSHOT, "rt", encoding="utf-8") as stream:
            pages = json.load(stream)
        self.assertEqual(rebuild(catalog, pages), catalog)
        self.assertEqual(len(pages), 42)
        self.assertEqual(len(catalog["tasks"]), 54)
        for task, entry in catalog["tasks"].items():
            with self.subTest(task=task):
                self.assertEqual(public_task_rules(task)["text"], "\n\n".join(entry["public_rules"]))
                if task.endswith("_random"):
                    self.assertEqual(entry["public_rules"], catalog["tasks"][task.removesuffix("_random")]["public_rules"])

    def test_html_extraction_preserves_complete_cells_and_rejects_missing_sections(self):
        html = '''<div class="sl-markdown-content"><table>
          <tr><td>Instruction</td><td>Use <b>all four</b> pieces &amp; return.</td></tr>
          <tr><td>Description</td><td>Compare x₁ and x₂; do not replace 4 with 5.</td></tr>
          <tr><td>Comment</td><td>Keep the original “example”.</td></tr></table>
          <h2>Scoring</h2><table><tr><th>Score</th><th>Condition</th></tr>
          <tr><td>15</td><td>Lift at least <b>5 cm</b>.</td></tr>
          <tr><td>100</td><td>All five pieces; gripper open.</td></tr></table></div>'''
        self.assertEqual(extract_sections(html), [
            "Instruction\n\nUse all four pieces & return.",
            "Description\n\nCompare x₁ and x₂; do not replace 4 with 5.",
            "Comment\n\nKeep the original “example”.",
            "Scoring\n\n| Score | Condition |\n| --- | --- |\n| 15 | Lift at least 5 cm. |\n| 100 | All five pieces; gripper open. |",
        ])
        with self.assertRaises(ValueError):
            extract_sections(html.replace("Description", "Unrecognized field"))
        with self.assertRaises(ValueError):
            extract_sections(html.replace("<h2>Scoring</h2>", ""))

    def test_documented_rule_and_unknown_task(self):
        result = public_task_rules("imitate_sorting_sequence")
        self.assertEqual(result["status"], "documented")
        self.assertIn("https://robodojo-benchmark.com/", result["source_url"])
        self.assertTrue(result["public_rules"])
        self.assertEqual(result["version"], 2)
        self.assertEqual(result["text"], "\n\n".join(result["public_rules"]))
        self.assertTrue(result["text"].startswith("Instruction\n"))
        self.assertIn("\n\nDescription\n", result["text"])
        self.assertIn("\n\nScoring\n", result["text"])
        self.assertIn("| Score | Condition |", result["text"])
        self.assertNotIn("examples are not this episode's answers", result["text"])
        unknown = public_task_rules("not_an_official_task")
        self.assertEqual(unknown["status"], "unavailable")
        self.assertIsNone(unknown["text"])

    def test_lookup_uses_only_frozen_catalog_without_runtime_evaluator(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "catalog.json"
            data = json.dumps(dict(version="test", tasks={"fixture": dict(
                source_url="https://robodojo-benchmark.com/doc/sim-tasks/fixture/",
                public_rules=["Use the order in the current instruction."])})).encode()
            path.write_bytes(data)
            with patch("ni_rua.cortex.context.public_task_rules.CATALOG_PATH", path):
                result = public_task_rules("fixture")
            self.assertEqual(result["catalog_sha256"], hashlib.sha256(data).hexdigest())
            self.assertEqual("Use the order in the current instruction.", result["text"])

    def test_legacy_archived_text_is_not_replaced_with_new_official_sections(self):
        text = "Official RoboDojo public task rules (old citation):\nArchived summary."
        self.assertEqual(actor_task_rules_text({"version": 1, "text": text}),
                         "RoboDojo task rules:\nArchived summary.")

    def test_invalid_catalog_rules_are_rejected(self):
        for rules in ([], [""], [None], "text"):
            with self.subTest(rules=rules), tempfile.TemporaryDirectory() as temporary:
                path = Path(temporary) / "catalog.json"
                path.write_text(json.dumps(dict(version=2, tasks={"fixture": dict(
                    source_url="https://robodojo-benchmark.com/doc/sim-tasks/fixture/",
                    public_rules=rules)})))
                with patch("ni_rua.cortex.context.public_task_rules.CATALOG_PATH", path), \
                        self.assertRaisesRegex(ValueError, "invalid_public_task_rules_catalog"):
                    public_task_rules("fixture")


class PublicRulesStartupTest(unittest.IsolatedAsyncioTestCase):
    async def test_benchmark_rules_are_delivered_and_archived_independently_of_operation_notes(self):
        with tempfile.TemporaryDirectory() as temporary:
            workbench, terminal = fixture_workbench(Path(temporary) / "run", task="imitate_sorting_sequence",
                                                     default_task_notes=False)
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                    await workbench.start("unused", benchmark=True)
                prompt = workbench.session.run_turn.call_args.args[0]
                manifest = json.loads((workbench.root / "manifest.json").read_text())
                recorded = next(r for r in workbench.store.records("configuration") if "prompt" in r)
                rules = manifest["public_task_rules"]
                self.assertEqual(recorded["public_task_rules"], rules)
                self.assertEqual(prompt.count(rules["text"]), 1)
                self.assertIn("Fixture native task.", prompt)
                self.assertIsNone(manifest["task_notes"])
                self.assertIn("cortex/context/robodojo_task_rules.json", manifest["source_sha256"])
                for name in ("execute", "reset", "evaluate"):
                    getattr(workbench.robot, name).assert_not_called()
            finally:
                await workbench.close()

    async def test_smoke_prompt_is_not_reinterpreted_as_native_task(self):
        with tempfile.TemporaryDirectory() as temporary:
            workbench, terminal = fixture_workbench(Path(temporary) / "run", task="imitate_sorting_sequence")
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                    await workbench.start("Only observe, do not sort.", benchmark=False)
                self.assertEqual(workbench.session.run_turn.call_args.args[0], "Only observe, do not sort.")
                manifest = json.loads((workbench.root / "manifest.json").read_text())
                self.assertIsNone(manifest["public_task_rules"])
            finally:
                await workbench.close()

    async def test_resume_preserves_rule_text_without_reinterpreting_old_runs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            first, terminal = fixture_workbench(root, task="imitate_sorting_sequence")
            legacy = dict(version=1, status="documented", task="imitate_sorting_sequence",
                          text="RoboDojo task rules:\nFrozen pre-verbatim summary.",
                          public_rules=["Frozen pre-verbatim summary."])
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal), \
                        patch("ni_rua.apps.gateway.workbench.public_task_rules", return_value=legacy):
                    await first.start("unused", benchmark=True)
            finally:
                await first.close()
            old = json.loads((root / "manifest.json").read_text())["public_task_rules"]
            resumed, terminal = fixture_workbench(root, resume=True)
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal), \
                        patch("ni_rua.apps.gateway.workbench.public_task_rules", side_effect=AssertionError("must use archived rules")):
                    await resumed.start("unused", benchmark=True)
                current = json.loads((root / "manifest.json").read_text())
                self.assertEqual(current["public_task_rules"], old)
                resumed.session.run_turn.assert_not_called()
            finally:
                await resumed.close()

# ==== tests/test_reasoning_options.py ====

KIMI = CatalogueModel(id="kimi-k3", name="Kimi K3", tools=True, image_input=True, reasoning=True,
                      reasoning_efforts=("low", "high", "max"), reasoning_toggle=True, effort_updates=False)


def values(control, key="value"):
    return [choice[key] for choice in control["choices"]]


def flags(control):
    return {choice["value"]: (choice["backend"], choice["model"]) for choice in control["choices"]}


class BackendTraitsTests(unittest.TestCase):
    def test_traits_are_read_from_backend_code(self):
        codex = backend_traits("codex", "openai_responses")
        self.assertEqual((codex.effort_choices, codex.live_effort, codex.service_tier, codex.thinking_display_choices),
                         (model_effort_choices(), True, True, ()))
        claude = backend_traits("claude_code", "anthropic_messages")
        self.assertEqual((claude.effort_choices, claude.live_effort, claude.service_tier),
                         (factory.NATIVE_EFFORT_CHOICES["claude_code"], False, False))
        for wire in ("anthropic_messages", "openai_responses"):
            opencode = backend_traits("opencode", wire)
            self.assertEqual((opencode.effort_choices, opencode.live_effort, opencode.service_tier),
                             (opencode_config.EFFORT_CHOICES[wire], True, wire == "openai_responses"))
        self.assertEqual(backend_traits("opencode", "anthropic_messages").thinking_display_choices,
                         opencode_config.THINKING_DISPLAY_CHOICES)
        self.assertEqual(backend_traits("opencode", "openai_responses").thinking_display_choices, ())
        with self.assertRaisesRegex(ValueError, "unknown_agent_backend:gemini"):
            backend_traits("gemini", "openai_responses")

    def test_output_follows_patched_backend_constants(self):
        patched = {"anthropic_messages": ("low", "max"), "openai_responses": ("high",)}
        with mock.patch.dict(opencode_config.EFFORT_CHOICES, patched):
            effort = reasoning_options("opencode", "anthropic_messages", None)["effort"]
        self.assertEqual([c["value"] for c in effort["choices"] if c["backend"]], ["low", "max"])
        self.assertEqual(effort["default"], "low")   # the schema default "high" is not a backend choice here
        with mock.patch.object(ClaudeCodeSession, "capabilities", AgentCapabilities(live_effort=True)):
            effort = reasoning_options("claude_code", "anthropic_messages", None)["effort"]
        self.assertEqual((effort["applies"], effort["scope"], effort["command"]), ("next_turn", "live", "set_effort"))


class ReasoningOptionsTests(unittest.TestCase):
    def test_codex_known_model_uses_the_native_catalogue(self):
        probes = {"priority": {"tier": "priority", "supported": True, "served_tier": "priority", "error": None},
                  "ultrafast": {"tier": "ultrafast", "supported": False, "served_tier": None, "error": None,
                                "rejection": "HTTP 400: service_tier rejected"}}
        options = reasoning_options("codex", "openai_responses", None, model_id="gpt-6.1-sol", effort="high",
                                    service_tier="priority", tier_probes=probes)
        effort = options["effort"]
        self.assertEqual((effort["available"], effort["value"], effort["default"], effort["command"],
                          effort["applies"], effort["scope"], effort["model_source"]),
                         (True, "high", "high", "set_effort", "next_turn", "live", "codex_models"))
        self.assertEqual(values(effort), list(model_effort_choices()))
        self.assertEqual(flags(effort)["none"], (True, False))
        self.assertEqual(flags(effort)["ultra"], (True, True))
        self.assertEqual(effort["supported"], ["low", "medium", "high", "xhigh", "max", "ultra"])
        self.assertTrue(all(choice["selectable"] for choice in effort["choices"]))   # backend accepts all of them
        ultra = effort["choices"][-1]
        self.assertEqual((ultra["description"], ultra["note"]),
                         ("Maximum reasoning with automatic task delegation", None))
        self.assertEqual(effort["choices"][0]["note"], "模型未声明支持")
        self.assertEqual((options["model"]["id"], options["model"]["listed"], options["model"]["native_codex"],
                          options["model"]["default_effort"]), ("gpt-6.1-sol", False, True, "low"))
        tier = options["service_tier"]
        self.assertEqual((tier["available"], tier["value"], tier["default"], tier["command"], tier["applies"],
                          tier["scope"], tier["model_source"]),
                         (True, "priority", "default", "set_service_tier", "next_model_request", "live",
                          "codex_models"))
        self.assertEqual(values(tier), list(SERVICE_TIERS))
        by_value = {choice["value"]: choice for choice in tier["choices"]}
        self.assertEqual((by_value["priority"]["label"], by_value["priority"]["model"],
                          by_value["priority"]["probe"]["supported"], by_value["priority"]["selectable"]),
                         ("Fast", True, True, True))
        self.assertEqual((by_value["ultrafast"]["model"], by_value["ultrafast"]["selectable"]), (False, False))
        self.assertEqual((by_value["default"]["probe"], by_value["default"]["selectable"]), (None, True))
        self.assertEqual(options["thinking_display"],
                         dict(available=False, label="思考显示", reason="backend_has_no_thinking_display",
                              message="Codex 没有思考显示选项"))

    def test_codex_listing_claims_win_over_the_native_catalogue(self):
        listed = CatalogueModel(id="gpt-6.1-sol", reasoning=True, reasoning_efforts=("low", "medium", "high"))
        effort = reasoning_options("codex", "openai_responses", listed)["effort"]
        self.assertEqual((effort["model_source"], effort["supported"]), ("catalogue", ["low", "medium", "high"]))
        self.assertEqual(flags(effort)["ultra"], (True, False))

    def test_codex_unknown_model(self):
        options = reasoning_options("codex", "openai_responses", None, model_id="my-custom-model", effort="none")
        effort, tier = options["effort"], options["service_tier"]
        self.assertEqual((effort["model_source"], effort["value"], effort["value_selectable"]), (None, "none", True))
        self.assertEqual(effort["supported"], list(model_effort_choices()))
        self.assertTrue(all(choice["model"] is None and choice["note"] is None for choice in effort["choices"]))
        self.assertEqual(tier["model_source"], None)
        self.assertTrue(all(choice["model"] is None and choice["probe"] is None for choice in tier["choices"]))
        self.assertEqual(options["model"], dict(id="my-custom-model", listed=False, native_codex=False, reasoning=None,
                                                efforts=None, toggle=None, default_effort=None, budget=None,
                                                effort_updates=None))

    def test_claude_code_effort_is_session_scoped(self):
        options = reasoning_options("claude_code", "anthropic_messages", KIMI, effort="ultra")
        effort = options["effort"]
        self.assertEqual((effort["command"], effort["scope"], effort["applies"], effort["applies_label"]),
                         ("handoff_run", "session", "next_session", "需要新会话"))
        self.assertEqual(values(effort), ["low", "medium", "high", "xhigh", "max"])
        self.assertEqual(effort["supported"], ["low", "high", "max"])
        self.assertEqual((effort["value"], effort["value_selectable"]), ("ultra", False))
        self.assertIsNone(effort["note"])   # effort_updates only matters for a live control
        self.assertEqual(options["service_tier"]["reason"], "service_tier_unsupported_by_backend")
        thinking = options["thinking_display"]
        self.assertEqual((thinking["available"], thinking["reason"]), (False, "claude_code_thinking_in_profile_env"))
        self.assertIn("backends.claude_code.env", thinking["message"])

    def test_opencode_anthropic_wire(self):
        options = reasoning_options("opencode", "anthropic_messages", KIMI, effort="high",
                                    thinking_display="summarized")
        effort = options["effort"]
        self.assertEqual((effort["command"], effort["applies"], effort["model_source"]),
                         ("set_effort", "next_turn", "catalogue"))
        self.assertEqual(flags(effort), {"low": (True, True), "medium": (True, False), "high": (True, True),
                                         "xhigh": (True, False), "max": (True, True)})
        self.assertEqual(effort["note"], "模型声明不支持在会话中途更改强度（由供应商裁决）")
        thinking = options["thinking_display"]
        self.assertEqual((thinking["available"], thinking["value"], thinking["default"], thinking["command"],
                          thinking["scope"], thinking["applies"]),
                         (True, "summarized", None, "handoff_run", "session", "next_session"))
        self.assertEqual(values(thinking), [None, *opencode_config.THINKING_DISPLAY_CHOICES])
        self.assertEqual(options["service_tier"]["available"], False)

    def test_opencode_responses_wire(self):
        options = reasoning_options("opencode", "openai_responses", KIMI)
        effort = options["effort"]
        # the backend's Responses choices plus the model's "max" (declared, but the backend refuses it)
        self.assertEqual(values(effort), ["none", "minimal", "low", "medium", "high", "xhigh", "max"])
        maximum = effort["choices"][-1]
        self.assertEqual((maximum["backend"], maximum["model"], maximum["selectable"], maximum["recommended"],
                          maximum["note"]), (False, True, False, False, "当前后端不支持"))
        self.assertEqual(effort["supported"], ["low", "high"])
        self.assertEqual(options["thinking_display"]["reason"],
                         "opencode_thinking_display_requires_anthropic_messages")

    def test_model_reporting_low_high_max_from_a_live_shaped_page(self):
        page = {"object": "list", "data": [{"id": "kimi-k3", "capabilities": {
            "tools": {"supported": True}, "reasoning": {"supported": True, "toggle": True,
                                                         "effort": ["low", "high", "max"]}}}]}
        entry = parse_models_page(page)[0][0]
        for backend, wire in (("claude_code", "anthropic_messages"), ("opencode", "anthropic_messages")):
            options = reasoning_options(backend, wire, entry.to_dict())   # the to_dict() form is accepted too
            self.assertEqual(options["effort"]["supported"], ["low", "high", "max"], backend)
            self.assertEqual(options["model"]["efforts"], ["low", "high", "max"])
            self.assertEqual(options["model"]["toggle"], True)

    def test_model_without_reasoning_and_unlisted_model(self):
        off = reasoning_options("opencode", "anthropic_messages", CatalogueModel(id="m", reasoning=False))["effort"]
        self.assertEqual((off["model_source"], off["supported"]), ("catalogue", []))
        self.assertTrue(all(choice["model"] is False and choice["selectable"] for choice in off["choices"]))
        self.assertEqual(off["default"], ModelConfig().effort)
        unlisted = reasoning_options("opencode", "anthropic_messages", None, model_id="gpt-6.1-sol")
        # the native Codex catalogue is Codex's own; other backends never borrow it
        self.assertEqual((unlisted["effort"]["model_source"], unlisted["model"]["native_codex"]), (None, False))

    def test_wire_without_effort_variants(self):
        effort = reasoning_options("opencode", "openai_chat", None)["effort"]
        self.assertEqual((effort["available"], effort["reason"]), (False, "effort_unsupported_on_wire"))

    def test_output_is_plain_json_and_tiers_can_be_given(self):
        options = reasoning_options("codex", "openai_responses", None, model_id="gpt-6.1-sol",
                                    service_tiers=("default", "priority"))
        self.assertEqual(json.loads(json.dumps(options)), options)
        self.assertEqual(values(options["service_tier"]), ["default", "priority"])
        with self.assertRaises(TypeError):
            reasoning_options("codex", "openai_responses", "gpt-6.1-sol")

# ==== tests/test_recovery_guidance.py ====

class RecoveryGuidanceTests(unittest.TestCase):
    def test_public_context_whitelist_and_full_attempt_budget(self):
        harness = SimpleNamespace(latest=dict(observation_id="obs-current", object_poses="forbidden"),
            action_budget=lambda: dict(native_steps_used=935, native_steps_limit=1050,
                native_steps_remaining=115, task_success="forbidden"),
            robot=Mock(), observer=Mock())
        context, prompt = recovery_guidance(harness, dict(status="away", task_score="forbidden"))
        self.assertEqual(context["observation_id"], "obs-current")
        self.assertEqual(context["action_budget"]["native_steps_remaining"], 115)
        self.assertEqual(context["measured_home_status"], "away")
        self.assertNotIn("forbidden", prompt)
        self.assertIn("release or verification", prompt)
        self.assertIn("withdrawal", prompt)
        self.assertIn("required final return", prompt)
        self.assertIn("one concrete recovery objective", prompt)
        self.assertIn("complete submitted batch", prompt)
        self.assertIn("settling has a separate additional step limit", prompt)
        self.assertNotIn("single-segment", prompt)
        harness.robot.assert_not_called()
        harness.observer.assert_not_called()

    def test_unknown_is_not_zero_and_fault_does_not_obscure_continuation(self):
        for budget in (None, lambda: {}, lambda: dict(native_steps_remaining=True),
                       Mock(side_effect=ValueError("missing budget"))):
            context, prompt = recovery_guidance(SimpleNamespace(action_budget=budget))
            self.assertIsNone(context["observation_id"])
            self.assertTrue(all(value is None for value in context["action_budget"].values()))
            self.assertEqual(context["measured_home_status"], "unknown")
            self.assertNotIn("native_steps_remaining\":0", prompt)

    def test_evidence_context_is_compact_and_does_not_generalize_failure(self):
        context, prompt = recovery_guidance(SimpleNamespace(), evidence=dict(
            version="recovery-evidence-v1", native_action_count=99, native_steps_remaining=11,
            tested_candidate_count=9, earlier_candidate_count=3, task_success="forbidden",
            tested_candidates=[dict(label=str(i), nominal_native_steps=30,
                exceeds_episode_budget=True, object_pose="forbidden") for i in range(9)]))
        self.assertEqual(context["version"], "recovery-guidance-v4")
        # The main request reminder supplies this once, alongside the current plan.
        self.assertNotIn(NEXT_STEP_INSTRUCTIONS, prompt)
        self.assertEqual(len(context["tested_recovery"]["tested_candidates"]), 6)
        self.assertNotIn("forbidden", prompt)
        self.assertIn("never all routes", prompt)
        self.assertIn("does not require consuming all remaining steps", prompt)

    def test_absent_evidence_preserves_previous_prompt(self):
        from ni_rua.cortex.supervision.recovery_guidance import INSTRUCTIONS
        context, prompt = recovery_guidance(SimpleNamespace(), evidence=None)
        self.assertEqual(context["version"], "recovery-guidance-v1")
        self.assertTrue(prompt.endswith(INSTRUCTIONS))
        self.assertNotIn(NEXT_STEP_INSTRUCTIONS, prompt)

# ==== tests/test_system_runtime.py ====

class SystemRuntimeTests(unittest.TestCase):
    @require(linux_runtime_tools, "publishes the Linux system runtime (ldd/ldconfig, glibc) into the sandbox")
    def test_grants_are_files_or_standard_library_namespaces_not_usr_roots(self):
        mounts = system_runtime_mounts()
        destinations = {str(destination): source for source, destination in mounts}
        for root in ('/usr', '/usr/local', '/usr/bin', '/usr/lib', '/bin', '/lib', '/lib64'):
            self.assertNotIn(root, destinations)
        for source, destination in mounts:
            self.assertTrue(source.is_absolute() and destination.is_absolute())
            self.assertEqual(source, source.resolve())
            self.assertNotIn('site-packages', destination.parts)
            self.assertNotIn('dist-packages', destination.parts)
            self.assertNotIn('/usr/local/src', str(destination))
            if source.is_dir():
                self.assertIn('python', str(source.parent))
        for command in ('sh', 'bash', 'env', 'find', 'sed', 'awk', 'python3'):
            self.assertIn('/usr/bin/' + command, destinations)
        self.assertTrue(any('sysconfigdata' in path for path in destinations))
        for soname in ('libstdc++.so.6', 'libpthread.so.0', 'libdl.so.2', 'librt.so.1'):
            self.assertTrue(any(Path(path).name == soname for path in destinations), soname)

    @require(linux_runtime_tools, "publishes the Linux system runtime (ldd/ldconfig, glibc) into the sandbox")
    def test_shell_pipelines_get_their_standard_tools_and_nothing_excluded(self):
        import os
        import shutil
        from ni_rua.cortex.session.system_runtime import EXCLUDED_COMMANDS, LOCALE_DIRECTORIES, SHELL_COMMANDS
        destinations = {str(destination): source for source, destination in system_runtime_mounts()}
        # Every listed command the host has is published under /usr/bin; the search and comparison tools that
        # shell pipelines assume are listed (their absence made every `grep` of the actor a wasted turn).
        for command in ('grep', 'diff', 'cmp', 'echo', 'test', 'seq', 'mktemp', 'sha256sum'):
            self.assertIn(command, SHELL_COMMANDS)
        for command in SHELL_COMMANDS:
            if shutil.which(command, path=os.defpath):
                self.assertIn('/usr/bin/' + command, destinations, command)
        self.assertIn('/usr/bin/grep', destinations)
        # `python` is the published system interpreter, never a different host python.
        self.assertEqual(destinations['/usr/bin/python'], destinations['/usr/bin/python3'])
        # Network, privilege, session and other-interpreter tools stay out, whatever the host has installed.
        from ni_rua.cortex.session.system_runtime import refuse_excluded
        refuse_excluded(SHELL_COMMANDS)
        with self.assertRaises(RuntimeError):   # the guard that runs at import and before every publication
            refuse_excluded((*SHELL_COMMANDS, 'curl'))
        for command in EXCLUDED_COMMANDS:
            self.assertNotIn('/usr/bin/' + command, destinations, command)
        self.assertTrue({'curl', 'ssh', 'sudo', 'setsid', 'perl'} <= EXCLUDED_COMMANDS)
        # The locale of the sandbox LANG is published as individual files (no directory grant).
        locale = next((Path(item) for item in LOCALE_DIRECTORIES if Path(item).is_dir()), None)
        if locale is not None:
            files = [path for path in destinations if Path(path).is_relative_to(locale)]
            self.assertIn(str(locale / 'LC_CTYPE'), files)
            self.assertTrue(all(destinations[path].is_file() for path in files))
            self.assertNotIn(str(locale), destinations)

    def test_stdlib_selection_excludes_synthetic_system_installed_applications(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary).resolve()  # the selection yields canonical paths
            for name in ('site-packages/robodojo', 'dist-packages/ni_rua', 'private_backend', 'encodings'):
                directory = root / name
                directory.mkdir(parents=True)
                (directory / '__init__.py').write_text('SYNTHETIC_PRIVATE_IMPLEMENTATION')
            (root / 'sitecustomize.py').write_text('SYNTHETIC_HOST_SETUP')
            (root / 'os.py').write_text('SYNTHETIC_STDLIB_FIXTURE')
            entries = list(stdlib_runtime_entries(root, {'os', 'encodings'}))
            self.assertEqual({entry.relative_to(root).as_posix() for entry in entries}, {'os.py', 'encodings'})

    @require(linux_runtime_tools, "publishes the Linux system runtime (ldd/ldconfig, glibc) into the sandbox")
    def test_library_dependencies_preserve_loader_alias_without_parent_grants(self):
        mounts = runtime_library_mounts([Path('/usr/bin/python3')])
        self.assertTrue(mounts)
        self.assertTrue(any('ld-linux' in str(source) for source, _ in mounts))
        for source, destination in mounts:
            self.assertTrue(source.is_file())
            self.assertEqual(source, source.resolve())
            self.assertNotIn(str(destination), ('/usr/lib', '/lib', '/lib64'))

    @require(linux_runtime_tools, "publishes the Linux system runtime (ldd/ldconfig, glibc) into the sandbox")
    def test_ldd_paths_with_spaces_remain_complete(self):
        with tempfile.TemporaryDirectory(prefix='runtime path ') as temporary:
            root = Path(temporary)
            executable, library = root / 'program', root / 'shared library.so'
            executable.write_bytes(b'\x7fELFsynthetic')
            library.write_bytes(b'synthetic shared library')
            result = SimpleNamespace(returncode=0, stdout=' libfixture.so => ' + str(library) + ' (0x1234)\n', stderr='')
            with mock.patch('ni_rua.cortex.session.system_runtime.subprocess.run', return_value=result):
                mounts = runtime_library_mounts([executable])
            self.assertEqual(mounts, ((library, library),))

# ==== tests/test_terminal.py ====

class TerminalTest(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_environment_does_not_inherit_host_variables(self):
        for environment in (None, {}, {"HOME": "/run-home", "CODEX_HOME": "/run-codex-home"}):
            with self.subTest(environment=environment), tempfile.TemporaryDirectory() as directory:
                code = ("import json,os; print(json.dumps({key: os.environ.get(key) for key in "
                        "['NI_RUA_HOST_SECRET', 'HOME', 'CODEX_HOME', 'TERM', 'COLORTERM']}))")
                terminal = Terminal(directory, directory, (sys.executable, "-c", code), env=environment)
                try:
                    with mock.patch.dict(os.environ, {"NI_RUA_HOST_SECRET": "host-only",
                            "HOME": "/host-home", "CODEX_HOME": "/host-codex-home"}):
                        await terminal.start("unix:///fixture.sock", "thread-fixture")
                    await asyncio.wait_for(terminal._exit_task, 3)
                    observed = json.loads(terminal.read()[0])
                    self.assertEqual(observed["NI_RUA_HOST_SECRET"], "host-only" if environment is None else None)
                    self.assertEqual(observed["HOME"], "/host-home" if environment is None else environment.get("HOME"))
                    self.assertEqual(observed["CODEX_HOME"], "/host-codex-home" if environment is None else environment.get("CODEX_HOME"))
                    self.assertEqual((observed["TERM"], observed["COLORTERM"]), ("xterm-256color", "truecolor"))
                finally:
                    await terminal.close()

    async def test_partial_startup_drain_does_not_block_on_empty_pty(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal = Terminal(directory, directory)
            master, slave = pty.openpty()
            terminal.master = master
            try:
                self.assertTrue(os.get_blocking(master))
                # Verify the syscall precondition without making a bad version
                # hang the suite inside a blocking os.read.
                from unittest import mock
                original_read = os.read
                def checked_read(fd, count):
                    self.assertFalse(os.get_blocking(fd))
                    return original_read(fd, count)
                with mock.patch('ni_rua.cortex.session.terminal.os.read', checked_read):
                    terminal._receive()
                self.assertEqual(terminal.cursor, 0)
            finally:
                os.close(slave)
                await terminal.close()

    async def test_durable_shared_pty_without_new_prompt(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "native_tui_fixture.py"
            fixture.write_text(
                "import json,os,sys,time\n"
                "assert os.isatty(0) and os.isatty(1)\n"
                "with open('/dev/tty', 'rb'): pass\n"
                "print(json.dumps(sys.argv[1:]),flush=True)\n"
                "time.sleep(.1)\n"
                "os.write(1, '\\x1b[32m未来输出\\x1b[0m'.encode())\n"
                "time.sleep(30)\n"
            )
            terminal = Terminal(directory, directory, (sys.executable, str(fixture)),
                                env={"TERM": "xterm", "COLORTERM": "fixture"})
            first = await terminal.start("unix:///fixture.sock", "thread-fixture")
            try:
                duplicate = await terminal.start("unix:///fixture.sock", "thread-fixture")
                self.assertEqual(first["pid"], duplicate["pid"])
                with self.assertRaises(ValueError):
                    await terminal.start("unix:///other.sock", "thread-fixture")

                async def observe():
                    data = b""
                    async for chunk, cursor in terminal.stream():
                        data += chunk
                        if "未来输出".encode() in data:
                            return data, cursor
                a, b = await asyncio.wait_for(asyncio.gather(observe(), observe()), 3)
                self.assertEqual(a, b)
                args = json.loads(a[0].splitlines()[0])
                self.assertEqual(args, ["resume", "--remote", "unix:///fixture.sock", "--no-alt-screen", "thread-fixture"])
                saved, cursor = terminal.read()
                self.assertEqual((saved, cursor), a)
                tail, end = terminal.read(cursor)
                self.assertEqual((tail, end), (b"", cursor))
                terminal.resize(80, 24)
                self.assertEqual(terminal.status()["cols"], 80)
                with self.assertRaises(ValueError):
                    terminal.resize(0, 24)
                with self.assertRaises(ValueError):
                    terminal.read(cursor + 1)
                # Disconnecting one reader does not own or close the shared PTY.
                subscriber = terminal.stream(cursor)
                waiting = asyncio.create_task(anext(subscriber))
                await asyncio.sleep(.01)
                waiting.cancel()
                await asyncio.gather(waiting, return_exceptions=True)
                await subscriber.aclose()
                self.assertIsNone(terminal.process.poll())
                self.assertEqual(terminal.status()["pid"], first["pid"])
                reconnect = terminal.stream(0)
                replay_chunk, replay_cursor = await anext(reconnect)
                await reconnect.aclose()
                self.assertEqual((replay_chunk, replay_cursor), a)
                prefix, partial_cursor = terminal.read(0, 7)
                tail, end = terminal.read(partial_cursor)
                self.assertEqual(prefix + tail, a[0])
                self.assertEqual(end, cursor)
            finally:
                await terminal.close()
            recovered = Terminal(directory, directory)
            self.assertEqual(recovered.read(), a)
            self.assertIsNotNone(terminal.status()["exit_code"])
            self.assertEqual(recovered.status()["exit_code"], terminal.status()["exit_code"])

    async def test_native_bootstrap_failure_is_preserved_and_persisted_on_exit(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "failed_tui.py"
            # This is the exact native error observed in the archived run, split
            # across writes so diagnosis cannot depend on PTY chunk boundaries.
            chunks = [b"\x1b[0mError: account/read failed during TUI ",
                      b"bootstrap: account/read failed: ",
                      b"workspace routing discovery failed (code -32603)\n\x1b[0 q"]
            fixture.write_text("import os,time\n" + "".join(
                f"os.write(1, {chunk!r}); time.sleep(.05)\n" for chunk in chunks
            ) + "raise SystemExit(1)\n")
            terminal = Terminal(directory, directory, (sys.executable, str(fixture)))
            await terminal.start("unix:///fixture.sock", "thread-fixture")
            metadata = Path(directory) / "terminal.json"

            async def persisted_exit():
                while True:
                    status = json.loads(metadata.read_text())
                    if status["exit_code"] is not None:
                        return status
                    await asyncio.sleep(.01)

            try:
                # No supervisor close is needed to make natural exit durable.
                status = await asyncio.wait_for(persisted_exit(), 3)
                self.assertEqual(status["exit_code"], 1)
                self.assertTrue(status["bootstrap_failed"])
                self.assertEqual(status["error"]["code"], "workspace_routing_discovery_failed")
                expected = b"".join(chunks).replace(b"\n", b"\r\n")
                self.assertEqual(terminal.read(), (expected, len(expected)))
                self.assertEqual(status["cursor"], len(expected))
                self.assertEqual(status["thread_id"], "thread-fixture")
                recovered = Terminal(directory, directory)
                self.assertEqual(recovered.status()["exit_code"], 1)
                self.assertEqual(recovered.status()["error"], status["error"])
                # An old archive with no diagnostic fields is diagnosed in memory.
                old_status = {key: value for key, value in status.items()
                              if key not in {"bootstrap_failed", "error"}}
                metadata.write_text(json.dumps(old_status))
                recovered = Terminal(directory, directory)
                self.assertTrue(recovered.status()["bootstrap_failed"])
                self.assertEqual(recovered.read(), (expected, len(expected)))
                self.assertEqual(json.loads(metadata.read_text()), old_status)
            finally:
                await terminal.close()

            # A later successful attachment must not inherit an earlier failure
            # from the append-only terminal log.
            clean = Terminal(directory, directory, (sys.executable, "-c", "print('ready')"))
            try:
                await clean.start("unix:///fixture.sock", "thread-next")
                status = await asyncio.wait_for(persisted_exit(), 3)
                self.assertEqual(status["exit_code"], 0)
                self.assertFalse(status["bootstrap_failed"])
                self.assertIsNone(status["error"])
                self.assertFalse(Terminal(directory, directory).status()["bootstrap_failed"])
            finally:
                await clean.close()

    def test_bootstrap_diagnosis_requires_native_error_prefix(self):
        for text in (b"", b"ready", b"Error: unrelated failure",
                     b"workspace routing discovery failed",
                     b"account/read failed during TUI bootstrap: quoted text"):
            with self.subTest(text=text):
                self.assertEqual(terminal_bootstrap_status(text),
                                 {"bootstrap_failed": False, "error": None})
        generic = terminal_bootstrap_status(
            b"Error: account/read failed during TUI bootstrap: account unavailable")
        self.assertTrue(generic["bootstrap_failed"])
        self.assertEqual(generic["error"]["code"], "account_read_bootstrap_failed")

    async def test_eof_waits_for_exit_without_sharing_subscriber_cancellation(self):
        with tempfile.TemporaryDirectory() as directory:
            release = Path(directory) / "release"
            fixture = Path(directory) / "eof_before_exit.py"
            fixture.write_text(
                "import os,signal,time\nfrom pathlib import Path\n"
                "signal.signal(signal.SIGHUP, signal.SIG_IGN)\n"
                "os.write(1,b'last output')\n"
                "os.close(0); os.close(1); os.close(2)\n"
                f"while not Path({str(release)!r}).exists(): time.sleep(.01)\n"
                "raise SystemExit(7)\n"
            )
            terminal = Terminal(directory, directory, (sys.executable, str(fixture)))
            await terminal.start("unix:///fixture.sock", "thread-fixture")

            async def observe():
                data = b""
                async for chunk, _ in terminal.stream():
                    data += chunk
                return data, terminal.status()

            observers = [asyncio.create_task(observe()) for _ in range(2)]
            try:
                async def eof():
                    while terminal.master is not None:
                        await asyncio.sleep(.01)
                await asyncio.wait_for(eof(), 3)
                self.assertIsNone(terminal.process.poll())
                self.assertTrue(all(not observer.done() for observer in observers))
                observers[0].cancel()
                with self.assertRaises(asyncio.CancelledError):
                    await observers[0]
                self.assertFalse(terminal._exit_task.cancelled())
                self.assertFalse(observers[1].done())
                release.touch()
                data, final = await asyncio.wait_for(observers[1], 3)
                self.assertEqual(data, b"last output")
                self.assertEqual(final["exit_code"], 7)
                saved = json.loads((Path(directory) / "terminal.json").read_text())
                self.assertEqual(saved["exit_code"], 7)
                self.assertEqual(saved["cursor"], len(data))
            finally:
                release.touch()
                await terminal.close()
                for observer in observers:
                    observer.cancel()
                await asyncio.gather(*observers, return_exceptions=True)

    async def test_scroll_validates_bounds_and_synthesizes_readonly_escape_sequences(self):
        with tempfile.TemporaryDirectory() as directory:
            terminal = Terminal(directory, directory)
            for bad_kwargs in (
                {"lines": 0},
                {"lines": True},
                {"lines": False},
                {"lines": "3"},
                {"lines": 41},
                {"lines": -41},
                {"action": "enter"},
                {"action": "\x1b[A"},
                {"action": 123},
                {"lines": 2, "action": "page_up"},
            ):
                with self.subTest(bad=bad_kwargs), self.assertRaisesRegex(ValueError, "invalid_terminal_scroll"):
                    terminal.scroll(**bad_kwargs)
            with self.assertRaises(ValueError):
                terminal.resize(True, 24)
            # Unstarted terminal is a safe False no-op after validation.
            self.assertFalse(terminal.scroll(lines=-3))
            self.assertFalse(terminal.scroll(action="bottom"))

            master, slave = pty.openpty()
            tty.setraw(slave)
            terminal.master = master
            terminal.resize(100, 32)
            try:
                os.set_blocking(slave, False)
                self.assertTrue(terminal.scroll(lines=-2))
                self.assertTrue(terminal.scroll(lines=3))
                self.assertTrue(terminal.scroll(action="line_up"))
                self.assertTrue(terminal.scroll(action="line_down"))
                self.assertTrue(terminal.scroll(action="page_up"))
                self.assertTrue(terminal.scroll(action="page_down"))
                self.assertTrue(terminal.scroll(action="top"))
                self.assertTrue(terminal.scroll(action="bottom"))
                expected = (
                    (b"\x1b[<64;50;10M" * 2)
                    + (b"\x1b[<65;50;10M" * 3)
                    + b"\x1b[<64;50;10M"
                    + b"\x1b[<65;50;10M"
                    + b"\x1b[5~"
                    + b"\x1b[6~"
                    + b"\x1b[H"
                    + b"\x1b[F"
                )
                # The PTY line discipline may deliver the writes in several chunks under load: read until the
                # expected byte count arrives (bounded), then compare the whole stream once.
                received = b""
                deadline = time.monotonic() + 5
                while len(received) < len(expected) and time.monotonic() < deadline:
                    if select.select([slave], [], [], 0.05)[0]:
                        received += os.read(slave, 4096)
                self.assertEqual(received, expected)
                # Partial mid-sequence PTY write preserves unwritten suffix without tearing escape sequences.
                writes = []
                scripted = iter([5, BlockingIOError(errno.EAGAIN, "full"), BlockingIOError(errno.EAGAIN, "full"), 7, 4])

                def partial_write(_fd, chunk):
                    step = next(scripted)
                    if isinstance(step, BaseException):
                        raise step
                    writes.append(bytes(chunk[:step]))
                    return step

                with mock.patch("ni_rua.cortex.session.terminal.os.write", side_effect=partial_write):
                    self.assertTrue(terminal.scroll(lines=-2))
                    self.assertEqual(terminal._pending_scroll, b";50;10M")
                    # While the suffix is blocked, new scroll calls shed cleanly without interleaving mid-sequence.
                    self.assertFalse(terminal.scroll(action="page_up"))
                    self.assertEqual(terminal._pending_scroll, b";50;10M")
                    # Once writable again, the suffix drains intact before the new sequence is written.
                    self.assertTrue(terminal.scroll(action="page_up"))
                    self.assertEqual(terminal._pending_scroll, b"")
                self.assertEqual(b"".join(writes), b"\x1b[<64;50;10M\x1b[5~")

                # Exited or closed terminal returns False without writing.
                terminal._exit_code = 0
                self.assertFalse(terminal.scroll(lines=-1))
                terminal._exit_code = None
                with mock.patch("ni_rua.cortex.session.terminal.os.write", side_effect=OSError(errno.EIO, "closed")):
                    self.assertFalse(terminal.scroll(action="page_down"))
            finally:
                os.close(slave)
                await terminal.close()
            self.assertFalse(terminal.scroll(action="top"))

    async def test_close_handles_sigkill_process_lookup_race_and_start_cleans_pty_fds(self):
        import subprocess
        with tempfile.TemporaryDirectory() as directory:
            # Invalid cols/rows in terminal.json are clamped/ignored instead of leaking PTY FDs on start().
            (Path(directory) / "terminal.json").write_text(json.dumps({"cols": -5, "rows": 99999}))
            terminal = Terminal(directory, directory, (sys.executable, "-c", "pass"))
            self.assertEqual((terminal.cols, terminal.rows), (100, 32))

            # Resize failure during start() closes both master and slave FDs and resets terminal.master.
            closed = []
            real_close = os.close
            with mock.patch("ni_rua.cortex.session.terminal.pty.openpty", return_value=(801, 802)), \
                    mock.patch.object(terminal, "resize", side_effect=OSError("ioctl_failed")), \
                    mock.patch("ni_rua.cortex.session.terminal.os.close",
                               side_effect=lambda fd: closed.append(fd) if fd in (801, 802) else real_close(fd)):
                with self.assertRaises(OSError):
                    await terminal.start("unix:///fixture.sock", "thread-fixture")
            self.assertEqual(sorted(closed), [801, 802])
            self.assertIsNone(terminal.master)

            # ProcessLookupError on SIGKILL after TimeoutExpired in close() still closes FDs and sets changed.
            master, slave = pty.openpty()
            os.close(slave)
            terminal.master = master
            terminal.log = (Path(directory) / "terminal.bin").open("ab")
            fake_proc = mock.MagicMock()
            fake_proc.pid = 77777
            fake_proc.poll.return_value = None
            fake_proc.wait.side_effect = [subprocess.TimeoutExpired("cmd", 5), 0]
            terminal.process = fake_proc
            terminal.changed.clear()
            with mock.patch("ni_rua.cortex.session.terminal.os.killpg", side_effect=ProcessLookupError):
                await terminal.close()
            self.assertIsNone(terminal.master)
            self.assertIsNone(terminal.log)
            self.assertTrue(terminal.changed.is_set())


if __name__ == "__main__":
    unittest.main()
