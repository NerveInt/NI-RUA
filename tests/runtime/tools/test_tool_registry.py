"""Actual catalog/invoke/program boundaries, typed evidence and plugin extension."""
from support.program_fixture import program_services
from ni_rua.core.presentation.actor_payload import lean_v2_display
from copy import deepcopy
from contextlib import closing
import hashlib
import json
import time
import threading
from pathlib import Path
import re
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ni_rua.core.config import Config
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs.program_review import action_evidence, live_observation_id, reviewed_plan
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.core.storage.store import Store
from ni_rua.core.tool_errors import ToolInputError
from ni_rua.core.tool_output import StructuredText, ToolOutput, ToolResponse, wire_response
from ni_rua.runtime.tools.builtins import create_registry, output_schemas
from ni_rua.runtime.tools.definitions import TOOLS, public_definitions
from ni_rua.runtime.tools.registry import ToolRegistry, ToolSpec, validator_for
from support.motion_fixtures import RETIRED_TOOLS
from support.fixture_robot import FixtureRobot

NI_RUA_TEST_TIER = "component"


class ToolRegistryTests(unittest.TestCase):
    def test_public_descriptions_do_not_recommend_retired_tools(self):
        legacy_compare = deepcopy(next(item for item in TOOLS if item['name'] == 'vision_compare'))
        definitions = public_definitions()
        for definition in definitions:
            for retired in RETIRED_TOOLS:
                with self.subTest(tool=definition['name'], retired=retired):
                    self.assertIsNone(re.search(r'\b' + re.escape(retired) + r'\b',
                                                definition['description']))
        compare = next(item for item in definitions if item['name'] == 'vision_compare')
        self.assertIn('optional vision_review when available', compare['description'])
        def validation_schema(value):
            if isinstance(value, dict):
                return {key: validation_schema(item) for key, item in value.items()
                        if key not in {'description', 'examples'}}
            if isinstance(value, list):
                return [validation_schema(item) for item in value]
            return value
        self.assertEqual(validation_schema(compare['inputSchema']),
                         validation_schema(legacy_compare['inputSchema']))
        self.assertEqual(next(item for item in TOOLS if item['name'] == 'vision_compare'), legacy_compare)
        names = [item['name'] for item in definitions]
        self.assertFalse(set(names) & RETIRED_TOOLS)
        self.assertFalse(any(name.startswith('scene_') for name in names))
        self.assertEqual(names.count('robot_action'), 1)
        plan_tool = next(item for item in definitions if item['name'] == 'task_plan')
        self.assertEqual(plan_tool['inputSchema']['properties']['plan']['type'], 'string')
        program_tool = next(item for item in definitions if item['name'] == 'program_run')
        self.assertNotIn('max_calls', program_tool['inputSchema']['properties'])

    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.store = Store(self.root/'run', dict(run_id='registry', episode_id='ep',
            env_instance_id='env', env_generation=1))
        self.addCleanup(self.store.close)
        self.robot = FixtureRobot(self.root)
        self.harness = Harness(self.store, self.robot)
        self.harness.thread_id = 'thread'
        self.harness.capabilities = {'fixture': True}
        self.calls = []
        self.plugin = ToolSpec(dict(type='function', name='fixture_read', description='Read one fixture value.',
            inputSchema=dict(type='object', properties={'value': {'type': 'integer'}},
                required=['value'], additionalProperties=False)), self.read,
            output_schemas={'data': dict(type='object', properties={'value': {'type': 'integer'}},
                required=['value'], additionalProperties=False, examples=[{'value': 7}])})
        self.harness.tool_registry.register(self.plugin)

    def read(self, runtime, args, correlation):
        self.calls.append(deepcopy(args))
        return ToolOutput({'value': args['value']}).response()

    def archived_results(self, store=None):
        store = self.store if store is None else store
        return [(record, json.loads(store.resolve_artifact(record['raw_ref'],
                    allowed_scopes=('observer',))[1]))
                for record in store.records('tool_result') if record.get('raw_ref')]

    def test_robot_observe_claims_delta_delivery_only_in_a_delta_run(self):
        """A run whose backend gets full delivery (the Harness repeats static observation metadata in every reply)
        is not told that summary leaves unchanged calibration out; a delta run keeps the Codex catalogue bytes."""
        from ni_rua.core.contracts import canonical
        from ni_rua.runtime.tools.definitions import descriptions
        delta, full = descriptions(), descriptions(context_delivery="full")
        self.assertEqual({name for name in delta if delta[name] != full[name]}, {"robot_observe"})
        # Nothing is claimed instead: the full-delivery text is the delta text without its last sentence.
        self.assertTrue(delta["robot_observe"].startswith(full["robot_observe"]))
        self.assertLess(len(full["robot_observe"]), len(delta["robot_observe"]))

        def catalogue(agent):
            # the prose catalogue: this test is about the delivery mode only (the catalogue forms are compared in
            # tool_contracts_fast.ToolCatalogProfileTests, and external gets compact_v1 under a layered profile)
            config = Config.from_dict({"agent": agent, "features": {"tool_catalog_profile": "full_v1"}})
            return config.agent.context_delivery, Workbench.actor_tools(SimpleNamespace(config=config))

        codex_mode, codex = catalogue({})
        self.assertEqual(codex_mode, "delta")
        self.assertEqual(canonical(codex), canonical(Workbench.actor_tools(
            SimpleNamespace(config=Config.from_dict({"features": {"tool_catalog_profile": "full_v1"}})))))
        for agent, expected in (
                ({"backend": "opencode"}, "delta"),                                  # 四后端默认均为 delta
                ({"backend": "opencode", "opencode": {"subagents": False}}, "delta"),
                ({"backend": "opencode", "opencode": {"context_delivery": "full"}}, "full"),
                ({"backend": "opencode", "opencode": {"subagents": False, "context_delivery": "full"}}, "full"),
                ({"backend": "claude_code"}, "delta"),
                ({"backend": "claude_code", "claude_code": {"context_delivery": "full"}}, "full"),
                ({"backend": "claude_code", "claude_code": {"subagents": False, "context_delivery": "full"}}, "full"),
                ({"backend": "external"}, "delta"),
                ({"backend": "external", "external": {"context_delivery": "full"}}, "full")):
            with self.subTest(agent=agent):
                mode, tools = catalogue(agent)
                self.assertEqual(mode, expected)
                changed = {new["name"] for old, new in zip(codex, tools, strict=True) if canonical(old) != canonical(new)}
                self.assertEqual(changed, {"robot_observe"} if expected == "full" else set())

    def test_robot_capabilities_isolated_and_stale_epoch_do_not_pollute_baseline(self):
        # 主链先建立能力基线
        first = json.loads(self.harness.handle_tool(dict(
            threadId="thread", turnId="turn", callId="cap-main-1",
            tool="robot_capabilities", arguments=dict(detail="summary"),
            contextScope="main",
        ))["contentItems"][0]["text"])
        self.assertEqual(first["capabilities_delivery"]["status"], "inline")

        # isolated 与过时 contextEpoch 均返回完整 inline 且不改写基线
        iso = json.loads(self.harness.handle_tool(dict(
            threadId="thread", turnId="turn", callId="cap-iso",
            tool="robot_capabilities", arguments=dict(detail="summary"),
            contextScope="isolated",
        ))["contentItems"][0]["text"])
        self.assertEqual(iso["capabilities_delivery"]["status"], "inline")
        self.assertTrue(iso["fixture"])

        stale = json.loads(self.harness.handle_tool(dict(
            threadId="thread", turnId="turn", callId="cap-stale",
            tool="robot_capabilities", arguments=dict(detail="summary"),
            contextScope="main", contextEpoch=("stale",),
        ))["contentItems"][0]["text"])
        self.assertEqual(stale["capabilities_delivery"]["status"], "inline")
        self.assertTrue(stale["fixture"])

        # 后续主链仍命中初次基线
        second = json.loads(self.harness.handle_tool(dict(
            threadId="thread", turnId="turn", callId="cap-main-2",
            tool="robot_capabilities", arguments=dict(detail="summary"),
            contextScope="main",
        ))["contentItems"][0]["text"])
        self.assertEqual(second["capabilities_delivery"]["status"], "unchanged")

    def test_native_function_definition_is_checked_before_registration(self):
        from dataclasses import replace
        bad_fields = [('type', 'namespace'), ('type', 7), ('name', None), ('name', []),
                      ('description', 5), ('description', {}), ('deferLoading', 'true')]
        for field, value in bad_fields:
            definition = dict(self.plugin.definition, **{field: value})
            with self.subTest(field=field, value=value), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=definition)])
            self.assertEqual(caught.exception.code, 'invalid_tool_definition')
            self.assertEqual(caught.exception.field_path, '/' + field)
        for field in ('type', 'name', 'description', 'inputSchema'):
            definition = deepcopy(self.plugin.definition)
            del definition[field]
            with self.subTest(missing=field), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=definition)])
            self.assertEqual(caught.exception.field_path, '/' + field)
        for schema in ({}, True, False):
            definition = dict(self.plugin.definition, inputSchema=schema, deferLoading=False,
                              extension_metadata={'source_url': 'https://example.test/author'})
            registry = ToolRegistry([replace(self.plugin, definition=definition)])
            self.assertEqual(registry.catalog(self.harness), [definition])
        from jsonschema.exceptions import SchemaError
        for schema in (None, [], {'type': 'integer', 'minimum': 'bad'}):
            with self.subTest(schema=schema), self.assertRaises(SchemaError):
                ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, inputSchema=schema))])

    def test_native_admission_constraints_reject_unsupported_function_declarations(self):
        from dataclasses import replace
        for name in ('', ' ', 'leading ', ' trailing', 'newline\n', 'dot.name', 'slash/name',
                     'unicode_中文', 'control\x00', 'x'*129, 'mcp', 'mcp__plugin'):
            with self.subTest(name=name), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, name=name))])
            self.assertEqual(caught.exception.field_path, '/name')
        for name in ('x'*128, 'legal-tool_0', '_', 'MCP', 'mcp_plugin'):
            registry = ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, name=name))])
            self.assertEqual(registry.names, frozenset({name}))
        for field, value in [('namespace', None), ('namespace', 'plugin'), ('exposeToContext', True),
                             ('exposeToContext', False)]:
            with self.subTest(field=field, value=value), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, **{field: value}))])
            self.assertEqual(caught.exception.field_path, '/' + field)
        for schema in ({'type': 'null'}, {'type': ['null']}):
            with self.subTest(schema=schema), self.assertRaises(ToolInputError) as caught:
                ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition, inputSchema=schema))])
            self.assertEqual(caught.exception.code, 'unsupported_tool_input_schema')
            self.assertEqual(caught.exception.field_path, '/inputSchema/type')
        registry = ToolRegistry([replace(self.plugin, definition=dict(self.plugin.definition,
            inputSchema={'type': ['object', 'null']}))])
        self.assertEqual(registry.names, frozenset({'fixture_read'}))

    def test_all_public_definitions_are_admitted_by_native_app_server_without_inference(self):
        from ni_rua.cortex.agents.codex.session import CodexSession
        source = self.root / 'native-source'
        source.mkdir()
        (source / 'config.toml').write_text('model_provider = "fixture"\n'
            '[model_providers.fixture]\nbase_url = "http://127.0.0.1:1/v1"\n')
        config = Config.from_dict({'model': {'codex_home': str(source)},
            'features': {'observation_profile': 'rgbd_proprio_calibrated_v1'}})
        runtime = SimpleNamespace(config=config, disabled_tools=(),
            observation_profile=config.features.observation_profile,
            capabilities={'unified_action': {'available': True}}, enable_wait=True,
            vision_compare=True, visual_reviewer=object(), action_policy=object(), programs=object())
        catalog = create_registry().catalog(runtime)
        self.assertEqual(len(catalog), 20)
        session = CodexSession(self.root / 'native-startup', self.store, lambda _: {},
            model='gpt-6.1-sol', effort='high', config=config)
        def close_runtime():
            bridge = session.client._proc if session.client else None
            session.close()
            if bridge:
                for stream in (bridge.stdin, bridge.stdout, bridge.stderr):
                    if stream and not stream.closed:
                        stream.close()
        self.addCleanup(close_runtime)
        self.assertTrue(session.start(catalog, 'Tool catalog admission only; no model turn.'))
        self.assertEqual(session.state, 'idle')
        self.assertIsNone(session.process.poll())
        self.assertEqual(session.model_proxy.request_count, 0)
        self.assertEqual(session.model_proxy.attempt_count, 0)

    def test_deferred_publication_keeps_schema_roles_and_availability_authoritative(self):
        from dataclasses import replace
        definition = deepcopy(self.plugin.definition)
        definition['deferLoading'] = True
        definition['inputSchema']['properties']['value'].update(minimum=0, maximum=10,
            description='Integer fixture value from 0 through 10.')
        registry = ToolRegistry([replace(self.plugin, definition=definition)])
        self.assertEqual(registry.catalog(self.harness), [definition])
        self.assertEqual(registry.invoke('fixture_read', {'value': 7}, {},
            runtime=self.harness, role='actor').data, {'value': 7})
        with self.assertRaises(ToolInputError):
            registry.invoke('fixture_read', {'value': 11}, {}, runtime=self.harness, role='actor')
        self.assertEqual(self.calls, [{'value': 7}])
        restricted = ToolRegistry([replace(self.plugin, definition=definition, roles=frozenset({'program'}))])
        self.assertEqual(restricted.catalog(self.harness), [])
        with self.assertRaises(ToolInputError) as caught:
            restricted.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
        self.assertEqual(caught.exception.code, 'tool_not_permitted_for_role')
        self.harness.disabled_tools = {'fixture_read'}
        self.assertEqual(registry.catalog(self.harness), [])
        with self.assertRaises(ToolInputError) as caught:
            registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
        self.assertEqual(caught.exception.code, 'tool_disabled_by_config:fixture_read')

    def test_plugin_input_examples_are_validated_at_registration(self):
        from dataclasses import replace
        definition = deepcopy(self.plugin.definition)
        definition['inputSchema']['examples'] = [{'value': 7}]
        valid = ToolRegistry([replace(self.plugin, definition=definition)])
        self.assertEqual(valid.catalog(self.harness)[0]['inputSchema']['examples'], [{'value': 7}])
        for example in ({'value': 'not an integer'}, {}, {'value': 7, 'extra': True}):
            definition['inputSchema']['examples'] = [example]
            with self.subTest(example=example), self.assertRaisesRegex(ValueError, 'invalid_tool_input_example'):
                ToolRegistry([replace(self.plugin, definition=definition)])

    def test_registered_contract_is_isolated_from_all_inspection_copies(self):
        from dataclasses import replace
        source = replace(self.plugin, roles={'actor', 'trusted_legacy'}, legacy_schema={'type': 'object'})
        registry = ToolRegistry([source])
        catalog = registry.catalog(self.harness)
        manifest = registry.manifest(self.harness)
        copies = [source, registry.spec('fixture_read', role='actor'),
                  registry.validate('fixture_read', {'value': 7}, role='actor'), registry._specs['fixture_read']]
        for copied in copies:
            copied.definition['name'] = 'changed-name'
            copied.definition['inputSchema']['properties']['value']['type'] = 'string'
            copied.output_schemas['data']['required'].append('missing')
            copied.legacy_schema['type'] = 'string'
        source.roles.add('program')
        registry._specs.clear()
        registry._specs['fixture_read'] = source
        edited_catalog = registry.catalog(self.harness)
        edited_catalog[0]['inputSchema'].clear()
        edited_manifest = registry.manifest(self.harness)
        edited_manifest['tools'][0]['output_schemas']['data'].clear()
        exposed_validator = validator_for(catalog[0]['inputSchema'])
        exposed_validator.schema['properties']['value']['type'] = 'string'
        self.assertEqual(registry.catalog(self.harness), catalog)
        self.assertEqual(registry.manifest(self.harness), manifest)
        registry.validate('fixture_read', {'value': 7}, role='actor')
        registry.validate('fixture_read', {'any_legacy_field': True}, role='trusted_legacy')
        with self.assertRaises(ToolInputError):
            registry.validate('fixture_read', {'value': 'text'}, role='actor')
        with self.assertRaises(ToolInputError):
            registry.spec('fixture_read', role='program')
        result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
        self.assertTrue(result.success)
        self.assertEqual(result.data, {'value': 7})

    def test_native_content_union_rejects_bad_parts_and_keeps_valid_neighbors(self):
        from dataclasses import replace
        valid = [dict(type='inputText', text='opaque {"hash":"business"}'),
                 dict(type='inputImage', imageUrl='data:image/png;base64,fixture'),
                 dict(type='inputAudio', audioUrl='data:audio/wav;base64,fixture')]
        for item, path in [(7, ''), (None, ''), ([], ''), ({}, '/type'),
                ({'type': []}, '/type'), ({'type': 'bogus', 'text': 'unsupported'}, '/type'),
                ({'type': 'inputText'}, '/text'), ({'type': 'inputText', 'text': 7}, '/text'),
                ({'type': 'inputText', 'text': True}, '/text'),
                ({'type': 'inputImage', 'imageUrl': 5}, '/imageUrl'),
                ({'type': 'inputImage', 'image_url': 'wrong spelling'}, '/imageUrl'),
                ({'type': 'inputAudio'}, '/audioUrl'), ({'type': 'inputAudio', 'audioUrl': []}, '/audioUrl')]:
            with self.subTest(item=item):
                original = ToolResponse([valid[0], item, *valid[1:]], data={'value': 7})
                registry = ToolRegistry([replace(self.plugin, handler=lambda *args: original)])
                result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
                self.assertIs(result.success, False)
                self.assertIs(result.data, original.data)
                self.assertEqual(result.contract_error['reason'], 'invalid_native_content_item')
                self.assertEqual(result.contract_error['field_path'], '/content/1' + path)
                shown = wire_response(result)
                self.assertEqual(shown['contentItems'][1:], valid)
                self.assertEqual(json.loads(shown['contentItems'][0]['text'])['output_contract_error'],
                                 result.contract_error)
        original = ToolResponse([StructuredText({'hash': 'business'}), *valid], data={'value': 7})
        registry = ToolRegistry([replace(self.plugin, handler=lambda *args: original)])
        result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
        self.assertIs(result, original)
        self.assertTrue(result.success)
        self.assertEqual(wire_response(result)['contentItems'][1:], valid)

    def test_other_output_failures_also_remove_illegal_native_parts(self):
        from dataclasses import replace
        for response, reason in [(ToolResponse([7], data={'value': 7}, success=1), 'result_success_not_boolean'),
                (ToolResponse([7], data={'value': 'bad'}), 'result_schema_mismatch'),
                (ToolResponse([7], data={'value': 7}, kind='unexpected'), 'unexpected_result_kind')]:
            registry = ToolRegistry([replace(self.plugin, handler=lambda *args: response)])
            result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
            self.assertEqual(result.contract_error['reason'], reason)
            shown = wire_response(result)
            self.assertIs(shown['success'], False)
            self.assertEqual(len(shown['contentItems']), 1)
            self.assertEqual(shown['contentItems'][0]['type'], 'inputText')

    def test_illegal_native_display_preserves_real_action_and_program_replay(self):
        from ni_rua.robot.motion.arx_actions import compile_action
        self.harness.start()
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolResponse([dict(type='inputImage', imageUrl=5)],
                kind='action', data=context.execute(action))
        definition = dict(type='function', name='fixture_illegal_native_display', description='Native output regression.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        args = {'observation_id': self.harness.latest['observation_id']}
        correlation = {'tool_call_id': 'illegal-native-once'}
        first = self.harness.invoke(definition['name'], args, correlation, role='actor')
        repeated = self.harness.invoke(definition['name'], args, correlation, role='actor')
        self.assertFalse(first.success)
        self.assertEqual(first.contract_error['reason'], 'invalid_native_content_item')
        self.assertEqual(first.contract_error['phase'], 'after_handler')
        self.assertEqual(first.data, repeated.data)
        self.assertEqual(self.robot.executions, 1)
        self.assertEqual(first.data['receipt'], self.store.read('action_receipt', first.data['request']['action_id']))
        record, archived = next((record, archived) for record, archived in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'illegal-native-once')
        self.assertEqual(record['archival_status'], 'complete')
        self.assertEqual(archived['wire_response']['contentItems'], [dict(type='inputImage', imageUrl=5)])
        self.assertEqual(archived['data'], first.data)
        self.assertEqual(archived['contract_error']['reason'], 'invalid_native_content_item')
        evidence = action_evidence(self.store, definition['name'], args, first, 'illegal-native-once', physical=True)
        self.assertEqual(evidence['status'], 'completed')
        self.assertFalse(evidence['dispatch_success'])
        shown = self.harness.actor_output.present(first)
        self.assertTrue(json.loads(shown['contentItems'][0]['text'])['output_contract_error']['physical_action_executed'])
        self.assertEqual(json.loads(shown['contentItems'][1]['text'])['receipt']['status'], 'completed')
        self.assertTrue(all(part['type'] == 'inputText' for part in shown['contentItems']))
        runner = ProgramRunner(program_services(self.harness))
        self.harness.programs = runner
        self.addCleanup(runner.close)
        request = dict(operation_id='execute-once', tool=definition['name'],
            arguments={'observation_id': self.harness.latest['observation_id']})
        source = ('from ni_rua_client import NiRuaClient\nc = NiRuaClient()\nrequest = ' + repr(request) +
                  '\nfirst = c._request(request)\nassert c._request(request) == first\nprint(first["success"])')
        job = runner.run(dict(source=source, allow_direct_actions=True),
            dict(thread_id='thread', turn_id='turn', tool_call_id='illegal-native-program'))
        self.assertEqual(job['status'], 'succeeded')
        self.assertEqual(job['call_summary']['calls'], 1)
        self.assertEqual(job['call_summary']['failed_calls'], 1)
        self.assertEqual(job['action_summary']['status_counts'], {'completed': 1})
        self.assertEqual(self.robot.executions, 2)

    def test_unknown_fields_reject_before_actor_and_program_handler(self):
        arguments = dict(detail='full', unexpected_argument=True)
        with self.assertRaises(ToolInputError) as error:
            self.harness.dispatch('robot_capabilities', arguments, {})
        self.assertEqual(error.exception.field_path, '/unexpected_argument')
        with self.assertRaises(ToolInputError):
            self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='bad',
                tool='fixture_read', arguments={'value': 1, 'extra': True}))
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        result = runner.run(dict(source="from ni_rua_client import NiRuaClient\n"
            "NiRuaClient().call('fixture_read', {'value': 1, 'extra': True})"),
            dict(thread_id='thread', turn_id='turn', tool_call_id='program-bad'))
        self.assertEqual(result['status'], 'succeeded')
        self.assertEqual(result['call_summary']['failed_calls'], 1)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.robot.executions, 0)

    def test_new_read_tool_advertises_and_invokes_through_both_real_ports(self):
        self.assertIn('fixture_read', {x['name'] for x in self.harness.tool_registry.catalog(self.harness)})
        response = self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='read',
            tool='fixture_read', arguments={'value': 7}))
        self.assertEqual(json.loads(response['contentItems'][0]['text']), {'value': 7})
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        result = runner.run(dict(source="from ni_rua_client import NiRuaClient\n"
            "print(NiRuaClient().call('fixture_read', {'value': 11})['success'])"),
            dict(thread_id='thread', turn_id='turn', tool_call_id='program-read'))
        self.assertEqual(result['status'], 'succeeded', result)
        self.assertEqual(result['call_summary']['failed_calls'], 0)
        self.assertEqual(self.calls, [{'value': 7}, {'value': 11}])
        self.assertEqual(self.robot.executions, 0)
        self.harness.disabled_tools = frozenset({'fixture_read'})
        self.assertNotIn('fixture_read', {x['name'] for x in self.harness.tool_registry.catalog(self.harness)})
        with self.assertRaisesRegex(ValueError, 'tool_disabled_by_config'):
            runner._invoke('fixture_read', {'value': 1}, {})

    def test_physical_plugin_cannot_bypass_program_invocation_authorization(self):
        executions = []
        def perform(runtime, args, correlation):
            executions.append(correlation['program_id'])
            return ToolOutput({'receipt': {'status': 'completed', 'executed_native_steps': 2}},
                kind='action').response()
        definition = dict(type='function', name='fixture_physical', description='Fixture physical port.',
            inputSchema=dict(type='object', properties={}, additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        source = "from ni_rua_client import NiRuaClient\nNiRuaClient().call('fixture_physical', {})"
        for authorized in (False, True):
            result = runner.run(dict(source=source, allow_direct_actions=authorized),
                dict(thread_id='thread', turn_id='turn', tool_call_id='physical-'+str(authorized)))
            self.assertEqual(result['status'], 'succeeded')
            self.assertEqual(result['call_summary']['failed_calls'], int(not authorized))
            if authorized:
                self.assertEqual(result['action_summary']['status_counts'], {'completed': 1})
                self.assertEqual(result['action_summary']['executed_native_steps_known'], 2)
            else:
                self.assertEqual(result['action_summary']['action_calls'], 0)
                self.assertEqual(executions, [])
        self.assertEqual(len(executions), 1)

    def test_plugin_ports_preserve_single_writer_admission_and_replay(self):
        self.harness.start()
        from ni_rua.robot.motion.arx_actions import compile_action
        contexts = []
        def perform(context, args, correlation):
            contexts.append(context)
            self.assertFalse(hasattr(context, 'robot'))
            self.assertFalse(hasattr(context, 'store'))
            action = compile_action('move', dict(observation_id=args['observation_id'],
                arm='left', position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]),
                context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_motion', description='Fixture writer port.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        args = {'observation_id': self.harness.latest['observation_id']}
        self.harness.accepting = False
        denied = self.harness.invoke('fixture_motion', args, {'tool_call_id': 'denied'}, role='actor')
        self.assertEqual(denied.data['receipt']['status'], 'rejected')
        self.assertEqual(self.robot.executions, 0)
        self.harness.accepting = True
        first = self.harness.invoke('fixture_motion', args, {'tool_call_id': 'once'}, role='actor')
        repeated = self.harness.invoke('fixture_motion', args, {'tool_call_id': 'once'}, role='actor')
        self.assertEqual(first.data, repeated.data)
        self.assertEqual(first.data['receipt']['status'], 'completed')
        self.assertEqual(self.robot.executions, 1)
        self.assertEqual(len(self.store.records('action_request')), 2)
        self.assertEqual(len(self.store.records('action_receipt')), 2)
        contexts[-1].identity['episode_id'] = 'local-copy'
        self.assertEqual(self.harness.identity['episode_id'], 'ep')
        read_contexts = []
        def read(context, arguments, correlation):
            read_contexts.append(context)
            return ToolOutput({}).response()
        readonly = deepcopy(definition)
        readonly['name'] = 'fixture_read_port'
        self.harness.tool_registry.register(ToolSpec(readonly, read, output_schemas={'data': {'type': 'object'}}))
        self.harness.invoke('fixture_read_port', args, {}, role='actor')
        self.assertIsNone(read_contexts[0].execute)
        reserved = deepcopy(readonly)
        reserved['name'] = 'scene_custom'
        with self.assertRaisesRegex(ValueError, 'reserved_tool_namespace'):
            self.harness.tool_registry.register(ToolSpec(reserved, read))

    def test_actor_action_status_preserves_request_and_failure_diagnostics_in_both_details(self):
        request = dict(action_id='history-action', based_on_observation_id='before',
            path=dict(waypoints=[dict(position_m=[.123456789, 0, 1])]))
        receipt = dict(status='stopped', error='motor_fault', executed_native_steps=3,
            executed_control_steps=4, stop_confirmation=dict(status='unknown'),
            execution_uncertain=True, future_diagnostic=dict(fault_code=7))
        self.store.save('action_request', request['action_id'], request)
        self.store.save('action_receipt', request['action_id'], receipt)
        self.harness.states[request['action_id']] = deepcopy(receipt)
        for detail in ('summary', 'full'):
            with self.subTest(detail=detail):
                reply = self.harness.handle_tool(dict(threadId='thread', turnId='turn',
                    callId='history-' + detail, tool='robot_action_status',
                    arguments=dict(action_id=request['action_id'], detail=detail)))
                self.assertTrue(reply['success'])
                result = json.loads(reply['contentItems'][0]['text'])
                # full: the stored numbers; summary: the default display (lean_v2), floats with six decimals
                shown = request if detail == 'full' else lean_v2_display(request)
                self.assertEqual(result['request'], shown)
                self.assertEqual(result['request']['path']['waypoints'][0]['position_m'][0],
                                 .123456789 if detail == 'full' else .123457)
                self.assertEqual(result['receipt'], receipt)
                if detail == 'summary':
                    # robot_action_status is loaded up front on every host: the reply names it itself.
                    self.assertIn('robot_action_status', self.harness.tool_registry.loaded_tool_names(self.harness))
                    named = dict(tool='robot_action_status', arguments=dict(action_id=request['action_id'], detail='full'))
                    self.assertEqual(result['detail'], named)
                    self.assertNotIn('full_result', result)
                    # A host that publishes every tool eagerly gets the same entry.
                    self.harness.deferred_tool_loading = False
                    eager = json.loads(self.harness.handle_tool(dict(threadId='thread', turnId='turn',
                        callId='history-eager', tool='robot_action_status',
                        arguments=dict(action_id=request['action_id'])))['contentItems'][0]['text'])
                    self.harness.deferred_tool_loading = True
                    self.assertEqual(eager['detail'], named)
                    self.assertNotIn('full_result', eager)
        self.assertEqual(self.store.read('action_receipt', request['action_id']), receipt)
        self.assertEqual(self.robot.executions, 0)

    def test_program_plugin_evidence_uses_request_action_id_without_changing_receipt(self):
        from ni_rua.robot.motion.arx_actions import compile_action
        self.harness.start()
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_request_id', description='Physical ID fixture.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        args = dict(observation_id=self.harness.latest['observation_id'])
        job = runner.run(dict(source='from ni_rua_client import NiRuaClient\n'
            'assert NiRuaClient().call("fixture_request_id", ' + repr(args) + ')["success"]',
            allow_direct_actions=True), dict(thread_id='thread', turn_id='turn', tool_call_id='request-id'))
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        request = self.store.records('action_request')[0]
        receipt = self.store.read('action_receipt', request['action_id'])
        self.assertNotIn('action_id', receipt)
        evidence = job['action_summary']['recent_actions'][0]
        self.assertEqual(evidence['action_id'], request['action_id'])
        self.assertEqual(evidence['status'], receipt['status'])
        self.assertEqual(self.robot.executions, 1)

    def test_interaction_history_actor_and_real_program_share_evidence_without_execution(self):
        self.harness.start()
        raw = self.robot.observe()
        raw['robot_state'].update(sim_step=raw['sim_step'], step_domain=raw['step_domain'])
        observation = self.harness.publish_observation(raw)
        request = dict(self.harness.identity, action_id='history-interaction',
            based_on_observation_id=observation['observation_id'],
            gripper_events=[dict(arm='left', goal_normalized=0.)], input_digest='a'*64)
        receipt = dict(status='rejected', reason='fixture-rejection', executed_native_steps=0,
                       execution_start_observation_id=None, result_observation_id=None)
        self.store.save('action_request', request['action_id'], request)
        self.harness.states[request['action_id']] = deepcopy(receipt)
        response = self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='interactions',
            tool='history_list_actions', arguments=dict(anchor='events')))
        self.assertTrue(response['success'])
        self.assertEqual([part['type'] for part in response['contentItems']], ['inputText'])
        text = response['contentItems'][0]['text']
        self.assertNotIn('a'*64, text)
        result = json.loads(text)
        self.assertEqual(result['actions'][0]['receipt'], receipt)
        self.assertEqual(next(iter(result['observations'].values()))['status'], 'available')
        self.assertEqual(result['actions'][0]['omitted_actions_before'], 0)
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        source = ('from ni_rua_client import NiRuaClient\n'
                  'r = NiRuaClient().call("history_list_actions", {"anchor": "events"})\n'
                  'assert r["success"] and r["kind"] == "data"\n'
                  'b = r["data"]["actions"][0]\n'
                  'assert b["receipt"]["status"] == "rejected"\n'
                  'assert b["receipt"]["executed_native_steps"] == 0\n'
                  'assert b["observation_ids"]["result"] is None\n'
                  'assert "input_digest" not in b["request"]\n'
                  'assert "result_observation_not_recorded" in b["evidence_gaps"]\n')
        job = runner.run(dict(source=source), dict(thread_id='thread', turn_id='turn', tool_call_id='history-program'))
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        self.assertEqual(self.robot.executions, 0)
        self.assertEqual(self.harness.latest, observation)
        self.assertEqual(self.harness.states[request['action_id']], receipt)

    def test_interaction_history_invalid_anchor_rejects_before_handler(self):
        from unittest.mock import patch
        with patch('ni_rua.core.presentation.history_output.interaction_blocks') as query:
            for arguments in (dict(anchor='grasped'), dict(anchor=True), dict(anchor='all', limit=0)):
                with self.subTest(arguments=arguments), self.assertRaises(ToolInputError):
                    self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId=str(arguments),
                        tool='history_list_actions', arguments=arguments))
            query.assert_not_called()
        self.assertEqual(self.robot.executions, 0)

    def test_wait_image_failures_preserve_actor_and_program_action_evidence(self):
        self.harness.start()
        self.harness.capabilities['bounded_wait'] = dict(max_native_steps=50)
        self.harness.capabilities['path_kinds'].append('hold')
        original_execute = self.robot.execute
        def execute(action, progress_callback=None, cancel_flag=None):
            progress_callback(dict(executed_steps=10, observation=self.robot.observe()))
            receipt = original_execute(action, progress_callback, cancel_flag)
            receipt.update(executed_native_steps=20, completed_native_steps=20)
            return receipt
        self.robot.execute = execute
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        original_content = self.harness.observation_content
        for role in ('actor', 'program'):
            for fault in ('temporal_sample', 'final_frame'):
                with self.subTest(role=role, fault=fault):
                    action_id = role + '-' + fault
                    args = dict(observation_id=self.harness.latest['observation_id'],
                        native_steps=20, action_id=action_id)
                    attempts = []
                    def content(bundle, correlation, *args, **kwargs):
                        temporal = kwargs.get('metadata_delivery') is False
                        attempts.append('temporal_sample' if temporal else 'final_frame')
                        if temporal == (fault == 'temporal_sample'):
                            raise OSError('fixture_image_read_failed')
                        return original_content(bundle, correlation, *args, **kwargs)
                    before = self.robot.executions
                    with patch.object(self.harness, 'observation_content', side_effect=content):
                        if role == 'actor':
                            params = dict(threadId='thread', turnId='turn', callId=action_id,
                                tool='robot_wait', arguments=args)
                            reply = self.harness.handle_tool(params)
                            self.assertEqual(self.harness.handle_tool(params), reply)
                        else:
                            request = dict(operation_id='wait-once', tool='robot_wait', arguments=args)
                            source = ('from ni_rua_client import NiRuaClient\nc = NiRuaClient()\nrequest = ' +
                                repr(request) + '\nfirst = c._request(request)\nassert c._request(request) == first\n')
                            job = runner.run(dict(source=source, allow_direct_actions=True),
                                dict(thread_id='thread', turn_id='turn', tool_call_id=action_id))
                            self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
                            evidence = job['action_summary']['recent_actions'][0]
                            self.assertEqual((evidence['action_id'], evidence['status']), (action_id, 'completed'))
                            self.assertEqual(evidence['completed_native_steps'], 20)
                            reply = json.loads(self.store.resolve_artifact(evidence['reply_ref'])[1])
                    self.assertIn(fault, attempts)
                    self.assertFalse(reply['success'])
                    warning = json.loads(reply['contentItems'][0]['text'])['output_contract_error']
                    self.assertEqual(warning['reason'], 'action_feedback_unavailable')
                    self.assertTrue(warning['physical_action_executed'])
                    result = json.loads(reply['contentItems'][1]['text'])
                    self.assertEqual(result['receipt']['status'], 'completed')
                    self.assertEqual(result['receipt']['completed_native_steps'], 20)
                    saved = self.store.read('action_receipt', action_id)
                    self.assertEqual(saved['status'], 'completed')
                    self.assertEqual(len(saved['wait_observation_ids']), 1)
                    self.assertEqual(self.robot.executions, before + 1)

    def test_policy_image_failure_retains_receipt_without_repeating_prediction_or_execution(self):
        self.harness.start()
        predictions, executions = [], []
        def infer(*args, **kwargs):
            predictions.append(kwargs)
            return dict(actions=[[0.] * 14])
        def execute(action, **kwargs):
            executions.append(deepcopy(action))
            return dict(status='completed', executed_native_steps=1, completed_native_steps=1,
                terminal_observation=self.robot.observe())
        self.harness.action_policy = SimpleNamespace(infer=infer)
        self.robot.execute = execute
        params = dict(threadId='thread', turnId='turn', callId='policy-feedback', tool='robot_policy_step',
            arguments=dict(observation_id=self.harness.latest['observation_id'], prompt='Fixture adjustment', prefix=1))
        with patch.object(self.harness, 'observation_content', side_effect=OSError('fixture_image_read_failed')):
            reply = self.harness.handle_tool(params)
            self.assertEqual(self.harness.handle_tool(params), reply)
        self.assertFalse(reply['success'])
        self.assertTrue(json.loads(reply['contentItems'][0]['text'])['output_contract_error']['physical_action_executed'])
        result = json.loads(reply['contentItems'][1]['text'])
        self.assertEqual(result['receipt']['status'], 'completed')
        self.assertEqual(result['receipt']['completed_native_steps'], 1)
        self.assertEqual(result['request']['type'], 'JointPrefixAction')
        self.assertEqual((len(predictions), len(executions)), (1, 1))
        self.assertEqual(self.store.read('action_receipt', executions[0]['action_id'])['status'], 'completed')

    def test_availability_matrix_catalog_and_invoke_share_rejection(self):
        registry = create_registry()
        runtime = SimpleNamespace(config=Config(), disabled_tools=(),
            observation_profile='rgb_proprio_calibrated_v1', motion_review=False,
            capabilities={'unified_action': {'available': True}}, enable_wait=True,
            vision_compare=True, visual_reviewer=object(), action_policy=object(), programs=object())
        self.assertIn('robot_action', {x['name'] for x in registry.catalog(runtime)})
        self.assertEqual(registry.spec('robot_action', role='program').effect_for({'mode': 'inspect'}), 'read')
        self.assertEqual(registry.spec('robot_action', role='program').effect_for({'mode': 'execute'}), 'physical')
        cases = [
            ('robot_action', 'config', Config.from_dict({'features': {'motion_execution_mode': 'feedback', 'motion_review': False}}), {}),
            ('robot_action', 'capabilities', {'unified_action': {'available': False}}, {}),
            ('robot_action', 'observation_profile', 'rgb_proprio_v1', {}),
            ('robot_wait', 'enable_wait', False, {'observation_id': 'o', 'native_steps': 1}),
            ('vision_review', 'visual_reviewer', None, {'observation_id': 'o', 'target': 'cup', 'condition': 'retained_with_gripper'}),
            ('geometry_query', 'observation_profile', 'rgb_proprio_v1', {'request': {'kind': 'help'}}),
            ('depth_query', 'observation_profile', 'rgb_proprio_calibrated_v1', {'request': {'kind': 'help'}}),
        ]
        for tool, field, value, args in cases:
            variant = deepcopy(runtime)
            setattr(variant, field, value)
            with self.subTest(tool=tool, field=field):
                self.assertNotIn(tool, {x['name'] for x in registry.catalog(variant)})
                spec = registry.spec(tool, role='actor')
                self.assertIsNotNone(registry.unavailable(spec, variant, args))
                with self.assertRaises(ValueError):
                    registry.invoke(tool, args, {}, runtime=variant, role='actor')
        for name in ('robot_move_ee', 'scene_query', 'program_run', 'robot_policy_step'):
            with self.assertRaises(ValueError):
                registry.spec(name, role='program')
        with self.assertRaises(ToolInputError):
            registry.validate('robot_wait', {'observation_id': 'o', 'native_steps': True}, role='actor')
        with self.assertRaises(ToolInputError):
            registry.validate('robot_action', {'timeout_s': float('nan')}, role='actor')

    def test_plugin_metadata_is_filtered_for_actor_and_program_but_preserved_in_raw_data(self):
        from ni_rua.core.contracts import digest
        payload = dict(version="a"*64, revision=3, solver_version="s1", hash="payload-hash",
            source_url="https://example.test/business", nested=dict(hash="b"*64, version=4,
                source_url="https://example.test/nested"))
        schema = dict(type="object", properties=dict(version={"type": "string"}, revision={"type": "integer"},
            solver_version={"type": "string"}, hash={"type": "string"}, source_url={"type": "string"},
            nested={"type": "object"}), required=list(payload), additionalProperties=False)
        definition = dict(type="function", name="fixture_named_business_fields", description="Public business fixture.",
            inputSchema=dict(type="object", properties={}, additionalProperties=False))
        image = dict(type="inputImage", imageUrl="data:image/png;base64,AAAA",
            hash="raw-image-hash", source_url="https://example.test/raw-image")
        self.harness.tool_registry.register(ToolSpec(definition,
            lambda *args: ToolOutput(deepcopy(payload), extra_content=(image,)).response(), output_schemas={"data": schema}))
        shown = self.harness.handle_tool(dict(threadId="thread", turnId="turn", callId="named-business",
            tool=definition["name"], arguments={}))
        self.assertEqual(json.loads(shown["contentItems"][0]["text"]),
            dict(revision=3, solver_version="s1", nested=dict(version=4)))
        self.assertEqual(shown['contentItems'][1], dict(type='inputImage', imageUrl=image['imageUrl']))
        with closing(Store(self.store.root, self.store.identity, read_only=True)) as reopened:
            record, saved = next((record, saved) for record, saved in self.archived_results(reopened)
                if record['correlation'].get('tool_call_id') == 'named-business')
            self.assertEqual(record['archival_status'], 'complete')
            self.assertEqual(saved['data'], payload)
            self.assertEqual(json.loads(saved['wire_response']['contentItems'][0]['text']), payload)
            self.assertEqual(saved['wire_response']['contentItems'][1], image)
            self.assertEqual(reopened.artifact(record['raw_ref'])[0]['access_scope'], 'observer')
            with self.assertRaisesRegex(ValueError, 'artifact_access_scope_denied'):
                reopened.resolve_artifact(record['raw_ref'], allowed_scopes=('actor', 'internal'))
        raw = self.harness.invoke(definition["name"], {}, {}, role="actor")
        self.assertEqual(raw.data, payload)
        self.assertEqual(json.loads(wire_response(raw)["contentItems"][0]["text"]), payload)
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+60, call_lock=threading.RLock(),
            job_id="program-business-fields", calls=0, correlation={},
            condition=threading.Condition(threading.RLock()),
            job=dict(job_id="program-business-fields", status="running"), workspace=self.root,
            model_wait_s=0, review_wait_s=0, action_evidence=[])
        call = dict(operation_id="read", tool=definition["name"], arguments={})
        first = runner.dispatch(entry, call)
        repeated = runner.dispatch(entry, call)
        self.assertEqual(first, repeated)
        self.assertEqual(json.loads(first["contentItems"][0]["text"]),
            dict(revision=3, solver_version="s1", nested=dict(version=4)))
        self.assertEqual(first['contentItems'][1], dict(type='inputImage', imageUrl=image['imageUrl']))
        identifier = entry["job_id"] + "-" + digest("read")[:16]
        self.assertEqual(self.store.read_state("program_call", identifier)["result"], first)
        self.assertIs(type(first), dict)
        self.assertEqual(json.loads(json.dumps(first)), first)
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['role'] == 'program' and record['correlation'].get('tool_call_id') == identifier)
        self.assertEqual(saved['data'], payload)
        self.assertEqual(json.loads(saved['wire_response']['contentItems'][0]['text']), payload)
        self.assertEqual(saved['wire_response']['contentItems'][1], image)

    def test_program_replay_projects_historical_wire_without_mutation_or_redispatch(self):
        from ni_rua.core.contracts import digest
        runner = ProgramRunner(self.harness.program_services())
        self.addCleanup(runner.close)
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+60,
            call_lock=threading.RLock(), job_id='program-' + 'a'*24, calls=0)
        request = dict(operation_id='historic', tool='fixture_read', arguments={'value': 7})
        action_id = 'action-plugin-' + 'b'*24
        payload = dict(action_id=action_id, status='completed', executed_native_steps=7,
            request_sha256='c'*64, nested=dict(hash='d'*64, position_m=[.123456789, 0, 1]))
        wire = dict(success=True, contentItems=[dict(type='inputText', text=json.dumps(payload)),
            dict(type='inputText', text='Archived proof: ' + 'e'*64),
            dict(type='inputImage', imageUrl='data:image/png;base64,AAAA', sha256='f'*64)])
        identifier = entry['job_id'] + '-' + digest(request['operation_id'])[:16]
        saved = dict(program_id=entry['job_id'], operation_id=request['operation_id'],
            input_digest=digest(request), status='completed', result=deepcopy(wire))
        self.store.set_state('program_call', identifier, saved)
        shown = runner.dispatch(entry, deepcopy(request))
        self.assertEqual(runner.dispatch(entry, deepcopy(request)), shown)
        value = json.loads(shown['contentItems'][0]['text'])
        self.assertEqual(value['status'], 'completed')
        self.assertEqual(value['executed_native_steps'], 7)
        self.assertEqual(value['nested'], dict(position_m=[.123456789, 0, 1]))
        self.assertNotIn('request_sha256', value)
        self.assertNotEqual(value['action_id'], action_id)
        self.assertEqual(self.harness.actor_output.restore(value['action_id']), action_id)
        for original in (action_id, 'c'*64, 'd'*64, 'e'*64, 'f'*64):
            self.assertNotIn(original, json.dumps(shown))
        self.assertEqual(shown['contentItems'][2], dict(type='inputImage', imageUrl='data:image/png;base64,AAAA'))
        self.assertEqual(self.store.read_state('program_call', identifier), saved)
        self.assertEqual(entry['calls'], 0)
        self.assertEqual(self.calls, [])
        self.assertEqual(self.robot.executions, 0)
        with self.assertRaisesRegex(ValueError, 'program_operation_content_conflict'):
            runner.dispatch(entry, dict(request, arguments={'value': 8}))

    def test_real_program_short_references_round_trip_through_action_status_and_artifact(self):
        from ni_rua.robot.motion.arx_actions import compile_action
        self.harness.start()
        seen_observations = []
        def perform(context, args, correlation):
            seen_observations.append(args['observation_id'])
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1], timeout_s=2),
                context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_alias_motion', description='Alias round trip fixture.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        source = '''import json, re
from ni_rua_client import NiRuaClient
c = NiRuaClient()
def data(reply):
    assert reply['success'], reply
    return json.loads(next(item['text'] for item in reply['contentItems'] if item['type'] == 'inputText'))
observed = data(c.call('robot_observe', {'detail': 'full'}))
observation_id = observed.get('observation_id') or observed['bundle']['observation_id']
assert not re.search(r'[0-9a-f]{16,}', observation_id), observation_id
request = {'operation_id': 'move-once', 'tool': 'fixture_alias_motion',
           'arguments': {'observation_id': observation_id}}
first = c._request(request)
assert c._request(request) == first
result = data(first)
action_id = result['request']['action_id']
assert not re.search(r'[0-9a-f]{16,}', action_id), action_id
status = data(c.call('robot_action_status', {'action_id': action_id, 'detail': 'full'}))
assert status['request']['action_id'] == action_id
assert status['receipt']['status'] == 'completed', status
reference = result['receipt']['actual_trace_ref']
assert reference.startswith('artifact:ref_'), reference
trace = json.loads(c.artifact(reference))
assert trace[0]['measured_fixture_state']['arms']['left']['ee_pose']['position_m'] == [0.009, 0, 1]
print(json.dumps({'observation_id': observation_id, 'action_id': action_id, 'trace_ref': reference}))
'''
        job = runner.run(dict(source=source, allow_direct_actions=True, timeout_s=10),
            dict(thread_id='thread', turn_id='turn', tool_call_id='alias-program'))
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        shown = json.loads(job['stdout_text'])
        raw_action = job['action_summary']['recent_actions'][0]['action_id']
        self.assertTrue(raw_action.startswith('action-plugin-'))
        self.assertEqual(self.harness.actor_output.restore(shown['action_id']), raw_action)
        self.assertEqual(self.harness.actor_output.restore(shown['observation_id']), seen_observations[0])
        self.assertEqual(self.store.read('action_request', raw_action)['based_on_observation_id'], seen_observations[0])
        self.assertEqual(self.store.read('action_receipt', raw_action)['status'], 'completed')
        self.assertEqual(job['action_summary']['status_counts'], {'completed': 1})
        self.assertEqual(job['call_summary']['calls'], 3)
        self.assertEqual(job['call_summary']['failed_calls'], 0)
        self.assertEqual(self.robot.executions, 1)
        self.assertEqual(len(seen_observations), 1)

    def test_actor_program_source_reuses_reply_aliases_without_rewriting_checksums_or_numbers(self):
        from ni_rua.core.contracts import digest
        self.harness.start()
        runner = ProgramRunner(self.harness.program_services())
        self.harness.programs = runner
        self.addCleanup(runner.close)
        observation_id = self.harness.actor_output.present(self.harness.latest['observation_id'])
        reference = self.store.put_artifact(b'numeric fixture\n', 'fixture_text', 'text/plain')
        alias = self.harness.actor_output.present(reference)
        source = ("import json\nfrom ni_rua_client import NiRuaClient\nc = NiRuaClient()\n"
            "expected = " + repr(observation_id) + "\n"
            "checksum = " + repr('a'*64) + "\n"
            "unrelated = " + repr('checksum-' + 'b'*64) + "\n"
            "precision = '0.12345678901234567890123456789'\n"
            "reply = c.call('history_view_observation', {'observation_id': expected, 'camera_id': 'fixture_camera'})\n"
            "assert reply['success'], reply\n"
            "value = json.loads(reply['contentItems'][0]['text'])\n"
            "assert value['bundle']['observation_id'] == expected, (expected, value)\n"
            "assert c.artifact(" + repr(alias) + ") == b'numeric fixture\\n'\n"
            "assert len(checksum) == 64 and checksum.count('a') == 64\n"
            "assert unrelated.startswith('checksum-') and unrelated.count('b') == 64\n"
            "assert precision == '0.12345678901234567890123456789'\n"
            "print(json.dumps({'observation_id': expected, 'precision': precision}))\n")
        arguments = dict(source=source, timeout_s=10, detail='full')
        params = dict(threadId='thread', turnId='turn', callId='source-alias-roundtrip',
            tool='program_run', arguments=arguments)
        reply = self.harness.handle_tool(params)
        self.assertTrue(reply['success'])
        shown = json.loads(reply['contentItems'][0]['text'])
        self.assertEqual(shown['status'], 'succeeded', shown.get('stderr_text'))
        job = self.store.states('program')[0]
        actual_source = self.store.resolve_artifact(job['source_ref'])[1]
        self.assertEqual(actual_source.decode(), source)
        self.assertEqual(job['source_sha256'], hashlib.sha256(actual_source).hexdigest())
        original_arguments = self.harness.actor_output.restore(dict(source=source, timeout_s=10))
        self.assertEqual(job['input_digest'], digest(original_arguments))
        self.assertEqual(json.loads(self.store.resolve_artifact(job['stdout_ref'])[1]),
            dict(observation_id=observation_id, precision='0.12345678901234567890123456789'))
        repeated = self.harness.handle_tool(params)
        self.assertEqual(json.loads(repeated['contentItems'][0]['text'])['status'], 'succeeded')
        self.assertEqual(len(self.store.states('program')), 1)
        self.assertEqual(job['calls'], 1)
        self.assertEqual(self.robot.executions, 0)

    def test_raw_reply_is_persisted_before_failed_actor_presentation(self):
        payload = dict(value=7, hash='raw-hash', source_url='https://example.test/raw')
        definition = dict(self.plugin.definition, name='fixture_before_display')
        self.harness.tool_registry.register(ToolSpec(definition,
            lambda *args: ToolOutput(payload).response(), output_schemas={'data': {'type': 'object'}}))
        with patch.object(self.harness.actor_output, 'present', side_effect=ValueError('fixture_display_failed')):
            with self.assertRaisesRegex(ValueError, 'fixture_display_failed'):
                self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='before-display',
                    tool=definition['name'], arguments={'value': 7}))
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'before-display')
        self.assertEqual(record['archival_status'], 'complete')
        self.assertEqual(saved['data'], payload)
        self.assertEqual(json.loads(saved['wire_response']['contentItems'][0]['text']), payload)

    def test_non_json_raw_data_records_archival_gap_and_keeps_serializable_receipt(self):
        data = dict(receipt=dict(status='completed', executed_native_steps=4),
            hash='raw-proof', invalid=object())
        definition = dict(self.plugin.definition, name='fixture_non_json_raw_data')
        self.harness.tool_registry.register(ToolSpec(definition,
            lambda *args: ToolResponse([StructuredText(dict(value=7))], data=data),
            output_schemas={'data': {'type': 'object'}}))
        shown = self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='non-json-archive',
            tool=definition['name'], arguments={'value': 7}))
        self.assertFalse(shown['success'])
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'non-json-archive')
        self.assertEqual(record['archival_status'], 'partial')
        self.assertEqual(saved['data'], dict(receipt=data['receipt'], hash='raw-proof'))
        self.assertIn('/data/invalid', {error['field_path'] for error in saved['archival_errors']})
        self.assertEqual(saved['contract_error']['reason'], 'result_data_not_json')
        self.assertEqual(saved['wire_response'],
            dict(success=True, contentItems=[dict(type='inputText', text='{"value":7}')]))
        self.assertNotIn('object at', json.dumps(saved))

    def test_archive_write_failure_preserves_executed_receipt_and_action_is_not_replayed(self):
        from ni_rua.robot.motion.arx_actions import compile_action
        self.harness.start()
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_archive_failure_motion', description='Archive failure fixture.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        params = dict(threadId='thread', turnId='turn', callId='archive-motion', tool=definition['name'],
            arguments=dict(observation_id=self.harness.latest['observation_id']))
        original = self.store.put_artifact
        def fail_archive(data, kind, mime_type, access_scope='actor'):
            if kind == 'tool_result':
                raise OSError('fixture_archive_write_failed')
            return original(data, kind, mime_type, access_scope)
        with patch.object(self.store, 'put_artifact', side_effect=fail_archive):
            shown = self.harness.handle_tool(params)
        self.assertFalse(shown['success'])
        warning = json.loads(shown['contentItems'][0]['text'])['output_contract_error']
        self.assertEqual(warning['reason'], 'raw_result_archival_failed')
        self.assertEqual(warning['phase'], 'after_handler')
        self.assertTrue(warning['physical_action_executed'])
        self.assertIn('do not automatically replay', warning['suggestion'])
        self.assertEqual(json.loads(shown['contentItems'][1]['text'])['receipt']['status'], 'completed')
        failed = next(record for record in self.store.records('tool_result')
            if record['correlation'].get('tool_call_id') == 'archive-motion')
        self.assertEqual(failed['archival_status'], 'failed')
        repeated = self.harness.handle_tool(params)
        self.assertTrue(repeated['success'])
        self.assertEqual(json.loads(repeated['contentItems'][0]['text'])['receipt']['status'], 'completed')
        self.assertEqual(self.robot.executions, 1)
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'archive-motion')
        self.assertEqual(saved['data']['receipt']['status'], 'completed')
        self.assertEqual(saved['data']['receipt'],
            self.store.read('action_receipt', saved['data']['request']['action_id']))

    def test_robot_information_source_is_archived_before_its_internal_projection(self):
        raw = dict(robot_id='fixture', available=True, section='all', hash='source-proof',
            source_url='https://example.test/robot', measured_q_rad=[1.23456789012345])
        with patch('ni_rua.robot.model.robot_information.robot_information', return_value=deepcopy(raw)):
            shown = self.harness.robot_information('all', 'fixture')
        self.assertEqual(shown, dict(robot_id='fixture', available=True, section='all',
            measured_q_rad=[1.23456789012345]))
        with closing(Store(self.store.root, self.store.identity, read_only=True)) as reopened:
            record, saved = next((record, saved) for record, saved in self.archived_results(reopened)
                if record['tool'] == 'robot_information' and record['role'] == 'internal')
            self.assertEqual(record['archival_status'], 'complete')
            self.assertEqual(saved['data'], raw)
            self.assertEqual(json.loads(saved['wire_response']['contentItems'][0]['text']), raw)

    def test_archive_deepcopy_recursion_failure_does_not_escape_or_replay_executed_action(self):
        from ni_rua.robot.motion.arx_actions import compile_action
        self.harness.start()
        nested = {}
        for _ in range(500):
            nested = dict(child=nested)
        part = dict(type='inputText', text='deep fixture log', extension=nested)
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolResponse([part], kind='action', data=context.execute(action))
        definition = dict(type='function', name='fixture_archive_recursion_motion', description='Deep archive fixture.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical',
            output_schemas=output_schemas('robot_wait')))
        args = dict(observation_id=self.harness.latest['observation_id'])
        correlation = dict(tool_call_id='archive-recursion')
        first = self.harness.invoke(definition['name'], args, correlation, role='actor')
        self.assertEqual(first.data['receipt']['status'], 'completed')
        record, saved = next((record, saved) for record, saved in self.archived_results()
            if record['correlation'].get('tool_call_id') == 'archive-recursion')
        self.assertEqual(record['archival_status'], 'partial')
        self.assertEqual(saved['data'], first.data)
        self.assertIn(dict(field_path='/wire_response', reason='not_json_serializable',
            error_type='RecursionError'), saved['archival_errors'])
        self.assertEqual(first.data['receipt'],
            self.store.read('action_receipt', first.data['request']['action_id']))
        evidence = action_evidence(self.store, definition['name'], args, first,
            'archive-recursion', physical=True)
        self.assertEqual(evidence['status'], 'completed')
        self.assertTrue(evidence['dispatch_success'])
        reply = json.loads(self.store.resolve_artifact(evidence['reply_ref'])[1])
        self.assertNotIn('reply_archive_error', evidence)
        self.assertEqual(reply['contentItems'][0]['text'], 'deep fixture log')
        extension = reply['contentItems'][0]['extension']
        for _ in range(500):
            self.assertEqual(list(extension), ['child'])
            extension = extension['child']
        self.assertEqual(extension, {})
        # The projected writer can handle this tree; fail its current archive boundary explicitly.
        with patch.object(self.store, 'put_artifact', side_effect=RecursionError('fixture_reply_archive')) as archive:
            failed_evidence = action_evidence(self.store, definition['name'], args, first,
                'archive-recursion', physical=True)
        archive.assert_called_once()
        self.assertEqual(archive.call_args.args[1], 'program_action_reply')
        self.assertEqual(failed_evidence['status'], 'completed')
        self.assertTrue(failed_evidence['dispatch_success'])
        self.assertEqual(failed_evidence['reply_archive_error']['error_type'], 'RecursionError')
        self.assertNotIn('reply_ref', failed_evidence)
        self.assertEqual(self.robot.executions, 1)
        repeated = self.harness.invoke(definition['name'], args, correlation, role='actor')
        self.assertEqual(repeated.data, first.data)
        self.assertEqual(self.robot.executions, 1)

    def test_non_json_display_does_not_lose_trusted_physical_receipt(self):
        from dataclasses import replace
        business = dict(receipt=dict(status="completed", executed_native_steps=4))
        bad = ToolResponse([StructuredText(dict(value=object()))], data=business, kind="action")
        spec = replace(self.plugin, handler=lambda *args: bad,
            output_schemas=output_schemas("robot_wait"))
        result = ToolRegistry([spec]).invoke("fixture_read", {"value": 7}, {}, runtime=self.harness, role="actor")
        self.assertFalse(result.success)
        self.assertIs(result.data, business)
        self.assertEqual(result.contract_error["reason"], "display_not_json")
        evidence = action_evidence(self.store, "fixture_read", {}, result, "physical", physical=True)
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence["executed_native_steps"], 4)
        shown = self.harness.actor_output.present(result)
        self.assertFalse(shown["success"])
        self.assertIn("do not automatically replay", json.loads(shown["contentItems"][0]["text"])
            ["output_contract_error"]["suggestion"])
        self.assertEqual(json.loads(shown["contentItems"][1]["text"]), business)

    def test_api_version_is_explicit_and_independent_of_input_schema_hash(self):
        from dataclasses import replace
        first = ToolRegistry([self.plugin])
        next_version = ToolRegistry([replace(self.plugin, api_version='2.1')])
        a = first.manifest(self.harness)['tools'][0]
        b = next_version.manifest(self.harness)['tools'][0]
        self.assertEqual(a['api_version'], '1.0')
        self.assertEqual(b['api_version'], '2.1')
        self.assertEqual(a['schema_hash'], b['schema_hash'])
        for invalid in ('', 'latest', '0.1', '1', None):
            with self.subTest(version=invalid), self.assertRaisesRegex(ValueError, 'invalid_tool_api_version'):
                ToolRegistry([replace(self.plugin, api_version=invalid)])

    def test_same_tool_name_with_changed_schema_never_reuses_old_validation(self):
        old = ToolRegistry([self.plugin])
        changed = deepcopy(self.plugin.definition)
        changed['inputSchema']['properties']['value']['minimum'] = 10
        new = ToolRegistry([ToolSpec(changed, self.read, output_schemas=self.plugin.output_schemas)])
        old.validate('fixture_read', {'value': 7}, role='actor')
        with self.assertRaises(ToolInputError) as error:
            new.validate('fixture_read', {'value': 7}, role='actor')
        self.assertEqual(error.exception.field_path, '/value')
        new.validate('fixture_read', {'value': 10}, role='actor')
        with self.assertRaises(ToolInputError):
            self.harness.tool_registry.validate('robot_action', {'timeout_s': 10**400}, role='actor')

    def test_broker_uses_typed_values_even_if_rendered_text_is_forged(self):
        plan = ToolResponse([{'type': 'inputText', 'text': 'not JSON'}],
            kind='plan', data={'status': 'planned', 'plan_id': 'actual-plan'})
        self.assertEqual(reviewed_plan(plan)['plan_id'], 'actual-plan')
        fake_wire = ToolOutput({'status': 'planned', 'plan_id': 'forged-plan'}).response()
        with self.assertRaisesRegex(ValueError, 'missing_plan'):
            reviewed_plan(fake_wire)
        action = ToolResponse([{'type': 'inputText', 'text': '{"receipt":{"status":"completed"}}'}],
            kind='action', data={'receipt': {'status': 'stopped', 'executed_native_steps': 3}})
        evidence = action_evidence(self.store, 'robot_action', {'mode': 'execute'}, action, 'call')
        self.assertEqual(evidence['status'], 'stopped')
        self.assertEqual(evidence['executed_native_steps'], 3)
        unknown = action_evidence(self.store, 'robot_action', {'mode': 'execute'}, wire_response(action), 'wire-only')
        self.assertEqual(unknown['status'], 'unknown')
        observed = ToolResponse([], data={'observation_id': 'actual-observation'}, kind='observation')
        self.assertEqual(live_observation_id('robot_observe', observed), 'actual-observation')
        self.assertIsNone(live_observation_id('robot_observe', wire_response(observed)))
        self.assertEqual(set(json.loads(json.dumps(wire_response(action)))), {'success', 'contentItems'})

    def test_output_declarations_examples_and_manifest_share_one_contract(self):
        from dataclasses import replace
        from jsonschema.exceptions import SchemaError
        with self.assertRaisesRegex(ValueError, 'tool_output_contract_required'):
            ToolRegistry([replace(self.plugin, output_schemas=None)])
        with self.assertRaises(SchemaError):
            ToolRegistry([replace(self.plugin, output_schemas={'data': {'type': 'not-a-type'}})])
        wrong_example = deepcopy(self.plugin.output_schemas)
        wrong_example['data']['examples'] = [{'value': 'seven'}]
        with self.assertRaisesRegex(ValueError, 'invalid_tool_output_example'):
            ToolRegistry([replace(self.plugin, output_schemas=wrong_example)])
        original = ToolRegistry([self.plugin]).manifest(self.harness)['tools'][0]
        changed = deepcopy(self.plugin.output_schemas)
        changed['data']['properties']['value']['minimum'] = 2
        updated = ToolRegistry([replace(self.plugin, output_schemas=changed)]).manifest(self.harness)['tools'][0]
        self.assertEqual(original['result_kinds'], ['data'])
        self.assertEqual(original['output_schemas'], self.plugin.output_schemas)
        self.assertEqual(original['schema_hash'], updated['schema_hash'])
        self.assertNotEqual(original['output_schema_hash'], updated['output_schema_hash'])
        # A malformed new plugin cannot use a wire-shaped reply to opt out.
        legacy = replace(self.plugin, roles=frozenset({'trusted_legacy'}), output_schemas=None)
        legacy_registry = ToolRegistry([legacy])
        self.assertEqual(legacy_registry.manifest(self.harness, role='trusted_legacy')['tools'][0]
            ['output_contract'], 'trusted_legacy_wire_unvalidated')
        with self.assertRaises(ToolInputError):
            legacy_registry.spec('fixture_read', role='actor')

    def test_invalid_read_output_is_reported_after_handler_at_actor_and_program_ports(self):
        from dataclasses import replace
        calls = []
        def malformed(context, arguments, correlation):
            calls.append(arguments)
            return ToolOutput({'value': 'not an integer'}).response()
        spec = replace(self.plugin, definition=dict(self.plugin.definition, name='fixture_bad_output'),
            handler=malformed)
        self.harness.tool_registry.register(spec)
        result = self.harness.handle_tool(dict(threadId='thread', turnId='turn', callId='bad-output',
            tool=spec.name, arguments={'value': 7}))
        self.assertFalse(result['success'])
        diagnostic = json.loads(result['contentItems'][0]['text'])['output_contract_error']
        self.assertEqual(diagnostic['field_path'], '/data/value')
        self.assertEqual(diagnostic['phase'], 'after_handler')
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        job = runner.run(dict(source="from ni_rua_client import NiRuaClient\n"
            "print(NiRuaClient().call('fixture_bad_output', {'value': 11})['success'])"),
            dict(thread_id='thread', turn_id='turn', tool_call_id='bad-output-program'))
        self.assertEqual(job['status'], 'succeeded')
        self.assertEqual(job['call_summary']['failed_calls'], 1)
        self.assertEqual(calls, [{'value': 7}, {'value': 11}])
        self.assertEqual(self.robot.executions, 0)

    def test_output_failure_preserves_executed_receipt_and_does_not_replay(self):
        self.harness.start()
        from ni_rua.robot.motion.arx_actions import compile_action
        def perform(context, args, correlation):
            action = compile_action('move', dict(observation_id=args['observation_id'], arm='left',
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolOutput(context.execute(action), kind='action').response()
        definition = dict(type='function', name='fixture_bad_motion_output', description='Post-action regression.',
            inputSchema=dict(type='object', properties={'observation_id': {'type': 'string'}},
                required=['observation_id'], additionalProperties=False))
        schema = output_schemas('robot_wait')
        schema['action']['required'].append('missing_output_field')
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect='physical', output_schemas=schema))
        args = {'observation_id': self.harness.latest['observation_id']}
        first = self.harness.invoke(definition['name'], args, {'tool_call_id': 'execute-once'}, role='actor')
        repeated = self.harness.invoke(definition['name'], args, {'tool_call_id': 'execute-once'}, role='actor')
        self.assertFalse(first.success)
        self.assertEqual(first.contract_error['phase'], 'after_handler')
        self.assertEqual(first.kind, 'action')
        self.assertEqual(first.data['receipt']['status'], 'completed')
        self.assertEqual(first.data, repeated.data)
        self.assertEqual(self.robot.executions, 1)
        saved = self.store.read('action_receipt', first.data['request']['action_id'])
        self.assertEqual(first.data['receipt'], saved)
        evidence = action_evidence(self.store, definition['name'], args, first, 'execute-once', physical=True)
        self.assertEqual(evidence['status'], 'completed')
        self.assertFalse(evidence['dispatch_success'])
        self.assertEqual(evidence.get('executed_native_steps'), saved.get('executed_native_steps'))

    def test_executed_action_with_non_json_display_remains_completed_and_replays_once(self):
        from ni_rua.robot.motion.arx_actions import compile_action
        self.harness.start()
        def perform(context, args, correlation):
            action = compile_action("move", dict(observation_id=args["observation_id"], arm="left",
                position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
            return ToolResponse([StructuredText(dict(invalid_display=object()))],
                kind="action", data=context.execute(action))
        definition = dict(type="function", name="fixture_non_json_motion_display", description="Display failure after execution.",
            inputSchema=dict(type="object", properties={"observation_id": {"type": "string"}},
                required=["observation_id"], additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition, perform, effect="physical",
            output_schemas=output_schemas("robot_wait")))
        args = {"observation_id": self.harness.latest["observation_id"]}
        first = self.harness.invoke(definition["name"], args, {"tool_call_id": "non-json-once"}, role="actor")
        repeated = self.harness.invoke(definition["name"], args, {"tool_call_id": "non-json-once"}, role="actor")
        self.assertFalse(first.success)
        self.assertEqual(first.contract_error["reason"], "display_not_json")
        self.assertEqual(first.data, repeated.data)
        self.assertEqual(first.data["receipt"]["status"], "completed")
        self.assertEqual(self.robot.executions, 1)
        self.assertEqual(first.data["receipt"], self.store.read("action_receipt", first.data["request"]["action_id"]))
        shown = self.harness.actor_output.present(first)
        self.assertTrue(json.loads(shown["contentItems"][0]["text"])["output_contract_error"]["physical_action_executed"])
        self.assertEqual(json.loads(shown["contentItems"][1]["text"])["receipt"]["status"], "completed")
        evidence = action_evidence(self.store, definition["name"], args, first, "non-json-once", physical=True)
        self.assertEqual(evidence["status"], "completed")
        self.assertEqual(evidence.get("executed_native_steps"), first.data["receipt"].get("executed_native_steps"))
        runner = ProgramRunner(program_services(self.harness))
        self.harness.programs = runner
        self.addCleanup(runner.close)
        source = ("from ni_rua_client import NiRuaClient\nprint(NiRuaClient().call(" +
                  repr(definition["name"]) + ", " + repr({"observation_id": self.harness.latest["observation_id"]}) + ")['success'])")
        job = runner.run(dict(source=source, allow_direct_actions=True),
            dict(thread_id="thread", turn_id="turn", tool_call_id="non-json-program"))
        self.assertEqual(job["status"], "succeeded")
        self.assertEqual(job["action_summary"]["status_counts"], {"completed": 1})
        # FixtureRobot does not report native step counts; retain that unknown.
        self.assertEqual(job["action_summary"]["actions_without_step_count"], 1)
        self.assertEqual(job["call_summary"]["failed_calls"], 1)
        self.assertEqual(self.robot.executions, 2)

    def test_typed_success_requires_boolean_without_coercing_truthy_values(self):
        from dataclasses import replace
        for success in (object(), 1, 0, "true", "false", None):
            with self.subTest(success_type=type(success).__name__, success_value=success if isinstance(success, (str, int)) else None):
                original = ToolResponse([StructuredText({"value": 7})], data={"value": 7}, success=success)
                spec = replace(self.plugin, handler=lambda *args: original)
                result = ToolRegistry([spec]).invoke("fixture_read", {"value": 7}, {}, runtime=self.harness, role="actor")
                self.assertIs(result.success, False)
                self.assertIs(result.data, original.data)
                self.assertEqual(result.kind, "data")
                self.assertEqual(result.contract_error["reason"], "result_success_not_boolean")
                self.assertEqual(result.contract_error["field_path"], "/success")
                self.assertEqual(result.contract_error["phase"], "after_handler")
                self.assertIs(json.loads(json.dumps(wire_response(result)))["success"], False)
        for success in (True, False):
            with self.subTest(valid_success=success):
                original = ToolResponse([StructuredText({"value": 7})], data={"value": 7}, success=success)
                spec = replace(self.plugin, handler=lambda *args: original)
                result = ToolRegistry([spec]).invoke("fixture_read", {"value": 7}, {}, runtime=self.harness, role="actor")
                self.assertIs(result, original)
                self.assertIsNone(result.contract_error)

    def test_executed_action_with_invalid_success_keeps_receipt_and_program_replay(self):
        from ni_rua.robot.motion.arx_actions import compile_action
        from ni_rua.core.contracts import digest
        self.harness.start()
        runner = ProgramRunner(program_services(self.harness))
        self.harness.programs = runner
        self.addCleanup(runner.close)
        for index, invalid_success in enumerate((object(), 1, "true")):
            with self.subTest(success_type=type(invalid_success).__name__):
                def perform(context, args, correlation):
                    action = compile_action("move", dict(observation_id=args["observation_id"], arm="left",
                        position_m=[.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]), context.identity, context.capabilities)
                    response = ToolOutput(context.execute(action), kind="action", projection="action_result").response()
                    response.success = invalid_success
                    return response
                definition = dict(type="function", name="fixture_invalid_success_" + str(index),
                    description="Invalid success after fixture execution.",
                    inputSchema=dict(type="object", properties={"observation_id": {"type": "string"}},
                        required=["observation_id"], additionalProperties=False))
                self.harness.tool_registry.register(ToolSpec(definition, perform, effect="physical",
                    output_schemas=output_schemas("robot_wait")))
                request = dict(operation_id="execute-once", tool=definition["name"],
                    arguments={"observation_id": self.harness.latest["observation_id"]})
                source = ("from ni_rua_client import NiRuaClient\nc = NiRuaClient()\nrequest = " +
                    repr(request) + "\nfirst = c._request(request)\nassert c._request(request) == first\nprint(first['success'])")
                job = runner.run(dict(source=source, allow_direct_actions=True),
                    dict(thread_id="thread", turn_id="turn", tool_call_id="invalid-success-program-" + str(index)))
                self.assertEqual(job["status"], "succeeded")
                self.assertEqual(job["action_summary"]["status_counts"], {"completed": 1})
                self.assertEqual(job["action_summary"]["actions_without_step_count"], 1)
                self.assertEqual(job["call_summary"]["calls"], 1)
                self.assertEqual(job["call_summary"]["failed_calls"], 1)
                self.assertEqual(self.robot.executions, 2 * index + 1)
                call_id = job["job_id"] + "-" + digest("execute-once")[:16]
                record = self.store.read_state("program_call", call_id)
                self.assertEqual(record["output_contract_error"]["reason"], "result_success_not_boolean")
                self.assertEqual(record["output_contract_error"]["field_path"], "/success")
                self.assertIs(record["result"]["success"], False)
                evidence = job["action_summary"]["recent_actions"][0]
                self.assertIs(evidence["dispatch_success"], False)
                reply = json.loads(self.store.resolve_artifact(evidence["reply_ref"])[1])
                self.assertIs(reply["success"], False)
                self.assertTrue(json.loads(reply["contentItems"][0]["text"])["output_contract_error"]["physical_action_executed"])
                recorded_action = json.loads(reply["contentItems"][1]["text"])
                original_action_id = self.harness.actor_output.restore(recorded_action['request']['action_id'])
                stored_receipt = self.store.read('action_receipt', original_action_id)
                self.assertEqual(recorded_action["receipt"],
                    self.harness.actor_output.present(dict(receipt=stored_receipt),
                        projection='action_result')['receipt'])
                archived = next(saved for item, saved in self.archived_results()
                    if item['correlation'].get('tool_call_id') == call_id)
                self.assertEqual(archived['data']['receipt'], stored_receipt)
                args = {"observation_id": self.harness.latest["observation_id"]}
                correlation = {"tool_call_id": "invalid-success-actor-" + str(index)}
                first = self.harness.invoke(definition["name"], args, correlation, role="actor")
                repeated = self.harness.invoke(definition["name"], args, correlation, role="actor")
                self.assertIs(first.success, False)
                self.assertEqual(first.kind, "action")
                self.assertEqual(first.data["receipt"]["status"], "completed")
                self.assertEqual(first.data, repeated.data)
                self.assertEqual(first.data["receipt"], self.store.read("action_receipt", first.data["request"]["action_id"]))
                self.assertEqual(self.robot.executions, 2 * index + 2)
                shown = self.harness.actor_output.present(first)
                self.assertIs(shown["success"], False)
                self.assertTrue(json.loads(shown["contentItems"][0]["text"])["output_contract_error"]["physical_action_executed"])
                self.assertEqual(json.loads(shown["contentItems"][1]["text"])["receipt"]["status"], "completed")

    def test_wrong_kind_and_wire_only_results_never_create_execution_evidence(self):
        from dataclasses import replace
        for bad in (ToolResponse([], data={'value': 7}, kind='action'),
                    ToolResponse([], data={'receipt': None}, kind='action'),
                    ToolResponse([], data={'receipt': {'status': []}}, kind='action'),
                    ToolResponse([], data={'value': 7}, kind=[]),
                    {'success': True, 'contentItems': None},
                    {'success': True, 'contentItems': [{'type': 'inputText',
                        'text': '{"receipt":{"status":"completed","executed_native_steps":9}}'}]}):
            registry = ToolRegistry([replace(self.plugin, handler=lambda *args: bad)])
            result = registry.invoke('fixture_read', {'value': 7}, {}, runtime=self.harness, role='actor')
            self.assertFalse(result.success)
            self.assertIsNotNone(result.contract_error)
            evidence = action_evidence(self.store, 'fixture_read', {}, result, 'call', physical=True)
            self.assertEqual(evidence['status'], 'unknown')
            self.assertNotIn('executed_native_steps', evidence)

    def test_malformed_physical_reply_is_unknown_without_losing_broker_accounting(self):
        self.harness.start()
        definition = dict(type='function', name='fixture_missing_receipt', description='Malformed receipt fixture.',
            inputSchema=dict(type='object', properties={}, additionalProperties=False))
        self.harness.tool_registry.register(ToolSpec(definition,
            lambda *args: ToolResponse([], kind='action', data={'receipt': None}),
            effect='physical', output_schemas=output_schemas('robot_wait')))
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        job = runner.run(dict(allow_direct_actions=True, source="from ni_rua_client import NiRuaClient\n"
            "print(NiRuaClient().call('fixture_missing_receipt', {})['success'])"),
            dict(thread_id='thread', turn_id='turn', tool_call_id='malformed-physical-program'))
        self.assertEqual(job['status'], 'succeeded')
        self.assertEqual(job['call_summary']['failed_calls'], 1)
        self.assertEqual(job['action_summary']['status_counts'], {'unknown': 1})
        self.assertEqual(job['action_summary']['actions_without_step_count'], 1)
        calls = self.store.states('program_call')
        self.assertEqual(calls[0]['output_contract_error']['phase'], 'after_handler')
        self.assertFalse(calls[0]['result']['success'])

    def test_source_archive_and_catalog_manifest_include_new_packages(self):
        self.robot.close = lambda: None
        config = Config.from_dict({'features': {'disabled_tools': ['fixture_read'],
            'grasp_memory_dir': str(self.root/'memory')}})
        workbench = Workbench(self.root/'workbench', robot=self.robot, config=config,
            tool_specs=[self.plugin], observation_profile='rgb_proprio_v1', motion_review=False)
        self.addCleanup(workbench.store.close)
        self.addCleanup(workbench.run_lock.close)
        if workbench.grasp_memory_store:
            self.addCleanup(workbench.grasp_memory_store.close)
        from support.paths import REPO_ROOT
        package = REPO_ROOT / 'ni_rua'
        for name in ('runtime/harness.py', 'runtime/tools/registry.py', 'apps/evaluation/report.py'):
            self.assertEqual(workbench.source_hashes[name], hashlib.sha256((package/name).read_bytes()).hexdigest())
        info, data = workbench.store.resolve_artifact(workbench.source_ref, allowed_scopes=("observer",))
        archived = json.loads(data)
        self.assertEqual(archived['runtime/tools/registry.py'], (package/'runtime/tools/registry.py').read_text())
        manifest = workbench.harness.tool_registry.manifest(workbench.harness)
        item = next(x for x in manifest['tools'] if x['name'] == 'fixture_read')
        self.assertFalse(item['available'])
        self.assertEqual(item['unavailable_reason'], 'tool_disabled_by_config:fixture_read')

    def test_native_app_server_tool_payload_stays_under_5000_bytes_without_collapsing(self):
        """验证全开 3D 开关下经真实 codex app-server 转换后的原生工具 Schema < 5000B、零折叠且字段描述完整送达。"""
        from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
        from codex_cli_bin import bundled_codex_path
        from openai_codex.client import CodexConfig
        from ni_rua.core.contracts import canonical
        from ni_rua.cortex.agents.codex.session import RecordedClient, write_direct_model_catalog
        from ni_rua.perception.external import reanchor_tracks

        all_on_tools = [
            dict(tool, deferLoading=False)
            for tool in public_definitions(
                vggt_height='default',
                object_table=True,
                scene3d=True,
                observe3d=(True, True, False),
            )
        ]
        captured_requests, notifications = [], []

        class LoopbackResponses(BaseHTTPRequestHandler):
            protocol_version = 'HTTP/1.1'
            def log_message(self, *_):
                pass
            def do_GET(self):
                body = b'{"data":[]}'
                self.send_response(200)
                self.send_header('Content-Type', 'application/json')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            def do_POST(self):
                captured_requests.append(json.loads(self.rfile.read(int(self.headers['Content-Length']))))
                item = {'type': 'message', 'id': 'msg-ok', 'role': 'assistant', 'status': 'completed',
                        'content': [{'type': 'output_text', 'text': 'ok', 'annotations': []}]}
                events = [
                    ('response.created', {'response': {'id': 'r1', 'status': 'in_progress', 'output': []}}),
                    ('response.output_item.added', {'output_index': 0, 'item': item}),
                    ('response.output_item.done', {'output_index': 0, 'item': item}),
                    ('response.completed', {'response': {'id': 'r1', 'status': 'completed', 'output': [item],
                        'usage': {'input_tokens': 5, 'output_tokens': 1, 'total_tokens': 6}}}),
                ]
                body = ''.join(f'event: {ev}\ndata: {json.dumps({"type": ev, **val})}\n\n' for ev, val in events).encode()
                self.send_response(200)
                self.send_header('Content-Type', 'text/event-stream')
                self.send_header('Content-Length', str(len(body)))
                self.end_headers()
                self.wfile.write(body)

        resp_server = ThreadingHTTPServer(('127.0.0.1', 0), LoopbackResponses)
        threading.Thread(target=lambda: resp_server.serve_forever(poll_interval=0.01), daemon=True).start()
        self.addCleanup(resp_server.server_close)
        self.addCleanup(resp_server.shutdown)

        croot = self.root / 'codex-native-check'
        codex_home = croot / 'codex-home'
        codex_home.mkdir(parents=True)
        catalog_path = write_direct_model_catalog(codex_home / 'models.json')
        base_url = f'http://127.0.0.1:{resp_server.server_address[1]}/v1'
        (codex_home / 'config.toml').write_text(
            'model = "gpt-6.1-sol"\nmodel_provider = "fixture"\n'
            f'model_catalog_json = {json.dumps(str(catalog_path))}\n'
            'approval_policy = "never"\nsandbox_mode = "danger-full-access"\n'
            'web_search = "disabled"\n'
            '[model_providers.fixture]\nname = "Loopback fixture"\n'
            f'base_url = {json.dumps(base_url)}\n'
            'wire_api = "responses"\nrequires_openai_auth = false\nsupports_websockets = false\n'
            '[features]\ncode_mode_host = false\nshell_tool = true\nunified_exec = true\n'
            'memories = false\napps = false\nplugins = false\n'
            '[features.code_mode]\nenabled = false\n',
            encoding='utf-8',
        )
        cstore = Store(croot / 'crun', dict(run_id='codex-schema', episode_id='ep', env_instance_id='env', env_generation=1))
        turn_completed = threading.Event()
        expected_turn_id = [None]

        def on_notification(message):
            notifications.append(message)
            if (message.get('method') == 'turn/completed'
                    and expected_turn_id[0] is not None
                    and message.get('params', {}).get('turn', {}).get('id') == expected_turn_id[0]):
                turn_completed.set()

        client = RecordedClient(
            CodexConfig(
                cwd=str(croot),
                launch_args_override=(
                    '/usr/bin/env', '-i', 'PATH=/usr/bin:/bin',
                    f'HOME={croot}', f'CODEX_HOME={codex_home}', f'TMPDIR={croot}',
                    str(bundled_codex_path()), '--no-daemon', 'app-server', '--listen', 'stdio://',
                ),
            ),
            cstore,
            lambda _p: {'success': True, 'contentItems': []},
            on_notification,
        )
        try:
            client.start()
            client.initialize()
            started = client.thread_start({
                'model': 'gpt-6.1-sol', 'modelProvider': 'fixture', 'cwd': str(croot),
                'approvalPolicy': 'never', 'sandbox': 'danger-full-access', 'dynamicTools': all_on_tools,
            })
            turn = client.turn_start(started.thread.id, 'Check tool definitions.')
            expected_turn_id[0] = turn.turn.id
            if any(m.get('method') == 'turn/completed' and m.get('params', {}).get('turn', {}).get('id') == turn.turn.id
                   for m in notifications):
                turn_completed.set()
            if not turn_completed.wait(20.0):
                self.fail('codex app-server turn did not complete')

            req0 = captured_requests[0]
            tool_specs = req0.get('tools', []) + [
                t for item in req0['input'] if item.get('type') == 'additional_tools' for t in item.get('tools', [])
            ]
            flat_specs = [s for s in tool_specs if s.get('type') == 'function'] + [
                t for s in tool_specs for t in s.get('tools', []) if t.get('type') == 'function'
            ]
            native_tools = {s['name']: s for s in flat_specs}

            def _schema_nodes(value, path=()):
                if isinstance(value, dict):
                    yield path, value
                    for key, child in value.items():
                        if key not in {'description', 'examples', 'default', 'enum', 'const', 'if', 'not'}:
                            yield from _schema_nodes(child, path + (key,))
                elif isinstance(value, list):
                    for index, child in enumerate(value):
                        yield from _schema_nodes(child, path + (index,))

            from ni_rua.apps.evaluation.codex_fork import measure_codex_0160_dynamic_tool_schema
            from ni_rua.cortex.agents.codex.session import _native_dynamic_tools
            from ni_rua.cortex.agents.mcp_server import portable_input_schema
            from ni_rua.robot.backends.so101 import deployment as so101_deployment
            from ni_rua.robot.model.embodiments import ARX_X5_PROFILE
            from ni_rua.runtime.tools.plugins import configured_tool_specs
            from support.paths import CONFIGS_ROOT

            for d in all_on_tools:
                tname = d['name']
                self.assertIn(tname, native_tools)
                self.assertEqual(native_tools[tname]['description'], d['description'], tname)
                schema_bytes = len(canonical(native_tools[tname]['parameters']).encode('utf-8'))
                measured = measure_codex_0160_dynamic_tool_schema(d['inputSchema'])
                self.assertEqual(measured['serialized_bytes'], schema_bytes, tname)
                self.assertEqual(measured['schema'], native_tools[tname]['parameters'], tname)
                self.assertLessEqual(schema_bytes, 5000, f'{tname} native schema {schema_bytes} > 5000')
                local_help = {p: n['description'] for p, n in _schema_nodes(d['inputSchema']) if 'description' in n}
                native_help = {p: n['description'] for p, n in _schema_nodes(native_tools[tname]['parameters']) if 'description' in n}
                for p, help_text in local_help.items():
                    self.assertTrue(native_help.get(p) == help_text or help_text in native_tools[tname]['description'], (tname, p))
            stage_props = native_tools['robot_action']['parameters']['properties']['stages']['items']['properties']
            for vec_field in ('grasp_position_m', 'angles_deg', 'position_m', 'quaternion_xyzw'):
                self.assertEqual(stage_props[vec_field]['items']['type'], 'number', vec_field)
            self.assertEqual(len(native_tools['vision_review']['parameters']['oneOf']), 2)
            portable_vr = portable_input_schema(
                next(d['inputSchema'] for d in all_on_tools if d['name'] == 'vision_review')
            )
            self.assertEqual(len(portable_vr['properties']), 11)
            self.assertEqual(len(portable_vr['required']), 5)
            geo_desc = native_tools['geometry_query']['description']
            for field, help_text in reanchor_tracks.FIELD_DESCRIPTIONS.items():
                self.assertIn(help_text, geo_desc, field)

            # G1 双本体门禁：验证 ARX-X5 与 SO-101 经 _native_dynamic_tools 发布后全部工具 <= 5000 B 且位姿/关节向量不坍缩
            so101_doc = json.loads((CONFIGS_ROOT / 'so101.example.json').read_text(encoding='utf-8'))
            so101_profile = so101_deployment.embodiment_profile(
                so101_deployment.validate_deployment(so101_doc, base=CONFIGS_ROOT)
            )
            g1_cfg = Config.from_dict({
                'features': {
                    'motion_review': False,
                    'external_perception': True,
                    'vggt_height': True,
                    'object_table': True,
                    'scene3d': True,
                }
            })
            for emb_name, profile, expected_vectors in (
                ('arx-x5', ARX_X5_PROFILE, ('grasp_position_m', 'angles_deg', 'position_m', 'quaternion_xyzw')),
                ('so101', so101_profile, ('grasp_position_m', 'joint_positions_deg')),
            ):
                rt = SimpleNamespace(
                    config=g1_cfg,
                    features=g1_cfg.features,
                    embodiment=profile,
                    disabled_tools=(),
                    observation_profile='rgb_proprio_calibrated_v1',
                    motion_review=False,
                    capabilities={'unified_action': {'available': True}},
                    enable_wait=True,
                    vision_compare=True,
                    visual_reviewer=object(),
                    action_policy=object(),
                    programs=object(),
                )
                reg = create_registry(configured_tool_specs(g1_cfg), runtime=rt)
                dyn_tools = _native_dynamic_tools(
                    reg.catalog(rt, role='actor'),
                    catalog_profile=g1_cfg.features.tool_catalog_profile,
                    supports_search_tool=False,
                )
                self.assertTrue(all(t.get('type') != 'namespace' and not t.get('deferLoading') for t in dyn_tools))
                by_tool = {t['name']: t for t in dyn_tools}
                for tname, tdef in by_tool.items():
                    m = measure_codex_0160_dynamic_tool_schema(tdef['inputSchema'])
                    self.assertLessEqual(
                        m['serialized_bytes'], 5000,
                        f"{emb_name}:{tname} native schema {m['serialized_bytes']} > 5000",
                    )
                ra_stage = by_tool['robot_action']['inputSchema']['properties']['stages']['items']['properties']
                ra_native = measure_codex_0160_dynamic_tool_schema(by_tool['robot_action']['inputSchema'])['schema']
                ra_native_stage = ra_native['properties']['stages']['items']['properties']
                for vf in expected_vectors:
                    if vf in ra_stage:
                        self.assertEqual(ra_native_stage[vf]['items']['type'], 'number', (emb_name, vf))
        finally:
            proc = client._proc
            client.close()
            if proc:
                for stream in (proc.stdout, proc.stderr):
                    if stream:
                        stream.close()
            client.server_pool.shutdown(wait=True)
            cstore.close()

    def test_optional_experimental_tools_default_off_and_respect_feature_gates(self):
        """验证默认关闭的可选工具 (robotic_view / stereo_guide / pose_plan) 默认不暴露且受真实配置与标定门禁约束。"""
        from ni_rua.apps.gateway.workbench import _configured_tool_specs

        def _build_registry(cfg):
            rt = SimpleNamespace(
                config=cfg,
                features=cfg.features,
                disabled_tools=cfg.features.disabled_tools,
                observation_profile=cfg.features.observation_profile,
                motion_review=cfg.features.motion_review,
                enable_wait=True,
                vision_compare=True,
                visual_reviewer=None,
                action_policy=None,
                capabilities=None,
                programs=True,
            )
            return rt, create_registry(_configured_tool_specs(cfg), runtime=rt)

        def _catalog_names(cfg, role='actor'):
            rt, registry = _build_registry(cfg)
            return {t['name'] for t in registry.catalog(rt, role=role)}

        default_rt, default_reg = _build_registry(Config())
        for role in ('actor', 'program'):
            default_names = {t['name'] for t in default_reg.catalog(default_rt, role=role)}
            for optional_tool in ('robotic_view', 'stereo_guide', 'pose_plan'):
                self.assertNotIn(optional_tool, default_names, (role, optional_tool))
                with self.assertRaises((KeyError, ValueError)):
                    default_reg.invoke(optional_tool, {}, {}, runtime=default_rt, role=role)

        enabled_cfg = Config.from_dict({
            'features': {
                'motion_review': False,
                'external_perception': True,
                'robotic_view': True,
                'stereo_guide': True,
                'pose_skill': True,
            }
        })
        for role in ('actor', 'program'):
            enabled_names = _catalog_names(enabled_cfg, role=role)
            for optional_tool in ('robotic_view', 'stereo_guide', 'pose_plan'):
                self.assertIn(optional_tool, enabled_names, (role, optional_tool))

        only_rv_cfg = Config.from_dict({'features': {'motion_review': False, 'robotic_view': True, 'stereo_guide': False}})
        only_sg_cfg = Config.from_dict({'features': {'motion_review': False, 'robotic_view': False, 'stereo_guide': True}})
        for role in ('actor', 'program'):
            rv_names = _catalog_names(only_rv_cfg, role=role)
            self.assertIn('robotic_view', rv_names)
            self.assertNotIn('stereo_guide', rv_names)
            sg_names = _catalog_names(only_sg_cfg, role=role)
            self.assertIn('stereo_guide', sg_names)
            self.assertNotIn('robotic_view', sg_names)

        # 不做任何测试侧配置篡改，直接以 robotic_view=True / stereo_guide=True + 未标定 profile 检验插件自身标定门禁及双角色调用拒绝
        uncalibrated_cfg = enabled_cfg.with_overrides({'features': {'observation_profile': 'rgb_proprio_v1'}})
        self.assertTrue(uncalibrated_cfg.features.robotic_view)
        self.assertTrue(uncalibrated_cfg.features.stereo_guide)
        uncal_rt, uncal_reg = _build_registry(uncalibrated_cfg)
        for role in ('actor', 'program'):
            uncalibrated_names = {t['name'] for t in uncal_reg.catalog(uncal_rt, role=role)}
            for gated_tool in ('robotic_view', 'stereo_guide'):
                self.assertNotIn(gated_tool, uncalibrated_names, (role, gated_tool))
                spec = uncal_reg.spec(gated_tool, role=role)
                self.assertIsNotNone(uncal_reg.unavailable(spec, uncal_rt, {}), (role, gated_tool))
                with self.assertRaises(ValueError):
                    uncal_reg.invoke(gated_tool, {}, {}, runtime=uncal_rt, role=role)

        disabled_cfg = enabled_cfg.with_overrides({
            'features': {'disabled_tools': ['robotic_view', 'stereo_guide', 'pose_plan']}
        })
        disabled_names = _catalog_names(disabled_cfg)
        for optional_tool in ('robotic_view', 'stereo_guide', 'pose_plan'):
            self.assertNotIn(optional_tool, disabled_names)

    def test_optional_view_tools_real_owner_security_and_freshness_boundaries(self):
        """验证启用可选视图工具 (robotic_view / stereo_guide) 时真实 reader 的安全门禁、新鲜度竞态拦截与零物理副作用。"""
        import cv2
        import numpy as np
        from scipy.spatial.transform import Rotation
        from unittest.mock import Mock
        from ni_rua.apps.gateway.workbench import _configured_tool_specs
        from ni_rua.core.contracts import canonical

        def _mat(xyz=(0.0, 0.0, 0.0), angles_deg=(0.0, 0.0, 0.0)):
            out = np.eye(4)
            out[:3, :3] = Rotation.from_euler('xyz', angles_deg, degrees=True).as_matrix()
            out[:3, 3] = xyz
            return out

        identity = dict(run_id='rv-sg-fixture', episode_id='episode', env_instance_id='env', env_generation=1)
        store = Store(self.root / 'run_rv_sg', identity)
        self.addCleanup(store.close)
        robot = SimpleNamespace(
            observe=Mock(side_effect=AssertionError('不得触发物理观测')),
            execute=Mock(side_effect=AssertionError('不得触发物理执行')),
            preview_batch=Mock(side_effect=AssertionError('不得触发原生预览')),
        )
        cfg = Config.from_dict({
            'features': {
                'observation_profile': 'rgb_proprio_calibrated_v1',
                'motion_review': False,
                'robotic_view': True,
                'stereo_guide': True,
            }
        })
        owner = SimpleNamespace(harness=None)
        specs = _configured_tool_specs(cfg, owner=owner)
        h = Harness(store, robot, config=cfg, tool_specs=specs)
        owner.harness = h
        h.thread_id = 'fixture-thread'
        h.capabilities = dict(frame_id='env_origin', tcp_config_id='arx-x5-link6-v1')

        link = _mat((0.2, -0.1, 0.3), (12.0, 25.0, 40.0))
        fixed_first = _mat((0.06, 0.01, 0.0), (0.0, 20.0, 0.0))
        fixed_second = _mat((0.038, -0.01, 0.0), (0.0, 0.0, 10.0))
        fixed = fixed_first @ fixed_second
        optical = _mat((0.01, 0.02, 0.03), (90.0, 0.0, 90.0))
        base_model = dict(
            base_link='base',
            joints=[
                dict(name='wrist', type='fixed', parent='base', child='link6', T_parent_joint=link.tolist()),
                dict(name='hand_to_camera_mount', type='fixed', parent='link6', child='camera_base', T_parent_joint=fixed_first.tolist()),
                dict(name='camera_joint', type='fixed', parent='camera_base', child='camera', T_parent_joint=fixed_second.tolist()),
            ],
        )
        src_pose = _mat((0.0, 0.0, 0.0), (0.0, 0.0, 0.0))
        tgt_pose = _mat((0.2, 0.0, 0.0), (0.0, 0.0, 0.0))
        sg_k = [[90.0, 0.0, 49.0], [0.0, 110.0, 39.0], [0.0, 0.0, 1.0]]
        seq = [0]

        def save_bundle(
            *,
            obs_identity=None,
            profile='rgb_proprio_calibrated_v1',
            rgb_scope='actor',
            cal_scope='actor',
            rgb_kind='rgb',
            state_step=5,
            wrist_arm='left',
            tcp_id='arx-x5-link6-v1',
            ee_dx=0.0,
            frame_reference='env_origin',
            distortion=None,
            distortion_model='pinhole_no_distortion',
            corrupt_rgb=False,
            import_scene=True,
        ):
            seq[0] += 1
            oid = f'obs-{seq[0]}'
            fid = f'frame-{seq[0]}'

            def put_json(val, kind, sc='actor'):
                return store.put_artifact(canonical(val).encode(), kind, 'application/json', sc)

            ee_pos = link[:3, 3].copy()
            ee_pos[0] += ee_dx
            state = dict(
                sim_step=state_step,
                step_domain='control_step',
                arms=dict(
                    left=dict(
                        source_kind='measurement',
                        q_rad=[],
                        joint_names=[],
                        ee_pose=dict(
                            source_kind='measurement',
                            frame_id='env_origin',
                            tcp_config_id=tcp_id,
                            position_m=ee_pos.tolist(),
                            quaternion_xyzw=Rotation.from_matrix(link[:3, :3]).as_quat().tolist(),
                        ),
                        gripper=dict(
                            joint_names=[],
                            position_m=[],
                            position_source_kind='measurement',
                            command_normalized=0.0,
                            command_source_kind='command',
                        ),
                    )
                ),
            )
            wrist_cal = dict(
                reference_frame_id='env_origin',
                cameras=dict(
                    wrist=dict(
                        arm=wrist_arm,
                        link_name='camera',
                        T_link_camera=optical.tolist(),
                        T_world_base=np.eye(4).tolist(),
                        K=[[30.0, 0.0, 10.0], [0.0, 30.0, 10.0], [0.0, 0.0, 1.0]],
                        distortion=[],
                        image_size=[20, 20],
                    )
                ),
            )
            wrist_sample = dict(
                frame_id=fid,
                T_world_camera=(link @ fixed @ optical).tolist(),
                T_world_link=(link @ fixed).tolist(),
                T_world_base=np.eye(4).tolist(),
                source_kind='robot_kinematics',
                sim_step=5,
                step_domain='control_step',
            )
            _, wrist_png = cv2.imencode('.png', np.zeros((20, 20, 3), np.uint8))
            cams = [
                dict(
                    camera_id='wrist',
                    frame_id=fid,
                    width=20,
                    height=20,
                    sim_step=5,
                    step_domain='control_step',
                    calibration_ref=put_json(wrist_cal, 'calibration', cal_scope),
                    pose_sample_ref=put_json(wrist_sample, 'camera_pose', cal_scope),
                    rgb_ref=store.put_artifact(wrist_png.tobytes(), 'rgb', 'image/png', rgb_scope),
                    pixel_transform='identity',
                )
            ]
            for idx, (cname, cpose) in enumerate((('source', src_pose), ('target', tgt_pose))):
                sg_cal = dict(
                    reference_frame_id=frame_reference if idx == 1 else 'env_origin',
                    K=sg_k,
                    distortion=list(distortion) if distortion is not None else [0.0, 0.0, 0.0, 0.0, 0.0],
                    distortion_model=distortion_model,
                    image_size=[100, 80],
                    T_world_camera=cpose.tolist(),
                )
                px = np.zeros((80, 100, 3), np.uint8)
                px[:, :, 0] = np.arange(100, dtype=np.uint8)
                px[:, :, 1] = idx * 90
                _, sg_png = cv2.imencode('.png', px)
                cams.append(
                    dict(
                        camera_id=cname,
                        frame_id=f'{fid}-{cname}',
                        width=100,
                        height=80,
                        sim_step=5,
                        step_domain='control_step',
                        calibration_ref=put_json(sg_cal, 'calibration', cal_scope),
                        rgb_ref=store.put_artifact(b'corrupt' if corrupt_rgb else sg_png.tobytes(), rgb_kind, 'image/png', rgb_scope),
                        pixel_transform='identity',
                    )
                )
            bundle = dict(
                schema_version='0.1',
                **(obs_identity or identity),
                world_kind='observed',
                observation_id=oid,
                observation_profile=profile,
                sim_step=5,
                step_domain='control_step',
                clock_domain='env/generation-1',
                calibration_id='fixture-calibration',
                robot_model_id='fixture-arm',
                robot_model_ref=put_json(base_model, 'robot_model', cal_scope),
                robot_state_ref=put_json(state, 'robot_state', cal_scope),
                cameras=cams,
            )
            store.save('observation', oid, bundle)
            if import_scene:
                h.scene.import_observation(bundle)
            h.latest = bundle
            return oid

        def rv_args(oid):
            return dict(
                observation_id=oid,
                camera_id='wrist',
                eye_position_m=[0.4, 0.1, 0.5],
                target_position_m=[0.1, 0.2, 0.2],
                up_vector=[0.0, 0.0, 1.0],
            )

        def sg_args(oid):
            return dict(
                observation_id=oid,
                source_camera_id='source',
                target_camera_id='target',
                source_pixel=[49.0, 30.0],
            )

        # 1) 正常真实 owner 往返：actor / program 等价、reader 本身零写记录、零哈希/路径泄露、不改写原图
        from ni_rua.runtime.tools.plugins.robotic_view import read_robotic_view
        from ni_rua.runtime.tools.plugins.stereo_guide import read_stereo_guide

        oid_ok = save_bundle()
        orig_read = store.read

        def _guarded_read(kind, identifier):
            self.assertEqual(kind, 'observation')
            return orig_read(kind, identifier)

        with patch.object(store, 'read', side_effect=_guarded_read), \
             patch.object(store, 'save', side_effect=AssertionError('read_robotic_view 不得写记录')), \
             patch.object(store, 'put_artifact', side_effect=AssertionError('read_robotic_view 不得写衍生构件')):
            read_robotic_view(h, rv_args(oid_ok), {})

        with patch.object(store, 'read', side_effect=_guarded_read), \
             patch.object(store, 'save', side_effect=AssertionError('read_stereo_guide 不得写记录')):
            sg_direct = read_stereo_guide(h, sg_args(oid_ok), {})
        self.assertEqual(sg_direct.data['status'], 'guided')

        rv_actor = h.invoke('robotic_view', rv_args(oid_ok), dict(tool_call_id='rv-ok'), role='actor')
        rv_prog = h.invoke('robotic_view', rv_args(oid_ok), dict(tool_call_id='rv-prog'), role='program')
        self.assertTrue(rv_actor.success, rv_actor.contract_error)
        self.assertEqual(rv_actor.data, rv_prog.data)
        self.assertEqual(rv_actor.data['status'], 'calculated')
        rv_wire = canonical(rv_actor.data) + canonical(h.actor_output.present(rv_actor))
        for forbidden in ('sha256', 'calibration_ref', 'robot_model_ref', str(self.root), 'artifact:'):
            self.assertNotIn(forbidden, rv_wire)

        orig_refs = [c['rgb_ref'] for c in h.latest['cameras'] if c['camera_id'] in ('source', 'target')]
        orig_digests = [hashlib.sha256(store.resolve_artifact(r)[1]).hexdigest() for r in orig_refs]
        sg_actor = h.invoke('stereo_guide', sg_args(oid_ok), dict(tool_call_id='sg-ok'), role='actor')
        sg_prog = h.invoke('stereo_guide', sg_args(oid_ok), dict(tool_call_id='sg-prog'), role='program')
        self.assertTrue(sg_actor.success, sg_actor.contract_error)
        self.assertEqual(sg_actor.data, sg_prog.data)
        self.assertEqual(sg_actor.data['status'], 'guided')
        np.testing.assert_allclose(sg_actor.data['target_line'], [0.0, 1.0, -30.0], atol=1e-10)
        self.assertEqual(orig_digests, [hashlib.sha256(store.resolve_artifact(r)[1]).hexdigest() for r in orig_refs])
        sg_wire = canonical(h.actor_output.present(sg_actor))
        self.assertNotIn('sha256', sg_wire)
        self.assertNotIn(str(self.root), sg_wire)

        def inv(name, payload, cid='call', role='actor'):
            return h.invoke(name, payload, dict(tool_call_id=cid), role=role)

        # 2) stale / missing observation、active action 与 unknown/stop gate 拒绝（新观测不得重开 stop 门禁）
        stale_oid = oid_ok
        cur_oid = save_bundle()
        for tool_name, stale_payload, cur_payload in (
            ('robotic_view', rv_args(stale_oid), rv_args(cur_oid)),
            ('stereo_guide', sg_args(stale_oid), sg_args(cur_oid)),
        ):
            with patch.object(store, 'put_artifact', side_effect=AssertionError('拒绝前不得写入衍生图像')):
                with self.assertRaisesRegex(ValueError, 'current_idle'):
                    inv(tool_name, stale_payload, 'stale')
                h.active_action = 'in-flight'
                with self.assertRaisesRegex(ValueError, 'current_idle'):
                    inv(tool_name, cur_payload, 'active')
                h.active_action = None
                saved_latest = h.latest
                h.latest = None
                with self.assertRaisesRegex(ValueError, 'current_idle'):
                    inv(tool_name, cur_payload, 'missing')
                h.latest = saved_latest

        h.accepting = False
        h.states['unknown-action'] = dict(status='status_unknown', stop_confirmation=False)
        states_snapshot = deepcopy(h.states)
        fresh_under_stop = save_bundle()
        for tool_name, payload in (('robotic_view', rv_args(fresh_under_stop)), ('stereo_guide', sg_args(fresh_under_stop))):
            with patch.object(store, 'put_artifact', side_effect=AssertionError('stop 门禁下不得写入衍生构件')):
                with self.assertRaisesRegex(ValueError, 'current_idle'):
                    inv(tool_name, payload, 'stop-gate')
        self.assertFalse(h.accepting)
        self.assertEqual(h.states, states_snapshot)
        h.accepting = True
        h.states.clear()

        # 3) 读取中途状态变化（最终新鲜度竞态检查）在返回或写入衍生图前拒绝
        orig_camera_fn = h.geometry.camera
        for race_mode in ('active', 'observation', 'stop'):
            race_oid = save_bundle()
            h.active_action = None
            h.accepting = True

            def _mutate_during_read(*a, _mode=race_mode, **kw):
                res = orig_camera_fn(*a, **kw)
                if _mode == 'active':
                    h.active_action = 'concurrent'
                elif _mode == 'observation':
                    h.latest = dict(h.latest, observation_id='other-obs')
                else:
                    h.accepting = False
                return res

            with patch.object(h.geometry, 'camera', side_effect=_mutate_during_read), \
                 patch.object(store, 'put_artifact', side_effect=AssertionError('竞态拒绝前不得写入衍生图像')):
                with self.assertRaisesRegex(ValueError, 'current_idle'):
                    inv('robotic_view', rv_args(race_oid), 'rv-race')
            h.active_action = None
            h.accepting = True
            race_oid_sg = save_bundle()
            with patch.object(h.geometry, 'camera', side_effect=_mutate_during_read), \
                 patch.object(store, 'put_artifact', side_effect=AssertionError('竞态拒绝前不得写入衍生图像')):
                with self.assertRaisesRegex(ValueError, 'current_idle'):
                    inv('stereo_guide', sg_args(race_oid_sg), 'sg-race')
            h.active_action = None
            h.accepting = True

        # 4) robotic_view 专属身份、observer 作用域、异源时钟、固定相机、外来 TCP 与安装位姿失配拒绝
        cross_oid = save_bundle(obs_identity=dict(identity, episode_id='other'))
        with self.assertRaisesRegex(ValueError, 'outside_current_episode'):
            inv('robotic_view', rv_args(cross_oid), 'rv-cross')

        obs_scope_oid = save_bundle(cal_scope='observer', import_scene=False)
        with patch.object(h.geometry, 'camera', side_effect=AssertionError('observer 构件必须在几何求解前拦截')):
            with self.assertRaises(ValueError):
                inv('robotic_view', rv_args(obs_scope_oid), 'rv-observer')

        unsync_oid = save_bundle(state_step=4)
        with self.assertRaisesRegex(ValueError, 'unsynchronized'):
            inv('robotic_view', rv_args(unsync_oid), 'rv-unsync')

        fixed_cam_oid = save_bundle(wrist_arm='center')
        with self.assertRaises(ValueError):
            inv('robotic_view', rv_args(fixed_cam_oid), 'rv-fixed')

        foreign_tcp_oid = save_bundle(tcp_id='other-tcp')
        with self.assertRaisesRegex(ValueError, 'native_link6'):
            inv('robotic_view', rv_args(foreign_tcp_oid), 'rv-tcp')

        mismatch_oid = save_bundle(ee_dx=0.02)
        with self.assertRaisesRegex(ValueError, 'camera_native_link6_mismatch'):
            inv('robotic_view', rv_args(mismatch_oid), 'rv-mismatch')

        # 5) stereo_guide 专属身份、observer RGB/标定权限、RGB kind、时钟/参考系、损坏图像与非零畸变拒绝（均零衍生写入）
        obs_rgb_oid = save_bundle(rgb_scope='observer', import_scene=False)
        with patch.object(h.scene, 'artifact_resolver', None):
            h.scene.import_observation(store.read('observation', obs_rgb_oid))
        with patch.object(store, 'put_artifact', side_effect=AssertionError('observer RGB 拒绝前不得写入衍生图像')):
            with self.assertRaisesRegex(ValueError, 'artifact_access_scope_denied'):
                inv('stereo_guide', sg_args(obs_rgb_oid), 'sg-obs-rgb')

        obs_cal_oid = save_bundle(cal_scope='observer', import_scene=False)
        with patch.object(h.geometry, 'camera', side_effect=AssertionError('observer 标定必须在几何求解前拒绝')), \
             patch.object(store, 'put_artifact', side_effect=AssertionError('observer 标定拒绝前不得写入衍生图像')):
            with self.assertRaisesRegex(ValueError, 'artifact_access_scope_denied'):
                inv('stereo_guide', sg_args(obs_cal_oid), 'sg-obs-cal')

        sg_reject_variants = (
            (dict(obs_identity=dict(identity, episode_id='other')), 'outside_current_episode'),
            (dict(rgb_kind='geometry_overlay'), 'formal_rgb'),
            (dict(profile='rgb_proprio_v1', import_scene=False), 'observation_profile|permitted_calibration'),
            (dict(state_step=4), 'unsynchronized|sim_step'),
            (dict(frame_reference='other'), 'reference_frame_mismatch'),
            (dict(corrupt_rgb=True), 'invalid_rgb|decode|image'),
            (dict(distortion_model='fisheye'), 'unsupported_distortion_model|zero_distortion'),
        )
        for variant, pattern in sg_reject_variants:
            with self.subTest(sg_variant=variant):
                bad_oid = save_bundle(**variant)
                with patch.object(store, 'put_artifact', side_effect=AssertionError('拒绝路径不得写入衍生图像')):
                    with self.assertRaisesRegex(ValueError, pattern):
                        inv('stereo_guide', sg_args(bad_oid), 'sg-bad')

        robot.observe.assert_not_called()
        robot.execute.assert_not_called()
        robot.preview_batch.assert_not_called()
        self.assertEqual(h.action_count, 0)
