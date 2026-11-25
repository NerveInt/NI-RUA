"""
- test_tool_errors.py: Rejected calls keep codes and expose field-specific, evidence-safe repairs.
- test_tool_parallel_contract.py: ToolSpec.parallel / ToolRegistry.parallel_safe: the registry-owned concurrency contract (no model, no robot).
"""
from support.program_fixture import program_services
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from jsonschema import Draft202012Validator
from ni_rua.core.tool_errors import error_payload, error_response, schema_input_error
from dataclasses import replace
from ni_rua.core.config import Config
from ni_rua.core.storage.store import Store
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.tools.builtins import create_registry, output_schemas
from ni_rua.runtime.tools.plugins.robotic_view import make_robotic_view_tool
from ni_rua.runtime.tools.plugins.stereo_guide import make_stereo_guide_tool
from ni_rua.runtime.tools.registry import ToolRegistry, ToolSpec
from support.fixture_robot import FixtureRobot
from support.paths import FIXTURES_ROOT

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_tool_errors.py ---

class ToolErrorsTests(unittest.TestCase):
    def test_boolean_schema_rejections_remain_actionable_failures(self):
        for schema, value in [({'type': 'object', 'properties': {'blocked': False}}, {'blocked': 1}),
                ({'oneOf': [False, {'type': 'number'}]}, 'bad'),
                ({'not': True}, 1),
                ({'type': 'object', 'required': ['kind'], 'properties': {'kind': False}}, {}),
                ({'type': 'object', 'required': ['blocked'], 'properties': {'blocked': False}}, {})]:
            error = next(Draft202012Validator(schema).iter_errors(value))
            result = error_payload(schema_input_error(error))
            self.assertEqual(result['code'], 'invalid_tool_arguments')
            self.assertTrue(result['suggestion'])

    def test_action_errors_explain_actual_legal_fields_and_options(self):
        from ni_rua.runtime.tools.builtins import create_registry
        from ni_rua.core.tool_errors import ToolInputError
        registry = create_registry()
        cases = [
            ({'id': 'jaw', 'kind': 'gripper', 'arm': 'left', 'gripper': 1}, 'goal_normalized'),
            ({'id': 'look', 'kind': 'observe'}, 'nonblank reason'),
            ({'id': 'home', 'kind': 'home', 'arm': 'left'}, 'joint/home/reset'),
            ({'id': 'move', 'kind': 'move', 'arm': 'left'}, 'grasp_position_m, angles_deg OR position_m, quaternion_xyzw'),
            ({'id': 'move', 'kind': 'move', 'arm': 'left', 'grasp_position_m': [0,0,1], 'angles_deg': [0,91,0]}, 'maximum'),
            ({'id': 'jaw', 'kind': 'gripper', 'arm': 'left', 'goal_normalized': 1, 'speed_preset': 'slow'}, 'gripper_speed_preset')]
        for stage, expected in cases:
            with self.subTest(stage=stage):
                with self.assertRaises(ToolInputError) as caught:
                    registry.validate('robot_action', {'observation_id': 'obs1', 'stages': [stage]}, role='actor')
                payload = error_payload(caught.exception)
                self.assertIn(expected, payload['suggestion'])
                self.assertTrue(payload['field_path'].startswith('/stages/0'))
                self.assertNotIn('executed_native_steps', payload)

    def test_runtime_action_modes_give_repair_without_replay_claim(self):
        for code, required in [('cached_batch_commands_cannot_be_changed', 'expected_cursor'),
                ('robot_action_plan_unknown_or_expired', 'never replay'),
                ('inspect_requires_one_plan_action_or_trajectory_reference', 'exactly one'),
                ('trajectory_native_steps_and_pagination_are_mutually_exclusive', 'max_samples')]:
            payload = error_payload(ValueError(code), tool='robot_action', arguments={'mode': 'execute'})
            self.assertEqual(payload['code'], code)
            self.assertIn(required, payload['suggestion'])

    def test_unavailable_and_unknown_tools_supply_catalog_guidance(self):
        from ni_rua.runtime.tools.builtins import create_registry
        from ni_rua.core.tool_errors import ToolInputError
        registry = create_registry()
        with self.assertRaises(ToolInputError) as caught:
            registry.validate('robot_home', {}, role='actor')
        self.assertIn('robot_action', caught.exception.suggestion)
        from unittest.mock import patch
        with patch.object(registry, 'catalog', return_value=[{'name': 'robot_observe'}]):
            issue = registry._unavailable_error('depth_not_enabled_for_run', 'geometry_query', None, 'actor')
        self.assertIn('robot_observe', issue.suggestion)
        self.assertIn('cannot enable', issue.suggestion)
        self.assertIn('do not', issue.suggestion.lower())

    def test_schema_arrays_keep_index_and_actual_without_full_schema_echo(self):
        schema = dict(type="object", properties=dict(values=dict(type="array", items=dict(type="number", minimum=0))))
        error = next(Draft202012Validator(schema).iter_errors(dict(values=[1, -2])))
        payload = error_payload(error, tool="fixture", arguments=dict(values=[1, -2]))
        self.assertEqual(payload["field_path"], "/values/1")
        self.assertEqual(payload["actual"], -2)
        self.assertEqual(payload["code"], "invalid_tool_arguments")
        self.assertNotIn("Failed validating", payload["error"])

    def test_release_fix_does_not_fabricate_evidence_or_invalid_released_state(self):
        for state in ("retained", "released_unverified"):
            args = dict(plan=dict(stages=[dict(operation_context=dict(state=state))]))
            payload = error_payload(ValueError("operation_context_release_evidence_required"), tool="task_plan", arguments=args)
            self.assertEqual(payload["field_path"], "/plan/stages/0/operation_context/release_observation_id")
            self.assertEqual(payload["minimal_example"]["state"], "unknown")
            self.assertNotIn("release_observation_id", payload["minimal_example"])
            self.assertIn("do not fabricate", payload["suggestion"])

    def test_plain_runtime_error_stays_failure_and_does_not_claim_zero_motion(self):
        response = error_response(ValueError("worker_connection_lost"), tool="robot_action", arguments=dict(mode="direct"))
        self.assertFalse(response["success"])
        value = json.loads(response["contentItems"][0]["text"])
        self.assertEqual(value["error"], "worker_connection_lost")
        self.assertEqual(value["field_path"], "")
        self.assertIn("no motion or task outcome is inferred", value["outcome_semantics"])
        self.assertNotIn("executed_native_steps", value)

    def test_depth_refresh_guidance_is_accepted_and_keeps_registered_camera(self):
        from ni_rua.perception.depth.depth_runtime import DepthRuntime
        payload = error_payload(ValueError('depth_target_requires_refresh'), tool='depth_query',
            arguments={'request': {'kind': 'region_geometry', 'target_id': 'target'}})
        request = payload['minimal_example']['request']
        self.assertEqual(request['kind'], 'refresh_target')
        self.assertNotIn('camera_id', request)
        measured = []
        def measure(arguments, correlation):
            measured.append(arguments)
            return {'status': 'supported', 'measurement_id': 'measurement'}
        owner = SimpleNamespace(_current=lambda observation_id: None, _targets=lambda: {},
            _target=lambda identifier, current=False: {'target_id': identifier, 'camera_id': 'original_camera'},
            _measure=measure, _save_targets=lambda targets: None,
            store=SimpleNamespace(event=lambda *args: None))
        result = DepthRuntime._register(owner, request, {})
        self.assertEqual(result['target_status'], 'tracked')
        self.assertEqual(measured[0]['camera_id'], 'original_camera')
        with self.assertRaisesRegex(ValueError, 'invalid_depth_target_fields'):
            DepthRuntime._register(owner, dict(request, camera_id='replacement_camera'), {})

    def test_malformed_arguments_cannot_break_error_formatting(self):
        args = dict(stages=None, plan=[])
        response = error_response(ValueError("invalid_task_plan: provide plan"), tool="robot_action", arguments=args)
        self.assertFalse(response["success"])
        self.assertIn("error", json.loads(response["contentItems"][0]["text"]))

    def test_model_and_program_boundaries_keep_failure_status_and_machine_fields(self):
        from ni_rua.cortex.agents.codex.session import RecordedClient
        from ni_rua.runtime.programs.runner import ProgramRunner
        from ni_rua.core.storage.store import Store
        from ni_rua.runtime.tools.builtins import create_registry
        registry = create_registry()
        def failure(*args):
            registry.validate('task_plan', arguments, role='actor')
            raise ValueError("operation_context_release_evidence_required")
        arguments = dict(plan=dict(stages=[dict(operation_context=dict(state="retained"))]))
        with tempfile.TemporaryDirectory() as directory:
            store=Store(Path(directory), dict(run_id="errors", episode_id="ep", env_instance_id="env", env_generation=1))
            runner=ProgramRunner(program_services(SimpleNamespace(store=store, dispatch=failure)))
            entry=dict(cancel=threading.Event(), deadline=time.monotonic()+30, call_lock=threading.RLock(),
                       job_id="job", calls=0, max_calls=1, correlation={})
            try:
                model = RecordedClient._approval(SimpleNamespace(tool_handler=failure, store=store), "item/tool/call",
                    dict(tool="task_plan", arguments=arguments))
                self.assertFalse(model["success"])
                model_value = json.loads(model["contentItems"][0]["text"])
                result=runner.dispatch(entry, dict(operation_id="call", tool="task_plan", arguments=arguments))
                self.assertFalse(result["success"])
                self.assertEqual(json.loads(result["contentItems"][0]["text"]), model_value)
                self.assertEqual(runner.dispatch(entry, dict(operation_id="call", tool="task_plan", arguments=arguments)), result)
                self.assertEqual(entry["calls"], 1)
            finally:
                runner.close();store.close()



# --- Folded from tests/test_tool_parallel_contract.py ---

GOLDEN = FIXTURES_ROOT / 'tool_parallel_classification.json'
# Tools that are parallel-safe regardless of arguments.
ALWAYS_PARALLEL = frozenset({'robot_information', 'robot_contact_overlay', 'robot_action_status', 'robot_capabilities',
                             'history_list_actions', 'history_view_observation', 'history_read_trace', 'program_status',
                             'vision_compare'})
REQUEST_DEPENDENT = frozenset({'geometry_query', 'depth_query', 'task_plan', 'operation_memory', 'pose_plan'})

# W2 重构相对 c4ff16a 历史基线表有意修正的并发分类差异（修正死分支与写副作用漏判）
_GOLDEN_PARALLEL_OVERRIDES = {
    # depth_query: target_status / skill_help 均为纯读；非字符串 kind=7 按保守串行
    ('depth_query', '{"request":{"kind":"skill_help"}}'): True,
    ('depth_query', '{"request":{"kind":"target_status"}}'): True,
    ('depth_query', '{"request":{"kind":7}}'): False,
    # geometry_query: 移除死键 commit_scene，非对象 request 按保守串行，写深度/目标表分支串行
    ('geometry_query', '{}'): False,
    ('geometry_query', '{"mode":"execute"}'): False,
    ('geometry_query', '{"operation":"read"}'): False,
    ('geometry_query', '{"observation_id":"obs-1"}'): False,
    ('geometry_query', '{"commit_scene":true}'): False,
    ('geometry_query', '{"commit_scene":false}'): False,
    ('geometry_query', '{"commit_scene":1}'): False,
    ('geometry_query', '{"commit_scene":0}'): False,
    ('geometry_query', '{"commit_scene":""}'): False,
    ('geometry_query', '{"commit_scene":"yes"}'): False,
    ('geometry_query', '{"commit_scene":null}'): False,
    ('geometry_query', '{"request":"not-an-object"}'): False,
    ('geometry_query', '{"request":{"commit_scene":true}}'): True,
    ('geometry_query', '{"request":{"kind":"fk","commit_scene":1}}'): True,
    ('geometry_query', '{"request":{"kind":"fk"},"commit_scene":true}'): True,
    ('geometry_query', '{"request":{"kind":"register_target"}}'): False,
    ('geometry_query', '{"request":{"kind":"grasp_candidates"}}'): False,
    # pose_plan: 仅 targets/place 为纯读可并发，grasp 写回合槽且无 request/非法 kind 按保守串行
    ('pose_plan', '{}'): False,
    ('pose_plan', '{"request":{}}'): False,
    ('pose_plan', '{"request":{"kind":"help"}}'): False,
    ('pose_plan', '{"mode":"execute"}'): False,
    ('pose_plan', '{"operation":"read"}'): False,
    ('pose_plan', '{"observation_id":"obs-1"}'): False,
    # robot_observe: 移除死分支 observe_parallel，一律串行
    ('robot_observe', '{"observation_id":"obs-1","crop":{"camera":"head"}}'): False,
    ('robot_observe', '{"observation_id":"","crop":{}}'): False,
    ('robot_observe', '{"observation_id":0,"crop":[]}'): False,
}


def full_registry():
    """Every tool the gateway can register: built-ins (pose_skill on) and both optional plugins."""
    return create_registry([make_robotic_view_tool(lambda *args: None), make_stereo_guide_tool(lambda *args: None)],
                           runtime=Config.from_dict({'features': {'pose_skill': True}}))


def fixture_spec(name, parallel):
    return ToolSpec(dict(type='function', name=name, description='Fixture tool.',
                         inputSchema=dict(type='object', additionalProperties=True)),
                    lambda context, arguments, correlation: None, parallel=parallel,
                    output_schemas=output_schemas('robot_wait'))


class GoldenClassificationTest(unittest.TestCase):
    """The table was produced by the pre-refactor RecordedClient._is_parallel_read_tool at c4ff16a."""

    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(GOLDEN.read_text())
        cls.registry = full_registry()

    def test_table_covers_every_registered_tool(self):
        self.assertEqual(sorted(self.registry.names), self.golden['registered_tools'])
        self.assertEqual({row['tool'] for row in self.golden['rows']}, set(self.registry.names))

    def test_registry_reproduces_every_golden_row(self):
        for row in self.golden['rows']:
            key = (row['tool'], json.dumps(row['arguments'], separators=(',', ':')))
            expected = _GOLDEN_PARALLEL_OVERRIDES.get(key, row['parallel'])
            with self.subTest(tool=row['tool'], arguments=row['arguments']):
                self.assertIs(self.registry.parallel_safe(row['tool'], row['arguments']), expected)

    def test_retired_allowlist_names_are_not_registered_and_are_ordered(self):
        retired = self.golden['retired_names_formerly_parallel']
        self.assertIn('scene_query', retired)
        for name in retired:
            with self.subTest(tool=name):
                self.assertNotIn(name, self.registry.names)
                self.assertFalse(self.registry.parallel_safe(name, {}))

    def test_declarations_match_the_former_allowlist_shape(self):
        for name in self.registry.names:
            spec = self.registry.spec(name, role='actor')
            with self.subTest(tool=name):
                if name in ALWAYS_PARALLEL:
                    self.assertIs(spec.parallel, True)
                elif name in REQUEST_DEPENDENT:
                    self.assertTrue(callable(spec.parallel))
                else:
                    self.assertIs(spec.parallel, False)

    def test_classification_does_not_use_the_read_effect(self):
        # robot_observe 始终串行（移除无 Schema 支持的死分支 observe_parallel）。
        spec = self.registry.spec('robot_observe', role='actor')
        self.assertEqual(spec.effect_for({}), 'read')
        self.assertIs(spec.parallel, False)
        self.assertFalse(self.registry.parallel_safe('robot_observe', {}))
        self.assertFalse(self.registry.parallel_safe('robot_observe', {'observation_id': 'obs-1', 'crop': {}}))


class ParallelSafeEdgeCasesTest(unittest.TestCase):
    def setUp(self):
        self.registry = full_registry()

    def test_unknown_or_non_string_tool_is_ordered(self):
        for name in ('not_registered', None, 7, ['robot_information'], {'tool': 'robot_information'}):
            with self.subTest(name=name):
                self.assertFalse(self.registry.parallel_safe(name, {}))

    def test_non_object_arguments_are_ordered(self):
        for arguments in (None, [], 'request', 3, ['observation_id']):
            for name in ('robot_information', 'geometry_query', 'robot_observe'):
                with self.subTest(tool=name, arguments=arguments):
                    self.assertFalse(self.registry.parallel_safe(name, arguments))

    def test_unhashable_request_kind_is_ordered_without_raising(self):
        for name, arguments in (('depth_query', {'request': {'kind': ['depth_points']}}),
                                ('operation_memory', {'request': {'kind': {'help': True}}})):
            with self.subTest(tool=name):
                self.assertFalse(self.registry.parallel_safe(name, arguments))

    def test_failing_or_non_boolean_classifier_is_ordered(self):
        def broken(arguments):
            raise RuntimeError('fixture classifier failure')
        registry = ToolRegistry([fixture_spec('fixture_broken', broken),
                                 fixture_spec('fixture_truthy', lambda arguments: 'yes')])
        self.assertFalse(registry.parallel_safe('fixture_broken', {}))
        # parallel_for normalises a callable's verdict to bool.
        self.assertTrue(registry.parallel_safe('fixture_truthy', {}))
        self.assertIs(registry.spec('fixture_truthy', role='actor').parallel_for({}), True)

    def test_parallel_contract_is_validated_at_registration(self):
        self.assertIs(fixture_spec('fixture_default', False).parallel, False)
        self.assertIs(ToolSpec(dict(type='function', name='fixture_plain', description='Fixture.',
                                    inputSchema={'type': 'object'}), None).parallel, False)
        for invalid in ('yes', 1, None, {'parallel': True}):
            with self.subTest(parallel=invalid), self.assertRaisesRegex(ValueError, 'invalid_tool_parallel:fixture_bad'):
                ToolRegistry([fixture_spec('fixture_bad', invalid)])

    def test_inspection_copies_keep_the_contract(self):
        registry = ToolRegistry([fixture_spec('fixture_read', True)])
        self.assertIs(registry.spec('fixture_read', role='program').parallel, True)
        self.assertIs(registry._specs['fixture_read'].parallel, True)
        self.assertIs(replace(registry.spec('fixture_read', role='actor'), parallel=False).parallel, False)
        self.assertTrue(registry.parallel_safe('fixture_read', {}))


class ManifestParallelFieldTest(unittest.TestCase):
    def test_manifest_reports_constant_and_request_dependent_contracts(self):
        registry = ToolRegistry([fixture_spec('fixture_parallel', True), fixture_spec('fixture_ordered', False),
                                 fixture_spec('fixture_dynamic', lambda arguments: bool(arguments.get('read')))])
        runtime = SimpleNamespace(disabled_tools=frozenset({'fixture_ordered'}))
        tools = {item['name']: item for item in registry.manifest(runtime)['tools']}
        self.assertIs(tools['fixture_parallel']['parallel'], True)
        self.assertIs(tools['fixture_ordered']['parallel'], False)
        self.assertEqual(tools['fixture_dynamic']['parallel'], 'request_dependent')
        # The field is reported for unavailable tools too, like effect.
        self.assertFalse(tools['fixture_ordered']['available'])
        json.dumps(registry.manifest(runtime))

    def test_harness_catalog_manifest_declares_builtin_concurrency(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        root = Path(temporary.name)
        store = Store(root / 'run', dict(run_id='parallel-manifest', episode_id='ep', env_instance_id='env',
                                         env_generation=1))
        self.addCleanup(store.close)
        harness = Harness(store, FixtureRobot(root))
        manifest = harness.tool_registry.manifest(harness)
        tools = {item['name']: item for item in manifest['tools']}
        self.assertTrue(tools)
        for name, item in tools.items():
            with self.subTest(tool=name):
                expected = (True if name in ALWAYS_PARALLEL else
                            'request_dependent' if name in REQUEST_DEPENDENT else False)
                self.assertEqual(item['parallel'], expected)
                self.assertIs(type(item['parallel']), str if expected == 'request_dependent' else bool)


class W2ToolDefinitionAndContractTests(unittest.TestCase):
    """W2 工具定义、几何批次门禁、观测采样来源、输出契约与帮助校验回归测试。"""

    def test_definitions_cleanup_conditionals_and_layered_guide_support(self):
        from ni_rua.core.tool_errors import ToolInputError
        from ni_rua.runtime.tools import catalog_guide
        from ni_rua.runtime.tools.definitions import TOOLS, public_definitions

        legacy_program = next(item for item in TOOLS if item['name'] == 'program_run')
        self.assertNotIn('max_calls', legacy_program['inputSchema']['properties'])
        legacy_policy = next(item for item in TOOLS if item['name'] == 'robot_policy_step')
        self.assertNotIn('Astra', legacy_policy['description'])
        legacy_view = next(item for item in TOOLS if item['name'] == 'history_view_observation')
        self.assertEqual(legacy_view['inputSchema']['properties']['scale']['default'], 1)

        defs_off = {d['name']: d for d in public_definitions(catalog_profile='layered_v3', vggt_height=False)}
        defs_on = {d['name']: d for d in public_definitions(catalog_profile='layered_v3', vggt_height='optional')}
        # program_status 收拢重复说明，不再内嵌整段 _PROGRAM_OUTPUT_HELP/_PROGRAM_VISUAL_HELP
        self.assertIn('program_run', defs_off['program_status']['description'])
        self.assertNotIn('no head/tail byte cap removes middle diagnostics', defs_off['program_status']['description'])

        # 确认新增 allOf 与 batch 继承字段均可被 catalog_guide 正常渲染（不退回 UnsupportedSchema）
        for name in ('program_run', 'robot_information', 'history_view_observation', 'geometry_query'):
            with self.subTest(tool=name):
                self.assertIn(catalog_guide._INTRO, defs_on[name]['description'])

        registry = create_registry()
        # robot_information 条件约束
        registry.validate('robot_information', {'section': 'resources'}, role='actor')
        registry.validate('robot_information', {'section': 'resource', 'resource_id': 'geometry_model'}, role='actor')
        for bad in ({'section': 'resource'}, {'section': 'default', 'resource_id': 'x'}, {'resource_id': 'x'}):
            with self.subTest(robot_information=bad), self.assertRaises(ToolInputError):
                registry.validate('robot_information', bad, role='actor')

        # program_run start/resume 条件约束
        registry.validate('program_run', {'source': 'print(1)'}, role='actor')
        registry.validate('program_run', {'operation': 'start', 'source_ref': 'artifact:1'}, role='actor')
        registry.validate('program_run', {'operation': 'resume', 'job_id': 'j1', 'review_id': 'r1',
                                          'decision': 'approve', 'detail': 'full'}, role='actor')
        for bad in (
            {},
            {'operation': 'start'},
            {'source': 'print(1)', 'job_id': 'j1'},
            {'operation': 'resume', 'job_id': 'j1'},
            {'operation': 'resume', 'job_id': 'j1', 'review_id': 'r1', 'decision': 'approve', 'timeout_s': 10},
        ):
            with self.subTest(program_run=bad), self.assertRaises(ToolInputError):
                registry.validate('program_run', bad, role='actor')

        # geometry_query.batch 继承字段：VGGT 关闭 4 项、开启 5 项，且保留省略 kind 兼容
        batch_off = next(b for b in defs_off['geometry_query']['inputSchema']['properties']['request']['oneOf']
                         if b.get('properties', {}).get('kind', {}).get('const') == 'batch')
        batch_on = next(b for b in defs_on['geometry_query']['inputSchema']['properties']['request']['oneOf']
                        if b.get('properties', {}).get('kind', {}).get('const') == 'batch')
        for field in ('plane_origin_m', 'plane_normal', 'plane_origin_alt_m', 'target_camera_id'):
            self.assertIn(field, batch_off['properties'])
            self.assertIn(field, batch_on['properties'])
        self.assertNotIn('plane_source', batch_off['properties'])
        self.assertIn('plane_source', batch_on['properties'])

        reg_vggt = create_registry(runtime=Config.from_dict({'features': {'vggt_height': True}}))
        reg_vggt.validate('geometry_query', {
            'request': {
                'observation_id': 'obs_1',
                'camera_id': 'cam_head',
                'plane_origin_m': [0.0, 0.0, 0.76],
                'plane_normal': [0.0, 0.0, 1.0],
                'plane_source': 'caller',
                'plane_origin_alt_m': [0.0, 0.0, 0.80],
                'target_camera_id': 'cam_right_wrist',
                'queries': [{'kind': 'pixel_plane', 'pixel': [100, 100]}],
            }
        }, role='actor')

    def test_geometry_batch_recursive_classifiers_admission_and_error_isolation(self):
        from unittest.mock import patch
        from ni_rua.core.tool_errors import ToolInputError
        from ni_rua.runtime.tools import handlers

        registry = full_registry()
        # 递归批次展开：嵌套写入项导致整批非并发且 effect='record'
        nested_write = {'request': {'kind': 'batch', 'queries': [
            {'kind': 'pixel_ray', 'pixel': [10, 10]},
            {'queries': [{'kind': 'reanchor_tracks'}]},
        ]}}
        nested_depth_write = {'request': {'queries': [
            {'kind': 'fk'},
            {'kind': 'batch', 'queries': [{'kind': 'depth_points'}]},
        ]}}
        nested_read = {'request': {'kind': 'batch', 'queries': [
            {'kind': 'pixel_ray'},
            {'kind': 'batch', 'queries': [{'kind': 'fk'}, {'kind': 'target_status'}]},
        ]}}
        self.assertFalse(registry.parallel_safe('geometry_query', nested_write))
        self.assertEqual(registry.effect_of('geometry_query', nested_write), 'record')
        self.assertFalse(registry.parallel_safe('geometry_query', nested_depth_write))
        self.assertEqual(registry.effect_of('geometry_query', nested_depth_write), 'record')
        self.assertTrue(registry.parallel_safe('geometry_query', nested_read))
        self.assertEqual(registry.effect_of('geometry_query', nested_read), 'read')
        # depth_query 四分法：target_status/skill_help/help 纯读，其余写测量或改目标表均串行
        for read_kind in ('help', 'skill_help', 'target_status'):
            self.assertTrue(registry.parallel_safe('depth_query', {'request': {'kind': read_kind}}))
            self.assertEqual(registry.effect_of('depth_query', {'request': {'kind': read_kind}}), 'read')
        for write_kind in ('depth_points', 'region_geometry', 'grasp_candidates', 'place_candidates',
                           'compare_regions', 'register_target', 'refresh_target', 'forget_target'):
            self.assertFalse(registry.parallel_safe('depth_query', {'request': {'kind': write_kind}}))
            self.assertEqual(registry.effect_of('depth_query', {'request': {'kind': write_kind}}), 'record')

        # 写前整批准入：batch 含被禁用/未启用的深度子项时，前序子项 0 次执行
        rt = SimpleNamespace(
            identity={'run_id': 'r', 'episode_id': 'e', 'env_instance_id': 'i', 'env_generation': 1},
            observation_profile='rgb_proprio_calibrated_v1',
            disabled_tools=frozenset(),
        )
        executed = []
        with patch.object(handlers, '_execute_geometry_leaf', side_effect=lambda *a, **kw: executed.append(a[2])):
            with self.assertRaises(ToolInputError) as caught:
                handlers.geometry_query(rt, {'request': {'kind': 'batch', 'queries': [
                    {'kind': 'reanchor_tracks', 'observation_id': 'obs_1'},
                    {'kind': 'depth_points', 'observation_id': 'obs_1'},
                ]}}, {})
            self.assertEqual(caught.exception.code, 'depth_not_enabled_for_run')
            self.assertEqual(executed, [])

        # 超深嵌套批次在执行前被拒绝
        deep = {'kind': 'fk'}
        for _ in range(10):
            deep = {'kind': 'batch', 'queries': [deep]}
        self.assertFalse(registry.parallel_safe('geometry_query', {'request': deep}))
        with self.assertRaises(ToolInputError) as caught:
            handlers.geometry_query(rt, {'request': deep}, {})
        self.assertEqual(caught.exception.code, 'invalid_geometry_batch_depth')

        # 逐项领域错误写回 invalid_input，非领域内部错误上抛
        from ni_rua.core.tool_output import ToolResponse
        def fake_leaf(context, runtime, sub_req, correlation):
            if sub_req.get('kind') == 'bad_domain':
                raise ToolInputError('invalid_pixel', field_path='/request/pixel')
            if sub_req.get('kind') == 'boom_internal':
                raise RuntimeError('internal_crash')
            return ToolResponse([], data={'query_type': sub_req['kind'], 'status': 'supported', 'value': {'ok': True}})

        with patch.object(handlers, '_execute_geometry_leaf', side_effect=fake_leaf):
            resp = handlers.geometry_query(rt, {'request': {'kind': 'batch', 'queries': [
                {'kind': 'fk'},
                {'kind': 'bad_domain'},
            ]}}, {})
            items = resp.data['value']['batch_results']
            self.assertEqual(items[0]['status'], 'supported')
            self.assertEqual(items[1]['status'], 'invalid_input')
            self.assertEqual(items[1]['reason'], 'invalid_pixel')
            with self.assertRaisesRegex(RuntimeError, 'internal_crash'):
                handlers.geometry_query(rt, {'request': {'kind': 'batch', 'queries': [
                    {'kind': 'fk'},
                    {'kind': 'boom_internal'},
                ]}}, {})

    def test_robot_observe_sets_captured_live_attribute_on_normal_and_fallback_paths(self):
        from unittest.mock import patch
        from ni_rua.runtime.tools import handlers

        tmp = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(tmp)
        store = Store(root / 'run', dict(run_id='live-flag', episode_id='ep', env_instance_id='env', env_generation=1))
        self.addCleanup(store.close)
        harness = Harness(store, FixtureRobot(root))
        harness.start()

        live_resp = handlers.robot_observe(harness, {}, {})
        self.assertIs(getattr(live_resp, 'captured_live', None), True)
        cam_id = harness.latest['cameras'][0]['camera_id']
        cached_resp = handlers.robot_observe(harness, {'camera_id': cam_id}, {})
        self.assertIs(getattr(cached_resp, 'captured_live', None), False)

        # 渲染异常回退到 output_contract_error 时同样保留 captured_live=True
        with patch.object(harness, 'observation_content', side_effect=RuntimeError('render_broken')):
            fallback_resp = handlers.robot_observe(harness, {'image_delivery': 'all'}, {})
        self.assertIs(fallback_resp.success, False)
        self.assertIs(getattr(fallback_resp, 'captured_live', None), True)

    def test_output_contracts_pose_plan_effect_and_help_validations(self):
        from ni_rua.core.tool_errors import ToolInputError
        from ni_rua.cortex.context.operation_memory import OperationMemory
        from ni_rua.perception.external import pose_skill
        from ni_rua.robot.model.geometry import GeometryTools
        from ni_rua.runtime.tools.output_contracts import CONTRACTS
        from ni_rua.runtime.tools.registry import validator_for

        # output_contracts: 删死契约 tool_guide、增 pose_plan、RECEIPT/PLAN 零新增必填并含新可选字段
        self.assertNotIn('tool_guide', CONTRACTS)
        self.assertIn('pose_plan', CONTRACTS)
        receipt_schema = CONTRACTS['robot_action']['action']['properties']['receipt']
        self.assertEqual(receipt_schema['required'], ['status'])
        for field in ('action_id', 'stop_confirmed', 'execution_uncertain', 'failure_feedback',
                      'contact_block_summary', 'first_divergence', 'grasp_contact_feedback',
                      'execution_interval', 'episode_terminal', 'failed_phase'):
            self.assertIn(field, receipt_schema['properties'])
        plan_schema = CONTRACTS['robot_action']['plan']
        self.assertEqual(plan_schema['required'], ['status'])
        for field in ('effective_tolerances', 'downgrade_reasons', 'downgraded_from_execute_if_clean',
                      'continuation_of', 'grasp_3d_review', 'inspection', 're_executed_stage_ids'):
            self.assertIn(field, plan_schema['properties'])
        # ACTION_INSPECTION 兼容类型化采样（不含 available/action_id）、无反馈回退与 null
        data_validator = validator_for(CONTRACTS['robot_action']['data'])
        self.assertTrue(data_validator.is_valid(None))
        self.assertTrue(data_validator.is_valid({'action_id': 'a1', 'available': False, 'trajectory_feedback': {}}))
        self.assertTrue(data_validator.is_valid({
            'trajectory_ref': 'artifact:t1', 'source_kind': 'candidate_command_trajectory_not_measurement',
            'samples': [], 'total_samples': 0, 'selection_mode': 'key_samples', 'detail': 'summary',
            'include_geometry': False, 'current': True, 'limitations': 'none',
        }))

        # pose_plan 副作用与并发分类：grasp 写槽串行，targets/place 纯读可并发且 place 不写槽
        registry = full_registry()
        self.assertEqual(registry.effect_of('pose_plan', {'request': {'kind': 'grasp'}}), 'record')
        self.assertFalse(registry.parallel_safe('pose_plan', {'request': {'kind': 'grasp'}}))
        self.assertEqual(registry.effect_of('pose_plan', {'request': {'kind': 'targets'}}), 'read')
        self.assertTrue(registry.parallel_safe('pose_plan', {'request': {'kind': 'targets'}}))
        self.assertEqual(registry.effect_of('pose_plan', {'request': {'kind': 'place'}}), 'read')
        self.assertTrue(registry.parallel_safe('pose_plan', {'request': {'kind': 'place'}}))
        bare_rt = SimpleNamespace()
        self.assertEqual(pose_skill._grasp_memory(bare_rt, create=False), {})
        self.assertFalse(hasattr(bare_rt, '_pose_skill_grasps'))

        # geometry.py help 补齐 batch
        help_res = GeometryTools(None).execute(dict(
            run_id='r', episode_id='e', env_instance_id='i', env_generation=1, schema_version='0.1', kind='help',
        ))
        self.assertIn('batch', help_res['requests'])

        # operation_memory help 多余字段报精确定位错误
        tmp = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(tmp) / 'mem', dict(run_id='m', episode_id='e', env_instance_id='i', env_generation=1))
        self.addCleanup(store.close)
        mem = OperationMemory(store)
        self.assertIn('scopes', mem.call({'kind': 'help'}))
        with self.assertRaises(ToolInputError) as caught:
            mem.call({'kind': 'help', 'scope': 'task'})
        self.assertEqual(caught.exception.code, 'invalid_operation_memory_fields')
        self.assertEqual(caught.exception.field_path, '/request')
        self.assertEqual(caught.exception.actual['unexpected'], ['scope'])


if __name__ == "__main__":
    unittest.main()
