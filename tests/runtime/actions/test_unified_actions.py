"""
- test_dual_arm_interface.py: Shared dual-arm Harness fixture and unified robot_action schema boundary checks.
- test_unified_actions.py: Public unified action conversion, cache lookup and harness execution boundary.
"""
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from support.tool_fixture import install_registry
from ni_rua.core.config import Config
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.runtime.harness import Harness
from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
from ni_rua.core.storage.store import Store
from support.fixture_robot import FixtureRobot
from copy import deepcopy
import json
from unittest.mock import Mock, patch
from ni_rua.runtime.actions.unified_actions import compile_request, dispatch as typed_dispatch
from ni_rua.core.tool_output import wire_response
from ni_rua.core.presentation.actor_payload import lean_v2_display
from ni_rua.robot.motion.batch_contract import validate_batch_action
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.core.tool_errors import ToolInputError
from ni_rua.runtime.tools.registry import validator_for
from support.motion_fixtures import DualArmHarnessBase

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_dual_arm_interface.py ---

IDENTITY = dict(run_id='dual-interface', episode_id='ep', env_instance_id='env', env_generation=1)


def capabilities():
    return dict(frame_id=FRAME_ID, tcp_config_id=TCP_CONFIG_ID, path_kinds=['backend_free', 'hold'],
        max_waypoints=8, dual_arm=True, gripper_after_motion=True,
        gripper_phases=['after_motion', 'during_motion'],
        action_speed=dict(available=True, waypoint_speed_scales=dict(available=True)))


class DualArmHarnessTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.store = Store(Path(temporary.name) / 'store', dict(IDENTITY))
        self.addCleanup(self.store.close)
        self.robot = FixtureRobot(temporary.name)
        observe = self.robot.observe
        self.robot.observe = lambda: dict(observe(), camera_calibration=dict(reference_frame_id='env_origin', cameras={}),
            camera_pose_sample=dict(cameras={}), robot_model=dict(base_link='base', joints=[]))
        caps = {**self.robot.start(), **capabilities()}
        self.robot.start = lambda: caps
        self.h = Harness(self.store, self.robot, config=Config.from_dict(dict(features=dict(
            motion_review=False, default_motion_speed_scale=1.25))))
        self.h.start()

    def test_unified_catalog_preserves_both_arm_schema_across_backend_modes(self):
        workbench = Workbench.__new__(Workbench)
        workbench.pi05_enabled = False
        workbench.visual_review_model = None
        for mode in ('resampled', 'feedback'):
            for supported in (None, True, False):
                with self.subTest(mode=mode, supported=supported):
                    workbench.config = Config.from_dict(dict(features=dict(
                        motion_review=False, motion_execution_mode=mode)))
                    workbench.harness = SimpleNamespace(capabilities=(None if supported is None else
                        {**capabilities(), 'dual_arm': supported}))
                    install_registry(workbench.harness, workbench.config)
                    tools = {tool['name']: tool for tool in workbench.actor_tools()}
                    self.assertEqual('robot_action' in tools, mode == 'resampled')
                    self.assertIn('operation_memory', tools)
                    self.assertNotIn('robot_move_dual_grasp', tools)
                    if mode == 'feedback':
                        continue
                    schema = tools['robot_action']['inputSchema']['properties']
                    self.assertEqual(schema['stages']['items']['properties']['arm']['enum'],
                                     ['left', 'right'])
                    self.assertIn('depends_on', schema['stages']['items']['properties'])



# --- Folded from tests/test_unified_actions.py ---

def dispatch(*args, **kwargs):
    """This legacy fixture inspects wire output at an explicit adapter."""
    return wire_response(typed_dispatch(*args, **kwargs))


class UnifiedActionTests(unittest.TestCase):
    setUp = DualArmHarnessBase.setUp

    def request(self):
        return dict(observation_id=self.h.latest['observation_id'], stages=[dict(id='reach', kind='move', arm='left',
            grasp_position_m=[.2,-.1,.8], angles_deg=[0,0,0])])

    def preview(self, args=None, call='preview'):
        def result(action):
            full=dict(schema_version='candidate-trajectory-v1', source_kind='candidate_command_trajectory_not_measurement',
                frame_id='env_origin',tcp_config_id='arx-x5-link6-v1', joint_names={'left':[], 'right':[]},
                samples=[dict(native_step=0,arms={arm:dict(q_rad=[0.]*6,jaw_goal_normalized=1.,phase='initial')
                    for arm in ('left','right')})], geometry=dict(status='unknown'),inter_arm_proximity=dict(status='unknown'),
                execution_binding='exact_cached_rows; no replanning or resampling')
            return dict(status='planned',plan_id='native-plan',planned_native_steps=2,trajectory_data=full,
                trajectory_evidence=dict(available=True),stage_costs={})
        self.robot.preview_batch=Mock(side_effect=result)
        with patch('ni_rua.runtime.actions.inspection._trajectory_image',return_value=(dict(status='unavailable'),[])):
            output=dispatch(self.h,args or self.request(),{'tool_call_id':call})
        return json.loads(output['contentItems'][0]['text'])

    def failed_preview_evidence(self, turn, failure, *, remaining=42, stale=False):
        from ni_rua.cortex.supervision.recovery_evidence import completed_turn_evidence

        args = self.request()
        correlation = dict(thread_id='thread', turn_id=turn, tool_call_id='call-' + turn)
        self.store.save('tool_call', correlation['tool_call_id'], dict(correlation=correlation,
            params=dict(tool='robot_action', arguments=args)))
        budget = dict(native_steps_used=5, native_steps_limit=47,
            native_steps_remaining=remaining, source='native_execution_accounting',
            unknown_values_are_null=True)
        self.h.action_budget = Mock(side_effect=[budget, budget])
        observation = self.h.current_observation()
        if stale:
            stale_reference = dict(observation, robot_state_ref='artifact:stale-state-reference')
            self.h.current_observation = Mock(return_value=stale_reference)
        self.robot.preview_batch = Mock(return_value=failure)
        response = typed_dispatch(self.h, args, correlation)
        self.assertEqual(response.kind, 'plan')
        events = self.store.turn_preview_events(thread_id='thread', turn_id=turn, episode_id='ep')
        self.assertEqual(len(events), 1)
        evidence = completed_turn_evidence(self.store, thread_id='thread', turn_id=turn,
            episode_id='ep', native_action_count=5, native_steps_remaining=remaining)
        return events[0]['payload'], evidence

    def test_failed_batch_budget_preview_is_recorded_as_a_candidate(self):
        safety = dict(enabled=True, status='reject', arms={'left': dict(status='reject',
            reasons=['insufficient_remaining_native_budget'], nominal_native_steps=47,
            episode_remaining_native_steps=42, exceeded_budgets={'episode': 42})})
        payload, evidence = self.failed_preview_evidence('budget', dict(status='failed',
            reason='motion_safety_rejected', motion_safety_checks=[safety], physics_advanced=False,
            executed_native_steps=0))
        self.assertEqual(payload['planned_native_steps'], 47)
        self.assertTrue(payload['zero_step_check_passed'])
        self.assertNotIn('same_measured_state_verified', payload)
        self.assertEqual(payload['state_binding']['robot_state_ref'],
                         self.store.read('observation', payload['observation_id'])['robot_state_ref'])
        self.assertEqual(len(evidence['candidates']), 1)
        candidate = evidence['candidates'][0]
        self.assertEqual(candidate['nominal_native_steps'], 47)
        self.assertFalse(candidate['feasible'])
        self.assertTrue(candidate['exceeds_episode_budget'])
        self.assertEqual(evidence['budget_rejected_count'], 1)

    def test_program_failed_batch_preview_reaches_recovery_ledger_once(self):
        self.program_failed_preview_evidence('preview')

    def test_program_review_generated_failed_preview_reaches_recovery_ledger_once(self):
        self.program_failed_preview_evidence('direct')

    def program_failed_preview_evidence(self, mode):
        from ni_rua.cortex.supervision.recovery_evidence import completed_turn_evidence

        args = dict(self.request(), mode=mode)
        correlation = dict(thread_id='thread', turn_id='program-preview', tool_call_id='program-call')
        self.h.action_budget = Mock(return_value=dict(native_steps_used=1082,
            native_steps_limit=1100, native_steps_remaining=18))
        safety = dict(enabled=True, status='reject', arms={'left': dict(status='reject',
            reasons=['insufficient_remaining_native_budget'], nominal_native_steps=71,
            episode_remaining_native_steps=18, exceeded_budgets={'episode': 18})})
        self.robot.preview_batch = Mock(return_value=dict(status='failed',
            reason='motion_safety_rejected', motion_safety_checks=[safety],
            physics_advanced=False, executed_native_steps=0))
        self.robot.execute = Mock()
        source = ('from ni_rua_client import NiRuaClient\nclient = NiRuaClient()\n'
                  'for _ in range(2):\n'
                  f'    result = client.call("robot_action", {args!r})\n')
        if mode == 'preview':
            source += ('    assert result["kind"] == "plan"\n'
                       '    assert result["data"]["status"] == "failed"\n')
        else:
            source += ('    assert result["kind"] == "error"\n'
                       '    assert result["success"] is False\n')
        self.store.save('tool_call', correlation['tool_call_id'], dict(correlation=correlation,
            params=dict(tool='program_run', arguments=dict(source=source))))
        runner = ProgramRunner(self.h.program_services())
        self.addCleanup(runner.close)
        job = runner.run(dict(source=source), correlation)
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        self.assertEqual(self.robot.preview_batch.call_count, 2)
        self.robot.execute.assert_not_called()
        self.assertFalse(job.get('pending_review'))
        self.assertEqual(self.store.records('action_request'), [])
        evidence = completed_turn_evidence(self.store, thread_id='thread', turn_id='program-preview',
            episode_id='ep', native_action_count=1082, native_steps_remaining=18)
        self.assertEqual(len(evidence['candidates']), 1)
        self.assertEqual(evidence['candidates'][0]['nominal_native_steps'], 71)
        self.assertTrue(evidence['candidates'][0]['exceeds_episode_budget'])
        self.assertEqual(evidence['budget_rejected_count'], 1)

    def test_failed_batch_ik_and_unknown_costs_are_candidates_without_budget_rejection(self):
        for turn, failure, expected_feasible in (
                ('ik-failure', dict(status='failed', reason='path_infeasible',
                    physics_advanced=False, executed_native_steps=0), False),
                ('unknown-failure', dict(status='failed', reason='preview_unavailable',
                    physics_advanced=False, executed_native_steps=0), None)):
            with self.subTest(turn=turn):
                _, evidence = self.failed_preview_evidence(turn, failure)
                self.assertEqual(len(evidence['candidates']), 1)
                self.assertIs(evidence['candidates'][0]['feasible'], expected_feasible)
                self.assertEqual(evidence['budget_rejected_count'], 0)

    def test_failed_batch_preview_with_nonzero_or_stale_state_keeps_request_credit(self):
        failure = dict(status='failed', reason='motion_safety_rejected', physics_advanced=False,
            executed_native_steps=0)
        _, nonzero = self.failed_preview_evidence('nonzero', dict(failure,
            physics_advanced=True, executed_native_steps=1))
        _, stale = self.failed_preview_evidence('stale', failure, stale=True)
        for evidence in (nonzero, stale):
            self.assertEqual(evidence['candidates'], [])
            self.assertEqual(evidence['budget_rejected_count'], 0)
            self.assertTrue(any(value.startswith('robot_action:') for value in evidence['activity']))

    def native_execution(self):
        self.robot.execute=Mock(return_value=dict(status='completed',reason=None,executed_native_steps=2,
            completed_native_steps=2,batch_status='completed',batch_cursor=2,completed_stage_ids=['reach'],
            pending_stage_ids=[],stage_results={},terminal_observation=self.robot.observe()))

    def test_stage_schema_rejects_static_mismatches_for_both_roles_and_compiler(self):
        grasp = self.request()['stages'][0]
        link = dict(id='reach', kind='move', arm='left', position_m=[.2,-.1,.8], quaternion_xyzw=[0,0,0,1])
        gripper = dict(id='jaw', kind='gripper', arm='left', goal_normalized=0.)
        observe = dict(id='check', kind='observe', reason='Check retention')
        invalid = [dict(grasp, reason='Move above the object'), dict(grasp, goal_normalized=0.),
            {key: value for key, value in grasp.items() if key != 'arm'},
            dict(id='empty', kind='move', arm='left'),
            {key: value for key, value in grasp.items() if key != 'grasp_position_m'},
            {key: value for key, value in grasp.items() if key != 'angles_deg'},
            {key: value for key, value in link.items() if key != 'position_m'},
            {key: value for key, value in link.items() if key != 'quaternion_xyzw'},
            dict(grasp, position_m=link['position_m']), dict(link, angles_deg=[0,0,0]),
            dict(link, contact_region='pad_center'), dict(link, contact_depth_m=0.),
            dict(grasp, contact_region='pad_center', contact_depth_m=0.),
            dict(grasp, speed_scale=1., speed_preset='normal'),
            dict(grasp, gripper_motion=dict(start_fraction=.2,end_fraction=.8)),
            dict(grasp, gripper_speed_scale=1.), dict(grasp, gripper_speed_preset='normal'),
            dict(grasp, gripper_goal_normalized=0., gripper_speed_scale=1., gripper_speed_preset='normal'),
            dict(gripper, gripper_speed_scale=1., gripper_speed_preset='normal'),
            dict(id='jaw', kind='gripper', arm='left'), dict(gripper, reason='Close jaws'),
            dict(gripper, position_m=link['position_m']), dict(gripper, speed_scale=1.),
            dict(observe, position_m=link['position_m']), dict(observe, goal_normalized=0.),
            dict(id='check', kind='observe'), dict(observe, reason=' \t\n'),
            dict(observe, arm='middle')]
        for field, values in (
                ('speed_scale', [.249, 2.001, True, float('inf')]),
                ('contact_depth_m', [-.0076, .0626]),
                ('position_tolerance_m', [0., .101]),
                ('orientation_tolerance_rad', [0., 3.15]),
                ('max_tracking_divergence_m', [.004, .151]),
                ('gripper_goal_normalized', [-.01, 1.01])):
            invalid.extend(dict(grasp, **{field: value}) for value in values)
        invalid.extend(dict(grasp, gripper_goal_normalized=0., gripper_speed_scale=value)
            for value in (.249, 1.001, False))
        invalid.extend(dict(grasp, angles_deg=value) for value in ([181,0,0], [0,91,0], [0,0,-181]))
        invalid.extend(dict(grasp, gripper_goal_normalized=0., gripper_motion=value) for value in (
            dict(start_fraction=-.1,end_fraction=.8), dict(start_fraction=1.,end_fraction=1.),
            dict(start_fraction=0.,end_fraction=0.), dict(start_fraction=0.,end_fraction=1.01)))
        for index, stage in enumerate(invalid):
            args = dict(observation_id=self.h.latest['observation_id'], stages=[stage])
            original = deepcopy(args)
            with self.subTest(index=index, stage=stage):
                for role in ('actor', 'program'):
                    with self.assertRaises(ToolInputError):
                        self.h.tool_registry.validate('robot_action', args, role=role)
                with patch('ni_rua.runtime.actions.unified_actions._next') as allocate:
                    with self.assertRaises(ValueError):
                        compile_request(self.h, args)
                    allocate.assert_not_called()
                self.assertEqual(args, original)

    def test_valid_stage_forms_share_schema_and_compile_without_mutation(self):
        grasp = self.request()['stages'][0]
        stages = [grasp, dict(grasp, contact_region='finger_middle', speed_scale=.25),
            dict(grasp, contact_depth_m=-.0075, speed_scale=2., angles_deg=[-180,-90,180]),
            dict(grasp, contact_depth_m=.0625, position_tolerance_m=.05),
            dict(grasp, tolerance_preset='coarse', stall_guard_preset='contact_stop'),
            dict(grasp, tolerance_preset='tight', orientation_tolerance_rad=.2, max_tracking_divergence_m=.03),
            dict(id='link', kind='move', arm='right', position_m=[.2,.1,.8], quaternion_xyzw=[0,0,0,1]),
            dict(id='jaw', kind='gripper', arm='left', goal_normalized=0., gripper_speed_scale=.25),
            dict(id='jaw', kind='gripper', arm='right', goal_normalized=1., gripper_speed_scale=1.),
            dict(id='check', kind='observe', reason='检查支撑'),
            dict(id='check', kind='observe', arm='right', reason='Check retention'),
            dict(grasp, gripper_goal_normalized=0., gripper_speed_preset='slow'),
            dict(grasp, gripper_goal_normalized=1., gripper_motion=dict(start_fraction=0.,end_fraction=1.))]
        actor = self.h.tool_registry.spec('robot_action', role='actor')
        program = self.h.tool_registry.spec('robot_action', role='program')
        self.assertEqual(actor, program)
        self.assertIsNot(actor.definition, program.definition)
        schema = validator_for(actor.definition['inputSchema'])
        for stage in stages:
            with self.subTest(stage=stage):
                args = dict(observation_id=self.h.latest['observation_id'], stages=[stage])
                original = deepcopy(args)
                self.assertTrue(schema.is_valid(args))
                for role in ('actor', 'program'):
                    self.h.tool_registry.validate('robot_action', args, role=role)
                action = compile_request(self.h, args)
                validate_batch_action(action, self.h.capabilities)
                self.assertEqual(args, original)

    def test_direct_compiler_preserves_existing_error_codes(self):
        grasp = self.request()['stages'][0]
        cases = [(dict(grasp, reason='move'), 'unsupported_robot_action_stage_fields'),
            ({key: value for key, value in grasp.items() if key != 'arm'}, 'robot_action_stage_requires_arm'),
            (dict(id='check',kind='observe',reason='check',speed_scale=1.), 'observe_requires_only_reason'),
            (dict(id='check',kind='observe',reason=' \n'), 'batch_observe_requires_reason'),
            (dict(grasp,position_m=[0,0,0]), 'batch_move_requires_one_pose_convention'),
            (dict(grasp,position_tolerance_m=0), 'invalid_position_tolerance_m'),
            (dict(grasp,contact_region='pad_center',contact_depth_m=0), 'contact_region_and_contact_depth_m_are_mutually_exclusive'),
            (dict(id='link',kind='move',arm='left',position_m=[0,0,1],quaternion_xyzw=[0,0,0,1],contact_depth_m=0), 'contact_selection_requires_grasp_pose'),
            (dict(grasp,speed_scale=1.,speed_preset='normal'), 'conflicting_speed_preset_and_speed_scale'),
            (dict(grasp,gripper_goal_normalized=0.,gripper_speed_scale=1.,gripper_speed_preset='normal'), 'conflicting_gripper_speed_preset_and_gripper_speed_scale'),
            (dict(grasp,gripper_goal_normalized=0.,gripper_speed_scale=1.01), 'invalid_gripper_speed_scale'),
            (dict(grasp,gripper_motion=dict(start_fraction=.2,end_fraction=.8)), 'invalid_gripper_motion')]
        for stage, code in cases:
            with self.subTest(code=code), self.assertRaisesRegex(ValueError, '^'+code):
                compile_request(self.h, dict(observation_id=self.h.latest['observation_id'], stages=[stage]))

    def test_move_reason_rejects_at_real_actor_and_program_ports_before_side_effects(self):
        args = self.request(); args['stages'][0]['reason'] = 'Move above the visible object'
        original = deepcopy(args)
        self.h.thread_id = 'thread'
        self.robot.preview_batch = Mock()
        self.robot.execute = Mock()
        runner = ProgramRunner(self.h.program_services())
        self.addCleanup(runner.close)
        with patch('ni_rua.runtime.actions.unified_actions.dispatch', wraps=dispatch) as handler, \
                patch('ni_rua.runtime.actions.unified_actions._next') as allocate:
            for mode in ('preview', 'direct'):
                with self.subTest(mode=mode), self.assertRaises(ToolInputError) as error:
                    self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='bad-'+mode,
                        tool='robot_action', arguments=dict(args, mode=mode)))
                self.assertEqual(error.exception.field_path, '/stages/0/reason')
            source = 'from ni_rua_client import NiRuaClient\nclient = NiRuaClient()\n'
            source += '\n'.join("client.call('robot_action', " + repr(dict(args, mode=mode)) + ')'
                for mode in ('preview', 'direct'))
            result = runner.run(dict(source=source),
                dict(thread_id='thread',turn_id='turn',tool_call_id='invalid-stage-program'))
            self.assertEqual(result['status'], 'succeeded', result)
            self.assertEqual(result['call_summary']['failed_calls'], 2)
            handler.assert_not_called()
            allocate.assert_not_called()
        self.robot.preview_batch.assert_not_called()
        self.robot.execute.assert_not_called()
        self.assertEqual(self.robot.executions, 0)
        self.assertEqual(self.store.records('action_request'), [])
        self.assertEqual(self.store.records('unified_action_plan'), [])
        self.assertFalse(getattr(self.h, 'unified_action_plans', {}))
        self.assertEqual(args, original)

    def test_conversion_keeps_grasp_reference_speed_and_input_unchanged(self):
        args=self.request(); args['stages'][0].update(contact_region='finger_middle',speed_preset='slow')
        original=deepcopy(args); action=compile_request(self.h,args)
        own=action['stages'][0]['action']
        self.assertAlmostEqual(own['path']['waypoints'][0]['left']['position_m'][2],.9226)
        self.assertEqual(own['speed_scale'],.8)
        self.assertEqual(args,original)

    def test_ee_target_remains_link6_and_mixed_conventions_reject(self):
        args=self.request(); stage=args['stages'][0]
        stage.pop('grasp_position_m'); stage.pop('angles_deg')
        stage.update(position_m=[.2,.1,.8],quaternion_xyzw=[0,0,0,1])
        action=compile_request(self.h,args)
        self.assertEqual(action['stages'][0]['action']['path']['waypoints'][0]['left']['position_m'],[.2,.1,.8])
        stage.update(grasp_position_m=[.2,.1,.8],angles_deg=[0,0,0])
        with self.assertRaisesRegex(ValueError,'one_pose_convention'): compile_request(self.h,args)

    def test_after_motion_jaw_expands_with_original_completion_dependency(self):
        args=self.request(); args['stages'][0].update(gripper_goal_normalized=0.,gripper_speed_preset='slow')
        args['stages'].append(dict(id='peer',arm='right',kind='gripper',goal_normalized=1.,depends_on=['reach']))
        original=deepcopy(args)
        action=compile_request(self.h,args); dependencies=validate_batch_action(action,self.h.capabilities)
        self.assertEqual([s['id'] for s in action['stages']],['reach/motion','reach','peer'])
        self.assertEqual(action['stages'][1]['action']['gripper_speed_scale'],.8)
        self.assertEqual(dependencies['peer'],{'reach'})
        self.assertEqual(action['stages'][0]['action']['gripper_events'], [])
        self.assertEqual(action['stages'][1]['action']['gripper_events'][0]['phase'], 'after_motion')
        self.assertEqual(args,original)

    def test_move_synchronized_jaw_stays_in_the_same_stage(self):
        args=self.request(); args['stages'][0].update(gripper_goal_normalized=0.,
            gripper_motion=dict(start_fraction=.2,end_fraction=.9))
        action=compile_request(self.h,args)
        self.assertEqual(len(action['stages']),1)
        self.assertEqual(action['stages'][0]['action']['gripper_events'][0]['phase'],'during_motion')

    def test_default_preview_has_short_id_evidence_and_zero_execution(self):
        plan=self.preview()
        self.assertEqual(plan['status'],'planned')
        self.assertEqual(plan['plan_id'],'p1')
        self.assertIn('trajectory_ref',plan)
        self.assertEqual(self.robot.executions,0)
        self.assertEqual(self.robot.preview_batch.call_count,1)

    def test_execute_cached_plan_and_idempotent_action_id_refresh_observation(self):
        plan=self.preview(); self.native_execution()
        self.h.observe()
        args=dict(mode='execute',plan_id=plan['plan_id'],action_id='same')
        first=dispatch(self.h,args,{'tool_call_id':'execute'})
        summary=json.loads(first['contentItems'][0]['text'])
        self.assertEqual(summary['receipt']['status'],'completed')
        self.assertEqual(self.robot.execute.call_args.args[0]['batch_plan_id'],'native-plan')
        dispatch(self.h,args,{'tool_call_id':'retry'})
        self.assertEqual(self.robot.execute.call_count,1)

    def test_automatic_jaw_checkpoint_delivers_formal_observation_to_actor_and_program(self):
        from support.program_fixture import program_services
        from support.motion_fixtures import jaws
        from support.motion_fixtures import periodic_pose
        from support.motion_fixtures import preview_adapter
        for role in ('actor', 'program'):
            with self.subTest(role=role):
                self.setUp()
                self.h.thread_id = 'thread'
                adapter, _, dispatched = preview_adapter(); jaws(adapter)
                self.robot.preview_batch = Mock(side_effect=adapter.preview_batch)
                def execute(action, **kwargs):
                    receipt = adapter.execute(action, lambda _: None, kwargs['cancel_flag'])
                    receipt.pop('before_observation', None)
                    if receipt.get('terminal_observation') is not None:
                        receipt['terminal_observation'] = self.robot.observe()
                    return receipt
                self.robot.execute = Mock(side_effect=execute)
                pose = periodic_pose([-.125, 0., 0., 0., 0., 0.])
                for invalid in (
                        [dict(id='check', kind='observe', reason='r', checkpoint_after_gripper=True)],
                        [dict(id='move', kind='move', arm='left', checkpoint_after_gripper=True, **pose)]):
                    with self.assertRaises(ValueError):
                        self.h.invoke('robot_action', dict(observation_id=self.h.latest['observation_id'], stages=invalid),
                            dict(tool_call_id='invalid'), role=role)
                args = dict(observation_id=self.h.latest['observation_id'], stages=[
                    dict(id='close', kind='gripper', arm='left', goal_normalized=0., checkpoint_after_gripper=True),
                    dict(id='carry', kind='move', arm='left', **pose)])
                with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
                    plan = self.h.invoke('robot_action', args, dict(tool_call_id='preview'), role=role).data
                self.assertEqual(plan['status'], 'planned')
                before = self.h.latest['observation_id']
                execute_args = dict(mode='execute', plan_id=plan['plan_id'])
                if role == 'actor':
                    reply = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='execute',
                        tool='robot_action', arguments=execute_args))
                    self.assertTrue(any(item['type'] == 'inputImage' for item in reply['contentItems']))
                else:
                    runner = self.h.programs = ProgramRunner(program_services(self.h))
                    self.addCleanup(runner.close)
                    source = ("from ni_rua_client import NiRuaClient\n"
                        "result = NiRuaClient().call('robot_action', " + repr(execute_args) + ")\n"
                        "assert result['kind'] == 'action'\n"
                        "assert result['data']['receipt']['reason'] == 'observation_checkpoint'\n")
                    job = runner.run(dict(source=source, allow_direct_actions=True),
                        dict(thread_id='thread', turn_id='turn', tool_call_id='program'))
                    self.assertEqual(job['status'], 'succeeded', job)
                    self.assertEqual(job['action_summary']['status_counts'], {'stopped': 1})
                    self.assertEqual(job['final_observation_id'], self.h.latest['observation_id'])
                receipt = self.store.records('action_receipt')[-1]
                self.assertEqual((receipt['status'], receipt['reason']), ('stopped', 'observation_checkpoint'))
                self.assertEqual(receipt['stage_results']['carry']['completed_steps'], 0)
                self.assertNotEqual(self.h.latest['observation_id'], before)
                self.assertEqual(receipt['result_observation_id'], self.h.latest['observation_id'])
                resumed = self.h.invoke('robot_action', execute_args, dict(tool_call_id='resume'), role='actor')
                self.assertEqual(resumed.data['receipt']['status'], 'completed')
                self.assertEqual(len(dispatched), plan['planned_native_steps'])

    def test_action_reply_is_laid_out_on_every_host_and_fitted_only_to_a_known_host_limit(self):
        from support.motion_fixtures import jaws, periodic_pose, preview_adapter
        from support.reply_defaults import with_reply_features
        from ni_rua.core.presentation.reply_budget import fit_reply, reply_cost_tokens
        replies = {}
        # off: features.budgeted_action_reply false; ordered: on, the host has no known limit; fitted: on, 400 bytes
        for name, (layout, budget) in dict(off=(False, None), ordered=(True, None), fitted=(True, 400)).items():
            self.setUp()
            self.h.thread_id = 'thread'
            self.assertIsNone(self.h.action_reply_budget)
            self.h.config = with_reply_features(self.h.config, budgeted_action_reply=layout)
            self.h.action_reply_budget = lambda: budget
            adapter, _, _ = preview_adapter(); jaws(adapter)
            self.robot.preview_batch = Mock(side_effect=adapter.preview_batch)
            def execute(action, **kwargs):
                receipt = adapter.execute(action, lambda _: None, kwargs['cancel_flag'])
                receipt.pop('before_observation', None)
                receipt['terminal_observation'] = self.robot.observe()
                return receipt
            self.robot.execute = Mock(side_effect=execute)
            args = dict(observation_id=self.h.latest['observation_id'], stages=[
                dict(id='close', kind='gripper', arm='left', goal_normalized=0.),
                dict(id='carry', kind='move', arm='left', **periodic_pose([-.125, 0., 0., 0., 0., 0.]))])
            with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
                preview = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='preview',
                    tool='robot_action', arguments=args))
                plan = json.loads(preview['contentItems'][0]['text'])
            replies[name] = (preview, self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='execute',
                tool='robot_action', arguments=dict(mode='execute', plan_id=plan['plan_id']))))
            events = [event['payload'] for event in self.store.events_by_type(('actor_reply.fitted',))]
            if budget is None:
                self.assertEqual(events, [])
        (preview_off, off), (_, ordered), (preview_on, on) = replies['off'], replies['ordered'], replies['fitted']
        # A host without a known limit: the same reply laid out by decision value, nothing replaced.
        self.assertEqual(ordered['contentItems'], fit_reply(off['contentItems'], None, layout=True))
        self.assertNotIn('by_reference', ''.join(item.get('text', '') for item in ordered['contentItems']))
        self.assertEqual(set(json.loads(ordered['contentItems'][0]['text'])) & {'stages', 'trajectory_feedback'}, set())
        # A preview is not an action reply; the budget does not touch it.
        self.assertEqual([item['type'] for item in preview_on['contentItems']],
                         [item['type'] for item in preview_off['contentItems']])
        self.assertEqual(list(json.loads(preview_on['contentItems'][0]['text'])),
                         list(json.loads(preview_off['contentItems'][0]['text'])))
        original = json.loads(off['contentItems'][0]['text'])
        self.assertEqual(set(original['stages']), {'close', 'carry'})
        self.assertIn('trajectory_feedback', original)
        # Same run with the budget: exactly what the pure function makes of the unfitted reply.
        self.assertEqual([item['type'] for item in on['contentItems']],
                         [item['type'] for item in fit_reply(off['contentItems'], 400)])
        head, observation, stages = (json.loads(item['text']) for item in on['contentItems'][:4]
                                     if item['type'] == 'inputText')
        self.assertEqual(head['receipt']['status'], 'completed')
        self.assertNotIn('stages', head)
        self.assertIn('robot_state', observation)
        self.assertEqual(events, [dict(applied=True, budget_tokens=100, by_reference=['stages'], fits=False,
            cost_tokens_before=events[0]['cost_tokens_before'], cost_tokens_after=reply_cost_tokens(on['contentItems']))])
        note = stages['stages_by_reference']
        self.assertEqual(sorted(note['stage_ids']), ['carry', 'close'])
        read = dict(note['read'])
        full = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='status',
            tool=read.pop('tool'), arguments=read))
        stored = json.loads(full['contentItems'][0]['text'])['receipt']['stage_results']
        self.assertEqual(set(stored), {'close', 'carry'})
        self.assertEqual(stored['carry']['completed_steps'], original['stages']['carry']['completed_steps'])
        self.assertEqual(self.robot.execute.call_count, 1)

    def test_static_once_drops_fixed_notes_of_real_replies_states_them_in_the_catalogue_and_its_reads_resolve(self):
        import dataclasses
        from support.motion_fixtures import jaws, periodic_pose, preview_adapter
        from ni_rua.core.contracts import canonical
        from ni_rua.core.presentation.reply_compaction import drop_static_once_report, unify_detail_entries
        from support.reply_defaults import with_reply_features
        replies, catalogues = {}, {}
        for switch in (False, True):
            self.setUp()
            self.h.thread_id = 'thread'
            self.h.config = with_reply_features(self.h.config, reply_static_once=switch)
            catalogues[switch] = self.h.tool_registry.catalog(self.h, role='actor')
            adapter, _, _ = preview_adapter(); jaws(adapter)
            self.robot.preview_batch = Mock(side_effect=adapter.preview_batch)
            def execute(action, **kwargs):
                receipt = adapter.execute(action, lambda _: None, kwargs['cancel_flag'])
                receipt.pop('before_observation', None)
                receipt['terminal_observation'] = self.robot.observe()
                return receipt
            self.robot.execute = Mock(side_effect=execute)
            args = dict(observation_id=self.h.latest['observation_id'], stages=[
                dict(id='close', kind='gripper', arm='left', goal_normalized=0.),
                dict(id='carry', kind='move', arm='left', **periodic_pose([-.125, 0., 0., 0., 0., 0.]))])
            with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
                preview = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='preview',
                    tool='robot_action', arguments=args))
                plan = json.loads(preview['contentItems'][0]['text'])
            replies[switch] = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='execute',
                tool='robot_action', arguments=dict(mode='execute', plan_id=plan['plan_id'])))
        off, on = replies[False]['contentItems'], replies[True]['contentItems']
        loaded = self.h.tool_registry.loaded_tool_names(self.h, role='actor')
        self.assertIn('robot_action', loaded)
        self.assertIn('robot_action_status', loaded)
        # This fixture has no stored trajectory (feedback unavailable), which is an exception: its notes all stay and
        # the delivered reply is the reply of the same run without the switch with only its read entries rewritten.
        plan = self.h.tool_registry.static_once_plan(self.h, role='actor')
        self.assertLessEqual({'budget', 'steps', 'trajectory', 'grasp_contact', 'joint_travel'}, set(plan.segments))
        kept, report = drop_static_once_report(off, plan=plan)
        self.assertIs(kept, off)
        self.assertEqual(report, dict(exception=True, dropped={}))
        self.assertEqual(on, unify_detail_entries(off, loaded))
        # A host that publishes the whole catalogue eagerly (the MCP hosts) loads every tool.
        self.h.deferred_tool_loading = False
        eager = self.h.tool_registry.loaded_tool_names(self.h, role='actor')
        self.assertEqual(eager, {tool['name'] for tool in catalogues[True]})
        self.h.deferred_tool_loading = True
        # A routine reply of the same producers (the preview, and the execution result without its unavailable
        # feedback) loses exactly the listed notes: the list matches what the real producers write.
        result = json.loads(off[0]['text'])
        result.pop('trajectory_feedback')
        routine = [dict(off[0], text=canonical(result)), *off[1:]]
        dropped, report = drop_static_once_report(routine, plan=plan)
        self.assertFalse(report['exception'])
        for key in ('semantics', 'executed_steps', 'completed_native_steps', 'executed_control_steps',
                    'unknown_values_are_null', *(('collision_scope',) if 'collision' in plan.segments else ())):
            self.assertGreaterEqual(report['dropped'].get(key, 0), 1, key)
        # collision_scope is left out only when it is the text the backend's capabilities publish
        self.assertEqual('collision' in plan.segments, isinstance((self.h.capabilities or {}).get('collision_scope'), str))
        before, after = json.loads(off[0]['text']), json.loads(on[0]['text'])
        self.assertEqual(after['receipt']['status'], 'completed')
        for key in ('endpoint_residual_m', 'completed_native_steps', 'executed_steps', 'completed_stage_ids'):
            self.assertEqual(after['receipt'][key], before['receipt'][key])
        self.assertEqual({name: stage.get('endpoint_error') for name, stage in after['stages'].items()},
                         {name: stage.get('endpoint_error') for name, stage in before['stages'].items()})
        # The read entry is a call that works as written; a diagnostic that happens to be named "detail" is not one.
        self.assertEqual(after['trajectory_feedback'], before['trajectory_feedback'])
        self.assertIsInstance(after['trajectory_feedback']['detail'], str)
        self.assertNotIn('next_action', after)
        entry = after['detail']
        self.assertEqual(entry, dict(tool='robot_action', arguments=before['next_action']))
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            read = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='detail',
                                           tool=entry['tool'], arguments=entry['arguments']))
        self.assertTrue(read['success'], read)
        self.assertEqual(self.robot.execute.call_count, 1)
        # The catalogue states the notes once, in robot_action only, and only with the switch on.
        plain, noted = ({tool['name']: tool for tool in catalogue} for catalogue in (catalogues[False], catalogues[True]))
        self.assertEqual({name for name in plain if plain[name] != noted[name]}, {'robot_action'})
        self.assertEqual(noted['robot_action']['description'].replace(plan.note, '', 1),
                         plain['robot_action']['description'])
        self.assertEqual(noted['robot_action']['inputSchema'], plain['robot_action']['inputSchema'])

    def test_old_checkpoint_replay_preserves_current_plan_and_default_continuation(self):
        from support.motion_fixtures import preview_adapter
        plan = self.preview(dict(observation_id=self.h.latest['observation_id'], stages=[
            dict(id='check' + str(i), kind='observe', reason='Inspect checkpoint') for i in range(3)]))
        adapter, _, _ = preview_adapter()
        def execute(action, **kwargs):
            if not getattr(adapter, '_batch_plans', None):
                native = adapter.preview_batch(action)
                adapter._batch_plans[action['batch_plan_id']] = adapter._batch_plans.pop(native['plan_id'])
            receipt = adapter.execute(action, lambda _: None, kwargs['cancel_flag'])
            receipt.pop('before_observation', None)
            if receipt.get('terminal_observation') is not None:
                receipt['terminal_observation'] = self.robot.observe()
            return receipt
        self.robot.execute = Mock(side_effect=execute)
        def run(action_id, call_id):
            return typed_dispatch(self.h, dict(mode='execute', plan_id=plan['plan_id'], action_id=action_id),
                dict(tool_call_id=call_id)).data['receipt']
        first = deepcopy(run('first-checkpoint', 'first'))
        run('second-checkpoint', 'second')
        current = deepcopy(self.h.unified_action_plans[plan['plan_id']]['summary'])
        self.assertEqual(current['plan_revision'], 2)
        self.assertEqual(run('first-checkpoint', 'replay-first'), first)
        self.assertEqual(self.h.unified_action_plans[plan['plan_id']]['summary'], current)
        self.assertEqual(self.robot.execute.call_count, 2)
        third = run('third-checkpoint', 'third')
        self.assertEqual((third['status'], third['reason']), ('stopped', 'observation_checkpoint'))
        self.assertEqual(third['batch_status'], 'awaiting_observation')
        self.assertEqual(third['batch_revision'], 3)
        self.assertEqual(self.robot.execute.call_count, 3)
        self.assertEqual(self.store.read('action_receipt', 'first-checkpoint'), first)

    def test_actor_feedback_failure_keeps_receipt_and_idempotent_retransmission(self):
        def fail_details(data, kind, *args, **kwargs):
            if kind == 'unified_action_result':
                raise OSError('fixture_detail_archive_failure')
            return original_put(data, kind, *args, **kwargs)
        for status in ('completed', 'stopped', 'failed'):
            for fault in ('observation', 'details'):
                with self.subTest(status=status, fault=fault):
                    # Each uncertain execution closes admission for its own run.
                    self.setUp()
                    self.h.thread_id = 'thread'
                    original_put = self.store.put_artifact
                    call_id = status + '-' + fault
                    plan = self.preview(call='preview-' + call_id)
                    self.native_execution()
                    self.robot.execute.return_value.update(status=status,
                        stop_confirmation=dict(confirmed=False), execution_uncertain=True)
                    params = dict(threadId='thread', turnId='turn', callId=call_id, tool='robot_action',
                        arguments=dict(mode='execute', plan_id=plan['plan_id'], action_id=call_id))
                    failure = (patch.object(self.h, 'observation_content', side_effect=ValueError('fixture_encoding_failure'))
                        if fault == 'observation' else patch.object(self.store, 'put_artifact', side_effect=fail_details))
                    with failure:
                        shown = self.h.handle_tool(params)
                    repeated = self.h.handle_tool(params)
                    self.assertEqual(repeated, shown)
                    self.assertFalse(shown['success'])
                    warning = json.loads(shown['contentItems'][0]['text'])['output_contract_error']
                    self.assertEqual(warning['reason'], 'action_feedback_unavailable')
                    self.assertTrue(warning['physical_action_executed'])
                    self.assertIn('do not automatically replay', warning['suggestion'])
                    receipt = json.loads(shown['contentItems'][1]['text'])['receipt']
                    self.assertEqual(receipt['status'], status)
                    self.assertEqual(receipt['completed_native_steps'], 2)
                    self.assertTrue(receipt['execution_uncertain'])
                    self.assertEqual(receipt['stop_confirmation'], dict(confirmed=False))
                    self.assertEqual(self.store.read('action_receipt', call_id)['status'], status)
                    self.assertEqual(self.robot.execute.call_count, 1)
                    self.assertFalse(self.h.accepting)

    def test_default_actor_failure_retains_joint_guard_numbers(self):
        cases = [dict(reason='batch_resume_first_joint_step_exceeds_limit',
                      first_joint_step_rad=[.009317894881990335, .10970402463135542],
                      max_joint_step_rad=[.10000000149011612, .10000000149011612]),
                 dict(reason='batch_segment_start_changed', max_joint_start_error_rad=.08408594131469727)]
        self.h.thread_id = 'thread'
        for index, diagnostics in enumerate(cases):
            with self.subTest(diagnostics=diagnostics):
                plan = self.preview(call='guard-preview-' + str(index))
                self.native_execution()
                self.robot.execute.return_value.update(status='failed', resume_validation={'large': 'not requested'},
                                                       **diagnostics)
                reply = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='guard-' + str(index),
                    tool='robot_action', arguments=dict(mode='execute', plan_id=plan['plan_id'],
                                                       action_id='guard-' + str(index))))
                receipt = json.loads(reply['contentItems'][0]['text'])['receipt']
                self.assertEqual(receipt['status'], 'failed')
                for key, value in diagnostics.items():
                    # the default display (lean_v2) prints floats with six decimals; typed data keeps every digit
                    self.assertEqual(receipt[key], lean_v2_display(value))
                    self.assertEqual(self.store.read('action_receipt', 'guard-' + str(index))[key], value)
                self.assertEqual(receipt['resume_validation'], {'large': 'not requested'})

    def test_program_feedback_failure_keeps_broker_evidence_and_does_not_replay(self):
        runner = ProgramRunner(self.h.program_services())
        self.h.programs = runner
        self.addCleanup(runner.close)
        for status in ('completed', 'stopped', 'failed'):
            with self.subTest(status=status):
                plan = self.preview(call='preview-program-' + status)
                self.native_execution()
                self.robot.execute.return_value['status'] = status
                action_id = 'program-action-' + status
                request = dict(operation_id='execute-once', tool='robot_action', arguments=dict(
                    mode='execute', plan_id=plan['plan_id'], action_id=action_id))
                source = ('from ni_rua_client import NiRuaClient\nc = NiRuaClient()\nrequest = ' +
                    repr(request) + '\nfirst = c._request(request)\nassert c._request(request) == first\n'
                    'assert not first["success"]\n')
                with patch.object(self.h, 'observation_content', side_effect=ValueError('fixture_encoding_failure')):
                    job = runner.run(dict(source=source, timeout_s=300, allow_direct_actions=True),
                        dict(thread_id='thread', turn_id='turn', tool_call_id='program-' + status))
                self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
                self.assertEqual(job['call_summary']['calls'], 1)
                self.assertEqual(job['call_summary']['failed_calls'], 1)
                evidence = job['action_summary']['recent_actions'][0]
                self.assertEqual(evidence['status'], status)
                self.assertEqual(evidence['action_id'], action_id)
                self.assertEqual(evidence['completed_native_steps'], 2)
                self.assertFalse(evidence['dispatch_success'])
                self.assertEqual(job['final_observation_id'], self.store.read('action_receipt', action_id)['result_observation_id'])
                self.assertEqual(self.robot.execute.call_count, 1)

    def test_missing_current_observation_rejects_new_preview_and_execution_but_keeps_replay(self):
        plan=self.preview(); self.native_execution()
        self.robot.execute.return_value['terminal_observation']=None
        args=dict(mode='execute',plan_id=plan['plan_id'],action_id='missing-capture')
        first=dispatch(self.h,args,{'tool_call_id':'first'})
        self.assertEqual(json.loads(first['contentItems'][0]['text'])['receipt']['status'],'completed')
        self.assertIsNone(self.h.latest)
        replay=dispatch(self.h,args,{'tool_call_id':'replay'})
        self.assertEqual(json.loads(replay['contentItems'][0]['text'])['receipt']['status'],'completed')
        with self.assertRaisesRegex(ValueError,'current_observation_unavailable'):
            dispatch(self.h,dict(args,action_id='new'),{'tool_call_id':'new'})
        request=dict(observation_id=plan['observation_id'], stages=[dict(
            id='reach',kind='move',arm='left',grasp_position_m=[.2,-.1,.8],angles_deg=[0,0,0])])
        with self.assertRaisesRegex(ValueError,'current_observation_unavailable'):
            dispatch(self.h,request,{'tool_call_id':'new-preview'})
        self.assertEqual(self.robot.execute.call_count,1)

    def test_inspect_reuses_artifact_even_when_optional_legacy_preview_flag_off(self):
        plan=self.preview()
        self.h.config=self.h.config.with_overrides({'features':{'trajectory_preview':False}})
        result=dispatch(self.h,dict(mode='inspect',plan_id=plan['plan_id']),{})
        texts=[json.loads(c['text']) for c in result['contentItems'] if c['type']=='inputText']
        self.assertEqual(texts[0]['plan_id'],plan['plan_id'])
        self.assertEqual(texts[1]['total_samples'],1)
        self.assertIn('exact cached',texts[1]['limitations'])
        self.assertEqual(self.robot.preview_batch.call_count,1)

    def test_execution_parameter_changes_are_rejected(self):
        plan=self.preview(); self.native_execution()
        with self.assertRaisesRegex(ValueError,'commands_cannot_be_changed'):
            dispatch(self.h,dict(mode='execute',plan_id=plan['plan_id'],stages=self.request()['stages']),{})
        self.robot.execute.assert_not_called()

    def test_postprocess_and_existing_trajectory_feedback_are_retained(self):
        plan=self.preview(); self.native_execution()
        self.h.unified_action_postprocess=Mock(side_effect=lambda result,correlation,**kw: dict(result,settling={'status':'stable'}))
        with patch('ni_rua.core.presentation.feedback_delivery.action_trajectory_feedback',return_value=({'available':True},[])) as feedback:
            result=dispatch(self.h,dict(mode='execute',plan_id=plan['plan_id']),{'tool_call_id':'exec'})
        first=json.loads(result['contentItems'][0]['text'])
        self.assertEqual(first['settling']['status'],'stable')
        self.assertTrue(first['trajectory_feedback']['available'])
        self.h.unified_action_postprocess.assert_called_once()
        feedback.assert_called_once()

    def test_recreated_harness_counters_skip_persisted_short_ids(self):
        self.preview(); self.native_execution()
        first=dispatch(self.h,dict(mode='execute',plan_id='p1'),{'tool_call_id':'execute'})
        action_id=json.loads(first['contentItems'][0]['text'])['action_id']
        delattr(self.h,'_unified_plan_sequence')
        delattr(self.h,'_unified_action_sequence')
        second=self.preview(call='another-preview')
        self.assertEqual(second['plan_id'],'p2')
        from ni_rua.runtime.actions.unified_actions import _next
        self.assertGreater(int(_next(self.h,'action')[1:]),int(action_id[1:]))

    def test_repeated_direct_tool_call_does_not_replan_execute_or_settle(self):
        self.native_execution()
        args=dict(self.request(),mode='direct')
        self.preview(args,call='direct')
        preview_calls=self.robot.preview_batch.call_count
        dispatch(self.h,args,{'tool_call_id':'direct'})
        self.assertEqual(self.robot.preview_batch.call_count,preview_calls)
        self.assertEqual(self.robot.execute.call_count,1)

    def test_settle_options_and_separate_max_budget_are_reviewable(self):
        args=self.request(); args['settle']=dict(roi=[0,0,20,20],max_native_steps=50)
        plan=self.preview(args)
        self.assertEqual(plan['settle']['max_native_steps'],50)
        self.assertEqual(plan['additional_settle_budget']['max_native_steps'],50)
        result=dispatch(self.h,dict(mode='inspect',plan_id=plan['plan_id']),{})
        self.assertEqual(json.loads(result['contentItems'][0]['text'])['settle'],plan['settle'])

    def test_explicit_proxy_geometry_is_bound_to_native_batch(self):
        args=self.request(); args.update(obstacles=[dict(id='box',min_m=[0,0,0],max_m=[.1,.1,.1],evidence='visible')],
            held_object=dict(min_m=[.2,0,.8],max_m=[.3,.1,.9],evidence='held estimate'),contact_entity_id='box')
        action=compile_request(self.h,args)
        self.assertEqual(action['review_proxies']['contact_entity_id'],'box')
        self.assertIn('left',action['review_proxies']['held_objects'])

    def test_inspect_action_and_raw_reference_read_measured_history(self):
        action=compile_request(self.h,self.request()); action['action_id']='archived'
        self.store.save('action_request','archived',action)
        common=dict(identity=dict(self.h.identity),observation_id=self.h.latest['observation_id'],action_id='archived',
            frame_id='env_origin',tcp_config_id='arx-x5-link6-v1')
        path=dict(common,source_kind='measured_action_trajectory',samples=[dict(native_step=0,
            arms={'left':dict(q_rad=[.123]*6,phase='measured')})])
        full=dict(common,schema_version='action-trajectory-feedback-v1',paths={'measured':path},summary={})
        reference=self.store.put_artifact(json.dumps(full).encode(),'action_trajectory_feedback','application/json')
        self.store.set_state('unified_action_feedback','archived',dict(trajectory_ref=reference,identity=dict(self.h.identity)))
        for selection in (dict(action_id='archived'),dict(trajectory_ref=reference,path_kind='measured')):
            output=dispatch(self.h,dict(mode='inspect',**selection),{})
            summary=json.loads(output['contentItems'][0]['text'])
            self.assertEqual(summary['source_kind'],'measured_action_trajectory')
            self.assertEqual(summary['samples'][0]['arms']['left']['q_rad'],[.123]*6)
        self.assertEqual(self.robot.executions,0)

    def test_execute_if_clean_and_inline_plan_update(self):
        self.native_execution()
        # 1) Clean preview with mode='execute_if_clean' and inline plan_update auto-executes in 1 call
        args = dict(
            self.request(),
            mode='execute_if_clean',
            plan_update=dict(
                plan='Task understanding: move arm and grasp. Approach + completion: execute clean batch.',
                attention=[dict(concern='Clearance', check='Observe final frame', when='After batch')],
            ),
        )
        first = self.preview(args, call='clean-auto-1')
        self.assertEqual(first['receipt']['status'], 'completed')
        self.assertEqual(self.robot.execute.call_count, 1)
        saved_plan = self.h.task_planning.current(self.h.identity)
        self.assertIn('execute clean batch', saved_plan['plan'])
        self.assertEqual(len(saved_plan['attention']), 1)

        # 2) When grasp_3d_review has advisory warnings/risk_flags, auto_execute_if_clean passes through and records advisory_notes
        with patch(
            'ni_rua.runtime.actions.grasp_preview_3d.audit_batch_grasp_3d',
            return_value=(
                dict(
                    observation_id=self.h.latest['observation_id'],
                    status='warnings_detected',
                    risk_flags=['warn_along_long_axis'],
                    stages=[],
                    warnings=['Closing along long axis'],
                ),
                [],
            ),
        ):
            warn_args = dict(
                self.request(),
                mode='preview',
                auto_execute_if_clean=True,
                plan_update=dict(attention=[]),
            )
            warn_res = dispatch(self.h, warn_args, {'tool_call_id': 'warn-pass-1'})
            self.assertTrue(warn_res['success'], warn_res)
            self.assertEqual(self.robot.execute.call_count, 2)
            self.assertEqual(self.h.task_planning.current(self.h.identity)['attention'], [])
            warn_header = json.loads(warn_res['contentItems'][0]['text'])
            self.assertEqual(warn_header['advisory_notes'][0]['code'], 'grasp_3d_review_warnings')

        # 3) Hard robot motion_safety reject downgrades execute_if_clean to cached plan_id; explicit execute runs cached plan
        self.h.observe()
        def _reject_once(action):
            full = dict(
                schema_version='candidate-trajectory-v1',
                source_kind='candidate_command_trajectory_not_measurement',
                frame_id='env_origin',
                tcp_config_id='arx-x5-link6-v1',
                joint_names={'left': [], 'right': []},
                samples=[dict(native_step=0, arms={arm: dict(q_rad=[0.] * 6, jaw_goal_normalized=1., phase='initial') for arm in ('left', 'right')})],
                geometry=dict(status='unknown'),
                inter_arm_proximity=dict(status='unknown'),
                execution_binding='exact_cached_rows; no replanning or resampling',
            )
            return dict(
                status='planned',
                plan_id='native-plan',
                planned_native_steps=2,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={},
                motion_safety_checks=[
                    dict(stage_id=action['stages'][0]['id'], enabled=True, status='reject', start_state_source='initial_measurement')
                ],
            )

        self.robot.preview_batch = Mock(side_effect=_reject_once)
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            blocked_wire = dispatch(
                self.h,
                dict(self.request(), mode='execute_if_clean', plan_update=dict(attention=[])),
                {'tool_call_id': 'safety-stop-1'},
            )
        blocked_summary = json.loads(blocked_wire['contentItems'][0]['text'])
        self.assertEqual(blocked_summary['status'], 'planned')
        self.assertTrue(blocked_summary.get('downgraded_from_execute_if_clean'))
        self.assertIn('motion_safety_reject', blocked_summary.get('downgrade_reasons', []))
        self.assertEqual(self.robot.execute.call_count, 2)

        exec_res = dispatch(
            self.h,
            dict(
                mode='execute',
                plan_id=blocked_summary['plan_id'],
                observation_id=self.h.latest['observation_id'],
                timeout_s=120,
                expected_cursor=0,
                expected_revision=0,
                plan_update=dict(
                    ledger=[
                        dict(id='L1', event='Object grasped', basis='visual', next_check='after lift'),
                        dict(id='L2', event='Step 2'),
                        dict(id='L3', event='Step 3'),
                        dict(id='L4', event='Step 4 truncated cleanly'),
                    ]
                ),
            ),
            {'tool_call_id': 'exec-downgraded-1'},
        )
        self.assertTrue(exec_res['success'], exec_res)
        self.assertEqual(self.robot.execute.call_count, 3)
        saved_ledger = self.h.task_planning.current(self.h.identity)['ledger']
        self.assertEqual(len(saved_ledger), 3)
        self.assertEqual(saved_ledger[0]['status'], 'pending')

        # 4) Closed-jaw closure without prior open stage emits neutral advisory_notes without blocking execute_if_clean
        self.h.observe()
        self.preview(dict(self.request(), mode='preflight'), call='restore-clean-stub')
        self.h.latest['grip_facts'] = {'right': {'measured_aperture_m': 0.002}}
        closed_jaw_args = dict(
            self.request(),
            mode='execute_if_clean',
            stages=[
                dict(
                    id='reach',
                    kind='move',
                    arm='right',
                    grasp_position_m=[0.2, -0.1, 0.8],
                    angles_deg=[0, 0, 0],
                ),
                dict(
                    id='close_jaw',
                    kind='gripper',
                    arm='right',
                    goal_normalized=0.0,
                ),
            ],
        )
        closed_exec = dispatch(self.h, closed_jaw_args, {'tool_call_id': 'closed-jaw-pass-1'})
        self.assertTrue(closed_exec['success'], closed_exec)
        self.assertEqual(self.robot.execute.call_count, 4)
        closed_header = json.loads(closed_exec['contentItems'][0]['text'])
        self.assertIn('closed_jaw_before_grasp', {n['code'] for n in closed_header.get('advisory_notes', [])})

        # Calling mode='execute_if_clean' with a cached plan_id (and no stages) executes that cached plan
        self.h.observe()
        cached_pv = self.preview(dict(self.request(), mode='preflight'), call='cached-pf-1')
        exec_via_eic = dispatch(
            self.h,
            dict(
                mode='execute_if_clean',
                plan_id=cached_pv['plan_id'],
                expected_cursor=0,
                expected_revision=0,
            ),
            {'tool_call_id': 'exec-via-eic-1'},
        )
        self.assertTrue(exec_via_eic['success'], exec_via_eic)
        self.assertEqual(self.robot.execute.call_count, 5)

        # 5) Top-level obstacle and clearance fields are admitted in preview/direct/execute_if_clean, and effective_tolerances has zero hidden inflation
        self.h.observe()
        obstacle_args = dict(
            self.request(),
            mode='preview',
            obstacle_object_ids=['obj_1'],
            obstacle_policy='persist',
            ignore_obstacle_ids=['obj_2'],
            target_exemption_radius_m=0.055,
            clearance_height_m=0.13,
            stages=[
                dict(
                    id='strict_5mm',
                    kind='move',
                    arm='right',
                    grasp_position_m=[0.2, -0.1, 0.8],
                    angles_deg=[0, 0, 0],
                    position_tolerance_m=0.005,
                ),
                dict(
                    id='contact_descent',
                    kind='move',
                    arm='right',
                    grasp_position_m=[0.2, -0.1, 0.77],
                    angles_deg=[0, 0, 0],
                    stall_guard_preset='contact_stop',
                ),
            ],
        )
        obstacle_summary = self.preview(obstacle_args, call='obstacle-fields-preview-1')
        self.assertEqual(obstacle_summary['status'], 'planned')
        eff_tols = obstacle_summary['effective_tolerances']
        self.assertEqual(eff_tols['strict_5mm']['position_tolerance_m'], 0.005)
        self.assertEqual(eff_tols['strict_5mm']['max_upward_dz_m'], 0.0)
        self.assertEqual(eff_tols['contact_descent']['max_upward_dz_m'], 0.04)

    def test_top_level_allow_jaw_symmetry_continuation_repair_and_ik_residual_advisory(self):
        from types import SimpleNamespace
        from ni_rua.robot.motion.batch_motion import motion_digest
        from ni_rua.runtime.actions.batch_repair import build_repaired_args
        from ni_rua.runtime.actions.plan_continuation import resolve_continuation_args
        from ni_rua.runtime.actions.unified_actions import (
            _collect_plan_risks_and_notes,
            _extract_measured_support_z_m,
            compile_request,
        )

        obs_id = (self.h.latest or self.h.observe())['observation_id']
        plan = self.preview(
            dict(
                observation_id=obs_id,
                allow_jaw_symmetry=False,
                stages=[
                    {'id': 's1', 'kind': 'move', 'arm': 'left', 'grasp_position_m': [0.20, 0.15, 0.85], 'angles_deg': [0.0, 0.0, 30.0]},
                    {'id': 's2', 'kind': 'move', 'arm': 'left', 'grasp_position_m': [0.20, 0.15, 0.82], 'angles_deg': [0.0, 0.0, 30.0]},
                ],
            ),
            call='u11-jaw-sym-preview',
        )
        saved_record = self.h.unified_action_plans[plan['plan_id']]
        self.assertIs(saved_record['action']['allow_jaw_symmetry'], False)
        for stage in saved_record['action']['stages']:
            self.assertIs(stage['allow_jaw_symmetry'], False)
        act_true = dict(deepcopy(saved_record['action']), allow_jaw_symmetry=True)
        self.assertNotEqual(motion_digest(saved_record['action']), motion_digest(act_true))

        repair_record = deepcopy(saved_record)
        repair_record['source_args'] = {**deepcopy(repair_record.get('source_args') or {}), 'max_downward_dz_m': 0.025}
        repair_record['source_stages'] = [
            {'id': 'grip', 'kind': 'move', 'arm': 'left', 'grasp_position_m': [0.20, 0.15, 0.80], 'angles_deg': [0.0, 0.0, 113.0], 'gripper_goal_normalized': 0.0},
            {'id': 'lift', 'kind': 'move', 'arm': 'left', 'grasp_position_m': [0.20, 0.15, 0.90], 'angles_deg': [0.0, 0.0, 113.0]},
            {'id': 'seat', 'kind': 'move', 'arm': 'left', 'grasp_position_m': [0.12, 0.15, 0.84], 'angles_deg': [0.0, 0.0, 90.0]},
        ]
        repair_record['raw_stages'] = deepcopy(repair_record['source_stages'])
        repair_record['summary'] = {**deepcopy(repair_record.get('summary') or {}), 'status': 'paused', 'completed_stage_ids': ['grip/motion', 'grip'], 'failed_stage_id': 'lift'}
        self.h.unified_action_plans['plan-repair-u11'] = repair_record
        cont_args, _ = resolve_continuation_args(
            self.h,
            {'observation_id': obs_id, 'source_plan_id': 'plan-repair-u11', 'start_from_stage_id': 'lift', 'stage_overrides': {'lift': {'grasp_position_m': [0.20, 0.15, 0.92]}}},
        )
        self.assertIs(cont_args['allow_jaw_symmetry'], False)
        self.assertEqual(cont_args['max_downward_dz_m'], 0.025)
        self.assertEqual(cont_args['stages'][0]['angles_deg'], [0.0, 0.0, 113.0])
        self.assertEqual(cont_args['stages'][1]['angles_deg'], [0.0, 0.0, 90.0])
        repaired_args = build_repaired_args(
            self.h,
            {'observation_id': obs_id, 'plan_id': 'plan-repair-u11', 'skip_failed_stage': True},
            lambda _h, pid: self.h.unified_action_plans[pid],
        )
        self.assertIs(repaired_args['allow_jaw_symmetry'], False)
        self.assertEqual(repaired_args['max_downward_dz_m'], 0.025)
        self.assertEqual([s['id'] for s in repaired_args['stages']], ['seat'])
        self.assertEqual(repaired_args['stages'][0]['angles_deg'], [0.0, 0.0, 90.0])
        with self.assertRaisesRegex(ValueError, 'invalid_allow_jaw_symmetry'):
            compile_request(self.h, {'observation_id': obs_id, 'allow_jaw_symmetry': 'invalid', 'stages': [{'id': 'obs_only', 'kind': 'observe', 'reason': 'check'}]})

        self.assertIsNone(_extract_measured_support_z_m(SimpleNamespace(), 'obs_1'))
        h_s3 = SimpleNamespace(_obs3d_memo=('obs_1', {'scene3d': ({'table_z': 0.765}, None, {})}))
        self.assertAlmostEqual(_extract_measured_support_z_m(h_s3, 'obs_1'), 0.765, places=6)
        blk, adv = _collect_plan_risks_and_notes(
            dict(
                status='planned',
                stage_costs={'hang/motion': dict(native_steps=18, planned_ik_residual_m=0.0022)},
                effective_tolerances={'hang': dict(tolerance_preset='tight', position_tolerance_m=0.003)},
            ),
            args=dict(stages=[]),
            harness=h_s3,
        )
        self.assertEqual(blk, [])
        self.assertEqual([n['stage_id'] for n in adv if n.get('code') == 'warn_planned_ik_residual_near_tolerance'], ['hang'])

        # 覆盖 motion_safety_rejected 步数预算与关节缠绕子因诊断及零元数据泄露
        from ni_rua.core.contracts import canonical
        from ni_rua.core.observation_policy import LEGACY_ACTOR_PROFILE
        from ni_rua.core.presentation.actor_presentation import metadata_findings

        budget_raw = dict(
            status='failed',
            reason='motion_safety_rejected',
            failed_stage_id='pour',
            failed_arm='left',
            planned_native_steps=190,
            motion_safety_checks=[
                dict(stage_id='lift', enabled=True, status='pass', arms=dict(left=dict(status='pass', reasons=[], nominal_native_steps=80))),
                dict(
                    stage_id='pour',
                    enabled=True,
                    status='reject',
                    arms=dict(
                        left=dict(
                            status='reject',
                            reasons=['insufficient_remaining_native_budget'],
                            nominal_native_steps=140,
                            episode_remaining_native_steps=120,
                            request_remaining_native_steps=250,
                            exceeded_budgets=dict(episode_remaining_native_steps=120),
                        )
                    ),
                ),
            ],
        )
        b_risks, b_notes = _collect_plan_risks_and_notes(budget_raw)
        self.assertEqual(len(b_risks), 1)
        self.assertEqual(b_risks[0]['code'], 'motion_safety_rejected')
        self.assertEqual(b_risks[0]['sub_reasons'], ['insufficient_remaining_native_budget'])
        self.assertEqual(b_risks[0]['episode_remaining_native_steps'], 120)
        self.assertEqual(b_risks[0]['request_remaining_native_steps'], 250)
        self.assertEqual([n['code'] for n in b_notes], ['motion_safety_diagnosis'])
        self.assertEqual(list(metadata_findings(canonical(dict(blocking_risks=b_risks, advisory_notes=b_notes)))), [])

        dummy_r_state = dict(arms=dict(left=dict(q_rad=[0.0, 0.8, 0.8, -0.5, 0.0, 0.0]), right=dict(q_rad=[0.0, 0.8, 0.8, -0.5, 0.0, 0.0])))
        dummy_harness = SimpleNamespace(
            observation_profile='rgb_proprio_calibrated_v1',
            current_observation=lambda: dict(observation_id='obs_1', robot_state=dummy_r_state),
            robot_state=dummy_r_state,
        )
        travel_raw = dict(
            status='failed',
            reason='motion_safety_rejected',
            failed_stage_id='twist_place/motion',
            failed_arm='left',
            planned_native_steps=42,
            motion_safety_checks=[
                dict(
                    stage_id='twist_place/motion',
                    enabled=True,
                    status='reject',
                    arms=dict(
                        left=dict(
                            status='reject',
                            reasons=['abnormal_joint_travel'],
                            witnesses=[dict(joint_name='left_joint6', reason='periodic_winding', total_travel_rad=5.42, signed_net_rad=-5.42, shortest_periodic_delta_rad=0.86)],
                            nominal_native_steps=42,
                            request_remaining_native_steps=250,
                            episode_remaining_native_steps=400,
                            exceeded_budgets={},
                        )
                    ),
                )
            ],
        )
        t_args = dict(observation_id='obs_1', stages=[dict(id='twist_place', kind='move', arm='left', grasp_position_m=[-0.15, 0.05, 0.88], angles_deg=[0.0, 0.0, -45.0])])
        t_risks, t_notes = _collect_plan_risks_and_notes(travel_raw, args=t_args, harness=dummy_harness)
        self.assertEqual(t_risks[0]['sub_reasons'], ['abnormal_joint_travel'])
        t_codes = [n['code'] for n in t_notes]
        self.assertIn('motion_safety_diagnosis', t_codes)
        self.assertNotIn('kinematic_reachability_diagnosis', t_codes)
        self.assertEqual(list(metadata_findings(canonical(dict(blocking_risks=t_risks, advisory_notes=t_notes)))), [])

        # 未标定档案下零步 path_infeasible 绝不越权伪造 6R 可达性诊断
        uncal_harness = SimpleNamespace(
            observation_profile=LEGACY_ACTOR_PROFILE,
            current_observation=lambda: dict(observation_id='obs_1', observation_profile=LEGACY_ACTOR_PROFILE, robot_state=dummy_r_state),
            robot_state=dummy_r_state,
        )
        _, uncal_notes = _collect_plan_risks_and_notes(
            dict(status='failed', reason='path_infeasible', failed_stage_id='twist_place', failed_arm='left', executed_native_steps=0),
            args=t_args,
            harness=uncal_harness,
        )
        self.assertNotIn('kinematic_reachability_diagnosis', [n['code'] for n in uncal_notes])

    def test_path_infeasible_approach_tilt_hint_passes_output_schema_and_presentation(self):
        from ni_rua.core.presentation.actor_output import ActorOutput
        from ni_rua.core.presentation.actor_presentation import ActorPresentation, metadata_findings

        hint_text = (
            "allow_approach_tilt=false left -15/+15/-30/+30 deg closing-axis tilts unsearched; "
            "setting allow_approach_tilt=true on the failed stage (or passing explicit "
            "angles_deg=[75.0, 0.0, 0.0] deg, keeping grasp_position_m fixed) resolves path_infeasible."
        )
        tilts = [[75.0, 0.0, 0.0], [-75.0, 0.0, 180.0]]
        raw_failure = dict(
            status='failed',
            reason='path_infeasible',
            failed_stage_id='box_grasp/motion',
            failed_arm='right',
            physics_advanced=False,
            executed_native_steps=0,
            feasible_approach_tilt_angles_deg=tilts,
            failure_hint=hint_text,
            failure_feedback=dict(
                schema_version='1',
                arm='right',
                allow_approach_tilt=False,
                feasible_approach_tilt_angles_deg=deepcopy(tilts),
                suggested_adjustments=[hint_text],
            ),
        )
        self.robot.preview_batch = Mock(return_value=deepcopy(raw_failure))
        args_base = dict(
            observation_id=self.h.latest['observation_id'],
            stages=[
                dict(
                    id='box_grasp',
                    kind='move',
                    arm='right',
                    grasp_position_m=[0.24, -0.14, 0.84],
                    angles_deg=[90.0, 0.0, 0.0],
                    gripper_goal_normalized=0.0,
                )
            ],
        )
        for mode in ('preview', 'execute_if_clean', 'direct'):
            with self.subTest(mode=mode):
                self.robot.preview_batch = Mock(return_value=deepcopy(raw_failure))
                call_args = dict(args_base, mode=mode)
                reply = self.h.invoke('robot_action', call_args, {'tool_call_id': f'tilt-{mode}'}, role='actor')
                self.assertIsNone(reply.contract_error, reply.contract_error)
                self.assertEqual(reply.data['status'], 'failed')
                self.assertEqual(reply.data['feasible_approach_tilt_angles_deg'], tilts)
                self.assertEqual(reply.data['failure_hint'], hint_text)
                self.assertEqual(reply.data['blocking_risks'][0]['feasible_approach_tilt_angles_deg'], tilts)
                self.assertIn('allow_approach_tilt=true', reply.data['blocking_risks'][0]['message'])
                tilt_notes = [n for n in reply.data['advisory_notes'] if n.get('code') == 'approach_tilt_available']
                self.assertEqual(len(tilt_notes), 1)
                self.assertEqual(tilt_notes[0]['stage_id'], 'box_grasp')
                self.assertEqual(tilt_notes[0]['feasible_approach_tilt_angles_deg'], tilts)
                if mode == 'direct':
                    self.assertEqual(reply.data['receipt']['feasible_approach_tilt_angles_deg'], tilts)

                presentation = ActorPresentation(self.store)
                for payload_profile, compact in (('lean_v1', False), ('lean_v2', False), ('lean_v2', True)):
                    out = ActorOutput(self.store, presentation, payload_profile=payload_profile, compact=compact)
                    wire = out.present(reply, projection='action_plan', tool='robot_action')
                    self.assertTrue(wire['success'], wire)
                    text = wire['contentItems'][0]['text']
                    self.assertEqual(metadata_findings(text), [])
                    decoded = json.loads(text)
                    self.assertEqual(decoded['feasible_approach_tilt_angles_deg'], tilts)
                    self.assertIn('allow_approach_tilt=true', decoded['failure_hint'])
                    self.assertEqual(
                        decoded['failure_feedback']['feasible_approach_tilt_angles_deg'],
                        tilts,
                    )

    def test_carry_slip_guard_public_schema_and_compile_request_wiring(self):
        obs_id = self.h.latest['observation_id']
        captured = []

        def preview_stub(action):
            captured.append(deepcopy(action))
            full = dict(
                schema_version='candidate-trajectory-v1',
                source_kind='candidate_command_trajectory_not_measurement',
                frame_id='env_origin',
                tcp_config_id='arx-x5-link6-v1',
                joint_names={'left': [], 'right': []},
                samples=[dict(native_step=0, arms={a: dict(q_rad=[0.0] * 6, jaw_goal_normalized=1.0, phase='initial') for a in ('left', 'right')})],
                geometry=dict(status='unknown'),
                inter_arm_proximity=dict(status='unknown'),
                execution_binding='exact_cached_rows; no replanning or resampling',
            )
            return dict(
                status='planned',
                plan_id='p-cs',
                planned_native_steps=2,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={s['id']: dict(nominal_native_steps=1) for s in action['stages']},
            )

        self.robot.preview_batch = Mock(side_effect=preview_stub)
        reply = self.h.invoke(
            'robot_action',
            dict(
                mode='preflight',
                observation_id=obs_id,
                grip_guard_preset='rigid_object',
                carry_slip_guard=True,
                carry_slip_scope='until_release',
                slip_consecutive_steps=2,
                stages=[
                    dict(id='close', kind='gripper', arm='left', goal_normalized=0.0, carry_slip_guard=True),
                    dict(id='lift', kind='move', arm='left', grasp_position_m=[0.2, -0.1, 0.85], angles_deg=[0, 0, 0]),
                ],
            ),
            {'tool_call_id': 'cs-preflight-1'},
            role='actor',
        )
        self.assertIsNone(reply.contract_error, reply.contract_error)
        self.assertEqual(reply.data['status'], 'planned')
        compiled = captured[-1]
        self.assertTrue(compiled['carry_slip_guard'])
        self.assertEqual(compiled['carry_slip_scope'], 'until_release')
        self.assertEqual(compiled['slip_consecutive_steps'], 2)
        self.assertTrue(compiled['stages'][0]['carry_slip_guard'])

    def test_post_execution_bookkeeping_crash_preserves_action_receipt_and_program_shape(self):
        plan = self.preview()
        self.native_execution()
        with patch(
            'ni_rua.runtime.actions.action_state_ledger.record_execution_outcome',
            side_effect=TypeError('corrupt ledger row'),
        ):
            actor_resp = self.h.invoke(
                'robot_action',
                dict(mode='execute', plan_id=plan['plan_id'], action_id='act-bookkeep-actor'),
                dict(tool_call_id='call-bookkeep-actor'),
                role='actor',
            )
        self.assertEqual(actor_resp.kind, 'action')
        self.assertEqual(actor_resp.data['action_id'], 'act-bookkeep-actor')
        self.assertEqual(actor_resp.data['receipt']['status'], 'completed')
        self.assertEqual(
            actor_resp.data['post_execution_bookkeeping_error'],
            dict(error_type='TypeError', action_replay_permitted=False),
        )
        self.assertEqual(actor_resp.contract_error['reason'], 'action_feedback_unavailable')
        self.assertEqual(self.robot.execute.call_count, 1)

        # Idempotent replay returns cached ToolResponse(kind='action') without re-executing
        replayed = self.h.invoke(
            'robot_action',
            dict(mode='execute', plan_id=plan['plan_id'], action_id='act-bookkeep-actor'),
            dict(tool_call_id='call-bookkeep-actor'),
            role='actor',
        )
        self.assertEqual(replayed.kind, 'action')
        self.assertEqual(self.robot.execute.call_count, 1)

        # Program caller receives flattened status / observation_id / stages on fallback
        self.h.programs = SimpleNamespace(admit_action=Mock(), unresolved_jobs=lambda **kw: [], entries={})
        self.h.observe()
        plan_prog = self.preview(self.request(), call='preview-prog')
        self.native_execution()
        with patch(
            'ni_rua.runtime.actions.plan_continuation.summarize_stage_progress',
            side_effect=KeyError('missing_stage'),
        ):
            prog_resp = self.h.invoke(
                'robot_action',
                dict(mode='execute', plan_id=plan_prog['plan_id'], action_id='act-bookkeep-prog'),
                dict(program_id='program-1', tool_call_id='call-bookkeep-prog'),
                role='program',
            )
        self.assertEqual(prog_resp.kind, 'action')
        self.assertEqual(prog_resp.data['status'], 'completed')
        self.assertEqual(prog_resp.data['action_id'], 'act-bookkeep-prog')
        self.assertIsInstance(prog_resp.data['stages'], dict)

    def test_robot_observe_post_capture_render_crash_preserves_new_observation_id_and_batch_geometry_status(self):
        prior_obs_id = self.h.latest['observation_id']
        orig_content = self.h.observation_content
        calls = {'n': 0}

        def fail_image_render(*args, **kwargs):
            calls['n'] += 1
            if kwargs.get('include_images', True) is not False:
                raise RuntimeError('image_encode_failed')
            return orig_content(*args, **kwargs)

        with patch.object(self.h, 'observation_content', side_effect=fail_image_render):
            obs_resp = self.h.invoke('robot_observe', {}, dict(tool_call_id='obs-render-fail'), role='actor')
        self.assertEqual(obs_resp.kind, 'observation')
        self.assertNotEqual(obs_resp.data['observation_id'], prior_obs_id)
        self.assertEqual(obs_resp.data['observation_id'], self.h.latest['observation_id'])
        self.assertEqual(obs_resp.contract_error['reason'], 'observation_render_unavailable')

        # Batch geometry_query reports invalid_input / unknown when all sub-queries fail
        bad_batch = self.h.invoke(
            'geometry_query',
            dict(
                request=dict(
                    kind='batch',
                    observation_id='nonexistent-obs',
                    queries=[dict(kind='pixel_ray', camera_id='cam_head', pixel=[10, 10])],
                )
            ),
            dict(tool_call_id='batch-geom-fail'),
            role='actor',
        )
        self.assertIn(bad_batch.data['status'], ('unknown', 'invalid_input'))
        self.assertNotEqual(bad_batch.data['status'], 'supported')



if __name__ == "__main__":
    unittest.main()
