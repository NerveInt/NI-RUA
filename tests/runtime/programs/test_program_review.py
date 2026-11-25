"""
- test_activity_sources.py: Actual native/worker output refreshes inactivity; polling and logs do not.
- test_program_review.py: Real isolated processes retain their stack while the main actor reviews.
- test_program_visual_feedback.py: Program-only visual feedback reaches the actor without another image encode.
"""
import io
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ni_rua.cortex.agents.codex.session import CodexSession
from ni_rua.core.config import Config
from ni_rua.robot.backends.robodojo.robot_process import RobotProcess
from ni_rua.core.storage.store import Store
from ni_rua.robot.backends.robodojo.worker_protocol import WIRE_PREFIX
from support.host_capabilities import program_isolation, require
from support.program_fixture import program_services
from copy import deepcopy
from ni_rua.runtime.programs.runner import ProgramRunner, program_remaining, review_wait
from ni_rua.runtime.programs.program_result import program_result
from ni_rua.core.tool_output import ToolOutput, ToolResponse, wire_response
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
import base64
from PIL import Image
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.core.tool_output import ToolOutput
from support.motion_fixtures import DualArmHarnessBase

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_activity_sources.py ---

class ActivitySourcesTest(unittest.TestCase):
    def test_native_activity_excludes_empty_deltas_status_and_foreign_threads(self):
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp = temp_dir.name
        if True:
            store = Store(Path(temp), dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
            self.addCleanup(store.close)
            session = CodexSession(temp, store, lambda _: {}, config=Config.from_dict({}))
            self.addCleanup(session.close)
            session.thread_id = 'owned'
            session.last_activity_at = 1
            for method, params in [
                ('thread/status/changed', {'threadId': 'owned'}),
                ('item/commandExecution/outputDelta', {'threadId': 'owned', 'delta': ''}),
                ('item/commandExecution/outputDelta', {'threadId': 'other', 'delta': 'output'}),
            ]:
                session._notification(dict(method=method, params=params))
                self.assertEqual(session.last_activity_at, 1)
            for method, extra in [
                ('item/commandExecution/outputDelta', {'delta': 'output'}),
                ('item/reasoning/textDelta', {'delta': 'thinking'}),
                ('item/tool/call', {'callId': 'tool'}),
                ('item/started', {'item': {'id': 'command', 'type': 'commandExecution'}}),
                ('item/completed', {'item': {'id': 'command', 'type': 'commandExecution'}}),
            ]:
                session.last_activity_at = 1
                with patch('ni_rua.cortex.agents.codex.session.time.monotonic', return_value=20):
                    session._notification(dict(method=method, params={'threadId': 'owned', **extra}))
                self.assertEqual(session.last_activity_at, 20)

    def test_robot_activity_requires_new_pending_worker_milestone(self):
        with tempfile.TemporaryDirectory() as temp:
            robot = RobotProcess(temp)
            robot.last_activity_at = 1
            robot.pending['request'] = (threading.Event(), {}, None)
            progress = dict(id='request', progress=dict(phase='executing', executed_steps=5, sim_step=10))
            def output():
                yield 'native log chatter\n'
                self.assertEqual(robot.last_activity_at, 1)
                yield WIRE_PREFIX + json.dumps(dict(id='stale', result={})) + '\n'
                self.assertEqual(robot.last_activity_at, 1)
                yield WIRE_PREFIX + json.dumps(progress) + '\n'
                self.assertEqual(robot.last_activity_at, 20)
                robot.last_activity_at = 2
                yield WIRE_PREFIX + json.dumps(progress) + '\n'
                self.assertEqual(robot.last_activity_at, 2)
                progress['progress']['executed_steps'] = 10
                yield WIRE_PREFIX + json.dumps(progress) + '\n'
                self.assertEqual(robot.last_activity_at, 20)
                robot.last_activity_at = 3
                yield WIRE_PREFIX + json.dumps(dict(id='request', result={})) + '\n'
                self.assertEqual(robot.last_activity_at, 20)
            robot.process = SimpleNamespace(stdout=output(), poll=lambda: 0)
            robot.log = io.StringIO()
            with patch('ni_rua.robot.backends.robodojo.robot_process.time.monotonic', return_value=20):
                robot._read()

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_program_output_advances_clock_without_polling_refresh(self):
        from support.program_fixture import program_services
        from ni_rua.runtime.programs.runner import ProgramRunner
        temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(temp_dir.cleanup)
        temp = temp_dir.name
        if True:
            store = Store(Path(temp) / 'run', dict(run_id='test', episode_id='ep', env_instance_id='env', env_generation=1))
            self.addCleanup(store.close)
            runner = ProgramRunner(program_services(SimpleNamespace(store=store)))
            self.addCleanup(runner.close)
            results = []
            source = (
                "from pathlib import Path\n"
                "import time\n"
                "print('first', flush=True)\n"
                "while not Path('emit_second').exists():\n"
                "    time.sleep(.01)\n"
                "print('second', flush=True)\n"
                "while not Path('allow_exit').exists():\n"
                "    time.sleep(.01)\n"
            )
            worker = threading.Thread(target=lambda: results.append(runner.run(
                dict(source=source, timeout_s=10),
                dict(thread_id='thread', turn_id='turn', tool_call_id='clock'))))
            worker.start()
            entry = None
            try:
                deadline = time.monotonic() + 5
                while time.monotonic() < deadline:
                    logs = list(runner.logs.glob('*.stdout'))
                    if logs and b'first' in logs[0].read_bytes() and runner.entries:
                        entry = next(iter(runner.entries.values()))
                        break
                    time.sleep(.02)
                else:
                    self.fail('program did not produce first output')
                time.sleep(.05)
                first = runner.last_activity_at
                time.sleep(.1)
                self.assertEqual(runner.last_activity_at, first)
                (entry['workspace'] / 'emit_second').write_text('1')
                while time.monotonic() < deadline and b'second' not in logs[0].read_bytes():
                    time.sleep(.02)
                time.sleep(.05)
                # 断言时钟推进发生在子进程存活期间，排除进程退出兜底刷新造成的假阳性
                self.assertIsNone(entry['process'].poll())
                self.assertGreater(runner.last_activity_at, first)
            finally:
                if entry is not None and entry['workspace'].is_dir():
                    (entry['workspace'] / 'emit_second').write_text('1')
                    (entry['workspace'] / 'allow_exit').write_text('1')
                worker.join(12)
            self.assertFalse(worker.is_alive())
            self.assertEqual(results[0]['status'], 'succeeded')



# --- Folded from tests/test_program_review.py ---

class ProgramReviewTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.store = Store(Path(self.temp.name)/'run', dict(run_id='review',episode_id='ep',
            env_instance_id='env',env_generation=1))
        self.addCleanup(self.store.close)
        self.calls = []
        self.plan_count = 0
        self.action_count = 0
        self.stale = False
        self.plan_cursor = 0
        self.plan_revision = 0
        self.harness = SimpleNamespace(store=self.store,lock=threading.RLock(),
            cancel=threading.Event(),active_context=None,dispatch=self.dispatch)
        self.runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(self.runner.close)
        self.identity = dict(thread_id='thread',turn_id='start',tool_call_id='start-call')

    @staticmethod
    def response(body):
        kind = 'action' if 'receipt' in body else 'plan' if 'plan_id' in body else 'data'
        return ToolOutput(body, kind=kind).response()

    def dispatch(self, name, args, correlation):
        self.calls.append((name,dict(args),dict(correlation)))
        if name == 'robot_wait':
            self.action_count += 1
            return self.response(dict(receipt=dict(action_id='wait-action',status='completed',
                result_observation_id='obs-after-wait',executed_native_steps=args['native_steps'])))
        if name != 'robot_action':
            return self.response(dict(test=True))
        mode = args.get('mode','preview')
        if mode == 'preview':
            self.plan_count += 1
            result = self.response(dict(status='planned',plan_id='plan-'+str(self.plan_count),
                observation_id=args['observation_id'],trajectory_evidence={'stages':args['stages']}))
            result.content.append(dict(type='inputImage',imageUrl='data:image/png;base64,AA=='))
            return result
        if mode == 'inspect':
            return self.response(dict(status='planned',plan_id=args['plan_id'],
                executed_batch_native_steps=self.plan_cursor,plan_revision=self.plan_revision,
                trajectory_evidence={'inspected':True}))
        self.action_count += 1
        status = 'rejected' if self.stale else 'completed'
        return self.response(dict(plan_id=args.get('plan_id'),receipt=dict(action_id='action-'+str(self.action_count),
            status=status,reason='stale_plan' if self.stale else None,
            result_observation_id=None if self.stale else 'obs-final-'+str(self.action_count),
            executed_native_steps=0 if self.stale else 2,stop_confirmed=True)))

    def start(self, source, **args):
        return self.runner.run(dict(source=source,timeout_s=10,**args),self.identity)

    def resume(self, job, decision='approve', **extra):
        return self.runner.run(dict(operation='resume',job_id=job['job_id'],
            review_id=job['pending_review']['review_id'],decision=decision,**extra),
            dict(self.identity,turn_id='review',tool_call_id='resume-call'))

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_real_pause_preserves_stack_and_stops_all_program_threads(self):
        source = '''import json, threading, time
from pathlib import Path
from ni_rua_client import NiRuaClient
client=NiRuaClient()
def heartbeat():
    while True:
        Path('heartbeat').write_text(str(time.monotonic()))
        time.sleep(.002)
threading.Thread(target=heartbeat,daemon=True).start()
retained={'value':73}
result=client.action([{'id':'move','arm':'left','kind':'gripper','goal_normalized':1}],observation_id='obs-start')
assert retained['value']==73
print(json.dumps({'retained':retained,'result':result}))
'''
        real_monotonic=time.monotonic
        offset=[0.0]
        with patch('ni_rua.runtime.programs.runner.time.monotonic',side_effect=lambda:real_monotonic()+offset[0]):
            began=time.monotonic();job=self.start(source)
            self.assertLess(time.monotonic()-began,3)
            self.assertEqual(job['status'],'awaiting_review')
            self.assertEqual(self.action_count,0)
            entry=self.runner.entries[job['job_id']]
            heartbeat=entry['workspace']/'heartbeat'
            first=heartbeat.read_text();remaining=program_remaining(entry)
            time.sleep(.02)
            offset[0]+=.1
            self.assertEqual(heartbeat.read_text(),first)
            self.assertAlmostEqual(program_remaining(entry),remaining,places=3)
            self.assertEqual(self.runner.review_content(job['job_id'])[-1]['type'],'inputImage')
            presented=program_result(self.store,job)
            self.assertEqual(presented['resume']['operation'],'resume')
            self.assertNotIn('data:image',json.dumps(presented))
            final=self.resume(job)
        self.assertEqual(final['status'],'succeeded',final.get('stderr_text'))
        self.assertEqual(self.action_count,1)
        self.assertEqual([args['mode'] for name,args,_ in self.calls if name=='robot_action'],['preview','inspect','execute'])
        self.assertEqual(self.calls[-1][1]['expected_cursor'],0)
        self.assertEqual(self.calls[-1][1]['expected_revision'],0)
        self.assertIn('"value": 73',final['stdout_text'])
        self.assertEqual(final['action_summary']['status_counts'],{'completed':1})
        self.assertEqual(final['final_observation_id'],'obs-final-1')
        self.assertGreater(final['review_wait_s'],.09)
        self.assertTrue(final['stop_confirmed'])
        repeated=self.resume(job)
        self.assertEqual(repeated,final)
        self.assertEqual(self.action_count,1)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_next_action_requires_new_review_and_stale_plan_is_not_replanned(self):
        job=self.start("from ni_rua_client import NiRuaClient\nc=NiRuaClient()\nfor i in range(2):\n c.action([{'id':'s','arm':'left','kind':'gripper','goal_normalized':1}],observation_id='obs-start')")
        second=self.resume(job)
        self.assertEqual(second['status'],'awaiting_review')
        self.assertNotEqual(second['pending_review']['review_id'],job['pending_review']['review_id'])
        self.assertEqual(self.action_count,1)
        replay=self.resume(job)
        self.assertEqual(replay['pending_review']['review_id'],second['pending_review']['review_id'])
        self.assertEqual(self.action_count,1)
        self.stale=True
        final=self.resume(second)
        self.assertEqual(final['action_summary']['status_counts'],{'completed':1,'rejected':1})
        self.assertEqual(self.plan_count,2)
        self.assertIsNone(final['final_observation_id'])

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_plan_advanced_by_another_caller_cannot_consume_previous_approval(self):
        job=self.start("from ni_rua_client import NiRuaClient\nNiRuaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')")
        self.plan_cursor=4
        final=self.resume(job)
        self.assertEqual(self.action_count,0)
        self.assertEqual(final['action_summary']['action_calls'],0)
        self.assertEqual(final['call_summary']['recent_errors'][0]['reason'],
                         'program_action_plan_changed_since_review')
        self.assertEqual(self.plan_count,1)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_zero_step_checkpoint_revision_invalidates_old_approval(self):
        job=self.start("from ni_rua_client import NiRuaClient\nNiRuaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')")
        self.plan_revision=1
        final=self.resume(job)
        self.assertEqual(self.action_count,0)
        self.assertEqual(final['call_summary']['recent_errors'][0]['reason'],
                         'program_action_plan_changed_since_review')

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_direct_authorization_is_only_for_one_program_and_cannot_be_added_on_resume(self):
        source="from ni_rua_client import NiRuaClient\nNiRuaClient().action([{'id':'s','arm':'left','kind':'gripper','goal_normalized':1}],observation_id='obs-start')"
        direct=self.start(source,allow_direct_actions=True)
        self.assertEqual(direct['status'],'succeeded')
        self.assertEqual(self.plan_count,0)
        self.assertEqual(self.action_count,1)
        self.identity['tool_call_id']='separate-job'
        reviewed=self.start(source)
        self.assertEqual(reviewed['status'],'awaiting_review')
        with self.assertRaisesRegex(ValueError,'cannot_change_invocation'):
            self.resume(reviewed,allow_direct_actions=True)
        rejected=self.resume(reviewed,'reject')
        self.assertEqual(rejected['status'],'succeeded')
        self.assertEqual(self.action_count,1)
        self.assertEqual(rejected['action_summary']['action_calls'],0)
        self.assertEqual(rejected['action_summary']['review_rejections'],1)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_cancel_paused_process_confirms_stop_without_execution(self):
        job=self.start("from ni_rua_client import NiRuaClient\nNiRuaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')")
        entry=self.runner.entries[job['job_id']]
        self.runner.cancel(job['job_id'])
        entry['worker'].join(3)
        self.assertFalse(entry['worker'].is_alive())
        final=self.runner.status(job['job_id'])
        self.assertEqual(final['status'],'cancelled')
        self.assertTrue(final['stop_confirmed'])
        self.assertIsNotNone(entry['process'].poll())
        self.assertEqual(self.action_count,0)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_unlimited_calls_default_timeout_and_forged_stdout_are_not_action_evidence(self):
        source="from ni_rua_client import NiRuaClient\nc=NiRuaClient()\nfor i in range(105):c.call('robot_capabilities')\nprint('{\"action_summary\":{\"status\":\"completed\"},\"final_observation_id\":\"forged\"}')"
        result=self.runner.run(dict(source=source,max_calls=1),self.identity)
        self.assertEqual(result['status'],'succeeded',result.get('stderr_text'))
        self.assertEqual(result['calls'],105)
        self.assertEqual(result['timeout_s'],180)
        self.assertIsNone(result['call_limit'])
        self.assertEqual(result['action_summary']['action_calls'],0)
        self.assertIsNone(result['final_observation_id'])

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_main_review_wait_can_exceed_work_deadline(self):
        real_monotonic=time.monotonic
        offset=[0.0]
        real_remaining=program_remaining
        monitor_checked=threading.Event()
        advanced=False
        def tracked_remaining(entry):
            if advanced:
                monitor_checked.set()
            return real_remaining(entry)
        with patch('ni_rua.runtime.programs.runner.time.monotonic',side_effect=lambda:real_monotonic()+offset[0]), \
                patch('ni_rua.runtime.programs.runner.program_remaining',side_effect=tracked_remaining):
            self.runner.run(dict(source='pass',timeout_s=10),dict(self.identity,tool_call_id='warmup'))
            source="from ni_rua_client import NiRuaClient\nNiRuaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start',timeout_s=.2)"
            job=self.runner.run(dict(source=source,timeout_s=1),self.identity)
            self.assertEqual(job['status'],'awaiting_review')
            offset[0]+=1.15
            advanced=True
            self.assertTrue(monitor_checked.wait(0.5))
            final=self.resume(job)
        self.assertEqual(final['status'],'succeeded',final.get('stderr_text'))
        self.assertGreater(final['review_wait_s'],1)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_robot_wait_is_reviewed_by_default_and_direct_only_with_invocation_authorization(self):
        self.harness.latest={'observation_id':'obs-start'}
        source="from ni_rua_client import NiRuaClient\nNiRuaClient().call('robot_wait',{'native_steps':2,'observation_id':'obs-start'})"
        job=self.start(source)
        self.assertEqual(job['pending_review']['kind'],'robot_wait')
        self.assertNotIn('plan_id',job['pending_review'])
        self.assertEqual(self.action_count,0)
        final=self.resume(job)
        self.assertEqual(final['action_summary']['executed_native_steps_known'],2)
        self.assertEqual(final['final_observation_id'],'obs-after-wait')
        self.assertEqual(self.action_count,1)
        self.identity['tool_call_id']='direct-wait'
        direct=self.start(source,allow_direct_actions=True)
        self.assertEqual(direct['status'],'succeeded')
        self.assertEqual(self.action_count,2)
        self.identity['tool_call_id']='stale-wait'
        stale=self.start(source)
        self.harness.latest={'observation_id':'different-observation'}
        rejected=self.resume(stale)
        self.assertEqual(self.action_count,2)
        self.assertEqual(rejected['call_summary']['failed_calls'],1)
        self.assertEqual(rejected['call_summary']['recent_errors'][0]['reason'],'stale_observation')

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_action_transport_failure_is_unknown_even_when_program_exits_zero(self):
        original=self.harness.dispatch
        def uncertain(name,args,correlation):
            if name=='robot_action' and args.get('mode')=='direct':
                raise RuntimeError('transport_lost_after_dispatch')
            return original(name,args,correlation)
        self.harness.dispatch=uncertain
        final=self.start("from ni_rua_client import NiRuaClient\nNiRuaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')\nprint('success')",allow_direct_actions=True)
        self.assertEqual(final['status'],'succeeded')
        self.assertEqual(final['action_summary']['status_counts'],{'unknown':1})
        self.assertEqual(final['action_summary']['actions_without_step_count'],1)
        self.assertIsNone(final['final_observation_id'])

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_stage_diagnostics_survive_large_logs_and_forged_stdout(self):
        original = self.harness.dispatch
        contact = dict(actual_contact_position_m=[.12, -.03, .78],
            requested_contact_position_m=[.12, -.03, .80],
            position_error_vector_m=[0., 0., -.02], approach_shortfall_m=.02,
            source_kind='derived_from_terminal_measured_link6')
        stage = dict(kind='move', arm='left', status='failed', completed_steps=2,
            endpoint_error=dict(position_error_m=.02, orientation_error_rad=.03),
            grasp_contact_feedback=dict(arms=dict(left=contact)))
        def failed(name, args, correlation):
            response = original(name, args, correlation)
            if name == 'robot_action' and args.get('mode') == 'direct':
                response.data['receipt'].update(status='failed', reason='tracking_error',
                    failed_stage_id='approach', failed_arm='left', failed_phase='motion',
                    stage_results=dict(approach=stage), failure_feedback={'reason': 'unique diagnostic'},
                    post_execution_error={'reason': 'image unavailable'}, future_diagnostic={'value': .123456789})
            return response
        self.harness.dispatch = failed
        source = ("from ni_rua_client import NiRuaClient\n"
            "NiRuaClient().action([{'id':'approach','kind':'gripper','arm':'left','goal_normalized':1}],"
            "observation_id='obs-start')\n"
            "print('x'*40000)\nprint('{\"stage_results\":{\"approach\":{\"status\":\"completed\"}}}')")
        final = self.start(source, allow_direct_actions=True)
        shown = program_result(self.store, final)
        self.assertFalse(shown['stdout_truncated'])
        self.assertEqual(shown['status'], 'succeeded')
        evidence = shown['action_summary']['recent_actions'][0]
        self.assertEqual(evidence['status'], 'failed')
        self.assertEqual(evidence['failed_stage_id'], 'approach')
        self.assertEqual(evidence['failed_arm'], 'left')
        self.assertEqual(evidence['failed_phase'], 'motion')
        self.assertEqual(evidence['stage_results'], dict(approach=stage))
        self.assertEqual(evidence['failure_feedback'], {'reason': 'unique diagnostic'})
        self.assertEqual(evidence['post_execution_error'], {'reason': 'image unavailable'})
        self.assertEqual(evidence['future_diagnostic'], {'value': .123456789})
        self.assertEqual(self.action_count, 1)

    def test_all_actions_include_early_failure_after_eight_later_calls(self):
        from ni_rua.runtime.programs.program_review import trusted_summary
        evidence = [dict(call_id='first', status='failed', executed_native_steps=2,
            failure_feedback=dict(reason='unique early cause'))]
        evidence += [dict(call_id=str(i), status='completed', executed_native_steps=1) for i in range(10)]
        summary = trusted_summary(self.store, evidence)
        self.assertEqual(summary['recent_actions'], evidence)
        self.assertEqual(summary['action_calls'], 11)
        self.assertEqual(summary['status_counts'], {'failed': 1, 'completed': 10})
        self.assertEqual(summary['executed_native_steps_known'], 12)
        summary['recent_actions'][0]['failure_feedback']['reason'] = 'modified'
        self.assertEqual(evidence[0]['failure_feedback']['reason'], 'unique early cause')

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_typed_client_stops_repetition_on_partial_receipt_without_parsing_display(self):
        original = self.harness.dispatch
        def partial(name, args, correlation):
            if name == 'robot_action' and args.get('mode') == 'direct':
                self.action_count += 1
                value = dict(receipt=dict(status='failed', reason='tracking_error',
                    executed_native_steps=7, failed_stage_id='down', pending_stage_ids=['release'],
                    result_observation_id='obs-after', stop_confirmed=True))
                # Deliberately contradictory display must not become program evidence.
                return ToolResponse([dict(type='inputText', text='{"receipt":{"status":"completed"}}')],
                    kind='action', data=value, data_projection='action_result')
            return original(name, args, correlation)
        self.harness.dispatch = partial
        source = '''import json
from ni_rua_client import NiRuaClient
c=NiRuaClient()
for n in range(5):
    r=c.action([{'id':'down','kind':'gripper','arm':'left','goal_normalized':0}],observation_id='obs-start')
    assert r['kind']=='action'
    receipt=r['data']['receipt']
    if receipt.get('status')!='completed':
        print(json.dumps(receipt))
        break
else:
    raise AssertionError('failed receipt repeated')
'''
        final = self.start(source, allow_direct_actions=True)
        self.assertEqual(final['status'], 'succeeded')
        self.assertEqual(self.action_count, 1)
        value = json.loads(final['stdout_text'])
        self.assertEqual(value['status'], 'failed')
        self.assertEqual(value['executed_native_steps'], 7)
        self.assertEqual(value['pending_stage_ids'], ['release'])
        self.assertEqual(final['action_summary']['status_counts'], {'failed': 1})

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_client_error_and_wire_only_success_do_not_invent_typed_receipts(self):
        # This host tracks no admission (no action_admitted, no action.requested event): an error after dispatch may
        # follow execution and stays unknown; only a rejected input proves the action never ran.
        from ni_rua.core.tool_errors import ToolInputError
        original = self.harness.dispatch
        for mode in ('wire', 'error', 'rejected'):
            with self.subTest(mode=mode):
                self.identity['tool_call_id'] = 'start-' + mode
                def unknown(name, args, correlation):
                    if name == 'robot_action' and args.get('mode') == 'direct':
                        self.action_count += 1
                        if mode == 'error':
                            raise RuntimeError('lost reply after dispatch')
                        if mode == 'rejected':
                            raise ToolInputError('stage_goal_out_of_range', field_path='stages[0].goal_normalized')
                        return dict(success=True, contentItems=[dict(type='inputText',
                            text='{"receipt":{"status":"completed","executed_native_steps":0}}')])
                    return original(name, args, correlation)
                self.harness.dispatch = unknown
                final = self.start("# " + mode + "\nimport json\nfrom ni_rua_client import NiRuaClient\nr=NiRuaClient().action([{'id':'s','kind':'gripper','arm':'left','goal_normalized':0}],observation_id='obs-start')\nprint(json.dumps({'kind':r['kind'],'data':r['data'],'success':r['success']}))", allow_direct_actions=True)
                value = json.loads(final['stdout_text'])
                self.assertEqual(value['kind'], 'untyped' if mode == 'wire' else 'error')
                if mode == 'wire':
                    self.assertIsNone(value['data'])
                else:
                    self.assertFalse(value['success'])
                if mode == 'error':
                    self.assertIn('lost reply', value['data']['error'])
                self.assertEqual(final['action_summary']['status_counts'],
                                 {} if mode == 'rejected' else {'unknown': 1})
        self.assertEqual(self.action_count, 3)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_review_and_action_reply_artifacts_follow_producer_projection(self):
        typed = []
        def response(body):
            value = deepcopy(body)
            value.update(schema_version='0.1', run_id='fixture', env_generation=1)
            if 'receipt' in value:
                value['receipt'].update(schema_version='0.1', versions={'worker': 'build'},
                    failure_feedback=dict(schema_version='diagnostic-v1', reason='retained diagnostic',
                                          error_m=0.123456789012345))
                kind, projection = 'action', 'action_result'
            else:
                value['plan_revision'] = 0
                kind, projection = 'plan', 'action_plan'
            value['diagnostics'] = dict(version='business-version', revision=9)
            result = ToolOutput(value, kind=kind, projection=projection).response()
            typed.append((result, deepcopy(value)))
            return result
        self.response = response
        job = self.start("from ni_rua_client import NiRuaClient\nNiRuaClient().action([{'id':'s','arm':'left','kind':'gripper','goal_normalized':1}],observation_id='obs-start')")
        self.assertEqual(job['status'], 'awaiting_review')
        preview = json.loads(self.store.resolve_artifact(job['pending_review']['preview_ref'])[1])
        shown_preview = json.loads(preview['contentItems'][0]['text'])
        self.assertEqual(shown_preview['plan_revision'], 0)
        self.assertEqual(preview['contentItems'][1], dict(type='inputImage', imageUrl='data:image/png;base64,AA=='))
        for key in ('schema_version', 'run_id', 'env_generation'):
            self.assertNotIn(key, shown_preview)
        final = self.resume(job)
        self.assertEqual(final['status'], 'succeeded')
        evidence = final['action_summary']['recent_actions'][0]
        reply = json.loads(self.store.resolve_artifact(evidence['reply_ref'])[1])
        shown_action = json.loads(reply['contentItems'][0]['text'])
        for key in ('schema_version', 'run_id', 'env_generation'):
            self.assertNotIn(key, shown_action)
        self.assertNotIn('versions', shown_action['receipt'])
        self.assertEqual(shown_action['receipt']['failure_feedback'],
                         dict(reason='retained diagnostic', error_m=0.123456789012345))
        self.assertEqual(shown_action['receipt']['status'], 'completed')
        self.assertEqual(shown_action['receipt']['executed_native_steps'], 2)
        self.assertEqual(shown_action['diagnostics'], dict(version='business-version', revision=9))
        self.assertEqual(self.action_count, 1)
        for response, original in typed:
            self.assertEqual(response.data, original)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_direct_planning_failure_records_zero_execution_and_keeps_last_observation(self):
        original = self.harness.dispatch
        feedback = dict(candidates=[dict(constraint_witness=dict(
            collision_model_witnesses=[dict(trajectory_sample_index=i) for i in range(12)]))])
        def rejected(name, args, correlation):
            if name == 'robot_observe':
                obs = ToolResponse([], kind='observation', data={'observation_id': 'obs-start'})
                obs.captured_live = True
                return obs
            if name == 'robot_action' and args.get('mode') == 'direct':
                return ToolResponse([], kind='plan', data=dict(status='failed', reason='path_infeasible',
                    failed_stage_id='hold', failed_arm='left', physics_advanced=False, executed_native_steps=0,
                    failure_feedback=feedback, future_planning_diagnostic=dict(reason='different approach needed')))
            return original(name, args, correlation)
        self.harness.dispatch = rejected
        final = self.start("from ni_rua_client import NiRuaClient\nc=NiRuaClient()\nc.call('robot_observe')\nc.action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')\nprint('motion completed')", allow_direct_actions=True)
        summary = final['action_summary']
        self.assertEqual(final['status'], 'succeeded')
        self.assertEqual(self.action_count, 0)
        self.assertEqual(summary['status_counts'], {'not_executed': 1})
        self.assertEqual(summary['executed_native_steps_known'], 0)
        self.assertEqual(summary['actions_without_step_count'], 0)
        evidence = summary['recent_actions'][0]
        self.assertEqual(evidence['reason'], 'path_infeasible')
        self.assertEqual(evidence['phase'], 'planning')
        self.assertEqual(evidence['failure_feedback'], feedback)
        self.assertEqual(evidence['future_planning_diagnostic'], dict(reason='different approach needed'))
        self.assertIsNone(evidence['action_id'])
        self.assertEqual(final['final_observation_id'], 'obs-start')

    def test_unproven_zero_execution_stays_unknown(self):
        from ni_rua.runtime.programs.program_review import action_evidence
        plan = dict(status='failed', reason='path_infeasible', physics_advanced=False, executed_native_steps=0)
        replies = [wire_response(ToolResponse([], kind='plan', data=plan)),
                   ToolResponse([], kind='data', data=plan)]
        invalid = ToolResponse([], kind='plan', data=plan)
        invalid.contract_error = {'code': 'invalid_tool_result'}
        replies.append(invalid)
        for change in ({'physics_advanced': None}, {'physics_advanced': 0},
                       {'executed_native_steps': None}, {'executed_native_steps': False},
                       {'executed_native_steps': 1}, {'status': 'planned'}):
            replies.append(ToolResponse([], kind='plan', data={**plan, **change}))
        for response in replies:
            with self.subTest(response=response):
                evidence = action_evidence(self.store, 'robot_action', {'mode': 'direct'}, response, 'call')
                self.assertEqual(evidence['status'], 'unknown')
                self.assertNotIn('executed_native_steps', evidence)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_settling_steps_and_latest_observation_are_in_trusted_program_summary(self):
        original=self.harness.dispatch
        def settled(name,args,correlation):
            response=original(name,args,correlation)
            if name=='robot_action' and args.get('mode')=='direct':
                body=dict(response.data)
                body.update(current_observation_id='obs-after-settling',
                    settling=dict(status='low_visible_motion',executed_native_steps=3))
                return self.response(body)
            return response
        self.harness.dispatch=settled
        final=self.start("from ni_rua_client import NiRuaClient\nNiRuaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')",allow_direct_actions=True)
        summary=final['action_summary']
        self.assertEqual(summary['executed_native_steps_known'],5)
        self.assertEqual(summary['action_native_steps_known'],2)
        self.assertEqual(summary['settling_native_steps_known'],3)
        self.assertEqual(final['final_observation_id'],'obs-after-settling')
        self.assertEqual(summary['recent_actions'][0]['motion_observation_id'],'obs-final-1')

    def test_setup_failure_returns_without_leaving_an_unresumable_job(self):
        with patch('ni_rua.runtime.programs.runner.socket.socketpair',side_effect=OSError('fixture socket failure')):
            final=self.start('print(1)')
        self.assertEqual(final['status'],'failed')
        self.assertTrue(final['stop_confirmed'])
        self.assertIn('fixture socket failure',final['reason'])

    def test_real_harness_unified_cached_plan_review_and_final_main_image_delivery(self):
        from runtime.actions.test_unified_actions import UnifiedActionTests
        fixture=UnifiedActionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        plan=fixture.preview()
        fixture.native_execution()
        runner=fixture.h.programs=ProgramRunner(program_services(fixture.h))
        self.addCleanup(runner.close)
        source=("from ni_rua_client import NiRuaClient\n"
                "NiRuaClient().call('robot_action',"+repr(dict(mode='execute',plan_id=plan['plan_id']))+")")
        reviewed=fixture.h.dispatch('program_run',dict(source=source),self.identity)
        job=json.loads(reviewed['contentItems'][0]['text'])
        self.assertEqual(job['status'],'awaiting_review',job)
        self.assertEqual(job['pending_review']['plan_id'],plan['plan_id'])
        pending_status=fixture.h.dispatch('program_status',dict(job_id=job['job_id']),
            dict(self.identity,tool_call_id='pending-status'))
        self.assertEqual(json.loads(pending_status['contentItems'][0]['text'])['status'],'awaiting_review')
        fixture.robot.execute.assert_not_called()
        completed=fixture.h.dispatch('program_run',dict(operation='resume',job_id=job['job_id'],
            review_id=job['pending_review']['review_id'],decision='approve'),
            dict(self.identity,turn_id='resume',tool_call_id='resume'))
        final=json.loads(completed['contentItems'][0]['text'])
        self.assertEqual(final['status'],'succeeded',final)
        self.assertEqual(final['action_summary']['status_counts'],{'completed':1})
        self.assertEqual(final['final_observation_id'],fixture.h.latest['observation_id'])
        self.assertTrue(any(item['type']=='inputImage' for item in completed['contentItems']))
        final_status=fixture.h.dispatch('program_status',dict(job_id=final['job_id'],detail='full'),
            dict(self.identity,tool_call_id='final-status'))
        self.assertEqual([part for part in final_status['contentItems'] if part['type']=='inputImage'],
            [part for part in completed['contentItems'] if part['type']=='inputImage'])
        self.assertEqual(fixture.robot.preview_batch.call_count,1)
        fixture.robot.execute.assert_called_once()
        self.assertEqual(fixture.robot.execute.call_args.args[0]['batch_expected_cursor'],0)
        self.assertEqual(fixture.robot.execute.call_args.args[0]['batch_expected_revision'],0)

    def test_program_implicit_action_deadline_fits_budget_without_changing_explicit_or_reviewed_values(self):
        from runtime.actions.test_unified_actions import UnifiedActionTests
        for direct in (False, True):
            for explicit in (False, True):
                with self.subTest(direct=direct, explicit=explicit):
                    fixture = UnifiedActionTests()
                    fixture.setUp()
                    self.addCleanup(fixture.doCleanups)
                    fixture.preview()
                    fixture.robot.preview_batch.reset_mock()
                    fixture.native_execution()
                    runner = fixture.h.programs = ProgramRunner(program_services(fixture.h))
                    self.addCleanup(runner.close)
                    args = dict(fixture.request(), mode='direct')
                    if explicit:
                        args['timeout_s'] = 120
                    source = ("from ni_rua_client import NiRuaClient\n"
                              "NiRuaClient().call('robot_action'," + repr(args) + ")")
                    with patch.object(runner, 'admit_action', wraps=runner.admit_action) as admission:
                        job = runner.run(dict(source=source, timeout_s=120,
                            allow_direct_actions=direct), self.identity)
                        if not direct:
                            self.assertEqual(job['status'], 'awaiting_review', job)
                            job = runner.run(dict(operation='resume', job_id=job['job_id'],
                                review_id=job['pending_review']['review_id'], decision='approve'), self.identity)
                        self.assertEqual(job['status'], 'succeeded', job)
                        request = fixture.store.records('action_request')[-1]
                        deadline = request['timeout_s']
                        self.assertEqual(admission.call_args.args[1], deadline)
                        if explicit:
                            self.assertEqual(deadline, 120)
                            fixture.robot.execute.assert_not_called()
                            receipt = fixture.store.records('action_receipt')[-1]
                            self.assertEqual(receipt['reason'], 'program_time_insufficient_for_action')
                        else:
                            self.assertGreater(deadline, 100)
                            self.assertLessEqual(deadline, 114)
                            fixture.robot.execute.assert_called_once()
                            self.assertEqual(fixture.robot.execute.call_args.args[0]['timeout_s'], deadline)
                            if not direct:
                                self.assertEqual(fixture.robot.preview_batch.call_args.args[0]['timeout_s'], deadline)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_implicit_program_deadline_rejects_insufficient_time_before_creating_a_plan(self):
        source = ("from ni_rua_client import NiRuaClient\n"
                  "NiRuaClient().call('robot_action', {'observation_id':'obs', 'mode':'direct',"
                  "'stages':[{'id':'close','kind':'gripper','arm':'left','goal_normalized':0.0}]})")
        with patch('ni_rua.runtime.programs.runner.program_remaining', return_value=6.0):
            job = self.start(source)
        self.assertEqual(job['status'], 'succeeded', job)
        self.assertEqual(job['call_summary']['recent_errors'][0]['reason'], 'program_time_insufficient_for_action')
        self.assertEqual(self.calls, [])

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_read_only_preview_is_available_without_an_action_time_reserve(self):
        source = ("from ni_rua_client import NiRuaClient\n"
                  "NiRuaClient().call('robot_action', {'observation_id':'obs', 'mode':'preview',"
                  "'stages':[{'id':'close','kind':'gripper','arm':'left','goal_normalized':0.0}]})")
        with patch('ni_rua.runtime.programs.runner.program_remaining', return_value=6.0):
            job = self.start(source)
        self.assertEqual(job['status'], 'succeeded', job)
        self.assertEqual(self.plan_count, 1)
        self.assertEqual(self.action_count, 0)
        self.assertEqual(job['call_summary']['recent_errors'], [])

    def test_reviewed_implicit_deadline_is_not_lowered_when_planning_consumes_the_reserve(self):
        from runtime.actions.test_unified_actions import UnifiedActionTests
        fixture = UnifiedActionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.preview()
        fixture.native_execution()
        runner = fixture.h.programs = ProgramRunner(program_services(fixture.h))
        self.addCleanup(runner.close)
        args = dict(fixture.request(), mode='direct')
        source = ("from ni_rua_client import NiRuaClient\n"
                  "NiRuaClient().call('robot_action'," + repr(args) + ")")
        job = runner.run(dict(source=source, timeout_s=120), self.identity)
        self.assertEqual(job['status'], 'awaiting_review', job)
        deadline = fixture.robot.preview_batch.call_args.args[0]['timeout_s']
        with patch('ni_rua.runtime.programs.runner.program_remaining', return_value=deadline + 4):
            runner.run(dict(operation='resume', job_id=job['job_id'],
                review_id=job['pending_review']['review_id'], decision='approve'), self.identity)
        self.assertEqual(fixture.store.records('action_request')[-1]['timeout_s'], deadline)
        self.assertEqual(fixture.store.records('action_receipt')[-1]['reason'], 'program_time_insufficient_for_action')
        fixture.robot.execute.assert_not_called()

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_wait_defaults_fit_program_budget_and_review_preserves_explicit_deadlines(self):
        for explicit in (None, 30.):
            with self.subTest(explicit=explicit):
                args = dict(observation_id='obs', native_steps=1)
                if explicit is not None:
                    args['timeout_s'] = explicit
                source = ("from ni_rua_client import NiRuaClient\nclient=NiRuaClient()\n"
                          "client.call('robot_wait'," + repr(args) + ")")
                job = self.runner.run(dict(source=source, timeout_s=20),
                    dict(self.identity, tool_call_id='wait-' + str(explicit)))
                self.assertEqual(job['status'], 'awaiting_review', job)
                saved = job['pending_review']['preview']['request']['timeout_s']
                if explicit is None:
                    self.assertGreater(saved, .1)
                    self.assertLessEqual(saved, 14)
                else:
                    self.assertEqual(saved, explicit)
                final = self.resume(job)
                self.assertEqual(final['status'], 'succeeded', final)
                self.assertEqual(self.calls[-1][1]['timeout_s'], saved)

    def test_real_harness_checkpoint_advanced_during_review_rejects_previous_approval(self):
        from runtime.actions.test_unified_actions import UnifiedActionTests
        fixture=UnifiedActionTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        plan=fixture.preview()
        fixture.native_execution()
        fixture.robot.execute.return_value.update(batch_status='awaiting_observation',batch_cursor=1,batch_revision=1,
            completed_native_steps=1,executed_native_steps=1,pending_stage_ids=['reach'])
        runner=ProgramRunner(program_services(fixture.h))
        self.addCleanup(runner.close)
        source=("from ni_rua_client import NiRuaClient\n"
                "NiRuaClient().call('robot_action',"+repr(dict(mode='execute',plan_id=plan['plan_id']))+")")
        job=runner.run(dict(source=source,timeout_s=10),self.identity)
        self.assertEqual(job['status'],'awaiting_review')
        fixture.h.dispatch('robot_action',dict(mode='execute',plan_id=plan['plan_id']),
            dict(self.identity,tool_call_id='another-main-action'))
        final=runner.run(dict(operation='resume',job_id=job['job_id'],
            review_id=job['pending_review']['review_id'],decision='approve'),self.identity)
        self.assertEqual(final['call_summary']['recent_errors'][0]['reason'],
                         'program_action_plan_changed_since_review')
        self.assertEqual(final['action_summary']['action_calls'],0)
        fixture.robot.execute.assert_called_once()
        self.assertEqual(fixture.robot.preview_batch.call_count,1)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_legacy_motion_and_nested_authorization_cannot_bypass_review(self):
        source="""import json
from ni_rua_client import NiRuaClient
c=NiRuaClient()
errors={}
for name in ('robot_move_grasp','robot_gripper','robot_policy_step','depth_move','program_run'):
    try:c.call(name,{'allow_direct_actions':True})
    except RuntimeError as exc:errors[name]=str(exc)
    else:raise AssertionError(name)
print(json.dumps(errors))
"""
        final=self.start(source)
        self.assertEqual(final['status'],'succeeded',final.get('stderr_text'))
        self.assertEqual(self.calls,[])
        self.assertEqual(self.action_count,0)
        errors=json.loads(final['stdout_text'])
        for legacy_name in ('robot_move_grasp','robot_gripper','depth_move'):
            self.assertEqual(errors[legacy_name],'program_action_requires_robot_action')
        for actor_only_name in ('robot_policy_step','program_run'):
            self.assertEqual(errors[actor_only_name],'program_tool_not_permitted')

    def test_real_depth_candidate_cannot_bypass_review_through_legacy_depth_move(self):
        from runtime.harness.test_harness_geometry_and_contact import DepthIntegrationTest
        fixture=DepthIntegrationTest()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        harness=fixture.harness()
        target=fixture.query(harness,kind='register_target',
            observation_id=harness.latest['observation_id'],camera_id='head',
            roi=[36,36,24,24],arm='left')
        candidates=fixture.query(harness,kind='grasp_candidates',target_id=target['target_id'])
        candidate=candidates['value']['candidates'][0]
        runner=ProgramRunner(program_services(harness))
        self.addCleanup(runner.close)
        arguments=dict(candidate_id=candidate['candidate_id'],arm='left',stage='approach',timeout_s=1)
        source=("from ni_rua_client import NiRuaClient\n"
                "try: NiRuaClient().call('depth_move',"+repr(arguments)+")\n"
                "except RuntimeError as error:\n"
                " assert str(error)=='program_action_requires_robot_action'\n"
                "else: raise AssertionError('legacy depth execution was allowed')\n")
        final=runner.run(dict(source=source,timeout_s=10),self.identity)
        self.assertEqual(final['status'],'succeeded',final.get('stderr_text'))
        self.assertEqual(fixture.robot.executions,0)
        self.assertEqual(final['action_summary']['action_calls'],0)
        self.assertEqual(final['review_decisions'],{})

    def test_review_wait_excludes_only_own_wall_time_and_cancellation_stays_active(self):
        now=[100.0];entry=dict(cancel=threading.Event(),deadline=110.0)
        other=dict(cancel=threading.Event(),deadline=110.0)
        with patch('ni_rua.runtime.programs.runner.time.monotonic',side_effect=lambda:now[0]):
            with review_wait(entry):
                now[0]=1000.0
                self.assertEqual(program_remaining(entry),10)
                self.assertEqual(program_remaining(other),-890)
            self.assertEqual(program_remaining(entry),10)
            self.assertEqual(entry['review_wait_s'],900)
            entry['cancel'].set()
            with self.assertRaisesRegex(ValueError,'channel_expired'):
                with review_wait(entry):pass



# --- Folded from tests/test_program_visual_feedback.py ---

class ProgramVisualFeedbackTests(unittest.TestCase):
    def setUp(self):
        DualArmHarnessBase.setUp(self)
        self.h.thread_id = 'visual-feedback'
        base_services = self.h.program_services()
        orig_invoke = base_services.invoke

        def _invoke_with_observe_provenance(name, arguments, correlation):
            resp = orig_invoke(name, arguments, correlation)
            # W2 合入前本地单测垫片：若 handlers.robot_observe 尚未挂载 captured_live，按真实采帧分支补齐；合入后自动透传真实属性
            if name == 'robot_observe' and isinstance(resp, ToolResponse) and not hasattr(resp, 'captured_live'):
                reused_cache = bool(
                    arguments.get('camera_id')
                    or arguments.get('focus_3d')
                    or (arguments.get('image_delivery') == 'head_only' and (arguments.get('roi') is not None or arguments.get('scale', 1) != 1))
                )
                resp.captured_live = not reused_cache
            return resp

        self.h.programs = ProgramRunner(replace(base_services, invoke=_invoke_with_observe_provenance))
        self.addCleanup(lambda: self.h.programs.close())

    def call(self, name, args, identifier):
        wire = self.h.handle_tool(dict(threadId=self.h.thread_id, turnId='turn', callId=identifier,
            tool=name, arguments=args))
        self.assertTrue(wire['success'], wire)
        return json.loads(wire['contentItems'][0]['text']), wire['contentItems']

    @staticmethod
    def images(parts):
        return [part for part in parts if part['type'] == 'inputImage']

    def source(self, tool, args):
        return ('from ni_rua_client import NiRuaClient\n'
                'client = NiRuaClient()\n'
                'reply = client.call(' + repr(tool) + ', ' + repr(args) + ')\n'
                "assert reply['success']\nprint('visual feedback requested')\n")

    def test_history_crop_exact_bytes_provenance_retry_full_context_and_restart(self):
        old = self.h.latest
        self.h.observe({'tool_call_id': 'new-observation'})
        args = dict(observation_id=old['observation_id'], camera_id=old['cameras'][0]['camera_id'],
                    roi=[0, 0, 1, 1], scale=2)
        source = self.source('history_view_observation', args)
        job, parts = self.call('program_run', dict(source=source, timeout_s=10), 'start')
        self.assertEqual(job['status'], 'succeeded', job)
        self.assertIsNone(job['final_observation_id'])
        feedback = job['visual_feedback'][0]
        self.assertEqual(feedback['tool'], 'history_view_observation')
        self.assertEqual(feedback['arguments']['roi'], args['roi'])
        raw_job = self.store.states('program')[0]
        saved = json.loads(self.store.resolve_artifact(raw_job['visual_feedback'][0]['reply_ref'])[1])
        self.assertEqual(self.images(parts), self.images(saved['contentItems']))
        self.assertEqual(len(self.images(parts)), 1)
        metadata = json.loads(saved['contentItems'][0]['text'])
        self.assertEqual(metadata['image_view']['source_roi'], args['roi'])
        self.assertEqual(metadata['image_view']['display_scale'], 2)
        self.assertFalse(metadata['observation_is_current'])
        self.assertIn('not a new current observation', json.dumps(parts))
        _, retried = self.call('program_run', dict(source=source, timeout_s=10), 'start')
        self.assertEqual(self.images(retried), self.images(parts))
        polled, compact = self.call('program_status', dict(job_id=job['job_id']), 'poll')
        self.assertEqual(self.images(compact), [])
        self.assertEqual(polled['visual_feedback'][0]['delivery']['previous_tool_call_id'], 'start')
        _, full = self.call('program_status', dict(job_id=job['job_id'], detail='full'), 'full')
        self.assertEqual(self.images(full), self.images(parts))
        self.store.set_state('runtime', 'image_context_epoch', {'epoch': 'reset'})
        _, restored = self.call('program_status', dict(job_id=job['job_id']), 'after-reset')
        self.assertEqual(self.images(restored), self.images(parts))
        self.h.programs.close()
        self.h.programs = ProgramRunner(self.h.program_services())
        _, restarted = self.call('program_status', dict(job_id=job['job_id'], detail='full'), 'after-restart')
        self.assertEqual(self.images(restarted), self.images(parts))
        self.assertEqual(self.robot.executions, 0)

    def test_preview_image_is_forwarded_even_without_printing_reply_or_executing(self):
        buffer = io.BytesIO()
        Image.new('RGB', (7, 5), 'red').save(buffer, format='PNG')
        image = dict(type='inputImage', imageUrl='data:image/png;base64,' + base64.b64encode(buffer.getvalue()).decode())
        original = self.h.programs.services.invoke
        def invoke(name, args, correlation):
            if name == 'robot_action':
                self.assertEqual(args['mode'], 'preview')
                return ToolOutput(dict(plan_id='fixture-plan', status='planned',
                    label='candidate trajectory; not execution'), extra_content=(image,), kind='plan').response()
            return original(name, args, correlation)
        self.h.programs.services = replace(self.h.programs.services, invoke=invoke)
        args = dict(mode='preview', observation_id=self.h.latest['observation_id'],
                    stages=[dict(id='open', arm='left', kind='gripper', goal_normalized=1.)])
        job, parts = self.call('program_run', dict(source=self.source('robot_action', args)), 'preview')
        self.assertEqual(job['status'], 'succeeded', job)
        self.assertEqual(self.images(parts), [image])
        self.assertIn('candidate trajectory; not execution', json.dumps(parts))
        self.assertEqual(job['action_summary']['action_calls'], 0)
        self.assertIsNone(job['final_observation_id'])
        self.assertNotIn('data:image', job['stdout_text'])
        multi_source = (
            'from ni_rua_client import NiRuaClient\n'
            'client = NiRuaClient()\n'
            'assert client.call("robot_action", ' + repr(args) + ')["success"]\n'
            'assert client.call("robot_action", ' + repr(args) + ')["success"]\n'
        )
        multi_job, multi_parts = self.call('program_run', dict(source=multi_source), 'multi-preview')
        self.assertEqual(len(multi_job['visual_feedback']), 1)
        self.assertEqual(len(self.images(multi_parts)), 1)
        current_obs_id = self.h.latest['observation_id']
        cam_id = self.h.latest['cameras'][0]['camera_id']
        crop_out, crop_parts = self.call('robot_observe', dict(camera_id=cam_id, roi=[0, 0, 1, 1], scale=2), 'roi-crop')
        self.assertEqual(crop_out['bundle']['observation_id'], 'obs_1')
        self.assertEqual(self.h.latest['observation_id'], current_obs_id)
        self.assertEqual(len(self.images(crop_parts)), 1)
        self.h.latest = None
        restored, parts = self.call('program_status', dict(job_id=job['job_id'], detail='full'), 'no-current')
        self.assertEqual(restored['status'], 'succeeded')
        self.assertEqual(self.images(parts), [image])
        header = next(json.loads(part['text']) for part in parts if part['type'] == 'inputText'
                      and 'program_visual_feedback' in part['text'])
        self.assertIsNone(header['current_observation_id_at_delivery'])
        self.assertEqual(self.robot.executions, 0)

    def test_failed_delivery_retries_images_and_authored_stdout_is_not_an_image(self):
        args = dict(observation_id=self.h.latest['observation_id'])
        with patch.object(self.h.actor_output, '_present_part', side_effect=ValueError('display failure')):
            failed = self.h.handle_tool(dict(threadId=self.h.thread_id, turnId='turn', callId='failed',
                tool='program_run', arguments=dict(source=self.source('history_view_observation', args))))
        self.assertFalse(failed['success'])
        job = self.store.states('program')[0]
        _, parts = self.call('program_status', dict(job_id=job['job_id']), 'recovery')
        self.assertTrue(self.images(parts))
        authored, parts = self.call('program_run', dict(source="print('data:image/png;base64,AAAA')"), 'authored')
        self.assertEqual(self.images(parts), [])
        self.assertEqual(authored['visual_feedback'], [])
        self.assertIn('omitted', authored['stdout_text'])
        self.assertEqual(self.robot.executions, 0)

    def test_context_change_during_final_frame_and_missing_visual_preserve_job(self):
        source = self.source('history_view_observation', dict(observation_id=self.h.latest['observation_id']))
        source += "client.call('robot_observe', {})\n"
        job, first = self.call('program_run', dict(source=source), 'with-final')
        self.assertEqual(job['status'], 'succeeded')
        self.assertIn('currentness flags and budgets are as of the source call', json.dumps(first))
        original = self.h.observation_content
        def context_reset(*args, **kwargs):
            parts = original(*args, **kwargs)
            self.store.set_state('runtime', 'image_context_epoch', {'epoch': 'during-final'})
            return parts
        with patch.object(self.h, 'observation_content', side_effect=context_reset):
            _, reset = self.call('program_status', dict(job_id=job['job_id']), 'during-reset')
        self.assertEqual(self.images(reset), self.images(first))
        raw_job = self.store.states('program')[0]
        missing_ref = raw_job['visual_feedback'][0]['reply_ref']
        resolve = self.store.resolve_artifact
        def missing(reference, *args, **kwargs):
            if reference == missing_ref:
                raise ValueError('fixture_missing_visual_blob')
            return resolve(reference, *args, **kwargs)
        with patch.object(self.store, 'resolve_artifact', side_effect=missing), \
                patch.object(self.h, 'observation_content', side_effect=ValueError('fixture_missing_final_frame')):
            degraded, parts = self.call('program_status', dict(job_id=job['job_id']), 'missing')
        self.assertEqual(degraded['status'], 'succeeded')
        self.assertEqual(degraded['action_summary'], job['action_summary'])
        self.assertEqual({error['source'] for error in degraded['visual_feedback_errors']},
                         {'supplementary', 'final_observation'})
        self.assertEqual(degraded['visual_feedback'][0]['delivery']['status'], 'unavailable')
        self.assertEqual(self.images(parts), [])
        _, recovered = self.call('program_status', dict(job_id=job['job_id']), 'blob-restored')
        self.assertEqual(self.images(recovered), self.images(first))
        self.assertEqual(self.robot.executions, 0)

    def test_running_job_publishes_visuals_before_terminal_boundary(self):
        source = self.source('history_view_observation', dict(observation_id=self.h.latest['observation_id']))
        source += "from pathlib import Path\nimport time\nwhile not Path('release').exists(): time.sleep(.01)\n"
        with ThreadPoolExecutor(max_workers=1) as pool:
            started = pool.submit(self.h.programs.run, dict(source=source, timeout_s=10),
                dict(thread_id=self.h.thread_id, turn_id='turn', tool_call_id='running-start'))
            deadline = time.monotonic() + 5
            job = None
            while time.monotonic() < deadline:
                jobs = self.store.states('program')
                job = next((item for item in jobs if item.get('visual_feedback')), None)
                if job:
                    break
                time.sleep(.01)
            try:
                self.assertIsNotNone(job, 'visual call must publish before process completion')
                status, parts = self.call('program_status', dict(job_id=job['job_id']), 'running-status')
                self.assertEqual(status['status'], 'running')
                self.assertTrue(self.images(parts))
            finally:
                for item in self.store.states('program'):
                    (self.h.programs.root / item['job_id'] / 'release').touch()
            self.assertEqual(started.result(timeout=5)['status'], 'succeeded')
        self.assertEqual(self.robot.executions, 0)

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    def test_python_environment_initializes_without_reading_run_private_parent(self):
        self.h.config = self.h.config.with_overrides({'features': {'sandbox_enabled': True}})
        self.h.programs.close()
        self.h.programs = ProgramRunner(self.h.program_services())
        private = self.store.root / 'private' / 'not-program-evidence.txt'
        private.parent.mkdir(exist_ok=True)
        private.write_text('fixture-private-data')
        source = ('from pathlib import Path\n'
                  'try:\n Path(' + repr(str(private)) + ").read_text()\n"
                  'except PermissionError:\n print("private parent remains unreadable")\n'
                  'else:\n raise AssertionError("program read private parent")\n')
        job, _ = self.call('program_run', dict(source=source), 'python-environment')
        self.assertEqual(job['status'], 'succeeded', job.get('stderr_text'))
        self.assertIn('private parent remains unreadable', job['stdout_text'])
        self.assertEqual(self.robot.executions, 0)

    def test_pre_admission_preview_crash_does_not_fabricate_unknown_action_or_latch_incomplete(self):
        from unittest.mock import Mock

        obs_id = self.h.latest['observation_id']
        entry = dict(
            job_id='program-preadmission',
            correlation=dict(thread_id=self.h.thread_id, turn_id='turn', tool_call_id='outer-prog'),
            cancel=threading.Event(),
            call_lock=threading.RLock(),
            budget_lock=threading.RLock(),
            calls=0,
            deadline=time.monotonic() + 30,
            model_wait_started_at=None,
            model_wait_s=0.0,
            review_wait_started_at=None,
            review_wait_s=0.0,
            allow_direct_actions=True,
            on_incomplete_action='block',
            latched_incomplete=None,
            final_observation_id=obs_id,
            action_evidence=[],
        )
        # Crash inside preview_batch BEFORE Harness.execute admits any action_request
        self.robot.preview_batch = Mock(side_effect=TypeError('preview signature mismatch'))
        reply = self.h.programs.dispatch(
            entry,
            dict(
                operation_id='op-preadmit-crash',
                tool='robot_action',
                arguments=dict(
                    mode='direct',
                    observation_id=obs_id,
                    stages=[dict(id='g1', kind='gripper', arm='left', goal_normalized=0.0)],
                ),
            ),
        )
        self.assertFalse(reply['success'])
        err_body = json.loads(reply['contentItems'][0]['text'])
        self.assertEqual(err_body['code'], 'internal_runtime_error')
        self.assertEqual(entry['action_evidence'], [])
        self.assertEqual(entry['final_observation_id'], obs_id)
        self.assertIsNone(entry['latched_incomplete'])
        saved_calls = [c for c in self.store.states('program_call') if c.get('operation_id') == 'op-preadmit-crash']
        self.assertEqual(len(saved_calls), 1)
        self.assertFalse(saved_calls[0].get('action_dispatch_attempted'))

        # Subsequent execute_if_clean downgrade during planning also records status='not_executed' without clearing final_observation_id
        def _downgraded_preview(action):
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
                plan_id='plan-downgraded',
                planned_native_steps=2,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={},
                motion_safety_checks=[
                    dict(stage_id=action['stages'][0]['id'], enabled=True, status='reject', start_state_source='initial_measurement')
                ],
            )

        self.robot.preview_batch = Mock(side_effect=_downgraded_preview)
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            downgrade_reply = self.h.programs.dispatch(
                entry,
                dict(
                    operation_id='op-downgrade',
                    tool='robot_action',
                    arguments=dict(
                        mode='execute_if_clean',
                        observation_id=obs_id,
                        stages=[dict(id='g2', kind='gripper', arm='left', goal_normalized=0.0)],
                    ),
                ),
            )
        self.assertTrue(downgrade_reply['success'])
        self.assertEqual(entry['action_evidence'][-1]['status'], 'not_executed')
        self.assertIs(entry['action_evidence'][-1]['physics_advanced'], False)
        self.assertEqual(entry['final_observation_id'], obs_id)
        self.assertIsNone(entry['latched_incomplete'])

    def test_raw_inline_source_fidelity_input_digest_and_collision_guard(self):
        import hashlib
        from ni_rua.core.contracts import digest

        raw_hex = 'f' * 64
        ref = self.store.put_artifact(b'payload', 'fixture_text', 'text/plain')
        short_alias = self.h.actor_output.present(ref)
        obs_alias = self.h.actor_output.present(self.h.latest['observation_id'])
        # 登记一个与合法 64 位十六进制常量同形的短 ID 映射，验证既不把变量名 id_1 展开成语法错误，也不把合法校验和逆映射成短 ID
        raw_id_for_hex = 'e' * 64
        id_alias = 'id_1'
        self.store.set_state('actor_id', id_alias, dict(alias=id_alias, original=raw_id_for_hex))

        source = (
            "import json\n"
            f"{id_alias} = 5\n"
            f"literal_hex = {raw_id_for_hex!r}\n"
            f"other_hex = {raw_hex!r}\n"
            f"debug_name = 'debug-{obs_alias}.json'\n"
            f"assert {id_alias} == 5\n"
            f"assert literal_hex == {raw_id_for_hex!r}\n"
            f"print(json.dumps({{'v': {id_alias}, 'hex_len': len(literal_hex), 'debug': debug_name}}))\n"
        )
        params = dict(
            threadId=self.h.thread_id,
            turnId='turn',
            callId='raw-inline-fidelity',
            tool='program_run',
            arguments=dict(source=source, timeout_s=10, detail='full'),
        )
        wire = self.h.handle_tool(params)
        self.assertTrue(wire['success'], wire)
        shown = json.loads(wire['contentItems'][0]['text'])
        self.assertEqual(shown['status'], 'succeeded', shown.get('stderr_text'))
        job = self.store.states('program')[-1]
        main_py = (self.h.programs.root / job['job_id'] / 'main.py').read_text()
        self.assertEqual(main_py, source)
        self.assertEqual(self.store.resolve_artifact(job['source_ref'])[1].decode(), source)
        self.assertEqual(job['source_sha256'], hashlib.sha256(source.encode()).hexdigest())
        self.assertNotIn('_raw_inline_source', job['correlation'])
        restored_args = self.h.actor_output.restore(dict(source=source, timeout_s=10))
        self.assertEqual(job['input_digest'], digest(restored_args))

        # 幂等重放相同源码直接返回原状态
        replay = self.h.handle_tool(params)
        self.assertTrue(replay['success'])
        self.assertEqual(json.loads(replay['contentItems'][0]['text'])['job_id'], shown['job_id'])

        # 同 callId 提交 restore() 后完全相同但原文不同的源码：handle_tool 层拦截 immutable_id_content_conflict，runner.run 层由 source_sha256 拦截 program_id_content_conflict
        colliding_source = source.replace(f"{id_alias} = 5", f"{raw_id_for_hex} = 5").replace(
            f"assert {id_alias} == 5", f"assert {raw_id_for_hex} == 5"
        ).replace(f"'v': {id_alias}", f"'v': {raw_id_for_hex}")
        self.assertNotEqual(colliding_source, source)
        self.assertEqual(
            self.h.actor_output.restore(dict(source=colliding_source, timeout_s=10)),
            restored_args,
        )
        with self.assertRaisesRegex(ValueError, 'immutable_id_content_conflict'):
            self.h.handle_tool(dict(params, arguments=dict(source=colliding_source, timeout_s=10)))
        with self.assertRaisesRegex(ValueError, 'program_id_content_conflict'):
            self.h.programs.run(
                restored_args,
                dict(
                    thread_id=self.h.thread_id,
                    turn_id='turn',
                    tool_call_id='raw-inline-fidelity',
                    _raw_inline_source=colliding_source,
                ),
            )

    def test_preflight_timeout_capping_and_low_budget_readonly_exemption(self):
        from unittest.mock import Mock

        obs_id = self.h.latest['observation_id']
        seen_timeouts = []

        def _preview_batch(action):
            seen_timeouts.append(action.get('timeout_s'))
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
                plan_id=f'native-pf-{len(seen_timeouts)}',
                planned_native_steps=1,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={},
            )

        self.robot.preview_batch = Mock(side_effect=_preview_batch)
        self.robot.execute = Mock(return_value=dict(
            status='completed',
            reason=None,
            executed_native_steps=1,
            completed_native_steps=1,
            batch_status='completed',
            batch_cursor=1,
            completed_stage_ids=['g'],
            pending_stage_ids=[],
            stage_results={'g': dict(status='completed', completed_steps=1)},
            terminal_observation=self.robot.observe(),
        ))
        stage = [dict(id='g', kind='gripper', arm='left', goal_normalized=0.0)]

        # 1) 剩余 60s 时 preflight + auto_execute_if_clean 自动裁剪默认超时并通过准入执行
        src_auto = (
            "from ni_rua_client import NiRuaClient\n"
            "res = NiRuaClient().call('robot_action', {"
            f"'mode': 'preflight', 'auto_execute_if_clean': True, 'observation_id': {obs_id!r}, 'stages': {stage!r}"
            "})\n"
            "assert res['kind'] == 'action', res\n"
        )
        job_auto = self.h.programs.run(
            dict(source=src_auto, timeout_s=60, allow_direct_actions=True),
            dict(thread_id=self.h.thread_id, turn_id='t', tool_call_id='pf-cap-60'),
        )
        self.assertEqual(job_auto['status'], 'succeeded', job_auto.get('stderr_text'))
        self.assertEqual(self.robot.execute.call_count, 1)
        self.assertLessEqual(seen_timeouts[-1], 49.0)
        self.assertGreater(seen_timeouts[-1], 40.0)

        # 2) 低剩余时间 (6.0s) 下：preflight + auto_execute_if_clean 拒收，纯 preflight 享有只读豁免
        obs_id_now = self.h.latest['observation_id']
        src_low = (
            "import json\n"
            "from ni_rua_client import NiRuaClient\n"
            "c = NiRuaClient()\n"
            f"ro = c.call('robot_action', {{'mode': 'preflight', 'observation_id': {obs_id_now!r}, 'stages': {stage!r}}})\n"
            f"auto_res = c.call('robot_action', {{'mode': 'preflight', 'auto_execute_if_clean': True, 'observation_id': {obs_id_now!r}, 'stages': {stage!r}}})\n"
            "print(json.dumps({'ro_kind': ro['kind'], 'auto_kind': auto_res['kind'], 'auto_err': auto_res['data']['code']}))\n"
        )
        with patch('ni_rua.runtime.programs.runner.program_remaining', return_value=6.0):
            job_low = self.h.programs.run(
                dict(source=src_low, timeout_s=20, allow_direct_actions=True),
                dict(thread_id=self.h.thread_id, turn_id='t', tool_call_id='pf-low-6'),
            )
        self.assertEqual(job_low['status'], 'succeeded', job_low.get('stderr_text'))
        self.assertEqual(
            json.loads(job_low['stdout_text'].strip()),
            {'ro_kind': 'plan', 'auto_kind': 'error', 'auto_err': 'program_time_insufficient_for_action'},
        )
        self.assertEqual(self.robot.execute.call_count, 1)

    def test_reviewed_action_validates_raw_presets_preserves_clean_downgrade_and_passthrough_fields(self):
        from unittest.mock import Mock

        obs_id = self.h.latest['observation_id']
        reject_safety = {'flag': False}

        def _preview_batch(action):
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
            out = dict(
                status='planned',
                plan_id='native-rev-' + str(action.get('action_id')),
                planned_native_steps=2,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                stage_costs={},
            )
            if reject_safety['flag']:
                out['motion_safety_checks'] = [
                    dict(stage_id=action['stages'][0]['id'], enabled=True, status='reject', start_state_source='initial_measurement')
                ]
            return out

        self.robot.preview_batch = Mock(side_effect=_preview_batch)
        self.robot.execute = Mock(return_value=dict(
            status='completed',
            reason=None,
            executed_native_steps=2,
            completed_native_steps=2,
            batch_status='completed',
            batch_cursor=2,
            batch_revision=0,
            completed_stage_ids=['g'],
            pending_stage_ids=[],
            stage_results={'g': dict(status='completed', completed_steps=2)},
            terminal_observation=self.robot.observe(),
        ))
        stage = [dict(id='g', kind='gripper', arm='left', goal_normalized=0.0)]

        with patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=(dict(status='unavailable'), [])):
            # 先创建一个干净缓存计划 p1，以及一个带安全阻断的缓存计划 p2
            p_clean = self.h.invoke(
                'robot_action',
                dict(mode='preflight', observation_id=obs_id, stages=stage),
                dict(thread_id=self.h.thread_id, turn_id='t', tool_call_id='seed-clean'),
                role='actor',
            ).data
            reject_safety['flag'] = True
            p_risky = self.h.invoke(
                'robot_action',
                dict(mode='preflight', observation_id=obs_id, stages=stage),
                dict(thread_id=self.h.thread_id, turn_id='t', tool_call_id='seed-risky'),
                role='actor',
            ).data
            reject_safety['flag'] = False

            # (a) 受审程序对 execute 夹带非法预设、错误 expected_revision 及 direct 非法预设均在审核前拦截（零审核挂起、零物理调用、零熔断）
            bad_src = (
                "import json\n"
                "from ni_rua_client import NiRuaClient\n"
                "c = NiRuaClient()\n"
                f"r_cmd = c.call('robot_action', {{'mode': 'execute', 'plan_id': {p_clean['plan_id']!r}, 'grip_guard_preset': 'zzz'}})\n"
                f"r_rev = c.call('robot_action', {{'mode': 'execute', 'plan_id': {p_clean['plan_id']!r}, 'expected_revision': 37}})\n"
                f"r_pre = c.call('robot_action', {{'mode': 'direct', 'observation_id': {obs_id!r}, 'stages': {stage!r}, 'grip_guard_preset': 'zzz'}})\n"
                "print(json.dumps({'cmd': r_cmd['data']['code'], 'rev': r_rev['data']['code'], 'pre_path': r_pre['data']['field_path'], 'pre_cat': r_pre['data']['error_category']}))\n"
            )
            bad_job = self.h.programs.run(
                dict(source=bad_src, timeout_s=20, allow_direct_actions=False),
                dict(thread_id=self.h.thread_id, turn_id='t', tool_call_id='rev-bad-inputs'),
            )
            self.assertEqual(bad_job['status'], 'succeeded', bad_job.get('stderr_text'))
            self.assertEqual(bad_job['review_decisions'], {})
            self.assertEqual(self.robot.execute.call_count, 0)
            self.assertEqual(self.h.internal_runtime_errors, [])
            bad_out = json.loads(bad_job['stdout_text'].strip())
            self.assertEqual(bad_out['cmd'], 'cached_batch_commands_cannot_be_changed')
            self.assertEqual(bad_out['rev'], 'batch_plan_revision_changed_since_review')
            self.assertEqual(bad_out['pre_path'], '/grip_guard_preset')
            self.assertEqual(bad_out['pre_cat'], 'input_error')

            # (b) 受审程序对含阻断风险的缓存计划调用 execute_if_clean：审核前直接返回零步零图降级计划，不挂起审核、不退化为无条件 execute
            risky_src = (
                "import json\n"
                "from ni_rua_client import NiRuaClient\n"
                "c = NiRuaClient()\n"
                f"res = c.call('robot_action', {{'mode': 'execute_if_clean', 'plan_id': {p_risky['plan_id']!r}}})\n"
                "imgs = [x for x in res.get('contentItems', []) if x.get('type') == 'inputImage']\n"
                "print(json.dumps({'kind': res['kind'], 'plan_id': res['data']['plan_id'], 'downgraded': res['data'].get('downgraded_from_execute_if_clean'), 'reasons': res['data'].get('downgrade_reasons'), 'imgs': len(imgs)}))\n"
            )
            risky_job = self.h.programs.run(
                dict(source=risky_src, timeout_s=20, allow_direct_actions=False),
                dict(thread_id=self.h.thread_id, turn_id='t', tool_call_id='rev-risky-clean'),
            )
            self.assertEqual(risky_job['status'], 'succeeded', risky_job.get('stderr_text'))
            self.assertEqual(risky_job['review_decisions'], {})
            self.assertEqual(self.robot.execute.call_count, 0)
            self.assertEqual(risky_job['action_summary']['status_counts'], {'not_executed': 1})
            risky_out = json.loads(risky_job['stdout_text'].strip())
            self.assertEqual(risky_out['kind'], 'plan')
            self.assertEqual(risky_out['plan_id'], p_risky['plan_id'])
            self.assertTrue(risky_out['downgraded'])
            self.assertIn('motion_safety_reject', risky_out['reasons'])
            self.assertEqual(risky_out['imgs'], 0)

            # (c) 清洁请求（兼容顶层历史别名 tolerance_preset='fine'）获批恢复时透传 image_delivery='none'、expected_cursor 与 expected_revision
            ok_src = (
                "from ni_rua_client import NiRuaClient\n"
                "c = NiRuaClient()\n"
                f"res = c.call('robot_action', {{'mode': 'execute_if_clean', 'observation_id': {obs_id!r}, 'stages': {stage!r}, 'tolerance_preset': 'fine', 'image_delivery': 'none'}})\n"
                "assert res['kind'] == 'action' and res['data']['receipt']['status'] == 'completed'\n"
            )
            paused = self.h.programs.run(
                dict(source=ok_src, timeout_s=20, allow_direct_actions=False),
                dict(thread_id=self.h.thread_id, turn_id='t', tool_call_id='rev-clean-approve'),
            )
            self.assertEqual(paused['status'], 'awaiting_review', paused)
            resumed = self.h.programs.run(
                dict(
                    operation='resume',
                    job_id=paused['job_id'],
                    review_id=paused['pending_review']['review_id'],
                    decision='approve',
                ),
                dict(thread_id=self.h.thread_id, turn_id='t2', tool_call_id='rev-clean-resume'),
            )
            self.assertEqual(resumed['status'], 'succeeded', resumed.get('stderr_text'))
            self.assertEqual(self.robot.execute.call_count, 1)
            exec_action = self.robot.execute.call_args.args[0]
            self.assertEqual(exec_action.get('image_delivery'), 'none')
            self.assertEqual(exec_action.get('batch_expected_cursor'), 0)
            self.assertEqual(exec_action.get('batch_expected_revision'), 0)

    def test_incomplete_latch_unlock_follows_clock_and_captured_live_and_client_mirrors_broker(self):
        from ni_rua.core.action_validation import embodiment_or_default

        rt_profile = replace(embodiment_or_default(None), clock='real_time')

        ident = dict(run_id='run-latch', episode_id='ep-1', env_instance_id='env-1', env_generation=1)
        temp = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(temp) / 'run', ident)
        self.addCleanup(store.close)

        state = {'latest': 'obs-1', 'captured_live': False, 'stop_confirmed': True, 'res_obs': 'obs-2'}

        def fake_dispatch(name, args, correlation):
            if name == 'robot_observe':
                oid = 'obs-live-3' if state['captured_live'] else state['latest']
                if state['captured_live']:
                    state['latest'] = oid
                resp = ToolResponse(
                    [dict(type='inputText', text=json.dumps({'observation_id': oid}))],
                    data=dict(observation_id=oid, cameras=[], **ident),
                    kind='observation',
                )
                resp.captured_live = state['captured_live']
                return resp
            if name == 'robot_wait':
                if state.get('complete_next'):
                    rcpt = dict(action_id='w-ok', status='completed', physics_advanced=True, executed_native_steps=1, result_observation_id=state['latest'], stop_confirmed=True)
                else:
                    if state['res_obs']:
                        state['latest'] = state['res_obs']
                    rcpt = dict(
                        action_id='w-inc',
                        status='stopped',
                        reason='contact_stop',
                        physics_advanced=True,
                        executed_native_steps=1,
                        result_observation_id=state['res_obs'],
                        stop_confirmed=state['stop_confirmed'],
                    )
                return ToolResponse([dict(type='inputText', text='{}')], data=dict(receipt=rcpt), kind='action')
            raise AssertionError(name)

        host = SimpleNamespace(
            store=store,
            lock=threading.RLock(),
            cancel=threading.Event(),
            active_context=None,
            dispatch=fake_dispatch,
        )
        base_srv = replace(
            program_services(host),
            sandbox_enabled=False,
            observation=lambda: dict(observation_id=state['latest'], observation_profile='actor_rgb_calibrated_v1'),
            current_identity=lambda: dict(ident),
        )

        # 1) 实时钟本体（clock='real_time'）：缓存复用的 robot_observe (captured_live=False) 绝不解锁，也不会抹掉已调用的 acknowledge_failure；唯有 captured_live=True 解锁
        rt_runner = ProgramRunner(replace(base_srv, embodiment=rt_profile))
        self.addCleanup(rt_runner.close)
        rt_src = (
            "import json\n"
            "from ni_rua_client import NiRuaClient\n"
            "c = NiRuaClient()\n"
            "c.call('robot_wait', {'observation_id': 'obs-1', 'native_steps': 1})\n"
            "# 缓存观测 (captured_live=False) 不得清锁\n"
            "c.observe(camera_id='cam_head', roi=[0, 0, 4, 4])\n"
            "assert c._unacknowledged_failed_action is not None\n"
            "blocked = False\n"
            "try:\n"
            "    c.call('robot_wait', {'observation_id': 'obs-2', 'native_steps': 1})\n"
            "except RuntimeError as exc:\n"
            "    blocked = getattr(exc, 'code', None) == 'program_blind_continuation_blocked'\n"
            "# 显式 acknowledge_failure 后再调缓存观测，不得被客户端本地私自清掉确认状态\n"
            "c.acknowledge_failure()\n"
            "c.observe(camera_id='cam_head', roi=[0, 0, 4, 4])\n"
            "assert c._failure_acknowledged is True\n"
            "print(json.dumps({'blocked': blocked}))\n"
        )
        rt_job = rt_runner.run(
            dict(source=rt_src, allow_direct_actions=True, timeout_s=20),
            dict(thread_id='t', turn_id='u', tool_call_id='rt-clock-gate'),
        )
        self.assertEqual(rt_job['status'], 'succeeded', rt_job.get('stderr_text'))
        self.assertEqual(json.loads(rt_job['stdout_text'].strip()), {'blocked': True})

        # 实时钟下真实新采帧 (captured_live=True) 解锁并更新终帧
        state.update(latest='obs-1', captured_live=True, complete_next=False)
        rt_live_src = (
            "from ni_rua_client import NiRuaClient\n"
            "c = NiRuaClient()\n"
            "c.call('robot_wait', {'observation_id': 'obs-1', 'native_steps': 1})\n"
            "c.observe()\n"
            "assert c._unacknowledged_failed_action is None\n"
        )
        rt_live_job = rt_runner.run(
            dict(source=rt_live_src, allow_direct_actions=True, timeout_s=20),
            dict(thread_id='t', turn_id='u', tool_call_id='rt-clock-live'),
        )
        self.assertEqual(rt_live_job['status'], 'succeeded', rt_live_job.get('stderr_text'))
        self.assertEqual(rt_live_job['final_observation_id'], 'obs-live-3')

        # 2) 仿真步进钟（默认 ARX-X5, clock='simulated_steps'）：三方 ID 一致且 stop_confirmed=True 时允许复用终帧解锁；stop_confirmed=False 时拒绝缓存解锁
        sim_runner = ProgramRunner(base_srv)
        self.addCleanup(sim_runner.close)
        state.update(latest='obs-1', captured_live=False, stop_confirmed=True, res_obs='obs-2', complete_next=False)
        sim_ok_src = (
            "from ni_rua_client import NiRuaClient\n"
            "c = NiRuaClient()\n"
            "c.call('robot_wait', {'observation_id': 'obs-1', 'native_steps': 1})\n"
            "c.observe(camera_id='cam_head', roi=[0, 0, 4, 4])\n"
            "assert c._unacknowledged_failed_action is None\n"
        )
        sim_ok = sim_runner.run(
            dict(source=sim_ok_src, allow_direct_actions=True, timeout_s=20),
            dict(thread_id='t', turn_id='u', tool_call_id='sim-3way-ok'),
        )
        self.assertEqual(sim_ok['status'], 'succeeded', sim_ok.get('stderr_text'))

        state.update(latest='obs-1', captured_live=False, stop_confirmed=False, res_obs='obs-2', complete_next=False)
        sim_unconfirmed_src = (
            "from ni_rua_client import NiRuaClient\n"
            "c = NiRuaClient()\n"
            "c.call('robot_wait', {'observation_id': 'obs-1', 'native_steps': 1})\n"
            "c.observe(camera_id='cam_head', roi=[0, 0, 4, 4])\n"
            "assert c._unacknowledged_failed_action is not None\n"
        )
        sim_unconfirmed = sim_runner.run(
            dict(source=sim_unconfirmed_src, allow_direct_actions=True, timeout_s=20),
            dict(thread_id='t', turn_id='u', tool_call_id='sim-3way-unconfirmed'),
        )
        self.assertEqual(sim_unconfirmed['status'], 'succeeded', sim_unconfirmed.get('stderr_text'))

    def test_client_ensure_completed_zero_step_not_executed_and_non_action_guard(self):
        from ni_rua.runtime.programs.ni_rua_client import ActionIncompleteError, NiRuaClient

        ok_action = {'kind': 'action', 'success': True, 'data': {'receipt': {'status': 'completed'}}}
        self.assertIs(NiRuaClient.ensure_completed(ok_action), ok_action)

        # 含同名 data.receipt.status='completed' 的非动作响应绝不可误判通过，也不可把 status 标为 completed
        spoofed = {'kind': 'data', 'success': True, 'data': {'receipt': {'status': 'completed'}}}
        with self.assertRaises(ActionIncompleteError) as spoof_ctx:
            NiRuaClient.ensure_completed(spoofed)
        self.assertIs(spoof_ctx.exception.response, spoofed)
        self.assertEqual(spoof_ctx.exception.receipt.get('status'), None)
        self.assertIn('status=unknown', str(spoof_ctx.exception))

        # 普通 preview 计划（无零步降级标记）保持 status=unknown
        plain_plan = {
            'kind': 'plan',
            'success': True,
            'data': {'plan_id': 'p-plain', 'status': 'planned', 'physics_advanced': False, 'executed_native_steps': 0},
        }
        with self.assertRaises(ActionIncompleteError) as plain_ctx:
            NiRuaClient.ensure_completed(plain_plan)
        self.assertIn('status=unknown', str(plain_ctx.exception))

        # 明确零步安全降级计划报 status=not_executed，保留 .response、plan_id、downgrade_reasons 与 next_action
        downgraded_plan = {
            'kind': 'plan',
            'success': True,
            'data': {
                'plan_id': 'p-dg',
                'status': 'planned',
                'downgraded_from_execute_if_clean': True,
                'downgrade_reasons': ['motion_safety_reject'],
                'physics_advanced': False,
                'executed_native_steps': 0,
                'next_action': {'mode': 'execute', 'plan_id': 'p-dg', 'expected_cursor': 0, 'expected_revision': 0},
            },
        }
        with self.assertRaises(ActionIncompleteError) as dg_ctx:
            NiRuaClient.ensure_completed(downgraded_plan)
        exc = dg_ctx.exception
        self.assertIs(exc.response, downgraded_plan)
        self.assertEqual(exc.plan_id, 'p-dg')
        self.assertEqual(exc.receipt['status'], 'not_executed')
        self.assertEqual(exc.receipt['downgrade_reasons'], ['motion_safety_reject'])
        self.assertEqual(exc.receipt['next_action']['plan_id'], 'p-dg')
        self.assertIn('status=not_executed', str(exc))
        self.assertIn('motion_safety_reject', str(exc))

        # 明确零步规划失败同样报 status=not_executed
        failed_plan = {
            'kind': 'plan',
            'success': True,
            'data': {
                'plan_id': 'p-fail',
                'status': 'failed',
                'reason': 'path_infeasible',
                'failed_stage_id': 'grasp',
                'physics_advanced': False,
                'executed_native_steps': 0,
            },
        }
        with self.assertRaises(ActionIncompleteError) as fail_ctx:
            NiRuaClient.ensure_completed(failed_plan)
        self.assertEqual(fail_ctx.exception.receipt['status'], 'not_executed')
        self.assertEqual(fail_ctx.exception.receipt['reason'], 'path_infeasible')
        self.assertIn('failed_stage_id=grasp', str(fail_ctx.exception))



if __name__ == "__main__":
    unittest.main()
