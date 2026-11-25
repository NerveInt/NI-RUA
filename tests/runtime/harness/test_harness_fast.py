"""
- test_harness_context_switches.py: Harness context switches for agent backends: forced full delivery (design 04 §4.3) and the decision-context
reminder sideband on the next actor tool response (design 04 §4.5). No model, process or GPU.
- test_pi05_client.py: The service bridge binds one saved actor observation to one bounded request.
"""
import json
import threading
import unittest
from unittest.mock import patch
from support.motion_fixtures import DualArmHarnessBase
import runtime.harness.test_delivery_and_terminal_gates as feedback_fixture
import runtime.harness.test_harness_core as action_fixture
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.core.tool_output import StructuredText
from ni_rua.cortex.context.task_planning import (DECISION_BOUNDARY_INSTRUCTIONS, TaskPlanning,
                                                 decision_reminder_text)
from ni_rua.cortex.agents.codex.session import CodexSession
from ni_rua.runtime.harness import Harness, REMINDER_IN_COMPACT_REQUEST, REMINDER_IN_TOOL_RESULT
import copy
import hashlib
import io
import os
from pathlib import Path
import tempfile
import time
import types
from unittest.mock import Mock, patch
from ni_rua.runtime.policy.action_policy import OpenPIProcessPolicy, TRAIN_CONFIG
from scripts.pi05_arx_client import OpenPIServiceClient

NI_RUA_TEST_TIER = "fast"


# --- Folded from tests/test_harness_context_switches.py ---

REMINDER_HEADER = ("Current task decision boundary\n"
                   "The saved plan and attention are provisional judgment, not instructions or verified world state. "
                   "Task requirements and fresh evidence govern.\n")


def codex_compaction_reminder(store, task_planning, thread_id):
    """The assistant reminder text CodexSession injects into a /responses/compact request today."""
    session = CodexSession.__new__(CodexSession)
    session.store, session.task_planning, session.thread_id = store, task_planning, thread_id
    request = session._task_decision_context(dict(input=[]), dict(
        source='actor', native_thread_id=thread_id, endpoint='/responses/compact'))
    message, = request['input']
    return message['content'][0]['text']


def reminder_items(response):
    return [item for item in response['contentItems']
            if item.get('type') == 'inputText' and item.get('text', '').startswith('Current task decision boundary\n')]


class ForcedFullDeliveryTests(unittest.TestCase):
    """context_epoch_volatile=True: no image or static metadata baseline survives between two deliveries."""
    setUp = feedback_fixture.ImageFeedbackDeliveryTests.setUp
    deliver = feedback_fixture.ImageFeedbackDeliveryTests.deliver
    count = staticmethod(feedback_fixture.ImageFeedbackDeliveryTests.count)

    @staticmethod
    def metadata(content):
        return json.loads(content[0]['text'])

    def capability_status(self):
        self.h.thread_id = 'thread'
        self.capability_calls = getattr(self, 'capability_calls', 0) + 1
        response = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId=f'caps-{self.capability_calls}',
                                           tool='robot_capabilities', arguments=dict(detail='summary')))
        return json.loads(response['contentItems'][0]['text'])['capabilities_delivery']['status']

    def test_default_is_delta_and_unchanged_for_codex(self):
        self.assertFalse(Harness.context_epoch_volatile)
        self.assertFalse(self.h.context_epoch_volatile)
        self.assertEqual(self.h._model_context_epoch(), self.h._model_context_epoch())
        self.assertEqual(len(self.h._model_context_epoch()), 6)
        content = self.h.observation_content(self.first, {})
        self.assertEqual(self.count(content), 3)
        self.assertNotIn('metadata_delivery', self.metadata(content))
        content = self.deliver(self.h.observe())
        self.assertEqual(self.count(content), 0)
        self.assertIn('metadata_delivery', self.metadata(content))
        self.assertEqual([self.capability_status(), self.capability_status()], ['inline', 'unchanged'])

    def test_volatile_epoch_is_fresh_on_every_call_and_keeps_the_delta_fields(self):
        self.h.context_epoch_volatile = True
        first, second = self.h._model_context_epoch(), self.h._model_context_epoch()
        self.assertNotEqual(first, second)
        self.assertEqual(first[:6], second[:6])
        self.assertEqual(len(first), 7)
        self.h.context_epoch_volatile = False
        self.assertEqual(self.h._model_context_epoch(), first[:6])

    def test_volatile_epoch_forces_every_image_and_complete_metadata_on_every_delivery(self):
        self.h.context_epoch_volatile = True
        self.assertEqual(self.count(self.h.observation_content(self.first, {})), 3)
        for _ in range(3):
            content = self.deliver(self.h.observe())
            self.assertEqual(self.count(content), 3)
            metadata = self.metadata(content)
            self.assertEqual(metadata['image_feedback']['omitted_camera_ids'], [])
            self.assertNotIn('metadata_delivery', metadata)
            self.assertIn('robot_state', metadata)
            self.assertIn('camera_calibration', metadata)
        self.assertEqual([self.capability_status() for _ in range(3)], ['inline'] * 3)
        # Back to delta: baselines form again from the next delivery.
        self.h.context_epoch_volatile = False
        self.deliver(self.h.observe())
        self.assertEqual(self.count(self.deliver(self.h.observe())), 0)


class DecisionReminderSidebandTests(unittest.TestCase):
    def setUp(self):
        DualArmHarnessBase.setUp(self)
        self.h.thread_id = 'thread'
        self.calls = 0

    def call(self, tool='robot_capabilities', arguments=None):
        self.calls += 1
        return self.h.handle_tool(dict(threadId='thread', turnId='turn', callId=f'call-{self.calls}', tool=tool,
                                       arguments=dict(detail='full') if arguments is None else arguments))

    def test_reply_compaction_switches_gate_the_actor_reply_only(self):
        """Off: the compaction is not invoked. On: the delivered items are the compaction of the presented ones."""
        import dataclasses
        from ni_rua.core.presentation import reply_compaction
        from support.reply_defaults import with_reply_features
        seen = []
        self.assertTrue(self.h.config.features.reply_same_as and self.h.config.features.reply_float32)  # the defaults
        self.h.config = with_reply_features(self.h.config)

        def spy(items, **switches):
            seen.append((copy.deepcopy(items), switches))
            return reply_compaction.compact_content_items(items, **switches)

        with patch('ni_rua.runtime.harness.compact_content_items', spy):
            self.assertFalse(self.h.config.features.reply_same_as or self.h.config.features.reply_float32)
            self.call()
            self.assertEqual(seen, [])
            self.h.config = dataclasses.replace(self.h.config, features=dataclasses.replace(
                self.h.config.features, reply_same_as=True, reply_float32=True))
            response = self.call(arguments=dict(detail='full'))
        (items, switches), = seen
        self.assertEqual(switches, dict(same_as=True, float32=True))
        self.assertEqual(response['contentItems'][:len(items)],
                         reply_compaction.compact_content_items(items, same_as=True, float32=True))
        self.assertTrue(response['success'])
        # a failing compaction delivers the presented reply as is (it must never hide a receipt)
        with patch('ni_rua.runtime.harness.compact_content_items', side_effect=RuntimeError('boom')):
            fallback = self.call(arguments=dict(detail='full'))
        self.assertTrue(fallback['success'])
        self.assertTrue(json.loads(fallback['contentItems'][0]['text']))

    def test_reply_switches_apply_in_one_order_and_every_combination_delivers_whole_pointers(self):
        """static notes out -> layout (profile, budget) -> (detail entries, compact encoding, same_as, float32) on the
        text items of the layout."""
        import dataclasses
        import itertools
        import core.presentation.test_actor_payload as reply_fixture
        from ni_rua.core.contracts import canonical
        from ni_rua.core.presentation import reply_budget, reply_compaction, reply_encoding

        items = reply_fixture.ReplyShapingOrderTests().reply()
        result = json.loads(items[0]['text'])
        noted = reply_fixture.ReplyStaticOnceTests().result()
        result['action_budget'] = noted['action_budget']
        result['receipt'].update(collision_scope=noted['receipt']['collision_scope'], completed_native_steps=42,
                                 action_budget=noted['receipt']['action_budget'],
                                 motion_safety_check_summary=noted['receipt']['motion_safety_check_summary'])
        items[0] = dict(items[0], text=canonical(result))
        loaded = self.h.tool_registry.loaded_tool_names(self.h, role='actor')
        from support.reply_defaults import action_result, with_reply_features
        base = with_reply_features(self.h.config)
        correlation = dict(thread_id='thread', turn_id='turn', tool_call_id='call', item_id='call')
        budget = 4 * reply_budget.reply_cost_tokens(items) // 2
        self.assertTrue(reply_budget.fit_reply_report(items, budget)[1]['by_reference'])

        plan = self.h.tool_registry.static_once_plan(self.h, role='actor')

        def expected(static, same_as, float32, layout, host, kind, profile, encoding):
            content = reply_compaction.drop_static_once(items, plan) if static else items
            steps = []
            if static:
                steps.append(lambda value: reply_compaction.unify_detail_entries(value, loaded))
            if encoding:
                steps.append(reply_encoding.encode_compact_items)
            if same_as or float32:
                steps.append(lambda value: reply_compaction.compact_content_items(value, same_as=same_as, float32=float32))

            def finish(value):
                for step in steps:
                    value = step(value)
                return value
            if kind == 'action' and (layout or profile != "full"):
                return reply_budget.fit_reply(content, budget if layout and host else None,
                                              compact=finish if steps else None, profile=profile, layout=layout)
            return finish(content)

        # layout = features.budgeted_action_reply; host = the agent host has a known text limit for one tool output
        for static, same_as, float32, layout, host, summary, encoding in itertools.product((False, True), repeat=7):
            profile = "summary_first" if summary else "full"
            self.h.config = dataclasses.replace(base, features=dataclasses.replace(
                base.features, reply_static_once=static, reply_same_as=same_as, reply_float32=float32,
                budgeted_action_reply=layout, action_reply_profile=profile, reply_compact_encoding=encoding))
            self.h.action_reply_budget = (lambda: budget) if host else (lambda: None)
            for kind in ('action', 'observation'):
                with self.subTest(static=static, same_as=same_as, float32=float32, layout=layout, host=host, kind=kind,
                                  profile=profile, encoding=encoding):
                    response = dict(success=True, contentItems=items)
                    shaped = self.h._shape_actor_reply(response, kind, correlation)
                    if not (static or same_as or float32 or encoding or ((layout or summary) and kind == 'action')):
                        self.assertIs(shaped, response)
                    self.assertEqual(shaped['contentItems'],
                                     expected(static, same_as, float32, layout, host, kind, profile, encoding))
                    self.assertEqual(reply_compaction.dangling_same_as(shaped['contentItems']), [])
                    self.assertEqual(sum(item['type'] == 'inputImage' for item in shaped['contentItems']), 3)
                    if layout and host and kind == 'action':
                        self.assertLessEqual(reply_budget.reply_cost_tokens(shaped['contentItems']), budget // 4)
                    if summary and kind == 'action':
                        self.assertNotIn('mmmm', ''.join(item.get('text', '') for item in shaped['contentItems']))
        # Every backend: with budgeted_action_reply and a host without a known limit the reply is laid out by
        # decision value and nothing is replaced; every value is still delivered.
        self.h.config = with_reply_features(base, budgeted_action_reply=True)
        self.h.action_reply_budget = lambda: None
        fitted_events = len(self.events('actor_reply.fitted'))
        ordered = self.h._shape_actor_reply(dict(success=True, contentItems=items), 'action', correlation)['contentItems']
        self.assertEqual(ordered, reply_budget.fit_reply(items, None, layout=True))
        self.assertNotEqual(ordered, items)
        self.assertEqual(action_result(ordered), json.loads(items[0]['text']))
        self.assertEqual([item for item in ordered if item['type'] == 'inputImage'],
                         [items[3], items[4], items[1]])
        self.assertEqual(len(self.events('actor_reply.fitted')), fitted_events)
        # A failed tool response keeps its notes whatever its JSON says.
        self.h.config = with_reply_features(base, reply_static_once=True)
        failed = self.h._shape_actor_reply(dict(success=False, contentItems=items), 'action', correlation)
        self.assertEqual(failed['contentItems'], reply_compaction.unify_detail_entries(items, loaded))
        self.assertIn('"collision_scope"', failed['contentItems'][0]['text'])
        # A full read of stored evidence skips the compact encoding (and the display rounding of lean_v2).
        self.h.config = with_reply_features(base, reply_compact_encoding=True, reply_float32=True)
        exact = self.h._shape_actor_reply(dict(success=True, contentItems=items), 'data', correlation, exact=True)
        self.assertEqual(exact['contentItems'], reply_compaction.compact_content_items(items, float32=True))

    def submit_plan(self):
        response = self.call('task_plan', dict(plan='Task understanding: move the cube. Approach: grasp, lift, verify.',
                                               ledger=[]))
        self.assertTrue(response['success'], response)

    def events(self, kind):
        return [event for event in self.store.events(limit=1000) if event['type'] == kind]

    def test_codex_channel_is_the_default_and_never_gets_the_sideband(self):
        self.assertEqual(Harness.decision_reminder_channel, REMINDER_IN_COMPACT_REQUEST)
        self.submit_plan()
        before = len(self.call()['contentItems'])
        self.h.model_context_reset('relay_context_discontinuity')
        self.assertFalse(self.h.decision_reminder_due)
        response = self.call()
        self.assertEqual(len(response['contentItems']), before)
        self.assertEqual(reminder_items(response), [])
        reset, = self.events('model_context.reset')
        self.assertEqual(reset['payload'], dict(reason='relay_context_discontinuity',
                                                decision_reminder_channel=REMINDER_IN_COMPACT_REQUEST,
                                                decision_reminder_due=False))
        self.assertEqual(self.events('model_context.decision_reminder_delivered'), [])

    def test_reminder_is_appended_once_to_the_next_response_with_the_codex_text(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        plain = self.call()
        self.assertEqual(reminder_items(plain), [])
        self.h.model_context_reset('native_context_compaction')
        self.assertTrue(self.h.decision_reminder_due)
        reminded = self.call()
        self.assertFalse(self.h.decision_reminder_due)
        self.assertEqual(len(reminded['contentItems']), len(plain['contentItems']) + 1)
        self.assertEqual(reminded['contentItems'][:-1], plain['contentItems'])
        self.assertEqual(reminded['success'], plain['success'])
        item = reminded['contentItems'][-1]
        self.assertEqual(set(item), {'type', 'text'})
        self.assertEqual(item['type'], 'inputText')
        # The same text and snapshot Codex carries in its compaction request, after the same egress scrub.
        codex_text = codex_compaction_reminder(self.store, self.h.task_planning, 'thread')
        self.assertEqual(decision_reminder_text(self.h.task_planning.decision_context(self.store.identity)),
                         codex_text)
        self.assertEqual(item['text'], ActorPresentation(self.store).present_text(codex_text))
        self.assertTrue(item['text'].startswith(REMINDER_HEADER + 'Decision boundary: '
                                                + DECISION_BOUNDARY_INSTRUCTIONS + '\n'))
        snapshot = json.loads(item['text'].split('\n')[-1])
        self.assertEqual((snapshot['revision'], snapshot['plan']),
                         (1, 'Task understanding: move the cube. Approach: grasp, lift, verify.'))
        self.assertEqual(snapshot['attention'], [])
        # The session's second presentation scrub (MCP bridge, as Codex's RecordedClient) leaves it unchanged.
        self.assertEqual(ActorPresentation(self.store).present(item), item)
        # Consumed exactly once.
        self.assertEqual(reminder_items(self.call()), [])
        delivered, = self.events('model_context.decision_reminder_delivered')
        self.assertEqual(delivered['payload'], dict(included=True, plan_revision=1))
        self.assertEqual(delivered['correlation']['tool_call_id'], f'call-{self.calls - 1}')
        reset, = self.events('model_context.reset')
        self.assertEqual(reset['payload']['decision_reminder_due'], True)

    def test_required_plan_missing_reminds_to_submit_it_like_codex(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.h.task_planning = TaskPlanning(self.store, required=True)
        self.h.model_context_reset('agent_resume_after_unclean_exit')
        item, = reminder_items(self.call())
        codex_text = codex_compaction_reminder(self.store, self.h.task_planning, 'thread')
        self.assertEqual(item['text'], ActorPresentation(self.store).present_text(codex_text))
        self.assertEqual(item['text'], REMINDER_HEADER + 'No plan is stored for this episode. '
                                                         'Submit your initial task_plan before physical actions.')

    def test_optional_planning_without_plan_adds_nothing_and_clears_the_flag(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.assertFalse(self.h.task_planning.required)
        before = len(self.call()['contentItems'])
        self.h.model_context_reset('relay_context_discontinuity')
        self.assertEqual(len(self.call()['contentItems']), before)
        self.assertFalse(self.h.decision_reminder_due)
        delivered, = self.events('model_context.decision_reminder_delivered')
        self.assertEqual(delivered['payload'], dict(included=False, plan_revision=None))

    def test_tool_failure_before_a_response_keeps_the_reminder_for_the_next_response(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        self.h.model_context_reset('relay_context_discontinuity')
        with patch.object(self.h.actor_output, 'present', side_effect=ValueError('synthetic presentation failure')):
            with self.assertRaisesRegex(ValueError, 'synthetic presentation failure'):
                self.call()
        self.assertTrue(self.h.decision_reminder_due)
        self.assertEqual(len(reminder_items(self.call())), 1)
        self.assertFalse(self.h.decision_reminder_due)

    def test_failed_tool_response_still_carries_the_reminder(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        self.h.model_context_reset('relay_context_discontinuity')
        with patch.object(self.h.actor_output, '_present_part', side_effect=ValueError('synthetic display failure')):
            response = self.call()
        self.assertFalse(response['success'])
        self.assertEqual(len(reminder_items(response)), 1)
        self.assertIs(response['contentItems'][-1], reminder_items(response)[0])

    def test_reminder_build_failure_returns_the_response_unchanged_and_stays_due(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        plain = self.call()
        self.h.model_context_reset('relay_context_discontinuity')
        with patch.object(self.h.task_planning, 'decision_context', side_effect=RuntimeError('store fault')):
            response = self.call()
        self.assertEqual(response, plain)
        self.assertTrue(self.h.decision_reminder_due)
        failed, = self.events('model_context.decision_reminder_failed')
        self.assertEqual(failed['payload'], dict(error_type='RuntimeError'))
        self.assertEqual(len(reminder_items(self.call())), 1)

    def test_parallel_responses_consume_one_reminder(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.submit_plan()
        self.h.model_context_reset('relay_context_discontinuity')
        barrier, results = threading.Barrier(8), []

        def deliver(index):
            barrier.wait()
            results.append(self.h._deliver_decision_reminder(
                dict(success=True, contentItems=[dict(type='inputText', text=str(index))]),
                dict(tool_call_id=f'parallel-{index}')))
        workers = [threading.Thread(target=deliver, args=(index,)) for index in range(8)]
        for worker in workers:
            worker.start()
        for worker in workers:
            worker.join(10)
        self.assertEqual(len(results), 8)
        self.assertEqual(sum(len(reminder_items(result)) for result in results), 1)
        self.assertEqual(len(self.events('model_context.decision_reminder_delivered')), 1)

    def test_non_envelope_and_unwritable_store_never_raise(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.h.task_planning = TaskPlanning(self.store, required=True)
        self.h.model_context_reset(None)
        self.assertEqual(self.events('model_context.reset')[-1]['payload']['reason'], 'unspecified')
        legacy = dict(success=True)
        self.assertIs(self.h._deliver_decision_reminder(legacy, {}), legacy)
        self.assertTrue(self.h.decision_reminder_due)
        with patch.object(self.store, 'event', side_effect=RuntimeError('read-only store')):
            self.h.model_context_reset('relay_context_discontinuity')
            response = self.h._deliver_decision_reminder(dict(success=True, contentItems=[]), {})
        self.assertFalse(self.h.decision_reminder_due)
        self.assertEqual(response['contentItems'], [dict(type='inputText', text=REMINDER_HEADER + (
            'No plan is stored for this episode. Submit your initial task_plan before physical actions.'))])


class ActionReceiptWithReminderTests(unittest.TestCase):
    """A physical action's typed receipt survives every reminder path; the action is never re-sent."""
    setUp = action_fixture.StageLabelExecutionTests.setUp

    def act(self, call_id='depth-check'):
        return self.h.handle_tool(dict(threadId='thread', turnId='turn', callId=call_id, tool='robot_action',
                                       arguments=self.args))

    def capabilities(self):
        return self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='after', tool='robot_capabilities',
                                       arguments=dict(detail='summary')))

    def receipt_of(self, response):
        for item in response['contentItems']:
            try:
                value = json.loads(item.get('text', ''))
            except ValueError:
                continue
            if isinstance(value, dict) and isinstance(value.get('receipt'), dict):
                return value['receipt']
        self.fail('no receipt in the response')

    def arm(self):
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        planned = self.h.handle_tool(dict(threadId='thread', turnId='turn', callId='plan', tool='task_plan',
                                          arguments=dict(plan='Reach, then inspect the insertion.', ledger=[])))
        self.assertTrue(planned['success'], planned)
        self.h.model_context_reset('relay_context_discontinuity')

    def test_reminder_rides_after_the_receipt(self):
        self.arm()
        response = self.act()
        self.assertTrue(response['success'])
        self.assertEqual(self.receipt_of(response)['executed_native_steps'], 1)
        self.assertEqual(len(reminder_items(response)), 1)
        self.assertIs(response['contentItems'][-1], reminder_items(response)[0])
        self.assertEqual(len(self.dispatched), 1)

    def test_reminder_failure_after_an_executed_action_keeps_the_receipt(self):
        self.arm()
        with patch.object(self.h.task_planning, 'decision_context', side_effect=RuntimeError('planning fault')):
            response = self.act()
        self.assertTrue(response['success'])
        self.assertEqual(self.receipt_of(response)['executed_native_steps'], 1)
        self.assertEqual(reminder_items(response), [])
        self.assertTrue(self.h.decision_reminder_due)
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(len(reminder_items(self.capabilities())), 1)
        self.assertEqual(len(self.dispatched), 1)
        self.assertEqual(self.store.read('action_receipt', 'label-action')['executed_native_steps'], 1)

    def test_display_failure_after_an_executed_action_keeps_the_receipt_and_the_reminder(self):
        self.arm()
        original = self.h.actor_output._present_part

        def fail_display(item, *args, **kwargs):
            if isinstance(item, StructuredText) and item.projection == 'action_display' and set(item.value) == {'receipt'}:
                return original(item, *args, **kwargs)
            raise ValueError('synthetic display failure')
        with patch.object(self.h.actor_output, '_present_part', side_effect=fail_display):
            response = self.act()
        self.assertFalse(response['success'])
        self.assertEqual(self.receipt_of(response)['executed_native_steps'], 1)
        self.assertEqual(len(reminder_items(response)), 1)
        self.assertIs(response['contentItems'][-1], reminder_items(response)[0])
        self.assertFalse(self.h.decision_reminder_due)
        self.assertEqual(len(self.dispatched), 1)



# --- Folded from tests/test_pi05_client.py ---

class ServiceClientCheck(unittest.TestCase):
    def setUp(self):
        self.connection = Mock()
        self.connection.recv.return_value = {"config": TRAIN_CONFIG, "action_shape": [50, 14]}
        self.connect = Mock(return_value=self.connection)
        codec = types.SimpleNamespace(packb=lambda value: value, unpackb=lambda value: value)
        self.modules = patch.dict("sys.modules", {
            "openpi_client": types.SimpleNamespace(msgpack_numpy=codec),
            "websockets": types.ModuleType("websockets"),
            "websockets.sync": types.ModuleType("websockets.sync"),
            "websockets.sync.client": types.SimpleNamespace(connect=self.connect),
        })
        self.modules.start()
        self.addCleanup(self.modules.stop)
        self.request = dict(run_dir="/tmp/actor-run", observation_id="obs-7")
        self.observation = dict(state=[0] * 14, images={"real": "saved-image"}, prompt="public task")
        # The loader reports the canonical run directory (/tmp is a symlink on macOS).
        self.evidence = dict(source_run_dir=str(Path("/tmp/actor-run").resolve()), observation=dict(observation_id="obs-7"),
                             public_instruction="public task")
        self.loader = patch("scripts.pi05_arx_client.load_policy_observation",
                            return_value=(self.observation, self.evidence)).start()
        self.addCleanup(patch.stopall)

    def test_saved_observation_prompt_identity_and_honest_metadata(self):
        client = OpenPIServiceClient("ws://127.0.0.1:8000", timeout=3)
        self.connection.recv.return_value = dict(actions=[[0.] * 14 for _ in range(50)])
        result = client.infer({**self.request, "prompt": "precise instruction"})
        self.loader.assert_called_once_with("/tmp/actor-run", "obs-7")
        self.connection.send.assert_called_once_with(self.observation)
        self.assertEqual(result["observation"], self.evidence["observation"])
        self.assertEqual(result["public_instruction"], "public task")
        self.assertEqual(result["policy_prompt"], "precise instruction")
        self.assertEqual(result["executed_steps"], 0)
        self.assertFalse(result["checkpoint_attested_by_server"])
        self.assertFalse(result["model_loaded_by_bridge"])
        self.assertNotIn("checkpoint_files_verified", result)
        self.assertNotIn("checkpoint_revision", result)
        self.assertEqual(self.connection.recv.call_args.kwargs, {"timeout": 3})
        self.assertEqual(self.connect.call_args.kwargs["open_timeout"], 3)
        client.close()
        client.close()
        self.connection.close.assert_called_once_with()

    def test_model_prompt_strips_public_and_override_hashes_without_changing_evidence(self):
        fingerprint = 'abcdef0123456789' * 4
        public = 'Move the cup. sha256=' + fingerprint + ' Keep it upright.'
        self.evidence['public_instruction'] = public
        self.evidence['robot_state_sha256'] = fingerprint
        client = OpenPIServiceClient('ws://127.0.0.1:8000', timeout=3)
        self.addCleanup(client.close)
        actions = [[0.] * 14 for _ in range(50)]
        self.connection.recv.return_value = dict(actions=actions)
        for override in (None, 'Lift the cup. checkpoint_sha256=' + fingerprint + ' Keep it upright.'):
            with self.subTest(override=override is not None):
                observation = {**self.observation, 'prompt': public}
                self.loader.return_value = (observation, self.evidence)
                result = client.infer({**self.request, 'prompt': override})
                sent = self.connection.send.call_args.args[0]
                self.assertNotIn(fingerprint, sent['prompt'])
                self.assertNotIn('sha256', sent['prompt'])
                self.assertIn('Keep it upright.', sent['prompt'])
                self.assertEqual(sent['state'], self.observation['state'])
                self.assertEqual(sent['images'], self.observation['images'])
                self.assertEqual(result['public_instruction'], public)
                self.assertEqual(result['robot_state_sha256'], fingerprint)
                self.assertEqual(result['policy_prompt'], sent['prompt'])
                self.assertEqual(result['actions'], actions)

    def test_wrong_configuration_or_shape_is_rejected_before_inference(self):
        for metadata in ({"config": "pi0_base", "action_shape": [50, 14]},
                         {"config": TRAIN_CONFIG, "action_shape": [50, 32]}, {}):
            with self.subTest(metadata=metadata):
                self.connection.reset_mock()
                self.connection.recv.return_value = metadata
                with self.assertRaises(ValueError):
                    OpenPIServiceClient("ws://localhost:8000")
                self.connection.send.assert_not_called()
                self.connection.close.assert_called_once()

    def test_handshake_timeout_does_not_reconnect(self):
        self.connection.recv.side_effect = TimeoutError("handshake deadline")
        with self.assertRaises(TimeoutError):
            OpenPIServiceClient("ws://localhost:8000", timeout=1)
        self.connect.assert_called_once()
        self.connection.close.assert_called_once()

    def test_refused_connection_has_no_retry(self):
        self.connect.side_effect = ConnectionRefusedError("offline")
        with self.assertRaises(ConnectionRefusedError):
            OpenPIServiceClient("ws://localhost:8000", timeout=1)
        self.connect.assert_called_once()

    def test_timeout_does_not_resend_or_consume_late_response(self):
        client = OpenPIServiceClient("ws://localhost:8000", timeout=1)
        self.connection.recv.side_effect = TimeoutError("response deadline")
        with self.assertRaises(TimeoutError):
            client.infer(self.request)
        with self.assertRaisesRegex(RuntimeError, "connection_closed"):
            client.infer(self.request)
        self.connection.send.assert_called_once()
        self.connect.assert_called_once()
        self.connection.close.assert_called_once()

    def test_invalid_server_actions_and_text_error_close_connection(self):
        invalid = [[0.] * 14 for _ in range(50)]
        invalid[3][4] = float("nan")
        for response in (dict(actions=invalid), dict(actions=[[0.] * 32 for _ in range(50)]),
                         "server traceback", {}):
            with self.subTest(response_type=type(response).__name__):
                self.connection.reset_mock()
                self.connection.recv.return_value = {"config": TRAIN_CONFIG, "action_shape": [50, 14]}
                client = OpenPIServiceClient("ws://localhost:8000")
                self.connection.recv.return_value = response
                with self.assertRaises((ValueError, RuntimeError, KeyError)):
                    client.infer(self.request)
                self.connection.close.assert_called_once()
                self.connection.send.assert_called_once()

    def test_noise_seed_and_bad_identity_fail_before_send(self):
        client = OpenPIServiceClient("ws://localhost:8000")
        with self.assertRaisesRegex(ValueError, "noise_seed"):
            client.infer({**self.request, "noise_seed": 0})
        self.loader.assert_not_called()
        self.connection.send.assert_not_called()
        bad = copy.deepcopy(self.evidence)
        bad["observation"]["observation_id"] = "other"
        self.loader.return_value = (self.observation, bad)
        with self.assertRaisesRegex(ValueError, "observation_mismatch"):
            client.infer(self.request)
        self.connection.send.assert_not_called()
        client.close()

    def test_invalid_url_or_timeout_never_connects(self):
        for url, timeout in (("http://localhost:8000", 1), ("", 1), ("ws://localhost", 0),
                             ("ws://localhost", float("inf"))):
            with self.assertRaises(ValueError):
                OpenPIServiceClient(url, timeout)
        self.connect.assert_not_called()


class LocalPolicyPromptCheck(unittest.TestCase):
    def test_local_inference_receives_clean_prompt_and_keeps_source_evidence(self):
        from ni_rua.runtime.policy.policy_workers import pi05_arx_inference as worker
        fingerprint = 'abcdef0123456789' * 4
        public = 'Move the cup. sha256=' + fingerprint + ' Keep it upright.'
        evidence = dict(public_instruction=public, robot_state_sha256=fingerprint)
        observation = dict(state=[0] * 14, images={'camera': 'saved pixels'}, prompt=public)
        actions = [[0.] * 14 for _ in range(50)]
        policy = types.SimpleNamespace(infer=Mock(return_value=dict(actions=types.SimpleNamespace(tolist=lambda: actions))))
        modules = {'openpi': types.ModuleType('openpi'),
                   'openpi.policies': types.SimpleNamespace(policy_config=types.SimpleNamespace(create_trained_policy=lambda *_: policy)),
                   'openpi.training': types.SimpleNamespace(config=types.SimpleNamespace(get_config=lambda _: None))}
        requests = [dict(run_dir='/tmp/run', observation_id='obs', prompt=prompt)
                    for prompt in (None, 'Lift the cup. sha256=' + fingerprint + ' Keep it upright.')]
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            stats = root / 'assets' / 'arx_x5_sim' / 'norm_stats.json'
            stats.parent.mkdir(parents=True)
            stats.write_text('{}')
            (root / 'pi05_arx_manifest.json').write_text(json.dumps(dict(revision=worker.CHECKPOINT_REVISION,
                train_config=TRAIN_CONFIG, files=[], repo_id='fixture')))
            output = io.StringIO()
            with patch.dict('sys.modules', modules), patch.object(worker, '__file__', str(root / 'worker.py')), \
                    patch.object(worker, 'NORM_SHA256', hashlib.sha256(stats.read_bytes()).hexdigest()), \
                    patch.object(worker, 'load_policy_observation', side_effect=lambda *_: (copy.deepcopy(observation), evidence)), \
                    patch('sys.argv', ['worker', '--checkpoint', str(root), '--stdio']), \
                    patch('sys.stdin', io.StringIO(''.join(json.dumps(request) + '\n' for request in requests))), \
                    patch('sys.stdout', output):
                worker.main()
        responses = [json.loads(line) for line in output.getvalue().splitlines()][1:]
        self.assertEqual(len(responses), 2)
        for call, response in zip(policy.infer.call_args_list, responses):
            sent = call.args[0]
            self.assertNotIn(fingerprint, sent['prompt'])
            self.assertNotIn('sha256', sent['prompt'])
            self.assertIn('Keep it upright.', sent['prompt'])
            self.assertEqual(sent['state'], observation['state'])
            self.assertEqual(sent['images'], observation['images'])
            self.assertEqual(response['public_instruction'], public)
            self.assertEqual(response['robot_state_sha256'], fingerprint)
            self.assertEqual(response['actions'], actions)
        self.assertEqual(policy.infer.call_count, 2)


class ProcessResponseCheck(unittest.TestCase):
    def setUp(self):
        read_fd, self.write_fd = os.pipe()
        self.stdout = os.fdopen(read_fd, "rb")
        self.addCleanup(self.stdout.close)
        self.addCleanup(os.close, self.write_fd)
        self.policy = OpenPIProcessPolicy.__new__(OpenPIProcessPolicy)
        self.policy.process = types.SimpleNamespace(stdout=self.stdout)
        self.policy.timeout = 0.05
        self.policy._response_buffer = bytearray()

    def test_partial_line_obeys_response_deadline(self):
        os.write(self.write_fd, b'{"status":"ready"}')
        # Release even a regressed blocking readline, so this test cannot hang.
        completion = threading.Timer(0.5, os.write, args=(self.write_fd, b"\n"))
        completion.start()
        try:
            started = time.monotonic()
            with self.assertRaisesRegex(TimeoutError, "timed out"):
                self.policy._receive()
            self.assertLess(time.monotonic() - started, 0.4)
        finally:
            completion.cancel()
            completion.join()

    def test_extra_complete_lines_are_preserved_without_another_pipe_read(self):
        os.write(self.write_fd, b'{"status":"ready"}\n{"status":"inference_completed","id":2}\n')
        self.assertEqual(self.policy._receive(), {"status": "ready"})
        with patch("ni_rua.runtime.policy.action_policy.select.select", side_effect=AssertionError("buffered response must not wait")):
            self.assertEqual(self.policy._receive(), {"status": "inference_completed", "id": 2})
        self.assertEqual(self.policy._response_buffer, b"")

    def test_partial_next_line_is_preserved_and_completed(self):
        os.write(self.write_fd, b'{"status":"ready"}\n{"status":')
        self.assertEqual(self.policy._receive(), {"status": "ready"})
        os.write(self.write_fd, b'"inference_completed"}\n')
        self.assertEqual(self.policy._receive(), {"status": "inference_completed"})


class ProcessSelectionCheck(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.log_path = Path(self.directory.name) / "pi05.log"
        self.process = Mock()
        self.process.poll.return_value = None
        self.popen = patch("ni_rua.runtime.policy.action_policy.subprocess.Popen", return_value=self.process).start()
        self.receive = patch.object(OpenPIProcessPolicy, "_receive", return_value={"status": "ready"}).start()
        self.addCleanup(patch.stopall)

    def test_service_bridge_does_not_take_checkpoint_or_gpu_flags(self):
        with patch.dict("os.environ", {}, clear=True):
            policy = OpenPIProcessPolicy("/openpi/python", "/unused/checkpoint", gpu=7,
                                         server_url="ws://localhost:8000", timeout=2, log_path=self.log_path)
        command = self.popen.call_args.args[0]
        self.assertTrue(command[1].endswith("pi05_arx_client.py"))
        self.assertNotIn("--checkpoint", command)
        self.assertNotIn("CUDA_VISIBLE_DEVICES", self.popen.call_args.kwargs["env"])
        with self.assertRaisesRegex(ValueError, "noise_seed"):
            policy.infer("/tmp/run", "obs-1", noise_seed=0)
        self.process.stdin.write.assert_not_called()
        policy.close()

    def test_local_model_mode_remains_available(self):
        policy = OpenPIProcessPolicy("/openpi/python", "/checkpoint", gpu=7, log_path=self.log_path)
        self.assertTrue(self.popen.call_args.args[0][1].endswith("pi05_arx_inference.py"))
        self.assertIn("--checkpoint", self.popen.call_args.args[0])
        self.assertEqual(self.popen.call_args.kwargs["env"]["CUDA_VISIBLE_DEVICES"], "7")
        policy.close()

    def test_timeout_terminates_bridge(self):
        policy = OpenPIProcessPolicy("/openpi/python", server_url="ws://localhost:8000", log_path=self.log_path)
        self.receive.side_effect = TimeoutError("deadline")
        with self.assertRaises(TimeoutError):
            policy.infer("/tmp/run", "obs-1")
        self.process.terminate.assert_called_once()
        self.assertTrue(policy.log.closed)

    def test_returned_identity_is_checked(self):
        for source, observation_id in (("/tmp/other", "obs-1"), ("/tmp/run", "obs-other")):
            with self.subTest(source=source, observation_id=observation_id):
                self.receive.return_value = {"status": "ready"}
                policy = OpenPIProcessPolicy("/openpi/python", server_url="ws://localhost:8000", log_path=self.log_path)
                self.receive.return_value = dict(status="inference_completed", source_run_dir=source,
                    observation=dict(observation_id=observation_id), actions=[[0.] * 14 for _ in range(50)])
                with self.assertRaisesRegex(ValueError, "observation_mismatch"):
                    policy.infer("/tmp/run", "obs-1")
                self.assertTrue(policy.log.closed)



if __name__ == "__main__":
    unittest.main()
