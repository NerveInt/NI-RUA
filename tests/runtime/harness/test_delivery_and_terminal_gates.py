"""
- test_feedback_delivery.py: Feedback hooks preserve exact action outcomes, phase identity and opt-outs.
- test_image_feedback_delivery.py: Selective delivery never removes archived or explicitly requested evidence.
- test_native_tui_attach.py: Native TUI attach for every backend (work package NATIVE-TUI): the generic ``Terminal`` command, the Workbench
wiring from ``session.terminal_attach_spec()`` and the backend specs (OpenCode 2.2.7: ``opencode --server``; Claude Code:
none, see docs/development/agent-backends.md#native-terminal-featuresterminal).

No native agent binary and no model: the TUI is a small Python fixture in a real PTY; the OpenCode session runs
against the scripted fake server of ``test_agent_opencode_session_unit``. The real-binary attach is in
``test_agent_opencode_terminal_smoke``.
"""
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from ni_rua.core.config import Config
from ni_rua.core.presentation.feedback_delivery import action_trajectory_feedback
from ni_rua.core.storage.store import Store
from support.motion_fixtures import DualArmHarnessBase
from unittest.mock import patch, Mock
from PIL import Image
from support.motion_fixtures import queue_case_fixture, setup_dual_arm_harness
from ni_rua.core.config import Config, cli_overrides
from scripts.benchmark_queue import command
from support.program_fixture import program_services
from support.tool_fixture import install_registry
import asyncio
import threading
import time
from unittest.mock import AsyncMock, Mock, patch
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.programs.runner import ProgramRunner, model_wait, program_remaining
from support.fixture_robot import FixtureRobot
import sys
from ni_rua.cortex.agents.base import AgentCapabilities, AgentSession
from ni_rua.cortex.session.terminal import Terminal
from support.fake_agent import FakeAgentSession
from support.agent_fixtures import OpenCodeSessionBase as SessionCase
from support.agent_fixtures import fake_session, fixture_harness, terminal_double
from support.paths import isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_image_feedback_delivery.py ---

class ImageFeedbackDeliveryTests(unittest.TestCase):
    def setUp(self):
        setup_dual_arm_harness(self)
        original = self.robot.observe
        self.paths = {}
        for camera in ('cam_head', 'cam_left_wrist', 'cam_right_wrist'):
            path = self.robot.path.parent / (camera+'.png')
            Image.new('RGB', (160,120), (70,70,70)).save(path)
            self.paths[camera] = path
        def observe():
            raw = original()
            raw['cameras'] = [dict(camera_id=key,png_path=str(path),width=160,height=120)
                              for key,path in self.paths.items()]
            return raw
        self.robot.observe = observe
        self.h.image_settings['feedback_policy'] = 'adaptive'
        self.first = self.h.observe()
        # Geometry validation itself has dedicated tests; this fixture has synthetic calibration.
        self.h.geometry.camera = Mock(return_value={})
        self.action = dict(path=dict(waypoints=[{'left':{}}]),gripper_events=[])
        self.receipt = dict(status='completed',reason=None)

    def deliver(self, bundle=None, **kwargs):
        return self.h.observation_content(bundle or self.h.latest, {},
            feedback_action=self.action,feedback_receipt=self.receipt, **kwargs)

    @staticmethod
    def count(content):
        return sum(item['type']=='inputImage' for item in content)

    @staticmethod
    def image_metadata(response):
        return next(value['image_feedback'] for item in response['contentItems']
                    if item['type'] == 'inputText'
                    for value in [json.loads(item['text'])]
                    if 'image_feedback' in value)

    def test_initial_full_then_all_unchanged_omission_and_explicit_retrieval(self):
        self.assertEqual(self.count(self.h.observation_content(self.first, {})), 3)
        current = self.h.observe()
        content = self.deliver(current)
        self.assertEqual(self.count(content),0)
        metadata = json.loads(content[0]['text'])['image_feedback']
        self.assertEqual(metadata['omitted_camera_ids'],list(self.paths))
        self.assertEqual([item['baseline_observation_id'] for item in metadata['omitted_views']],
                         [self.first['observation_id']]*3)
        self.assertEqual(len(self.store.read('observation',current['observation_id'])['cameras']),3)
        before = self.robot.executions
        response = self.h.dispatch('history_view_observation',dict(observation_id=current['observation_id'],
            camera_id='cam_right_wrist'),{'tool_call_id':'retrieve'})
        self.assertEqual(self.count(response['contentItems']),1)
        self.assertEqual(self.robot.executions,before)
        self.assertEqual(self.h._last_delivered_views['cam_right_wrist']['observation_id'],current['observation_id'])

    def test_program_internal_and_crop_do_not_advance_model_baseline(self):
        self.h.observation_content(self.first,{})
        new = self.h.observe()
        self.assertEqual(self.count(self.h.observation_content(new, {'program_id':'internal'},
            feedback_action=self.action,feedback_receipt=self.receipt)),3)
        self.h.observation_content(new,{},camera_id='cam_right_wrist',roi=[0,0,20,20])
        self.assertEqual(self.h._last_delivered_views['cam_right_wrist']['observation_id'],self.first['observation_id'])
        self.deliver(new)
        self.assertEqual(self.h._last_delivered_views['cam_right_wrist']['observation_id'],self.first['observation_id'])

    def test_changed_view_included_and_failures_or_checkpoints_include_all(self):
        self.h.observation_content(self.first,{})
        Image.new('RGB',(160,120),(100,70,70)).save(self.paths['cam_right_wrist'])
        self.assertEqual(self.count(self.deliver(self.h.observe())),1)
        for status,reason in [('failed','tracking_error'),('stopped','observation_checkpoint'),('unknown',None)]:
            self.receipt=dict(status=status,reason=reason)
            self.assertEqual(self.count(self.deliver(self.h.observe())),3)
        self.receipt=dict(status='completed',reason=None)
        self.action={}
        self.assertEqual(self.count(self.deliver(self.h.observe())),0)
        self.receipt=dict(status='completed',execution_uncertain=True)
        self.assertEqual(self.count(self.deliver(self.h.observe())),3)

    def test_selector_error_preserves_completed_receipt_and_full_rgb(self):
        with patch('ni_rua.perception.view_selection.select_observation_views',side_effect=ValueError('bad comparison')):
            self.assertEqual(self.count(self.deliver()),3)

    def test_action_uncertainty_is_full_for_top_level_arm_and_jaw_evidence(self):
        self.h.observation_content(self.first, {})
        for updates in (dict(error='observation uncertainty'), dict(execution_uncertain=True),
                        dict(stop_confirmed=False), dict(stop_confirmation=False),
                        dict(stop_confirmation=dict(confirmed=False)),
                        dict(arm_results=dict(left=dict(execution_uncertain=True))),
                        dict(gripper_execution=dict(right=dict(execution_uncertain=True)))):
            with self.subTest(updates=updates):
                self.receipt = dict(status='completed', **updates)
                self.assertEqual(self.count(self.deliver(self.h.observe())), 3)

    def test_default_all_has_no_filter_and_numeric_feedback_has_no_images(self):
        self.assertEqual(Config().images.feedback_policy,'all')
        self.h.observation_content(self.first,{})
        self.h.image_settings['feedback_policy']='all'
        content=self.deliver(self.h.observe())
        self.assertEqual(self.count(content),3)
        self.assertNotIn('image_feedback',json.loads(content[0]['text']))
        self.assertEqual(self.count(self.deliver(numeric_only=True)),0)

    def test_action_dispatch_passes_real_canonical_request_to_selector(self):
        self.h.observation_content(self.first,{})
        original=self.robot.execute
        def execute(*args,**kwargs):
            result=original(*args,**kwargs); result['reason']=None; return result
        self.robot.execute=execute
        self.h.capabilities['bounded_wait']=dict(max_native_steps=50)
        result=self.h.dispatch('robot_wait',dict(observation_id=self.h.latest['observation_id'],
            native_steps=5),{'tool_call_id':'motion'})
        self.assertEqual(self.count(result['contentItems']),0)
        receipt=json.loads(result['contentItems'][0]['text'])['receipt']
        self.assertEqual(receipt['status'],'completed')
        self.assertEqual(self.robot.executions,1)

    def test_config_and_queue_option_are_validated(self):
        cfg=Config().with_overrides(cli_overrides({'image_feedback_policy':'adaptive'}))
        self.assertEqual(cfg.images.feedback_policy,'adaptive')
        qroot = self.enterContext(tempfile.TemporaryDirectory())
        qcase = queue_case_fixture(qroot)
        argv=command(dict(qcase,image_feedback_policy='adaptive'),1,config=Config())
        self.assertEqual(argv[argv.index('--image-feedback-policy')+1],'adaptive')
        for invalid in ('none', True, None):
            with self.assertRaises(ValueError):
                command(dict(qcase,image_feedback_policy=invalid),1,config=Config())

    def test_compaction_marker_and_new_session_invalidate_previously_seen_views(self):
        self.h.observation_content(self.first,{})
        self.assertEqual(self.count(self.deliver(self.h.observe())),0)
        self.store.set_state('runtime','image_context_epoch','compact-1')
        self.assertEqual(self.count(self.deliver(self.h.observe())),3)
        self.assertEqual(self.count(self.deliver(self.h.observe())),0)
        self.store.set_state('runtime','codex',{'app_server_instance_id':'replacement'})
        self.assertEqual(self.count(self.deliver(self.h.observe())),3)

    def test_failed_actor_projection_restores_full_views(self):
        self.h.thread_id = 'image-projection-failure'
        for method in ('_present_part', 'present'):
            with self.subTest(method=method):
                self.h.observation_content(self.h.latest, {})
                self.assertEqual(self.count(self.deliver(self.h.observe())), 0)
                params = dict(threadId=self.h.thread_id, turnId='turn', callId='failed-' + method,
                              tool='robot_observe', arguments={})
                with patch.object(self.h.actor_output, method,
                                  side_effect=ValueError('synthetic image display failure')):
                    if method == 'present':
                        with self.assertRaisesRegex(ValueError, 'synthetic image display failure'):
                            self.h.handle_tool(params)
                    else:
                        self.assertFalse(self.h.handle_tool(params)['success'])
                marker = self.store.read_state('runtime', 'image_context_epoch')
                self.assertEqual(marker['source'], 'tool_output_failed')
                self.assertEqual(marker['tool_call_id'], params['callId'])
                self.assertEqual(self.count(self.deliver(self.h.observe())), 3)
        self.assertEqual(self.robot.executions, 0)

    def test_invalid_calibration_keeps_inactive_view(self):
        self.h.observation_content(self.first,{})
        self.h.geometry.camera.side_effect=ValueError('missing_calibration')
        self.assertEqual(self.count(self.deliver(self.h.observe())),3)

    def test_compaction_invalidates_before_asynchronous_proxy_callback(self):
        from ni_rua.cortex.relay.server import ModelRequestProxy
        events=[]
        proxy=ModelRequestProxy('http://127.0.0.1:1',on_event=events.append)
        self.h.model_proxy=proxy
        self.h.observation_content(self.first,{})
        self.assertEqual(self.count(self.deliver(self.h.observe())),0)
        proxy._record('model_call_started',endpoint='/responses/compact',logical_call_id='compact-sync')
        self.assertEqual(events,[])
        self.assertIsNone(self.store.read_state('runtime','image_context_epoch'))
        self.assertEqual(self.count(self.deliver(self.h.observe())),3)

    def test_native_window_compaction_events_restore_all_views_without_compact_endpoint(self):
        from ni_rua.cortex.agents.codex.session import CodexSession
        session=CodexSession(self.store.root,self.store,lambda value:{})
        self.addCleanup(session.close)
        session.thread_id='actor-thread'
        self.h.observation_content(self.first,{})
        self.assertEqual(self.count(self.deliver(self.h.observe())),0)
        for phase in ('item/started','item/completed'):
            # Actual October experiment event shape; this path calls /responses.
            session._notification(dict(method=phase,params=dict(threadId='actor-thread',
                turnId='actor-turn',item=dict(type='contextCompaction',id='window-compact'))))
            marker=self.store.read_state('runtime','image_context_epoch')
            self.assertEqual(marker['source'],'native_context_compaction')
            self.assertEqual(marker['phase'],phase)
            self.assertEqual(self.count(self.deliver(self.h.observe())),3)
            self.assertEqual(self.count(self.deliver(self.h.observe())),0)

    def test_other_thread_compaction_and_ordinary_items_do_not_reset_actor_views(self):
        from ni_rua.cortex.agents.codex.session import CodexSession
        session=CodexSession(self.store.root,self.store,lambda value:{})
        self.addCleanup(session.close)
        session.thread_id='actor-thread'
        self.h.observation_content(self.first,{})
        for thread,kind in [('background-thread','contextCompaction'),('actor-thread','reasoning')]:
            session._notification(dict(method='item/started',params=dict(threadId=thread,
                turnId='turn',item=dict(type=kind,id='not-actor-compaction'))))
            self.assertIsNone(self.store.read_state('runtime','image_context_epoch'))
            self.assertEqual(self.count(self.deliver(self.h.observe())),0)



# --- Folded from tests/test_feedback_delivery.py ---

class FeedbackCompositionTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.identity = dict(run_id="r", episode_id="e", env_instance_id="env", env_generation=1)
        self.store = Store(Path(directory.name), self.identity)
        self.addCleanup(self.store.close)
        self.h = SimpleNamespace(store=self.store, identity=self.identity, config=Config())

    def test_program_internal_and_disabled_feature_do_not_render(self):
        with patch("ni_rua.runtime.actions.trajectory_feedback.build_action_trajectory_feedback") as build:
            self.assertEqual(action_trajectory_feedback(self.h, {}, {}, {"program_id":"p"}), (None, []))
            self.h.config = self.h.config.with_overrides({"features":{"trajectory_preview":False}})
            self.assertEqual(action_trajectory_feedback(self.h, {}, {}, {}), (None, []))
        build.assert_not_called()


class ActionFeedbackDispatchTests(unittest.TestCase):
    setUp = DualArmHarnessBase.setUp

    def test_renderer_failure_does_not_hide_completed_receipt_or_repeat_execution(self):
        self.h.capabilities["bounded_wait"] = dict(max_native_steps=50)
        with patch("ni_rua.runtime.actions.trajectory_feedback.build_action_trajectory_feedback", side_effect=RuntimeError("image rendering failed")):
            reply = self.h.dispatch("robot_wait", dict(observation_id=self.h.latest["observation_id"],
                native_steps=5), {"tool_call_id":"move"})
        result = json.loads(reply["contentItems"][0]["text"])
        self.assertTrue(reply["success"])
        self.assertEqual(result["receipt"]["status"], "completed")
        self.assertFalse(result["trajectory_feedback"]["available"])
        self.assertEqual(self.robot.executions, 1)
        stored = self.store.read("action_receipt", result["request"]["action_id"])
        self.assertEqual(self.h.execute(result["request"])["receipt"], stored)
        self.assertEqual(self.robot.executions, 1)


class OptionalImageBookkeepingTests(unittest.TestCase):
    setUp = ImageFeedbackDeliveryTests.setUp

    def move(self):
        original = self.robot.execute
        def completed(*args, **kwargs):
            result = original(*args, **kwargs)
            result["reason"] = None
            return result
        self.robot.execute = completed
        self.h.capabilities["bounded_wait"] = dict(max_native_steps=50)
        return self.h.dispatch("robot_wait", dict(observation_id=self.h.latest["observation_id"],
            native_steps=5), {"tool_call_id":"bookkeeping-move"})

    def test_baseline_failure_preserves_receipt_images_and_future_full_delivery(self):
        with patch("ni_rua.perception.view_selection.baseline_record", side_effect=RuntimeError("optional baseline failed")):
            reply = self.move()
        result = json.loads(reply["contentItems"][0]["text"])
        self.assertTrue(reply["success"])
        self.assertEqual(result["receipt"]["status"], "completed")
        self.assertEqual(sum(item["type"] == "inputImage" for item in reply["contentItems"]), 3)
        self.assertEqual(self.h._last_delivered_views, {})
        self.assertEqual(self.robot.executions, 1)

    def test_counterfactual_byte_failure_does_not_hide_already_executed_motion(self):
        self.h.observation_content(self.first, {})
        with patch("ni_rua.runtime.harness.image_url", side_effect=RuntimeError("optional telemetry failed")):
            reply = self.move()
        result = json.loads(reply["contentItems"][0]["text"])
        self.assertTrue(reply["success"])
        self.assertEqual(result["receipt"]["status"], "completed")
        self.assertEqual(self.robot.executions, 1)
        metrics = self.store.records("delivery")[-1]["image_metrics"]
        self.assertIsNone(metrics["omitted_transport_bytes"])
        self.assertEqual(metrics["omitted_transport_bytes_unavailable"], "RuntimeError")
        stored = self.store.read("action_receipt", result["request"]["action_id"])
        self.assertEqual(stored["status"], result["receipt"]["status"])
        self.assertEqual(stored["result_observation_id"], result["receipt"]["result_observation_id"])
        self.assertEqual(self.h.execute(result["request"])["receipt"], stored)
        self.assertEqual(self.robot.executions, 1)

    def test_all_policy_does_not_depend_on_adaptive_baseline_bookkeeping(self):
        self.h.image_settings["feedback_policy"] = "all"
        with patch("ni_rua.perception.view_selection.baseline_record", side_effect=RuntimeError("unused")) as record:
            reply = self.move()
        record.assert_not_called()
        self.assertTrue(reply["success"])
        self.assertEqual(json.loads(reply["contentItems"][0]["text"])["receipt"]["status"], "completed")



# --- Folded from tests/test_native_terminal_gates.py ---

"""CPU regression evidence for native termination and delayed policy admission."""





IDENTITY = dict(run_id="native-gate-fixture", episode_id="ep", env_instance_id="env", env_generation=1)
TERMINAL = dict(ended=True, success=True, reason="native_success", native_action_count=1, native_step_limit=200)


class JointRobot(FixtureRobot):
    def execute(self, action, progress_callback=None, cancel_flag=None):
        self.executions += 1
        return dict(status="completed", terminal_observation=self.observe(), executed_steps=len(action["actions"]))


class Policy:
    def __init__(self, before_return=None):
        self.calls = 0
        self.before_return = before_return

    def infer(self, *args, **kwargs):
        self.calls += 1
        if self.before_return:
            self.before_return()
        return dict(actions=[[0.0] * 14 for _ in range(50)])


class NativeTerminalGateCheck(unittest.TestCase):
    def harness(self, directory):
        store = Store(Path(directory) / "store", dict(IDENTITY))
        self.addCleanup(store.close)
        robot = JointRobot(directory)
        harness = Harness(store, robot)
        harness.start()
        return harness, robot

    def test_policy_retry_reuses_prediction_and_receipt_after_observation_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            harness, robot = self.harness(directory)
            harness.action_policy = Policy()
            args = dict(observation_id=harness.latest["observation_id"], action_id="same-id", prefix=1,
                        prompt="Close the gripper around the selected object.")
            with self.assertRaisesRegex(ValueError, "invalid_tool_arguments"):
                harness.dispatch("robot_policy_step", {key:value for key,value in args.items() if key != "prompt"},
                                 dict(tool_call_id="missing-local-intent"))
            self.assertEqual(harness.action_policy.calls, 0)
            first = harness.dispatch("robot_policy_step", args, dict(tool_call_id="first"))
            repeated = harness.dispatch("robot_policy_step", args, dict(tool_call_id="repeat"))
            # Idempotency preserves the request/receipt and saved RGB. Static
            # observation conventions may be referenced on the second model
            # delivery instead of retransmitted verbatim.
            self.assertEqual(first["success"], repeated["success"])
            self.assertEqual(first["contentItems"][0], repeated["contentItems"][0])
            self.assertEqual([item for item in first["contentItems"] if item["type"] == "inputImage"],
                             [item for item in repeated["contentItems"] if item["type"] == "inputImage"])
            observations = [next(json.loads(item["text"]) for item in reply["contentItems"]
                if item["type"] == "inputText" and "robot_state" in json.loads(item["text"]))
                for reply in (first, repeated)]
            for field in ("current_observation_id", "observation_is_current"):
                self.assertEqual(observations[0][field], observations[1][field])
            self.assertEqual(observations[0]["bundle"]["observation_id"], observations[1]["bundle"]["observation_id"])
            self.assertEqual(observations[0]["robot_state"]["arms"], observations[1]["robot_state"]["arms"])
            for field in ("native_steps_used", "native_steps_limit", "native_steps_remaining"):
                self.assertEqual(observations[0]["current_action_budget"][field],
                                 observations[1]["current_action_budget"][field])
            self.assertEqual(harness.action_policy.calls, 1)
            self.assertEqual(robot.executions, 1)
            with self.assertRaisesRegex(ValueError, "action_id_content_conflict"):
                harness.dispatch("robot_policy_step", {**args, "prefix": 2}, dict(tool_call_id="changed"))
            self.assertEqual(harness.action_policy.calls, 1)

    def test_delayed_inference_cannot_outlive_program_or_observation(self):
        for interruption in ("cancel", "deadline", "stale_observation", "native_terminal"):
            with self.subTest(interruption=interruption), tempfile.TemporaryDirectory() as directory:
                harness, robot = self.harness(directory)
                runner = ProgramRunner(program_services(harness))
                harness.programs = runner
                entry = dict(job_id="program-fixture", cancel=threading.Event(), deadline=time.monotonic() + 30,
                             call_lock=threading.Lock(), calls=0, max_calls=3,
                             correlation=dict(thread_id="thread", turn_id="turn", tool_call_id="outer"))
                runner.entries[entry["job_id"]] = entry
                harness.store.set_state("program", entry["job_id"], dict(job_id=entry["job_id"], status="running"))

                def interrupt():
                    if interruption == "cancel":
                        runner.cancel(entry["job_id"])
                    elif interruption == "deadline":
                        entry["deadline"] = time.monotonic() - 1
                    elif interruption == "stale_observation":
                        harness.observe()
                    else:
                        harness._native_progress(dict(TERMINAL))

                harness.action_policy = Policy(interrupt)
                # Exercise the retained backend gate directly. The actor program
                # broker now rejects legacy policy motion before inference.
                result = harness.dispatch("robot_policy_step",
                    dict(observation_id=harness.latest["observation_id"], prefix=1,
                         prompt="Make a short local grasp adjustment."),
                    dict(program_id=entry['job_id'],tool_call_id='one'))
                receipt = json.loads(result["contentItems"][0]["text"])["receipt"]
                self.assertEqual(receipt["status"], "rejected")
                self.assertEqual(robot.executions, 0)
                self.assertEqual(harness.action_count, 0)
                self.assertEqual(harness.action_policy.calls, 1)

    def test_program_reserves_action_time_before_starting_motion(self):
        for remaining, expected in ((4, "rejected"), (20, "completed")):
            with self.subTest(remaining=remaining), tempfile.TemporaryDirectory() as directory:
                harness, robot = self.harness(directory)
                runner = ProgramRunner(program_services(harness))
                harness.programs = runner
                identifier = "program-fixture"
                runner.entries[identifier] = dict(job_id=identifier, cancel=threading.Event(),
                    deadline=time.monotonic() + remaining)
                harness.action_policy = Policy()
                result = harness.dispatch("robot_policy_step", dict(observation_id=harness.latest["observation_id"],
                    prefix=1, timeout_s=5, prompt="Make a short local grasp adjustment."),
                    dict(program_id=identifier, tool_call_id="bounded"))
                receipt = json.loads(result["contentItems"][0]["text"])["receipt"]
                self.assertEqual(receipt["status"], expected)
                self.assertEqual(robot.executions, int(expected == "completed"))
                if expected == "rejected":
                    self.assertEqual(receipt["reason"], "program_time_insufficient_for_action")

    def test_next_action_reserves_original_remaining_time_after_own_sol_wait(self):
        with tempfile.TemporaryDirectory() as directory:
            harness, robot = self.harness(directory)
            runner = harness.programs = ProgramRunner(program_services(harness))
            entry = dict(job_id="program-fixture", cancel=threading.Event(), deadline=120.0)
            runner.entries[entry["job_id"]] = entry
            harness.action_policy = Policy()
            clock = [100.0]
            with patch("ni_rua.runtime.programs.runner.time.monotonic", side_effect=lambda: clock[0]):
                with model_wait(entry):
                    clock[0] += 1000
                self.assertEqual(program_remaining(entry), 20.0)
                result = harness.dispatch("robot_policy_step", dict(observation_id=harness.latest["observation_id"],
                    prefix=1, timeout_s=5, prompt="Make a short local grasp adjustment."),
                    dict(program_id=entry["job_id"], tool_call_id="after-sol"))
            receipt = json.loads(result["contentItems"][0]["text"])["receipt"]
            self.assertEqual(receipt["status"], "completed")
            self.assertEqual(robot.executions, 1)

    def test_native_success_survives_actor_interrupt_failure_and_can_be_evaluated(self):
        with tempfile.TemporaryDirectory() as directory:
            harness, robot = self.harness(directory)
            harness._native_progress(dict(TERMINAL))
            workbench = Workbench.__new__(Workbench)
            workbench.harness, workbench.store, workbench.robot = harness, harness.store, robot
            workbench.session = SimpleNamespace(state="running", interrupt=Mock(side_effect=RuntimeError("synthetic interrupt failure")), close=Mock())
            workbench.control_responses, workbench.status, workbench.starting = {}, "running", False
            workbench.identity, workbench.replay = dict(IDENTITY), False
            robot.evaluate = Mock(return_value=dict(task_success=True, normalized_score=1.0))
            response = asyncio.run(workbench.stop("native-stop"))
            self.assertEqual(response["status"], "stopped")
            workbench.session.close.assert_called_once()
            self.assertTrue(workbench.observer_evaluate()["result"]["task_success"])
            self.assertFalse(harness.accepting)

    def test_native_stop_waits_for_cancelled_program_cleanup_before_scoring(self):
        for interrupt_error in (False, True):
            with self.subTest(interrupt_error=interrupt_error), tempfile.TemporaryDirectory() as directory:
                harness, robot = self.harness(directory)
                harness._native_progress(dict(TERMINAL))
                entries = {"program": dict(job_id="program")}
                cleanup = threading.Event()
                def cancel(identifier):
                    def finish():
                        entries.pop(identifier, None)
                        cleanup.set()
                    threading.Timer(.05, finish).start()
                harness.programs = SimpleNamespace(
                    cancel_all=lambda: [cancel(identifier) for identifier in list(entries)],
                    unresolved_jobs=lambda: [],
                    snapshot=lambda: dict(active_jobs=list(entries), unresolved_jobs=[], pending_reviews=[]))
                workbench = Workbench.__new__(Workbench)
                workbench.harness, workbench.store, workbench.robot = harness, harness.store, robot
                workbench.session = SimpleNamespace(state="completed", close=Mock(),
                    interrupt=Mock(side_effect=RuntimeError("interrupt failed") if interrupt_error else None))
                workbench.control_responses, workbench.status, workbench.starting = {}, "running", False
                workbench.identity, workbench.replay = dict(IDENTITY), False
                robot.evaluate = Mock(return_value=dict(task_success=True, normalized_score=1.0))
                response = asyncio.run(workbench.stop("drain-program"))
                self.assertEqual(response["status"], "stopped")
                self.assertTrue(cleanup.is_set())
                self.assertFalse(entries)
                self.assertTrue(workbench.observer_evaluate()["result"]["task_success"])

    def test_terminal_during_first_turn_start_preserves_worker_and_manifest(self):
        from ni_rua.core.config import Config
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory, dict(IDENTITY))
        self.addCleanup(store.close)
        workbench = Workbench.__new__(Workbench)
        workbench.config = Config()
        harness = SimpleNamespace(accepting=True, episode_terminal=None, start=Mock(), capabilities={},
            latest=dict(observation_profile="rgb_proprio_v1"), scene=SimpleNamespace(capabilities=lambda: {}))
        install_registry(harness, workbench.config)
        workbench.harness, workbench.root, workbench.store = harness, Path(directory), store
        workbench.identity, workbench.replay, workbench.resume = dict(IDENTITY), False, False
        workbench.pi05_enabled, workbench.visual_review_model = False, None
        workbench.enable_wait, workbench.tracking_guard = False, False
        workbench.joint_replan = False
        workbench.completion_guidance = "none"
        workbench.observation_profile = "rgb_proprio_v1"
        workbench.motion_review = False
        workbench.vision_compare = True
        workbench.grasp_memory_store = None
        workbench.actor_payload_profile = "full_v1"
        workbench.robot = SimpleNamespace(close=Mock())
        workbench.terminal, workbench.starting = None, True
        workbench.task, workbench.seed, workbench.layout_id = "fixture", 0, 0
        workbench.source_hashes, workbench.source_ref = {}, None

        def first_turn(prompt):
            harness.episode_terminal, harness.accepting = dict(TERMINAL), False

        workbench.session = SimpleNamespace(start=Mock(return_value="thread"), run_turn=first_turn,
            close=Mock(), thread_id="thread", actor_dir=Path(directory) / "actor", env={}, socket_path="unused",
            cli_executable=Path("/fixture/native/codex"), command_prefix=["fixture-sandbox"],
            model="fixture", provider="fixture", effort="high", price_card=None, process=SimpleNamespace(pid=1))
        terminal = SimpleNamespace(start=AsyncMock(), close=AsyncMock(), session_id="terminal", process=SimpleNamespace(pid=2))
        with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal) as create_terminal:
            asyncio.run(workbench.start("synthetic startup"))
        self.assertEqual(create_terminal.call_args.kwargs["command_prefix"][:2], ["fixture-sandbox", "/fixture/native/codex"])
        workbench.robot.close.assert_not_called()
        workbench.session.close.assert_not_called()
        self.assertTrue((Path(directory) / "manifest.json").is_file())
        self.assertFalse(workbench.starting)
        self.assertTrue(harness.episode_terminal["success"])



# --- Folded from tests/test_native_tui_attach.py ---

# A stand-in attach TUI: prints its argv and a marker, reports whether anything ever arrived on its stdin (the PTY
# slave), then stays alive until the terminal is closed.
TUI_FIXTURE = (
    "import json, os, select, sys, time\n"
    "assert os.isatty(0) and os.isatty(1)\n"
    "print(json.dumps(sys.argv[1:]), flush=True)\n"
    "print('FIXTURE-TUI-RENDERED', flush=True)\n"
    "readable, _, _ = select.select([0], [], [], 1.0)\n"
    "print('STDIN-INPUT' if readable else 'NO-STDIN-INPUT', flush=True)\n"
    "time.sleep(60)\n"
)


async def read_until(terminal, marker, timeout=10.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while asyncio.get_running_loop().time() < deadline:
        data, _ = terminal.read(0, 1048576)
        if marker in data:
            return data
        await asyncio.sleep(0.05)
    raise AssertionError(f"{marker!r} never appeared in the terminal log: {terminal.read(0, 1048576)[0][-500:]!r}")


class ExplicitCommandTerminalTests(unittest.IsolatedAsyncioTestCase):
    async def test_explicit_command_runs_exactly_and_never_receives_input(self):
        with tempfile.TemporaryDirectory() as directory:
            fixture = Path(directory) / "attach_tui.py"
            fixture.write_text(TUI_FIXTURE)
            command = [sys.executable, str(fixture), "attach", "http://127.0.0.1:4096", "--session", "ses_fixture"]
            terminal = Terminal(directory, directory, command_prefix=[sys.executable], env={"FIXTURE": "1"},
                                command=command)
            try:
                status = await terminal.start("http://127.0.0.1:4096", "ses_fixture")
                self.assertEqual((status["mode"], status["thread_id"]), ("read_only", "ses_fixture"))
                terminal.resize(90, 30)        # the only thing a browser may send: no bytes reach the TUI's stdin
                data = await read_until(terminal, b"STDIN-INPUT")
                self.assertEqual(json.loads(data.splitlines()[0]), command[2:])
                self.assertIn(b"FIXTURE-TUI-RENDERED", data)
                self.assertIn(b"NO-STDIN-INPUT", data)
                saved = json.loads((Path(directory) / "terminal.json").read_text())
                self.assertEqual((saved["command"], saved["socket_uri"]), (command, "http://127.0.0.1:4096"))
                self.assertNotIn("FIXTURE", json.dumps(saved))      # the environment is never persisted
                pid = terminal.process.pid
            finally:
                await terminal.close()
            self.assertIsNotNone(terminal.process.poll())
            self.assertEqual(terminal.status()["pid"], pid)
            self.assertIsNotNone(terminal.status()["exit_code"])
            self.assertFalse(hasattr(terminal, "write"))            # no input path exists at all

    async def test_http_endpoint_needs_an_explicit_command(self):
        with tempfile.TemporaryDirectory() as directory:
            codex = Terminal(directory, directory, (sys.executable, "-c", "pass"))
            with self.assertRaisesRegex(ValueError, "remote endpoint"):
                await codex.start("http://127.0.0.1:4096", "thread")      # the Codex remote TUI keeps its schemes
            self.assertIsNone(codex.process)
            explicit = Terminal(directory, directory, command=[sys.executable, "-c", "pass"])
            with self.assertRaisesRegex(ValueError, "remote endpoint"):
                await explicit.start("file:///etc/passwd", "thread")
            with self.assertRaisesRegex(ValueError, "remote endpoint"):
                await explicit.start("http://127.0.0.1:4096", "")


class WorkbenchAttachTests(unittest.IsolatedAsyncioTestCase):
    async def workbench(self, capabilities, spec):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        create, created = fake_session(capabilities, backend_kind="opencode")
        robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
        with patch("ni_rua.apps.gateway.workbench.create_session", side_effect=create):
            workbench = Workbench(Path(temporary.name) / "run", robot=robot, visual_review_model=None,
                                  config=Config.from_dict({"features": {"self_briefing": False}}))
        session, = created
        session.attach_spec = spec
        workbench.harness = fixture_harness(workbench)
        return workbench, session

    async def test_spec_command_is_passed_to_the_terminal(self):
        spec = dict(cwd=Path("/fixture/actor"), command_prefix=["/fixture/opencode"], env={"A": "1"},
                    command=["/fixture/opencode", "attach", "http://127.0.0.1:1", "--session", "ses_1"],
                    attach_uri="http://127.0.0.1:1", thread_id="ses_1")
        workbench, _ = await self.workbench(AgentCapabilities(terminal_attach=True), spec)
        terminal = terminal_double()
        with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal) as create_terminal:
            await workbench.start("Smoke instruction.")
        self.addAsyncCleanup(workbench.close)
        create_terminal.assert_called_once_with(workbench.root, spec["cwd"], command_prefix=spec["command_prefix"],
                                                env=spec["env"], command=spec["command"])
        terminal.start.assert_awaited_once_with(spec["attach_uri"], spec["thread_id"])

    async def test_real_pty_terminal_from_a_backend_spec_reaches_snapshot_manifest_and_close(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        fixture = Path(directory.name) / "attach_tui.py"
        fixture.write_text(TUI_FIXTURE)
        command = [sys.executable, str(fixture), "attach", "http://127.0.0.1:1", "--session", "ses_live"]
        spec = dict(cwd=Path(directory.name), command_prefix=[sys.executable], env={}, command=command,
                    attach_uri="http://127.0.0.1:1", thread_id="ses_live")
        workbench, _ = await self.workbench(AgentCapabilities(terminal_attach=True), spec)
        await workbench.start("Smoke instruction.")
        try:
            terminal = workbench.terminal
            self.assertIsInstance(terminal, Terminal)
            data = await read_until(terminal, b"STDIN-INPUT")
            self.assertIn(b"FIXTURE-TUI-RENDERED", data)
            self.assertIn(b"NO-STDIN-INPUT", data)
            snapshot = workbench.snapshot()
            self.assertEqual((snapshot["terminal"]["mode"], snapshot["terminal"]["thread_id"],
                              snapshot["agent_backend"]), ("read_only", "ses_live", "opencode"))
            manifest = json.loads((workbench.root / "manifest.json").read_text())
            self.assertEqual((manifest["terminal_session_id"], manifest["terminal_pid"]),
                             (terminal.session_id, terminal.process.pid))
            process = terminal.process
        finally:
            report = await workbench.close()
        self.assertEqual(report["steps"]["terminal"], "completed")
        self.assertIsNotNone(process.poll())

    async def test_backend_spec_none_skips_the_terminal(self):
        workbench, _ = await self.workbench(AgentCapabilities(terminal_attach=True), None)
        with patch("ni_rua.apps.gateway.workbench.Terminal") as create_terminal:
            await workbench.start("Smoke instruction.")
        self.addAsyncCleanup(workbench.close)
        create_terminal.assert_not_called()
        self.assertIsNone(workbench.snapshot()["terminal"])
        manifest = json.loads((workbench.root / "manifest.json").read_text())
        self.assertEqual((manifest["terminal_session_id"], manifest["terminal_pid"]), (None, None))


class OpenCodeAttachSpecTests(SessionCase):
    def test_capability_and_spec_follow_the_session_lifecycle(self):
        session = self.make()
        self.assertTrue(session.capabilities.terminal_attach)
        self.assertIsNone(session.terminal_attach_spec())              # not started: no server, port or session
        session.start([], "Be careful.")
        spec = session.terminal_attach_spec()
        url = f"http://{session.client.host}:{session.client.port}"
        # OpenCode 2.2.7 form (``attach`` no longer exists): ``opencode --server <url> --session <id> <dir>``.
        self.assertEqual(spec, dict(
            cwd=session.actor_dir, command_prefix=[session.launch.executable],
            command=[session.launch.executable, "--server", url, "--session", session.thread_id,
                     str(session.actor_dir)],
            env=session.launch.env, attach_uri=url, thread_id=session.thread_id))
        # The server's own isolated environment; the password travels only in it (the TUI reads it from there).
        password = spec["env"]["OPENCODE_PASSWORD"]
        self.assertTrue(password)
        self.assertFalse(any(password in argument for argument in spec["command"]))
        for name in ("XDG_CONFIG_HOME", "XDG_DATA_HOME", "XDG_STATE_HOME", "OPENCODE_CONFIG_DIR",
                     "OPENCODE_DISABLE_AUTOUPDATE", "OPENCODE_DISABLE_MODELS_FETCH", "OPENCODE_DISABLE_PROJECT_CONFIG"):
            self.assertEqual(spec["env"][name], session.launch.env[name])
        self.assertIsNot(spec["env"], session.launch.env)              # a copy: the Terminal cannot alter the launch
        self.assertTrue(session.provenance()["capabilities"]["terminal_attach"])
        session.close()
        self.assertIsNone(session.terminal_attach_spec())

    def test_sandboxed_server_has_no_terminal(self):
        session = self.make()
        session.start([], "Be careful.")
        self.assertIsNotNone(session.terminal_attach_spec())
        session.sandbox_enabled = True       # a sandboxed server is reachable only through its inbound Unix socket
        self.assertIsNone(session.terminal_attach_spec())


class ClaudeCodeAttachTests(unittest.TestCase):
    def test_claude_code_declares_no_terminal(self):
        """Investigation result (docs, agent-backends.md): the SDK-driven ``--print`` session cannot be attached."""
        from ni_rua.cortex.agents.claude_code.session import ClaudeCodeSession
        self.assertFalse(ClaudeCodeSession.capabilities.terminal_attach)
        self.assertIs(ClaudeCodeSession.terminal_attach_spec, AgentSession.terminal_attach_spec)


class DeltaScopeIsolationAndTruncationTests(unittest.TestCase):
    """验证 5 类消费点隔离、isolated 不触发 prepare(None)、子链不抢吃提醒、宿主超限本地失效与 adaptive 快照守卫。"""

    setUp = ImageFeedbackDeliveryTests.setUp
    deliver = ImageFeedbackDeliveryTests.deliver
    count = staticmethod(ImageFeedbackDeliveryTests.count)

    def test_isolated_never_clears_main_metadata_or_view_baselines(self):
        # 1) 主链首帧播种 MetadataDelivery 与 adaptive 视图基线
        first = self.h.observation_content(self.first, {"context_scope": "main"})
        self.assertEqual(self.count(first), 3)
        self.assertNotIn("metadata_delivery", json.loads(first[0]["text"]))
        self.assertTrue(self.h._metadata_delivery._delivered)
        saved_views = dict(self.h._last_delivered_views)
        self.assertEqual(set(saved_views), {"cam_head", "cam_left_wrist", "cam_right_wrist"})

        # 2) isolated 子链调用返回完整元数据与全部图像，且严禁调用 prepare(None) 清空主基线
        obs2 = self.h.observe()
        with patch.object(self.h._metadata_delivery, "prepare", wraps=self.h._metadata_delivery.prepare) as prep_spy, \
             patch("ni_rua.perception.view_selection.select_observation_views") as sel_spy:
            iso = self.h.observation_content(
                obs2, {"context_scope": "isolated"},
                feedback_action=self.action, feedback_receipt=self.receipt,
            )
        prep_spy.assert_not_called()
        sel_spy.assert_not_called()
        self.assertEqual(self.count(iso), 3)
        iso_meta = json.loads(iso[0]["text"])
        self.assertNotIn("metadata_delivery", iso_meta)
        self.assertIn("camera_calibration", iso_meta)
        self.assertEqual(self.h._last_delivered_views, saved_views)
        self.assertTrue(self.h._metadata_delivery._delivered)

        # 3) 过时纪元直接调 MetadataDelivery.prepare(..., live_epoch=E2) 也不得清空主基线
        stale_proj = self.h._metadata_delivery.prepare(iso_meta, epoch=("stale",), live_epoch=self.h._model_context_epoch())
        self.assertNotIn("metadata_delivery", stale_proj.metadata)
        self.assertFalse(self.h._metadata_delivery.commit(stale_proj))
        self.assertTrue(self.h._metadata_delivery._delivered)

        # 4) 后续主链调用仍命中首帧基线（metadata static_refs 生效 + 视图省略生效）
        obs3 = self.h.observe()
        next_main = self.h.observation_content(
            obs3, {"context_scope": "main"},
            feedback_action=self.action, feedback_receipt=self.receipt,
        )
        self.assertEqual(self.count(next_main), 0)
        self.assertIn("metadata_delivery", json.loads(next_main[0]["text"]))

    def test_isolated_subagent_cannot_steal_main_decision_reminder(self):
        from ni_rua.runtime.decision_context import REMINDER_IN_TOOL_RESULT
        self.h.thread_id = "main-thread"
        self.h.task_planning.required = True
        self.h.decision_reminder_channel = REMINDER_IN_TOOL_RESULT
        self.h.model_context_reset("relay_context_discontinuity")
        self.assertTrue(self.h.decision_reminder_due)

        # isolated 调用不得消费决策提醒
        iso_reply = self.h.handle_tool(dict(
            threadId="main-thread", turnId="turn", callId="sub-call",
            tool="robot_capabilities", arguments=dict(detail="summary"),
            contextScope="isolated",
        ))
        self.assertTrue(self.h.decision_reminder_due)
        self.assertEqual(len(iso_reply["contentItems"]), 1)
        self.assertEqual(self.store.events_by_type(["model_context.decision_reminder_delivered"], limit=10), [])

        # 随后的 main 调用收到决策提醒并清除 due 标记
        main_reply = self.h.handle_tool(dict(
            threadId="main-thread", turnId="turn", callId="main-call",
            tool="robot_capabilities", arguments=dict(detail="summary"),
            contextScope="main",
        ))
        self.assertFalse(self.h.decision_reminder_due)
        self.assertEqual(len(main_reply["contentItems"]), 2)
        delivered_ev, = self.store.events_by_type(["model_context.decision_reminder_delivered"], limit=10)
        self.assertTrue(delivered_ev["payload"]["included"])
        self.assertEqual(delivered_ev["correlation"]["tool_call_id"], "main-call")

    def test_host_truncation_limit_invalidates_delivery_locally(self):
        self.h.thread_id = "trunc-thread"
        # 先建立基线
        first = self.h.handle_tool(dict(
            threadId="trunc-thread", turnId="turn", callId="cap-1",
            tool="robot_capabilities", arguments=dict(detail="summary"),
        ))
        self.assertEqual(json.loads(first["contentItems"][0]["text"])["capabilities_delivery"]["status"], "inline")

        # 1) Claude 语义：text_only_without_images=True 且纯文本超字节上限 -> 立刻失效基线
        self.h.host_truncation_limit = lambda: (10, None, True)
        second = self.h.handle_tool(dict(
            threadId="trunc-thread", turnId="turn", callId="cap-2",
            tool="robot_capabilities", arguments=dict(detail="summary"),
        ))
        self.assertEqual(json.loads(second["contentItems"][0]["text"])["capabilities_delivery"]["status"], "unchanged")
        marker = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual((marker["source"], marker["tool_call_id"]), ("tool_output_failed", "cap-2"))

        # 失效后下一条自动恢复完整 inline
        self.h.host_truncation_limit = None
        third = self.h.handle_tool(dict(
            threadId="trunc-thread", turnId="turn", callId="cap-3",
            tool="robot_capabilities", arguments=dict(detail="summary"),
        ))
        self.assertEqual(json.loads(third["contentItems"][0]["text"])["capabilities_delivery"]["status"], "inline")

        # 2) Claude 语义：text_only_without_images=True 但回复含图片 -> 跳过字节上限，不失效基线
        self.h.image_settings["feedback_policy"] = "all"
        self.h.host_truncation_limit = lambda: (10, None, True)
        before_epoch = self.store.read_state("runtime", "image_context_epoch")
        obs_reply = self.h.handle_tool(dict(
            threadId="trunc-thread", turnId="turn", callId="obs-img-ok",
            tool="robot_observe", arguments={},
        ))
        self.assertTrue(any(item["type"] == "inputImage" for item in obs_reply["contentItems"]))
        self.assertEqual(self.store.read_state("runtime", "image_context_epoch"), before_epoch)

        # 3) Codex 语义：text_only_without_images=False 即使含图片也触发超字节失效
        self.h.host_truncation_limit = lambda: (10, None, False)
        self.h.handle_tool(dict(
            threadId="trunc-thread", turnId="turn", callId="obs-img-over",
            tool="robot_observe", arguments={},
        ))
        self.assertEqual(self.store.read_state("runtime", "image_context_epoch")["tool_call_id"], "obs-img-over")

        # 4) OpenCode 语义：超行数上限触发失效
        self.h.host_truncation_limit = lambda: (16_000_000, 0, False)
        self.h.handle_tool(dict(
            threadId="trunc-thread", turnId="turn", callId="cap-lines-over",
            tool="robot_capabilities", arguments=dict(detail="summary"),
        ))
        self.assertEqual(self.store.read_state("runtime", "image_context_epoch")["tool_call_id"], "cap-lines-over")

    def test_adaptive_view_snapshot_and_auto_roi_baseline_guard(self):
        # 1) 传递浅拷贝快照给 select_observation_views（防止共享可变字典）
        self.h.observation_content(self.first, {"context_scope": "main"})
        seen_previous = []
        from ni_rua.perception import view_selection
        real_select = view_selection.select_observation_views

        def spy_select(bundle, images, previous, **kwargs):
            seen_previous.append(previous)
            return real_select(bundle, images, previous, **kwargs)

        with patch("ni_rua.perception.view_selection.select_observation_views", side_effect=spy_select):
            self.deliver(self.h.observe())
        self.assertEqual(len(seen_previous), 1)
        self.assertIsNot(seen_previous[0], self.h._last_delivered_views)

        # 2) focus_roi 触发自动裁剪（effective_roi 非 None）不得覆盖未裁剪的视图基线
        baseline_head_obs = self.h._last_delivered_views["cam_head"]["observation_id"]
        self.h.image_settings["action_feedback_policy"] = "focus_roi"
        obs_crop = self.h.observe()
        with patch("ni_rua.runtime.observation_pipeline._action_focus_roi", return_value=[0, 0, 64, 64]):
            self.deliver(obs_crop)
        self.assertEqual(self.h._last_delivered_views["cam_head"]["observation_id"], baseline_head_obs)

        # 3) 过时纪元调用不得更新或清空当前活跃纪元的视图基线
        stale_obs = self.h.observe()
        self.h.observation_content(
            stale_obs,
            {"context_scope": "main", "context_epoch": ("stale-epoch",)},
            feedback_action=self.action,
            feedback_receipt=self.receipt,
        )
        self.assertEqual(self.h._last_delivered_views["cam_head"]["observation_id"], baseline_head_obs)


if __name__ == "__main__":
    unittest.main()
