"""Consolidated tests:

- test_benchmark_terminal_policy: Official benchmark endings cannot be supplied by actor prose or turn status.
- test_mixed_backend_benchmark: P4-1: mixed-backend benchmark integration (design 04 sections 4.6, 5.5, 7.3, 9.6).
- test_robot_matrix: CPU contract tests for the fixed matrix; no real simulator acceptance claim.
"""

import json
import asyncio
import threading
import io
from types import SimpleNamespace
from PIL import Image
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ni_rua.core.storage.store import Store
from ni_rua.runtime.harness import Harness
from ni_rua.core.contracts import digest
from ni_rua.perception.scene.store import SceneStore
from ni_rua.runtime.supervision import wait_turn
from ni_rua.apps.evaluation.episode import observer_summary
from ni_rua.apps.library.scoring import classify_episode_result
from support.geometry_fixture import IDENTITY, recorded_plane
from scripts.benchmark_report import summarize as script_summarize
from ni_rua.cortex.session.usage import account_usage, load_price_card
from support.fixture_robot import FixtureRobot
import time
from unittest.mock import Mock, patch
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.cortex.supervision.completion_guidance import COMPLETION_GUIDANCE_MODES, HOOK_COMPLETION_GUIDANCE
from ni_rua.apps.evaluation.episode import BENCHMARK_CONTINUATION, wait_benchmark
from ni_rua.apps.gateway.workbench import Workbench
from support.host_capabilities import program_isolation, require
from unittest import mock
from support.agent_fixtures import MCP_IDENTITY, McpSessionBase as McpCase, FakeProxy, StubSession
from support.paths import REPO_ROOT
from ni_rua.apps.evaluation.report import build_report, summarize
from ni_rua.apps.library.experiment_comparisons import ExperimentGroups
from ni_rua.apps.library.experiments import Catalog, atomic_json, register_run
from ni_rua.core.config import Config
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents.codex.session import CodexSession, MODEL_COMPLETION_COUNTING
from ni_rua.cortex.session.usage import (COUNTING_EQUIVALENCE, RelayUsageTotal, add_served_model, counting_basis,
                                         responses_counters)
from scripts import benchmark_queue as queue
from contextlib import ExitStack
import numpy as np
from ni_rua.robot.backends.robodojo.robodojo import RobotBackend
from ni_rua.apps.evaluation.robot_matrix import run_checks, run_validation
from support.motion_fixtures import _DualRuntime


NI_RUA_TEST_TIER = "component"

# --- test_benchmark.py ---
class BenchmarkReportCheck(unittest.TestCase):
    def test_model_limit_is_recognized_before_delayed_callback_and_failed_turn(self):
        async def check():
            directory = self.enterContext(tempfile.TemporaryDirectory())
            store = Store(directory, IDENTITY)
            self.addCleanup(store.close)
            harness = Harness(store, FixtureRobot(directory))
            harness.start()
            done = threading.Event()
            done.set()
            session = SimpleNamespace(turn_done=done, model_limit_reached=threading.Event(),
                                      model_proxy=SimpleNamespace(limit_notified=True))
            self.assertFalse(await wait_turn(session, asyncio.Event(), None, harness))
            self.assertEqual(harness.wait_stop_reason, "model_request_limit")
            result = dict(valid_episode=True, task_success=False, normalized_score=0,
                          native_episode=dict(ended=False), model_budget_exhausted=True,
                          termination_reason=harness.wait_stop_reason)
            self.assertEqual(classify_episode_result(result, "failed")["primary_outcome"], "scored_failure")
            self.assertEqual(classify_episode_result({**result, "infrastructure_error": "robot transport error"},
                                                    "failed")["primary_outcome"], "unscored_infrastructure")
            self.assertEqual(classify_episode_result(result, "unknown")["primary_outcome"], "unscored_infrastructure")
        asyncio.run(check())

    def test_partial_native_score_is_not_a_policy_failure(self):
        partial = dict(valid_episode=True, task_success=False, normalized_score=0,
                       native_episode=dict(ended=False), infrastructure_error="request timeout")
        value = classify_episode_result(partial, "failed")
        self.assertEqual(value["primary_outcome"], "unscored_infrastructure")
        self.assertIsNone(value["task_success"])
        self.assertFalse(value["partial_native_result"]["task_success"])
        native = {**partial, "native_episode": dict(ended=True), "task_success": True, "normalized_score": 1}
        self.assertEqual(classify_episode_result(native, "completed")["primary_outcome"], "scored_success")
        failure = {**partial, "infrastructure_error": None, "termination_reason": "model_request_limit"}
        self.assertEqual(classify_episode_result(failure, "stopped")["primary_outcome"], "scored_failure")
        self.assertEqual(classify_episode_result({**partial, "infrastructure_error": None}, "stopped")["primary_outcome"], "unscored_interrupted")

    def test_budgets_remain_private_and_native_terminal_closes_motion(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(directory) / "store", IDENTITY)
        self.addCleanup(store.close)
        robot = FixtureRobot(directory)
        harness = Harness(store, robot)
        initial = harness.start()
        self.assertIsNone(harness.max_actions)
        def call(name, args=None):
            result = harness.dispatch(name, args or {}, dict(tool_call_id="budget-test"))
            return json.loads(result["contentItems"][0]["text"])
        self.assertNotIn("runtime_budget", call("robot_capabilities"))
        self.assertNotIn("runtime_budget", call("history_view_observation", dict(observation_id=initial["observation_id"])))
        execute = robot.execute
        def native_execute(*args, **kwargs):
            return {**execute(*args, **kwargs), "episode_terminal": dict(ended=True, success=True, reason="success", native_action_count=17, native_step_limit=200)}
        robot.execute = native_execute
        harness.capabilities.update(bounded_wait=dict(max_native_steps=50), path_kinds=["backend_free", "hold"])
        args = dict(observation_id=initial["observation_id"], native_steps=5, action_id="terminal")
        result = call("robot_wait", args)
        self.assertNotIn("episode_terminal", result["receipt"])
        self.assertNotIn("runtime_budget", result)
        self.assertTrue(harness.episode_done.is_set())
        self.assertFalse(harness.accepting)
        self.assertEqual(store.read_state("runtime", "native_episode")["native_action_count"], 17)
        rejected = call("robot_wait", {**args, "action_id":"after-success", "observation_id":harness.latest["observation_id"]})
        self.assertEqual(rejected["receipt"]["status"], "rejected")
        self.assertEqual(robot.executions, 1)

    def test_inspection_continues_without_action_and_no_custom_benchmark_deadline(self):
        async def check():
            directory = self.enterContext(tempfile.TemporaryDirectory())
            store = Store(directory, IDENTITY)
            self.addCleanup(store.close)
            harness = Harness(store, FixtureRobot(directory))
            initial = harness.start()
            for _ in range(16):
                self.assertTrue(harness.dispatch("robot_capabilities", {}, {})["success"])
                self.assertTrue(harness.dispatch("history_view_observation",
                    dict(observation_id=initial["observation_id"]), {})["success"])
            self.assertTrue(harness.dispatch("robot_observe", {}, {})["success"])
            self.assertEqual(harness.robot.executions, 0)
            session = SimpleNamespace(turn_done=threading.Event())
            harness._native_progress(dict(ended=True, success=False, reason="step_limit", native_action_count=200, native_step_limit=200))
            self.assertFalse(await wait_turn(session, asyncio.Event(), None, harness))
            self.assertIsNone(harness.wall_deadline)
            self.assertEqual(harness.wait_stop_reason, "native_episode_ended")
        asyncio.run(check())

    def test_developer_snapshot_remains_immutable_but_cannot_enter_actor_tools(self):
        fixture = json.loads((REPO_ROOT / "contracts/fixtures/scene_plane_box.json").read_text())
        directory = self.enterContext(tempfile.TemporaryDirectory())
        identity = {key: fixture["observations"][0][key] for key in IDENTITY}
        store = Store(Path(directory) / "store", identity)
        self.addCleanup(store.close)
        harness = Harness(store, object())
        harness.scene = SceneStore(Path(directory) / "scene")
        for observation in fixture["observations"]: harness.scene.import_observation(observation)
        harness.scene.apply_patch(fixture["patches"][0])
        with self.assertRaisesRegex(ValueError, "unknown_tool"):
            harness.dispatch("scene_snapshot", dict(scene_version=1, anchor_observation_id="obs-scene-0"), {})
        manifest = harness.scene.snapshot(identity, 1, "obs-scene-0")
        self.assertEqual(manifest, harness.scene.read_snapshot(identity, manifest["snapshot_id"]))
        expected = manifest.pop("manifest_digest")
        self.assertEqual(digest(manifest), expected)

    def test_evaluation_is_current_and_stop_confirmed(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, IDENTITY)
            try:
                score = dict(task_success=True, normalized_score=1, valid_episode=True,
                    native_reward=1.0, native_process_score=100.0,
                    authority="installed RoboDojo task.run_reward / RewardManager",
                    success_authority="locked_native_terminal")
                ref = store.put_artifact(json.dumps({**IDENTITY, "result": score}).encode(), "native_evaluation", "application/json", "observer")
                def evaluate(): store.event("observer.evaluated", {"result_ref": ref})
                evaluate()
                self.assertEqual(observer_summary(store, IDENTITY, "completed"), score)
                self.assertEqual(observer_summary(store, IDENTITY, "unknown"), {})
                self.assertEqual(observer_summary(store, {**IDENTITY, "env_generation": 2}, "completed"), {})
                for change in ("action.requested", "environment.reset_requested", "environment.recovery_reset"):
                    store.event(change, {})
                    self.assertEqual(observer_summary(store, IDENTITY, "completed"), {})
                    evaluate()
                    self.assertEqual(observer_summary(store, IDENTITY, "completed"), score)
            finally:
                store.close()

    def test_optional_crop_preserves_original_and_records_pixel_transform(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, IDENTITY)
            try:
                observation = recorded_plane(store)[0]
                observation.update(observation_profile="rgb_proprio_v1", calibration_id=None)
                observation.pop("robot_model_ref")
                observation["cameras"][0].pop("calibration_ref")
                observation["robot_state_ref"] = store.put_artifact(b'{"arms":{}}', "robot_state", "application/json")
                store.save("observation", observation["observation_id"], observation)
                harness = Harness(store, object())
                original_ref = observation["cameras"][0]["rgb_ref"]
                original_bytes = store.resolve_artifact(original_ref)[1]
                result = harness.dispatch("history_view_observation", dict(observation_id="obs-0", camera_id="head", roi=[100, 80, 40, 30], scale=2), {})
                view = json.loads(result["contentItems"][0]["text"])["image_view"]
                self.assertTrue(view["original_rgb_available"])
                self.assertTrue(view["rgb_available"])
                delivered = store.records("delivery")[-1]["images"][0]
                self.assertEqual(delivered["original_rgb_ref"], original_ref)
                self.assertEqual(view["pixel_transform"], [[2, 0, -199.5], [0, 2, -159.5], [0, 0, 1]])
                with Image.open(io.BytesIO(store.resolve_artifact(delivered["rgb_ref"])[1])) as image:
                    self.assertEqual(image.size, (80, 60))
                self.assertEqual(store.resolve_artifact(original_ref)[1], original_bytes)
                self.assertEqual(len(result["contentItems"]), 2)
                with self.assertRaisesRegex(ValueError, "out_of_bounds"):
                    harness.dispatch("history_view_observation", dict(observation_id="obs-0", camera_id="head", roi=[600, 400, 100, 100]), {})
            finally:
                store.close()

    def test_shutdown_interrupts_long_turn_wait(self):
        async def check():
            session = SimpleNamespace(turn_done=threading.Event())
            shutdown = asyncio.Event()
            waiter = asyncio.create_task(wait_turn(session, shutdown, 1200))
            await asyncio.sleep(.01)
            shutdown.set()
            self.assertFalse(await asyncio.wait_for(waiter, .5))
            session.turn_done.set()
            self.assertTrue(await wait_turn(session, asyncio.Event(), 1200))
        asyncio.run(check())

    def test_calibrated_overlay_cannot_cross_into_actor_store(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            main, twin = Store(root / "main", IDENTITY), Store(root / "twin", IDENTITY)
            try:
                harness = Harness(main, object())
                harness.scene = SceneStore(root / "scene", twin.resolve_artifact, twin.put_artifact)
                for observation in recorded_plane(twin): harness.scene.import_observation(observation)
                with self.assertRaisesRegex(ValueError, "tool_unavailable"):
                    harness.dispatch("geometry_query", {"request": {"kind": "project_points",
                        "observation_id": "obs-0", "camera_id": "head", "points_m": [[0, 0, 2]]}}, {})
                self.assertFalse(main.events())
            finally:
                main.close(); twin.close()

    def test_costs_and_failure_remain_distinct_from_motion_completion(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = dict(run_id="pilot", episode_id="ep", env_instance_id="env", env_generation=1)
            store = Store(root, identity)
            try:
                store.save("tool_call", "call", {"params": {"tool": "robot_move_ee"}})
                store.save("delivery", "model", {"delivered_to_actor": True, "images": [{}, {}, {}]})
                store.save("delivery", "program", {"delivered_to_actor": False, "images": [{}]})
            finally:
                store.close()
            (root / "manifest.json").write_text(json.dumps(dict(task="general_pickup", seed=0, layout_id=0)))
            (root / "summary.json").write_text(json.dumps(dict(status="completed", action_count=1,
                valid_episode=True, task_success=False, normalized_score=0,
                token_usage={"total": {"input_tokens": 12000, "cached_input_tokens": 10000, "output_tokens": 500}})))
            rollout_dir = root / "private" / "codex-home" / "sessions"
            rollout_dir.mkdir(parents=True)
            with (rollout_dir / "synthetic.jsonl").open("w") as stream:
                for kind in ("custom_tool_call_output", "function_call_output"):
                    stream.write(json.dumps({"type": "response_item", "payload": {"type": kind,
                        "output": [{"type": "input_image"}, {"type": "input_text", "text": "fixture"}]}}) + "\n")
                stream.write(json.dumps({"type": "event_msg", "payload": {"type": "item_completed", "output": [{"type": "input_image"}]}}) + "\n")
            result = script_summarize(root)
            self.assertFalse(result["task_success"])
            self.assertEqual(result["uncached_input_tokens"], 2000)
            self.assertEqual(result["actor_image_deliveries"], 3)
            self.assertEqual(result["native_model_input_images"], 2)
            self.assertEqual(result["actor_tools"], {"robot_move_ee": 1})
            self.assertEqual(result["token_accounting"]["tokens"]["uncached_input_tokens"], 2000)
            self.assertIsNone(result["token_accounting"]["rate_card_estimate"]["total"])
            # A frozen per-run provider plus an explicit price card can price old
            # archives without consulting today's global model/configuration.
            (root / "manifest.json").write_text(json.dumps(dict(model="fixture", task="general_pickup")))
            (root / "private/codex-home/config.toml").write_text('model = "fixture"\nmodel_provider = "test-provider"\n')
            saved = json.loads((root / "summary.json").read_text())
            saved["token_usage"]["total"]["cache_write_input_tokens"] = 0
            (root / "summary.json").write_text(json.dumps(saved))
            card = root / "synthetic-price-card.json"
            card.write_text(json.dumps(dict(version="synthetic", source="synthetic_fixture", currency="USD", rates=[dict(
                provider="test-provider", model="fixture", uncached_input_per_million="2",
                cached_input_per_million="0.5", output_per_million="8")])))
            priced = script_summarize(root, load_price_card(card))["token_accounting"]
            self.assertEqual(priced["provider"], "test-provider")
            self.assertEqual(priced["rate_card_estimate"]["total"], "0.013")
            self.assertEqual(priced["identity_source"], "private/codex-home/config.toml")
            (root / "private/codex-home/config.toml").write_text('model = "different-model"\nmodel_provider = "test-provider"\n')
            self.assertIsNone(script_summarize(root, load_price_card(card))["token_accounting"]["rate_card_estimate"]["total"])
            # A TUI failure after model work leaves a cost-bearing summary before
            # manifest creation. Preserve its frozen estimate, or reprice with
            # an explicit card, without using today's/global model config.
            saved.update(status="failed", infrastructure_error="synthetic terminal startup failure")
            saved["token_usage"].update(thread_id="failed-thread", turn_id="turn", model_context_window=258400)
            saved["token_accounting"] = account_usage(saved["token_usage"], model="fixture",
                                                       provider="test-provider", price_card=load_price_card(card))
            (root / "manifest.json").unlink()
            (root / "summary.json").write_text(json.dumps(saved))
            failed = script_summarize(root)
            self.assertEqual(failed["status"], "failed")
            self.assertEqual(failed["conditions"]["model"], "fixture")
            self.assertEqual(failed["token_accounting"]["rate_card_estimate"]["total"], "0.013")
            self.assertEqual(failed["token_accounting"]["identity_source"], "summary.token_accounting")
            changed_card = json.loads(card.read_text())
            changed_card["rates"][0]["uncached_input_per_million"] = "4"
            card.write_text(json.dumps(changed_card))
            self.assertEqual(script_summarize(root, load_price_card(card))["token_accounting"]["rate_card_estimate"]["total"], "0.017")
            self.assertEqual(script_summarize(root)["token_accounting"]["rate_card_estimate"]["total"], "0.013")
            for key, changed in (("thread_id", "foreign-thread"), ("model_context_window", 999999),
                                 ("tokens", {})):
                stale = {**saved, "token_accounting": {**saved["token_accounting"], key: changed}}
                (root / "summary.json").write_text(json.dumps(stale))
                self.assertIsNone(script_summarize(root)["token_accounting"]["rate_card_estimate"]["total"])
                self.assertIsNone(script_summarize(root, load_price_card(card))["token_accounting"]["rate_card_estimate"]["total"])
            (root / "summary.json").write_text(json.dumps(dict(status="failed", task_success=None)))
            failed = script_summarize(root)
            self.assertIsNone(failed["task_success"])
            self.assertIsNone(failed["token_usage"])
            self.assertIsNone(failed["uncached_input_tokens"])


# --- test_benchmark_terminal_policy.py ---
class BenchmarkTerminalPolicyCheck(unittest.IsolatedAsyncioTestCase):
    def workbench(self, mode="none"):
        done = threading.Event()
        done.set()
        session = SimpleNamespace(state="completed", turn_done=done, turn_id="first-turn", thread_id="same-thread",
            turn_result={"status": "completed", "answer": "I have completed the task successfully."},
            model_limit_reached=threading.Event(), model_proxy=SimpleNamespace(limit_notified=False, request_count=3))
        harness = SimpleNamespace(episode_done=threading.Event(), episode_terminal=None, accepting=True,
            native_action_count=0, started_at=time.monotonic(), last_native_progress_at=time.monotonic(), active_action=None,
            home_check=SimpleNamespace(check=Mock(return_value={"status": "away", "initial_observation_id": None})))
        return SimpleNamespace(session=session, harness=harness, completion_guidance=mode, store=SimpleNamespace(event=Mock()),
            identity={"episode_id": "fixture-episode"},
            current_unknown_actions=lambda: False, has_active_programs=lambda: False, control_lock=asyncio.Lock())

    def paused_program(self, workbench):
        from support.motion_fixtures import ProgramReviewFixture
        fixture = ProgramReviewFixture()
        self.addCleanup(fixture.close)
        job = fixture.start("from ni_rua_client import NiRuaClient\nNiRuaClient().action([{'id':'hold','kind':'gripper','arm':'left','goal_normalized':1}],observation_id='obs-start')")
        self.assertEqual(job['status'], 'awaiting_review')
        workbench.harness.programs = fixture.runner
        workbench.has_active_programs = lambda: Workbench.has_active_programs(workbench)
        workbench.program_continuation_state = lambda: Workbench.program_continuation_state(workbench)
        return fixture, job

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    async def test_review_paused_program_gets_continuation_and_can_resume_same_process(self):
        workbench = self.workbench()
        fixture, job = self.paused_program(workbench)
        entry = fixture.runner.entries[job['job_id']]
        self.assertTrue(workbench.has_active_programs())
        self.assertFalse(workbench.program_continuation_state()['blocked'])
        def continue_turn(prompt):
            self.assertIn(job['job_id'], prompt)
            self.assertIn(job['pending_review']['review_id'], prompt)
            self.assertIn("operation='resume'", prompt)
            self.assertEqual(fixture.action_count, 0)
            final = fixture.resume(job)
            self.assertEqual(final['status'], 'succeeded')
            self.assertEqual(fixture.action_count, 1)
            workbench.harness.episode_done.set()
        workbench.session.run_turn = Mock(side_effect=continue_turn)
        self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
        workbench.session.run_turn.assert_called_once()
        entry['worker'].join(1)
        self.assertFalse(workbench.has_active_programs())

    async def test_running_uncertain_or_unconfirmed_program_still_blocks_continuation(self):
        for state in ('running', 'unknown', 'unconfirmed_pause', 'cancel_requested', 'missing_review'):
            with self.subTest(state=state):
                workbench = self.workbench()
                cancel = threading.Event()
                if state == 'cancel_requested': cancel.set()
                job = dict(status=state if state in {'running', 'unknown'} else 'awaiting_review',
                    pending_review=dict(review_id='review', decision='pending'))
                if state == 'missing_review': job.pop('pending_review')
                entry = dict(job_id='program', job=job, cancel=cancel,
                    condition=threading.Condition(), process=SimpleNamespace(poll=lambda: None),
                    process_paused=state != 'unconfirmed_pause')
                from ni_rua.runtime.programs.runner import ProgramRunner
                runner = ProgramRunner.__new__(ProgramRunner)
                runner.lock, runner.entries = threading.RLock(), {'program': entry}
                runner.store = SimpleNamespace(states=lambda _: [])
                workbench.harness.programs = runner
                workbench.program_continuation_state = lambda: Workbench.program_continuation_state(workbench)
                workbench.session.run_turn = Mock()
                self.assertTrue(workbench.program_continuation_state()['blocked'])
                self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                workbench.session.run_turn.assert_not_called()

    @require(program_isolation, "bounded actor programs run under the Landlock + seccomp program sandbox")
    async def test_stop_still_cancels_and_reaps_review_paused_program(self):
        workbench = self.workbench()
        fixture, job = self.paused_program(workbench)
        entry = fixture.runner.entries[job['job_id']]
        workbench.control_responses = {}
        workbench.starting = False
        workbench.harness.request_stop = Mock(side_effect=lambda: setattr(workbench.harness, 'accepting', False))
        workbench.session.interrupt = Mock(side_effect=lambda: setattr(workbench.session, 'state', 'interrupted'))
        self.assertTrue(workbench.has_active_programs())
        result = await Workbench.stop(workbench, 'stop-paused-program')
        self.assertEqual(result['status'], 'stopped')
        self.assertFalse(workbench.has_active_programs())
        self.assertIsNotNone(entry['process'].poll())
        self.assertEqual(fixture.action_count, 0)
        final = fixture.runner.status(job['job_id'])
        self.assertEqual(final['status'], 'cancelled')
        self.assertTrue(final['stop_confirmed'])

    async def test_early_final_continues_same_episode_until_official_success_or_failure(self):
        for success in (False, True):
            with self.subTest(native_success=success):
                workbench = self.workbench()
                harness, session = workbench.harness, workbench.session
                before_clock, proxy = harness.started_at, session.model_proxy
                def continue_turn(prompt):
                    self.assertEqual(prompt, "任务还没完成，请继续。")
                    session.turn_id = "second-turn"
                    session.model_proxy.request_count += 1
                    harness.native_action_count = 4
                    harness.episode_terminal = {"ended": True, "success": success,
                        "reason": "native_success" if success else "native_step_limit"}
                    harness.episode_done.set()
                    harness.accepting = False
                session.run_turn = Mock(side_effect=continue_turn)
                self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                session.run_turn.assert_called_once()
                self.assertEqual(session.thread_id, "same-thread")
                self.assertIs(session.model_proxy, proxy)
                self.assertEqual(session.model_proxy.request_count, 4)
                self.assertEqual(harness.started_at, before_clock)
                self.assertEqual(harness.actor_continuation_count, 1)
                self.assertEqual(harness.wait_stop_reason, "native_episode_ended")

    async def test_repeated_early_finals_do_not_reset_model_budget(self):
        for mode in COMPLETION_GUIDANCE_MODES:
            with self.subTest(mode=mode):
                workbench = self.workbench(mode)
                session, harness = workbench.session, workbench.harness
                before_progress, before_start = harness.last_native_progress_at, harness.started_at
                proxy = session.model_proxy
                def continue_turn(prompt):
                    session.model_proxy.request_count += 1
                    if session.model_proxy.request_count == 5:
                        session.model_proxy.limit_notified = True
                session.run_turn = Mock(side_effect=continue_turn)
                self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                expected = HOOK_COMPLETION_GUIDANCE if mode in {"hook", "both"} else BENCHMARK_CONTINUATION
                self.assertEqual(session.run_turn.call_count, 2)
                self.assertTrue(all(call.args == (expected,) for call in session.run_turn.call_args_list))
                self.assertIs(session.model_proxy, proxy)
                self.assertEqual(session.model_proxy.request_count, 5)
                self.assertEqual(session.thread_id, "same-thread")
                self.assertEqual(harness.actor_continuation_count, 2)
                self.assertEqual(harness.wait_stop_reason, "model_request_limit")
                self.assertEqual(harness.last_native_progress_at, before_progress)
                self.assertEqual(harness.started_at, before_start)
                self.assertFalse(harness.episode_done.is_set())
                for count, call in enumerate(workbench.store.event.call_args_list, 1):
                    self.assertEqual(call.args[0], "benchmark.actor_turn_continuation")
                    self.assertEqual(call.args[1], dict(count=count, previous_turn_id="first-turn",
                        thread_id="same-thread", reason="actor_turn_completed_before_native_terminal",
                        completion_guidance=mode, trigger="turn.completed", prompt=expected, initial_observation_id=None,
                        continuation_kind="home_reminder" if mode in {"hook", "both"} else "generic",
                        home_check={"status": "away", "initial_observation_id": None} if mode in {"hook", "both"} else None))

    async def test_native_terminal_budget_timeout_and_unknown_never_launch_continuation(self):
        for mode in COMPLETION_GUIDANCE_MODES:
            for condition in ("native_success", "native_failure", "budget", "timeout", "progress_timeout",
                              "unknown", "shutdown", "not_accepting", "active_action", "active_program"):
                with self.subTest(mode=mode, condition=condition):
                    workbench = self.workbench(mode)
                    shutdown = asyncio.Event()
                    workbench.session.run_turn = Mock()
                    if condition.startswith("native_"):
                        workbench.harness.episode_terminal = {"ended": True, "success": condition == "native_success"}
                        workbench.harness.episode_done.set()
                    if condition == "budget": workbench.session.model_limit_reached.set()
                    if condition == "timeout": workbench.harness.started_at -= 1801; workbench.harness.last_native_progress_at -= 1801
                    if condition == "progress_timeout":
                        workbench.harness.native_action_count = 1
                        workbench.harness.started_at -= 1801
                        workbench.harness.last_native_progress_at -= 1801
                    if condition == "unknown": workbench.current_unknown_actions = lambda: True
                    if condition == "shutdown": shutdown.set()
                    if condition == "not_accepting": workbench.harness.accepting = False
                    if condition == "active_action": workbench.harness.active_action = "unfinished"
                    if condition == "active_program": workbench.has_active_programs = lambda: True
                    self.assertFalse(await wait_benchmark(workbench, shutdown))
                    workbench.session.run_turn.assert_not_called()
                    workbench.store.event.assert_not_called()
                    workbench.harness.home_check.check.assert_not_called()

    async def test_turn_failure_or_interruption_never_launches_hook(self):
        for mode in COMPLETION_GUIDANCE_MODES:
            for state in ("failed", "interrupted"):
                with self.subTest(mode=mode, state=state):
                    workbench = self.workbench(mode)
                    workbench.session.state = state
                    workbench.session.run_turn = Mock()
                    self.assertTrue(await wait_benchmark(workbench, asyncio.Event()))
                    workbench.session.run_turn.assert_not_called()
                    workbench.store.event.assert_not_called()

    async def test_hook_uses_completed_turn_status_without_matching_actor_prose(self):
        for answer in ("Done.", "I cannot continue.", "", None):
            with self.subTest(answer=answer):
                workbench = self.workbench("hook")
                workbench.session.turn_result["answer"] = answer
                workbench.session.run_turn = Mock(side_effect=lambda prompt: workbench.harness.episode_done.set())
                self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                workbench.session.run_turn.assert_called_once_with(HOOK_COMPLETION_GUIDANCE)

    async def test_home_or_unknown_uses_exact_original_continuation(self):
        for mode in ("hook", "both"):
            for status in ("home", "unknown", "unavailable"):
                with self.subTest(mode=mode, status=status):
                    workbench = self.workbench(mode)
                    if status == "unavailable":
                        workbench.harness.home_check = None
                    else:
                        workbench.harness.home_check.check.return_value = {"status": status}
                    workbench.session.run_turn = Mock(side_effect=lambda prompt: workbench.harness.episode_done.set())
                    self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                    workbench.session.run_turn.assert_called_once_with("任务还没完成，请继续。")
                    record = workbench.store.event.call_args.args[1]
                    self.assertEqual(record["continuation_kind"], "generic")
                    self.assertEqual(record["home_check"]["status"], "unknown" if status == "unavailable" else status)

    async def test_hook_rechecks_supervisor_and_unresolved_work_after_control_lock(self):
        for condition in ("native", "budget", "timeout", "unknown", "active_action", "active_program"):
            with self.subTest(condition=condition):
                workbench = self.workbench("hook")
                shutdown = asyncio.Event()
                workbench.session.run_turn = Mock()
                class ContendedLock:
                    async def __aenter__(self):
                        if condition == "native": workbench.harness.episode_done.set()
                        if condition == "budget": workbench.session.model_limit_reached.set()
                        if condition == "timeout": workbench.harness.started_at -= 1801; workbench.harness.last_native_progress_at -= 1801
                        if condition == "unknown": workbench.current_unknown_actions = lambda: True
                        if condition == "active_action": workbench.harness.active_action = "unfinished"
                        if condition == "active_program": workbench.has_active_programs = lambda: True
                    async def __aexit__(self, *arguments):
                        pass
                workbench.control_lock = ContendedLock()
                self.assertFalse(await wait_benchmark(workbench, shutdown))
                workbench.session.run_turn.assert_not_called()
                workbench.store.event.assert_not_called()
                workbench.harness.home_check.check.assert_not_called()

    async def test_dev_exit_after_turn_retains_single_turn_behavior(self):
        workbench = self.workbench()
        workbench.session.run_turn = Mock()
        self.assertTrue(await wait_turn(workbench.session, asyncio.Event(), 10, workbench.harness))
        workbench.session.run_turn.assert_not_called()
        self.assertFalse(workbench.harness.episode_done.is_set())

    async def test_disconnect_after_an_early_final_is_infrastructure_failure(self):
        workbench = self.workbench()
        workbench.session.state = "disconnected"
        workbench.session.run_turn = Mock()
        with self.assertRaisesRegex(RuntimeError, "benchmark_actor_disconnected"):
            await wait_benchmark(workbench, asyncio.Event())
        workbench.session.run_turn.assert_not_called()

    def test_existing_codex_continuation_reuses_thread_and_transport(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        store = Store(temporary.name, {})
        self.addCleanup(store.close)
        session = CodexSession.__new__(CodexSession)
        session.store, session.presentation = store, ActorPresentation(store)
        session.state, session.thread_id, session.model, session.effort = "completed", "same-thread", "fixture", "high"
        session.projection_lock = threading.Lock()
        session.turn_revision = 0
        session.turn_id, session.turn_result = "previous", None
        session.turn_done = threading.Event()
        session.turn_done.set()
        session.model_proxy = SimpleNamespace(request_count=169)
        session.client = SimpleNamespace(turn_start=Mock(return_value=SimpleNamespace(turn=SimpleNamespace(id="next"))))
        self.assertEqual(session.run_turn("continue"), "next")
        self.assertEqual(session.client.turn_start.call_args.args[0], "same-thread")
        self.assertEqual(session.model_proxy.request_count, 169)
        self.assertFalse(session.turn_done.is_set())

    async def test_pending_response_does_not_exempt_thirty_minutes_of_silence(self):
        h, s = self.workbench().harness, self.workbench().session
        h.started_at = h.last_native_progress_at = time.monotonic() - 1801
        s.model_proxy.pending_requests = True
        s.model_proxy.model_wait_seconds = 100000.0
        self.assertFalse(await wait_turn(s, asyncio.Event(), harness=h))
        self.assertEqual(h.wait_stop_reason, "runtime_inactivity_timeout")

    async def test_planner_gpu_oom_stops_turn_as_unscored_retryable_infrastructure(self):
        from ni_rua.apps.evaluation.retry import oom_evidence
        from ni_rua.robot.backends.robodojo.robot_process import GPU_OOM_FAILURE
        w = self.workbench()
        w.session.turn_done.clear()
        w.harness.robot = SimpleNamespace(failure="worker disconnected")
        w.harness.episode_done.set()
        self.assertFalse(await wait_turn(w.session, asyncio.Event(), harness=w.harness))
        self.assertEqual(w.harness.wait_stop_reason, "native_episode_ended")
        w.harness.episode_done.clear()
        w.harness.robot.failure = GPU_OOM_FAILURE
        self.assertFalse(await wait_turn(w.session, asyncio.Event(), harness=w.harness))
        self.assertEqual(w.harness.wait_stop_reason, "robot_gpu_out_of_memory")
        # A planner OOM is not a valid zero: it stays out of the score and qualifies for OOM retry.
        result = classify_episode_result(dict(benchmark=True, termination_reason=w.harness.wait_stop_reason,
            infrastructure_error=GPU_OOM_FAILURE, cleanup_status="completed", valid_episode=True,
            task_success=False, normalized_score=0, native_episode={"ended": False}), "unknown")
        self.assertEqual(result["primary_outcome"], "unscored_infrastructure")
        self.assertIsNone(result["normalized_score"])
        with tempfile.TemporaryDirectory() as directory:
            evidence = oom_evidence({"summary": result, "case": {"run_dir": directory}})
        self.assertEqual(evidence["signature"], "robot_gpu_out_of_memory:")

    def test_transport_recovery_gate_requires_settled_paused_world(self):
        w = self.workbench()
        w.closed = False
        block = lambda: Workbench.transport_recovery_block(w)
        self.assertIsNone(block())
        for change, reason in ((lambda: setattr(w.harness, "active_action", "action-1"), "action_in_progress"),
                               (lambda: setattr(w, "current_unknown_actions", lambda: True), "action_status_unknown"),
                               (lambda: setattr(w, "has_active_programs", lambda: True), "program_active"),
                               (lambda: w.harness.episode_done.set(), "native_episode_ended"),
                               (lambda: setattr(w.harness, "accepting", False), "harness_not_accepting"),
                               (lambda: setattr(w, "closed", True), "harness_not_accepting")):
            with self.subTest(reason=reason):
                change()
                self.assertEqual(block(), reason)
                w = self.workbench()
                w.closed = False

    async def test_inactivity_boundary_and_infrastructure_classification(self):
        w = self.workbench()
        w.harness.started_at = w.harness.last_native_progress_at = 100.0
        # Freeze only the supervision module's clock. Patching time.monotonic itself also freezes the asyncio loop
        # clock, so a wait_turn that reached its asyncio.sleep would never wake and the test would hang instead of
        # failing; with the loop clock real, wait_for turns such a regression into a failure.
        with patch("ni_rua.runtime.supervision.time", SimpleNamespace(monotonic=lambda: 1899.999)):
            self.assertTrue(await asyncio.wait_for(wait_turn(w.session, asyncio.Event(), harness=w.harness), 5))
        with patch("ni_rua.runtime.supervision.time", SimpleNamespace(monotonic=lambda: 1900.0)):
            self.assertFalse(await asyncio.wait_for(wait_turn(w.session, asyncio.Event(), harness=w.harness), 5))
        result = classify_episode_result(dict(benchmark=True, termination_reason=w.harness.wait_stop_reason,
            infrastructure_error=w.harness.wait_stop_reason, cleanup_status="completed",
            valid_episode=True, task_success=False, normalized_score=0, native_episode={"ended": False}), "stopped")
        self.assertEqual(result["primary_outcome"], "unscored_infrastructure")
        self.assertIsNone(result["task_success"])
        self.assertIsNone(result["normalized_score"])

    async def test_each_activity_source_keeps_a_long_running_case_alive(self):
        for source in ("session", "proxy", "robot", "programs", "native"):
            with self.subTest(source=source):
                w = self.workbench()
                h, s = w.harness, w.session
                h.started_at = h.last_native_progress_at = time.monotonic() - 100000
                h.robot, h.programs = SimpleNamespace(), SimpleNamespace()
                if source == "native":
                    h.last_native_progress_at = time.monotonic()
                else:
                    owner = {"session": s, "proxy": s.model_proxy, "robot": h.robot, "programs": h.programs}[source]
                    owner.last_activity_at = time.monotonic()
                self.assertTrue(await wait_turn(s, asyncio.Event(), harness=h))
                self.assertIsNone(h.wait_stop_reason)
                h.last_native_progress_at = h.started_at
                if source != "native": owner.last_activity_at = time.monotonic() - 1801
                self.assertFalse(await wait_turn(s, asyncio.Event(), harness=h))
                self.assertEqual(h.wait_stop_reason, "runtime_inactivity_timeout")

    async def test_terminal_api_error_after_retry_policy_surfaces_without_continuation(self):
        workbench=self.workbench()
        workbench.session.model_proxy.transport_blocked=True
        workbench.session.model_proxy.blocked_error={'error_type':'provider_error','automatic_retries':12,'retry_exhausted':True}
        workbench.session.run_turn=Mock()
        with self.assertRaisesRegex(RuntimeError,'model_transport_terminal_error'):
            await wait_benchmark(workbench,asyncio.Event())
        workbench.session.run_turn.assert_not_called()

    def test_actor_final_is_not_terminal_and_task_limits_score_failure_from_current_snapshot(self):
        partial = dict(benchmark=True, valid_episode=True, task_success=True, normalized_score=1.,
            native_episode={"ended": False, "success": None}, native_reward=1., native_process_score=100.,
            authority="installed RoboDojo task.run_reward / RewardManager", success_authority="native_final_check")
        cases = [("completed", None, "unscored_interrupted"),
                 ("stopped", "model_request_limit", "scored_failure"),
                 ("stopped", "official_no_action_progress_timeout", "scored_failure")]
        for status, reason, expected in cases:
            with self.subTest(status=status, reason=reason):
                result = classify_episode_result({**partial, "termination_reason": reason}, status)
                self.assertEqual(result["primary_outcome"], expected)
                if expected == "scored_failure":
                    self.assertTrue(result["scored_episode"])
                    self.assertFalse(result["task_success"])
                    self.assertEqual(result["normalized_score"], 1.)
                    self.assertEqual(result["native_process_score"], 100.)
                else:
                    self.assertFalse(result["scored_episode"])
                    self.assertIsNone(result["task_success"])
                    self.assertIsNone(result["normalized_score"])
                    self.assertEqual(result["partial_native_result"]["native_process_score"], 100.)
                    self.assertEqual(result["partial_native_result"]["success_authority"], "native_final_check")
                self.assertEqual(result["termination_authority"], "harness")
        native = {**partial, "native_episode": {"ended": True, "success": True}, "success_authority": "locked_native_terminal"}
        self.assertEqual(classify_episode_result(native, "completed")["primary_outcome"], "scored_success")


# --- test_mixed_backend_benchmark.py ---
CODEX_COUNTING = "delivered_usable_actor_helper_compact_responses_v2"
NEW_COUNTING = "delivered_usable_generation_completions_v1"
BACKENDS = ("codex", "claude_code", "opencode")


def responses_usage(input_tokens, output_tokens, cached=0, reasoning=0):
    return dict(input_tokens=input_tokens, output_tokens=output_tokens, total_tokens=input_tokens + output_tokens,
                input_tokens_details=dict(cached_tokens=cached), output_tokens_details=dict(reasoning_tokens=reasoning))


class MixedManifestTests(unittest.TestCase):
    """One queue manifest holding Codex, Claude Code and OpenCode cases, 2 seeds each."""

    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.checkout = self.root / "checkout"
        (self.checkout / "ni_rua/apps/gateway").mkdir(parents=True)
        (self.checkout / "ni_rua/core").mkdir(parents=True)
        (self.checkout / "ni_rua/apps/gateway/cli.py").write_text("parser.add_argument('--config')\nConfig.load()\n")
        self.cases = [dict(cwd=str(self.checkout), run_dir=str(self.root / f"run-{backend}-{seed}"), task="stack_bowls",
                           seed=seed, layout_id=0, title=f"{backend} seed {seed}", tag=[backend],
                           experiment_condition=backend, agent_backend=backend, model="shared-model", effort="high")
                      for backend in BACKENDS for seed in (0, 1)]

    def test_mixed_manifest_validates_and_every_case_carries_its_backend_flag(self):
        manifest = queue.configured_manifest({"cases": self.cases}, Config())
        self.assertEqual(len(manifest["cases"]), 6)
        self.assertEqual(len({case["run_dir"] for case in manifest["cases"]}), 6)
        for case in manifest["cases"]:
            args = queue.command(case, 0, config=Config())
            for flag, value in (("--agent-backend", case["agent_backend"]), ("--model", "shared-model"),
                                ("--effort", "high"), ("--seed", str(case["seed"]))):
                self.assertEqual(args[args.index(flag) + 1], value, (case["agent_backend"], flag))
        # The shared config is never mutated by a case's backend: backend is a per-case condition.
        self.assertNotIn("agent", manifest["ni_rua_config"])

    def test_frozen_settings_record_each_backend_and_only_env_names(self):
        for backend in BACKENDS:
            settings = queue.agent_settings(Config(), backend)
            self.assertEqual(settings["agent_backend"], backend)
            self.assertNotIn("sk-", json.dumps(settings))

    def test_evidence_path_warning_when_backends_mix_evidence_clients(self):
        def config(backend, client="native"):
            return Config.from_dict({"features": {"visual_review_model": "gpt-6.1-sol"},
                                     "agent": {"backend": backend, "evidence": {"client": client}}})
        mixed = queue.evidence_path_warnings([config("codex"), config("claude_code"), config("opencode")])
        self.assertEqual([item["kind"] for item in mixed], ["comparison_evidence_path_mismatch"])
        self.assertEqual(mixed[0]["agent_backends"], ["claude_code", "codex", "opencode"])
        self.assertEqual(mixed[0]["evidence_clients"], {"codex_agent": [0], "relay": [1, 2]})
        self.assertEqual(queue.evidence_path_warnings(
            [config("codex", "relay"), config("claude_code"), config("opencode")]), [])

    def test_project_config_for_runtimes_supports_raw_and_migrated_v1_checkouts(self):
        from ni_rua.core.config import MOTION_THRESHOLD_FIELDS
        from ni_rua.core.config.thresholds import _DEFAULT_MOTION_THRESHOLDS

        repo_schema = queue.runtime_config_schema(REPO_ROOT)
        v1_checkout_schema = {
            name: set(fields_set)
            for name, fields_set in repo_schema.items()
            if name != "EmbodimentConfig"
        }
        v1_checkout_schema["Config"].discard("embodiment")
        v1_checkout_schema["FeaturesConfig"] = set(v1_checkout_schema["FeaturesConfig"]) | set(MOTION_THRESHOLD_FIELDS)

        migrated_default = Config.from_dict({
            "version": 1,
            "features": {**dict(_DEFAULT_MOTION_THRESHOLDS), "self_briefing": False},
        }).to_dict()
        projected_default = queue.project_config_for_runtimes(migrated_default, [v1_checkout_schema])
        self.assertEqual(projected_default["version"], 1)
        self.assertNotIn("embodiment", projected_default)

        migrated_override = Config.from_dict({
            "version": 1,
            "features": {**dict(_DEFAULT_MOTION_THRESHOLDS), "tight_position_tolerance_m": 0.0045},
        }).to_dict()
        projected_override = queue.project_config_for_runtimes(migrated_override, [v1_checkout_schema])
        self.assertEqual(projected_override["version"], 1)
        self.assertNotIn("embodiment", projected_override)
        self.assertEqual(projected_override["features"]["tight_position_tolerance_m"], 0.0045)


class ServedModelAndRelayUsageTests(unittest.TestCase):
    def test_counting_basis_treats_equivalent_strings_as_one_basis(self):
        self.assertEqual(counting_basis(CODEX_COUNTING), counting_basis(NEW_COUNTING))
        self.assertEqual(counting_basis(NEW_COUNTING), "all_delivered_usable_generation_completions")
        self.assertEqual(set(COUNTING_EQUIVALENCE.values()), {"all_delivered_usable_generation_completions"})
        # Undeclared strings are their own basis; missing stays unknown.
        self.assertEqual(counting_basis("custom_v1"), "custom_v1")
        self.assertIsNone(counting_basis(None))
        self.assertIsNone(counting_basis(""))

    def test_relay_total_makes_unknown_sticky_and_seeds_only_the_same_thread(self):
        total = RelayUsageTotal()
        total.add(dict(input_tokens=10, output_tokens=2))
        record = total.add(dict(input_tokens=5, output_tokens=1, cached_input_tokens=0, cache_write_input_tokens=0,
                                reasoning_output_tokens=0, total_tokens=6), thread_id="t")
        self.assertEqual((record["total"]["input_tokens"], record["calls"], record["unknown_calls"]), (15, 2, 1))
        self.assertIsNone(record["total"]["cached_input_tokens"])
        seeded = RelayUsageTotal(record, "t")
        self.assertEqual(seeded.calls, 2)
        self.assertEqual(RelayUsageTotal(record, "other").calls, 0)
        self.assertEqual(add_served_model(add_served_model({}, "m"), None), {"m": 1, "unknown": 1})

    def test_responses_counters_use_the_account_responses_mapping(self):
        counters = responses_counters(responses_usage(100, 20, cached=40, reasoning=5))
        self.assertEqual((counters["input_tokens"], counters["cached_input_tokens"], counters["output_tokens"],
                          counters["reasoning_output_tokens"], counters["total_tokens"]), (100, 40, 20, 5, 120))
        self.assertIsNone(counters["cache_write_input_tokens"])
        self.assertTrue(all(value is None for value in responses_counters(None).values()))


class McpServedModelsTests(McpCase):
    def test_served_models_aggregate_delivered_completions_and_survive_resume(self):
        session = self.session()
        session._begin_turn()
        session.model_proxy = FakeProxy([])
        session._relay_used, session._relay_attempts, session._attempt_usage = 0, 0, {}
        usage = {"input_tokens": 10, "output_tokens": 3}
        for index, served in enumerate(("alias-1", "alias-1", None, "alias-2"), 1):
            call = f"t:{index}"
            session.model_proxy.attempt_count = index
            session.model_proxy.request_count = min(index, 3)
            session._model_event(dict(kind="model_request_finished", logical_call_id=call, usage=usage,
                                      usage_known=True, served_model=served))
            if index == 4:   # an undelivered call (failed) must not be counted as served
                session._model_event(dict(kind="model_call_finished", logical_call_id=call))
            else:
                session._model_event(dict(kind="model_call_completed", logical_call_id=call,
                                          native_thread_id="thread-1"))
        requests = self.store.read_state("runtime", "model_requests")
        self.assertEqual(requests["served_models"], {"alias-1": 2, "unknown": 1})
        self.assertEqual(requests["completed_calls"], 3)
        self.assertEqual(self.store.read_state("runtime", "model_usage_relay")["calls"], 3)
        resumed = self.session()
        resumed._start_relay = None
        self.store.set_state("runtime", "model_requests", dict(requests, counting=resumed.counting))
        with mock.patch("ni_rua.cortex.agents.mcp_session.ModelRequestProxy") as proxy:
            proxy.return_value.start.return_value = None
            try:
                type(resumed)._start_relay(resumed, "http://127.0.0.1:1", upstream_auth_headers={},
                                           request_projection=lambda request: request, default_thread_id="thread-1",
                                           image_settings=dict(reuse_exact_images=False, trim_oversized_requests=False,
                                                               trim_max_inline_images=1))
            except Exception:   # only the seeded counter is under test, not the full relay construction
                pass
        self.assertEqual(resumed._served_models, {"alias-1": 2, "unknown": 1})


class CodexRelayRecordTests(unittest.TestCase):
    """CodexSession._prepare wires the relay measurement and the served-model ledger into the proxy callback."""

    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory()))
        self.store = Store(self.root, dict(MCP_IDENTITY, run_id="codex-run"))
        self.addCleanup(self.store.close)
        home = self.root / "fixture-config"
        home.mkdir()
        (home / "config.toml").write_text('model_provider = "fixture"\n[model_providers.fixture]\nname = "fixture"\n'
                                          'base_url = "http://127.0.0.1:1"\n')
        self.session = CodexSession(self.root, self.store, lambda value: {},
                                    config=Config.from_dict({"model": {"codex_home": str(home)}}))
        self.session.thread_id, self.session.turn_id, self.session.state = "thread-owned", "turn-owned", "running"
        self.addCleanup(self.session.close)

    def callback(self):
        with mock.patch("ni_rua.cortex.agents.codex.session.ModelRequestProxy", side_effect=RuntimeError("reached")) as proxy:
            with self.assertRaisesRegex(RuntimeError, "reached"):
                self.session._prepare()
        self.session.model_proxy = SimpleNamespace(request_count=0, attempt_count=0, max_retries=12,
                                                   transport_blocked=False, blocked_error=None,
                                                   transport_recovery_summary=dict(policy=None, rounds=0))
        self.addCleanup(setattr, self.session, "model_proxy", None)
        return proxy.call_args.kwargs["on_event"]

    def deliver(self, emit, call, usage, served, completed=True):
        emit(dict(kind="model_request_finished", logical_call_id=call, usage=usage, usage_known=usage is not None,
                  served_model=served))
        emit(dict(kind="model_call_completed" if completed else "model_call_finished", logical_call_id=call))

    def test_codex_runs_record_relay_usage_and_served_models(self):
        emit = self.callback()
        self.deliver(emit, "c1", responses_usage(100, 20, cached=40, reasoning=5), "served-a")
        self.deliver(emit, "c2", responses_usage(50, 10, cached=10, reasoning=1), "served-a")
        self.deliver(emit, "c3", responses_usage(7, 7), "served-b", completed=False)   # not delivered: not counted
        relay = self.store.read_state("runtime", "model_usage_relay")
        self.assertEqual((relay["calls"], relay["total"]["input_tokens"], relay["total"]["cached_input_tokens"],
                          relay["total"]["output_tokens"], relay["total"]["reasoning_output_tokens"]),
                         (2, 150, 50, 30, 6))
        # Codex does not report cache writes: that counter (and so the all-fields unknown count) stays unknown.
        self.assertIsNone(relay["total"]["cache_write_input_tokens"])
        requests = self.store.read_state("runtime", "model_requests")
        self.assertEqual(requests["served_models"], {"served-a": 2})
        self.assertEqual(requests["counting"], MODEL_COMPLETION_COUNTING)
        self.assertEqual(counting_basis(requests["counting"]), counting_basis(NEW_COUNTING))

    def test_resume_without_relay_record_keeps_earlier_calls_unknown_and_seeded_ledger_continues(self):
        self.store.set_state("runtime", "model_requests", dict(completed_calls=5, http_attempt_count=5,
                             counting=MODEL_COMPLETION_COUNTING, served_models={"old": 5}))
        emit = self.callback()
        self.deliver(emit, "c1", responses_usage(10, 2), "new")
        relay = self.store.read_state("runtime", "model_usage_relay")
        self.assertEqual((relay["calls"], relay["unknown_calls"]), (6, 6))
        self.assertIsNone(relay["total"]["input_tokens"])
        self.assertEqual(self.store.read_state("runtime", "model_requests")["served_models"], {"old": 5, "new": 1})


def make_run(root, *, manifest, summary, state=None, events=()):
    store = Store(root, dict(run_id=manifest.get("run_id", root.name), episode_id="e", env_instance_id="i",
                             env_generation=1))
    try:
        for (kind, identifier), value in (state or {}).items():
            store.set_state(kind, identifier, value)
        for kind, payload in events:
            store.event(kind, payload, source="private_model_transport")
    finally:
        store.close()
    (root / "manifest.json").write_text(json.dumps(manifest))
    (root / "summary.json").write_text(json.dumps(summary))
    return root


def relay_state(calls, inp, out, thread="t"):
    total = dict(input_tokens=inp, cached_input_tokens=0, cache_write_input_tokens=0, output_tokens=out,
                 reasoning_output_tokens=0, total_tokens=inp + out)
    return dict(total=total, last=total, model_context_window=None, thread_id=thread, turn_id="u", raw_ref=None,
                unknown_calls=0, calls=calls)


class ReportArchiveTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.base = Path(directory.name)

    def run_dir(self, name, backend, **extra):
        """A finished run of ``backend`` as the matching session leaves it behind."""
        root = self.base / name
        root.mkdir()
        counting = CODEX_COUNTING if backend == "codex" else NEW_COUNTING
        relay = relay_state(3, 300, 30)
        block = dict(kind=backend, cli=dict(version="9.9", executable="/x", sha256="ab"), counting=counting,
                     evidence_client="codex_agent" if backend == "codex" else "relay",
                     native_effort=dict(variant="high") if backend == "opencode" else dict(effort="high"))
        manifest = dict(run_id=name, model="configured-model", provider="provider-a", seed=0, layout_id=0,
                        agent_backend=block, **extra)
        summary = dict(status="completed", agent_backend=backend, token_usage=relay,
                       served_models={"served-model": 3}, model_usage_relay=relay,
                       model_requests=dict(counting=counting, completed_calls=3, served_models={"served-model": 3}))
        return make_run(root, manifest=manifest, summary=summary)

    def test_each_backend_reports_conditions_usage_served_models_and_counting_basis(self):
        report = build_report([self.run_dir(f"run-{backend}", backend) for backend in BACKENDS])
        rows = {row["conditions"]["agent_backend"]: row for row in report["runs"]}
        self.assertEqual(set(rows), set(BACKENDS))
        for backend, row in rows.items():
            conditions = row["conditions"]
            self.assertEqual((conditions["agent_cli"], conditions["model"]), ("9.9", "configured-model"))
            self.assertEqual(conditions["evidence_client"], "codex_agent" if backend == "codex" else "relay")
            self.assertEqual(conditions["counting_basis"], "all_delivered_usable_generation_completions")
            self.assertEqual(row["served_models"], {"served-model": 3})
            self.assertEqual(row["model_usage_relay_source"], "recorded")
            self.assertEqual(row["model_usage_relay"]["calls"], 3)
            self.assertEqual(row["usage_methods"]["input_tokens"], dict(native=300, relay=300, delta=0))
        # The raw strings stay visible even though the basis is one.
        self.assertEqual({row["counting"] for row in rows.values()}, {CODEX_COUNTING, NEW_COUNTING})
        self.assertEqual(rows["opencode"]["conditions"]["native_effort"], {"variant": "high"})
        self.assertEqual(rows["claude_code"]["conditions"]["native_effort"], {"effort": "high"})

    def test_relay_accounting_source_label_is_kept_not_rewritten_to_codex(self):
        root = self.run_dir("run-cc", "claude_code")
        summary = json.loads((root / "summary.json").read_text())
        from ni_rua.cortex.session.usage import account_usage
        saved = account_usage(summary["token_usage"], model="configured-model", provider="provider-a",
                              source="relay.anthropic_messages.usage.sum")
        saved.update(thread_id="t", turn_id="u")
        (root / "summary.json").write_text(json.dumps(dict(summary, token_accounting=saved)))
        self.assertEqual(summarize(root)["token_accounting"]["source"], "relay.anthropic_messages.usage.sum")

    def test_pre_refactor_codex_archive_reads_as_codex_and_derives_relay_usage_from_events(self):
        root = self.base / "old-codex"
        root.mkdir()
        events = []
        for index, (usage, served) in enumerate(((responses_usage(100, 20, 40, 5), "served-a"),
                                                 (responses_usage(60, 10, 0, 0), "served-a")), 1):
            events.append(("model.model_request_finished", dict(logical_call_id=f"c{index}", usage=usage,
                                                                 usage_known=True, served_model=served)))
            events.append(("model.model_call_completed", dict(logical_call_id=f"c{index}")))
        events.append(("model.model_request_finished", dict(logical_call_id="c3", usage=None, usage_known=False)))
        make_run(root, manifest=dict(run_id="old", codex_sdk="0.160.0", codex_cli="0.160.0", model="m", provider="p"),
                 summary=dict(status="completed", model_requests=dict(counting=CODEX_COUNTING, completed_calls=2)),
                 events=events)
        row = summarize(root)
        conditions = row["conditions"]
        self.assertEqual((conditions["agent_backend"], conditions["agent_cli"], conditions["evidence_client"]),
                         ("codex", "0.160.0", "codex_agent"))
        self.assertEqual(conditions["counting_basis"], "all_delivered_usable_generation_completions")
        self.assertEqual(row["model_usage_relay_source"], "derived_from_model_events")
        self.assertEqual((row["model_usage_relay"]["calls"], row["model_usage_relay"]["total"]["input_tokens"],
                          row["served_models"]), (2, 160, {"served-a": 2}))

    def test_an_unsettled_summary_gives_way_to_the_archives_later_relay_records(self):
        # model_records_settled: false = the summary's relay copies may miss the last completions; the archive's
        # runtime records (the relay's last write) replace them. Absent (older archives) and true read the summary.
        later = relay_state(4, 410, 41)
        runtime = {("runtime", "model_requests"): dict(counting=NEW_COUNTING, completed_calls=4,
                                                       served_models={"served-model": 4}),
                   ("runtime", "model_usage_relay"): later, ("runtime", "token_usage"): later}
        rows = {}
        for flag in ("absent", True, False):
            root = self.run_dir(f"run-{flag}", "opencode")
            summary = json.loads((root / "summary.json").read_text())
            if flag != "absent":
                summary["model_records_settled"] = flag
            (root / "summary.json").write_text(json.dumps(summary))
            store = Store(root, dict(run_id=root.name, episode_id="e", env_instance_id="i", env_generation=1))
            try:
                for (kind, identifier), value in runtime.items():
                    store.set_state(kind, identifier, value)
            finally:
                store.close()
            rows[flag] = summarize(root)
        for flag in ("absent", True):
            with self.subTest(model_records_settled=flag):
                row = rows[flag]
                self.assertEqual((row["model_records_settled"], row["model_records_possibly_incomplete"]),
                                 (None if flag == "absent" else True, False))
                self.assertEqual((row["served_models"], row["model_requests"]["completed_calls"],
                                  row["model_usage_relay"]["calls"], row["token_usage"]["output_tokens"]),
                                 ({"served-model": 3}, 3, 3, 30))
        row = rows[False]
        self.assertEqual((row["model_records_settled"], row["model_records_possibly_incomplete"]), (False, False))
        self.assertEqual((row["served_models"], row["model_requests"]["completed_calls"],
                          row["model_usage_relay"]["calls"], row["token_usage"]["output_tokens"]),
                         ({"served-model": 4}, 4, 4, 41))
        self.assertEqual(row["token_accounting"]["tokens"]["output_tokens"], 41)
        self.assertEqual(row["usage_methods"]["output_tokens"], dict(native=41, relay=41, delta=0))

    def test_an_unsettled_summary_without_archive_records_is_marked_possibly_incomplete(self):
        root = self.run_dir("run-unsettled-bare", "claude_code")
        summary = json.loads((root / "summary.json").read_text())
        (root / "summary.json").write_text(json.dumps(dict(summary, model_records_settled=False)))
        row = summarize(root)
        self.assertEqual((row["model_records_settled"], row["model_records_possibly_incomplete"]), (False, True))
        self.assertEqual((row["served_models"], row["model_usage_relay"]["calls"]), ({"served-model": 3}, 3))

    def test_archive_without_any_relay_evidence_reports_unknown_not_zero(self):
        root = self.base / "bare"
        root.mkdir()
        make_run(root, manifest=dict(run_id="bare", agent_backend=dict(kind="opencode")), summary=dict(status="unknown"))
        row = summarize(root)
        self.assertIsNone(row["model_usage_relay"])
        self.assertIsNone(row["usage_methods"])
        self.assertIsNone(row["served_models"])
        self.assertIsNone(row["conditions"]["evidence_client"])
        self.assertIsNone(row["conditions"]["counting_basis"])


class LibraryComparisonTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name).resolve()
        self.path = self.root / "benchmark" / "panel" / "manifest.json"
        self.row = dict(case_id="pick__standard__g0__l0", task="pick", variant="standard", eval_seed=0, layout_id=0)

    def register(self, name, recorded):
        run = self.root / "benchmark" / "panel" / "runs" / name
        register_run(run, category="benchmark", task="pick", run_id=name)
        atomic_json(run / "summary.json", dict(normalized_score=1, task_success=True, valid_episode=True,
                                               scored_episode=True, status="completed"))
        manifest = dict(task="pick", seed=0, layout_id=0)
        if recorded is not None:
            manifest["agent_backend"] = recorded
        atomic_json(run / "manifest.json", manifest)
        return run

    def groups(self, specs, config=None, **extra):
        cases = []
        for condition, (name, recorded, override) in specs.items():
            run = self.register(name, recorded)
            cases.append(dict(run_dir=str(run), cwd="/source/x", task="pick", seed=0, layout_id=0,
                              experiment_condition=condition, tag=[self.row["case_id"]], **override))
        manifest = dict(benchmark_group="t", method="paired", protocol_cases=[self.row], cases=cases, **extra)
        if config is not None:
            manifest["ni_rua_config"] = config
        atomic_json(self.path, manifest)
        catalog = Catalog(self.root)
        catalog.scan()
        return {g["condition"]: g for g in ExperimentGroups(self.root).snapshot(catalog)["groups"]}

    def test_baseline_and_candidate_differing_only_in_backend(self):
        groups = self.groups({
            "baseline": ("old-codex", None, {}),   # archive from before backend selection
            "candidate": ("cc", dict(kind="claude_code", evidence_client="relay", native_effort=dict(effort="high")),
                          {"agent_backend": "claude_code"})}, config=Config().to_dict())
        base, candidate = groups["baseline"]["conditions"], groups["candidate"]["conditions"]
        self.assertEqual((base["agent_backend"], base["evidence_client"]), ("codex", "codex_agent"))
        self.assertEqual((candidate["agent_backend"], candidate["evidence_client"]), ("claude_code", "relay"))
        self.assertEqual(candidate["native_effort"], '{"effort":"high"}')
        self.assertEqual(groups["candidate"]["baseline_group_id"], groups["baseline"]["id"])
        differing = {key for key in {*base, *candidate} if base.get(key) != candidate.get(key)}
        self.assertEqual(differing, {"agent_backend", "evidence_client", "native_effort"})
        self.assertEqual(groups["candidate"]["warnings"], [])

    def test_baseline_label_pairs_the_backend_arm_with_the_codex_arm(self):
        groups = self.groups({"baseline": ("b", None, {}),
                              "cand": ("c", dict(kind="opencode", evidence_client="relay"), {"agent_backend": "opencode"})})
        self.assertEqual(groups["cand"]["baseline_group_id"], groups["baseline"]["id"])
        self.assertEqual(groups["baseline"]["conditions"]["agent_backend"], "codex")
        self.assertEqual(groups["cand"]["conditions"]["agent_backend"], "opencode")

    def test_codex_arm_on_relay_evidence_is_its_own_condition(self):
        config = Config.from_dict({"agent": {"evidence": {"client": "relay"}}}).to_dict()
        groups = self.groups({"baseline": ("b", None, {}), "cand": ("c", None, {"agent_backend": "claude_code"})},
                             config=config)
        self.assertEqual(groups["baseline"]["conditions"]["evidence_client"], "relay")
        self.assertEqual(groups["cand"]["conditions"]["evidence_client"], "relay")

    def test_recorded_backend_is_the_true_provenance_and_conflicts_are_flagged(self):
        groups = self.groups({"baseline": ("b", dict(kind="opencode"), {})}, config=Config().to_dict())
        group = groups["baseline"]
        self.assertEqual(group["conditions"]["agent_backend"], "opencode")
        self.assertIn("recorded_agent_backend_differs_from_configured", group["warnings"])


# --- test_robot_matrix.py ---
class MatrixChecks(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        runtime = self.runtime = _DualRuntime()
        adapter = runtime.adapter
        picture = FixtureRobot(directory.name)
        original_state = runtime.state

        def state():
            value = original_state()
            value.update(data_origin='synthetic_fixture', source_kind='measurement')
            for side, arm in value['arms'].items():
                arm['ee_pose']['position_m'][2] += arm['q_rad'][1]
                arm['gripper']['goal_normalized'] = adapter.goals[side]
            return value

        def plan(start, target, origin):
            goal = np.array(start, copy=True)
            goal[0], goal[1] = target[0], target[2] - 1.
            return dict(status='Success', position=np.stack([start, goal]))

        for planner in runtime.planners.values():
            planner.plan_path = plan
        adapter.state = state
        adapter.observe = lambda: dict(picture.observe(), robot_state=state(), sim_step=adapter.control_step,
            observation_profile='rgb_proprio_calibrated_v1',
            camera_calibration=dict(reference_frame_id='env_origin', cameras={}),
            camera_pose_sample=dict(cameras={}), robot_model=dict(base_link='base', joints=[]))
        robot = self.robot = Mock()
        robot.compile_action = RobotBackend.compile_action
        robot.validate_action = RobotBackend.validate_action
        robot.start.return_value = dict(adapter.capabilities(), data_origin='synthetic_fixture', observer_video=False)
        robot.observe.side_effect = adapter.observe
        robot.preview_batch.side_effect = adapter.preview_batch
        robot.execute = lambda action, progress_callback=None, cancel_flag=None: adapter.execute(
            action, progress_callback, cancel_flag)
        identity = runtime.harness.identity
        store = Store(Path(directory.name) / 'store', identity)
        self.addCleanup(store.close)
        self.harness = Harness(store, robot, config=Config.from_dict(dict(features=dict(
            observation_profile='rgb_proprio_calibrated_v1', motion_execution_mode='resampled',
            completion_guidance='none', enable_wait=True))))
        self.harness.start()
        stack = self.enterContext(ExitStack())
        stack.enter_context(patch('ni_rua.runtime.actions.inspection._trajectory_image', return_value=({}, [])))
        stack.enter_context(patch('ni_rua.core.presentation.feedback_delivery.action_trajectory_feedback', return_value=(None, [])))

    def test_matrix_uses_real_compiler_harness_and_native_cpu_scheduler(self):
        report = {}
        original = self.robot.execute
        run_checks(self.harness, report)
        self.assertTrue(all(item['passed'] for item in report['checks'].values()))
        self.assertEqual(report['checks']['single_checkpoint_pause']['receipt']['status'], 'stopped')
        self.assertEqual(report['checks']['bounded_wait']['receipt']['executed_native_steps'], 3)
        self.assertGreater(report['checks']['confirmed_stop']['receipt']['executed_native_steps'], 0)
        self.assertFalse(self.harness.accepting)
        self.assertIs(self.robot.execute, original)
        self.assertEqual(len([item for item in report['calls'] if item['tool'] == 'robot_action']), 9)

    def test_preview_clock_mutation_stops_matrix_before_execution(self):
        preview = self.robot.preview_batch.side_effect
        def advances(action):
            result = preview(action)
            self.runtime.adapter.control_step += 1
            return result
        self.robot.preview_batch.side_effect = advances
        report = {}
        with self.assertRaisesRegex(RuntimeError, 'single_checkpoint_preview'):
            run_checks(self.harness, report)
        self.assertFalse(report['checks']['single_checkpoint_preview']['passed'])
        self.assertEqual(self.runtime.commands, [])

    def test_stop_failure_preserves_unknown_and_restores_backend_method(self):
        original = self.robot.execute
        def lose_ack(action, **kwargs):
            if action.get('hold_native_steps') == 50:
                raise RuntimeError('synthetic stop acknowledgement lost')
            return original(action, **kwargs)
        self.robot.execute = lose_ack
        report = {}
        with self.assertRaisesRegex(RuntimeError, 'confirmed_stop'):
            run_checks(self.harness, report)
        self.assertIs(self.robot.execute, lose_ack)
        self.assertEqual(report['checks']['confirmed_stop']['receipt']['status'], 'unknown')
        self.assertFalse(self.harness.accepting)

    def test_completed_stage_labels_do_not_replace_measured_dual_overlap(self):
        original = self.robot.execute
        def no_overlap(action, **kwargs):
            receipt = original(action, **kwargs)
            if any(stage['id'] == 'right_out' for stage in action.get('stages', [])):
                initial = receipt['actual_trace'][0]['state']['arms']['right']['q_rad']
                for sample in receipt['actual_trace']:
                    sample['state']['arms']['right']['q_rad'] = list(initial)
            return receipt
        self.robot.execute = no_overlap
        report = {}
        with self.assertRaisesRegex(RuntimeError, 'dual_measured_overlap'):
            run_checks(self.harness, report)
        self.assertTrue(report['checks']['unequal_dual_stage_coverage']['passed'])
        self.assertFalse(report['checks']['dual_measured_overlap']['passed'])


class MatrixOwnerChecks(unittest.TestCase):
    def test_invalid_parameters_fail_before_allocating_backend_or_store(self):
        for options in (dict(displacement_m=0), dict(displacement_m=float('nan')),
                        dict(jaw_goal=True), dict(jaw_goal=.49)):
            with self.subTest(options=options), patch('ni_rua.apps.evaluation.robot_matrix.Store') as store, \
                    patch('ni_rua.apps.evaluation.robot_matrix.RobotBackend') as robot:
                with self.assertRaises(ValueError):
                    run_validation('/unused-validation', gpu=0, **options)
                store.assert_not_called()
                robot.assert_not_called()

    def test_owner_persists_failure_and_releases_other_resources(self):
        for failure in ('harness_init', 'start', 'stop', 'robot_close', 'reader_exit', 'store_close', None):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                store = stack.enter_context(patch('ni_rua.apps.evaluation.robot_matrix.Store')).return_value
                robot = stack.enter_context(patch('ni_rua.apps.evaluation.robot_matrix.RobotBackend')).return_value
                constructor = stack.enter_context(patch('ni_rua.apps.evaluation.robot_matrix.Harness'))
                harness = constructor.return_value
                harness.capabilities = {'data_origin': 'synthetic_fixture'}
                harness.programs = None
                stack.enter_context(patch('ni_rua.apps.evaluation.robot_matrix.run_checks'))
                stack.enter_context(patch('ni_rua.apps.evaluation.provenance.source_binding', return_value={}))
                robot.save_video.return_value = {'videos': []}
                robot.close.return_value = dict(process_exit_confirmed=True, reader_exit_confirmed=True,
                    native_close_acknowledged=True)
                if failure == 'reader_exit':
                    robot.close.return_value['reader_exit_confirmed'] = False
                elif failure is not None:
                    operation = dict(harness_init=constructor, start=harness.start, stop=harness.request_stop,
                        robot_close=robot.close, store_close=store.close)[failure]
                    operation.side_effect = RuntimeError('synthetic ' + failure)
                report = run_validation(directory, gpu=0)
                self.assertEqual(report['passed'], failure is None)
                self.assertIsNone(report['task_success'])
                self.assertEqual(json.loads((Path(directory) / 'validation-report.json').read_text()), report)
                self.assertEqual(json.loads((Path(directory) / 'summary.json').read_text())['status'],
                    'completed' if failure is None else 'failed')
                robot.close.assert_called_once()
                store.close.assert_called_once()

    def test_unresolved_program_is_reconciled_after_robot_close_before_storage(self):
        for unresolved in (False, True):
            with self.subTest(unresolved=unresolved), tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
                store = stack.enter_context(patch('ni_rua.apps.evaluation.robot_matrix.Store')).return_value
                robot = stack.enter_context(patch('ni_rua.apps.evaluation.robot_matrix.RobotBackend')).return_value
                harness = stack.enter_context(patch('ni_rua.apps.evaluation.robot_matrix.Harness')).return_value
                harness.capabilities = {}
                stack.enter_context(patch('ni_rua.apps.evaluation.provenance.source_binding', return_value={}))
                stack.enter_context(patch('ni_rua.apps.evaluation.robot_matrix.run_checks',
                    side_effect=RuntimeError('program_cleanup_confirmed')))
                robot.close.return_value = dict(process_exit_confirmed=True, reader_exit_confirmed=True,
                    native_close_acknowledged=True)
                def close_programs():
                    robot.close.assert_called_once()
                    store.close.assert_not_called()
                    return dict(active_jobs=['program-x'] if unresolved else [], unresolved_jobs=[])
                harness.programs.close.side_effect = close_programs
                report = run_validation(directory, gpu=0)
                harness.programs.close.assert_called_once()
                self.assertFalse(report['passed'])
                self.assertEqual(store.close.call_count, 0 if unresolved else 1)
                self.assertEqual('store_close_deferred' in report, unresolved)



if __name__ == "__main__":
    unittest.main()
