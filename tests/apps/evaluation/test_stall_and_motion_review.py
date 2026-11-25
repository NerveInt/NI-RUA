"""Consolidated tests:

- test_actor_progress_stall: Completed-final progress budgets use public execution, never wording or GT.
- test_actor_stall: Repeated completed inability answers stop a benchmark for native evaluation.
"""

import asyncio
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch
from ni_rua.cortex.supervision.actor_stall import (
    ACTOR_PROGRESS_IDLE_LIMIT, ACTOR_PROGRESS_RECOVERY_LIMIT, ACTOR_PROGRESS_REASON,
    ActorProgressTracker, completed_turn_activity, recovery_fingerprints,
)
from ni_rua.apps.evaluation.episode import wait_benchmark
from ni_rua.core.storage.store import Store
from unittest.mock import Mock
from ni_rua.cortex.supervision.actor_stall import (
    ACTOR_STALL_LIMIT, ACTOR_STALL_REASON, ActorStallTracker, inability_reason,
)
from ni_rua.cortex.agents.codex.session import CodexSession
from ni_rua.apps.evaluation.episode import BENCHMARK_CONTINUATION, wait_benchmark
from ni_rua.apps.library.scoring import classify_episode_result
import json
from unittest.mock import patch
from scripts.motion_review_eval import ANALYZER_VERSION, CASES, PROFILE, prepare, sha, source_files, summarize, summarize_case, token_report, verify
from support.paths import REPO_ROOT


NI_RUA_TEST_TIER = "component"

# --- test_actor_progress_stall.py ---
class ProgressTrackerTests(unittest.TestCase):
    def observe(self, tracker, turn, **changes):
        values = dict(thread_id="thread", turn_id=f"turn-{turn}", episode_id="episode",
                      native_action_count=606, activity=())
        values.update(changes)
        return tracker.observe(**values)

    def test_baseline_then_three_empty_recovery_turns(self):
        tracker = ActorProgressTracker()
        first = self.observe(tracker, 1)
        self.assertTrue(first["baseline"])
        self.assertEqual(first["count"], 0)
        for index in range(1, ACTOR_PROGRESS_IDLE_LIMIT + 1):
            result = self.observe(tracker, index + 1)
            self.assertEqual(result["count"], index)
            self.assertEqual(result["triggered"], index == ACTOR_PROGRESS_IDLE_LIMIT)
        self.assertEqual(result["reason"], "idle_final_limit")

    def test_distinct_investigation_is_bounded_at_six_continuations(self):
        tracker = ActorProgressTracker()
        self.observe(tracker, 0)
        for index in range(1, ACTOR_PROGRESS_RECOVERY_LIMIT + 1):
            result = self.observe(tracker, index, activity={f"geometry_query:hypothesis-{index}"})
            self.assertEqual(result["idle_count"], 0)
            self.assertEqual(result["triggered"], index == ACTOR_PROGRESS_RECOVERY_LIMIT)
        self.assertEqual(result["reason"], "recovery_final_limit")

    def test_baseline_activity_and_repeated_tools_never_get_new_credit(self):
        tracker = ActorProgressTracker()
        activity = {"robot_capabilities:same", "geometry_query:same"}
        self.observe(tracker, 0, activity=activity)
        for index in range(1, 4):
            result = self.observe(tracker, index, activity=activity)
            self.assertEqual(result["new_activity_count"], 0)
        self.assertTrue(result["triggered"])

    def test_native_execution_resets_both_limits_even_after_partial_failure(self):
        tracker = ActorProgressTracker()
        for index in range(3):
            self.observe(tracker, index)
        # Success/receipt status is deliberately absent. Executed steps count,
        # including partial motion; an infeasible zero-step attempt does not.
        progressed = self.observe(tracker, 3, native_action_count=607)
        self.assertTrue(progressed["baseline"])
        self.assertEqual(progressed["count"], 0)
        for index in range(4, 7):
            result = self.observe(tracker, index, native_action_count=607)
            self.assertEqual(result["triggered"], index == 6)

    def test_context_change_resets_but_duplicate_turn_is_not_another_final(self):
        tracker = ActorProgressTracker()
        self.observe(tracker, 0)
        self.observe(tracker, 1)
        self.assertIsNone(self.observe(tracker, 1, activity={"made-up"}))
        self.assertEqual(tracker.no_native_finals, 1)
        for key, value in [("episode_id", "new-episode"), ("thread_id", "new-thread")]:
            result = self.observe(tracker, 1, **{key: value})
            self.assertTrue(result["baseline"])
            self.assertFalse(result["triggered"])

    def test_unknown_or_regressed_counts_do_not_establish_a_stall(self):
        tracker = ActorProgressTracker()
        self.observe(tracker, 0)
        for index, count in enumerate([None, True, -1, 605, 606.0], 1):
            result = self.observe(tracker, index, native_action_count=count)
            self.assertEqual(result["status"], "unknown")
            self.assertFalse(result["triggered"])
        self.assertEqual(tracker.native_action_count, 606)
        self.assertEqual(tracker.no_native_finals, 0)

    def test_recorded_airborne_93_terminal_events_replay(self):
        # Public-event reduction of a recorded airborne-handoff run:
        # 8 finals at606, 22 at611, 62 at612, then one failed turn. New activity on
        # finals2(9),3(3),4(2); remaining turns are prose or repeated activity.
        # Keep literals self-contained; no archived run, model or robot required.
        counts = [606] * 8 + [611] * 22 + [612] * 63
        novelty = {
            2: ["geometry_query"] * 2 + ["history_view_observation"] * 2 + ["program_run"] * 2 + ["robot_preview_cost"] * 3,
            3: ["history_view_observation"] * 2 + ["scene_capabilities"],
            4: ["geometry_query", "program_run"],
        }
        tracker, first_stop = ActorProgressTracker(), None
        self.assertEqual(len(counts), 93)
        completed = 0
        for ordinal, count in enumerate(counts, 1):
            if ordinal == 93:  # Budget-ended failed turn is not another final.
                continue
            completed += 1
            activity = {f"{name}:recorded-{ordinal}-{i}" for i, name in enumerate(novelty.get(ordinal, []))}
            result = self.observe(tracker, ordinal, native_action_count=count, activity=activity)
            if result["triggered"] and first_stop is None:
                first_stop = ordinal
        self.assertEqual(completed, 92)
        self.assertEqual(first_stop, 7)  # Six continued turns, not 92.


class RecoveryFingerprintTests(unittest.TestCase):
    def test_unified_stage_names_do_not_buy_credit_but_routes_and_dependencies_do(self):
        first = dict(stages=[dict(id="reach", arm="left", kind="move", position_m=[0, 0, 1]),
            dict(id="jaw", arm="right", kind="gripper", goal_normalized=0, depends_on=["reach"])])
        renamed = dict(mode="preview", stages=[dict(id="a", arm="left", kind="move", position_m=[0, 0, 1]),
            dict(id="b", arm="right", kind="gripper", goal_normalized=0, depends_on=["a"])])
        original = recovery_fingerprints("robot_action", first)
        self.assertTrue(original)
        self.assertEqual(original, recovery_fingerprints("robot_action", renamed))
        renamed["stages"][1]["depends_on"] = []
        self.assertNotEqual(original, recovery_fingerprints("robot_action", renamed))

    def test_unified_inspect_uses_plan_commands_and_individual_samples(self):
        plan = dict(identity=dict(episode_id="e"), action=dict(stages=[
            dict(id="reach", arm="left", kind="move", position_m=[0, 0, 1])]))
        plans = {"p1": plan, "p2": plan}
        def inspect(identifier, steps):
            return recovery_fingerprints("robot_action", dict(mode="inspect", plan_id=identifier,
                native_steps=steps), episode_id="e", plans=plans)
        self.assertEqual(inspect("p1", [0, 5]), inspect("p2", [5, 0]))
        self.assertEqual(len(inspect("p2", [5, 10]) - inspect("p1", [0, 5])), 1)

    def test_operation_memory_queries_are_investigation_but_records_are_not(self):
        self.assertTrue(recovery_fingerprints("operation_memory", dict(request=dict(kind="query", query="grasp"))))
        self.assertFalse(recovery_fingerprints("operation_memory", dict(request=dict(kind="record", summary="new words"))))

    def test_stored_short_observation_aliases_restore_before_fingerprinting(self):
        with tempfile.TemporaryDirectory() as temporary:
            store = Store(Path(temporary), dict(run_id="r", episode_id="e", env_instance_id="env", env_generation=1))
            try:
                store.save("observation", "original", dict(observation_id="original", episode_id="e", sim_step=12))
                store.set_state("actor_id", "obs_1", dict(alias="obs_1", original="original"))
                for turn, observation in (("raw", "original"), ("short", "obs_1")):
                    store.save("tool_call", turn, dict(correlation=dict(thread_id="t", turn_id=turn),
                        params=dict(tool="history_view_observation", arguments=dict(observation_id=observation))))
                self.assertEqual(completed_turn_activity(store, thread_id="t", turn_id="raw", episode_id="e"),
                                 completed_turn_activity(store, thread_id="t", turn_id="short", episode_id="e"))
            finally:
                store.close()

    def test_preview_labels_order_ids_and_timeout_do_not_refresh_same_candidates(self):
        a = dict(observation_id="obs-first", timeout_s=45, candidates=[
            dict(label="left", move=dict(arm="left", grasp_position_m=[1, 2, 3])),
            dict(label="right", move=dict(arm="right", grasp_position_m=[1, 2, 3]))])
        b = dict(observation_id="obs-recaptured", timeout_s=120, candidates=[
            dict(label="renamed right", move=dict(arm="right", grasp_position_m=[1, 2, 3])),
            dict(label="renamed left", move=dict(arm="left", grasp_position_m=[1, 2, 3]))])
        self.assertEqual(recovery_fingerprints("robot_preview_cost", a),
                         recovery_fingerprints("robot_preview_cost", b))
        b["candidates"][0]["move"]["grasp_position_m"] = [1, 2, 4]
        self.assertEqual(len(recovery_fingerprints("robot_preview_cost", b)
                             - recovery_fingerprints("robot_preview_cost", a)), 1)

    def test_same_paused_observation_id_is_not_a_new_view_but_historical_step_is(self):
        observations = {"obs-1": dict(episode_id="e", sim_step=600),
                        "obs-2": dict(episode_id="e", sim_step=600),
                        "obs-3": dict(episode_id="e", sim_step=500)}
        def fingerprint(identifier):
            return recovery_fingerprints("history_view_observation",
                dict(observation_id=identifier, camera_id="cam_left_wrist", roi=[1, 2, 3, 4]),
                observations=observations, episode_id="e")
        self.assertEqual(fingerprint("obs-1"), fingerprint("obs-2"))
        self.assertNotEqual(fingerprint("obs-1"), fingerprint("obs-3"))
        self.assertEqual(fingerprint("missing-1"), fingerprint("missing-2"))

    def test_observe_and_plan_prose_do_not_refresh_recovery(self):
        for tool in ("robot_observe", "task_plan", "not_a_supported_tool"):
            self.assertEqual(recovery_fingerprints(tool, {"text": "A new claim", "revision": 500}), set())

    def test_program_formatting_and_comments_do_not_count_as_new_computation(self):
        a = recovery_fingerprints("program_run", dict(source="x = 1\nprint(x)\n", timeout_s=30))
        b = recovery_fingerprints("program_run", dict(source="# another label\nx=1\nprint( x )", timeout_s=60))
        self.assertEqual(a, b)

    def test_program_observation_uuid_churn_is_not_new_recovery(self):
        observations = {"obs-first": dict(episode_id="e", sim_step=100),
                        "obs-again": dict(episode_id="e", sim_step=100)}
        def fingerprint(observation_id):
            return recovery_fingerprints("program_run", {"source": f"client.observe('{observation_id}')"},
                                         observations=observations, episode_id="e")
        self.assertEqual(fingerprint("obs-first"), fingerprint("obs-again"))

    def test_trajectory_inspection_steps_are_individual_bounded_evidence(self):
        base = dict(trajectory_ref="artifact:trajectory", camera_id="cam_head")
        original = recovery_fingerprints("robot_inspect_trajectory", {**base, "native_steps": [0, 5]})
        reordered = recovery_fingerprints("robot_inspect_trajectory", {**base, "native_steps": [5, 0]})
        extended = recovery_fingerprints("robot_inspect_trajectory", {**base, "native_steps": [5, 10]})
        self.assertEqual(original, reordered)
        self.assertEqual(len(extended - original), 1)
        tracker = ActorProgressTracker()
        tracker.observe(thread_id="t", turn_id="0", episode_id="e", native_action_count=0, activity=original)
        for index in range(1, 7):
            activity = recovery_fingerprints("robot_inspect_trajectory", {**base, "native_steps": [10 + index]})
            result = tracker.observe(thread_id="t", turn_id=str(index), episode_id="e", native_action_count=0, activity=activity)
        self.assertTrue(result["triggered"])
        self.assertEqual(result["reason"], "recovery_final_limit")

    def test_actual_vision_review_name_and_recaptured_ids(self):
        observations = {"obs-a": dict(episode_id="e", sim_step=10),
                        "obs-b": dict(episode_id="e", sim_step=20),
                        "obs-c": dict(episode_id="e", sim_step=20)}
        base = dict(before_observation_id="obs-a", camera_id="cam_head", roi=[1, 2, 3, 4], target="bottle")
        first = recovery_fingerprints("vision_review", {**base, "after_observation_id": "obs-b"},
                                      observations=observations, episode_id="e")
        again = recovery_fingerprints("vision_review", {**base, "after_observation_id": "obs-c"},
                                      observations=observations, episode_id="e")
        self.assertEqual(len(first), 1)
        self.assertEqual(first, again)
        self.assertTrue(recovery_fingerprints("operation_verify", {"request": {"kind": "help"}}))
        self.assertFalse(recovery_fingerprints("visual_review", base))

    def test_store_reader_scopes_thread_turn_and_public_record_types(self):
        records = {
            "observation": [dict(observation_id="o", episode_id="e", sim_step=0)],
            "tool_call": [
                dict(correlation=dict(thread_id="t", turn_id="current"), params=dict(tool="geometry_query", arguments={"request": {"kind": "pixel_ray", "pixel": [1, 2]}})),
                dict(correlation=dict(thread_id="other", turn_id="current"), params=dict(tool="program_run", arguments={"source": "print(1)"})),
                dict(correlation=dict(thread_id="t", turn_id="old"), params=dict(tool="robot_capabilities", arguments={})),
            ],
        }
        store = SimpleNamespace(records=Mock(side_effect=lambda kind: records[kind]))
        result = completed_turn_activity(store, thread_id="t", turn_id="current", episode_id="e")
        self.assertEqual(len(result), 1)
        self.assertTrue(next(iter(result)).startswith("geometry_query:"))
        self.assertEqual({call.args[0] for call in store.records.call_args_list}, {"observation", "tool_call"})


class ProgressBenchmarkTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        temporary_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.store = Store(Path(temporary_dir), dict(run_id="progress", episode_id="e", env_instance_id="env", env_generation=1))
        self.addCleanup(self.store.close)
        done = threading.Event()
        done.set()
        self.session = SimpleNamespace(state="completed", turn_done=done, turn_id="t1", thread_id="thread",
            model_limit_reached=threading.Event(), model_proxy=SimpleNamespace(limit_notified=False, request_count=1),
            completed_final_answer=Mock(side_effect=AssertionError("bounded policy must not parse prose")))
        self.harness = SimpleNamespace(episode_done=threading.Event(), episode_terminal=None, accepting=True,
            native_action_count=606, started_at=time.monotonic(), last_native_progress_at=time.monotonic(), active_action=None)
        self.workbench = SimpleNamespace(session=self.session, harness=self.harness, completion_guidance="none",
            store=self.store, config=SimpleNamespace(features=SimpleNamespace(bounded_continuation=True)),
            identity=dict(episode_id="e"), current_unknown_actions=lambda: False, has_active_programs=lambda: False,
            control_lock=asyncio.Lock())

    def next_turn(self, prompt):
        self.session.turn_id = f"t{self.session.run_turn.call_count + 1}"
        self.session.model_proxy.request_count += 1
        self.assertLess(self.session.run_turn.call_count, 20)

    async def run_benchmark(self):
        return await asyncio.wait_for(wait_benchmark(self.workbench, asyncio.Event()), 3)

    async def test_wording_independent_stop_and_auditable_events(self):
        self.session.run_turn = Mock(side_effect=self.next_turn)
        self.assertFalse(await self.run_benchmark())
        self.assertEqual(self.session.run_turn.call_count, 3)
        self.session.completed_final_answer.assert_not_called()
        self.assertEqual(self.harness.wait_stop_reason, ACTOR_PROGRESS_REASON)
        self.assertFalse(self.harness.episode_done.is_set())
        events = [x["payload"] for x in self.store.events() if x["type"] == "benchmark.actor_progress"]
        self.assertEqual([x["count"] for x in events], [0, 1, 2, 3])
        self.assertTrue(events[-1]["triggered"])
        continuations = [x["payload"] for x in self.store.events()
                         if x["type"] == "benchmark.actor_turn_continuation"]
        self.assertEqual(len(continuations), 3)
        for item in continuations:
            self.assertEqual(item["recovery_context"]["version"], "recovery-guidance-v1")
            self.assertIn("one concrete recovery objective", item["prompt"])
            self.assertIn("complete attempt", item["prompt"])

    async def test_evidence_switch_stops_after_two_unchanged_recovery_finals(self):
        self.workbench.config.features.recovery_evidence = True
        self.session.run_turn = Mock(side_effect=self.next_turn)
        self.assertFalse(await self.run_benchmark())
        self.assertEqual(self.session.run_turn.call_count, 2)
        self.assertEqual(self.harness.wait_stop_reason, ACTOR_PROGRESS_REASON)
        self.assertEqual(self.harness.actor_stall["reason"], "no_new_evidence_final_limit")
        continuations = [x["payload"] for x in self.store.events()
                         if x["type"] == "benchmark.actor_turn_continuation"]
        self.assertTrue(all(item["recovery_context"]["version"] == "recovery-guidance-v4"
                            for item in continuations))
        self.assertTrue(all("never all routes" in item["prompt"] for item in continuations))

    async def test_native_terminal_callback_during_new_evidence_read_still_wins(self):
        from ni_rua.cortex.supervision.recovery_evidence import completed_turn_evidence
        self.workbench.config.features.recovery_evidence = True
        calls = 0
        def evidence(*args, **kwargs):
            nonlocal calls
            result = completed_turn_evidence(*args, **kwargs)
            calls += 1
            if calls == 3:
                self.harness.episode_done.set()
            return result
        self.session.run_turn = Mock(side_effect=self.next_turn)
        with patch("ni_rua.cortex.supervision.recovery_evidence.completed_turn_evidence", side_effect=evidence):
            self.assertFalse(await self.run_benchmark())
        self.assertEqual(self.harness.wait_stop_reason, "native_episode_ended")
        self.assertEqual(self.session.run_turn.call_count, 2)

    async def test_native_execution_callback_during_new_evidence_read_resets_plateau(self):
        from ni_rua.cortex.supervision.recovery_evidence import completed_turn_evidence
        self.workbench.config.features.recovery_evidence = True
        calls = 0
        def evidence(*args, **kwargs):
            nonlocal calls
            result = completed_turn_evidence(*args, **kwargs)
            calls += 1
            if calls == 3:
                self.harness.native_action_count += 1
            return result
        self.session.run_turn = Mock(side_effect=self.next_turn)
        with patch("ni_rua.cortex.supervision.recovery_evidence.completed_turn_evidence", side_effect=evidence):
            self.assertFalse(await self.run_benchmark())
        self.assertEqual(self.session.run_turn.call_count, 4)
        self.assertEqual(self.harness.actor_stall["native_action_count"], 607)

    async def test_actual_motion_gets_a_fresh_recovery_budget(self):
        def advance(prompt):
            self.next_turn(prompt)
            if self.session.run_turn.call_count == 2:
                self.harness.native_action_count += 1
        self.session.run_turn = Mock(side_effect=advance)
        self.assertFalse(await self.run_benchmark())
        self.assertEqual(self.session.run_turn.call_count, 5)
        self.assertEqual(self.harness.native_action_count, 607)

    async def test_legacy_false_keeps_non_inability_finals_until_existing_budget(self):
        self.workbench.config.features.bounded_continuation = False
        self.session.completed_final_answer = Mock(return_value="Same physical state. No new result.")
        def advance(prompt):
            self.next_turn(prompt)
            if self.session.run_turn.call_count == 9:
                self.session.model_proxy.limit_notified = True
        self.session.run_turn = Mock(side_effect=advance)
        self.assertFalse(await self.run_benchmark())
        self.assertEqual(self.session.run_turn.call_count, 9)
        self.assertEqual(self.harness.wait_stop_reason, "model_request_limit")
        self.assertFalse(any(e["type"] == "benchmark.actor_progress" for e in self.store.events()))
        from ni_rua.apps.evaluation.episode import BENCHMARK_CONTINUATION
        self.assertTrue(all(call.args[0] == BENCHMARK_CONTINUATION for call in self.session.run_turn.call_args_list))

    async def test_native_terminal_and_unresolved_work_take_priority(self):
        for condition in ("native", "unknown", "active_action", "active_program", "not_accepting"):
            with self.subTest(condition=condition):
                self.harness.episode_done.clear()
                self.harness.accepting = True
                self.harness.active_action = None
                self.workbench.current_unknown_actions = lambda: False
                self.workbench.has_active_programs = lambda: False
                self.session.run_turn = Mock()
                if condition == "native": self.harness.episode_done.set()
                elif condition == "unknown": self.workbench.current_unknown_actions = lambda: True
                elif condition == "active_action": self.harness.active_action = "action-inflight"
                elif condition == "active_program": self.workbench.has_active_programs = lambda: True
                else: self.harness.accepting = False
                self.assertFalse(await self.run_benchmark())
                self.session.run_turn.assert_not_called()
                self.assertNotEqual(self.harness.wait_stop_reason, ACTOR_PROGRESS_REASON)

    async def test_terminal_callback_during_evidence_read_wins_stall_race(self):
        calls = 0
        def activity(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                self.harness.episode_done.set()
            return set()
        self.session.run_turn = Mock(side_effect=self.next_turn)
        with patch("ni_rua.apps.evaluation.episode.completed_turn_activity", side_effect=activity):
            self.assertFalse(await self.run_benchmark())
        self.assertEqual(self.harness.wait_stop_reason, "native_episode_ended")
        self.assertEqual(self.session.run_turn.call_count, 3)

    async def test_native_accounting_callback_during_evidence_read_gets_new_baseline(self):
        calls = 0
        def activity(*args, **kwargs):
            nonlocal calls
            calls += 1
            if calls == 4:
                self.harness.native_action_count += 1
            return set()
        self.session.run_turn = Mock(side_effect=self.next_turn)
        with patch("ni_rua.apps.evaluation.episode.completed_turn_activity", side_effect=activity):
            self.assertFalse(await self.run_benchmark())
        self.assertEqual(self.session.run_turn.call_count, 6)
        self.assertEqual(self.harness.actor_stall["native_action_count"], 607)

    async def test_normal_long_running_turn_is_not_a_completed_final(self):
        self.session.state = "running"
        self.session.turn_done.clear()
        task = asyncio.create_task(wait_benchmark(self.workbench, asyncio.Event()))
        await asyncio.sleep(.07)
        self.assertFalse(task.done())
        self.assertFalse(any(e["type"] == "benchmark.actor_progress" for e in self.store.events()))
        self.harness.episode_done.set()
        self.assertFalse(await asyncio.wait_for(task, 1))


# --- test_actor_stall.py ---
# Exact first and last final answers from the recorded bottle-run incident.
FIRST_RECORDED_FINAL = (
    "Both return commands completed using the recovered initial positions and orientations. "
    "Both grippers are open and empty.\n\n"
    "The environment still has not ended: three bottles are inside the bin, while the blue "
    "bottle remains outside beyond either arm’s reach. The task cannot be completed with "
    "the available tools."
)
LAST_RECORDED_FINAL = (
    "Both return commands completed using the specified initial observation.\n\n"
    "The remaining blue bottle landed outside the bin and is beyond either arm’s reach. "
    "Three bottles are inside, both grippers are empty and open, and no feasible recovery "
    "action remains."
)
# Chronological first five completed finals from the recorded-run audit. Keep
# these literal: the regression must not depend on an archived run or /tmp.
FIRST_FIVE_RECORDED_FINALS = (
    "Both return commands completed using the recovered initial positions and orientations. "
    "Both grippers are open and empty.\n\n"
    "The environment still has not ended: three bottles are inside the bin, while the blue "
    "bottle remains outside beyond either arm’s reach. The task cannot be completed with "
    "the available tools.",
    "Both arms have returned successfully to the specified observation’s initial link6 poses, "
    "with grippers open and empty.\n\n"
    "Placement remains incomplete: three bottles are inside the bin; the blue bottle is "
    "outside, beyond both arms’ reach. No feasible recovery action is available with the "
    "permitted tools.",
    "The return requirement is satisfied and verified for both arms using the specified "
    "initial observation.\n\n"
    "The task remains incomplete because the blue bottle is outside the bin, approximately "
    "1.9 m from the nearest arm base—beyond its reach. Three bottles are inside. No further "
    "feasible action is available without external repositioning.",
    "Both arms are already at the recovered initial poses, with empty, open grippers.\n\n"
    "The unresolved issue is the blue bottle outside the bin and beyond either arm’s reach. "
    "Three bottles are inside; completion cannot be achieved with the available tools.",
    "Unable to complete: the blue bottle landed outside the bin, beyond both arms’ reach. "
    "Three bottles are inside.\n\n"
    "Both arms have successfully returned to the initial link6 poses recovered from "
    "`obs-ab87c2ad4f3b45a191186c68ec233027`, with grippers open and empty.",
)
IDENTITY = dict(run_id="synthetic-actor-stall", episode_id="episode-1",
                env_instance_id="fixture-env", env_generation=1)


class ActorStallTrackerChecks(unittest.TestCase):
    def observe(self, tracker, turn, **changes):
        values = dict(thread_id="thread-1", turn_id=turn, episode_id="episode-1",
                      native_action_count=12, text="I cannot continue.")
        values.update(changes)
        return tracker.observe(**values)

    def test_recorded_and_explicit_bilingual_inability_declarations_match(self):
        for text in (FIRST_RECORDED_FINAL, LAST_RECORDED_FINAL,
                     "I cannot continue.", "We are unable to complete the task.",
                     "I can't safely proceed.", "No safe next action is available.",
                     "我无法继续完成任务。", "无法完成任务，已经没有可行的恢复动作。",
                     "没有可执行动作。", "不能继续执行。"):
            with self.subTest(text=text):
                self.assertIsInstance(inability_reason(text), str)

    def test_completion_temporary_failure_and_continuation_plans_do_not_match(self):
        for text in (None, "", "The task is complete.", "Both arms are home.",
                     "The last movement failed; I will retry with a different pose.",
                     "I cannot continue with this pose. I will try another approach.",
                     "I cannot complete this route. I'll try a different pose.",
                     'The tool returned "I cannot continue." I am checking the result.',
                     "> I cannot continue.\nThat was the previous answer.",
                     "```text\nI cannot continue.\n```\nThe example above describes an old run.",
                     "当前无法完成任务，下一步尝试移动左臂。",
                     "工具无法继续执行；我会尝试恢复。"):
            with self.subTest(text=text):
                self.assertIsNone(inability_reason(text))

    def test_fifth_unique_inability_triggers_and_records_evidence(self):
        tracker = ActorStallTracker()
        self.assertEqual(ACTOR_STALL_LIMIT, 5)
        self.assertEqual(ACTOR_STALL_REASON, "actor_repeated_inability")
        for count in range(1, 6):
            answer = FIRST_RECORDED_FINAL if count == 1 else LAST_RECORDED_FINAL
            result = self.observe(tracker, f"turn-{count}", text=answer)
            self.assertEqual(result["count"], count)
            self.assertEqual(result["limit"], 5)
            self.assertEqual(result["triggered"], count == 5)
            self.assertTrue(result["matched_reason"])
            self.assertEqual(result["answer_excerpt"], answer)
            self.assertEqual(result["turn_id"], f"turn-{count}")
            self.assertEqual(result["thread_id"], "thread-1")
            self.assertEqual(result["episode_id"], "episode-1")
            self.assertEqual(result["native_action_count"], 12)

    def test_first_five_chronological_recorded_finals_trigger_exactly_on_fifth(self):
        tracker = ActorStallTracker()
        self.assertEqual(len(FIRST_FIVE_RECORDED_FINALS), 5)
        for count, text in enumerate(FIRST_FIVE_RECORDED_FINALS, 1):
            with self.subTest(completed_final=count):
                result = self.observe(tracker, f"recorded-turn-{count}", text=text)
                self.assertTrue(result["matched_reason"])
                self.assertEqual(result["count"], count)
                self.assertEqual(result["triggered"], count == 5)

    def test_local_recovery_problem_with_new_plan_and_quoted_declarations_do_not_count(self):
        for text in (
            "No feasible recovery is available with this grasp. I will try another approach.",
            'The previous answer said "completion cannot be achieved with the available tools."',
            '> The permitted tools cannot recover the fourth bottle.\nThis quotes an earlier assessment.',
            '```text\nFurther progress requires external repositioning.\n```\nThis is archived text.',
        ):
            with self.subTest(text=text):
                self.assertIsNone(inability_reason(text))
                tracker = ActorStallTracker()
                result = self.observe(tracker, "new-turn", text=text)
                self.assertEqual(result["count"], 0)
                self.assertFalse(result["triggered"])

    def test_native_progress_or_counter_reset_starts_a_new_streak(self):
        for next_action_count in (13, 0):
            with self.subTest(next_action_count=next_action_count):
                tracker = ActorStallTracker()
                for count in range(1, 5):
                    self.observe(tracker, f"turn-{count}")
                result = self.observe(tracker, "turn-5", native_action_count=next_action_count)
                self.assertEqual(result["count"], 1)
                self.assertFalse(result["triggered"])

    def test_thread_and_episode_changes_reset_even_if_turn_id_is_reused(self):
        for changes in ({"thread_id": "thread-2"}, {"episode_id": "episode-2"}):
            with self.subTest(changes=changes):
                tracker = ActorStallTracker()
                for count in range(1, 5):
                    self.observe(tracker, f"turn-{count}")
                result = self.observe(tracker, "turn-4", **changes)
                self.assertEqual(result["count"], 1)
                self.assertFalse(result["triggered"])

    def test_non_inability_final_clears_streak(self):
        tracker = ActorStallTracker()
        for count in range(1, 5):
            self.observe(tracker, f"turn-{count}")
        result = self.observe(tracker, "turn-5", text="I will inspect and move the left arm.")
        self.assertEqual(result["count"], 0)
        self.assertIsNone(result["matched_reason"])
        self.assertFalse(result["triggered"])
        self.assertEqual(self.observe(tracker, "turn-6")["count"], 1)

    def test_missing_identity_and_duplicate_turns_never_increment(self):
        tracker = ActorStallTracker()
        self.assertEqual(self.observe(tracker, "turn-1")["count"], 1)
        for changes in ({"turn_id": None}, {"thread_id": None}, {"episode_id": None}):
            self.assertIsNone(self.observe(tracker, "ignored", **changes))
        self.assertIsNone(self.observe(tracker, "turn-1"))
        self.assertEqual(self.observe(tracker, "turn-2")["count"], 2)
        self.assertIsNone(self.observe(tracker, "turn-1"))
        self.assertEqual(self.observe(tracker, "turn-3")["count"], 3)

    def test_duplicate_after_native_progress_clears_streak_without_counting(self):
        tracker = ActorStallTracker()
        for count in range(1, 5):
            self.observe(tracker, f"turn-{count}")
        self.assertIsNone(self.observe(tracker, "turn-4", native_action_count=13))
        result = self.observe(tracker, "turn-5", native_action_count=13)
        self.assertEqual(result["count"], 1)
        self.assertFalse(result["triggered"])


class CompletedActorFinalChecks(unittest.TestCase):
    def setUp(self):
        temporary_dir = self.enterContext(tempfile.TemporaryDirectory())
        self.store = Store(Path(temporary_dir) / "store", IDENTITY)
        self.addCleanup(self.store.close)
        self.session = CodexSession(temporary_dir, self.store, lambda params: None)
        self.session.thread_id = "owned-thread"
        self.session.turn_id = "current-turn"
        self.session.state = "completed"
        self.session.turn_done.set()

    def item(self, identifier, **changes):
        item = dict(id=identifier, thread_id="owned-thread", turn_id="current-turn",
                    type="agentMessage", phase="final_answer", status="completed",
                    text="I cannot continue.")
        item.update(changes)
        self.store.set_state("codex_item", identifier, item)

    def test_only_completed_current_turn_final_agent_messages_are_read(self):
        self.item("foreign-thread", thread_id="other-thread")
        self.item("earlier-turn", turn_id="previous-turn")
        self.item("commentary", phase="commentary")
        self.item("partial-final", status="running")
        self.item("tool-output", type="dynamicToolCall")
        self.item("unphased", phase=None)
        self.item("empty", text=" ")
        self.assertIsNone(self.session.completed_final_answer())
        self.item("owned-final", text=LAST_RECORDED_FINAL)
        self.assertEqual(self.session.completed_final_answer(), LAST_RECORDED_FINAL)

    def test_completed_items_are_not_read_before_completed_turn(self):
        self.item("final")
        for state in ("running", "failed", "interrupted", "disconnected"):
            with self.subTest(state=state):
                self.session.state = state
                self.assertIsNone(self.session.completed_final_answer())
        self.session.state = "completed"
        self.session.turn_done.clear()
        self.assertIsNone(self.session.completed_final_answer())

    def test_real_notification_projection_preserves_final_phase_and_scope(self):
        self.session.state = "running"
        self.session.turn_done.clear()
        for identifier, phase in (("commentary", "commentary"), ("final", "final_answer")):
            self.session._notification(dict(method="item/completed", params=dict(
                threadId="owned-thread", turnId="current-turn",
                item=dict(id=identifier, type="agentMessage", phase=phase,
                          text=FIRST_RECORDED_FINAL))))
        self.assertIsNone(self.session.completed_final_answer())
        self.session._notification(dict(method="turn/completed", params=dict(
            threadId="owned-thread", turn=dict(id="current-turn", status="completed"))))
        self.assertEqual(self.session.completed_final_answer(), FIRST_RECORDED_FINAL)


class ActorStallBenchmarkChecks(unittest.IsolatedAsyncioTestCase):
    def workbench(self):
        done = threading.Event()
        done.set()
        session = SimpleNamespace(state="completed", turn_done=done, turn_id="turn-1",
            thread_id="same-thread", turn_result={"status": "completed"},
            model_limit_reached=threading.Event(),
            model_proxy=SimpleNamespace(limit_notified=False, request_count=3),
            completed_final_answer=Mock(return_value=LAST_RECORDED_FINAL))
        harness = SimpleNamespace(episode_done=threading.Event(), episode_terminal=None,
            accepting=True, native_action_count=12, started_at=time.monotonic(),
            last_native_progress_at=time.monotonic(), active_action=None)
        return SimpleNamespace(session=session, harness=harness, completion_guidance="none",
            store=SimpleNamespace(event=Mock()), identity=dict(IDENTITY),
            current_unknown_actions=lambda: False, has_active_programs=lambda: False,
            control_lock=asyncio.Lock())

    async def test_five_finals_stop_before_sixth_turn_and_keep_native_clocks(self):
        workbench = self.workbench()
        session, harness = workbench.session, workbench.harness
        started, progress, proxy = harness.started_at, harness.last_native_progress_at, session.model_proxy
        def continue_turn(prompt):
            self.assertEqual(prompt, BENCHMARK_CONTINUATION)
            session.turn_id = f"turn-{session.run_turn.call_count + 1}"
            proxy.request_count += 1
            self.assertLessEqual(session.run_turn.call_count, 4, "a sixth actor turn must never start")
        session.run_turn = Mock(side_effect=continue_turn)
        finished = await asyncio.wait_for(wait_benchmark(workbench, asyncio.Event()), timeout=5)
        self.assertFalse(finished)
        self.assertEqual(session.run_turn.call_count, 4)
        self.assertEqual(session.completed_final_answer.call_count, 5)
        self.assertEqual(harness.wait_stop_reason, ACTOR_STALL_REASON)
        self.assertEqual(harness.actor_stall["count"], 5)
        self.assertTrue(harness.actor_stall["triggered"])
        self.assertEqual(harness.actor_continuation_count, 4)
        self.assertFalse(harness.episode_done.is_set())
        self.assertEqual(session.turn_id, "turn-5")
        self.assertEqual(session.thread_id, "same-thread")
        self.assertIs(session.model_proxy, proxy)
        self.assertEqual(proxy.request_count, 7)
        self.assertEqual(harness.native_action_count, 12)
        self.assertEqual(harness.started_at, started)
        self.assertEqual(harness.last_native_progress_at, progress)
        evidence = [call.args[1] for call in workbench.store.event.call_args_list
                    if call.args[0] == "benchmark.actor_inability"]
        self.assertEqual([item["count"] for item in evidence], [1, 2, 3, 4, 5])
        self.assertTrue(evidence[-1]["triggered"])

    async def test_native_progress_during_continuation_requires_five_new_finals(self):
        workbench = self.workbench()
        session, harness = workbench.session, workbench.harness
        def continue_turn(prompt):
            calls = session.run_turn.call_count
            session.turn_id = f"turn-{calls + 1}"
            if calls == 4:
                harness.native_action_count += 1
            self.assertLessEqual(calls, 8)
        session.run_turn = Mock(side_effect=continue_turn)
        self.assertFalse(await asyncio.wait_for(wait_benchmark(workbench, asyncio.Event()), timeout=5))
        self.assertEqual(session.run_turn.call_count, 8)
        self.assertEqual(harness.wait_stop_reason, ACTOR_STALL_REASON)
        self.assertEqual(harness.actor_stall["count"], 5)

    async def test_safety_recheck_precedes_reading_and_counting_final(self):
        for condition in ("native", "budget", "timeout", "unknown", "active_action", "active_program"):
            with self.subTest(condition=condition):
                workbench = self.workbench()
                session, harness = workbench.session, workbench.harness
                session.run_turn = Mock()
                class ContendedLock:
                    async def __aenter__(self):
                        if condition == "native": harness.episode_done.set()
                        if condition == "budget": session.model_limit_reached.set()
                        if condition == "timeout":
                            harness.started_at -= 1801
                            harness.last_native_progress_at -= 1801
                        if condition == "unknown": workbench.current_unknown_actions = lambda: True
                        if condition == "active_action": harness.active_action = "unfinished"
                        if condition == "active_program": workbench.has_active_programs = lambda: True
                    async def __aexit__(self, *arguments):
                        pass
                workbench.control_lock = ContendedLock()
                self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                session.run_turn.assert_not_called()
                session.completed_final_answer.assert_not_called()
                workbench.store.event.assert_not_called()


class ActorStallScoringChecks(unittest.TestCase):
    def snapshot(self, **changes):
        result = dict(benchmark=True, termination_reason=ACTOR_STALL_REASON,
            valid_episode=True, task_success=False, normalized_score=.4,
            native_process_score=40, native_reward=0, native_episode={"ended": False},
            authority="installed RoboDojo task.run_reward / RewardManager",
            success_authority="native_final_check")
        result.update(changes)
        return result

    def test_current_native_failure_is_scored_without_inventing_native_termination(self):
        snapshot = self.snapshot()
        result = classify_episode_result(snapshot, "stopped")
        self.assertEqual(result["primary_outcome"], "scored_failure")
        self.assertTrue(result["scored_episode"])
        self.assertFalse(result["task_success"])
        self.assertEqual(result["normalized_score"], .4)
        self.assertEqual(result["native_process_score"], 40)
        self.assertEqual(result["termination_authority"], "harness")
        self.assertFalse(result["native_episode"]["ended"])
        self.assertNotIn("partial_native_result", result)
        self.assertEqual(snapshot, self.snapshot(), "classification must not mutate input")

    def test_task_limit_is_failure_even_when_current_snapshot_reports_success(self):
        for success, score in ((True, 1.), (False, 0.)):
            with self.subTest(success=success):
                result = classify_episode_result(self.snapshot(task_success=success, normalized_score=score,
                                                               native_process_score=score * 100), "stopped")
                self.assertTrue(result["scored_episode"])
                self.assertEqual(result["primary_outcome"], "scored_failure")
                self.assertEqual(result["normalized_score"], score)
                self.assertIs(result["task_success"], False)

    def test_unknown_and_infrastructure_remain_unscored(self):
        for status, changes in (("unknown", {}), ("stopped", {"infrastructure_error": "robot disconnected"}),
                                ("failed", {})):
            with self.subTest(status=status, changes=changes):
                result = classify_episode_result(self.snapshot(**changes), status)
                self.assertEqual(result["primary_outcome"], "unscored_infrastructure")
                self.assertFalse(result["scored_episode"])
                self.assertIsNone(result["normalized_score"])
                self.assertIsNone(result["task_success"])
                self.assertEqual(result["partial_native_result"]["normalized_score"], .4)

    def test_invalid_native_evaluation_does_not_score_an_inability_stop(self):
        invalid = [{"valid_episode": False}, {"valid_episode": None}, {"valid_episode": 1},
                   {"task_success": None}, {"task_success": "false"}, {"task_success": 0}]
        invalid += [{"normalized_score": score} for score in
                    (None, "0.4", True, float("nan"), float("inf"), -.1, 1.1)]
        for changes in invalid:
            with self.subTest(changes=changes):
                result = classify_episode_result(self.snapshot(**changes), "stopped")
                self.assertFalse(result["scored_episode"])
                self.assertIsNone(result["normalized_score"])
                if "valid_episode" in changes:
                    self.assertIsNone(result["task_success"])
                else:
                    self.assertFalse(result["task_success"])
        snapshot = self.snapshot()
        del snapshot["normalized_score"]
        self.assertFalse(classify_episode_result(snapshot, "stopped")["scored_episode"])
        self.assertFalse(classify_episode_result(snapshot, "stopped")["task_success"])

    def test_task_limits_score_current_failure_and_external_interruption_remains_excluded(self):
        for reason, outcome in (("official_no_action_progress_timeout", "scored_failure"),
                                ("official_startup_no_action_timeout", "scored_failure"),
                                ("model_request_limit", "scored_failure"),
                                ("development_wall_budget", "scored_failure"),
                                ("shutdown-signal", "unscored_interrupted")):
            with self.subTest(reason=reason):
                result = classify_episode_result(self.snapshot(termination_reason=reason), "stopped")
                self.assertEqual(result["primary_outcome"], outcome)
                if outcome == "scored_failure":
                    self.assertTrue(result["scored_episode"])
                    self.assertFalse(result["task_success"])
                    self.assertEqual(result["normalized_score"], .4)
                else:
                    self.assertFalse(result["scored_episode"])
                    self.assertIsNone(result["normalized_score"])
                    self.assertEqual(result["partial_native_result"]["normalized_score"], .4)


# --- test_motion_review_eval.py ---
class MotionReviewEvaluationTest(unittest.TestCase):
    def setUp(self):
        self.root = Path(self.enterContext(tempfile.TemporaryDirectory())).resolve()
        root = self.root
        self.source, self.native, self.experiment = (root / name for name in ("source", "native", "experiment"))
        for name in ("apps/gateway/cli", "apps/gateway/workbench", "robot/motion/motion_review"):
            path = self.source / "ni_rua" / (name + ".py")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("# frozen test source\n")
        for task, _, seed, layout, *_ in CASES:
            for name in (f"task/RoboDojo/tasks/{task}.py", f"Assets/Eval_Layout/RoboDojo/arx_x5/{seed}/{task}_{layout}.json"):
                path = self.native / name
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text("{}\n")
        with patch('ni_rua.core.config.LOCAL_CONFIG_PATH', root / 'absent.local.json'):
            prepare(self.source, self.experiment, self.native)
        self.manifest = json.loads((self.experiment / "manifest.json").read_text())
        self.protocol = json.loads((self.experiment / "protocol.json").read_text())

    def test_pairs_have_only_intended_difference_and_isolated_empty_memory(self):
        self.assertEqual(self.protocol["model"]["model"], "gpt-6.1-sol")
        self.assertEqual(self.protocol["model"]["effort"], "high")
        self.assertEqual(self.protocol["model"]["codex_home"], str(Path("~/.codex").expanduser().resolve()))
        cases = self.manifest["cases"]
        self.assertEqual(len(cases), 12)
        self.assertEqual(len({case["grasp_memory_dir"] for case in cases}), 12)
        self.assertFalse(cases[0]["motion_review"])
        self.assertTrue(cases[1]["motion_review"])
        self.assertEqual(cases[0]["task"], "classify_objects_by_language")
        self.assertEqual(sum(pair["order"][0] == "control" for pair in self.protocol["pairs"]), 3)
        for pair in self.protocol["pairs"]:
            a, b = [case for case in cases if pair["pair"] in case["tag"]]
            self.assertEqual({key for key in a if a[key] != b[key]},
                             {"run_dir", "title", "tag", "motion_review", "grasp_memory_dir"})
            self.assertTrue(a["enable_wait"])
            self.assertEqual(a["completion_guidance"], "both")
            self.assertEqual(a["observation_profile"], PROFILE)
        self.assertTrue(all(not Path(case["grasp_memory_dir"]).exists() for case in cases))
        self.assertEqual(self.manifest["max_parallel"], 1)
        self.assertEqual(self.manifest["memory_per_case_mib"], 10240)
        self.assertEqual(self.protocol["execution"]["memory_per_case_mib"], 10240)
        self.assertEqual(self.manifest["oom_retry"], {"max_retries": 0})
        for name in ("pilot-manifest.json", "remaining-manifest.json"):
            phase = json.loads((self.experiment / name).read_text())
            self.assertEqual(phase["oom_retry"], {"max_retries": 0})
            self.assertEqual(phase["memory_per_case_mib"], 10240)
        self.assertEqual(self.protocol["execution"]["oom_retry"], {"max_retries": 0})

    def test_fixed_protocol_freezes_machine_config_before_queue_launch(self):
        from ni_rua.core.config import Config
        from scripts.benchmark_queue import queue_config
        path = self.root / "machine.json"
        path.write_text(json.dumps({
            "backend": {"python": "/machine/native/python", "root": "/wrong/native"},
            "model": {"name": "fixture-model", "effort": "low", "codex_home": "/machine/auth"},
            "features": {"enable_wait": False, "vision_compare": False, "default_task_notes": False,
                         "actor_payload_profile": "lean_v1", "disabled_tools": ["robot_preview_motion"]},
            "images": {"transport_format": "jpeg", "jpeg_quality": 87},
            "run": {"resume": True, "max_actions": 4, "prompt": "unrelated task"},
            "queue": {"max_parallel": 4, "memory_per_case_mib": 1, "oom_retry": {"max_retries": 2}},
        }))
        destination = self.root / "configured-experiment"
        prepare(self.source, destination, self.native, path)
        manifest = json.loads((destination / "manifest.json").read_text())
        protocol = json.loads((destination / "protocol.json").read_text())
        self.assertEqual(manifest["ni_rua_config"], protocol["ni_rua_config"])
        frozen = Config.from_dict(manifest["ni_rua_config"])
        self.assertEqual(frozen.backend.python, "/machine/native/python")
        self.assertEqual(frozen.backend.root, str(self.native))
        self.assertEqual(frozen.model.codex_home, "/machine/auth")
        self.assertEqual((frozen.model.name, frozen.model.effort), ("fixture-model", "low"))
        self.assertEqual(protocol["model"]["model"], "fixture-model")
        self.assertEqual(protocol["model"]["effort"], "low")
        self.assertEqual(protocol["model"]["codex_home"], "/machine/auth")
        self.assertEqual(frozen.features.actor_payload_profile, "full_v1")
        self.assertTrue(frozen.features.enable_wait)
        self.assertTrue(frozen.features.vision_compare)
        self.assertTrue(frozen.features.default_task_notes)
        self.assertEqual(frozen.features.disabled_tools, ())
        self.assertIsNone(frozen.run.max_actions)
        self.assertIsNone(frozen.run.prompt)
        self.assertFalse(frozen.run.resume)
        self.assertEqual((frozen.images.transport_format, frozen.images.jpeg_quality), ("jpeg", 87))
        self.assertEqual((frozen.queue.max_parallel, frozen.queue.memory_per_case_mib,
                          frozen.queue.oom_retry.max_retries), (1, 10240, 0))
        path.write_text('{"features": {"enable_wait": false}}')
        with patch("scripts.benchmark_queue.experiment_environment", side_effect=AssertionError("frozen")):
            self.assertEqual(queue_config(manifest), frozen)
        for name in ("pilot-manifest.json", "remaining-manifest.json"):
            self.assertEqual(json.loads((destination / name).read_text())["ni_rua_config"], frozen.to_dict())
        self.assertTrue(verify(destination)["valid"])

    def test_source_hashes_include_dependency_pins_config_and_vendored_source(self):
        names = ("requirements.lock.txt", "environment.yml", "configs/policy.yaml", "third_party/pinned/module.py", "third_party/pinned/README.md")
        for name in names:
            path = self.source / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text("fixture\n")
        snapshot = source_files(self.source)
        self.assertTrue(set(names) <= snapshot.keys())

    def test_hash_verification_detects_source_and_native_drift_and_refuses_overwrite(self):
        self.assertTrue(verify(self.experiment)["valid"])
        (self.source / "ni_rua/apps/gateway/workbench.py").write_text("changed\n")
        native_file = next(iter(self.protocol["native_input_sha256"]))
        (self.native / native_file).write_text("changed\n")
        result = verify(self.experiment)
        self.assertFalse(result["valid"])
        self.assertIn("source:ni_rua/apps/gateway/workbench.py", result["differences"])
        self.assertIn("native:" + native_file, result["differences"])
        with self.assertRaises(FileExistsError):
            prepare(self.source, self.experiment, self.native)

    def test_missing_runs_remain_unscored_and_unknown(self):
        report = summarize(self.experiment)
        self.assertEqual(report["outcomes"]["control"],
                         dict(planned=6, valid_scored=0, successes=0, unscored_or_unknown=6))
        self.assertTrue(all(pair["outcome"] == "unresolved" for pair in report["pairs"]))
        for run in report["runs"]:
            self.assertIsNone(run["tools"])
            self.assertIsNone(run["task_success"])
            self.assertIsNone(run["token_usage"]["accounting"]["total"]["tokens"]["total_tokens"]["total"])

    def test_report_preserves_unscored_process_evidence_without_promoting_it(self):
        case = self.manifest["cases"][0]
        root = Path(case["run_dir"])
        root.mkdir(parents=True, exist_ok=True)
        partial = dict(normalized_score=0.4, native_process_score=40, task_success=False)
        (root / "summary.json").write_text(json.dumps(dict(status="stopped", valid_episode=True,
            scored_episode=False, primary_outcome="unscored_harness_timeout", normalized_score=None,
            task_success=None, native_process_score=40, partial_native_result=partial,
            termination_authority="harness")))
        value = summarize(self.experiment)
        row = next(item for item in value["runs"] if item["run_dir"] == str(root))
        self.assertIsNone(row["normalized_score"])
        self.assertIsNone(row["task_success"])
        self.assertEqual(row["native_process_score"], 40)
        self.assertEqual(row["partial_native_result"], partial)
        self.assertEqual(value["outcomes"][row["condition"]]["valid_scored"], 0)
        self.assertEqual(value["outcomes"][row["condition"]]["successes"], 0)

    def test_saved_preview_and_terminal_evidence_are_reported_without_visual_inference(self):
        for case in self.manifest["cases"][:2]:
            root = Path(case["run_dir"])
            identity = dict(run_id=root.name, episode_id="ep", env_instance_id="env", env_generation=1)
            store = Store(root, identity)
            try:
                store.save("tool_call", "call", dict(params=dict(tool="robot_preview_motion")))
                store.save("motion_preview", "plan", dict(review=dict(status="blocked")))
                store.event("motion.preview_failed", dict(status="ik_failed"))
                store.event("motion.preview_failed", dict(status="failed", reason="path_infeasible", planner_failure_stage="inverse_kinematics"))
                store.save("action_receipt", "ik", dict(status="failed", reason="path_infeasible", planner_failure_stage="inverse_kinematics"))
                store.save("action_receipt", "path", dict(status="failed", reason="path_infeasible", planner_failure_stage="path_planning"))
            finally:
                store.close()
            manifest = dict(model=self.protocol["model"]["model"], effort=self.protocol["model"]["effort"], observation_profile=PROFILE,
                            motion_review=case["motion_review"], actor_payload_profile="full_v1", enable_wait=True,
                            model_request_limit=170, completion_guidance=dict(mode="both"),
                            grasp_memory_dir=case["grasp_memory_dir"])
            summary = dict(valid_episode=True, scored_episode=True, task_success=case["motion_review"],
                           native_episode=dict(ended=True, native_step_limit=1100))
            (root / "manifest.json").write_text(json.dumps(manifest))
            (root / "summary.json").write_text(json.dumps(summary))
        report = summarize(self.experiment)
        self.assertEqual(report["pairs"][0]["outcome"], "review_win")
        run = report["runs"][1]
        self.assertTrue(all(value is True for value in run["configuration_checks"].values()))
        self.assertEqual(run["preview_statuses"], {"blocked": 1, "planning_failed:ik_failed": 1, "planning_failed:failed": 1})
        self.assertEqual(run["explicit_ik_failure_receipts"], 1)
        self.assertEqual(run["explicit_ik_failure_previews"], 1)
        self.assertEqual(run["failed_receipts_missing_reason"], 0)
        self.assertEqual(run["tools"]["robot_preview_motion"], 1)
        self.assertIsNone(run["visual_mechanisms"]["empty_lifts"])

    def test_saved_legacy_model_is_checked_without_reloading_future_defaults(self):
        config = self.root / "legacy.json"
        config.write_text(json.dumps({"model": {"name": "gpt-6-astra", "effort": "medium",
                                               "codex_home": "/historical/auth"}}))
        destination = self.root / "historical-experiment"
        prepare(self.source, destination, self.native, config)
        frozen = json.loads((destination / "manifest.json").read_text())
        case = frozen["cases"][0]
        run = Path(case["run_dir"])
        run.mkdir(parents=True)
        (run / "manifest.json").write_text(json.dumps(dict(model="gpt-6-astra", effort="medium")))
        (run / "summary.json").write_text("{}")
        with patch("ni_rua.core.config.Config.load", side_effect=AssertionError("history reloaded")):
            report = summarize(destination)
        self.assertTrue(report["runs"][0]["configuration_checks"]["model"])
        self.assertTrue(report["runs"][0]["configuration_checks"]["effort"])
        self.assertEqual(report["runs"][0]["expected"]["effort"], "medium")
        # Earlier protocol files recorded model/effort before unified snapshots.
        protocol_path = destination / "protocol.json"
        protocol = json.loads(protocol_path.read_text())
        del protocol["ni_rua_config"]
        protocol_path.write_text(json.dumps(protocol))
        with patch("ni_rua.core.config.Config.load", side_effect=AssertionError("history reloaded")):
            report = summarize(destination)
        self.assertTrue(report["runs"][0]["configuration_checks"]["model"])
        self.assertTrue(report["runs"][0]["configuration_checks"]["effort"])

    def test_attempt_ledger_keeps_unknown_usage_and_excludes_cumulative_double_count(self):
        root = self.root / "usage"
        (root / "private").mkdir(parents=True)
        events = [dict(kind="model_request_finished", logical_call_id="1", attempt_index=1,
                       source="actor", usage=dict(input_tokens=80, output_tokens=20, total_tokens=100)),
                  dict(kind="model_request_started", logical_call_id="2", attempt_index=1, source="actor")]
        (root / "private/model-requests-test.jsonl").write_text("\n".join(json.dumps(item) for item in events))
        report = token_report(root, {}, dict(token_usage=dict(total=dict(total_tokens=9999))))
        total = report["accounting"]["total"]["tokens"]["total_tokens"]
        self.assertEqual(total, dict(total=None, known_subtotal=100, known_runs=1, unknown_runs=1))

    def test_program_attempts_merge_identity_evidence_and_keep_actor_and_receipts_separate(self):
        case = self.manifest["cases"][0]
        root = Path(case["run_dir"])
        store = Store(root, dict(run_id=root.name, episode_id="ep", env_instance_id="env", env_generation=1))
        try:
            store.save("tool_call", "outer", dict(params=dict(tool="program_run", callId="outer")))
            # Equal names/arguments are separate calls; even an equal ID in the
            # other source must not erase a direct actor invocation.
            for identifier in ("op1", "actor2"):
                store.save("tool_call", identifier, dict(params=dict(tool="robot_move_ee", callId=identifier)))
            for identifier, tool, status in (("op1", "robot_move_ee", "completed"),
                                             ("op2", "robot_move_ee", "dispatching")):
                store.set_state("program_call", identifier, dict(program_id="job", operation_id=identifier, status=status))
                correlation = dict(program_id="job", tool_call_id=identifier)
                store.event("program.call_started", dict(tool=tool), correlation)
                store.event("program.call_started", dict(tool=tool), correlation)
                if status == "completed":
                    store.event("program.call_completed", dict(tool=tool), correlation)
            # A mirrored tool record is evidence for op1, not another attempt.
            store.save("tool_call", "mirror", dict(params=dict(tool="robot_move_ee"),
                correlation=dict(program_id="job", tool_call_id="op1")))
            store.event("program.call_started", dict(tool="geometry_query"),
                        dict(program_id="job", tool_call_id="event-only"))
            store.set_state("program_call", "unnamed", dict(program_id="job", status="dispatching"))
            store.save("action_receipt", "action1", dict(status="rejected", reason="fixture"))
        finally:
            store.close()
        (root / "manifest.json").write_text("{}")
        (root / "summary.json").write_text("{}")
        run = summarize_case(case)
        self.assertEqual(run["tool_attempts_by_source"], {
            "actor": {"program_run": 1, "robot_move_ee": 2},
            "program": {"robot_move_ee": 2, "geometry_query": 1, "unknown": 1}})
        self.assertEqual(run["all_tool_attempts"], {"program_run": 1, "robot_move_ee": 4, "geometry_query": 1, "unknown": 1})
        self.assertEqual(run["tools"]["robot_move_ee"], 4)
        self.assertEqual(run["receipt_statuses"], {"rejected": 1})
        self.assertEqual(run["tool_count_evidence"]["unnamed_program_attempts"], 1)
        self.assertEqual(run["tool_count_evidence"]["identity_coverage"], "complete")
        self.assertEqual(run["analyzer"]["version"], ANALYZER_VERSION)
        self.assertEqual(run["analyzer"]["sha256"], sha(REPO_ROOT / "scripts/motion_review_eval.py"))
        report = summarize(self.experiment)
        self.assertEqual(report["analyzer"], run["analyzer"])
        self.assertEqual(report["frozen_runtime_source"]["root"], str(self.source))
        self.assertEqual(run["evidence_errors"], [])


class MotionReviewGeometryTests(unittest.TestCase):
    """扫掠包围盒间隙几何反例测试 (迁自 test_motion_review.py)。"""

    def test_box_distance_does_not_invent_collision_at_expanded_cube_corner(self):
        import math
        import numpy as np
        from ni_rua.robot.motion.motion_review import _segment_box_clearance

        a = np.array([[1.8, 1.8, 0.0], [2.0, -2.0, 0.0]])
        b = np.array([[1.8, 1.8, 0.0], [2.0, 2.0, 0.0]])
        clearance = _segment_box_clearance(a, b, np.array([-1.0] * 3), np.array([1.0] * 3), np.array([1.0, 0.9]))
        np.testing.assert_allclose(clearance, [math.sqrt(1.28) - 1, 0.1], atol=1e-12)

    def test_internal_runtime_errors_earn_zero_novelty_credit_and_classify_as_unscored_infrastructure(self):
        import time
        from types import SimpleNamespace
        from ni_rua.apps.evaluation.episode import episode_summary
        from ni_rua.apps.evaluation.retry import oom_evidence
        from ni_rua.cortex.supervision.actor_stall import ActorProgressTracker, completed_turn_activity
        from ni_rua.cortex.supervision.recovery_evidence import completed_turn_evidence

        tmp = self.enterContext(tempfile.TemporaryDirectory())
        run_dir = Path(tmp) / "run-crash"
        identity = dict(run_id="run-crash", episode_id="ep-1", env_instance_id="env-1", env_generation=1)
        store = Store(run_dir, identity)
        self.addCleanup(store.close)
        store.save("observation", "obs-1", dict(**identity, observation_id="obs-1", sim_step=0, robot_state_ref="artifact:s"))

        # Turn 1 baseline
        tracker = ActorProgressTracker(evidence_aware=True)
        obs0 = tracker.observe(
            thread_id="t1",
            turn_id="turn-0",
            episode_id="ep-1",
            native_action_count=0,
            activity=set(),
            evidence=completed_turn_evidence(store, thread_id="t1", turn_id="turn-0", episode_id="ep-1", native_action_count=0),
        )
        self.assertTrue(obs0["baseline"])

        # Turn 1 continuation: direct robot_action crashes with TypeError + program_run whose only broker call crashes
        corr_direct = dict(thread_id="t1", turn_id="turn-1", tool_call_id="call-direct-1")
        store.save(
            "tool_call",
            "call-direct-1",
            dict(
                params=dict(
                    callId="call-direct-1",
                    tool="robot_action",
                    arguments=dict(
                        mode="preview",
                        observation_id="obs-1",
                        stages=[dict(id="s1", kind="gripper", arm="left", goal_normalized=0.0)],
                    ),
                ),
                correlation=corr_direct,
            ),
        )
        store.event(
            "tool.internal_runtime_error",
            dict(tool="robot_action", error_type="TypeError", call_id="call-direct-1"),
            corr_direct,
        )

        corr_prog = dict(thread_id="t1", turn_id="turn-1", tool_call_id="call-prog-1")
        store.save(
            "tool_call",
            "call-prog-1",
            dict(
                params=dict(callId="call-prog-1", tool="program_run", arguments=dict(source="x = 1\n")),
                correlation=corr_prog,
            ),
        )
        store.set_state("program", "program-p1", dict(job_id="program-p1", calls=1, correlation=corr_prog))
        store.event(
            "program.call_started",
            dict(tool="robot_action", arguments=dict(mode="direct", observation_id="obs-1", stages=[])),
            dict(thread_id="t1", turn_id="turn-1", program_id="program-p1", tool_call_id="op-1"),
        )
        store.event(
            "tool.internal_runtime_error",
            dict(tool="robot_action", error_type="TypeError", call_id="op-1", program_id="program-p1"),
            dict(thread_id="t1", turn_id="turn-1", program_id="program-p1", tool_call_id="op-1"),
        )

        act1 = completed_turn_activity(store, thread_id="t1", turn_id="turn-1", episode_id="ep-1")
        ev1 = completed_turn_evidence(store, thread_id="t1", turn_id="turn-1", episode_id="ep-1", native_action_count=0)
        self.assertEqual(act1, set())
        self.assertEqual(ev1["activity"], [])
        obs1 = tracker.observe(
            thread_id="t1",
            turn_id="turn-1",
            episode_id="ep-1",
            native_action_count=0,
            activity=act1,
            evidence=ev1,
        )
        self.assertEqual(obs1["new_activity_count"], 0)
        self.assertEqual(obs1["idle_count"], 1)

        # Episode summary with tripped internal runtime error -> unscored_infrastructure and NOT GPU OOM retryable
        session = SimpleNamespace(
            thread_id="t1",
            turn_result={"status": "interrupted"},
            model="gpt-fixture",
            provider="openai",
            price_card=None,
            model_proxy=None,
            model_limit_reached=None,
        )
        harness = SimpleNamespace(
            action_count=0,
            wait_stop_reason="internal_runtime_error",
            internal_runtime_tripped=dict(
                tool="robot_action",
                error_type="TypeError",
                message="unsupported operand type(s)",
            ),
        )
        summary = episode_summary(
            store,
            identity,
            session,
            harness,
            benchmark=True,
            final_status="failed",
            started_at=time.monotonic() - 1.0,
        )
        self.assertEqual(summary["primary_outcome"], "unscored_infrastructure")
        self.assertFalse(summary["scored_episode"])
        self.assertIsNone(summary["normalized_score"])
        self.assertTrue(summary["infrastructure_error"].startswith("internal_runtime_error: robot_action: TypeError:"))
        self.assertIsNone(oom_evidence({"case": {"run_dir": str(run_dir)}, "summary": summary}))



if __name__ == "__main__":
    unittest.main()
