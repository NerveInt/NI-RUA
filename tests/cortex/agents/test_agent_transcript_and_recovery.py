"""Merged cortex tests:
- tests/test_actor_presentation_actor_dir.py: The native agent's working directory reaches the model verbatim; other digests stay scrubbed.
- tests/test_lean_reminders.py: Decision-context reminder preservation on ``/responses/compact`` without altering ``/responses``.
- tests/test_recovery_evidence.py: Evidence-aware continuation keeps alternatives and bounds redundant finals.
"""

NI_RUA_TEST_TIER = "component"

from copy import deepcopy
import json
import os
from pathlib import Path
import tempfile
import unittest
from ni_rua.core.config import Config
from ni_rua.core.presentation.actor_presentation import ActorPresentation, metadata_findings
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents.transcript import TranscriptWriter
from ni_rua.cortex.context.task_planning import TaskPlanning, decision_reminder_text
from ni_rua.cortex.session.usage import COUNTING_EQUIVALENCE, TOKEN_FIELDS, account_usage
from unittest.mock import patch
from ni_rua.cortex.supervision.actor_stall import ActorProgressTracker, completed_turn_activity
from ni_rua.cortex.supervision.recovery_evidence import completed_turn_evidence
import threading
import cv2
import numpy as np
from types import SimpleNamespace
from ni_rua.cortex.session.visual_review import VisualReviewer, MODEL
from ni_rua.cortex.context import operation_verification

# ==== tests/test_actor_presentation_actor_dir.py ====

DIGEST = "0123456789abcdef" * 2


class ActorDirectoryTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        base = Path(os.path.realpath(temporary.name)) / "runs" / "so101" / "task"
        self.root = base / "run-e8def49aea524689b206ef9e1626c2f7"
        identity = dict(run_id="presentation", episode_id="ep", env_instance_id="env", env_generation=1)
        self.store = Store(self.root, identity)
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.actor = str(self.root / "actor")

    def test_actor_dir_and_files_survive_while_other_digests_are_removed(self):
        text = "cwd: %s and %s/scripts/a.py id %s end" % (self.actor, self.actor, DIGEST)
        shown = self.presentation.present_text(text)
        self.assertEqual(shown, "cwd: %s and %s/scripts/a.py id  end" % (self.actor, self.actor))

    def test_actor_dir_inside_json_text(self):
        shown = self.presentation.present_text(json.dumps(dict(cwd=self.actor, file=self.actor + "/x.npy", h=DIGEST)))
        value = json.loads(shown)
        self.assertEqual((value["cwd"], value["file"]), (self.actor, self.actor + "/x.npy"))
        self.assertNotIn(DIGEST, shown)
        line = self.presentation.present_text("log " + self.actor + "\n" + json.dumps(dict(cwd=self.actor)) + "\n")
        self.assertEqual(line.count(self.actor), 2)

    def test_realpath_variant_is_kept(self):
        link = Path(self.root.parent) / "link"
        link.symlink_to(self.root)
        presentation = ActorPresentation(type("S", (), dict(root=link, lock=self.store.lock,
                                                          states=self.store.states))())
        real = os.path.realpath(str(link / "actor"))
        shown = presentation.present_text("a %s/f b %s/f" % (link / "actor", real))
        self.assertEqual(shown, "a %s/f b %s/f" % (link / "actor", real))

    def test_other_run_paths_are_scrubbed_as_before(self):
        private = str(self.root) + "/private/x"
        shown = self.presentation.present_text(private + " " + self.actor)
        self.assertEqual(shown, str(self.root.parent) + "/run-/private/x " + self.actor)
        self.assertEqual(self.presentation.present_text(str(self.root) + "/actor2"),
                         str(self.root.parent) + "/run-/actor2")

    def test_published_harness_source_named_by_the_instructions_survives(self):
        from ni_rua.cortex.context.actor_instructions import instructions, legacy_observation_instructions
        source = self.root / "private" / "actor-source"
        text = instructions(legacy_observation_instructions(), None, source_root=source)
        self.assertIn(f"Read the public Harness implementation under {source}.", self.presentation.present_text(text))
        shown = self.presentation.present_text("cat %s/ni_rua/runtime/harness.py id %s" % (source, DIGEST))
        self.assertEqual(shown, "cat %s/ni_rua/runtime/harness.py id " % source)
        value = json.loads(self.presentation.present_text(json.dumps(dict(path=str(source) + "/ni_rua/x.py"))))
        self.assertEqual(value["path"], str(source) + "/ni_rua/x.py")

    def test_harness_source_realpath_variant_is_kept(self):
        link = Path(self.root.parent) / "link"
        link.symlink_to(self.root)
        presentation = ActorPresentation(type("S", (), dict(root=link, lock=self.store.lock,
                                                          states=self.store.states))())
        real = os.path.realpath(str(link / "private" / "actor-source"))
        shown = presentation.present_text("a %s/f b %s/f" % (link / "private" / "actor-source", real))
        self.assertEqual(shown, "a %s/f b %s/f" % (link / "private" / "actor-source", real))

    def test_a_sentence_period_after_the_directory_keeps_it(self):
        shown = self.presentation.present_text("Work in %s. Then %s/x.py." % (self.actor, self.actor))
        self.assertEqual(shown, "Work in %s. Then %s/x.py." % (self.actor, self.actor))

    def test_other_private_paths_stay_scrubbed(self):
        parent = str(self.root.parent)
        for name in ("private/actor-source2", "private/actor-source-old/x", "private/actor-source.bak",
                     "private/opencode/instructions/NI-RUA.md", "private/tmp/a", "private/codex-home", "actor.bak"):
            with self.subTest(path=name):
                self.assertEqual(self.presentation.present_text(str(self.root) + "/" + name),
                                 parent + "/run-/" + name)

    def test_store_without_root(self):
        class Bare:
            lock = self.store.lock
            states = staticmethod(self.store.states)
        presentation = ActorPresentation(Bare())
        self.assertEqual(presentation.present_text("/x/run-e8def49aea524689b206ef9e1626c2f7/actor"), "/x/run-/actor")

# ==== tests/test_agent_transcript.py ====

IDENTITY = dict(run_id="check", episode_id="ep", env_instance_id="env", env_generation=1)


def usage(**kw):
    return {key: kw.get(key, 1) for key in TOKEN_FIELDS}


class TranscriptWriterCheck(unittest.TestCase):
    def setUp(self):
        self.dir = self.enterContext(tempfile.TemporaryDirectory())
        self.store = Store(self.dir, IDENTITY)
        self.addCleanup(self.store.close)
        self.writer = TranscriptWriter(self.store, backend="claude_code", thread_id="thr")

    def finals(self, turn):
        return self.store.completed_final_texts("thr", turn)

    def test_rows_round_trip_through_readers(self):
        w = self.writer
        turn = w.turn_started()
        w.user_message("do it", item_id="u1")
        w.reasoning("thinking", message_id="m1", item_id="r1")
        w.tool_started("call-1", "observe", {"a": 1}, message_id="m1")
        w.tool_completed("call-1", success=True, content_items=[dict(type="inputText", text="ok")], raw_ref="art:1")
        w.command("c1", "ls", aggregated_output="x", exit_code=0, message_id="m1")
        w.file_change("f1", [dict(path="a", kind="update", diff="d")])
        w.native_tool("n1", "Read", dict(path="a"), result="body")
        w.compaction(item_id="k1")
        w.error("boom", item_id="e1")
        w.assistant_text("all done", message_id="m2", item_id="a1", native_id="nat")
        w.turn_completed("completed")
        row = self.store.transcript_item("call-1")
        self.assertEqual((row["type"], row["tool"], row["success"], row["status"], row["backend"], row["turn_id"]),
                         ("dynamicToolCall", "observe", True, "completed", "claude_code", turn))
        self.assertEqual(row["arguments"], {"a": 1})
        self.assertEqual(row["raw_refs"], ["art:1"])
        self.assertIsInstance(row["durationMs"], int)
        self.assertEqual(self.store.transcript_item("a1")["native_id"], "nat")
        self.assertEqual(self.store.transcript_item("u1")["content"], [dict(type="text", text="do it")])
        self.assertEqual(self.store.transcript_item("r1")["summary"], ["thinking"])
        self.assertEqual(self.store.transcript_item("e1")["error"], dict(message="boom"))
        self.assertEqual(self.finals(turn), ["all done"])
        page = self.store.review_page()
        self.assertEqual([i["id"] for i in page["items"]],
                         ["u1", "r1", "call-1", "c1", "f1", "n1", "k1", "e1", "a1"])
        self.assertTrue(any(e["type"] == "agent.turn_completed" for e in self.store.events()))
        linked = [e for e in page["events"] if e["correlation"].get("turn_id") == turn]
        self.assertEqual(linked, [])  # review_page links events by item/tool-call id only

    def test_tool_failure_and_unknown_status(self):
        self.writer.turn_started()
        self.writer.tool_started("c", "t", {})
        self.writer.tool_completed("c", success=False)
        self.assertEqual(self.store.transcript_item("c")["status"], "failed")
        self.assertEqual(self.store.transcript_item("c")["tool"], "t")

    def test_final_answer_last_text_message_wins_commentary_before(self):
        w = self.writer
        turn = w.turn_started()
        w.assistant_text("first final?", message_id="m1", item_id="a1")
        w.assistant_text("plan", message_id="m2", item_id="a2")
        w.tool_started("t1", "observe", {}, message_id="m2")
        w.tool_completed("t1", success=True)
        w.assistant_text("part one", message_id="m3", item_id="a3")
        w.assistant_text("part two", message_id="m3", item_id="a4")
        self.assertEqual(self.finals(turn), [])  # nothing stamped before the terminal
        w.turn_completed("completed")
        phases = {i: self.store.transcript_item(i)["phase"] for i in ("a1", "a2", "a3", "a4")}
        self.assertEqual(phases, dict(a1="commentary", a2="commentary", a3="final_answer", a4="final_answer"))
        self.assertEqual(self.finals(turn), ["part one", "part two"])
        w.turn_completed("completed")  # idempotent
        self.assertEqual(self.finals(turn), ["part one", "part two"])

    def test_tool_only_last_message_does_not_become_final(self):
        w = self.writer
        turn = w.turn_started()
        w.assistant_text("summary", message_id="m1", item_id="a1")
        w.assistant_text("calling", message_id="m2", item_id="a2")
        w.tool_started("t1", "observe", {}, message_id="m2")
        w.tool_completed("t1", success=True)
        w.tool_started("t2", "observe", {}, message_id="m3")  # last message is tool-only
        w.turn_completed("completed")
        self.assertEqual(self.store.transcript_item("a2")["phase"], "commentary")
        self.assertEqual(self.finals(turn), ["summary"])

    def test_interrupted_and_failed_turns_have_no_final(self):
        for status in ("interrupted", "failed"):
            turn = self.writer.turn_started()
            self.writer.assistant_text("partial", message_id="m" + status, item_id="a" + status)
            self.writer.turn_completed(status, error="x")
            self.assertEqual(self.finals(turn), [])
            self.assertIsNone(self.store.transcript_item("a" + status)["phase"])
        with self.assertRaises(ValueError):
            self.writer.turn_completed("done")

    def test_turns_are_isolated_and_subagent_rows_excluded(self):
        w = self.writer
        t1 = w.turn_started()
        w.assistant_text("one", message_id="m1", item_id="a1")
        w.turn_completed("completed")
        t2 = w.turn_started()
        w.assistant_text("two", message_id="m2", item_id="a2")
        w.assistant_text("child says", message_id="cm", item_id="child", parent_item_id="task-1")
        w.turn_completed("completed")
        self.assertEqual(self.finals(t1), ["one"])
        self.assertEqual(self.finals(t2), ["two"])
        child = self.store.transcript_item("child")
        self.assertEqual(child["parent_item_id"], "task-1")
        self.assertIsNone(child["phase"])
        # a subagent-only tail must not displace the actor's final
        w2 = self.writer
        t3 = w2.turn_started()
        w2.assistant_text("actor", message_id="m3", item_id="a3")
        w2.assistant_text("later child", message_id="cm2", item_id="child2", parent_item_id="task-1")
        w2.turn_completed("completed")
        self.assertEqual(self.finals(t3), ["actor"])

    def test_streaming_update_keeps_stamped_phase_and_row_position(self):
        w = self.writer
        turn = w.turn_started()
        w.assistant_text("he", message_id="m", item_id="a", status="inProgress")
        w.assistant_text("hello", message_id="m", item_id="a")
        w.turn_completed("completed")
        w.assistant_text("hello", message_id="m", item_id="a")  # late replay must not drop the stamp
        self.assertEqual(self.finals(turn), ["hello"])

    def test_steer_row_event_and_permission_denial(self):
        w = self.writer
        turn = w.turn_started()
        w.steer("look left", item_id="s1")
        row = self.store.transcript_item("s1")
        self.assertEqual((row["type"], row["steer"], row["turn_id"]), ("userMessage", True, turn))
        w.permission_denied("Bash", tool_use_id="tu1", reason="not allowed")
        events = {e["type"]: e for e in self.store.events() if e["type"].startswith("agent.")}
        self.assertEqual(events["agent.turn_steered"]["correlation"]["item_id"], "s1")
        denial = events["agent.permission_denied"]
        self.assertEqual((denial["source"], denial["payload"]["tool"], denial["correlation"]["tool_call_id"],
                          denial["correlation"]["thread_id"]), ("agent_backend", "Bash", "tu1", "thr"))
        with self.assertRaises(ValueError):
            w.event("codex.turn")

    def test_usage_sum_and_unknown_is_sticky(self):
        w = self.writer
        w.turn_started()
        w.add_usage(usage(input_tokens=10, output_tokens=3, total_tokens=13), model_context_window=1000)
        record = w.add_usage(usage(input_tokens=5, output_tokens=2, total_tokens=7, cache_write_input_tokens=None,
                                   reasoning_output_tokens=None))
        self.assertEqual(record["total"]["input_tokens"], 15)
        self.assertEqual(record["total"]["total_tokens"], 20)
        self.assertIsNone(record["total"]["cache_write_input_tokens"])
        self.assertIsNone(record["total"]["reasoning_output_tokens"])
        self.assertEqual(record["unknown_calls"], 1)
        self.assertIsNone(record["raw_ref"])
        self.assertEqual(record["last"]["input_tokens"], 5)
        record = w.add_usage(usage())  # later known counters never resurrect an unknown total
        self.assertIsNone(record["total"]["cache_write_input_tokens"])
        self.assertEqual(self.store.read_state("runtime", "token_usage"), record)
        relay = self.store.read_state("runtime", "model_usage_relay")
        self.assertEqual((relay["calls"], relay["total"]), (3, record["total"]))
        accounting = w.write_accounting(source="relay.anthropic_messages.usage.sum", model="m", provider="p")
        self.assertEqual(accounting["source"], "relay.anthropic_messages.usage.sum")
        self.assertIsNone(accounting["tokens"]["cache_write_input_tokens"])
        self.assertEqual(accounting["tokens"]["input_tokens"], 16)

    def test_usage_resumes_from_store_for_same_thread_only(self):
        self.writer.add_usage(usage(input_tokens=4))
        again = TranscriptWriter(self.store, backend="claude_code", thread_id="thr")
        self.assertEqual(again.add_usage(usage(input_tokens=4))["total"]["input_tokens"], 8)
        other = TranscriptWriter(self.store, backend="claude_code", thread_id="other")
        self.assertEqual(other.add_usage(usage(input_tokens=4))["total"]["input_tokens"], 4)


class UsageAccountingCheck(unittest.TestCase):
    def test_account_usage_default_source_unchanged_and_override(self):
        snapshot = dict(total={key: 1 for key in TOKEN_FIELDS}, thread_id="t")
        self.assertEqual(account_usage(snapshot)["source"], "codex.thread/tokenUsage/updated.total")
        self.assertEqual(account_usage(snapshot, source="relay.x.usage.sum")["source"], "relay.x.usage.sum")

    def test_counting_equivalence(self):
        self.assertEqual(set(COUNTING_EQUIVALENCE.values()), {"all_delivered_usable_generation_completions"})
        self.assertIn("delivered_usable_actor_helper_compact_responses_v2", COUNTING_EQUIVALENCE)
        self.assertIn("delivered_usable_generation_completions_v1", COUNTING_EQUIVALENCE)

# ==== tests/test_lean_reminders.py ====

PREFIX = "Current task decision boundary\nThe saved plan and attention"
LONG_OBS = "observation-" + "a" * 32


def body(value):
    return json.dumps(value, ensure_ascii=False, separators=(",", ":"))


def native(count):
    """Native Codex input after ``count`` tool steps."""
    items = [
        {"type": "message", "role": "developer", "content": [{"type": "input_text", "text": "rules"}]},
        {"type": "message", "role": "user", "content": [{"type": "input_text", "text": "task"}]},
    ]
    for index in range(count):
        call_id = f"call_{index}"
        items += [
            {"type": "reasoning", "encrypted_content": f"enc{index}"},
            {"type": "function_call", "name": "robot_observe", "arguments": "{}", "call_id": call_id},
            {
                "type": "function_call_output",
                "call_id": call_id,
                "output": [{"type": "input_text", "text": json.dumps({"observation_id": f"obs_{index + 1}"})}],
            },
        ]
    return items


def reminders(items):
    return [
        item["content"][0]["text"]
        for item in items
        if isinstance(item, dict)
        and item.get("role") == "assistant"
        and isinstance(item.get("content"), list)
        and item["content"]
        and str(item["content"][0].get("text", "")).startswith(PREFIX)
    ]


class CompactionDecisionReminderTests(unittest.TestCase):
    """CodexSession._task_decision_context + _present_model_request contract."""

    def session(self, *, required=True):
        try:
            from ni_rua.cortex.agents.codex.session import CodexSession
        except ImportError as error:
            self.skipTest(str(error))
        temporary = self.enterContext(tempfile.TemporaryDirectory())
        root = Path(temporary)
        self.identity = dict(run_id="rem", episode_id="episode", env_instance_id="environment", env_generation=1)
        store = Store(root / "run", self.identity)
        self.addCleanup(store.close)
        config = Config.from_dict({"model": {"codex_home": str(root / "fixture-config")}})
        session = CodexSession(store.root, store, lambda _: {}, config=config)
        self.addCleanup(session.close)
        session.thread_id = "thread-owned"
        session.task_planning = TaskPlanning(store, required=required)
        return session

    def forward(self, session, items, *, endpoint="/responses", source="actor", thread_id=None):
        identity = dict(
            source=source,
            native_thread_id=session.thread_id if thread_id is None else thread_id,
            endpoint=endpoint,
        )
        with_ctx = session._task_decision_context({"input": deepcopy(items)}, identity)
        return session._present_model_request(with_ctx)["input"]

    def test_normal_responses_extend_strict_prefix_without_injected_reminders(self):
        session = self.session(required=True)
        obs = dict(self.identity, observation_id=LONG_OBS)
        session.task_planning.call(dict(plan="Move and verify target.", ledger=[]), self.identity, obs, {})

        first = self.forward(session, native(2), endpoint="/responses")
        second = self.forward(session, native(3), endpoint="/responses")
        self.assertEqual(reminders(first), [])
        self.assertEqual(reminders(second), [])
        self.assertEqual(body(second[: len(first)]), body(first))

    def test_compact_endpoint_injects_deduplicates_and_scrubs_decision_reminder(self):
        session = self.session(required=True)
        # Before any plan is submitted, required=True injects the no-plan boundary reminder on /responses/compact.
        unplanned = self.forward(session, native(1), endpoint="/responses/compact")
        self.assertEqual(len(reminders(unplanned)), 1)
        self.assertEqual(reminders(unplanned)[0], decision_reminder_text(None))

        # Optional planning with no stored plan leaves /responses/compact untouched.
        optional_session = self.session(required=False)
        self.assertEqual(reminders(self.forward(optional_session, native(1), endpoint="/responses/compact")), [])

        # Stored plan with raw internal observation_id is injected once (replacing stale reminders) and scrubbed to obs_N.
        obs = dict(self.identity, observation_id=LONG_OBS)
        session.task_planning.call(
            dict(
                plan="Pick the red block and check clearance.",
                attention=[dict(concern="Check jaw gap.", check="Compare wrist RGB.", when="Before lift.")],
                ledger=[],
            ),
            self.identity,
            obs,
            {},
        )
        stale_reminder = {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": PREFIX + "\nStale snapshot"}],
        }
        compacted = self.forward(session, native(2) + [stale_reminder], endpoint="/responses/compact")
        found = reminders(compacted)
        self.assertEqual(len(found), 1)
        self.assertNotIn("Stale snapshot", found[0])
        self.assertEqual(metadata_findings(found[0]), [])
        self.assertNotIn(LONG_OBS, found[0])
        snapshot = json.loads(found[0].splitlines()[-1])
        self.assertEqual(snapshot["plan_id"], "P1")
        self.assertEqual(snapshot["revision"], 1)
        self.assertTrue(snapshot["plan_observation_id"].startswith("obs_"))

        # Helper threads or foreign thread IDs never receive the actor's decision context.
        self.assertEqual(reminders(self.forward(session, native(1), endpoint="/responses/compact", source="helper")), [])
        self.assertEqual(
            reminders(self.forward(session, native(1), endpoint="/responses/compact", thread_id="other-thread")), []
        )

# ==== tests/test_recovery_evidence.py ====

class RecoveryEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = self.enterContext(tempfile.TemporaryDirectory())
        self.store = Store(Path(self.tmp), dict(run_id="r", episode_id="e",
                           env_instance_id="env", env_generation=1))
        self.addCleanup(self.store.close)
        self.store.save("observation", "o", dict(observation_id="o", episode_id="e", sim_step=990))
        self.tracker = ActorProgressTracker(evidence_aware=True)

    def candidate(self, turn, *, label="approach", z=.8, cost=30, remaining=11,
                  status="rejected", binding=99, valid=True, thread="t", trajectory=False,
                  old_state=False, duplicate=False, reason=None, program=None):
        correlation = dict(thread_id=thread, turn_id=turn, tool_call_id="call-" + turn)
        args = dict(observation_id="o", include_trajectory=False,
                    candidates=[dict(label=label, move=dict(arm="left", grasp_position_m=[0., 0., z]))])
        if duplicate:
            args["candidates"].append(dict(label=label + "-again",
                move=dict(arm="left", grasp_position_m=[0., 0., z])))
        params = dict(tool="robot_preview_cost", arguments=args)
        if program:
            correlation["program_id"] = program
            self.store.event("program.call_started", params, correlation)
        else:
            self.store.save("tool_call", "call-" + turn, dict(correlation=correlation, params=params))
        payload = dict(status="completed", same_measured_state_verified=valid,
                       zero_step_check_passed=valid, observation_id="o",
                       state_binding=dict(native_action_count=binding, sha256="a" * 64),
                       state_binding_after=dict(native_action_count=binding + int(old_state), sha256="b" * 64
                           if old_state else "a" * 64),
                       action_budget=dict(native_steps_used=binding, native_steps_remaining=remaining),
                       candidates=[dict(label=label, nominal_native_steps=cost, status=status,
                                        feasible=status == "planned", reason=(reason or
                                            (None if status == "planned" else "motion_safety_rejected")))])
        if status == "rejected":
            payload["candidates"][0]["motion_safety"] = dict(status="reject", arms={"left": dict(
                status="reject", reasons=["insufficient_remaining_native_budget"],
                nominal_native_steps=cost, episode_remaining_native_steps=remaining,
                exceeded_budgets={"episode": remaining})})
        if duplicate:
            payload["candidates"].append(dict(payload["candidates"][0], label=label + "-again"))
        if trajectory:
            payload["candidates"][0]["trajectory_ref"] = "artifact:" + turn
        self.store.event("motion.cost_preview", payload, correlation)

    def evidence(self, turn, *, remaining=11, count=99):
        return completed_turn_evidence(self.store, thread_id="t", turn_id=turn,
                                       episode_id="e", native_action_count=count,
                                       native_steps_remaining=remaining)

    def batch_candidate(self, turn, *, cost=30, z=.8, count=99, zero_step=True):
        correlation = dict(thread_id="t", turn_id=turn, tool_call_id="call-" + turn)
        self.store.set_state("actor_id", "obs_1", dict(alias="obs_1", original="o"))
        args = dict(observation_id="obs_1", stages=[dict(id=turn, arm="left", kind="move", position_m=[0, 0, z])])
        self.store.save("tool_call", "call-" + turn, dict(correlation=correlation,
            params=dict(tool="robot_action", arguments=args)))
        self.store.event("action.batch_preview", dict(plan_id="p-" + turn, status="planned",
            observation_id="o", physics_advanced=not zero_step, executed_native_steps=0 if zero_step else 1,
            planned_native_steps=cost, trajectory_ref="artifact:" + turn,
            action_budget=dict(native_steps_used=count, native_steps_remaining=11)), correlation)

    def test_unified_batch_costs_keep_existing_budget_recovery_limit(self):
        self.observe("baseline")
        for turn, z in (("first", .7), ("second", .9)):
            self.batch_candidate(turn, z=z)
            result = self.observe(turn)
            self.assertEqual(result["triggered"], turn == "second")
            self.assertEqual(result["new_activity_count"], 0)
        self.assertEqual(result["reason"], "budget_rejection_final_limit")
        self.assertEqual(result["idle_limit"], 2)
        self.assertEqual(result["recovery_limit"], 6)

    def test_unified_renamed_affordable_batch_is_same_evidence(self):
        self.batch_candidate("baseline", cost=7)
        self.observe("baseline")
        self.batch_candidate("renamed", cost=7)
        result = self.observe("renamed")
        self.assertEqual(result["new_activity_count"], 0)
        self.assertEqual(result["recovery_evidence"]["tested_candidate_count"], 1)
        self.assertEqual(self.evidence("renamed")["candidates"][0]["cost_scope"],
                         "complete_submitted_batch_excluding_optional_settle")

    def test_unified_stale_or_nonzero_preview_keeps_request_credit(self):
        for turn, options in (("stale", dict(count=98)), ("advanced", dict(zero_step=False))):
            self.batch_candidate(turn, **options)
            evidence = self.evidence(turn)
            self.assertEqual(evidence["candidates"], [])
            self.assertEqual(evidence["budget_rejected_count"], 0)
            self.assertTrue(evidence["activity"][0].startswith("robot_action:"))

    def test_program_cost_preview_uses_existing_budget_witness_checks(self):
        self.candidate("program-cost", program="program-1")
        evidence = self.evidence("program-cost")
        self.assertEqual(len(evidence["candidates"]), 1)
        self.assertEqual(evidence["budget_rejected_count"], 1)

    def test_program_batch_preview_requires_prior_request_in_same_context(self):
        for mismatch in (None, "program_id", "turn_id", "thread_id", "episode_id",
                         "missing_program", "late_request", "stale_budget", "observation"):
            turn = "program-" + str(mismatch)
            with self.subTest(mismatch=mismatch):
                correlation = dict(thread_id="t", turn_id=turn, program_id="program-1", tool_call_id="inner-call")
                request_correlation = dict(correlation)
                if mismatch in ("program_id", "turn_id", "thread_id"):
                    request_correlation[mismatch] = "other"
                elif mismatch == "missing_program":
                    request_correlation.pop("program_id")
                request = dict(tool="robot_action", arguments=dict(observation_id="other" if
                    mismatch == "observation" else "o", stages=[dict(id="reach", arm="left",
                        kind="move", position_m=[0, 0, .8])]))
                payload = dict(status="planned", observation_id="o", physics_advanced=False,
                    executed_native_steps=0, planned_native_steps=30,
                    action_budget=dict(native_steps_used=98 if mismatch == "stale_budget" else 99,
                        native_steps_remaining=11))
                if mismatch == "late_request":
                    self.store.event("action.batch_preview", payload, correlation)
                with patch.dict(self.store.identity, episode_id="other" if mismatch == "episode_id" else "e"):
                    self.store.event("program.call_started", request, request_correlation)
                if mismatch != "late_request":
                    self.store.event("action.batch_preview", payload, correlation)
                evidence = self.evidence(turn)
                self.assertEqual(len(evidence["candidates"]), 1 if mismatch is None else 0)
                self.assertEqual(evidence["budget_rejected_count"], 1 if mismatch is None else 0)

    def observe(self, turn, *, remaining=11, count=99):
        return self.tracker.observe(thread_id="t", turn_id=turn, episode_id="e",
            native_action_count=count, activity=completed_turn_activity(self.store,
                thread_id="t", turn_id=turn, episode_id="e"),
            evidence=self.evidence(turn, remaining=remaining, count=count))

    def test_distinct_unaffordable_endpoints_get_two_recovery_turns_not_six(self):
        self.observe("baseline")
        self.candidate("first", z=.7)
        first = self.observe("first")
        self.assertFalse(first["triggered"])  # One failed candidate never ends recovery.
        self.assertEqual(first["idle_count"], 1)
        self.candidate("second", z=.9)
        second = self.observe("second")
        self.assertTrue(second["triggered"])
        self.assertEqual(second["reason"], "budget_rejection_final_limit")
        self.assertEqual(second["recovery_evidence"]["tested_candidate_count"], 2)
        self.assertIn("not global task infeasibility", second["recovery_evidence"]["scope"])

    def test_viable_alternative_after_budget_rejection_keeps_opportunity(self):
        self.observe("baseline")
        self.candidate("first")
        self.observe("first")
        self.candidate("alternative", z=.6, cost=7, status="planned")
        result = self.observe("alternative")
        self.assertFalse(result["triggered"])
        self.assertEqual(result["idle_count"], 0)
        self.assertEqual(result["new_activity_count"], 1)

    def test_repeated_affordable_candidate_and_renamed_ids_are_not_new(self):
        self.candidate("baseline", cost=7, status="planned")
        self.observe("baseline")
        for turn in ("first", "second"):
            self.candidate(turn, label=turn, cost=7, status="planned")
            result = self.observe(turn)
            self.assertEqual(result["new_activity_count"], 0)
        self.assertTrue(result["triggered"])
        self.assertEqual(result["reason"], "no_new_evidence_final_limit")
        self.assertEqual(result["recovery_evidence"]["tested_candidate_count"], 1)

    def test_changed_result_of_same_candidate_is_new(self):
        self.candidate("baseline")
        self.observe("baseline")
        self.candidate("changed", cost=8, status="planned")
        result = self.observe("changed")
        self.assertEqual(result["idle_count"], 0)
        self.assertFalse(result["triggered"])

    def test_actually_delivered_trajectory_can_add_new_information_once(self):
        self.candidate("baseline", cost=8, status="planned")
        self.observe("baseline")
        self.candidate("with-trajectory", cost=8, status="planned", trajectory=True)
        self.assertEqual(self.observe("with-trajectory")["new_activity_count"], 1)
        self.candidate("repeated-trajectory", cost=8, status="planned", trajectory=True)
        self.assertEqual(self.observe("repeated-trajectory")["new_activity_count"], 0)

    def test_new_route_geometry_is_credit_even_when_candidate_over_budget(self):
        self.candidate("baseline")
        self.observe("baseline")
        self.candidate("geometry", trajectory=True)
        result = self.observe("geometry")
        self.assertEqual(result["new_activity_count"], 1)
        self.assertEqual(result["idle_count"], 0)

    def test_default_graphs_on_new_unaffordable_endpoints_do_not_refresh_idle(self):
        self.observe("baseline")
        for index in (1, 2):
            turn = str(index)
            self.candidate(turn, z=index / 10, trajectory=True)
            result = self.observe(turn)
            self.assertEqual(result["triggered"], index == 2)

    def test_unknown_mismatched_or_boolean_budget_never_rejects(self):
        for index, budget in enumerate((None, True, 12)):
            turn = "turn-" + str(index)
            self.candidate(turn, z=.7 + index / 10)
            evidence = self.evidence(turn, remaining=budget)
            self.assertEqual(evidence["budget_rejected_count"], 0)
            self.assertTrue(evidence["activity"])
            self.assertFalse(evidence["candidates"][0]["exceeds_episode_budget"])

    def test_stale_or_unverified_preview_preserves_request_credit(self):
        for turn, args in (("stale", dict(binding=98)), ("changed", dict(valid=False))):
            self.candidate(turn, **args)
            evidence = self.evidence(turn)
            self.assertEqual(evidence["candidates"], [])
            self.assertTrue(evidence["activity"][0].startswith("robot_preview_cost:"))

    def test_infeasible_unknown_cost_is_not_budget_proof(self):
        self.candidate("first", cost=None, status="failed")
        evidence = self.evidence("first")
        self.assertEqual(evidence["budget_rejected_count"], 0)
        self.assertTrue(evidence["activity"])

    def test_ik_rejection_is_recorded_as_infeasible_without_budget_failure(self):
        self.candidate("ik", cost=None, status="failed", reason="path_infeasible")
        evidence = self.evidence("ik")
        self.assertEqual(len(evidence["candidates"]), 1)
        self.assertFalse(evidence["candidates"][0]["feasible"])
        self.assertEqual(evidence["candidates"][0]["reason"], "path_infeasible")
        self.assertEqual(evidence["budget_rejected_count"], 0)

    def test_old_state_binding_cannot_create_budget_rejection_or_feasibility(self):
        self.candidate("old", old_state=True)
        evidence = self.evidence("old")
        self.assertEqual(evidence["candidates"], [])
        self.assertEqual(evidence["budget_rejected_count"], 0)
        self.assertTrue(evidence["activity"][0].startswith("robot_preview_cost:"))

    def test_duplicate_candidate_requests_count_once(self):
        self.candidate("duplicate", duplicate=True)
        evidence = self.evidence("duplicate")
        self.assertEqual(len(evidence["candidates"]), 1)
        self.assertEqual(evidence["budget_rejected_count"], 1)

    def test_changed_result_for_same_candidate_is_kept(self):
        self.candidate("changed-result", cost=7, status="planned")
        correlation = dict(thread_id="t", turn_id="changed-result", tool_call_id="call-changed-result-2")
        args = dict(observation_id="o", include_trajectory=False,
                    candidates=[dict(label="approach", move=dict(arm="left", grasp_position_m=[0., 0., .8]))])
        self.store.save("tool_call", correlation["tool_call_id"], dict(correlation=correlation,
            params=dict(tool="robot_preview_cost", arguments=args)))
        self.store.event("motion.cost_preview", dict(status="completed", same_measured_state_verified=True,
            zero_step_check_passed=True, observation_id="o",
            state_binding=dict(native_action_count=99, sha256="a" * 64),
            state_binding_after=dict(native_action_count=99, sha256="a" * 64),
            action_budget=dict(native_steps_used=99, native_steps_remaining=11),
            candidates=[dict(label="approach", nominal_native_steps=8, status="planned", feasible=True)]),
            correlation)
        evidence = self.evidence("changed-result")
        self.assertEqual(len(evidence["candidates"]), 2)
        self.assertEqual({value["nominal_native_steps"] for value in evidence["candidates"]}, {7, 8})
        self.assertEqual(sum(value.startswith("preview_result:") for value in evidence["activity"]), 2)

    def test_budget_witness_does_not_override_affordable_or_unknown_cost(self):
        self.candidate("affordable-witness", cost=10, remaining=11)
        evidence = self.evidence("affordable-witness")
        self.assertEqual(evidence["budget_rejected_count"], 0)
        self.assertFalse(evidence["candidates"][0]["exceeds_episode_budget"])

    def test_actual_delivered_rgb_not_observation_uuid_is_evidence(self):
        def delivery(turn, sha, *, delivered=True):
            obs_id = "obs-" + turn
            self.store.save("observation", obs_id, dict(observation_id=obs_id, episode_id="e", sim_step=990))
            self.store.save("delivery", "delivery-" + turn, dict(observation_id=obs_id,
                correlation=dict(thread_id="t", turn_id=turn), delivered_to_actor=delivered,
                images=[dict(camera_id="cam_head", source_sha256=sha * 64)]))
        delivery("baseline", "a")
        self.observe("baseline")
        delivery("same", "a")
        self.assertEqual(self.observe("same")["new_activity_count"], 0)
        delivery("new", "b")
        self.assertEqual(self.observe("new")["new_activity_count"], 1)
        delivery("undelivered", "c", delivered=False)
        self.assertEqual(self.evidence("undelivered")["delivered_rgb_count"], 0)

    def test_read_only_data_scope_and_thread_scope(self):
        self.candidate("other", thread="not-ours")
        self.store.event("observer.evaluated", {"task_success": "forbidden"})
        with patch.object(self.store, "events", side_effect=AssertionError("no unfiltered event scan")), \
             patch.object(self.store, "records", wraps=self.store.records) as records:
            evidence = self.evidence("other")
        self.assertFalse(evidence["activity"])
        self.assertFalse(evidence["candidates"])
        self.assertEqual({call.args[0] for call in records.call_args_list}, {"observation", "tool_call", "delivery"})
        self.assertNotIn("forbidden", str(evidence))

    def test_native_progress_clears_ledger_and_duplicate_turn_is_ignored(self):
        self.candidate("baseline")
        self.observe("baseline")
        self.assertIsNone(self.observe("baseline"))
        result = self.observe("progress", remaining=10, count=100)
        self.assertTrue(result["baseline"])
        self.assertEqual(result["recovery_evidence"]["tested_candidate_count"], 0)

    def test_native_callback_race_ignores_old_evidence(self):
        self.candidate("baseline")
        evidence = self.evidence("baseline")
        result = self.tracker.observe(thread_id="t", turn_id="baseline", episode_id="e",
                                     native_action_count=100, evidence=evidence)
        self.assertTrue(result["baseline"])
        self.assertEqual(result["idle_limit"], 3)
        self.assertIsNone(self.tracker.recovery_context())

    def test_unknown_native_accounting_hides_prior_plateau_ledger(self):
        self.candidate("baseline")
        self.observe("baseline")
        self.assertIsNotNone(self.tracker.recovery_context())
        result = self.observe("unknown", count=None, remaining=None)
        self.assertEqual(result["status"], "unknown")
        self.assertFalse(result["triggered"])
        self.assertIsNone(self.tracker.recovery_context())

    def test_feature_off_preserves_v1_three_and_six_limits(self):
        self.tracker = ActorProgressTracker(evidence_aware=False)
        self.observe("baseline")
        for index in range(1, 7):
            turn = str(index)
            self.candidate(turn, z=index / 10)
            result = self.observe(turn)
            self.assertEqual(result["triggered"], index == 6)
        self.assertEqual(result["idle_limit"], 3)

# ==== tests/test_visual_review.py ====

class VisualReviewTest(unittest.TestCase):
    def test_operation_packet_hides_hashes_but_retains_private_execution_evidence(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory, dict(run_id='r', episode_id='e', env_instance_id='i', env_generation=1))
        self.addCleanup(store.close)
        rgb = store.put_artifact(cv2.imencode('.png', np.zeros((20, 20, 3), np.uint8))[1].tobytes(), 'rgb', 'image/png')
        source = dict(rgb_ref=rgb, width=20, height=20)
        fingerprint = 'a' * 64
        action_id = 'action-' + 'b' * 24
        execution = dict(status='completed', blockers=[], actions=[dict(action_id=action_id,
            request_sha256=fingerprint, receipt_sha256=fingerprint,
            requested=dict(policy_provenance=dict(checkpoint_sha256=fingerprint)),
            feedback=dict(status='completed', executed_steps=7))],
            after_robot_state=dict(sha256=fingerprint, robot_state_ref=rgb, state=dict(measured_position=[.1, .2, .3])))
        answer = {name: (spec['enum'][0] if 'enum' in spec else [] if spec['type'] == 'array' else 'Visible evidence')
                  for name, spec in operation_verification.ANSWER_SCHEMA['properties'].items()}
        answer['assessment'] = 'uncertain'
        reviewer = VisualReviewer(SimpleNamespace(model=MODEL))
        with patch('ni_rua.cortex.session.visual_review.comparison_inputs', return_value=(None, [source, source])), \
                patch.object(operation_verification, 'execution_inputs', return_value=execution), \
                patch.object(reviewer, '_request', return_value=dict(model=MODEL, status='completed', answer=answer,
                    thread_id='helper-thread', turn_id='helper-turn', context_policy='fresh', backend='sdk')) as request:
            returned = reviewer.review_operation(store, {}, phase='grasp', target='block ' + fingerprint,
                expected_result='retained ' + fingerprint, destination='tray ' + fingerprint,
                action_ids=[action_id], before_observation_id='before', after_observation_id='after', camera_id='head', roi=[1, 1, 4, 4])
        payload = request.call_args.args[0]
        prompt = payload['input'][0]['text']
        for private in (fingerprint, action_id, rgb, 'request_sha256', 'receipt_sha256', 'checkpoint_sha256'):
            self.assertNotIn(private, prompt)
        packet = json.loads(prompt.split('Recorded execution evidence (commands are labeled requested; robot states are measured): ')[1])
        self.assertEqual(packet['actions'][0]['feedback']['executed_steps'], 7)
        self.assertEqual(packet['after_robot_state']['state']['measured_position'], [.1, .2, .3])
        self.assertEqual(ActorPresentation(store).restore(packet)['actions'][0]['action_id'], action_id)
        self.assertEqual(returned['execution'], execution)
        event = store.events()[-1]['payload']
        for key, value in dict(agent_thread_id='helper-thread', agent_turn_id='helper-turn',
                               context_policy='fresh', backend='sdk').items():
            self.assertNotIn(key, returned)
            self.assertEqual(event[key], value)
        self.assertEqual(event['execution']['actions'][0]['request_sha256'], fingerprint)
        self.assertEqual(event['image_encodings'][0]['source_sha256'], store.artifact(rgb)[0]['sha256'])
        archived = json.loads(store.resolve_artifact(event['request_ref'], ('observer',))[1])
        self.assertEqual(archived, payload)

    def test_only_actor_images_exact_model_and_private_usage(self):
        identity = dict(run_id='r',episode_id='e',env_instance_id='i',env_generation=1)
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory,identity)
        self.addCleanup(store.close)
        rgb = store.put_artifact(cv2.imencode('.png', np.zeros((20, 20, 3), np.uint8))[1].tobytes(),'rgb','image/png')
        sources = [dict(rgb_ref=rgb,width=20,height=20),dict(rgb_ref=rgb,width=20,height=20)]
        for index, name in enumerate(('before','after')):
            state = dict(sim_step=index, step_domain='fixture', arms=dict(left=dict(
                source_kind='measurement', q_rad=[index * .1], qdot_rad_s=[index * .2], joint_names=['q1'],
                ee_pose=dict(position_m=[index * .01, 0., 1.], quaternion_xyzw=[0., 0., 0., 1.], source_kind='measurement'),
                gripper=dict(goal_normalized=0., goal_source_kind='command', position_m=[.02, .02],
                             joint_names=['jaw1', 'jaw2'], position_source_kind='measurement'))))
            state_ref = store.put_artifact(json.dumps(state).encode(), 'robot_state', 'application/json')
            store.save('observation',name,dict(**identity, observation_id=name,observation_profile='rgb_proprio_calibrated_v1', robot_state_ref=state_ref,
                sim_step=index,step_domain='fixture',cameras=[dict(camera_id='head',rgb_ref=rgb,width=20,height=20)]))
        client = SimpleNamespace(base_url='http://127.0.0.1:1',_token='test-secret',model=MODEL)
        reviewer = VisualReviewer(client)
        answer = dict(visibility_after='visible',motion='displaced',after_center_px=[10,10],retained_with_gripper='uncertain',evidence='Occluded fingers.')
        result = dict(model=MODEL,status='completed',usage=dict(output_tokens=80),answer=answer,
                      thread_id='helper-thread', turn_id='helper-turn', context_policy='fresh', backend='sdk')
        def response():
            return result
        args=dict(before_observation_id='before',after_observation_id='after',camera_id='head',roi=[1,1,4,4],target='visible cup')
        with patch.object(reviewer, '_request') as request:
            request.side_effect=lambda *a, **kw: response()
            returned=reviewer.review(store,identity,**args)
            self.assertEqual(returned['answer'],answer)
            self.assertNotIn('usage',returned)
            for key, value in dict(agent_thread_id='helper-thread', agent_turn_id='helper-turn',
                                   context_policy='fresh', backend='sdk').items():
                self.assertNotIn(key, returned)
                self.assertEqual(store.events()[-1]['payload'][key], value)
            self.assertEqual([source['observation_id'] for source in returned['sources']], ['before','after'])
            payload=request.call_args.args[0]
            self.assertEqual(payload['model'],MODEL)
            self.assertEqual(sum(p['type']=='image' for p in payload['input']),2)
            bound = [json.loads(part['text'].split('\n', 1)[1]) for part in payload['input']
                     if part['type'] == 'text' and part['text'].startswith(('BEFORE\n', 'AFTER\n'))]
            self.assertEqual([item['sim_step'] for item in bound], [0, 1])
            self.assertEqual([item['robot_state']['arms']['left']['q_rad'] for item in bound], [[0.], [.1]])
            self.assertEqual(returned['sources'][1]['observation_state']['robot_state']['arms']['left']['qdot_rad_s'], [.2])
            result['model']='substituted-model'
            with self.assertRaisesRegex(ValueError,'unverified'):
                reviewer.review(store,identity,**args)

    def test_condition_needs_visible_target_and_failed_request_keeps_unknown_usage(self):
        identity = dict(run_id='r',episode_id='e',env_instance_id='i',env_generation=1)
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(directory,identity)
        self.addCleanup(store.close)
        rgb = store.put_artifact(cv2.imencode('.png', np.zeros((20, 20, 3), np.uint8))[1].tobytes(),'rgb','image/png')
        sources = [dict(rgb_ref=rgb,width=20,height=20)]*2
        reviewer = VisualReviewer(SimpleNamespace(base_url='http://127.0.0.1:1',_token='test-secret',model=MODEL))
        args=dict(before_observation_id='before',after_observation_id='after',camera_id='head',roi=[1,1,4,4],
                  target='visible cup',condition='released_from_gripper')
        answer = dict(visibility_after='not_visible',motion='uncertain',after_center_px=None,
                      retained_with_gripper='not_visually_supported',condition_assessment='met',evidence='Hidden target.')
        raw = dict(model=MODEL,status='completed',answer=answer)
        with patch('ni_rua.cortex.session.visual_review.comparison_inputs',return_value=(None,sources)), patch.object(reviewer,'_request',return_value=raw) as request:
            self.assertEqual(reviewer.review(store,identity,**args)['answer']['condition_assessment'],'uncertain')
            self.assertIn('absence of retention evidence alone', request.call_args.args[0]['input'][0]['text'])
            request.side_effect=TimeoutError('fixture')
            with self.assertRaises(TimeoutError):
                reviewer.review(store,identity,**args)
        failure=store.events()[-1]
        self.assertEqual(failure['type'],'visual.model_review_failed')
        self.assertIsNone(failure['payload']['usage'])
        self.assertTrue(failure['payload']['request_id'])

    def test_cancel_event_is_forwarded_to_official_agent(self):
        from unittest.mock import Mock
        cancel = threading.Event()
        client = SimpleNamespace(model=MODEL, review_evidence=Mock(side_effect=TimeoutError("cancelled")))
        reviewer = VisualReviewer(client)
        with self.assertRaises(TimeoutError):
            reviewer._request(dict(input=[], outputSchema={}), "cancel-fixture", cancel_event=cancel, deadline=0)
        client.review_evidence.assert_called_once_with(input_items=[], output_schema={}, request_id="cancel-fixture", cancel_event=cancel)

if __name__ == "__main__":
    unittest.main()
