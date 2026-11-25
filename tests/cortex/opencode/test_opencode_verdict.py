"""``turn_verdict`` / ``SettleGuard`` / ``IncompleteTurnGuard`` against message lists captured from the real opencode
2.2.7 binary (design 11 sections 1.6-1.8; fixtures under ``tests/fixtures/opencode/v2/``, built by
``build_v2_fixtures.py``; each file's ``capture`` names the probe).

One fixture per behaviour of design 11 section 5 (Acceptance, OC2-C): steer, late steer, abort in a tool, abort in the
stream, provider 400, retried and exhausted 500, stream-error continuation, ``length`` continuation, foreground and
background subagents, compaction ok/failed, crash without ``idle``; plus the base tool turn on two wires, native
tools, a model switch, and a rejected permission request (no ``idle`` at all). ``Timeline`` covers the API order
(never id order) and non-atomic paged reads. Intermediate states are captured snapshots (``snapshots[i]`` with the ``active``/``inbox``
read at that moment) or prefixes of a captured list (the messages that existed at that point; OpenCode appends).
Snapshots of the design probe family were read after the run went idle and recorded no ``active``/``inbox``; the tests
pass ``QUIET`` (nothing active) for them.
"""

NI_RUA_TEST_TIER = "fast"

import copy
import json
import unittest
from pathlib import Path
from ni_rua.cortex.agents.opencode.verdict import (EXECUTION_INTERRUPTED, INCOMPLETE_ERROR, PERMISSION_REJECTED,
                                                   SETTLE_MIN_GAP_S, IncompleteTurnGuard, SettleGuard, TurnVerdict,
                                                   active_sessions, adoption_boundary, error_code, final_text_of,
                                                   inbox_ids, last_idle_id, status_type, timeline, turn_verdict)
from support.paths import FIXTURES_ROOT

FIXTURES = FIXTURES_ROOT / "opencode" / "v2"
QUIET = {"data": {}}                       # GET /api/session/active with nothing active


def load(name):
    return json.loads((FIXTURES / f"{name}.json").read_text())


def final(fixture):
    """Message list of the last captured snapshot."""
    return fixture["snapshots"][-1]["messages"]


def snapshot(fixture, label):
    return next(s for s in fixture["snapshots"] if s["label"] == label)


def running_of(*sessions):
    return {"data": {session: {"type": "running"} for session in sessions}}


def index_of(messages, predicate):
    return next(i for i, message in enumerate(messages) if predicate(message))


def by_text(messages, text):
    """Index of the first message whose own text, or one of whose text items, contains ``text``."""
    return index_of(messages, lambda m: text in (m.get("text") or "") or any(
        text in (item.get("text") or "") for item in m.get("content") or ()))


def upto(messages, text):
    """Prefix of ``messages`` up to and including the first message that contains ``text``."""
    return messages[:by_text(messages, text) + 1]


def verdict(messages, boundary, **kw):
    kw.setdefault("active", QUIET)
    return turn_verdict(messages, boundary=boundary, **kw)


class Completion(unittest.TestCase):
    def test_tool_turn_on_both_wires(self):
        for name in ("tool_turn", "tool_turn_responses"):
            with self.subTest(name):
                fixture = load(name)
                messages, (prompt,) = final(fixture), fixture["prompts"]
                result = verdict(messages, prompt)
                self.assertEqual((result.status, result.final_text, result.error, result.reason, result.idle),
                                 ("completed", "FINAL-ANSWER-1", None, "succeeded", True))
                self.assertEqual((result.message_id, result.idle_id), (messages[-2]["id"], messages[-1]["id"]))
                self.assertIsNone(result.settle_key)
                # the commentary text shares its message with the tool call: never the final text
                self.assertNotIn("Looking.", result.final_text)

    def test_running_until_the_idle_after_the_boundary(self):
        fixture = load("tool_turn")
        messages, (prompt,) = final(fixture), fixture["prompts"]
        main = fixture["session_id"]
        for end in range(1, len(messages)):               # every prefix before the idle message
            with self.subTest(prefix=end):
                result = verdict(messages[:end], prompt)
                self.assertEqual((result.status, result.reason, result.idle), ("running", "awaiting_idle", True))
        result = verdict(messages, prompt, active=running_of(main))
        self.assertEqual((result.status, result.reason, result.idle), ("running", "active", False))
        # the prompt's id exists before its message does: unknown -> pending, waiting in the inbox -> queued
        for boundary in (None, "msg_not_in_snapshot"):
            self.assertEqual(verdict(messages, boundary).reason, "user_message_pending")
        self.assertEqual(verdict(messages[:0], prompt, inbox={"data": [{"id": prompt, "type": "user"}]}).reason,
                         "queued")

    def test_native_tools_and_model_switch_turns(self):
        fixture = load("native_tools")
        result = verdict(final(fixture), fixture["prompts"][0])
        self.assertEqual((result.status, result.final_text), ("completed", "NATIVE-DONE"))
        fixture = load("model_switch")
        messages, (first, second) = final(fixture), fixture["prompts"]
        self.assertEqual(verdict(messages[:by_text(messages, "A1") + 2], first).final_text, "A1")
        result = verdict(messages, second)            # the model-switched message sits between the two turns
        self.assertEqual((result.status, result.final_text), ("completed", "A2"))

    def test_verdict_does_not_mutate_its_input(self):
        fixture = load("subagent_background")
        messages = final(fixture)
        before = copy.deepcopy(messages)
        verdict(messages, fixture["prompts"][0], children=True, inbox={"data": []})
        self.assertEqual(messages, before)


class Steer(unittest.TestCase):
    def test_steer_during_a_tool_call(self):
        fixture = load("steer")
        prompt, steer = fixture["prompts"]
        captured = snapshot(fixture, "steer_in_inbox")
        self.assertEqual(inbox_ids(captured["inbox"]), [steer])
        self.assertEqual(verdict(captured["messages"], steer, active=captured["active"]).reason, "active")
        # the same moment with nothing active: the steer is still waiting in the inbox
        self.assertEqual(verdict(captured["messages"], steer, inbox=captured["inbox"]).reason, "queued")
        result = verdict(final(fixture), steer, active=fixture["snapshots"][-1]["active"])
        self.assertEqual((result.status, result.final_text), ("completed", "ANSWER-AFTER-STEER"))

    def test_late_steer_during_the_final_text(self):
        """1.19.5 completed on FIRST-FINAL unless the steer's message was matched in time; 2.x writes one idle after
        the extra model call, so the FIRST-FINAL message alone never ends the turn."""
        fixture = load("steer_late")
        messages, (prompt, steer) = final(fixture), fixture["prompts"]
        self.assertEqual(verdict(upto(messages, "FIRST-FINAL"), prompt).reason, "awaiting_idle")
        self.assertEqual(verdict(upto(messages, "LATE-STEER"), steer).reason, "awaiting_idle")
        for boundary in (prompt, steer):
            result = verdict(messages, boundary)
            self.assertEqual((result.status, result.final_text), ("completed", "SECOND-FINAL"))


class Interrupts(unittest.TestCase):
    def test_abort_in_a_tool_then_the_next_turn(self):
        fixture = load("interrupt_tool")
        messages, (first, second) = final(fixture), fixture["prompts"]
        turn1 = messages[:index_of(messages, lambda m: m["type"] == "idle") + 1]
        result = verdict(turn1, first)
        self.assertEqual((result.status, result.reason, result.final_text, result.error),
                         ("interrupted", "interrupted", None, None))
        self.assertEqual(result.message_id, turn1[-2]["id"])
        self.assertEqual(verdict(turn1, first, interrupt_requested=True).reason, "interrupt_requested")
        result = verdict(messages, second)
        self.assertEqual((result.status, result.final_text), ("completed", "SHOULD-NOT-APPEAR-IF-ABORTED"))

    def test_abort_in_the_stream(self):
        fixture = load("interrupt_stream")
        messages = final(fixture)
        result = verdict(messages, fixture["prompts"][0])
        self.assertEqual((result.status, result.message_id, result.final_text),
                         ("interrupted", messages[1]["id"], None))

    def test_interrupt_requested_wins_over_a_late_success(self):
        fixture = load("tool_turn")
        result = verdict(final(fixture), fixture["prompts"][0], interrupt_requested=True)
        self.assertEqual((result.status, result.reason, result.final_text),
                         ("interrupted", "interrupt_requested", None))

    def test_interrupt_cascades_to_a_foreground_subagent(self):
        fixture = load("interrupt_foreground_subagent")
        result = verdict(final(fixture), fixture["prompts"][0], children=True)
        self.assertEqual((result.status, result.reason), ("interrupted", "interrupted"))
        (child,) = fixture["children"].values()
        self.assertEqual(child["messages"][-1]["outcome"], "interrupted")


class Errors(unittest.TestCase):
    def test_empty_success_after_filtered_step_preserves_failure_until_new_work(self):
        fixture = load("error_400")
        prompt, = fixture["prompts"]
        user, step, failed = final(fixture)
        refusal = {"type": "provider.content-filter"}
        step = dict(step, error=refusal, finish="content-filter", rawFinish="refusal")
        failed = dict(failed, error=refusal)
        empty = {"type": "idle", "id": "empty", "outcome": "succeeded"}
        events = [user, step, failed, empty, dict(empty, id="empty-again")]
        # One host snapshot or an empty native wake-up adopted after the earlier terminal was recorded.
        for boundary in (prompt, failed["id"], empty["id"]):
            with self.subTest(boundary=boundary):
                result = verdict(events, boundary)
                self.assertEqual((result.status, result.error, result.idle_id, result.message_id, result.final_text),
                                 ("failed", "opencode_turn_failed:provider.content-filter", failed["id"], step["id"],
                                  None))
        self.assertEqual(verdict(events, prompt, interrupt_requested=True).reason, "interrupt_requested")
        self.assertEqual(verdict(events, prompt, inbox=[{"id": "next"}]).reason, "inbox_pending")

        # A real tool-only step may recover without producing final-answer text.
        tool_step = dict(final(load("tool_turn"))[1], id="after-retry")
        tool_step["content"] = [item for item in tool_step["content"] if item["type"] == "tool"]
        recovered = verdict(events + [tool_step, dict(empty, id="recovered")], prompt)
        self.assertEqual((recovered.status, recovered.final_text, recovered.message_id, recovered.idle_id),
                         ("completed", None, "after-retry", "recovered"))
        # A later host submission has its own verdict, including when it ends without an assistant step.
        new_turn = verdict(events + [dict(user, id="new-prompt"), dict(empty, id="new-idle")], "new-prompt")
        self.assertEqual((new_turn.status, new_turn.error, new_turn.message_id), ("completed", None, None))

    def test_provider_errors_fail_with_a_text_free_code(self):
        for name, code in (("error_400", "opencode_turn_failed:provider.invalid-request:400"),
                           ("error_500_exhausted", "opencode_turn_failed:provider.internal:500"),
                           ("compaction_failed", "opencode_turn_failed:compaction.failed")):
            with self.subTest(name):
                fixture = load(name)
                result = verdict(final(fixture), fixture["prompts"][0])
                self.assertEqual((result.status, result.error, result.reason, result.final_text),
                                 ("failed", code, "failed", None))
                for text in ("ni_rua_model_request_failed", "boom", "template", "body"):
                    self.assertNotIn(text, result.error)

    def test_failed_idle_without_error_falls_back_to_the_step_error(self):
        fixture = load("error_400")
        messages = copy.deepcopy(final(fixture))
        del messages[-1]["error"]
        self.assertEqual(verdict(messages, fixture["prompts"][0]).error,
                         "opencode_turn_failed:provider.invalid-request:400")
        messages[1].pop("error")
        self.assertEqual(verdict(messages, fixture["prompts"][0]).error, "opencode_turn_failed:unknown")
        messages[-1]["outcome"] = "a-new-outcome"
        self.assertEqual(verdict(messages, fixture["prompts"][0]).status, "failed")

    def test_retried_500_recovers(self):
        fixture = load("error_500_retried")
        prompt = fixture["prompts"][0]
        retrying = snapshot(fixture, "retrying")
        self.assertIn("retry", retrying["messages"][-1])
        self.assertEqual(verdict(retrying["messages"], prompt, active=retrying["active"]).reason, "active")
        self.assertEqual(verdict(retrying["messages"], prompt).reason, "awaiting_idle")
        result = verdict(final(fixture), prompt)
        self.assertEqual((result.status, result.final_text), ("completed", "RECOVERED-500"))

    def test_stream_error_and_length_continuations(self):
        for name, final_text, partial in (("stream_error_continuation", "RECOVERED-SE", "partial text her"),
                                          ("length_continuation", "done", "cut")):
            with self.subTest(name):
                fixture = load(name)
                messages, prompt = final(fixture), fixture["prompts"][0]
                synthetic = [i for i, m in enumerate(messages) if m["type"] == "synthetic"]
                self.assertTrue(synthetic)
                for index in synthetic:                 # each continuation point: still running
                    self.assertEqual(verdict(messages[:index + 1], prompt).reason, "awaiting_idle")
                result = verdict(messages, prompt)
                self.assertEqual((result.status, result.final_text), ("completed", final_text))
                self.assertNotIn(partial, result.final_text)


class Subagents(unittest.TestCase):
    def test_foreground_subagent_holds_the_turn_and_settles(self):
        fixture = load("subagent_foreground")
        main, prompt = fixture["session_id"], fixture["prompts"][0]
        (child,) = fixture["children"]
        captured = snapshot(fixture, "child_running")
        self.assertEqual(active_sessions(captured["active"]), {main, child})
        self.assertEqual(verdict(captured["messages"], prompt, active=captured["active"]).reason, "active")
        # only the child active: every active session of the private server holds the turn by default
        self.assertEqual(verdict(captured["messages"], prompt, active=running_of(child)).reason, "active")
        self.assertEqual(verdict(captured["messages"], prompt, active=running_of(child), sessions={main}).reason,
                         "awaiting_idle")
        candidate = verdict(final(fixture), prompt, children=True)
        self.assertEqual((candidate.status, candidate.final_text), ("completed", "MAIN-AFTER-FG-CHILD"))
        self.assertIsNotNone(candidate.settle_key)
        guard = SettleGuard()
        self.assertEqual((guard.observe(candidate, 10.0).status, guard.observe(candidate, 10.2).reason),
                         ("running", "settling"))
        self.assertIs(guard.observe(candidate, 10.0 + SETTLE_MIN_GAP_S), candidate)

    def test_background_subagent_report_runs_inside_the_held_turn(self):
        fixture = load("subagent_background")
        main, prompt = fixture["session_id"], fixture["prompts"][0]
        (child,) = fixture["children"]
        first = snapshot(fixture, "first_idle_child_running")
        self.assertEqual(active_sessions(first["active"]), {child})
        # the main session went idle with an answer while its background child runs: the turn is held
        self.assertEqual(verdict(first["messages"], prompt, active=first["active"], children=True).reason, "active")
        premature = verdict(first["messages"], prompt, active=first["active"], sessions={main})
        self.assertEqual(premature.final_text, "MAIN-FINAL-BEFORE-CHILD-DONE")      # why the child must count
        messages = final(fixture)
        report = index_of(messages, lambda m: m["type"] == "synthetic")
        self.assertGreater(report, by_text(messages, "MAIN-FINAL-BEFORE-CHILD-DONE"))
        self.assertLess(messages[report]["id"], messages[report - 1]["id"])  # minted earlier, listed after the idle
        # the child left `active`, its report waits in the inbox
        self.assertEqual(verdict(first["messages"], prompt, inbox=[messages[report]], children=True).reason,
                         "inbox_pending")
        # the report was delivered but the main session's new execution was not seen active (non-atomic reads)
        self.assertEqual(verdict(messages[:report + 1], prompt, children=True).reason, "execution_pending")
        self.assertEqual(verdict(messages[:report + 2], prompt, children=True).reason, "execution_pending")
        # the child just left `active` and its report is not enqueued yet: only the settle gap catches this
        guard = SettleGuard()
        racing = verdict(first["messages"], prompt, children=True)
        self.assertEqual((racing.status, racing.final_text), ("completed", "MAIN-FINAL-BEFORE-CHILD-DONE"))
        self.assertEqual(guard.observe(racing, 1.0).reason, "settling")
        done = verdict(messages, prompt, children=True)
        self.assertEqual(guard.observe(done, 1.6).reason, "settling")         # another snapshot: settle again
        confirmed = guard.observe(done, 1.6 + SETTLE_MIN_GAP_S)
        self.assertEqual((confirmed.status, confirmed.final_text, confirmed.idle_id),
                         ("completed", "MAIN-AFTER-CHILD-REPORT", messages[-1]["id"]))
        # a model switch after the last idle is no execution
        switched = messages + [dict(id="msg_zz_switch", type="model-switched", model=dict(id="m", providerID="relay"))]
        self.assertEqual(verdict(switched, prompt).final_text, "MAIN-AFTER-CHILD-REPORT")

    def test_background_report_after_the_turn_ended_is_an_adopted_turn(self):
        fixture = load("subagent_background")
        first, messages = snapshot(fixture, "first_idle_child_running")["messages"], final(fixture)
        boundary = last_idle_id(first)
        self.assertEqual(boundary, first[-1]["id"])
        self.assertEqual(verdict(first, boundary).reason, "awaiting_idle")
        result = verdict(messages, boundary)
        self.assertEqual((result.status, result.final_text), ("completed", "MAIN-AFTER-CHILD-REPORT"))
        (child_messages,) = [c["messages"] for c in fixture["children"].values()]
        self.assertEqual(verdict(child_messages, child_messages[0]["id"]).final_text, "CHILD-RESULT")


class Compaction(unittest.TestCase):
    def test_compaction_inside_a_turn(self):
        fixture = load("compaction")
        messages, prompt = final(fixture), fixture["prompts"][0]
        compaction = index_of(messages, lambda m: m["type"] == "compaction")
        self.assertEqual(messages[compaction]["status"], "completed")
        self.assertEqual(verdict(messages[:compaction + 1], prompt).reason, "awaiting_idle")
        result = verdict(messages, prompt)
        self.assertEqual((result.status, result.final_text), ("completed", "FINAL-AFTER-COMPACTION"))
        self.assertNotIn("MARKER-SUMMARY", result.final_text)


class Crash(unittest.TestCase):
    def test_server_killed_mid_tool_writes_no_idle(self):
        fixture = load("crash_without_idle")
        first, second, third = fixture["prompts"]
        running, crashed = snapshot(fixture, "tool_running"), snapshot(fixture, "after_crash")
        self.assertEqual(verdict(running["messages"], second, active=running["active"]).reason, "active")
        result = verdict(crashed["messages"], second, active=crashed["active"], inbox=crashed["inbox"])
        self.assertEqual((result.status, result.reason, result.idle), ("running", "awaiting_idle", True))
        guard = IncompleteTurnGuard()
        self.assertIs(guard.observe(result, 100.0), result)
        self.assertIs(guard.observe(result, 101.9), result)
        self.assertEqual((guard.observe(result, 102.0).status, guard.observe(result, 102.0).error),
                         ("failed", INCOMPLETE_ERROR))
        self.assertEqual(guard.observe(result, 103.0, interrupt_requested=True).status, "interrupted")
        # the turn before the crash, and the turn after the restart (OpenCode repaired the dead call itself)
        messages = final(fixture)
        self.assertEqual(verdict(upto(messages, "A1") + [messages[by_text(messages, "A1") + 1]], first).final_text,
                         "A1")
        dead = messages[by_text(messages, "T2 long tool") + 1]["content"][0]["state"]
        self.assertEqual((dead["status"], dead["error"]["type"]), ("error", "aborted"))
        result = verdict(messages, third)
        self.assertEqual((result.status, result.final_text), ("completed", "A3-after-restart"))


class PermissionRejected(unittest.TestCase):
    """OpenCode 2.2.7 writes NO idle when a permission request is rejected: the tool ends ``aborted`` ("The user
    declined this tool call"), the step ``aborted``, ``session.execution.interrupted`` (``reason: shutdown``); the next
    prompt runs normally. NI-RUA answers every ask ``reject`` (design 11 section 1.3)."""

    def test_rejection_ends_the_turn_without_an_idle(self):
        fixture = load("permission_rejected")
        first, second = fixture["prompts"]
        asked, rejected = snapshot(fixture, "permission_asked"), snapshot(fixture, "after_reject")
        self.assertEqual(verdict(asked["messages"], first, active=asked["active"]).reason, "active")
        self.assertEqual((asked["messages"][-1]["content"][0]["state"]["status"], asked["inbox"]), ("running", []))
        self.assertFalse(any(m["type"] == "idle" for m in rejected["messages"]))
        step = rejected["messages"][-1]
        self.assertEqual((step["finish"], step["error"]["type"], step["content"][0]["state"]["error"]["type"]),
                         ("error", "aborted", "aborted"))
        args = dict(active=rejected["active"], inbox=rejected["inbox"])
        result = verdict(rejected["messages"], first, permission_rejected=True, **args)
        self.assertEqual((result.status, result.error, result.reason, result.message_id, result.idle_id,
                          result.settle_key), ("failed", PERMISSION_REJECTED, "permission_rejected", step["id"], None,
                                               None))
        self.assertIs(SettleGuard().observe(result, 0.0), result)           # terminal at once, no guard needed
        self.assertEqual(verdict(rejected["messages"], first, interrupt_requested=True, **args).status, "interrupted")
        self.assertIsNotNone(verdict(rejected["messages"], first, children=True, permission_rejected=True, **args)
                             .settle_key)
        # the session missed the fact: still terminal, confirmed by the short settle gap (not the incomplete guard)
        unknown = verdict(rejected["messages"], first, **args)
        self.assertEqual((unknown.status, unknown.error), ("failed", EXECUTION_INTERRUPTED))
        guard = SettleGuard()
        self.assertEqual(guard.observe(unknown, 5.0).reason, "settling")
        self.assertIs(guard.observe(unknown, 5.0 + SETTLE_MIN_GAP_S), unknown)
        # the inbox still holds work: not yet
        self.assertEqual(verdict(rejected["messages"], first, inbox=[{"id": second}], permission_rejected=True)
                         .reason, "inbox_pending")
        result = verdict(final(fixture), second)
        self.assertEqual((result.status, result.final_text), ("completed", "AFTER-REJECT-ANSWER"))

    def test_an_aborted_step_followed_by_an_idle_is_an_ordinary_interrupt(self):
        fixture = load("interrupt_tool")
        messages = final(fixture)
        self.assertEqual([m["type"] for m in messages[:3]], ["user", "assistant", "idle"])
        # user + aborted step, the interrupt's idle not listed yet: the host asked for it
        self.assertEqual(verdict(messages[:2], fixture["prompts"][0], interrupt_requested=True).status, "interrupted")
        # once the idle is listed the idle decides, whatever the host facts say
        self.assertEqual(verdict(messages[:3], fixture["prompts"][0], permission_rejected=True).reason, "interrupted")
        empty = dict(id="empty", type="idle", outcome="succeeded")
        self.assertEqual(verdict(messages[:3] + [empty], fixture["prompts"][0]).reason, "interrupted")

    def test_adoption_boundary_after_a_rejected_execution(self):
        fixture = load("permission_rejected")
        rejected = snapshot(fixture, "after_reject")["messages"]
        self.assertEqual(adoption_boundary(rejected), rejected[-1]["id"])
        self.assertEqual(verdict(rejected, adoption_boundary(rejected)).reason, "awaiting_idle")
        background = load("subagent_background")
        first = snapshot(background, "first_idle_child_running")["messages"]
        self.assertEqual(adoption_boundary(first), last_idle_id(first))
        self.assertIsNone(adoption_boundary(first[:2]))
        crashed = snapshot(load("crash_without_idle"), "after_crash")["messages"]
        self.assertEqual(adoption_boundary(crashed), last_idle_id(crashed))     # a dead step ended nothing


class Timeline(unittest.TestCase):
    def test_api_order_is_kept_and_never_sorted_by_id(self):
        fixture = load("subagent_background")
        messages = final(fixture)
        self.assertNotEqual([m["id"] for m in messages], sorted(m["id"] for m in messages))
        self.assertEqual([m["id"] for m in timeline(messages)], [m["id"] for m in messages])
        # the report is delivered, its execution not listed yet: in API order it follows the idle (running); an id
        # sort would move it before the idle and end the turn on the answer written before the child finished
        delivered = messages[:next(i for i, m in enumerate(messages) if m["type"] == "synthetic") + 1]
        self.assertEqual(verdict(delivered, fixture["prompts"][0]).reason, "execution_pending")
        resorted = sorted(delivered, key=lambda m: m["id"])
        self.assertEqual(verdict(resorted, fixture["prompts"][0]).final_text, "MAIN-FINAL-BEFORE-CHILD-DONE")

    def test_non_atomic_pages(self):
        fixture = load("tool_turn")
        messages, prompt = final(fixture), fixture["prompts"][0]
        # page 1 read while the final step was still open, page 2 (read later) already holds the idle
        stale = copy.deepcopy(messages)
        stale[2].pop("finish")
        stale[2]["time"].pop("completed")
        self.assertEqual(verdict(stale, prompt).reason, "stale_snapshot")
        running_tool = copy.deepcopy(messages)
        running_tool[1]["content"][2]["state"] = dict(status="running", input={"camera": "front"}, metadata={})
        self.assertEqual(verdict(running_tool, prompt).reason, "stale_snapshot")
        # the same message on two pages: first position, freshest copy
        overlapped = stale[:3] + messages[2:]
        self.assertEqual([m["id"] for m in timeline(overlapped)], [m["id"] for m in messages])
        self.assertEqual(verdict(overlapped, prompt).final_text, "FINAL-ANSWER-1")
        # a timeline that grew between pages is just a longer prefix
        self.assertEqual(verdict(messages + [dict(id="msg_zz_more", type="user", text="x")], prompt).reason,
                         "execution_pending")
        self.assertEqual(timeline([{"id": 1}, "x", {"no": "id"}]), [])


class Guards(unittest.TestCase):
    def test_incomplete_guard_resets_on_activity_and_terminals(self):
        guard = IncompleteTurnGuard(min_gap_s=1.0)
        idle = TurnVerdict("running", reason="awaiting_idle", idle=True)
        guard.observe(idle, 0.0)
        guard.observe(TurnVerdict("running", reason="active"), 0.5)
        self.assertIs(guard.observe(idle, 1.5), idle)              # re-armed at 1.5
        self.assertEqual(guard.observe(idle, 2.5).error, INCOMPLETE_ERROR)
        done = TurnVerdict("completed", reason="succeeded", idle=True)
        self.assertIs(guard.observe(done, 3.0), done)
        self.assertIsNone(guard.first_idle_at)
        with self.assertRaises(ValueError):
            IncompleteTurnGuard(min_gap_s=-1)

    def test_settle_guard_passes_verdicts_without_a_key_and_resets(self):
        guard = SettleGuard(min_gap_s=0.0)
        plain = TurnVerdict("completed", reason="succeeded", idle=True)
        self.assertIs(guard.observe(plain, 0.0), plain)
        keyed = TurnVerdict("completed", reason="succeeded", idle=True, settle_key=(3, "msg_c"))
        self.assertEqual(guard.observe(keyed, 0.0).reason, "settling")
        guard.observe(TurnVerdict("running", reason="active"), 0.1)
        self.assertEqual(guard.observe(keyed, 0.2).reason, "settling")
        self.assertIs(guard.observe(keyed, 0.2), keyed)
        with self.assertRaises(ValueError):
            SettleGuard(min_gap_s=float("nan"))


class Helpers(unittest.TestCase):
    def test_status_and_active_forms(self):
        for value, expected in (({"type": "running"}, "running"), (None, "idle"), ("", "idle"), ("running", "running"),
                                ({"data": {"ses_a": {"type": "running"}}}, "running"), ({}, "idle")):
            self.assertEqual(status_type(value, "ses_a"), expected)
        with self.assertRaises(TypeError):
            status_type(3)
        self.assertEqual(active_sessions({"data": {"ses_a": {"type": "running"}}}), {"ses_a"})
        self.assertEqual(active_sessions({"ses_a": {"type": "running"}, "ses_b": None}), {"ses_a"})
        self.assertEqual(active_sessions(["ses_a", 7]), {"ses_a"})
        self.assertEqual(active_sessions(None), frozenset())
        with self.assertRaises(TypeError):
            active_sessions("ses_a")
        self.assertEqual(inbox_ids(None), [])
        self.assertEqual(inbox_ids([{"id": "msg_a"}, {"x": 1}, "y"]), ["msg_a"])

    def test_error_code_keeps_only_type_and_status(self):
        self.assertEqual(error_code({"type": "provider.internal", "message": "secret text", "status": 503}),
                         "opencode_turn_failed:provider.internal:503")
        self.assertEqual(error_code({"type": "provider stack trace: secret", "status": True}),
                         "opencode_turn_failed:unknown")
        self.assertEqual(error_code(None, "p"), "p:unknown")

    def test_final_text_and_last_idle(self):
        self.assertIsNone(final_text_of({"content": [{"type": "text", "text": "  "}, {"type": "tool"}]}))
        self.assertEqual(final_text_of({"content": [{"type": "text", "text": "a"}, {"type": "reasoning", "text": "r"},
                                                    {"type": "text", "text": "b"}]}), "a\nb")
        self.assertIsNone(last_idle_id([]))
        self.assertEqual(turn_verdict({"data": final(load("tool_turn"))}, boundary=load("tool_turn")["prompts"][0])
                         .status, "completed")                 # a whole page mapping is accepted

if __name__ == "__main__":
    unittest.main()
