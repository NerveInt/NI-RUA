"""``OpenCodeProjector``: OpenCode 2.x messages and ``/api/event`` frames -> neutral transcript rows (design 11 section
1.6), plus the archive read path of 1.19.5 runs (design 11 section 4, ``LegacyArchive1195``).

Inputs are lists and frames captured from the real 2.2.7 binary (``tests/fixtures/opencode/v2``; see
``test_agent_opencode_verdict`` for the fixture conventions). ``Host`` plays the two other writers of the same
transcript exactly as the session and the MCP transport do: the host ``userMessage`` row is written before the prompt
request and bound to the id the request returned (``bind_submission``), and every registry tool call that reached the
MCP endpoint has its ``dynamicToolCall`` row under a host call id (OpenCode sends none).
"""

NI_RUA_TEST_TIER = "component"

import copy
import json
import tempfile
import unittest
from pathlib import Path
from ni_rua.apps.evaluation.native_evidence import load_native_evidence
from ni_rua.core.contracts import new_id
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents.opencode.projector import (INTERRUPTED_TOOL_ERROR, OpenCodeProjector, TOOL_NAME_PREFIX,
                                                     compaction_message_id, subagent_links)
from ni_rua.cortex.agents.transcript import TranscriptWriter
from cortex.opencode.test_opencode_verdict import by_text, final, load, snapshot, upto, verdict
from support.paths import FIXTURES_ROOT

IDENTITY = dict(run_id="check", episode_id="ep", env_instance_id="env", env_generation=1)
ALL_FIXTURES = sorted(path.stem for path in (FIXTURES_ROOT / "opencode" / "v2").glob("*.json"))
# never in a row: tool image bytes, reasoning signatures/encrypted content, native item ids, provider error bodies
NOISE = ("iVBOR", "fake-signature-0001", "reasoningEncryptedContent", "rs_fake", "fc_fake", "api_error",
         "invalid_request_error", "overloaded_error")


def registry_items(messages):
    return [(message["id"], item) for message in messages if message["type"] == "assistant"
            for item in message.get("content") or () if item["type"] == "tool"
            and item["name"].startswith(TOOL_NAME_PREFIX)]


def text_of(content):
    return "\n".join(item["text"] for item in content or () if item.get("type") == "text")


class Host:
    """Session + MCP transport stand-in sharing one store with the projector under test."""

    def __init__(self, store, thread_id):
        self.store, self.thread_id = store, thread_id
        self.writer = TranscriptWriter(store, backend="opencode", thread_id=thread_id)
        self.projector = OpenCodeProjector(self.writer)
        self.called = {}                       # (message id, tool id) -> host call id

    def prompt(self, text, native_id=None):
        turn = self.writer.turn_started()
        item = self.writer.user_message(text)
        if native_id is not None:
            self.projector.bind_submission(item, native_id)
        return turn, item

    def steer(self, text, native_id=None):
        item = self.writer.steer(text)
        if native_id is not None:
            self.projector.bind_submission(item, native_id)
        return item

    def mcp(self, messages, *, skip=()):
        """Transport rows for every registry call of ``messages`` that reached MCP and has none yet (in order)."""
        for message_id, item in registry_items(messages):
            state, key = item["state"], (message_id, item["id"])
            if key in self.called or item["id"] in skip or state["status"] not in ("running", "completed", "error"):
                continue
            call_id = new_id("call")
            self.called[key] = call_id
            self.writer.tool_started(call_id, item["name"][len(TOOL_NAME_PREFIX):], state["input"])
            if state["status"] != "running":
                text = text_of(state.get("content")) or (state.get("error") or {}).get("message", "")
                self.writer.tool_completed(call_id, success=state["status"] == "completed",
                                           content_items=[dict(type="inputText", text=text)])

    def rows(self):
        return {row["id"]: row for row in self.store.states(self.store.TRANSCRIPT_KIND)
                if row.get("thread_id") == self.thread_id}

    def of_type(self, kind):
        return [row for row in self.rows().values() if row["type"] == kind]


class Base(unittest.TestCase):
    def setUp(self):
        self.dir = self.enterContext(tempfile.TemporaryDirectory())
        self.store = Store(self.dir, IDENTITY)
        self.addCleanup(self.store.close)

    def host(self, fixture, store=None):
        return Host(store or self.store, fixture["session_id"])

    def finals(self, host, turn):
        return self.store.completed_final_texts(host.thread_id, turn)

    def writes(self):
        """Last ``state_updates`` sequence: unchanged <=> no row was really written."""
        with self.store.lock:
            return self.store.db.execute("SELECT COALESCE(MAX(seq), 0) FROM state_updates").fetchone()[0]

    def run_single_turn(self, name):
        """Prompt (bound to the captured id), transport rows, sync of every captured snapshot, turn end."""
        fixture = load(name)
        host = self.host(fixture)
        turn, item = host.prompt("PROMPT", fixture["prompts"][0])
        for captured in fixture["snapshots"]:
            host.mcp(captured["messages"])
            host.projector.sync(captured["messages"])
        result = verdict(final(fixture), fixture["prompts"][-1])
        host.writer.turn_completed(result.status, error=result.error)
        return fixture, host, turn, item, result

    def assert_clean(self, host):
        text = json.dumps(host.rows())
        for noise in NOISE:
            self.assertNotIn(noise, text)


class ToolTurn(Base):
    def test_rows_final_answer_and_no_second_registry_row(self):
        for name in ("tool_turn", "tool_turn_responses"):
            with self.subTest(name):
                self.doCleanups()
                self.setUp()
                fixture, host, turn, item, result = self.run_single_turn(name)
                messages = final(fixture)
                tool_message, final_message = messages[1], messages[2]
                rows = host.rows()
                (message_id, call), = registry_items(messages)
                transport = rows[host.called[(message_id, call["id"])]]
                self.assertEqual((transport["native_id"], transport["message_id"], transport["type"]),
                                 ("call_fake_0_0", tool_message["id"], "dynamicToolCall"))
                self.assertNotIn(f"{message_id}:{call['id']}", rows)            # no row of its own
                self.assertEqual(host.of_type("mcpToolCall"), [])
                users = host.of_type("userMessage")
                self.assertEqual([(u["id"], u["native_id"]) for u in users], [(item, fixture["prompts"][0])])
                reasoning = rows[f"{tool_message['id']}:reasoning:0"]
                self.assertEqual((reasoning["summary"], reasoning["message_id"]), (["think-1"], tool_message["id"]))
                commentary = rows[f"{tool_message['id']}:text:0"]
                self.assertEqual((commentary["text"], commentary["phase"], commentary["status"]),
                                 ("Looking.", "commentary", "completed"))
                answer = rows[f"{final_message['id']}:text:0"]
                self.assertEqual((answer["text"], answer["phase"], answer["native_id"]),
                                 ("FINAL-ANSWER-1", "final_answer", final_message["id"]))
                self.assertEqual(self.finals(host, turn), [result.final_text])
                self.assertTrue(all(r["turn_id"] == turn and r["backend"] == "opencode" for r in rows.values()))
                self.assert_clean(host)

    def test_snapshot_facts(self):
        fixture = load("tool_turn")
        host = self.host(fixture)
        host.prompt("PROMPT", fixture["prompts"][0])
        result = host.projector.sync(final(fixture))
        self.assertEqual(result.kind, "snapshot")
        self.assertEqual(result.data["unmatched_registry"], ["call_fake_0_0"])   # its transport row is not written yet
        host.mcp(final(fixture))
        result = host.projector.sync({"data": final(fixture), "cursor": {}})      # a page mapping is accepted
        self.assertEqual((result.data["messages"], result.data["boundary"], result.data["pending_submissions"],
                          result.data["unmatched_registry"], result.data["compactions"], result.data["subagents"],
                          result.data["deferred_users"]), (4, fixture["prompts"][0], 0, [], [], [], []))

    def test_reprojection_is_idempotent_for_every_fixture(self):
        for name in ALL_FIXTURES:
            with self.subTest(name):
                fixture = load(name)
                host = self.host(fixture)
                host.prompt("PROMPT", fixture["prompts"][0])
                for prompt in fixture["prompts"][1:]:
                    host.steer("STEER", prompt)
                children = {child: Host(self.store, child) for child in fixture["children"]}
                for captured in fixture["snapshots"]:
                    host.mcp(captured["messages"])
                    host.projector.sync(captured["messages"])
                for child, entry in fixture["children"].items():
                    children[child].mcp(entry["messages"])
                    children[child].projector.sync(entry["messages"])
                before, seq = self.store.states(self.store.TRANSCRIPT_KIND), self.writes()
                for _ in range(2):
                    host.projector.sync(final(fixture))
                    host.projector.sync(copy.deepcopy(final(fixture)))
                OpenCodeProjector(host.writer).sync(final(fixture))        # a fresh projector (no caches)
                for child, entry in fixture["children"].items():
                    children[child].projector.sync(entry["messages"])
                    OpenCodeProjector(children[child].writer).sync(entry["messages"])
                self.assertEqual(self.writes(), seq)
                self.assertEqual(self.store.states(self.store.TRANSCRIPT_KIND), before)
                self.assert_clean(host)

    def test_two_turns_keep_their_rows_and_a_model_switch_is_reported_once(self):
        fixture = load("model_switch")
        messages, (first, second) = final(fixture), fixture["prompts"]
        host = self.host(fixture)
        turn1, _ = host.prompt("T1", first)
        host.projector.sync(messages[:by_text(messages, "A1") + 2])
        host.writer.turn_completed("completed")
        first_rows = host.rows()
        turn2, _ = host.prompt("T2", second)
        result = host.projector.sync(messages)
        switch = next(m for m in messages if m["type"] == "model-switched")
        self.assertEqual(result.data["model_switches"], [dict(message_id=switch["id"], model=dict(
            id="second-model", providerID="relay"), previous=None)])
        self.assertEqual(host.projector.sync(messages).data["model_switches"], [])
        rows = host.rows()
        self.assertEqual({k: v["turn_id"] for k, v in rows.items() if k in first_rows},
                         {k: v["turn_id"] for k, v in first_rows.items()})
        self.assertEqual({v["turn_id"] for k, v in rows.items() if k not in first_rows}, {turn2})
        self.assertNotIn(switch["id"], rows)
        host.writer.turn_completed("completed")
        self.assertEqual((self.finals(host, turn1), self.finals(host, turn2)), (["A1"], ["A2"]))


    def test_a_call_id_reused_in_a_later_message_gets_its_own_transport_row(self):
        """Some providers number calls per response ("call_0"); a linkage is (message id, tool id)."""
        fixture = load("model_switch")
        messages, (first, second) = final(fixture), fixture["prompts"]
        tool_turn = load("tool_turn")
        call_message = copy.deepcopy(final(tool_turn)[1])
        turns = []
        host = self.host(fixture)
        for index, (prompt, user) in enumerate(((first, messages[0]), (second, messages[4]))):
            turn, _ = host.prompt(user["text"], prompt)
            reply = dict(call_message, id=f"msg_constructed_call_{index}")
            reply["content"] = [dict(reply["content"][2], id="call_0", state=dict(
                reply["content"][2]["state"], input={"camera": f"cam-{index}"}))]
            listed = [m for m in messages if m["type"] != "assistant" and m["type"] != "idle"][:index * 2 + 1] + [reply]
            host.mcp([reply])
            host.projector.sync(listed)
            host.writer.turn_completed("completed")
            turns.append((turn, reply["id"]))
        calls = sorted(host.of_type("dynamicToolCall"), key=lambda row: row["arguments"]["camera"])
        self.assertEqual([(row["native_id"], row["message_id"], row["turn_id"]) for row in calls],
                         [("call_0", message_id, turn) for turn, message_id in turns])


class Users(Base):
    def test_unbound_submission_defers_its_native_message(self):
        """The reconcile thread may sync between the prompt request's answer and ``bind_submission``: the native
        user message must not become a second row."""
        fixture = load("tool_turn")
        messages, prompt = final(fixture), fixture["prompts"][0]
        host = self.host(fixture)
        _turn, item = host.prompt("PROMPT")                                   # request in flight
        self.assertEqual(host.projector.submissions(), (None, 1))
        result = host.projector.sync(messages[:1])
        self.assertEqual((result.data["deferred_users"], result.data["boundary"], result.data["pending_submissions"]),
                         ([prompt], None, 1))
        self.assertEqual(verdict(messages[:1], result.data["boundary"]).reason, "user_message_pending")
        self.assertTrue(host.projector.bind_submission(item, prompt))
        result = host.projector.sync(messages)
        self.assertEqual((result.data["deferred_users"], result.data["boundary"]), ([], prompt))
        self.assertEqual([(u["id"], u["native_id"]) for u in host.of_type("userMessage")], [(item, prompt)])
        self.assertFalse(host.projector.bind_submission("item-missing", prompt))
        with self.assertRaises(ValueError):
            host.projector.bind_submission(item, "")

    def test_withdrawn_submission_and_native_users_without_a_host_row(self):
        fixture = load("tool_turn")
        host = self.host(fixture)
        _turn, item = host.prompt("PROMPT")
        self.assertTrue(host.projector.withdraw_submission(item))           # the request got a 4xx
        self.assertEqual(host.projector.submissions(), (None, 0))
        host.projector.sync(final(fixture))                                  # e.g. a resumed history
        native = host.rows()[fixture["prompts"][0]]
        self.assertEqual((native["type"], native["source"], native["synthetic"], native["native_id"],
                          native["content"][0]["text"]),
                         ("userMessage", "native", False, fixture["prompts"][0], "Take one observation then answer."))
        self.assertFalse(host.projector.withdraw_submission("item-missing"))

    def test_steer_rows_get_their_ids_and_the_queued_steer_is_the_boundary(self):
        fixture = load("steer")
        prompt, steer = fixture["prompts"]
        host = self.host(fixture)
        turn, prompt_row = host.prompt("P1 wait then answer", prompt)
        steer_row = host.steer("STEER-1", steer)
        captured = snapshot(fixture, "steer_in_inbox")
        result = host.projector.sync(captured["messages"])
        self.assertEqual((result.data["boundary"], result.data["pending_submissions"]), (steer, 0))
        self.assertEqual(verdict(captured["messages"], result.data["boundary"], inbox=captured["inbox"]).reason,
                         "queued")
        host.mcp(final(fixture))
        result = host.projector.sync(final(fixture))
        rows = host.rows()
        self.assertEqual((rows[prompt_row]["native_id"], rows[steer_row]["native_id"], rows[steer_row]["steer"]),
                         (prompt, steer, True))
        self.assertEqual(len(host.of_type("userMessage")), 2)
        host.writer.turn_completed(verdict(final(fixture), result.data["boundary"]).status)
        self.assertEqual(self.finals(host, turn), ["ANSWER-AFTER-STEER"])

    def test_late_steer_phases(self):
        fixture = load("steer_late")
        prompt, steer = fixture["prompts"]
        host = self.host(fixture)
        turn, _ = host.prompt("P1", prompt)
        host.steer("LATE-STEER", steer)
        host.projector.sync(upto(final(fixture), "FIRST-FINAL"))
        host.projector.sync(final(fixture))
        host.writer.turn_completed("completed")
        self.assertEqual({row["text"]: row["phase"] for row in host.of_type("agentMessage")},
                         {"FIRST-FINAL": "commentary", "SECOND-FINAL": "final_answer"})
        self.assertEqual(self.finals(host, turn), ["SECOND-FINAL"])


class NativeTools(Base):
    def test_write_edit_shell_glob_read_grep(self):
        fixture, host, turn, _item, result = self.run_single_turn("native_tools")
        messages = final(fixture)
        rows = {row["native_id"]: row for row in host.rows().values() if row["type"] != "userMessage"
                and row["native_id"].startswith("call_")}
        self.assertEqual((rows["call_fake_0_0"]["type"], rows["call_fake_0_0"]["changes"], rows["call_fake_0_0"]["status"]),
                         ("fileChange", [dict(path="notes/a.txt", kind="write", diff=None)], "completed"))
        edit = rows["call_fake_1_0"]
        (change,) = edit["changes"]
        self.assertEqual((edit["type"], change["path"], change["kind"], edit["status"]),
                         ("fileChange", "notes/a.txt", "update", "completed"))
        self.assertIn("+gamma", change["diff"])
        failed_edit = rows["call_fake_2_0"]
        self.assertEqual((failed_edit["status"], failed_edit["error"]["type"]), ("failed", "tool.execution"))
        self.assertTrue(failed_edit["error"]["message"].startswith("Could not find oldString"))
        shell = rows["call_fake_3_0"]
        self.assertEqual((shell["type"], shell["command"], shell["exitCode"], shell["status"]),
                         ("commandExecution", "cat notes/a.txt; echo done-err >&2; exit 3", 3, "completed"))
        self.assertEqual(shell["aggregatedOutput"], "alpha\ngamma\ndone-err\n\nExit code: 3")
        for call, tool, needle in (("call_fake_4_0", "glob", "notes/a.txt"), ("call_fake_5_0", "read", "2: gamma"),
                                   ("call_fake_6_0", "grep", "Line 2: gamma")):
            row = rows[call]
            self.assertEqual((row["type"], row["tool"], row["status"]), ("mcpToolCall", tool, "completed"))
            self.assertIn(needle, row["result"])
        missing = rows["call_fake_7_0"]
        self.assertEqual((missing["tool"], missing["status"], missing["error"]),
                         ("read", "failed", {"message": "File not found: notes/missing.txt", "type": "tool.execution"}))
        self.assertEqual(host.rows()[f"{messages[1]['id']}:reasoning:0"]["summary"], ["plan the edits"])
        self.assertTrue(all(row["turn_id"] == turn for row in host.rows().values()))
        self.assertEqual(self.finals(host, turn), ["NATIVE-DONE"])
        self.assertEqual(result.final_text, "NATIVE-DONE")

    def test_patch_changes_from_metadata_and_from_the_input(self):
        fixture, host, *_ = self.run_single_turn("patch_tool")
        message = final(fixture)[1]
        row = host.rows()[f"{message['id']}:call_fake_0_0"]
        self.assertEqual([(c["path"], c["kind"]) for c in row["changes"]],
                         [("p/new.txt", "add"), ("p/old.txt", "update"), ("p/gone.txt", "delete")])
        self.assertIn("+new line", row["changes"][1]["diff"])
        running = copy.deepcopy(message)
        tool = running["content"][0]
        tool["state"] = dict(status="running", input=tool["state"]["input"], metadata={})
        running["id"] = "msg_constructed_running_patch"
        running.pop("finish")
        host.projector.sync([running])
        row = host.rows()["msg_constructed_running_patch:call_fake_0_0"]
        self.assertEqual((row["status"], row["changes"]),
                         ("inProgress", [dict(path="p/new.txt", kind="add", diff=None),
                                         dict(path="p/old.txt", kind="update", diff=None),
                                         dict(path="p/gone.txt", kind="delete", diff=None)]))

    def test_running_shell_then_abandoned_row_stays_final(self):
        fixture = load("shell_running")
        host = self.host(fixture)
        host.prompt("P1 slow shell", fixture["prompts"][0])
        captured = snapshot(fixture, "shell_running")
        host.projector.sync(captured["messages"])
        row_id = f"{captured['messages'][1]['id']}:call_fake_0_0"
        row = host.rows()[row_id]
        self.assertEqual((row["status"], row["aggregatedOutput"], row["exitCode"]), ("inProgress", None, None))
        host.projector.sync(final(fixture))
        row = host.rows()[row_id]
        self.assertEqual((row["status"], row["aggregatedOutput"], row["exitCode"]), ("completed", "slept\n", 0))
        # a dead step (server gone, turn failed by the guard): abandon, and a later native update cannot reopen it
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, IDENTITY)
            try:
                other = self.host(fixture, store)
                other.prompt("P1 slow shell", fixture["prompts"][0])
                other.projector.sync(captured["messages"])
                self.assertEqual(other.projector.abandon_open_items(), [row_id])
                other.projector.sync(final(fixture))
                abandoned = other.rows()[row_id]
                self.assertEqual((abandoned["status"], abandoned["interrupted"], abandoned["error"]),
                                 ("failed", True, {"message": INTERRUPTED_TOOL_ERROR}))
                self.assertEqual(other.projector.abandon_open_items(), [])
            finally:
                store.close()


class AbortsAndErrors(Base):
    def test_abort_during_a_registry_call(self):
        fixture = load("interrupt_tool")
        messages, (first, second) = final(fixture), fixture["prompts"]
        turn1_messages = messages[:by_text(messages, "P1 long wait") + 3]
        host = self.host(fixture)
        host.prompt("P1 long wait", first)
        host.mcp(turn1_messages)                                     # the MCP request was in flight
        host.projector.sync(turn1_messages)
        (call,) = host.of_type("dynamicToolCall")
        self.assertEqual((call["native_id"], call["tool"]), ("call_fake_0_0", "wait_s"))
        self.assertEqual((host.of_type("mcpToolCall"), host.of_type("error")), ([], []))   # aborted: no error row
        host.writer.turn_completed("interrupted")
        turn2, _ = host.prompt("P2 after interrupt", second)
        host.projector.sync(messages)
        host.writer.turn_completed("completed")
        self.assertEqual(self.finals(host, turn2), ["SHOULD-NOT-APPEAR-IF-ABORTED"])

    def test_registry_call_refused_before_mcp_becomes_one_failed_row(self):
        fixture = load("interrupt_tool")
        messages = final(fixture)
        host = self.host(fixture)
        host.prompt("P1 long wait", fixture["prompts"][0])
        for _ in range(2):
            host.projector.sync(messages[:3])
        row = host.rows()[f"{messages[1]['id']}:call_fake_0_0"]
        self.assertEqual((row["type"], row["tool"], row["status"], row["native_id"], row["error"]["type"]),
                         ("mcpToolCall", "ni_rua_wait_s", "failed", "call_fake_0_0", "aborted"))
        self.assertEqual(len(host.of_type("mcpToolCall")), 1)

    def test_abort_during_the_stream_writes_no_rows(self):
        fixture = load("interrupt_stream")
        host = self.host(fixture)
        host.prompt("P1", fixture["prompts"][0])
        host.projector.sync(final(fixture))
        self.assertEqual([row["type"] for row in host.rows().values()], ["userMessage"])

    def test_step_errors_carry_no_provider_text(self):
        for name, code in (("error_400", "opencode_step_failed:provider.invalid-request:400"),
                           ("error_500_exhausted", "opencode_step_failed:provider.internal:500")):
            with self.subTest(name):
                fixture, host, turn, *_ = self.run_single_turn(name)
                message = final(fixture)[1]
                self.assertEqual([(e["id"], e["error"]) for e in host.of_type("error")],
                                 [(f"{message['id']}:error", {"message": code})])
                self.assertEqual(host.of_type("agentMessage"), [])
                for word in ("ni_rua_model_request_failed", "boom", "invalid_request", "api_error"):
                    self.assertNotIn(word, json.dumps(host.rows()))
                self.assertEqual(self.finals(host, turn), [])

    def test_retried_500_rows(self):
        fixture, host, turn, _item, result = self.run_single_turn("error_500_retried")
        message = final(fixture)[1]
        self.assertEqual([(e["id"], e["error"]) for e in host.of_type("error")],
                         [(f"{message['id']}:retry:2", {"message": "opencode_retry:2:provider.internal:500"})])
        self.assertEqual((self.finals(host, turn), result.final_text), (["RECOVERED-500"], "RECOVERED-500"))
        self.assertNotIn("boom", json.dumps(host.rows()))

    def test_stream_error_and_length_continuations(self):
        for name, final_text, partial, synthetic_start in (
                ("stream_error_continuation", "RECOVERED-SE", "partial text her", "The previous response was"),
                ("length_continuation", "done", "cut", "Your last response hit the output token limit")):
            with self.subTest(name):
                fixture, host, turn, *_ = self.run_single_turn(name)
                synthetic = [row for row in host.of_type("userMessage") if row.get("source") == "native"]
                self.assertTrue(synthetic and all(row["synthetic"] for row in synthetic))
                self.assertTrue(all(row["content"][0]["text"].startswith(synthetic_start) for row in synthetic))
                partials = [row for row in host.of_type("agentMessage") if row["text"] == partial]
                self.assertEqual(len(partials), len(synthetic))
                self.assertTrue(all(row["phase"] == "commentary" for row in partials))
                self.assertEqual(self.finals(host, turn), [final_text])
                self.assertNotIn("overloaded", json.dumps(host.rows()))
        # each failed step keeps its retry record (attempt n schedules step n+1); a step error is not a turn failure
        errors = sorted(row["error"]["message"] for row in Host(self.store, load("stream_error_continuation")
                                                                ["session_id"]).of_type("error"))
        self.assertEqual(errors, sorted(["opencode_step_failed:provider.internal:200"] * 8
                                        + [f"opencode_retry:{n}:provider.internal:200" for n in range(2, 10)]))


class PermissionRejected(Base):
    def test_rejected_permission_rows_and_the_next_turn(self):
        fixture = load("permission_rejected")
        first, second = fixture["prompts"]
        host = self.host(fixture)
        turn1, _ = host.prompt("P1 read env", first)
        asked = snapshot(fixture, "permission_asked")["messages"]
        host.projector.sync(asked)
        row_id = f"{asked[1]['id']}:call_fake_0_0"
        self.assertEqual(host.rows()[row_id]["status"], "inProgress")
        rejected = snapshot(fixture, "after_reject")["messages"]
        host.projector.sync(rejected)
        row = host.rows()[row_id]
        self.assertEqual((row["type"], row["tool"], row["status"], row["error"]),
                         ("mcpToolCall", "read", "failed",
                          {"message": "The user declined this tool call", "type": "aborted"}))
        self.assertEqual(host.of_type("error"), [])                         # an aborted step is no error row
        result = verdict(rejected, host.projector.submissions()[0], permission_rejected=True)
        host.writer.turn_completed(result.status, error=result.error)
        self.assertEqual((result.status, self.finals(host, turn1)), ("failed", []))
        turn2, _ = host.prompt("P2 after reject", second)
        host.projector.sync(final(fixture))
        host.writer.turn_completed("completed")
        self.assertEqual(self.finals(host, turn2), ["AFTER-REJECT-ANSWER"])
        self.assertEqual(host.rows()[row_id]["turn_id"], turn1)


class Timeline(Base):
    def store_order(self, host):
        return [row["id"] for row in self.store.states(self.store.TRANSCRIPT_KIND)
                if row.get("thread_id") == host.thread_id]

    def test_rows_are_written_in_api_order_not_id_order(self):
        fixture = load("subagent_background")
        messages = final(fixture)
        host = self.host(fixture)
        host.prompt("P1 spawn background", fixture["prompts"][0])
        host.projector.sync(messages)
        order = self.store_order(host)
        report = next(m["id"] for m in messages if m["type"] == "synthetic")
        before = f"{next(m['id'] for m in messages if m['type'] == 'assistant' and m['content'][0]['type'] == 'text')}:text:0"
        self.assertLess(report, before.split(":")[0])                    # the report's id is the smaller one ...
        self.assertLess(order.index(before), order.index(report))       # ... and its row still comes after
        self.assertEqual([entry["id"] for entry in self.store.review_page(limit=50)["items"]], order)

    def test_non_atomic_pages_converge(self):
        fixture = load("tool_turn")
        messages = final(fixture)
        host = self.host(fixture)
        turn, _ = host.prompt("PROMPT", fixture["prompts"][0])
        host.mcp(messages)
        stale = copy.deepcopy(messages[:3])                              # page 1, read while the answer streamed
        stale[2].pop("finish")
        stale[2]["time"].pop("completed")
        stale[2]["content"][0]["text"] = "FINAL-ANS"
        host.projector.sync(stale + messages[2:])                        # page 2 repeats the message, fresher
        answer = host.rows()[f"{messages[2]['id']}:text:0"]
        self.assertEqual((answer["text"], answer["status"]), ("FINAL-ANSWER-1", "completed"))
        host.projector.sync(stale)                                       # an older read never reopens a step
        answer = host.rows()[f"{messages[2]['id']}:text:0"]
        self.assertEqual((answer["text"], answer["status"]), ("FINAL-ANSWER-1", "completed"))
        host.projector.sync(messages)
        host.writer.turn_completed("completed")
        self.assertEqual(self.finals(host, turn), ["FINAL-ANSWER-1"])
        seq = self.writes()
        host.projector.sync(messages[:2])                                # a shorter (earlier) read deletes nothing
        host.projector.sync(messages)
        self.assertEqual(self.writes(), seq)


class Subagents(Base):
    def test_foreground_subagent_rows_and_links(self):
        fixture = load("subagent_foreground")
        (child_id, child), = fixture["children"].items()
        host = self.host(fixture)
        turn, _ = host.prompt("P1 spawn foreground", fixture["prompts"][0])
        running = snapshot(fixture, "child_running")["messages"]
        self.assertEqual(host.projector.sync(running).data["subagents"], [])   # no sessionID while it runs
        result = host.projector.sync(final(fixture))
        item_id = f"{final(fixture)[1]['id']}:call_fake_0_0"
        self.assertEqual(result.data["subagents"], [dict(session_id=child_id, item_id=item_id,
                                                         message_id=final(fixture)[1]["id"], background=False)])
        row = host.rows()[item_id]
        self.assertEqual((row["type"], row["tool"], row["status"], row["arguments"]["agent"]),
                         ("mcpToolCall", "subagent", "completed", "general"))
        self.assertIn("FG-CHILD-RESULT", row["result"])
        sub = Host(self.store, child_id)
        sub.mcp(child["messages"])
        sub.projector.sync(child["messages"])
        first, = sub.of_type("userMessage")
        self.assertEqual((first["source"], first["synthetic"]), ("native", False))
        (call,) = sub.of_type("dynamicToolCall")
        self.assertEqual(call["native_id"], "call_fake_1_0")
        self.assertEqual([r["text"] for r in sub.of_type("agentMessage")], ["FG-CHILD-RESULT"])
        host.writer.turn_completed("completed")
        self.assertEqual(self.finals(host, turn), ["MAIN-AFTER-FG-CHILD"])

    def test_background_subagent_report_is_a_synthetic_row_of_the_held_turn(self):
        fixture = load("subagent_background")
        (child_id, _child), = fixture["children"].items()
        host = self.host(fixture)
        turn, _ = host.prompt("P1 spawn background", fixture["prompts"][0])
        first = snapshot(fixture, "first_idle_child_running")["messages"]
        links = host.projector.sync(first).data["subagents"]
        self.assertEqual([(link["session_id"], link["background"]) for link in links], [(child_id, True)])
        host.projector.sync(final(fixture))
        report, = [row for row in host.of_type("userMessage") if row.get("source") == "native"]
        self.assertTrue(report["synthetic"])
        self.assertIn("CHILD-RESULT", report["content"][0]["text"])
        host.writer.turn_completed("completed")
        self.assertEqual({row["text"]: row["phase"] for row in host.of_type("agentMessage")},
                         {"MAIN-FINAL-BEFORE-CHILD-DONE": "commentary", "MAIN-AFTER-CHILD-REPORT": "final_answer"})
        self.assertEqual(subagent_links(final(fixture)), links)


class Compaction(Base):
    def test_compaction_row_with_its_summary(self):
        fixture, host, turn, *_ = self.run_single_turn("compaction")
        message = next(m for m in final(fixture) if m["type"] == "compaction")
        row = host.rows()[message["id"]]
        self.assertEqual((row["type"], row["status"], row["reason"], row["native_id"], row["turn_id"]),
                         ("contextCompaction", "completed", "auto", message["id"], turn))
        self.assertIn("MARKER-SUMMARY", row["summary"])
        self.assertFalse(any("MARKER-SUMMARY" in r["text"] for r in host.of_type("agentMessage")))
        self.assertEqual(self.finals(host, turn), ["FINAL-AFTER-COMPACTION"])
        self.assertEqual(host.projector.sync(final(fixture)).data["compactions"], [])       # reported once
        # a resumed process adopts the history without signalling the old compaction again
        self.assertEqual(Host(self.store, host.thread_id).projector.sync(final(fixture)).data["compactions"], [])

    def test_failed_compaction(self):
        fixture, host, turn, _item, result = self.run_single_turn("compaction_failed")
        message = next(m for m in final(fixture) if m["type"] == "compaction")
        row = host.rows()[message["id"]]
        self.assertEqual((row["status"], row["error"], row.get("summary")),
                         ("failed", {"message": "opencode_compaction_failed:compaction.failed"}, None))
        self.assertEqual((result.status, self.finals(host, turn)), ("failed", []))
        self.assertNotIn("template", json.dumps(host.rows()))


class Crash(Base):
    def test_crash_leaves_the_transport_row_to_its_owner(self):
        fixture = load("crash_without_idle")
        first, second, third = fixture["prompts"]
        host = self.host(fixture)
        host.prompt("T1", first)
        host.projector.sync(fixture["snapshots"][0]["messages"][:3])
        host.writer.turn_completed("completed")
        host.prompt("T2 long tool", second)
        running = snapshot(fixture, "tool_running")["messages"]
        host.mcp(running)                                            # ni_rua_wait_s reached MCP, then SIGKILL
        host.projector.sync(running)
        before = self.store.states(self.store.TRANSCRIPT_KIND)
        host.projector.sync(snapshot(fixture, "after_crash")["messages"])
        second_process = Host(self.store, host.thread_id)            # a new process adopts the history
        result = second_process.projector.sync(snapshot(fixture, "after_crash")["messages"])
        self.assertEqual(self.store.states(self.store.TRANSCRIPT_KIND), before)
        self.assertEqual((result.data["boundary"], result.data["deferred_users"]), (None, []))
        host.writer.turn_completed("failed", error="opencode_turn_incomplete")
        turn3, _ = second_process.prompt("T3 after crash", third)
        second_process.projector.sync(final(fixture))
        second_process.writer.turn_completed("completed")
        self.assertEqual(self.finals(second_process, turn3), ["A3-after-restart"])
        (wait,) = second_process.of_type("dynamicToolCall")
        self.assertEqual((wait["status"], wait["native_id"]), ("inProgress", "call_fake_1_0"))   # transport-owned


class Events(Base):
    @staticmethod
    def normal(rows):
        """Rows keyed by their native-derived id; host rows (random ``item-``/``call-`` ids) by type + native id;
        turn ids and measured durations differ between the two stores."""
        def key(row):
            return (row["type"], row.get("native_id")) if row["id"].startswith(("item-", "call-")) else row["id"]
        return {key(row): {k: v for k, v in row.items() if k not in ("id", "turn_id", "durationMs")}
                for row in rows.values()}

    def replay_then_sync(self, name, *, frames=None):
        """Rows from the fixture's frames followed by a sync of the final list, and rows from the sync alone (fresh
        store); both with the same host prompt/steer rows and transport rows."""
        fixture = load(name)
        results = []
        for replay in (True, False):
            directory = tempfile.TemporaryDirectory()
            store = Store(directory.name, IDENTITY)
            try:
                host = self.host(fixture, store)
                host.prompt("PROMPT", fixture["prompts"][0])
                for prompt in fixture["prompts"][1:]:
                    host.steer("STEER", prompt)
                host.mcp(final(fixture))
                if replay:
                    for frame in (fixture["events"] if frames is None else frames(fixture["events"])):
                        host.projector.project_event(frame)
                host.projector.sync(final(fixture))
                results.append(self.normal(host.rows()))
            finally:
                store.close()
                directory.cleanup()
        return results

    def test_event_replay_matches_snapshot_projection(self):
        for name in ("tool_turn", "tool_turn_responses", "native_tools", "steer", "subagent_foreground",
                     "compaction", "length_continuation"):
            with self.subTest(name):
                live, synced = self.replay_then_sync(name)
                self.assertEqual(live, synced)

    def test_event_gap_then_snapshot_reconciles(self):
        live, synced = self.replay_then_sync("steer", frames=lambda frames: frames[:len(frames) // 2])
        self.assertEqual(live, synced)

    def test_retry_frames_add_one_row_per_attempt(self):
        fixture = load("error_500_exhausted")
        host = self.host(fixture)
        host.prompt("P1", fixture["prompts"][0])
        kinds = [host.projector.project_event(frame).kind for frame in fixture["events"]]
        self.assertEqual(kinds.count("retry"), 10)
        message_id = final(fixture)[1]["id"]
        host.projector.sync(final(fixture))
        errors = sorted(row["id"] for row in host.of_type("error"))
        self.assertEqual(errors, sorted([f"{message_id}:error"] + [f"{message_id}:retry:{n}" for n in range(2, 12)]))
        self.assertNotIn("boom", json.dumps(host.rows()))

    def test_live_text_rows_follow_the_stream(self):
        fixture = load("tool_turn")
        host = self.host(fixture)
        host.prompt("PROMPT", fixture["prompts"][0])
        answer = final(fixture)[2]["id"]
        frames = fixture["events"]
        delta = next(i for i, f in enumerate(frames) if f["type"] == "session.text.delta"
                     and f["data"]["assistantMessageID"] == answer)
        for frame in frames[:delta + 1]:
            host.projector.project_event(frame)
        row = host.rows()[f"{answer}:text:0"]
        self.assertEqual((row["text"], row["status"]), ("FINAL-ANSWER-1", "inProgress"))
        for frame in frames[delta + 1:]:
            host.projector.project_event(frame)
        self.assertEqual(host.rows()[f"{answer}:text:0"]["status"], "completed")

    def test_frames_the_projector_cannot_place(self):
        fixture = load("tool_turn")
        host = self.host(fixture)
        session = fixture["session_id"]
        project = host.projector.project_event
        self.assertEqual(project(dict(type="session.text.delta", data=dict(
            sessionID=session, assistantMessageID="msg_unknown", ordinal=0, delta="z"))).data["applied"], False)
        self.assertNotIn("msg_unknown:text:0", host.rows())
        self.assertTrue(project(dict(type="session.text.ended", data=dict(
            sessionID=session, assistantMessageID="msg_unknown", ordinal=0, text="whole"))).data["applied"])
        self.assertEqual(host.rows()["msg_unknown:text:0"]["text"], "whole")
        self.assertEqual(project(dict(type="session.tool.success", data=dict(
            sessionID=session, assistantMessageID="msg_unknown", id="call_x", content=[]))).kind, "deferred")
        self.assertEqual(project(dict(type="session.text.delta", data=dict(sessionID="ses_other"))).kind, "ignored")
        self.assertEqual(project(dict(type="server.connected", data={})).kind, "other")
        self.assertEqual(project("not a frame").kind, "other")
        self.assertEqual(project(dict(type="session.compaction.ended", data=dict(sessionID=session))).kind, "other")
        phases = [project(frame).data["phase"] for frame in fixture["events"]
                  if frame["type"].startswith("session.execution.")]
        self.assertEqual(phases, ["started", "succeeded"])

    def test_compaction_frames_are_the_early_signal(self):
        for name, status in (("compaction", "completed"), ("compaction_failed", "failed")):
            with self.subTest(name):
                fixture = load(name)
                message = next(m for m in final(fixture) if m["type"] == "compaction")
                host = self.host(fixture)
                host.prompt("PROMPT", fixture["prompts"][0])
                started = [f for f in fixture["events"] if f["type"] == "session.compaction.started"]
                self.assertEqual(compaction_message_id(started[0]["id"]), message["id"])
                projected = [host.projector.project_event(f) for f in fixture["events"]
                             if f["type"].startswith("session.compaction.")]
                signals = [p for p in projected if p.kind == "compaction"]
                self.assertEqual(signals[0].data, dict(message_id=message["id"], phase="started", reason="auto",
                                                       new=True))
                self.assertEqual(host.rows()[message["id"]]["status"], status)
                self.assertEqual(host.projector.sync(final(fixture)).data["compactions"], [])   # one signal
                self.assertEqual(host.rows()[message["id"]]["status"], status)
        self.assertIsNone(compaction_message_id("msg_x"))

    def test_subagent_progress_frame_links_a_running_child(self):
        fixture = load("subagent_foreground")
        (child_id,) = fixture["children"]
        host = self.host(fixture)
        host.prompt("P1 spawn foreground", fixture["prompts"][0])
        links = [p.data for p in map(host.projector.project_event, fixture["events"]) if p.kind == "subagent"]
        item_id = f"{final(fixture)[1]['id']}:call_fake_0_0"
        self.assertEqual(links[0], dict(session_id=child_id, item_id=item_id, message_id=final(fixture)[1]["id"]))
        self.assertEqual(host.projector.project_event(dict(type="session.created", data=dict(
            sessionID=child_id, parentID=fixture["session_id"]))).kind, "ignored")      # the session's business


LEGACY = FIXTURES_ROOT / "opencode" / "legacy_1_19_5" / "archive.json"


def legacy_archives():
    return json.loads(LEGACY.read_text())["archives"]


def open_legacy_archive(directory, name):
    """Lay one frozen OpenCode 1.19.5 archive (``legacy_1_19_5/archive.json``) into a fresh run directory as the 1.19.5
    run left it: transcript rows (store order), ``agent.*`` events, ``runtime/agent`` and ``agent-raw.jsonl``.
    Returns ``(store, manifest, archive)``; nothing here parses native 1.19.5 messages."""
    archive = legacy_archives()[name]
    store = Store(directory, IDENTITY)
    for row in archive["rows"]:
        store.set_transcript_item(row["id"], row)
    for event in archive["events"]:
        store.event(event["type"], event["payload"] or {}, event["correlation"], source="agent_backend")
    if archive.get("runtime_agent"):
        store.set_state("runtime", "agent", archive["runtime_agent"])
    for record in archive.get("agent_raw") or ():
        store.raw_agent(record["direction"], record["message"], channel=record["channel"])
    manifest = archive.get("manifest") or dict(thread_id=archive["thread_id"], agent_backend=dict(kind="opencode"))
    return store, manifest, archive


class LegacyArchive1195(unittest.TestCase):
    """Design 11 section 4: a run archived by the 1.19.5 backend keeps opening after the 2.x port. Replay and the
    viewer read neutral data only (transcript rows, ``runtime/agent``, ``manifest.agent_backend``, ``agent-raw.jsonl``),
    so the checks go through those readers: ``Store.completed_final_texts`` / ``review_page``, ``load_native_evidence``,
    ``read_agent_identity`` and ``OpenCodeSession.enter_replay`` (constructing the session starts nothing). The rows
    of every 1.19.5 projector scenario (formerly checked against the 1.19.5 native shapes) are frozen there with the
    finals and terminals the 1.19.5 code decided."""

    def setUp(self):
        self.dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.dir.cleanup)

    def open(self, name, directory=None):
        store, manifest, archive = open_legacy_archive(directory or self.dir.name, name)
        self.addCleanup(store.close)
        return store, manifest, archive

    def test_every_archive_opens_through_the_neutral_readers(self):
        for name, frozen in legacy_archives().items():
            with self.subTest(name):
                directory = self.enterContext(tempfile.TemporaryDirectory())
                store, manifest, archive = self.open(name, directory)
                thread, turns = archive["thread_id"], archive["turns"]
                self.assertEqual([store.completed_final_texts(thread, turn) for turn in turns],
                                 frozen.get("finals") or [store.completed_final_texts(thread, turn) for turn in turns])
                evidence = load_native_evidence(directory, manifest, store, require_texts=True, profile="report")
                expected = [dict(id=turn, status=status) for turn, status in zip(turns, archive["statuses"]) if status]
                self.assertEqual(evidence.turn_terminals, expected)
                self.assertEqual([entry["id"] for entry in store.review_page(limit=500)["items"]],
                                 [row["id"] for row in archive["rows"]])
                rows = store.states(store.TRANSCRIPT_KIND)
                self.assertTrue(all(row["backend"] == "opencode" and row["thread_id"] for row in rows))
                for noise in ("iVBOR", "SIG-SECRET", "provider text", "denied by", "overloaded"):
                    self.assertNotIn(noise, json.dumps(rows))

    def test_row_shapes_of_the_1_19_5_projector(self):
        archives = legacy_archives()
        rows = {name: {row["id"]: row for row in archive["rows"]} for name, archive in archives.items()}
        basic = list(rows["basic_anthropic_two_turns"].values())
        self.assertEqual(len([r for r in basic if r["type"] == "dynamicToolCall"]), 5)
        self.assertTrue(all(r.get("native_id", "").startswith("call_fake_") for r in basic
                            if r["type"] == "dynamicToolCall"))
        bash, = [r for r in basic if r["type"] == "commandExecution"]
        self.assertEqual((bash["command"], bash["exitCode"], bash["id"][:4]), ("env | sort", 0, "prt_"))
        failed, = [r for r in basic if r["type"] == "dynamicToolCall" and r["status"] == "failed"]
        self.assertEqual((failed["tool"], failed["native_id"], failed["success"]), ("fail", "call_fake_2_0", False))
        native = {r.get("tool"): r for r in rows["native_tools"].values() if r["type"] == "mcpToolCall"}
        self.assertEqual(set(native), {"grep", "glob", "read"})
        steer = rows["steer_during_final"].values()
        self.assertEqual({r["text"]: r["phase"] for r in steer if r["type"] == "agentMessage"},
                         {"FIRST-FINAL": "commentary", "SECOND-FINAL": "final_answer"})
        error, = [r for r in rows["error_400"].values() if r["type"] == "error"]
        self.assertEqual(error["error"], {"message": "opencode_turn_failed:APIError:400"})
        slow, = [r for r in rows["restart"].values() if r["type"] == "dynamicToolCall" and r["tool"] == "slow"]
        self.assertEqual(slow["status"], "inProgress")
        parts = rows["constructed_parts"]
        self.assertEqual(parts["prt_1"]["changes"], [dict(path="/runs/x/actor/a.py", kind="update", diff="-a\n+b")])
        self.assertEqual(parts["prt_3"]["changes"], [dict(path="/runs/x/actor/c.txt", kind="add", diff="+x")])
        self.assertEqual((parts["prt_4"]["type"], parts["prt_4"]["summary"]), ("reasoning", ["think first"]))
        self.assertEqual(parts["prt_5"]["error"], {"message": "opencode_retry:2:APIError:529"})
        aborted = [r for r in rows["abort_tool"].values() if r["type"] == "dynamicToolCall"]
        self.assertEqual([(r["tool"], r["native_id"]) for r in aborted], [("slow", "call_fake_10_0")])

    def test_smoke_run_final_answers_terminals_and_prompts(self):
        store, manifest, archive = self.open("smoke_run")
        thread, turns = archive["thread_id"], archive["turns"]
        self.assertEqual([store.completed_final_texts(thread, turn) for turn in turns],
                         [["FINAL"], ["FINAL-STEER"], [], ["FINAL-2"], []])
        evidence = load_native_evidence(self.dir.name, manifest, store, require_texts=True)
        self.assertTrue(evidence.log_present)
        self.assertEqual(evidence.turn_terminals, [dict(id=turns[0], status="completed"),
                                                   dict(id=turns[1], status="completed"),
                                                   dict(id=turns[2], status="interrupted"),
                                                   dict(id=turns[3], status="completed")])
        self.assertEqual(evidence.submitted_prompts, [["Look at the scene, then answer."], ["GATE-S: wait for the gate."],
                                                      ["GATE-A: wait for the gate."], ["PLAIN: answer directly."],
                                                      ["BASH: wait in the shell."]])
        self.assertEqual([request["turn_id"] for request in evidence.interrupt_requests], [turns[2]])
        call, = [row for row in archive["rows"] if row["type"] == "dynamicToolCall"]
        self.assertEqual(evidence.tool_item_ids, {call["id"]})
        self.assertEqual(sorted(evidence.completed_turn_texts), ["FINAL", "FINAL-2", "FINAL-STEER", "late answer"])

    def test_smoke_run_rows_keep_their_1_19_5_ids_and_fields(self):
        store, _manifest, archive = self.open("smoke_run")
        rows = {row["id"]: row for row in store.states(store.TRANSCRIPT_KIND)}
        texts = {row["text"]: row for row in rows.values() if row["type"] == "agentMessage"}
        self.assertTrue(all(row["id"].startswith("prt_") for row in texts.values()))   # 1.19.5 part ids, opaque
        self.assertEqual({text: row["phase"] for text, row in texts.items()},
                         {"FINAL": "final_answer", "late answer": "commentary", "FINAL-STEER": "final_answer",
                          "FINAL-2": "final_answer"})
        steer, = [row for row in rows.values() if row.get("steer")]
        self.assertEqual((steer["content"][0]["text"], steer["native_id"][:4]), ("STEER-NOW: answer the steer.", "msg_"))
        bash, = [row for row in rows.values() if row["type"] == "commandExecution"]
        self.assertEqual((bash["status"], bash["interrupted"], bash["command"]), ("failed", True, "sleep 600"))
        error, = [row for row in rows.values() if row["type"] == "error"]
        self.assertTrue(error["id"].endswith(":error"))
        self.assertEqual(error["error"], {"message": "opencode_turn_failed:APIError:400"})

    def test_smoke_run_identity_manifest_and_replay(self):
        from ni_rua.core.config import Config
        from ni_rua.cortex.agents.base import read_agent_identity
        from ni_rua.cortex.agents.opencode.session import OpenCodeSession
        store, manifest, archive = self.open("smoke_run")
        identity = read_agent_identity(store)
        self.assertEqual((identity["backend"], identity["cli_version"], identity["thread_id"]),
                         ("opencode", "1.19.5", archive["thread_id"]))
        block = manifest["agent_backend"]
        settings = block.get("native_settings") or {}
        self.assertEqual(((block.get("cli") or {}).get("version"), len(settings.get("flags") or {}),
                          settings.get("compaction_prune"), settings.get("title_agent"),
                          sorted((block.get("native_effort") or {}).get("provider_options") or {})),
                         ("1.19.5", 9, False, "disabled", ["effort", "thinking"]))
        viewer = OpenCodeSession(self.dir.name, store, None, config=Config())
        viewer.enter_replay(archive["thread_id"])
        self.assertEqual((viewer.state, viewer.thread_id, viewer.owned_processes()),
                         ("replay", archive["thread_id"], []))
        viewer.close()
        self.assertEqual(viewer.state, "closed")
        self.assertEqual(len(store.states(store.TRANSCRIPT_KIND)), len(archive["rows"]))   # replay wrote nothing

    def test_compaction_rows_from_the_1_19_5_projector(self):
        store, _manifest, archive = self.open("compaction_rows")
        thread, turns = archive["thread_id"], archive["turns"]
        self.assertEqual([store.completed_final_texts(thread, turn) for turn in turns],
                         [["TURN-1-DONE"], ["TURN-2-DONE"], ["TURN-3-DONE"]])
        rows = store.states(store.TRANSCRIPT_KIND)
        compaction, = [row for row in rows if row["type"] == "contextCompaction"]
        summary, = [row for row in rows if row["type"] == "agentMessage" and row["text"].startswith("SUMMARY:")]
        self.assertEqual((summary["parent_item_id"], summary["phase"], compaction["turn_id"]),
                         (compaction["id"], None, turns[2]))
        synthetic, = [row for row in rows if row.get("source") == "native"]
        self.assertTrue(synthetic["synthetic"])

if __name__ == "__main__":
    unittest.main()
