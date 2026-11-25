"""External agent backend in a real Workbench run (design 20 sections 2.2-2.5; review 19 M1, M3, M7, S1 and S2).

``support.external_backend.ExternalRun`` starts the real Workbench, Harness, Store and ``ExternalSession`` (MCP on an
ephemeral loopback port) over the deterministic ARX-X5 and SO-101 robots of ``support.embodiment_golden``;
``FakeMcpClient`` plays the user's agent. No model, network, GPU, agent binary or real arm.
"""

NI_RUA_TEST_TIER = "component"

import asyncio
import json
import os
import re
import signal
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest import mock
from support.external_backend import RUN_MARKER, ExternalRun, host_startup
from support.fake_mcp_client import FakeMcpClient
from ni_rua.apps.evaluation.episode import episode_summary
from ni_rua.apps.evaluation.native_evidence import load_native_evidence
from ni_rua.apps.evaluation.report import _agent_conditions, build_report
from ni_rua.apps.gateway.extensions import agent_settings
from ni_rua.apps.gateway.workbench import DEFAULT_PROMPT
from ni_rua.core.presentation.actor_presentation import metadata_findings
from ni_rua.cortex.context.actor_instructions import (
    EXTERNAL_AGENT_SOURCE_INSTRUCTIONS,
    EXTERNAL_URDF_ON_DEMAND_CLAUSE,
    _local_source_instructions,
)
from ni_rua.cortex.agents.base import AgentCapabilityError, session_turn_blocks_operator
from ni_rua.cortex.agents.external.session import CONSOLE_COMMANDS_UNSUPPORTED, MANIFEST_NULL_FIELDS

UNSUPPORTED = ("prompt", "steer", "reply", "set_model", "set_effort", "set_service_tier", "apply_agent_settings",
               "handoff_run", "retry_model_requests")


def wait_until(predicate, timeout=30.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def briefing(client):
    result = client.call_tool("run_begin", {})
    assert not result.result["isError"], result.result
    instructions, rest = result.result["content"][:2]
    return instructions["text"], json.loads(rest["text"]), result


class BriefingTextTests(unittest.TestCase):
    """The briefing is the host-mode startup instruction text for the same profile, scrubbed, except the source-access
    block (review 23 S1): an external agent is told nothing about its own tools, sandbox or network, and no path."""

    def check(self, embodiment):
        from ni_rua.robot.model.robot_information import robot_information as read_robot_information
        # 外部模式默认 delta 投递，tools/list 与托管 Codex 保持字节一致（末尾多 run_begin / run_finish）；
        # robot_urdf_delivery="file" 默认省略 46k-60k 原始 URDF 并引导按需调 robot_information（零路径泄露）；
        # 显式 robot_urdf_delivery="inline" 时仍与托管 inline 简报严格对齐。
        startup_inline = host_startup(embodiment, {"features": {"tool_catalog_profile": "compact_v1",
                                                                "robot_urdf_delivery": "inline"}})
        host = startup_inline["instructions"]
        robot_id = "arx-x5" if embodiment == "x5" else "so101"
        urdf = read_robot_information(robot_id)["urdf"]
        with ExternalRun(embodiment) as run:
            client = run.client()
            published = client.list_tools()
            self.assertEqual([tool["name"] for tool in published[-2:]], ["run_begin", "run_finish"])
            dumped = lambda tools: [json.dumps(tool, ensure_ascii=False, separators=(",", ":")) for tool in tools]
            differing = [(ours, theirs) for ours, theirs in zip(published[:-2], startup_inline["catalog"], strict=True)
                         if dumped([ours]) != dumped([theirs])]
            self.assertEqual(differing, [])
            text, _, _ = briefing(client)
            run_dir = str(run.run_dir)
            self.assertNotIn(urdf, text)
            self.assertNotIn("Robot URDF\n", text)
            self.assertIn(EXTERNAL_URDF_ON_DEMAND_CLAUSE, text)
            self.assertIn("robot_information", text)
            self.assertEqual(run.session.briefing_text, text)
            self.assertEqual(metadata_findings(text), [])
            self.assertNotIn(run_dir, text)
            self.assertNotIn("/private/", text)
            self.assertNotIn("robot.urdf", text)
            for claim in ("Web search", "working directory", "sandbox isolation"):
                self.assertNotIn(claim, text)
            instance = run.store.read_state("runtime", "agent")["instance_id"]
            for secret in (run.session.thread_id, run.session.turn_id, run.session.bridge.url, "Bearer",
                           run.workbench.identity["episode_id"], instance):
                self.assertNotIn(secret, text)
            self.assertIsNone(re.search(r"\b[0-9a-f]{32,}\b", text))
            # 按需调用只读工具 robot_information 仍完整返回原始 URDF
            info_reply = client.call_tool("robot_information", {})
            self.assertFalse(info_reply.result["isError"])
            info_payload = json.loads(info_reply.result["content"][0]["text"])
            self.assertEqual(info_payload["urdf"], urdf)
        with ExternalRun(embodiment, overrides={"features": {"robot_urdf_delivery": "inline"}}) as run:
            text, _, _ = briefing(run.client())
            hosted_block = _local_source_instructions(RUN_MARKER + "/private/actor-source")
            self.assertEqual(host.count(hosted_block), 1)
            self.assertEqual(text, host.replace(hosted_block, EXTERNAL_AGENT_SOURCE_INSTRUCTIONS))
            self.assertIn(urdf, text)

    def test_x5_default_profile(self):
        self.check("x5")

    def test_so101_example_profile(self):
        self.check("so101")


class TaskSourceTests(unittest.TestCase):
    def test_without_prompt_there_is_no_task_and_no_default_prompt(self):
        # 19 M1: the development-acceptance prompt moves the arm; it must never become the external task.
        for prompt in (None, DEFAULT_PROMPT):
            with self.subTest(prompt=prompt is not None), ExternalRun("x5", prompt=prompt) as run:
                _, data, _ = briefing(run.client())
                self.assertEqual(data["task"], {"source": "none", "text": None})
                self.assertIsNone(run.workbench.config.run.prompt)
                effective = json.loads((run.run_dir / "effective_config.json").read_text())
                self.assertIsNone(effective["run"]["prompt"])
                self.assertEqual(run.store.events_by_type(["run.turn_settings"], limit=10), [])
                self.assertNotIn("Development acceptance", json.dumps(data))

    def test_operator_prompt_is_the_task(self):
        with ExternalRun("so101", prompt="Wave once, then report the gripper opening.") as run:
            text, data, _ = briefing(run.client())
            self.assertEqual(data["task"], {"source": "operator_prompt",
                                            "text": "Wave once, then report the gripper opening."})
            self.assertEqual(data["turn"], {"opened": True})
            self.assertEqual(data["budget"], run.harness.action_budget())
            self.assertNotIn("native_task", data)
            self.assertIn("client.artifact(ref) are available only inside program_run", text)

    def test_robodojo_external_run_delivers_native_task_rules_and_default_task_notes(self):
        # RoboDojo 外部模式虽禁用 --benchmark，但 run_begin 必须透传原生任务指令、公开任务规则与 default_task_notes，且 --resume 保持一致
        from ni_rua.cortex.agents.external.session import ExternalSession
        with tempfile.TemporaryDirectory(prefix="ni-rua-external-robodojo-brief-") as directory:
            root = Path(directory)
            observed_at_start = []
            orig_start = ExternalSession.start

            def spy_start(session, *args, **kwargs):
                observed_at_start.append((session.native_task, session.task_notes))
                return orig_start(session, *args, **kwargs)

            with mock.patch.object(ExternalSession, "start", autospec=True, side_effect=spy_start):
                with ExternalRun("x5", root=root, prompt=None, overrides={"run": {"task": "build_tower"}}) as run:
                    _, data, _ = briefing(run.client())
                    self.assertEqual(data["task"], {"source": "none", "text": None})
                    self.assertIn("native_task", data)
                    self.assertEqual(data["native_task"]["task"], "build_tower")
                    self.assertTrue(data["native_task"]["instruction"])
                    self.assertTrue(data["native_task"]["public_task_rules"])
                    self.assertTrue(data.get("task_notes"))
                    manifest = json.loads((run.run_dir / "manifest.json").read_text())
                    self.assertIsNotNone(manifest["public_task_rules"])
                    self.assertTrue(manifest["task_notes"])
                # --resume 同样须在 session.start() 打开端口前即注入归档的 native_task 与 task_notes
                with ExternalRun("x5", root=root, resume=True) as resumed:
                    _, rdata, _ = briefing(resumed.client())
                    self.assertEqual(rdata["native_task"], data["native_task"])
                    self.assertEqual(rdata["task_notes"], data["task_notes"])
            self.assertEqual(len(observed_at_start), 2)
            for native_task, task_notes in observed_at_start:
                self.assertIsNotNone(native_task)
                self.assertTrue(task_notes)


class BracketInRunTests(unittest.TestCase):
    def test_run_begin_tool_call_record_has_a_turn_id_and_the_gate_holds(self):
        with ExternalRun("x5") as run:
            client = run.client()
            refused = client.call_tool("robot_capabilities", {"detail": "summary"})
            self.assertIn("external_briefing_required", refused.result["content"][0]["text"])
            self.assertEqual(run.store.records("tool_call"), [])     # nothing reached the Harness
            briefing(client)
            served = client.call_tool("robot_capabilities", {"detail": "summary"})
            self.assertFalse(served.result["isError"])
            records = run.store.records("tool_call")
            self.assertEqual([record["params"]["tool"] for record in records], ["run_begin", "robot_capabilities"])
            self.assertEqual({record["correlation"]["turn_id"] for record in records}, {run.session.turn_id})
            self.assertIsNotNone(run.session.turn_id)
            finish = client.call_tool("run_finish", {"summary": "Read the capabilities; nothing moved."})
            self.assertFalse(finish.result["isError"])
            self.assertEqual(run.session.completed_final_answer(), "Read the capabilities; nothing moved.")
            self.assertIn("external_turn_closed", client.call_tool("robot_observe", {}).result["content"][0]["text"])

    def test_replies_and_catalogue_carry_the_default_reply_delivery_on_both_robots(self):
        """The reply settings are not Codex-only: an external host gets the notes in tools/list, the shaped reply
        through tools/call, and detail entries that name tools directly (every tool is published eagerly)."""
        from ni_rua.core.presentation import reply_compaction
        for embodiment, segments in (("x5", {"budget", "collision", "steps", "grasp_contact", "joint_travel"}),
                                     ("so101", set())):
            with self.subTest(embodiment=embodiment), ExternalRun(embodiment) as run:
                client = run.client()
                briefing(client)
                harness = run.harness
                self.assertIsNone(run.session.tool_output_text_budget_bytes())   # no host limit is known: only ordered
                self.assertFalse(harness.deferred_tool_loading)
                published = {tool["name"]: tool for tool in client.list_tools()}
                self.assertEqual(harness.tool_registry.loaded_tool_names(harness, role="actor"), set(published))
                self.assertIn("robot_action_status", published)
                plan = harness.tool_registry.static_once_plan(harness, role="actor")
                self.assertLessEqual(segments, set(plan.segments))
                if embodiment == "so101":   # no native step budget, joint-travel admission or grasp-contact fields
                    self.assertEqual(set(plan.segments) - {"trajectory", "program"}, set())
                description = published["robot_action"]["description"]
                for marker in (" Execution reply layout:", " Compact reply encoding:", plan.note):
                    self.assertEqual(description.count(marker), 1, marker[:40])
                self.assertEqual(sum("Compact reply encoding" in tool["description"] for tool in published.values()), 1)
                shaped, original = [], harness._shape_actor_reply

                def spy(response, kind, correlation, exact=False):
                    result = original(response, kind, correlation, exact=exact)
                    shaped.append((response, result, exact))
                    return result
                harness._shape_actor_reply = spy
                observed = client.call_tool("robot_observe", {})
                full = client.call_tool("robot_capabilities", {"detail": "full"})
                harness._shape_actor_reply = original
                self.assertFalse(observed.result["isError"])
                (presented, delivered, _), (_, _, exact) = shaped
                self.assertTrue(exact)   # a full read keeps the stored numbers
                texts = [item["text"] for item in delivered["contentItems"] if item["type"] == "inputText"]
                self.assertEqual([part["text"] for part in observed.result["content"] if part["type"] == "text"][:len(texts)],
                                 texts)
                self.assertNotEqual(delivered["contentItems"], presented["contentItems"])
                self.assertEqual(reply_compaction.dangling_same_as(delivered["contentItems"]), [])
                self.assertFalse(full.result["isError"])

    def test_rebriefing_appends_the_decision_reminder(self):
        with ExternalRun("x5") as run:
            client = run.client()
            briefing(client)
            _, data, result = briefing(client)
            self.assertEqual(data["turn"], {"opened": False})
            self.assertEqual(len(result.result["content"]), 3)       # instructions, data, decision reminder
            self.assertNotIn("startup_grounding", data)
            self.assertFalse(any(part.get("type") == "image" for part in result.result["content"]))
            reset, = [event["payload"] for event in run.store.events_by_type(["model_context.reset"], limit=10)]
            self.assertEqual(reset["reason"], "external_rebriefing")

    def test_capability_delivery_is_deferred_until_run_begin_and_deduplicates_in_delta(self):
        # external 后端不在 Workbench.start() 预播种能力基线，改为 run_begin 送达简报时同步播种
        with ExternalRun("x5") as run:
            self.assertIsNone(run.harness._capability_delivery._baseline)
            client = run.client()
            briefing(client)
            self.assertIsNotNone(run.harness._capability_delivery._baseline)
            cap = client.call_tool("robot_capabilities", {"detail": "summary"})
            self.assertFalse(cap.result["isError"])
            payload = json.loads(cap.result["content"][0]["text"])
            self.assertEqual(payload["capabilities_delivery"]["status"], "unchanged")
        # 显式 context_delivery="full" 时保持每轮自包含，不播种基线
        with ExternalRun("x5", overrides={"agent": {"external": {"context_delivery": "full"}}}) as run:
            self.assertTrue(run.harness.context_epoch_volatile)
            client = run.client()
            briefing(client)
            self.assertIsNone(run.harness._capability_delivery._baseline)
            cap = client.call_tool("robot_capabilities", {"detail": "summary"})
            self.assertFalse(cap.result["isError"])
            payload = json.loads(cap.result["content"][0]["text"])
            self.assertEqual(payload["capabilities_delivery"]["status"], "inline")

    def test_startup_inline_grounding_five_gates_on_run_begin(self):
        # 门禁全通：新鲜 simulated_steps 首次 run_begin 附带 obs_1 结构化文本与概览图，且不调用 robot.observe()
        with ExternalRun("x5") as run:
            client = run.client()
            with mock.patch.object(run.workbench.robot, "observe", wraps=run.workbench.robot.observe) as observe_spy:
                _, data, result = briefing(client)
            observe_spy.assert_not_called()
            content = result.result["content"]
            self.assertGreaterEqual(len(content), 4)
            self.assertEqual(content[0]["type"], "text")
            self.assertEqual(content[1]["type"], "text")
            self.assertEqual(content[2]["type"], "text")
            self.assertIn("Initial scene grounding", content[2]["text"])
            self.assertIn("obs_1", content[2]["text"])
            images = [part for part in content[3:] if part.get("type") == "image"]
            self.assertGreaterEqual(len(images), 1)
            sg = data.get("startup_grounding")
            self.assertIsInstance(sg, dict)
            self.assertEqual(sg["observation_id"], "obs_1")
            self.assertEqual(sg["image_count"], len(images))
            self.assertTrue(run.session._startup_grounding_delivered)
            # 门禁 3（后续新回合）：run_finish 后的下一次 run_begin 绝不回放旧 obs_1
            self.assertFalse(client.call_tool("run_finish", {"summary": "done"}).result["isError"])
            _, next_data, next_result = briefing(client)
            self.assertEqual(next_data["turn"], {"opened": True})
            self.assertNotIn("startup_grounding", next_data)
            self.assertEqual(len(next_result.result["content"]), 2)
            self.assertFalse(any(part.get("type") == "image" for part in next_result.result["content"]))

        # 门禁 2（real_time 真机）：SO-101 严禁把启动旧 obs_1 冒充当前画面
        with ExternalRun("so101") as run:
            _, data, result = briefing(run.client())
            self.assertNotIn("startup_grounding", data)
            self.assertEqual(len(result.result["content"]), 2)
            self.assertFalse(any(part.get("type") == "image" for part in result.result["content"]))

        # 门禁 1a（features.startup_inline_grounding=False）
        with ExternalRun("x5", overrides={"features": {"startup_inline_grounding": False}}) as run:
            _, data, result = briefing(run.client())
            self.assertNotIn("startup_grounding", data)
            self.assertEqual(len(result.result["content"]), 2)

        # 门禁 4（action_count > 0）
        with ExternalRun("x5") as run:
            run.harness.action_count = 1
            _, data, result = briefing(run.client())
            self.assertNotIn("startup_grounding", data)
            self.assertEqual(len(result.result["content"]), 2)

        # 门禁 5（latest != initial_observation_id）
        with ExternalRun("x5") as run:
            run.harness.latest = dict(run.harness.latest, observation_id="obs-after-reset")
            _, data, result = briefing(run.client())
            self.assertNotIn("startup_grounding", data)
            self.assertEqual(len(result.result["content"]), 2)

        # 门禁 1b（resume=True 不附带首轮接地）
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-external-grounding-resume-"))
        root = Path(directory)
        with ExternalRun("x5", root=root):
            pass
        with ExternalRun("x5", root=root, resume=True) as resumed:
            _, data, result = briefing(resumed.client())
            self.assertNotIn("startup_grounding", data)
            self.assertFalse(any(part.get("type") == "image" for part in result.result["content"]))


class ProvenanceTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.shared_run = cls.enterClassContext(ExternalRun("x5"))

    def test_manifest_and_summary_record_unknown_instead_of_host_values(self):
        # 19 M7.
        run = self.shared_run
        manifest = json.loads((run.run_dir / "manifest.json").read_text())
        block = manifest["agent_backend"]
        self.assertEqual((block["kind"], block["information_boundary"], block["model_relay"], block["counting"]),
                         ("external", "mcp_boundary_only", None, None))
        for key in ("model", "provider", "effort", "price_card", "model_request_limit", "no_progress_timeout_s"):
            with self.subTest(key=key):
                self.assertIn(key, manifest)
                self.assertIsNone(manifest[key])
        for key in MANIFEST_NULL_FIELDS:
            self.assertIsNone(manifest[key])
        for key in ("codex_sdk", "codex_cli", "app_server_pid"):
            self.assertNotIn(key, manifest)
        self.assertEqual(manifest["effective_config"]["agent"]["backend"], "external")
        summary = episode_summary(run.store, run.workbench.identity, run.session, run.harness, benchmark=False,
                                  final_status="stopped", started_at=0.0)
        for key in ("token_usage", "token_accounting", "served_models", "model_usage_relay", "model_requests"):
            with self.subTest(key=key):
                self.assertIsNone(summary[key])
        self.assertEqual(summary["agent_backend"], "external")
        conditions = _agent_conditions(manifest)
        self.assertEqual((conditions["agent_backend"], conditions["information_boundary"],
                          conditions["exploratory"]), ("external", "mcp_boundary_only", True))
        host = {"agent_backend": {"kind": "claude_code", "counting": "x", "cli": {"version": "2.1.321"}}}
        self.assertNotIn("exploratory", _agent_conditions(host))

    def test_report_and_audit_read_an_external_archive(self):
        with ExternalRun("x5") as run:
            client = run.client()
            briefing(client)
            client.call_tool("run_finish", {"summary": "done"})
            manifest = json.loads((run.run_dir / "manifest.json").read_text())
            evidence = load_native_evidence(run.run_dir, manifest, run.store, profile="audit")
            self.assertEqual(evidence.tool_item_ids, {record["params"]["callId"]
                                                      for record in run.store.records("tool_call")})
            self.assertEqual([turn["status"] for turn in evidence.turn_terminals], ["completed"])
            row, = build_report([run.run_dir])["runs"]
            self.assertEqual(row["report_status"], "readable", row.get("report_error"))
            self.assertEqual((row["conditions"]["agent_backend"], row["conditions"]["exploratory"]),
                             ("external", True))
            self.assertIsNone(row["served_models"])
            self.assertIsNone(row["token_accounting"]["tokens"]["total_tokens"])

    def test_client_attached_event_is_self_reported(self):
        run = self.shared_run
        run.client(name="opencode", version="2.2.7")
        attached, = [event["payload"] for event in run.store.events_by_type(["agent.client_attached"], limit=10)]
        self.assertEqual(attached["client_info"], {"name": "opencode", "version": "2.2.7", "self_reported": True})


class ConsoleTests(unittest.TestCase):
    def control(self, run, command, **fields):
        return run.control(dict(request_id="req-" + command, command=command, **fields))

    def test_commands_that_need_an_agent_ni_rua_runs_are_refused(self):
        # 19 S2.
        self.assertEqual(CONSOLE_COMMANDS_UNSUPPORTED, frozenset(UNSUPPORTED))
        with ExternalRun("x5") as run:
            for command in UNSUPPORTED:
                with self.subTest(command=command), self.assertRaisesRegex(
                        AgentCapabilityError, "^" + command + "_unsupported_by_backend$"):
                    self.control(run, command, prompt="Do it.", model="m", effort="high", tier="priority")
            response = run.call(lambda: run.workbench.control_dispatch(
                dict(request_id="http-prompt", command="prompt", prompt="Do it."), None))
            self.assertEqual(response.status, 400)
            self.assertEqual(json.loads(response.body)["error"], "prompt_unsupported_by_backend")
            self.assertEqual(run.session.task, {"text": None, "source": "none"})

    def test_reset_closes_an_open_turn_first(self):
        # 19 S1: an agent that never called run_finish must not block the operator's reset.
        with ExternalRun("x5") as run:
            client = run.client()
            briefing(client)
            self.assertEqual(run.session.state, "running")
            with mock.patch.object(run.harness, "reset", return_value={"observation_id": "obs-reset"}) as reset:
                result = self.control(run, "reset")
            self.assertEqual((result["status"], result["observation_id"]), ("reset", "obs-reset"))
            reset.assert_called_once()
            self.assertEqual(run.session.state, "interrupted")
            closed, = [event["payload"] for event in run.store.events_by_type(["agent.turn_closed_by_operator"],
                                                                               limit=10)]
            self.assertEqual(closed, {"reason": "operator_reset"})
            self.assertIn("external_turn_closed", client.call_tool("robot_observe", {}).result["content"][0]["text"])

    def test_reset_still_refuses_while_an_action_runs(self):
        with ExternalRun("x5") as run:
            briefing(run.client())
            run.harness.active_action = "act-1"
            try:
                with self.assertRaisesRegex(ValueError, "reset_requires_idle_turn_and_motion"):
                    self.control(run, "reset")
            finally:
                run.harness.active_action = None
            self.assertEqual(run.session.state, "running")            # nothing was closed

    def test_stop_closes_the_turn_and_motion_admission(self):
        with ExternalRun("x5") as run:
            client = run.client()
            briefing(client)
            result = run.call(lambda: run.workbench.stop("operator-stop"))
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(run.session.state, "interrupted")
            self.assertFalse(run.harness.accepting)

    def test_agent_bar_payload(self):
        with ExternalRun("x5", overrides={"agent": {"profiles_dir": "/nonexistent-ni-rua-profiles"}}) as run:
            view = run.workbench.console_state()["settings"]["agent"]
            self.assertEqual(view["backend"], "external")
            self.assertEqual((view["model"], view["profile"], view["provider"], view["effort"], view["wire_api"]),
                             (None, None, None, None, None))
            self.assertEqual(view["external"], {"endpoint": run.url, "client": None, "last_call_at": None,
                                           "replay": False})
            client = run.client(name="claude-code", version="2.1.321")
            briefing(client)
            with mock.patch.object(agent_settings, "runtime_problem", return_value=None):
                options = run.call(lambda: run.workbench.agent_options())
            external = options["current"]["external"]
            self.assertEqual(external["client"], {"name": "claude-code", "version": "2.1.321", "self_reported": True})
            self.assertIsNotNone(external["last_call_at"])
            self.assertFalse(options["handoff"]["allowed"])
            self.assertIn("handoff_unsupported_by_backend",
                          [blocker["code"] for blocker in options["handoff"]["blockers"]])

    def test_handoff_to_external_is_refused(self):
        with ExternalRun("x5") as run:
            with self.assertRaisesRegex(ValueError, "^handoff_unsupported_by_backend: an external agent run starts "
                                                    "only from the command line"):
                run.workbench._handoff_selection({"backend": "external", "profile": "p"})



class OperatorTurnTests(unittest.TestCase):
    """Review 23 M2: an open external turn never blocks an operator command; an active action still does."""

    def control(self, run, command, **fields):
        return run.control(dict(request_id="req-" + command + "-" + str(len(fields)), command=command, **fields))

    def assertNotBlockedByTheTurn(self, call, idle_code):
        try:
            call()
        except ValueError as error:
            self.assertNotIn(idle_code, str(error))
            self.assertNotIn("codex_turn_running", str(error))

    def test_operator_commands_with_an_open_turn(self):
        with ExternalRun("x5") as run:
            briefing(run.client())
            workbench = run.workbench
            self.assertEqual(run.session.state, "running")
            self.assertFalse(session_turn_blocks_operator(run.session))
            self.assertEqual(workbench.resume_blockers(), [])
            self.assertEqual(workbench.calibration_blockers(False), [])
            workbench.require_observer_idle()                         # no exception
            for command in ("go_home", "calibrate", "reload_calibration"):
                with self.subTest(command=command):
                    self.assertNotBlockedByTheTurn(lambda: self.control(run, command),
                                                   command + "_requires_idle_turn_and_motion")
            self.assertNotBlockedByTheTurn(lambda: self.control(run, "handheld_capture", action="start"),
                                           "handheld_capture_requires_idle_turn_and_motion")
            result = self.control(run, "resume")                      # nothing unknown: admission reopens
            self.assertEqual(result["status"], "resumed")
            self.assertEqual(run.session.state, "running")            # the agent's turn stays open
            # Motion still protects the robot: an active action blocks every one of them.
            run.harness.active_action = "act-1"
            try:
                self.assertIn("action_in_progress", workbench.resume_blockers())
                self.assertIn("action_in_progress", workbench.calibration_blockers(False))
                with self.assertRaisesRegex(ValueError, "observer_operation_requires_idle_run"):
                    workbench.require_observer_idle()
                with self.assertRaisesRegex(ValueError, "go_home_requires_idle_turn_and_motion"):
                    self.control(run, "go_home", again=True)
                with self.assertRaisesRegex(ValueError, "handheld_capture_requires_idle_turn_and_motion"):
                    self.control(run, "handheld_capture", action="start", again=True)
            finally:
                run.harness.active_action = None

    def test_stop_does_not_wait_for_a_turn_the_agent_reopens(self):
        with ExternalRun("x5") as run:
            briefing(run.client())
            # The agent's run_begin lands right after the stop closed its turn: modelled by an interrupt that
            # leaves the turn open. The stop must still end promptly as stopped, not wait 45 s and report unknown.
            with mock.patch.object(run.session, "interrupt", lambda: None):
                result = run.call(lambda: run.workbench.stop("operator-stop"), timeout=30)
            self.assertEqual(result["status"], "stopped")
            self.assertEqual(run.session.state, "running")
            self.assertFalse(run.harness.accepting)

    def test_host_sessions_keep_blocking(self):
        from support.fake_agent import FakeAgentSession
        session = FakeAgentSession()
        session.state = "running"
        self.assertTrue(session_turn_blocks_operator(session))
        session.state = "idle"
        self.assertFalse(session_turn_blocks_operator(session))
        self.assertFalse(session_turn_blocks_operator(None))


class ResumeTaskTests(unittest.TestCase):
    """Review 23 M3: a resume keeps the archived task; a resume --prompt replaces it, in the manifest and briefing."""

    def resumed(self, root, prompt=None):
        overrides = {"run": {"prompt": prompt}} if prompt else None
        with ExternalRun("x5", root=root, resume=True, overrides=overrides) as run:
            _, data, _ = briefing(run.client())
            manifest = json.loads((run.run_dir / "manifest.json").read_text())
            return data["task"], manifest["effective_config"]["run"]["prompt"]

    def test_resume_keeps_or_replaces_the_task(self):
        with tempfile.TemporaryDirectory(prefix="ni-rua-external-resume-") as directory:
            root = Path(directory)
            first = "Stack the red block on the blue block."
            with ExternalRun("x5", root=root, prompt=first) as run:
                _, data, _ = briefing(run.client())
                self.assertEqual(data["task"], {"source": "operator_prompt", "text": first})
            operator = {"source": "operator_prompt", "text": first}
            self.assertEqual(self.resumed(root), (operator, first))
            self.assertEqual(self.resumed(root), (operator, first))        # a second resume keeps it too
            second = "Now put the green block on top."
            self.assertEqual(self.resumed(root, second), ({"source": "operator_prompt", "text": second}, second))
            self.assertEqual(self.resumed(root), ({"source": "operator_prompt", "text": second}, second))

    def test_a_run_without_task_stays_without_one(self):
        with tempfile.TemporaryDirectory(prefix="ni-rua-external-resume-") as directory:
            root = Path(directory)
            with ExternalRun("x5", root=root):
                pass
            self.assertEqual(self.resumed(root), ({"source": "none", "text": None}, None))

    def test_historical_archive_without_context_delivery_backfills_full_on_resume_and_replay(self):
        # 翻默认后恢复历史缺少 context_delivery 的归档：优先从 manifest.agent_backend.context_delivery 回填，缺失回退 "full"
        from ni_rua.apps.gateway.workbench import Workbench
        from ni_rua.core.config import Config
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-external-legacy-resume-"))
        root = Path(directory)
        with ExternalRun("x5", root=root, prompt="Legacy task."):
            pass
        manifest_path = root / "run" / "manifest.json"
        original_manifest = json.loads(manifest_path.read_text())

        # 场景 1：旧 external 归档省略 agent.external，manifest.agent_backend.context_delivery="full" -> 回填 "full"
        legacy_full = json.loads(json.dumps(original_manifest))
        legacy_full["effective_config"]["agent"] = {"backend": "external"}
        legacy_full["agent_backend"]["context_delivery"] = "full"
        manifest_path.write_text(json.dumps(legacy_full, indent=2))
        with ExternalRun("x5", root=root, resume=True) as resumed:
            self.assertEqual(resumed.workbench.config.agent.external.context_delivery, "full")
            self.assertEqual(resumed.session.context_delivery, "full")
            self.assertTrue(resumed.harness.context_epoch_volatile)

        # 场景 2：旧 external 归档自定义端口但缺少 context_delivery 且 agent_backend 也缺该键 -> 回退 "full"
        legacy_absent = json.loads(json.dumps(original_manifest))
        legacy_absent["effective_config"]["agent"] = {"backend": "external", "external": {"mcp_port": 0}}
        legacy_absent["agent_backend"].pop("context_delivery", None)
        manifest_path.write_text(json.dumps(legacy_absent, indent=2))
        with ExternalRun("x5", root=root, resume=True) as resumed:
            self.assertEqual(resumed.workbench.config.agent.external.context_delivery, "full")
            self.assertEqual(resumed.session.context_delivery, "full")

        # 场景 3：新 external 归档省略 agent.external 但 agent_backend.context_delivery="delta" -> 保留 "delta"
        modern_delta = json.loads(json.dumps(original_manifest))
        modern_delta["effective_config"]["agent"] = {"backend": "external"}
        modern_delta["agent_backend"]["context_delivery"] = "delta"
        manifest_path.write_text(json.dumps(modern_delta, indent=2))
        with ExternalRun("x5", root=root, resume=True) as resumed:
            self.assertEqual(resumed.workbench.config.agent.external.context_delivery, "delta")
            self.assertEqual(resumed.session.context_delivery, "delta")
            self.assertFalse(resumed.harness.context_epoch_volatile)

        # 场景 4：历史 claude_code / opencode 归档缺少 context_delivery 或旧版 subagents=True 硬覆写为 full 时，
        # replay 均须回填 "full" 以保历史语义与 tools_digest；现代 recorded="delta" 归档则保持 "delta"
        for backend in ("claude_code", "opencode"):
            with self.subTest(backend=backend, case="missing_key"):
                hosted = json.loads(json.dumps(original_manifest))
                hosted["effective_config"]["agent"] = {"backend": backend, backend: {}}
                hosted["agent_backend"] = {"kind": backend}
                manifest_path.write_text(json.dumps(hosted, indent=2))
                wb = Workbench(root / "run", replay=True, config=Config.from_dict({"run": {"replay": True}}))
                try:
                    self.assertEqual(getattr(wb.config.agent, backend).context_delivery, "full")
                finally:
                    wb.store.close()
                    wb.run_lock.close()
            with self.subTest(backend=backend, case="legacy_subagents_override"):
                hosted = json.loads(json.dumps(original_manifest))
                hosted["effective_config"]["agent"] = {
                    "backend": backend,
                    backend: {"subagents": True, "context_delivery": "delta"},
                }
                hosted["agent_backend"] = {"kind": backend, "context_delivery": "full"}
                manifest_path.write_text(json.dumps(hosted, indent=2))
                wb = Workbench(root / "run", replay=True, config=Config.from_dict({"run": {"replay": True}}))
                try:
                    self.assertEqual(getattr(wb.config.agent, backend).context_delivery, "full")
                finally:
                    wb.store.close()
                    wb.run_lock.close()
            with self.subTest(backend=backend, case="modern_subagents_delta"):
                hosted = json.loads(json.dumps(original_manifest))
                hosted["effective_config"]["agent"] = {
                    "backend": backend,
                    backend: {"subagents": True, "context_delivery": "delta"},
                }
                hosted["agent_backend"] = {"kind": backend, "context_delivery": "delta"}
                manifest_path.write_text(json.dumps(hosted, indent=2))
                wb = Workbench(root / "run", replay=True, config=Config.from_dict({"run": {"replay": True}}))
                try:
                    self.assertEqual(getattr(wb.config.agent, backend).context_delivery, "delta")
                finally:
                    wb.store.close()
                    wb.run_lock.close()


class InFlightTests(unittest.TestCase):
    """Review 23 S6: estop, a failed reset and shutdown while an external agent's call is in flight."""

    def gate(self, run):
        """Hold every Harness call until the returned event is set; returns ``(event, calls)``."""
        release, calls, handler = threading.Event(), [], run.session.tool_handler

        def held(params):
            calls.append(params["tool"])
            release.wait(30)
            return handler(params)
        run.session.tool_handler = held
        return release, calls

    def test_estop_with_a_call_in_flight(self):
        with ExternalRun("x5") as run:
            client = run.client()
            briefing(client)
            turn = run.session.turn_id
            release, calls = self.gate(run)
            pending = client.start_call("robot_capabilities", {"detail": "summary"})
            self.assertTrue(wait_until(lambda: calls == ["robot_capabilities"]))
            result = run.control(dict(request_id="estop-1", command="estop"))
            self.assertEqual(result["admission"], "closed")
            self.assertEqual(run.session.state, "interrupted")
            release.set()
            reply = pending.wait(30)
            self.assertFalse(reply.result["isError"])                 # the admitted call returns its real result
            self.assertEqual(calls, ["robot_capabilities"])           # never re-sent
            record, = [item for item in run.store.records("tool_call") if item["params"]["tool"] == "robot_capabilities"]
            self.assertEqual(record["correlation"]["turn_id"], turn)
            self.assertIn("the operator stopped the run",
                          client.call_tool("robot_observe", {}).result["content"][0]["text"])

    def test_a_refused_reset_keeps_the_turn(self):
        # Review 23 S4: the turn is released only once the reset happened.
        with ExternalRun("x5") as run:
            briefing(run.client())
            with mock.patch.object(run.harness, "reset", side_effect=ValueError("unsupported_capability:reset")):
                with self.assertRaisesRegex(ValueError, "unsupported_capability:reset"):
                    run.control(dict(request_id="reset-1", command="reset"))
            self.assertEqual(run.session.state, "running")
            self.assertEqual(run.store.events_by_type(["agent.turn_closed_by_operator"], limit=10), [])

    def test_shutdown_with_a_call_in_flight(self):
        with ExternalRun("x5") as run:
            client = run.client()
            briefing(client)
            release, calls = self.gate(run)
            failures = []
            server = run.session.bridge.server
            report_failure = server.on_delivery_failure
            server.on_delivery_failure = lambda call_id, reason: (failures.append(reason),
                                                                   report_failure(call_id, reason))
            pending = client.start_call("robot_capabilities", {"detail": "summary"})
            self.assertTrue(wait_until(lambda: calls == ["robot_capabilities"]))
            closing = asyncio.run_coroutine_threadsafe(run.workbench.close(), run.loop)
            self.assertTrue(wait_until(lambda: run.session.closing))
            release.set()
            report = closing.result(60)
            pending.wait(30)
            self.assertEqual(report["errors"], [])
            self.assertEqual(failures, ["tool_reply_delivery_uncertain"])   # the endpoint closed with the call open
            self.assertEqual(calls, ["robot_capabilities"])
            self.assertTrue(run.session.close_report["clean_shutdown"])     # the stop closed the turn first


class ServeTests(unittest.IsolatedAsyncioTestCase):
    """Review 23 S6: the real ``serve`` of an interactive external run prints the endpoint and finalises summary.json."""

    async def test_serve_prints_the_endpoint_and_finalises_the_summary(self):
        from ni_rua.apps.gateway.extensions import handoff
        from ni_rua.apps.gateway.server import serve
        from ni_rua.core.config import Config
        from support.fixture_robot import FixtureRobot
        directory = tempfile.TemporaryDirectory(prefix="ni-rua-external-serve-")
        self.addCleanup(directory.cleanup)
        root = Path(directory.name).resolve()
        config = Config.from_dict({
            "features": {"grasp_memory_dir": str(root / "memory"), "observation_profile": "rgb_proprio_v1",
                         "motion_review": False, "terminal": False, "record_video": False},
            "agent": {"backend": "external", "external": {"mcp_port": 0}},
            "run": {"experiment_root": str(root / "exp"), "category": "external", "task": "fixture", "port": 0,
                    "prompt": "Report the gripper opening."}})
        lines = []

        class ClosingRobot(FixtureRobot):
            def close(self):
                return {"native_close_acknowledged": True, "process_exit_confirmed": True,
                        "reader_exit_confirmed": True, "returncode": 0, "errors": []}
        with mock.patch("ni_rua.apps.gateway.workbench.create_backend",
                        side_effect=lambda **kwargs: ClosingRobot(kwargs["run_dir"])), \
                mock.patch("ni_rua.apps.gateway.server._say", side_effect=lines.append):
            serving = asyncio.create_task(serve(handoff.launch_args(config)))
            endpoint = None
            for _ in range(1500):
                endpoint = next((line.split()[4] for line in lines
                                 if isinstance(line, str) and line.startswith("External agent MCP endpoint")), None)
                if endpoint or serving.done():
                    break
                await asyncio.sleep(0.02)
            self.assertIsNotNone(endpoint, lines)
            banner, = [line for line in lines if line.startswith("External agent MCP endpoint")]
            self.assertIn("(no bearer;", banner)

            def agent():
                client = FakeMcpClient(endpoint, protocol_version="2025-11-25")
                client.handshake()
                begin = client.call_tool("run_begin", {})
                finish = client.call_tool("run_finish", {"summary": "Reported the opening; nothing moved."})
                return begin.result, finish.result
            begin, finish = await asyncio.to_thread(agent)
            self.assertFalse(begin["isError"] or finish["isError"])
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.wait_for(serving, 60)
        run_dir, = [path.parent for path in (root / "exp").rglob("summary.json")]
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["agent_backend"], summary["agent_turn_status"]),
                         ("completed", "external", "completed"))
        for key in ("token_usage", "token_accounting", "served_models", "model_usage_relay", "model_requests"):
            self.assertIsNone(summary[key], key)
        self.assertIsNone(summary["infrastructure_error"])
        self.assertEqual(summary["cleanup_status"], "completed")
        manifest = json.loads((run_dir / "manifest.json").read_text())
        self.assertEqual(manifest["agent_backend"]["kind"], "external")
        self.assertIsNone(manifest["model_request_limit"])

    async def test_serve_shutdown_with_open_external_turn_records_interrupted_summary(self):
        # 外部回合尚未调用 run_finish 就收到停机信号时，summary.json 的 agent_turn_status 须为 interrupted（与事件流一致）
        from ni_rua.apps.gateway.extensions import handoff
        from ni_rua.apps.gateway.server import serve
        from ni_rua.core.config import Config
        from support.fixture_robot import FixtureRobot
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-external-serve-open-"))
        root = Path(directory).resolve()
        config = Config.from_dict({
            "agent": {"backend": "external", "external": {"mcp_port": 0}},
            "features": {"grasp_memory_dir": str(root / "memory"), "observation_profile": "rgb_proprio_v1",
                         "motion_review": False, "terminal": False, "record_video": False},
            "run": {"experiment_root": str(root / "exp"), "category": "external", "task": "fixture", "port": 0,
                    "prompt": "Hold open turn."},
        })
        lines = []

        class ClosingRobot(FixtureRobot):
            def close(self):
                return {"native_close_acknowledged": True, "process_exit_confirmed": True,
                        "reader_exit_confirmed": True, "returncode": 0, "errors": []}

        with mock.patch("ni_rua.apps.gateway.workbench.create_backend",
                        side_effect=lambda **kwargs: ClosingRobot(kwargs["run_dir"])), \
                mock.patch("ni_rua.apps.gateway.server._say", side_effect=lines.append):
            serving = asyncio.create_task(serve(handoff.launch_args(config)))
            endpoint = None
            for _ in range(1500):
                endpoint = next((line.split()[4] for line in lines
                                 if isinstance(line, str) and line.startswith("External agent MCP endpoint")), None)
                if endpoint or serving.done():
                    break
                await asyncio.sleep(0.02)
            self.assertIsNotNone(endpoint, lines)

            def open_only():
                client = FakeMcpClient(endpoint, protocol_version="2025-11-25")
                client.handshake()
                return client.call_tool("run_begin", {}).result

            begin = await asyncio.to_thread(open_only)
            self.assertFalse(begin["isError"])
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.wait_for(serving, 60)
        run_dir, = [path.parent for path in (root / "exp").rglob("summary.json")]
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["agent_backend"], summary["agent_turn_status"]),
                         ("stopped", "external", "interrupted"))
        self.assertEqual(summary["cleanup_status"], "completed")

    async def test_serve_interactive_run_finish_and_observer_evaluate_persists_scored_summary_and_evaluation(self):
        # 交互模式调用 run_finish + POST /v1/observer/evaluate 后停机，summary.json 与 evaluation.json 须正确落盘计分
        import aiohttp
        from ni_rua.apps.gateway.extensions import handoff
        from ni_rua.apps.gateway.server import serve
        from ni_rua.core.config import Config
        from support.fixture_robot import FixtureRobot
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-external-serve-eval-"))
        root = Path(directory).resolve()
        config = Config.from_dict({
            "agent": {"backend": "external", "external": {"mcp_port": 0}},
            "features": {"grasp_memory_dir": str(root / "memory"), "observation_profile": "rgb_proprio_v1",
                         "motion_review": False, "terminal": False, "record_video": False},
            "run": {"experiment_root": str(root / "exp"), "category": "external", "task": "fixture", "port": 0,
                    "prompt": "Finish and evaluate."},
        })
        lines = []

        class EvaluatingRobot(FixtureRobot):
            def evaluate(self, final_check=True):
                return {"task_success": False, "normalized_score": 0.4, "native_process_score": 40.0,
                        "native_reward": 0.0, "valid_episode": True, "success_authority": "native_final_check",
                        "authority": "fixture"}

            def close(self):
                return {"native_close_acknowledged": True, "process_exit_confirmed": True,
                        "reader_exit_confirmed": True, "returncode": 0, "errors": []}

        with mock.patch("ni_rua.apps.gateway.workbench.create_backend",
                        side_effect=lambda **kwargs: EvaluatingRobot(kwargs["run_dir"])), \
                mock.patch("ni_rua.apps.gateway.server._say", side_effect=lines.append):
            serving = asyncio.create_task(serve(handoff.launch_args(config)))
            endpoint = gateway_url = None
            for _ in range(1500):
                for line in lines:
                    if isinstance(line, str) and line.startswith("Workbench http://"):
                        gateway_url = line.split()[1].rstrip(";")
                    if isinstance(line, str) and line.startswith("External agent MCP endpoint"):
                        endpoint = line.split()[4]
                if (endpoint and gateway_url) or serving.done():
                    break
                await asyncio.sleep(0.02)
            self.assertIsNotNone(endpoint, lines)
            self.assertIsNotNone(gateway_url, lines)

            def finish_agent():
                client = FakeMcpClient(endpoint, protocol_version="2025-11-25")
                client.handshake()
                b = client.call_tool("run_begin", {}).result
                f = client.call_tool("run_finish", {"summary": "Partial progress done."}).result
                return b, f

            b, f = await asyncio.to_thread(finish_agent)
            self.assertFalse(b["isError"] or f["isError"])
            async with aiohttp.ClientSession() as http:
                async with http.post(f"{gateway_url}/v1/observer/evaluate", json={}) as resp:
                    self.assertEqual(resp.status, 200)
                    eval_body = await resp.json()
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.wait_for(serving, 60)
        run_dir, = [path.parent for path in (root / "exp").rglob("summary.json")]
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["status"], "completed")
        self.assertEqual(summary["primary_outcome"], "scored_failure")
        self.assertTrue(summary["scored_episode"])
        self.assertEqual(summary["task_success"], False)
        self.assertAlmostEqual(summary["normalized_score"], 0.4)
        evaluation_path = run_dir / "evaluation.json"
        self.assertTrue(evaluation_path.is_file())
        saved_eval = json.loads(evaluation_path.read_text())
        self.assertEqual(saved_eval["result_ref"], eval_body["result_ref"])
        self.assertEqual(saved_eval["result"], eval_body["result"])
        from ni_rua.apps.evaluation.audit import audit
        report = audit(run_dir)
        self.assertTrue(report["passed"], report)

    async def test_serve_interactive_reset_after_run_finish_marks_stopped_and_drops_stale_evaluation(self):
        # 交互模式 run_finish + observer/evaluate 之后若发生环境重置再停机，须判 stopped 且不落盘过期 evaluation.json
        import aiohttp
        from ni_rua.apps.gateway.extensions import handoff
        from ni_rua.apps.gateway.server import serve
        from ni_rua.core.config import Config
        from support.fixture_robot import FixtureRobot
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-external-serve-reset-"))
        root = Path(directory).resolve()
        config = Config.from_dict({
            "agent": {"backend": "external", "external": {"mcp_port": 0}},
            "features": {"grasp_memory_dir": str(root / "memory"), "observation_profile": "rgb_proprio_v1",
                         "motion_review": False, "terminal": False, "record_video": False},
            "run": {"experiment_root": str(root / "exp"), "category": "external", "task": "fixture", "port": 0,
                    "prompt": "Finish then reset."},
        })
        lines = []

        class ResettableEvaluatingRobot(FixtureRobot):
            def start(self):
                return {**super().start(), "reset": True}

            def reset(self, seed=0, layout_id=0):
                return {"observation": self.observe()}

            def evaluate(self, final_check=True):
                return {"task_success": True, "normalized_score": 1.0, "native_process_score": 100.0,
                        "native_reward": 1.0, "valid_episode": True, "success_authority": "native_final_check",
                        "authority": "fixture"}

            def close(self):
                return {"native_close_acknowledged": True, "process_exit_confirmed": True,
                        "reader_exit_confirmed": True, "returncode": 0, "errors": []}

        with mock.patch("ni_rua.apps.gateway.workbench.create_backend",
                        side_effect=lambda **kwargs: ResettableEvaluatingRobot(kwargs["run_dir"])), \
                mock.patch("ni_rua.apps.gateway.server._say", side_effect=lines.append):
            serving = asyncio.create_task(serve(handoff.launch_args(config)))
            endpoint = gateway_url = None
            for _ in range(1500):
                for line in lines:
                    if isinstance(line, str) and line.startswith("Workbench http://"):
                        gateway_url = line.split()[1].rstrip(";")
                    if isinstance(line, str) and line.startswith("External agent MCP endpoint"):
                        endpoint = line.split()[4]
                if (endpoint and gateway_url) or serving.done():
                    break
                await asyncio.sleep(0.02)
            self.assertIsNotNone(endpoint, lines)
            self.assertIsNotNone(gateway_url, lines)

            def finish_agent():
                client = FakeMcpClient(endpoint, protocol_version="2025-11-25")
                client.handshake()
                b = client.call_tool("run_begin", {}).result
                f = client.call_tool("run_finish", {"summary": "Finished before reset."}).result
                return b, f

            b, f = await asyncio.to_thread(finish_agent)
            self.assertFalse(b["isError"] or f["isError"])
            async with aiohttp.ClientSession() as http:
                async with http.post(f"{gateway_url}/v1/observer/evaluate", json={}) as resp:
                    self.assertEqual(resp.status, 200)
                async with http.post(f"{gateway_url}/v1/control", json={"request_id": "reset-after-finish", "command": "reset"}) as resp:
                    self.assertEqual(resp.status, 200)
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.wait_for(serving, 60)
        run_dir, = [path.parent for path in (root / "exp").rglob("summary.json")]
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual(summary["status"], "stopped")
        self.assertFalse(summary["scored_episode"])
        self.assertIsNone(summary["normalized_score"])
        self.assertFalse((run_dir / "evaluation.json").exists())

    async def test_serve_interactive_native_terminal_sets_completed_even_without_run_finish(self):
        # 交互模式原生环境已达成终态时，即使停机前未调用 run_finish，summary.json 的 status 仍应为 completed
        from ni_rua.apps.gateway.extensions import handoff
        from ni_rua.apps.gateway.server import serve
        from ni_rua.core.config import Config
        from support.fixture_robot import FixtureRobot
        directory = self.enterContext(tempfile.TemporaryDirectory(prefix="ni-rua-external-serve-terminal-"))
        root = Path(directory).resolve()
        config = Config.from_dict({
            "agent": {"backend": "external", "external": {"mcp_port": 0}},
            "features": {"grasp_memory_dir": str(root / "memory"), "observation_profile": "rgb_proprio_v1",
                         "motion_review": False, "terminal": False, "record_video": False},
            "run": {"experiment_root": str(root / "exp"), "category": "external", "task": "fixture", "port": 0,
                    "prompt": "Native terminal."},
        })
        lines = []
        created_workbenches = []
        orig_wb = None

        class ClosingRobot(FixtureRobot):
            def close(self):
                return {"native_close_acknowledged": True, "process_exit_confirmed": True,
                        "reader_exit_confirmed": True, "returncode": 0, "errors": []}

        from ni_rua.apps.gateway import server as gateway_server
        orig_wb = gateway_server.Workbench

        def spy_wb(*args, **kwargs):
            wb = orig_wb(*args, **kwargs)
            created_workbenches.append(wb)
            return wb

        with mock.patch("ni_rua.apps.gateway.workbench.create_backend",
                        side_effect=lambda **kwargs: ClosingRobot(kwargs["run_dir"])), \
                mock.patch("ni_rua.apps.gateway.server.Workbench", side_effect=spy_wb), \
                mock.patch("ni_rua.apps.gateway.server._say", side_effect=lines.append):
            serving = asyncio.create_task(serve(handoff.launch_args(config)))
            endpoint = None
            for _ in range(1500):
                endpoint = next((line.split()[4] for line in lines
                                 if isinstance(line, str) and line.startswith("External agent MCP endpoint")), None)
                if endpoint or serving.done():
                    break
                await asyncio.sleep(0.02)
            self.assertIsNotNone(endpoint, lines)
            await asyncio.to_thread(lambda: (c := FakeMcpClient(endpoint, protocol_version="2025-11-25"), c.handshake(), c.call_tool("run_begin", {})))
            wb = created_workbenches[0]
            terminal = {"ended": True, "success": True, "reason": "native_success"}
            wb.harness.episode_terminal = terminal
            wb.store.set_state("runtime", "native_episode", terminal)
            os.kill(os.getpid(), signal.SIGINT)
            await asyncio.wait_for(serving, 60)
        run_dir, = [path.parent for path in (root / "exp").rglob("summary.json")]
        summary = json.loads((run_dir / "summary.json").read_text())
        self.assertEqual((summary["status"], summary["agent_turn_status"]), ("completed", "interrupted"))

if __name__ == "__main__":
    unittest.main()
