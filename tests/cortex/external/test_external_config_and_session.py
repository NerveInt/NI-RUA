"""Merged cortex tests:
- tests/test_agent_external_config.py: External agent backend: configuration, launch gate, factory, queue and the bracket tool contracts.
- tests/test_agent_external_session.py: ExternalSession and its session bracket (design 20 sections 2.1-2.4; review 19 M3, M8, S3, N2 and B3).
- tests/test_agent_external_transport.py: External agent backend, transport layer (design 20 sections 2.1, 2.2 and 2.4; review 19 S7 and M5).
"""

NI_RUA_TEST_TIER = "component"

import json
import socket
import tempfile
import unittest
from pathlib import Path
from ni_rua.apps.gateway.workbench import _configured_tool_specs
from ni_rua.core.config import Config, ConfigError
from ni_rua.core.storage.store import Store
from ni_rua.cortex.agents import factory
from ni_rua.cortex.agents.external.session import ExternalSession
from ni_rua.runtime.tools.builtins import create_registry
from ni_rua.runtime.tools.plugins import run_bracket
from ni_rua.runtime.tools.registry import ToolRegistry
from scripts import benchmark_queue as queue
from support.agent_fixtures import gaps
import threading
import time
from support.fake_mcp_client import FakeMcpClient
from ni_rua.core.config import Config
from ni_rua.core.presentation.actor_presentation import metadata_findings
from ni_rua.core.tool_output import ToolOutput, wire_response
from ni_rua.cortex.agents.base import AgentCapabilityError
from ni_rua.cortex.agents.external.session import (CLIENT_FIELD_MAX_CHARS, EXTERNAL_MCP_INSTRUCTIONS,
                                                   INFORMATION_BOUNDARY, RUN_FINISH_ORIGIN, ExternalSession)
from ni_rua.runtime.tools.plugins.run_bracket import make_run_bracket_tools
from ni_rua.runtime.tools.registry import ToolRegistry, ToolSpec
import errno
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.cortex.agents.mcp_server import NiRuaMcpServer
from ni_rua.cortex.agents.mcp_session import McpToolBridge
from support.agent_fixtures import full_registry
from support.paths import REPO_ROOT

# ==== tests/test_agent_external_config.py ====

IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)


def external(document=None, port=0):
    base = {"agent": {"backend": "external", "external": {"mcp_port": port}}}
    from ni_rua.core.config import _merge
    return Config.from_dict(_merge(base, document or {}))


class ConfigTests(unittest.TestCase):
    def test_section_defaults_bounds_and_snapshot_omission(self):
        self.assertEqual(Config().agent.external.mcp_port, 8790)
        self.assertEqual(Config().agent.external.context_delivery, "delta")
        self.assertEqual(Config.from_dict({"agent": {"backend": "external"}}).agent.context_delivery, "delta")
        self.assertNotIn("agent", Config().to_dict())
        snapshot = Config.from_dict({"agent": {"backend": "external"}}).to_dict()
        self.assertEqual(snapshot["agent"]["backend"], "external")
        self.assertNotIn("external", snapshot["agent"])       # omitted while unchanged
        self.assertNotIn("external", Config.from_dict({"agent": {"backend": "claude_code"}}).to_dict()["agent"])
        self.assertEqual(external(port=9001).to_dict()["agent"]["external"],
                         {"mcp_port": 9001, "context_delivery": "delta"})
        self.assertEqual(Config.from_dict(external(port=9001).to_dict()).agent.external.mcp_port, 9001)
        full_cfg = external({"agent": {"external": {"context_delivery": "full"}}}, port=8790)
        self.assertEqual(full_cfg.agent.context_delivery, "full")
        self.assertEqual(full_cfg.to_dict()["agent"]["external"], {"mcp_port": 8790, "context_delivery": "full"})
        self.assertEqual(Config.from_dict(full_cfg.to_dict()).agent.external.context_delivery, "full")
        with self.assertRaisesRegex(ConfigError, "invalid_config_choice:ExternalConfig.context_delivery"):
            Config.from_dict({"agent": {"external": {"context_delivery": "bogus"}}})
        for port, code in ((-1, "config_below_minimum"), (65536, "config_above_maximum"),
                           ("8790", "invalid_config_type"), (True, "invalid_config_type")):
            with self.subTest(port=port), self.assertRaisesRegex(ConfigError, code + ":ExternalConfig.mcp_port"):
                Config.from_dict({"agent": {"external": {"mcp_port": port}}})
        with self.assertRaisesRegex(ConfigError, "unknown_config_keys:config.agent.external"):
            Config.from_dict({"agent": {"external": {"mcp_auth": "none"}}})

    def test_backend_choice_and_cli(self):
        from ni_rua.apps.gateway import cli as gateway_cli
        from ni_rua.core.config import cli_overrides, namespace_values
        self.assertEqual(Config.from_dict({"agent": {"backend": "external"}}).agent.backend, "external")
        source = (REPO_ROOT / "ni_rua/apps/gateway/cli.py").read_text()
        self.assertIn("choices=('codex', 'claude_code', 'opencode', 'external')", source)
        self.assertEqual(factory.BACKENDS, ("codex", "claude_code", "opencode"))   # hosted backends unchanged
        self.assertEqual(factory.EXTERNAL_BACKEND, "external")
        parsed = vars(gateway_cli.build_parser().parse_args(["--agent-backend", "external", "--external-mcp-port", "9123"]))
        overrides = cli_overrides(parsed)
        self.assertEqual(overrides, {"agent": {"backend": "external", "external": {"mcp_port": 9123}}})
        loaded = Config().with_overrides(overrides)
        self.assertEqual(loaded.agent.external.mcp_port, 9123)
        self.assertEqual(namespace_values(loaded)["external_mcp_port"], 9123)


class LaunchGateTests(unittest.TestCase):
    def assertRefused(self, config, prefix):
        with self.assertRaises(ConfigError) as caught:
            factory.backend_requirements(config)
        self.assertTrue(str(caught.exception).startswith(prefix), str(caught.exception))
        return str(caught.exception)

    def test_interactive_external_launch_passes(self):
        self.assertIsNone(factory.backend_requirements(external()))

    def test_refusals(self):
        cases = (
            ({"run": {"benchmark": True}}, "external_backend_is_interactive_only: remove --benchmark"),
            ({"run": {"exit_after_turn": True}}, "external_backend_is_interactive_only: remove --benchmark"),
            ({"features": {"sandbox_enabled": True}}, "external_backend_cannot_sandbox: NI-RUA does not start the "
                                                     "agent"),
            ({"features": {"visual_review_model": "gpt-6.1-sol"}},
             "external_backend_has_no_model_relay: set features.visual_review_model=null"),
            ({"model": {"price_card": "/tmp/card.json"}},
             "external_backend_has_no_model_relay: set model.price_card=null"),
            ({"features": {"disabled_tools": ["run_begin", "robot_wait"]}},
             "external_bracket_tool_cannot_be_disabled: remove run_begin from features.disabled_tools"),
        )
        for document, prefix in cases:
            with self.subTest(document=document):
                self.assertRefused(external(document), prefix)

    def test_benchmark_only_features_are_dormant_and_not_refused(self):
        # self_briefing, startup_inline_grounding (default true), bounded_continuation, recovery_evidence,
        # default_task_notes and the completion hook act only in hosted/benchmark runs, which are refused or gated above.
        config = external({"features": {"self_briefing": True, "startup_inline_grounding": True,
                                        "bounded_continuation": True, "recovery_evidence": True,
                                        "default_task_notes": True, "completion_guidance": "hook", "terminal": True}})
        self.assertIsNone(factory.backend_requirements(config))
        self.assertFalse(Config().features.self_briefing)
        self.assertTrue(Config().features.startup_inline_grounding)

    def test_a_taken_port_is_refused_before_anything_starts(self):
        holder = socket.socket()
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        self.addCleanup(holder.close)
        port = holder.getsockname()[1]
        message = self.assertRefused(external(port=port), f"external_mcp_port_in_use: 127.0.0.1:{port} is already "
                                                          "in use; another external gateway may be running")
        self.assertIn("set agent.external.mcp_port to a free port", message)
        self.assertEqual(message.count(";"), 1)                 # one sentence
        holder.close()
        self.assertIsNone(factory.backend_requirements(external(port=port)))

    def test_port_zero_is_never_probed(self):
        self.assertIsNone(factory.external_port_problem(0))


class FactoryTests(unittest.TestCase):
    def test_create_session_builds_an_unstarted_external_session(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(Path(directory) / "run", IDENTITY)
            try:
                handler = object()
                session = factory.create_session(external(), Path(directory) / "run", store, handler)
                self.assertIsInstance(session, ExternalSession)
                self.assertEqual((session.state, session.tool_handler, session.mcp_port), ("not_started", handler, 0))
            finally:
                store.close()


class QueueTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        checkout = self.root / "checkout"
        (checkout / "ni_rua/apps/gateway").mkdir(parents=True)
        (checkout / "ni_rua/core").mkdir(parents=True)
        (checkout / "ni_rua/apps/gateway/cli.py").write_text("parser.add_argument('--config')\nConfig.load()\n")
        self.case = dict(cwd=str(checkout), run_dir=str(self.root / "run"), task="stack_bowls", seed=0, layout_id=0,
                         title="agent", tag=[])

    def test_external_cases_are_not_queueable(self):
        for case, config in (({**self.case, "agent_backend": "external"}, None),
                             (self.case, Config.from_dict({"agent": {"backend": "external"}}))):
            with self.subTest(config=config is not None), self.assertRaisesRegex(
                    ValueError, "^external_backend_not_queueable: nobody would attach an agent to a queued case"):
                queue.command(case, 0, config=config)
        self.assertIn("--benchmark", queue.command({**self.case, "agent_backend": "claude_code"}, 0))


class BracketContractTests(unittest.TestCase):
    """The two tools' contracts: every field documented, bounds in prose, roles, effect and ordering."""

    @classmethod
    def setUpClass(cls):
        cls.begin, cls.finish = run_bracket.make_run_bracket_tools(lambda correlation: None,
                                                                   lambda summary, correlation: None)

    def test_registration(self):
        for spec, kind in ((self.begin, "briefing"), (self.finish, "run_finish")):
            with self.subTest(tool=spec.name):
                self.assertEqual((spec.roles, spec.effect, spec.parallel, spec.api_version),
                                 (frozenset({"actor"}), "session", False, "1.0"))
                self.assertEqual(list(spec.output_schemas), [kind])
        registry = ToolRegistry([self.begin, self.finish])
        self.assertEqual(registry.effect_of("run_begin", {}), "session")
        self.assertEqual(registry.effect_of("run_finish", {"summary": "x"}), "session")
        self.assertFalse(registry.parallel_safe("run_begin", {}))
        with self.assertRaisesRegex(TypeError, "run_bracket_callbacks_required"):
            run_bracket.make_run_bracket_tools(None, lambda *args: None)

    def test_inputs(self):
        self.assertEqual(self.begin.definition["inputSchema"],
                         {"type": "object", "properties": {}, "required": [], "additionalProperties": False})
        summary = self.finish.definition["inputSchema"]["properties"]["summary"]
        self.assertEqual((summary["type"], summary["minLength"], summary["maxLength"]), ("string", 1, 16000))
        self.assertEqual(self.finish.definition["inputSchema"]["required"], ["summary"])

    def test_every_bound_is_stated_in_prose(self):
        for spec in (self.begin, self.finish):
            with self.subTest(tool=spec.name):
                self.assertEqual(gaps(spec.definition), [])

    def test_every_field_is_documented_in_the_description(self):
        begin, finish = self.begin.definition["description"], self.finish.definition["description"]
        for field in ("No inputs", "instructions (text)", "task.text (text or null)", "task.source",
                      "operator_prompt", "none (no task was given)", "budget", "native_steps_used",
                      "native_steps_limit", "native_steps_remaining", "per_request_native_limit",
                      "turn.opened (boolean)", "external_briefing_required", "external_turn_closed",
                      "semantics", "source names the accounting", "unknown_values_are_null", "no physical effect",
                      "does not judge task success", "context compaction", "refused with nothing executed"):
            with self.subTest(tool="run_begin", field=field):
                self.assertIn(field, begin)
        for field in ("summary (string, required, 1 to 16000 characters, no default)", "turn_closed (boolean)",
                      "no physical", "external_turn_closed", "judge task success", "never claim success",
                      "does not stop helper programs", "program_status", "decision=reject"):
            with self.subTest(tool="run_finish", field=field):
                self.assertIn(field, finish)
        for description in (begin, finish):
            self.assertNotIn("takeover", description)
            self.assertNotIn("decision_context", description)
            self.assertNotIn("evaluator", description)    # true also on a real arm, where no evaluator exists

    def test_only_the_external_backend_registers_them(self):
        for backend in ("codex", "claude_code", "opencode"):
            with self.subTest(backend=backend):
                config = Config.from_dict({"agent": {"backend": backend}})
                names = create_registry(_configured_tool_specs(config), runtime=config).names
                self.assertFalse({"run_begin", "run_finish"} & names)
        config = external()
        names = create_registry(_configured_tool_specs(config), runtime=config).names
        self.assertTrue({"run_begin", "run_finish"} <= names)

    def test_handlers_need_their_owner(self):
        begin, finish = (spec for spec in _configured_tool_specs(external()) if spec.name.startswith("run_"))
        with self.assertRaisesRegex(RuntimeError, "run_bracket_owner_not_bound"):
            begin.handler(None, {}, {})
        with self.assertRaisesRegex(RuntimeError, "run_bracket_owner_not_bound"):
            finish.handler(None, {"summary": "x"}, {})

    def test_output_schemas_accept_the_session_data(self):
        from ni_rua.runtime.tools.registry import validator_for
        validator_for(run_bracket.BRIEFING_OUTPUT).validate(dict(
            instructions="x", task=dict(text=None, source="none"), budget={}, turn=dict(opened=True)))
        validator_for(run_bracket.BRIEFING_OUTPUT).validate(dict(
            instructions="x", task=dict(text=None, source="none"), budget={}, turn=dict(opened=True),
            startup_grounding=dict(observation_id="obs_1", overview_camera_id="cam_head",
                                   image_kind="raw_rgb", image_count=1, obs3d_included=False)))
        self.assertIn("startup_grounding", self.begin.definition["description"])
        validator_for(run_bracket.RUN_FINISH_OUTPUT).validate(dict(turn_closed=True))
        self.assertFalse(validator_for(run_bracket.RUN_FINISH_OUTPUT).is_valid(dict(turn_closed=False)))
        json.dumps([run_bracket.BRIEFING_OUTPUT, run_bracket.RUN_FINISH_OUTPUT])

# ==== tests/test_agent_external_session.py ====

IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
INSTRUCTIONS = "Robot instructions.\nUse metres and radians."
BRIEFING_REQUIRED = ("external_briefing_required: call run_begin first; it returns the operating instructions and "
                     "the task")
FINISHED = "external_turn_closed: run_finish closed the turn; call run_begin to start the next task"
STOPPED = ("external_turn_closed: the operator stopped the run; motion stays refused until the operator resumes; call "
           "run_begin to re-read the task before continuing")
RESET = ("external_turn_closed: the operator reset the environment; earlier observations and plans no longer describe "
         "it; call run_begin to re-read the task before continuing")
TIMEOUT = 10


def wait_until(predicate, timeout=TIMEOUT):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


def fixture_spec(name, parallel):
    return ToolSpec(dict(type="function", name=name, description=f"Fixture {name}.",
                         inputSchema={"type": "object", "properties": {}, "additionalProperties": True}),
                    lambda context, arguments, correlation: ToolOutput({"tool": name}).response(),
                    roles=frozenset({"actor"}), effect="read" if parallel else "physical", parallel=parallel,
                    output_schemas={"data": {}})


class FakeHarness:
    """``Harness.handle_tool`` stand-in: correlation from the params, the registry invocation, the wire envelope."""

    def __init__(self, store=None):
        self.calls, self.gates = [], {}
        self.session = None
        self.store = store
        begin, finish = make_run_bracket_tools(
            lambda correlation: self.session.run_begin(correlation, budget={"native_steps_used": 0}),
            lambda summary, correlation: self.session.run_finish(summary, correlation))
        self.registry = ToolRegistry([fixture_spec("look", True), fixture_spec("move", False), begin, finish])
        runtime = type("Runtime", (), dict(disabled_tools=(), identity=IDENTITY, latest=None, capabilities=None,
                                          action_budget=lambda self: {}, action=lambda self, i: None))()
        self.runtime = runtime

    def _model_context_epoch(self):
        return (
            self.session.thread_id if self.session is not None else None,
            self.store.read_state("runtime", "image_context_epoch") if self.store is not None else None,
        )

    def tools(self):
        return [spec.definition for spec in self.registry._specs.values()]

    def __call__(self, params):
        self.calls.append(params)
        gate = self.gates.get(params["tool"])
        if gate is not None:
            gate.wait(TIMEOUT)
        correlation = dict(thread_id=params["threadId"], turn_id=params["turnId"], tool_call_id=params["callId"])
        response = self.registry.invoke(params["tool"], params["arguments"], correlation, runtime=self.runtime,
                                        role="actor")
        return wire_response(response)


class SessionCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.store = Store(self.root / "run", IDENTITY)
        self.addCleanup(self.store.close)
        self.resets = []

    def session(self, *, prompt=None, start=True, config=None, port=0, resume=None):
        harness = self.harness = FakeHarness(self.store)
        config = config or Config.from_dict({"agent": {"backend": "external", "external": {"mcp_port": port}}})
        session = ExternalSession(self.root / "run", self.store, harness, config=config)
        session.harness = harness
        harness.session = session
        session.tool_effect = harness.registry.effect_of
        session.tool_parallel_safe = harness.registry.parallel_safe
        session.tool_parallel_hints = {spec.name: spec.parallel for spec in harness.registry._specs.values()}
        session.on_context_reset = self.resets.append
        self.addCleanup(session.close)
        if start:
            session.start(harness.tools(), INSTRUCTIONS, resume)
            session.run_turn(prompt)
        return session

    def client(self, session, **kwargs):
        client = FakeMcpClient(session.bridge.url, protocol_version="2025-11-25", **kwargs)
        client.handshake()
        return client

    @staticmethod
    def text(result):
        return result.result["content"][0]["text"]

    def events(self, kind):
        return [event["payload"] for event in self.store.events_by_type([kind], limit=1000)]


class GateTests(SessionCase):
    def test_calls_before_run_begin_are_refused_without_running_anything(self):
        session = self.session()
        client = self.client(session)
        result = client.call_tool("look", {})
        self.assertTrue(result.result["isError"])
        self.assertIn(BRIEFING_REQUIRED, self.text(result))
        self.assertIn('"minimal_example":{"arguments":{},"tool":"run_begin"}', self.text(result))
        self.assertIn('"actual":{"briefed":false,"turn_open":false}', self.text(result))
        self.assertEqual(self.harness.calls, [])
        self.assertEqual(self.store.states(self.store.TRANSCRIPT_KIND), [])
        self.assertEqual(session.state, "idle")
        # Visible to the operator (review 23 nit 9): one observer event per refused call.
        self.assertEqual(self.events("agent.tool_refused"), [{"code": "external_briefing_required", "tool": "look"}])

    def test_run_begin_opens_the_turn_at_admission_so_its_call_carries_the_turn_id(self):
        session = self.session(prompt="Stack the cubes.")
        client = self.client(session)
        result = client.call_tool("run_begin", {})
        self.assertFalse(result.result["isError"])
        params, = self.harness.calls
        self.assertIsNotNone(params["turnId"])
        self.assertEqual(params["turnId"], session.turn_id)
        self.assertEqual(session.state, "running")
        self.assertEqual(self.store.transcript_item(params["callId"])["turn_id"], params["turnId"])
        started, = [event for event in self.store.events_by_type(["agent.turn_started"], limit=10)]
        self.assertEqual((started["payload"], started["correlation"]["turn_id"]), ({"origin": "external"},
                                                                                   params["turnId"]))
        # Display: the instructions as one plain text item, then task / turn / budget as JSON.
        instructions, rest = result.result["content"]
        self.assertEqual(instructions["text"], INSTRUCTIONS)
        self.assertIn('"task":{"source":"operator_prompt","text":"Stack the cubes."}', rest["text"])
        self.assertIn('"turn":{"opened":true}', rest["text"])
        # 19 S3: the first run_begin of a run reports no context loss.
        self.assertEqual(self.resets, [])

    def test_every_caller_is_served_while_the_turn_is_open(self):
        # B3: the host's subagents carry other session ids; no caller identity is enforced.
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        for meta in ({"claudecode/toolUseId": "toolu_sub"}, {"ai.opencode/sessionID": "child-session"},
                     {"sessionId": "codex-subagent", "threadId": "codex-subagent"}):
            with self.subTest(meta=meta):
                self.assertFalse(client.call_tool("look", {}, meta=meta).result["isError"])
        other = self.client(session, client_name="opencode", client_version="2.2.7")
        self.assertFalse(other.call_tool("move", {}).result["isError"])
        self.assertEqual({params["turnId"] for params in self.harness.calls}, {session.turn_id})

    def test_run_finish_closes_the_turn_and_later_calls_are_refused(self):
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        turn = session.turn_id
        result = client.call_tool("run_finish", {"summary": "Moved 5 mm; obs_3 shows the cube on the plate."})
        self.assertFalse(result.result["isError"])
        self.assertEqual(self.text(result), '{"turn_closed":true}')
        self.assertEqual((session.state, session.turn_result["status"]), ("completed", "completed"))
        refused = client.call_tool("look", {})
        self.assertIn(FINISHED, self.text(refused))
        self.assertEqual(session.turn_id, turn)

    def test_run_finish_summary_is_the_final_answer_with_an_origin_marker(self):
        # 19 N2: replay must not present the summary as native prose.
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        client.call_tool("run_finish", {"summary": "Nothing moved: the gripper is not calibrated."})
        rows = [row for row in self.store.states(self.store.TRANSCRIPT_KIND) if row["type"] == "agentMessage"]
        row, = rows
        self.assertEqual((row["origin"], row["phase"], row["status"], row["turn_id"]),
                         (RUN_FINISH_ORIGIN, "final_answer", "completed", session.turn_id))
        self.assertNotIn("native_id", row)
        self.assertEqual(session.completed_final_answer(), "Nothing moved: the gripper is not calibrated.")

    def test_run_finish_outside_a_turn_records_its_summary_in_a_turn_of_its_own(self):
        session = self.session()
        client = self.client(session)
        self.assertFalse(client.call_tool("run_finish", {"summary": "I cannot start."}).result["isError"])
        self.assertEqual(session.completed_final_answer(), "I cannot start.")
        # That turn was never briefed: the gate still asks for run_begin.
        self.assertIn(BRIEFING_REQUIRED, self.text(client.call_tool("look", {})))

    def test_run_finish_validates_its_summary(self):
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        for arguments in ({}, {"summary": ""}, {"summary": "x" * 16001}, {"summary": 3}):
            with self.subTest(arguments=arguments):
                self.assertTrue(client.call_tool("run_finish", arguments).result["isError"])
        self.assertEqual(session.state, "running")
        self.assertFalse(client.call_tool("run_finish", {"summary": "x" * 16000}).result["isError"])

    def test_a_failing_effect_classifier_takes_the_gated_path(self):
        session = self.session()

        def broken(name, arguments):
            raise RuntimeError("classifier down")
        session.tool_effect = broken
        client = self.client(session)
        self.assertIn(BRIEFING_REQUIRED, self.text(client.call_tool("run_begin", {})))
        self.assertEqual(session.state, "idle")

    def test_calls_after_close_are_stale(self):
        session = self.session()
        client = self.client(session)
        session.closing = True
        self.assertIn("native_tool_call_stale_connection", self.text(client.call_tool("run_begin", {})))


class FailedBracketTests(SessionCase):
    """Review 23 M1: the gate serves a turn only after its briefing, and a failed opening call closes its turn."""

    def assertStillGated(self, session, client):
        self.assertIn(BRIEFING_REQUIRED, self.text(client.call_tool("look", {})))
        self.assertNotIn("look", [params["tool"] for params in self.harness.calls])
        self.assertNotEqual(session.state, "running")

    def test_a_failed_run_finish_on_a_fresh_session_opens_nothing(self):
        session = self.session()
        client = self.client(session)
        self.assertTrue(client.call_tool("run_finish", {}).result["isError"])
        self.assertStillGated(session, client)
        completed, = self.events("agent.turn_completed")
        self.assertEqual(completed, {"status": "interrupted", "error": {"message": "external_bracket_call_failed"}})

    def test_a_run_begin_with_a_spurious_argument_opens_nothing(self):
        session = self.session()
        client = self.client(session)
        self.assertTrue(client.call_tool("run_begin", {"reason": "start"}).result["isError"])
        self.assertStillGated(session, client)
        self.assertFalse(client.call_tool("run_begin", {}).result["isError"])
        self.assertFalse(client.call_tool("look", {}).result["isError"])

    def test_a_failing_bracket_handler_opens_nothing(self):
        session = self.session()
        client = self.client(session)
        original = session.run_begin

        def broken(correlation, budget):
            raise RuntimeError("store unavailable")
        session.run_begin = broken
        self.assertTrue(client.call_tool("run_begin", {}).result["isError"])
        self.assertStillGated(session, client)
        session.run_begin = original

    def test_post_execution_transcript_and_refusal_logging_failures_preserve_contract(self):
        session = self.session()
        client = self.client(session)
        # BracketRefusal when transcript.event("agent.tool_refused") raises still returns refusal cleanly.
        orig_event = session.transcript.event
        session.transcript.event = lambda kind, *a, **kw: (
            (_ for _ in ()).throw(RuntimeError("event_store_failed"))
            if kind == "agent.tool_refused"
            else orig_event(kind, *a, **kw)
        )
        refused = client.call_tool("look", {})
        self.assertTrue(refused.result["isError"])
        self.assertIn(BRIEFING_REQUIRED, self.text(refused))
        session.transcript.event = orig_event

        # Unknown release_turn reason is rejected immediately rather than poisoning _close_reason.
        self.assertFalse(client.call_tool("run_begin", {}).result["isError"])
        with self.assertRaisesRegex(ValueError, "^unknown_turn_close_reason:bogus$"):
            session.release_turn("bogus")
        self.assertEqual(session.state, "running")

        # Post-execution tool_completed failure never discards an executed physical receipt.
        orig_completed = session.transcript.tool_completed
        session.transcript.tool_completed = lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("store_closed"))
        moved = client.call_tool("move", {})
        self.assertFalse(moved.result["isError"])
        self.assertIn('"tool":"move"', self.text(moved))
        session.transcript.tool_completed = orig_completed

    def test_a_failed_bracket_call_inside_a_briefed_turn_keeps_it(self):
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        self.assertTrue(client.call_tool("run_finish", {}).result["isError"])
        self.assertTrue(client.call_tool("run_begin", {"x": 1}).result["isError"])
        self.assertEqual(session.state, "running")
        self.assertFalse(client.call_tool("look", {}).result["isError"])

    def test_nothing_is_admitted_before_the_task_was_set(self):
        # Review 23 S3: a connected agent's call between start() and run_turn() is stale, never briefed without a task.
        session = self.session(start=False)
        session.start(self.harness.tools(), INSTRUCTIONS)
        client = self.client(session)
        self.assertIn("native_tool_call_stale_connection", self.text(client.call_tool("run_begin", {})))
        self.assertEqual(self.harness.calls, [])
        session.run_turn("Pick up the cube.")
        _, rest = client.call_tool("run_begin", {}).result["content"]
        self.assertIn('"text":"Pick up the cube."', rest["text"])


class ClientTests(SessionCase):
    def test_a_cancelled_call_keeps_running_and_is_reported_uncertain(self):
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        self.harness.gates["move"] = gate = threading.Event()
        pending = client.start_call("move", {})
        self.assertTrue(wait_until(lambda: any(params["tool"] == "move" for params in self.harness.calls)))
        client.cancel(pending.request_id)
        self.assertTrue(wait_until(lambda: self.events("agent.tool_reply_unknown")))
        gate.set()
        self.assertTrue(wait_until(lambda: session.bridge._active == 0))
        self.assertEqual([params["tool"] for params in self.harness.calls].count("move"), 1)   # never re-sent
        self.assertEqual(self.events("agent.tool_reply_unknown"), [{"reason": "tool_reply_delivery_uncertain"}])
        self.assertEqual(session.state, "running")
        self.assertFalse(client.call_tool("look", {}).result["isError"])

    def test_two_clients_share_one_turn(self):
        session = self.session()
        first = self.client(session, client_name="claude-code", client_version="2.1.321")
        second = self.client(session, client_name="claude-code-subagent", client_version="2.1.321")
        first.call_tool("run_begin", {})
        _, rest = second.call_tool("run_begin", {}).result["content"]
        self.assertIn('"turn":{"opened":false}', rest["text"])
        self.assertEqual(self.resets, ["external_rebriefing"])
        self.assertFalse(second.call_tool("run_finish", {"summary": "done by the second client"}).result["isError"])
        self.assertIn(FINISHED, self.text(first.call_tool("look", {})))
        self.assertEqual(session.completed_final_answer(), "done by the second client")


class DispatchOrderTests(SessionCase):
    """19 M3: the bracket state is evaluated in dispatch order, not at HTTP arrival."""

    def test_a_read_dispatched_before_run_begin_ran_is_refused_with_nothing_executed(self):
        # Instrumented interleaving: the read reached the dispatcher first and its worker starts only after run_begin's
        # admission opened the turn. The turn is open but not briefed yet, so the read is refused and runs nothing.
        session = self.session()
        client = self.client(session)
        original, release = session._dispatch_turn, threading.Event()

        def delayed(job):
            if job["tool"] == "look":
                release.wait(TIMEOUT)
            return original(job)
        session._dispatch_turn = delayed
        read = client.start_call("look", {})
        self.assertTrue(wait_until(lambda: session.bridge._active == 1))   # the read is admitted and dispatched
        begin = client.start_call("run_begin", {})
        self.assertTrue(wait_until(lambda: session.state == "running"))    # run_begin's admission opened the turn
        release.set()
        read, begin = read.wait(TIMEOUT), begin.wait(TIMEOUT)
        self.assertIn(BRIEFING_REQUIRED, self.text(read))
        # Review 23 fix verification nit: the refusal reports the turn as open (opened by run_begin, not yet briefed).
        self.assertIn('"actual":{"briefed":false,"turn_open":true}', self.text(read))
        self.assertFalse(begin.result["isError"], begin.result)
        self.assertEqual([params["tool"] for params in self.harness.calls], ["run_begin"])
        self.assertFalse(client.call_tool("look", {}).result["isError"])   # repeated after the briefing: served

    def test_concurrent_run_begin_and_read_without_instrumentation(self):
        # Real concurrency, both send orders. The invariant that holds: run_begin is always served; the read is served
        # (after the briefing, in its turn) or refused with nothing executed; every Harness call carries the turn id.
        for first in ("look", "run_begin") * 10:
            with self.subTest(first=first):
                session = self.session()
                client = self.client(session)
                calls = [client.start_call(name, {}) for name in (first, "look" if first == "run_begin" else "run_begin")]
                results = {name: call.wait(TIMEOUT) for name, call in
                           zip((first, "look" if first == "run_begin" else "run_begin"), calls)}
                self.assertFalse(results["run_begin"].result["isError"])
                tools = [params["tool"] for params in self.harness.calls]
                if results["look"].result["isError"]:
                    self.assertIn(BRIEFING_REQUIRED, self.text(results["look"]))
                    self.assertEqual(tools, ["run_begin"])
                else:
                    self.assertEqual(tools, ["run_begin", "look"])
                self.assertEqual({params["turnId"] for params in self.harness.calls}, {session.turn_id})
                session.close()

    def test_run_begin_admitted_before_a_read_serves_both(self):
        session = self.session()
        client = self.client(session)
        self.harness.gates["run_begin"] = gate = threading.Event()
        begin = client.start_call("run_begin", {})
        self.assertTrue(wait_until(lambda: session.state == "running"))
        read = client.start_call("look", {})
        gate.set()
        self.assertFalse(begin.wait(TIMEOUT).result["isError"])
        self.assertFalse(read.wait(TIMEOUT).result["isError"])
        self.assertEqual([params["tool"] for params in self.harness.calls], ["run_begin", "look"])

    def test_a_read_dispatched_after_run_finish_is_refused_although_admitted_in_the_turn(self):
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        self.harness.gates["move"] = gate = threading.Event()
        move = client.start_call("move", {})
        self.assertTrue(wait_until(lambda: any(params["tool"] == "move" for params in self.harness.calls)))
        finish = client.start_call("run_finish", {"summary": "done"})
        self.assertTrue(wait_until(lambda: session.bridge._active == 2))
        read = client.start_call("look", {})
        self.assertTrue(wait_until(lambda: session.bridge._active == 3))
        self.assertEqual(session.state, "running")                         # all three admitted in the open turn
        gate.set()
        self.assertFalse(move.wait(TIMEOUT).result["isError"])
        self.assertFalse(finish.wait(TIMEOUT).result["isError"])
        self.assertIn(FINISHED, self.text(read.wait(TIMEOUT)))

    def test_run_begin_dispatched_after_run_finish_opens_the_next_turn(self):
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        first = session.turn_id
        self.harness.gates["run_finish"] = gate = threading.Event()
        finish = client.start_call("run_finish", {"summary": "first task done"})
        self.assertTrue(wait_until(lambda: session.bridge._active == 1))
        begin = client.start_call("run_begin", {})
        self.assertTrue(wait_until(lambda: session.bridge._active == 2))
        gate.set()
        finish.wait(TIMEOUT)
        result = begin.wait(TIMEOUT)
        self.assertIn('"turn":{"opened":true}', result.result["content"][1]["text"])
        self.assertNotEqual(session.turn_id, first)
        self.assertEqual(session.state, "running")
        self.assertFalse(client.call_tool("look", {}).result["isError"])


class RebriefingTests(SessionCase):
    def test_run_begin_inside_an_open_turn_reports_context_loss_once_per_call(self):
        # 19 S3: no decision_context field; the reset makes the Harness append the reminder to this result.
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        self.assertEqual(self.resets, [])
        again = client.call_tool("run_begin", {})
        self.assertIn('"turn":{"opened":false}', again.result["content"][1]["text"])
        self.assertEqual(self.resets, ["external_rebriefing"])
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual(epoch["source"], "external_rebriefing")
        client.call_tool("run_finish", {"summary": "done"})
        client.call_tool("run_begin", {})   # a new task in a new turn: nothing was lost
        self.assertEqual(self.resets, ["external_rebriefing"])
        # 非首次开回合必须刷新 image_context_epoch 清空旧基线，但不触发 on_context_reset 决策提醒
        new_epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual(new_epoch["source"], "external_new_turn")
        self.assertNotEqual(new_epoch["marker"], epoch["marker"])

    def test_new_non_doctor_client_initialize_refreshes_epoch_without_reset(self):
        session = self.session()
        # doctor 探测不刷新基线
        session._client_initialized({"clientInfo": {"name": "ni-rua-plugin-doctor", "version": "0.2.0"}})
        self.assertIsNone(self.store.read_state("runtime", "image_context_epoch"))
        # 首个真实客户端不刷新基线
        session._client_initialized({"clientInfo": {"name": "claude-code", "version": "2.1.321"}})
        self.assertIsNone(self.store.read_state("runtime", "image_context_epoch"))
        # 真实客户端附着后 doctor 探测仍不刷新基线
        session._client_initialized({"clientInfo": {"name": "ni-rua-plugin-doctor", "version": "0.2.0"}})
        self.assertIsNone(self.store.read_state("runtime", "image_context_epoch"))
        # 第二个真实客户端接入刷新 image_context_epoch，且不触发 on_context_reset
        session._client_initialized({"clientInfo": {"name": "opencode", "version": "2.2.7"}})
        epoch = self.store.read_state("runtime", "image_context_epoch")
        self.assertEqual(epoch["source"], "external_client_attached")
        self.assertEqual(epoch["client_name"], "opencode")
        self.assertEqual(self.resets, [])


class DeltaLeaseTests(SessionCase):
    """非阻塞单主写回租约、OpenCode 子会话 isolated 隔离与 full 模式行为回归测试。"""

    def test_single_main_lease_degrades_overlapping_parallel_call_to_isolated_without_blocking(self):
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        self.assertEqual(self.harness.calls[0]["contextScope"], "main")
        self.harness.calls.clear()

        first_entered = threading.Event()
        release_first = threading.Event()
        orig_call = self.harness.__call__
        call_counter = {"look": 0}

        def gated_handler(params):
            if params["tool"] == "look":
                call_counter["look"] += 1
                idx = call_counter["look"]
                if idx == 1:
                    first_entered.set()
                    release_first.wait(TIMEOUT)
            return orig_call(params)

        session.tool_handler = gated_handler
        first = client.start_call("look", {"seq": 1})
        self.assertTrue(first_entered.wait(TIMEOUT))
        # 首个主链在途调用持有单主租约时，重叠并行调用不阻塞并立即降级 isolated
        second = client.call_tool("look", {"seq": 2})
        self.assertFalse(second.result["isError"])
        release_first.set()
        self.assertFalse(first.wait(TIMEOUT).result["isError"])
        self.assertTrue(wait_until(lambda: session.bridge._lease_call_id is None))
        # 结算释放租约后，下一次主链调用重新拿到 main + contextEpoch
        third = client.call_tool("look", {"seq": 3})
        self.assertFalse(third.result["isError"])

        by_seq = {p["arguments"]["seq"]: p for p in self.harness.calls}
        self.assertEqual(by_seq[1]["contextScope"], "main")
        self.assertIn("contextEpoch", by_seq[1])
        self.assertEqual(by_seq[2]["contextScope"], "isolated")
        self.assertNotIn("contextEpoch", by_seq[2])
        self.assertEqual(by_seq[3]["contextScope"], "main")
        self.assertIn("contextEpoch", by_seq[3])

    def test_opencode_child_session_meta_is_isolated_while_main_session_stays_main(self):
        session = self.session()
        client = self.client(session, client_name="opencode", client_version="2.2.7")
        client.call_tool("run_begin", {}, meta={"ai.opencode/sessionID": "ses-main"})
        self.harness.calls.clear()
        self.assertFalse(client.call_tool("look", {"who": "main"},
                                          meta={"ai.opencode/sessionID": "ses-main"}).result["isError"])
        self.assertFalse(client.call_tool("look", {"who": "child"},
                                          meta={"ai.opencode/sessionID": "ses-child"}).result["isError"])
        by_who = {p["arguments"]["who"]: p for p in self.harness.calls}
        self.assertEqual(by_who["main"]["contextScope"], "main")
        self.assertIn("contextEpoch", by_who["main"])
        self.assertEqual(by_who["child"]["contextScope"], "isolated")
        self.assertNotIn("contextEpoch", by_who["child"])

    def test_full_mode_keeps_run_begin_main_and_regular_tools_isolated(self):
        cfg = Config.from_dict({"agent": {"backend": "external",
                                          "external": {"mcp_port": 0, "context_delivery": "full"}}})
        session = self.session(config=cfg)
        client = self.client(session)
        client.call_tool("run_begin", {})
        client.call_tool("look", {})
        self.assertEqual([p["contextScope"] for p in self.harness.calls], ["main", "isolated"])
        self.assertNotIn("contextEpoch", self.harness.calls[1])

    def test_abandoned_call_does_not_claim_main_lease(self):
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        self.harness.calls.clear()
        job = dict(tool="look", arguments={}, call_id="call-abandoned", native_call_id=None,
                   native_thread_id=None, is_abandoned=lambda: True,
                   thread_id=session.thread_id, turn_id=session.turn_id, bracket=False, started=False)
        session.bridge._dispatched(job)
        self.assertIsNone(session.bridge._lease_call_id)
        self.assertEqual(self.harness.calls[-1]["contextScope"], "isolated")
        self.assertNotIn("contextEpoch", self.harness.calls[-1])


class OperatorTests(SessionCase):
    def test_interrupt_closes_the_open_turn(self):
        session = self.session()
        client = self.client(session)
        session.interrupt()                                         # idle: nothing to close
        self.assertEqual(self.events("agent.turn_closed_by_operator"), [])
        client.call_tool("run_begin", {})
        session.interrupt()
        self.assertEqual((session.state, session.turn_result["status"]), ("interrupted", "interrupted"))
        self.assertEqual(self.events("agent.turn_closed_by_operator"), [{"reason": "operator_stop"}])
        self.assertIn(STOPPED, self.text(client.call_tool("look", {})))

    def test_release_turn_for_a_console_reset(self):
        session = self.session()
        client = self.client(session)
        self.assertFalse(session.release_turn("operator_reset"))
        client.call_tool("run_begin", {})
        self.assertTrue(session.release_turn("operator_reset"))
        self.assertEqual(session.state, "interrupted")
        self.assertEqual(self.events("agent.turn_closed_by_operator"), [{"reason": "operator_reset"}])
        self.assertTrue(session.operator_turn_release)
        self.assertIn(RESET, self.text(client.call_tool("look", {})))


class EndpointTests(SessionCase):
    def test_initialize_carries_the_fixed_instructions_without_a_bearer(self):
        session = self.session()
        self.assertEqual(session.bridge.headers, {})
        result = FakeMcpClient(session.bridge.url).initialize()
        self.assertEqual(result["instructions"], EXTERNAL_MCP_INSTRUCTIONS)
        self.assertEqual(result["serverInfo"]["name"], "ni_rua")
        self.assertEqual(metadata_findings(result["instructions"]), [])

    def test_instructions_tell_the_model_the_bracket_and_that_cancel_does_not_stop_the_arm(self):
        # 19 M8: Esc / cancel in the host agent keeps the admitted action running.
        text = EXTERNAL_MCP_INSTRUCTIONS
        self.assertIn("Call run_begin first, and call it again after any context compaction", text)
        self.assertIn("Tool results are the only source of robot facts.", text)
        self.assertIn("Cancelling a tool call in the agent does not stop robot motion; use the NI-RUA console stop, "
                      "hold or estop.", text)
        self.assertIn("run_finish", text)
        # Spike 21: no host re-lists tools, so a gateway restarted with another robot needs a reconnect.
        self.assertIn("ask the user to reconnect the ni_rua MCP server or start a new session", text)

    def test_client_attached_records_self_reported_client_info(self):
        session = self.session()
        self.assertEqual(session.client_view()["client"], None)
        self.client(session, client_name="claude-code", client_version="2.1.321")
        attached, = self.events("agent.client_attached")
        self.assertEqual(attached, {"client_info": {"name": "claude-code", "version": "2.1.321",
                                                    "self_reported": True},
                                    "protocol_version": "2025-11-25", "via": "http"})
        self.client(session, client_name="x" * 1000, client_version="")
        self.assertEqual(session.client_info, {"name": "x" * CLIENT_FIELD_MAX_CHARS, "version": None,
                                               "self_reported": True})

    def test_client_view_is_the_agent_bar_payload(self):
        session = self.session()
        client = self.client(session, client_name="codex-mcp-client", client_version="0.160.0")
        self.assertIsNone(session.client_view()["last_call_at"])
        client.call_tool("look", {})                                # refused, but still a call the agent made
        view = session.client_view()
        self.assertEqual(set(view), {"endpoint", "client", "last_call_at", "replay"})
        self.assertIs(view["replay"], False)
        self.assertEqual(view["endpoint"], session.bridge.url)
        self.assertEqual(view["client"], {"name": "codex-mcp-client", "version": "0.160.0", "self_reported": True})
        self.assertRegex(view["last_call_at"], r"^\d{4}-\d\d-\d\dT\d\d:\d\d:\d\d\.\d+Z$")

    def test_client_view_marks_a_replay_of_an_archived_run(self):
        session = self.session(start=False)
        session.enter_replay("archived-thread")
        view = session.client_view()
        self.assertIs(view["replay"], True)
        self.assertEqual((view["endpoint"], view["client"], view["last_call_at"]), (None, None, None))

    def test_a_taken_port_fails_the_start_with_one_actionable_sentence(self):
        # 19 M5: one external gateway per port.
        holder = socket.socket()
        holder.bind(("127.0.0.1", 0))
        holder.listen()
        self.addCleanup(holder.close)
        port = holder.getsockname()[1]
        session = self.session(start=False, port=port)
        harness = self.harness
        with self.assertRaises(ValueError) as caught:
            session.start(harness.tools(), INSTRUCTIONS)
        self.assertEqual(str(caught.exception), (
            f"external_mcp_port_in_use: 127.0.0.1:{port} is already in use; another external gateway may be running "
            "(stop it first), or set agent.external.mcp_port to a free port"))
        self.assertEqual(session.state, "closed")


class ContractTests(SessionCase):
    def test_task_comes_only_from_run_turn(self):
        session = self.session(start=False)
        with self.assertRaisesRegex(ValueError, "session_not_connected"):
            session.run_turn("x")
        session.start(self.harness.tools(), INSTRUCTIONS)
        self.assertEqual(session.task, {"text": None, "source": "none"})
        self.assertIsNone(session.run_turn(None))
        self.assertEqual(session.task, {"text": None, "source": "none"})
        self.assertIsNone(session.run_turn("  "))
        self.assertEqual(session.task, {"text": None, "source": "none"})
        self.assertIsNone(session.run_turn("Pick up the cube."))
        self.assertEqual(session.task, {"text": "Pick up the cube.", "source": "operator_prompt"})
        self.assertEqual(session.state, "idle")

    def test_backend_declarations(self):
        session = self.session()
        self.assertEqual((session.backend_kind, session.context_delivery, session.counting, session.model_proxy),
                         ("external", "delta", None, None))
        self.assertEqual(session.owned_processes(), [])
        self.assertFalse(any(value for key, value in session.capabilities.to_dict().items()
                             if key != "compaction_signal"))
        self.assertEqual(session.capabilities.compaction_signal, "none")
        self.assertEqual((session.model, session.provider, session.effort, session.price_card), (None,) * 4)
        for call, code in ((lambda: session.set_model("m"), "set_model_unsupported_by_backend"),
                           (lambda: session.set_effort("high"), "set_effort_unsupported_by_backend"),
                           (lambda: session.evidence_agent(), "evidence_unsupported_by_backend"),
                           (lambda: session.steer("x"), "steer_unsupported_by_backend")):
            with self.subTest(code=code), self.assertRaisesRegex(AgentCapabilityError, code):
                call()
        provenance = session.provenance()
        self.assertEqual((provenance["kind"], provenance["information_boundary"], provenance["model_relay"],
                          provenance["counting"]), ("external", INFORMATION_BOUNDARY, None, None))
        self.assertEqual(provenance["context_delivery"], "delta")
        self.assertEqual(provenance["context_delivery_policy"], {"effective": "delta", "configured": "delta"})
        full_session = self.session(start=False, config=Config.from_dict(
            {"agent": {"backend": "external", "external": {"mcp_port": 0, "context_delivery": "full"}}}))
        self.assertEqual(full_session.context_delivery, "full")
        self.assertEqual(full_session.provenance()["context_delivery_policy"],
                         {"effective": "full", "configured": "full"})
        identity = self.store.read_state("runtime", "agent")
        self.assertEqual((identity["backend"], identity["model"], identity["wire_api"], identity["pid"]),
                         ("external", None, None, None))

    def test_host_tool_output_truncation_limit_by_client_name(self):
        session = self.session()
        self.assertEqual(session.host_tool_output_truncation_limit(), (None, None, False))
        for name, expected in (
            ("claude-code", (500_000, None, True)),
            ("codex-mcp-client", (400_000, None, False)),
            ("opencode", (16_000_000, 100_000, False)),
            ("unknown-client", (None, None, False)),
        ):
            with self.subTest(client=name):
                session._client_initialized({"clientInfo": {"name": name, "version": "1.0.0"}})
                self.assertEqual(session.host_tool_output_truncation_limit(), expected)
        # doctor 探测不覆盖已附着真实 Agent 的截断阈值
        session._client_initialized({"clientInfo": {"name": "claude-code", "version": "2.1.321"}})
        session._client_initialized({"clientInfo": {"name": "ni-rua-plugin-doctor", "version": "0.2.0"}})
        self.assertEqual(session.host_tool_output_truncation_limit(), (500_000, None, True))

    def test_a_price_card_is_refused(self):
        with self.assertRaisesRegex(ValueError, "^external_backend_has_no_model_relay: set model.price_card=null"):
            ExternalSession(self.root / "run", self.store, None, price_card={"rates": []},
                            config=Config.from_dict({"agent": {"backend": "external"}}))

    def test_close_is_clean_and_idempotent(self):
        session = self.session()
        client = self.client(session)
        client.call_tool("run_begin", {})
        client.call_tool("run_finish", {"summary": "done"})
        report = session.close()
        self.assertTrue(report["clean_shutdown"])
        self.assertIs(session.close(), report)
        self.assertTrue(self.store.read_state("runtime", "agent")["clean_shutdown"])

    def test_resume_takes_the_archived_task_and_requires_a_new_briefing(self):
        session = self.session()
        thread = session.thread_id
        session.close()                       # clean
        self.store.set_state("runtime", "agent", dict(self.store.read_state("runtime", "agent"),
                                                      clean_shutdown=False))
        config = Config.from_dict({"agent": {"backend": "external", "external": {"mcp_port": 0}},
                                   "run": {"prompt": "Archived task."}})
        resumed = self.session(start=False, config=config)
        resumed.start(self.harness.tools(), INSTRUCTIONS, thread)
        self.assertEqual(resumed.thread_id, thread)
        self.assertEqual(resumed.task, {"text": "Archived task.", "source": "operator_prompt"})
        self.assertEqual(self.resets, ["agent_resume_after_unclean_exit"])
        client = self.client(resumed)
        self.assertIn(BRIEFING_REQUIRED, self.text(client.call_tool("look", {})))

# ==== tests/test_agent_external_transport.py ====

IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
SCHEMA = {"type": "object", "properties": {"x": {"type": "number", "description": "x in m, default 0"}}}
TOOLS = [{"type": "function", "name": "look", "description": "Read-only view.", "inputSchema": SCHEMA}]


def free_port():
    probe = socket.socket()
    probe.bind(("127.0.0.1", 0))
    try:
        return probe.getsockname()[1]
    finally:
        probe.close()


class EffectOfTests(unittest.TestCase):
    """19 S7: effect callables run on unvalidated arguments at the transport; effect_of must never raise."""

    @classmethod
    def setUpClass(cls):
        cls.registry = full_registry()

    def test_static_and_request_dependent_effects(self):
        self.assertEqual(self.registry.effect_of("robot_observe", {}), "read")
        self.assertEqual(self.registry.effect_of("robot_wait", {}), "physical")
        self.assertEqual(self.registry.effect_of("task_plan", {"operation": "read"}), "read")
        self.assertEqual(self.registry.effect_of("task_plan", {"operation": "create"}), "record")
        self.assertEqual(self.registry.effect_of("robot_action", {"mode": "execute"}), "physical")
        self.assertEqual(self.registry.effect_of("robot_action", {"mode": "preview"}), "plan")
        self.assertEqual(self.registry.effect_of("robot_action", {"auto_execute_if_clean": True}), "physical")

    def test_failing_effect_callables_and_bad_arguments_yield_none(self):
        # action_effect raises KeyError for an unknown mode; memory_effect raises on a non-object request.
        for name, arguments in (("robot_action", {"mode": "teleport"}), ("operation_memory", {"request": "x"}),
                                ("geometry_query", {"request": 3}), ("robot_action", None), ("robot_action", [1]),
                                ("task_plan", "read")):
            with self.subTest(tool=name, arguments=arguments):
                self.assertIsNone(self.registry.effect_of(name, arguments))
        for name in ("no_such_tool", None, 7, ""):
            with self.subTest(tool=name):
                self.assertIsNone(self.registry.effect_of(name, {}))

    def test_non_string_verdicts_and_static_effects_ignore_arguments(self):
        spec = ToolSpec(dict(type="function", name="odd", description="Fixture.", inputSchema={"type": "object"}),
                        lambda context, arguments, correlation: None, effect=lambda arguments: 42,
                        output_schemas={"data": {}})
        registry = ToolRegistry([spec])
        self.assertIsNone(registry.effect_of("odd", {}))
        # A static effect does not depend on the arguments (the call is still validated at execution).
        self.assertEqual(self.registry.effect_of("robot_wait", "garbage"), "physical")

    def test_every_published_actor_tool_is_classified_without_raising(self):
        for name in sorted(self.registry.names):
            with self.subTest(tool=name):
                self.assertIn(self.registry.effect_of(name, {}),
                              (None, "read", "plan", "record", "shared_memory", "physical"))

    def test_no_host_tool_declares_the_session_effect(self):
        registry = create_registry(runtime=Config())
        self.assertNotIn("session", {self.registry.effect_of(name, {}) for name in self.registry.names})
        self.assertNotIn("session", {registry.effect_of(name, {}) for name in registry.names})


class ServerCase(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.store = Store(directory.name, IDENTITY)
        self.addCleanup(self.store.close)
        self.presentation = ActorPresentation(self.store)
        self.calls, self.initialized = [], []

    def server(self, **kwargs):
        def call_tool(name, arguments, **ids):
            self.calls.append(name)
            return {"success": True, "contentItems": [{"type": "inputText", "text": "ok"}]}
        server = NiRuaMcpServer(tools=TOOLS, presentation=self.presentation, call_tool=call_tool,
                                 raw_log=lambda direction, message: None,
                                 on_delivery_failure=lambda call_id, reason: None, **kwargs)
        self.addCleanup(server.close)
        return server.start()


class HostDefaultsTests(ServerCase):
    def test_host_mode_keeps_ephemeral_port_bearer_and_no_instructions(self):
        server = self.server()
        self.assertEqual((server.port, server.require_bearer, server.instructions, server.on_initialize),
                         (0, True, None, None))
        self.assertEqual(set(server.headers), {"Authorization"})
        self.assertTrue(server.headers["Authorization"].startswith("Bearer "))
        self.assertEqual(len(server.token), 43)   # secrets.token_urlsafe(32), unchanged
        self.assertNotEqual(server.server.server_port, 0)
        self.assertNotEqual(server.server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR), 0)
        self.assertEqual(FakeMcpClient(server.url).call_tool("look", {}).status, 401)
        client = FakeMcpClient(server.url, token=server.token)
        self.assertNotIn("instructions", client.initialize())
        self.assertFalse(client.call_tool("look", {}).result["isError"])

    def test_bridge_defaults_are_the_host_mode_ones(self):
        handler = object()
        session = type("Session", (), dict(tool_handler=handler, tool_parallel_safe=None, store=self.store,
                                            presentation=self.presentation))()
        bridge = McpToolBridge(session, TOOLS)
        self.addCleanup(bridge.join)
        self.assertIs(bridge.dispatcher.handler, handler)
        self.assertEqual((bridge.server.port, bridge.server.require_bearer, bridge.server.instructions),
                         (0, True, None))
        own = object()
        other = McpToolBridge(session, TOOLS, handler=own, server_options=dict(require_bearer=False))
        self.addCleanup(other.join)
        self.assertIs(other.dispatcher.handler, own)
        self.assertEqual(other.server.headers, {})


def eventually(condition, timeout_s=10.0):
    """``condition()`` once it holds, or its last value after ``timeout_s`` seconds. The endpoint sends the
    ``initialize`` reply before it calls ``on_initialize`` (``NiRuaMcpServer``), so a client can read the reply
    before the observer has run; under load it does."""
    deadline = time.monotonic() + timeout_s
    while not condition() and time.monotonic() < deadline:
        time.sleep(0.01)
    return condition()


class ExternalOptionsTests(ServerCase):
    def test_fixed_port_and_collision(self):
        port = free_port()
        server = self.server(port=port, require_bearer=False)
        self.assertEqual(server.url, f"http://127.0.0.1:{port}/mcp")
        second = NiRuaMcpServer(tools=TOOLS, presentation=self.presentation, call_tool=None,
                                 raw_log=lambda *args: None, on_delivery_failure=lambda *args: None, port=port)
        with self.assertRaises(OSError) as caught:
            second.start()
        self.assertEqual(caught.exception.errno, errno.EADDRINUSE)
        second.close()   # a failed start leaves nothing to release
        self.assertNotEqual(server.server.socket.getsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR), 0)
        # A served call leaves the server-closed connection in TIME_WAIT; a restarted gateway still rebinds at once.
        self.assertFalse(FakeMcpClient(server.url).call_tool("look", {}).result["isError"])
        server.close()
        again = self.server(port=port, require_bearer=False)
        self.assertFalse(FakeMcpClient(again.url).call_tool("look", {}).result["isError"])

    def test_no_bearer_keeps_the_host_check(self):
        server = self.server(require_bearer=False)
        self.assertEqual((server.headers, server.token), ({}, None))
        self.assertFalse(FakeMcpClient(server.url).call_tool("look", {}).result["isError"])
        # A bearer, right or wrong, is simply not looked at.
        self.assertFalse(FakeMcpClient(server.url, token="anything").call_tool("look", {}).result["isError"])
        port = server.server.server_port
        for host in ("evil.example:%d" % port, "127.0.0.1:1"):
            self.assertEqual(FakeMcpClient(server.url, headers={"Host": host}).call_tool("look", {}).status, 403)
        self.assertEqual(self.calls, ["look", "look"])

    def test_no_bearer_refuses_a_foreign_origin(self):
        # Review 23 M5: a web page's cross-site request (a simple text/plain POST) must not reach the robot tools.
        server = self.server(require_bearer=False)
        port = server.server.server_port
        for origin in ("https://evil.example", "http://127.0.0.1:1", "null", f"http://localhost:{port}.evil"):
            with self.subTest(origin=origin):
                client = FakeMcpClient(server.url, headers={"Origin": origin, "Content-Type": "text/plain"})
                self.assertEqual(client.call_tool("look", {}).status, 403)
                self.assertEqual(client.request("initialize").status, 403)
        self.assertEqual(self.calls, [])
        for origin in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
            with self.subTest(origin=origin):
                result = FakeMcpClient(server.url, headers={"Origin": origin}).call_tool("look", {})
                self.assertFalse(result.result["isError"])
        self.assertFalse(FakeMcpClient(server.url).call_tool("look", {}).result["isError"])   # no Origin: MCP hosts

    def test_host_mode_keeps_its_bearer_path_for_origins(self):
        server = self.server()
        client = FakeMcpClient(server.url, token=server.token, headers={"Origin": "https://evil.example"})
        self.assertFalse(client.call_tool("look", {}).result["isError"])   # unchanged: the bearer is the control
        self.assertEqual(FakeMcpClient(server.url, headers={"Origin": "https://evil.example"}).call_tool(
            "look", {}).status, 401)

    def test_initialize_instructions_and_observer(self):
        def observe(params):
            self.initialized.append(params)
        server = self.server(require_bearer=False, instructions="Call run_begin first.", on_initialize=observe)
        client = FakeMcpClient(server.url, client_name="claude-code", client_version="2.1.321")
        result = client.initialize()
        self.assertEqual(result["instructions"], "Call run_begin first.")
        self.assertEqual(result["serverInfo"]["name"], "ni_rua")
        self.assertTrue(eventually(lambda: self.initialized))
        self.assertEqual(self.initialized[0]["clientInfo"], {"name": "claude-code", "version": "2.1.321"})

    def test_a_failing_observer_never_stops_the_endpoint(self):
        def broken(params):
            raise RuntimeError("boom")
        server = self.server(require_bearer=False, on_initialize=broken)
        client = FakeMcpClient(server.url)
        client.initialize()
        self.assertEqual(eventually(lambda: server.errors), [("on_initialize", "RuntimeError('boom')")])
        self.assertFalse(client.call_tool("look", {}).result["isError"])

    def test_invalid_options_are_refused_at_construction(self):
        for options, code in ((dict(port=-1), "invalid_mcp_port"), (dict(port=65536), "invalid_mcp_port"),
                              (dict(port="8790"), "invalid_mcp_port"), (dict(port=True), "invalid_mcp_port"),
                              (dict(require_bearer=0), "invalid_mcp_require_bearer"),
                              (dict(instructions=""), "invalid_mcp_instructions"),
                              (dict(instructions=3), "invalid_mcp_instructions"),
                              (dict(on_initialize="x"), "invalid_mcp_on_initialize")):
            with self.subTest(options=options), self.assertRaisesRegex(ValueError, "^" + code + "$"):
                NiRuaMcpServer(tools=TOOLS, presentation=self.presentation, call_tool=None,
                                raw_log=lambda *args: None, on_delivery_failure=lambda *args: None, **options)

    def test_concurrent_initialize_calls_each_reach_the_observer(self):
        server = self.server(require_bearer=False, on_initialize=self.initialized.append)
        threads = [threading.Thread(target=FakeMcpClient(server.url).initialize) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join(10)
        self.assertTrue(eventually(lambda: len(self.initialized) == 4), len(self.initialized))

    def test_doctor_probe_does_not_overwrite_attached_real_agent(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(directory) / "run", IDENTITY)
        self.addCleanup(store.close)
        session = ExternalSession(store.root, store, lambda _: {}, config=external(port=0))
        self.addCleanup(session.close)
        session.start([], INSTRUCTIONS)
        # 无真实 Agent 时，doctor 探测正常显示
        session._client_initialized({"clientInfo": {"name": "ni-rua-plugin-doctor", "version": "0.2.0"}})
        self.assertEqual(session.client_view()["client"]["name"], "ni-rua-plugin-doctor")
        # 真实 Agent 附着后，后续 doctor 探测不得覆盖真实 Agent 展示（但审计事件仍正常记录）
        session._client_initialized({"clientInfo": {"name": "claude-code", "version": "2.1.325"}})
        session._client_initialized({"clientInfo": {"name": "ni-rua-plugin-doctor", "version": "0.2.0"}})
        self.assertEqual(session.client_view()["client"]["name"], "claude-code")
        self.assertEqual(session.client_view()["client"]["version"], "2.1.325")
        attached = store.events_by_type(["agent.client_attached"], limit=10)
        self.assertEqual([e["payload"]["client_info"]["name"] for e in attached],
                         ["ni-rua-plugin-doctor", "claude-code", "ni-rua-plugin-doctor"])

if __name__ == "__main__":
    unittest.main()
