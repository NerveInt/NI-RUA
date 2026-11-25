"""Consolidated tests:

- test_completion_guidance: Public completion guidance is auditable and never executes robot actions.
- test_recovery_gateway_integration: Continuation cannot race public terminal or unresolved-execution callbacks.
"""

import asyncio
from contextlib import redirect_stderr
from copy import deepcopy
import io
from itertools import product
import json
from pathlib import Path
import re
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from ni_rua.cortex.supervision.completion_guidance import (
    COMPLETION_GUIDANCE_MODES, HOOK_COMPLETION_GUIDANCE, INITIAL_COMPLETION_GUIDANCE, hook_completion_guidance,
)
from ni_rua.apps.evaluation.episode import BENCHMARK_CONTINUATION, wait_benchmark
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.apps.gateway.cli import main
from ni_rua.runtime.harness import Harness
from ni_rua.runtime.actions.home_check import HomePoseCheck, HOME_CHECK_POLICY
from ni_rua.core.config import Config
from ni_rua.cortex.context.public_task_rules import public_task_rules
from ni_rua.cortex.supervision.recovery_guidance import (
    EVIDENCE_INSTRUCTIONS, EVIDENCE_VERSION, INSTRUCTIONS, VERSION, recovery_guidance,
)
from ni_rua.core.storage.store import Store
from ni_rua.runtime.tools.registry import validator_for
from support.fixture_robot import FixtureRobot
from support.paths import CONFIGS_ROOT, isolate_home
from ni_rua.cortex.session.usage import load_price_card
from scripts.summarize_mixed_usage import summarize_run, report
import base64
import hashlib
from ni_rua.apps.evaluation.native_evidence import (
    AgentRawEvidence, CodexRawEvidence, NativeEvidence, load_native_evidence)
from unittest.mock import Mock, patch
from ni_rua.apps.evaluation.episode import wait_benchmark
from ni_rua.cortex.supervision.recovery_evidence import completed_turn_evidence


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"

# --- test_completion_guidance.py ---
RETIRED_TOOLS = frozenset({
    "robot_move_ee", "robot_move_grasp", "robot_move_dual_grasp",
    "robot_execute_ee", "robot_gripper", "robot_execute_queue",
    "robot_preview_cost", "robot_preview_motion", "robot_execute_plan",
    "robot_inspect_trajectory", "grasp_plan", "grasp_memory",
    "manipulation_run", "skill_run",
    "operation_verify", "depth_move",
})


def measured_state(offset=0):
    return {"frame_id": "env_origin", "tcp_config_id": "arx-x5-link6-v1",
            "sim_step": 0, "step_domain": "adapter_control_step", "arms": {
        side: {"ee_pose": {"position_m": [offset, 0, 1], "quaternion_xyzw": [0, 0, 0, 1],
                           "frame_id": "env_origin", "tcp_config_id": "arx-x5-link6-v1",
                           "source_kind": "measurement"}} for side in ("left", "right")}}


def saved_observation(store, identifier, offset=0):
    return {**store.identity, "schema_version": "0.1", "world_kind": "observed",
            "observation_id": identifier, "sim_step": 0, "step_domain": "adapter_control_step",
            "robot_state_ref": store.put_artifact(json.dumps(measured_state(offset)).encode(),
                                                  "robot_state", "application/json"),
            "task_instruction": "Arrange the visible digits."}


class CompletionGuidanceContractCheck(unittest.TestCase):
    def test_final_home_is_conditional_on_public_task_rules(self):
        for prompt in (INITIAL_COMPLETION_GUIDANCE, HOOK_COMPLETION_GUIDANCE):
            self.assertIn("If the public task rules require return-home", prompt)
            self.assertIn("For tasks without a public return-home rule", prompt)
            self.assertNotIn("Returning home is still unmet", prompt)
        self.assertIn("pose measurement, not an additional task completion condition",
                      HOOK_COMPLETION_GUIDANCE)

    def test_make_kong_does_not_gain_a_final_home_requirement(self):
        rules = public_task_rules("make_kong")
        self.assertEqual(rules["status"], "documented")
        self.assertIsNone(re.search(r"return.*(?:home|origin|initial)|back.*(?:origin|initial)",
                                   " ".join(rules["public_rules"]), re.I))
        reminder = hook_completion_guidance("initial-observation")
        self.assertIn("If the public task rules require return-home", reminder)
        self.assertNotIn("Returning home is still unmet", reminder)
        self.assertIn('"observation_id": "initial-observation"', reminder)


class CompletionGuidanceStartupCheck(unittest.IsolatedAsyncioTestCase):
    def assert_current_public_guidance(self, prompt, definitions, *, has_pose_guidance):
        tools = {item['name']: item for item in definitions}
        for retired in RETIRED_TOOLS:
            self.assertIsNone(re.search(r'\b' + re.escape(retired) + r'\b', prompt))
        for name in re.findall(r'\b(?:robot|history|vision|program)_[a-z_]+\b', prompt):
            if name != 'robot_state':
                self.assertIn(name, tools)
        if has_pose_guidance:
            self.assertIn("kind='move'", prompt)
            for field in ('id', 'arm', 'position_m', 'quaternion_xyzw', 'plan_id'):
                self.assertIn(field, prompt)
            schema = tools['robot_action']['inputSchema']
            pose = measured_state()['arms']['left']['ee_pose']
            validator_for(schema).validate(dict(mode='preview', observation_id='current-observation',
                stages=[dict(id='return_left', kind='move', arm='left',
                             position_m=pose['position_m'], quaternion_xyzw=pose['quaternion_xyzw'])]))
            validator_for(schema).validate(dict(mode='execute', plan_id='returned-cached-plan'))

    async def test_four_conditions_record_actual_initial_and_continuation_prompts(self):
        for requested_mode, recovery_evidence in product((None, *COMPLETION_GUIDANCE_MODES), (False, True)):
            mode = requested_mode or "hook"
            with self.subTest(requested_mode=requested_mode, recovery_evidence=recovery_evidence), \
                    tempfile.TemporaryDirectory() as temporary:
                robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
                options = {} if requested_mode is None else {"completion_guidance": requested_mode}
                config = Config().with_overrides({"features": {"recovery_evidence": recovery_evidence}})
                workbench = Workbench(Path(temporary) / "run", robot=robot, visual_review_model=None,
                                      config=config, **options)
                self.assertEqual(workbench.completion_guidance, mode)
                self.assertEqual(workbench.config.features.recovery_evidence, recovery_evidence)
                self.assertEqual(workbench.harness.home_check is not None, mode in {"hook", "both"})
                initial = saved_observation(workbench.store, "initial-observation")
                home_check = HomePoseCheck(workbench.store) if mode in {"hook", "both"} else None
                if home_check is not None:
                    home_check.capture_initial(initial)
                done = threading.Event()
                done.set()
                harness = SimpleNamespace(
                    accepting=True, active_action=None, states={}, action_count=0,
                    latest=initial, home_check=home_check, capabilities={},
                    action_budget=lambda: dict(native_steps_used=0, native_steps_limit=None, native_steps_remaining=None),
                    episode_done=threading.Event(), episode_terminal=None,
                    native_action_count=0, started_at=time.monotonic(), last_native_progress_at=time.monotonic(),
                    scene=SimpleNamespace(capabilities=lambda: {}), start=Mock(),
                )
                for field in ('tool_registry', 'config', 'disabled_tools', 'observation_profile',
                              'motion_review', 'enable_wait', 'vision_compare', 'visual_reviewer',
                              'action_policy', 'programs'):
                    setattr(harness, field, getattr(workbench.harness, field))
                harness.request_stop = lambda: setattr(harness, "accepting", False)
                session = SimpleNamespace(
                    thread_id="fixture-thread", turn_id="first-turn", state="completed", turn_done=done,
                    model="fixture", provider="fixture", effort="high", price_card=None,
                    cli_executable=Path("/fixture/codex"), command_prefix=["fixture-sandbox"], actor_dir=workbench.root,
                    socket_path=workbench.root / "fixture.sock", env={}, process=SimpleNamespace(pid=-1),
                    start=Mock(return_value="fixture-thread"), run_turn=Mock(), close=Mock(),
                    model_proxy=SimpleNamespace(limit_notified=False, request_count=1),
                )
                workbench.harness, workbench.session = harness, session
                terminal = SimpleNamespace(start=AsyncMock(), close=AsyncMock(),
                    session_id="fixture-terminal", process=SimpleNamespace(pid=-1))
                try:
                    with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                        await workbench.start("unused", benchmark=True, task_notes="Fixture public note.",
                                              task_notes_source="fixture-public")
                    session.run_turn.assert_called_once()
                    initial_prompt = session.run_turn.call_args.args[0]
                    definitions = session.start.call_args.args[0]
                    self.assert_current_public_guidance(initial_prompt, definitions,
                        has_pose_guidance=mode in {'initial', 'both'})
                    self.assertIn("Arrange the visible digits.", initial_prompt)
                    self.assertIn("Fixture public note.", initial_prompt)
                    self.assertEqual(initial_prompt.count(INITIAL_COMPLETION_GUIDANCE),
                                     int(mode in {"initial", "both"}))
                    self.assertNotIn(HOOK_COMPLETION_GUIDANCE, initial_prompt)
                    manifest = json.loads((workbench.root / "manifest.json").read_text())
                    record = manifest["completion_guidance"]
                    expected_recovery_version = EVIDENCE_VERSION if recovery_evidence else VERSION
                    expected_recovery_instructions = EVIDENCE_INSTRUCTIONS if recovery_evidence else INSTRUCTIONS
                    self.assertEqual(record["recovery_guidance"], dict(
                        version=expected_recovery_version, instructions=expected_recovery_instructions,
                        context="current actor observation and native budget; measured home status"))
                    self.assertEqual(record["mode"], mode)
                    self.assertEqual(record["initial_prompt"],
                                     INITIAL_COMPLETION_GUIDANCE if mode in {"initial", "both"} else None)
                    self.assertEqual(record["initial_trigger"],
                                     "initial_actor_prompt" if mode in {"initial", "both"} else None)
                    expected = hook_completion_guidance("initial-observation") if mode in {"hook", "both"} else BENCHMARK_CONTINUATION
                    if home_check is not None:
                        self.assertIsNone(record["continuation_prompt"])
                        self.assertEqual(record["continuation_policy"], HOME_CHECK_POLICY)
                        self.assertEqual(record["continuation_prompts"], {
                            "away": expected, "home": BENCHMARK_CONTINUATION, "unknown": BENCHMARK_CONTINUATION})
                        self.assertEqual(record["home_reference"], home_check.reference_record())
                    else:
                        self.assertEqual(record["continuation_prompt"], expected)
                    self.assertEqual(record["initial_observation_id"], "initial-observation")
                    self.assertNotIn("initial-observation", initial_prompt)
                    self.assertEqual(record["continuation_trigger"], "turn.completed")
                    self.assertEqual(record["continuation_condition"], "actor_turn_completed_before_native_terminal")
                    prompt_records = [value for value in workbench.store.records("configuration") if "prompt" in value]
                    self.assertEqual(len(prompt_records), 1)
                    self.assertEqual(prompt_records[0]["prompt"], initial_prompt)
                    self.assertEqual(prompt_records[0]["completion_guidance"], record)
                    self.assertIn("cortex/supervision/completion_guidance.py", manifest["source_sha256"])
                    self.assertIn("runtime/actions/home_check.py", manifest["source_sha256"])

                    # Later observations cannot replace the saved episode-start reference.
                    harness.latest = saved_observation(workbench.store, "current-observation", .2)
                    if home_check is not None:
                        home_check.update_current(harness.latest)
                    session.run_turn.side_effect = lambda prompt: harness.episode_done.set()
                    self.assertFalse(await wait_benchmark(workbench, asyncio.Event()))
                    self.assertEqual(session.run_turn.call_count, 2)
                    if workbench.config.features.bounded_continuation:
                        evidence = None
                        if recovery_evidence:
                            evidence = dict(version="recovery-evidence-v1", native_action_count=0,
                                native_steps_remaining=None, no_native_finals=0, no_new_evidence_finals=0,
                                no_new_evidence_limit=2, tested_candidate_count=0, tested_candidates=[],
                                earlier_candidate_count=0,
                                scope="nominal candidates at recorded measured state and budget; not global task infeasibility")
                            self.assertEqual(harness.actor_stall["recovery_evidence"], evidence)
                            self.assertEqual(harness.actor_stall["version"], "bounded_completed_final_evidence_v2")
                        else:
                            self.assertNotIn("recovery_evidence", harness.actor_stall)
                            self.assertEqual(harness.actor_stall["version"], "bounded_completed_final_progress_v1")
                        expected_context, recovery = recovery_guidance(harness,
                            home_check.check(workbench.identity) if home_check else None, evidence=evidence)
                        self.assertEqual(expected_context["version"], expected_recovery_version)
                        expected += "\n" + recovery
                    self.assertEqual(session.run_turn.call_args.args, (expected,))
                    self.assert_current_public_guidance(session.run_turn.call_args.args[0], definitions,
                        has_pose_guidance=mode in {'hook', 'both'})
                    self.assertNotIn(INITIAL_COMPLETION_GUIDANCE, expected)
                    event = next(event for event in workbench.store.events()
                                 if event["type"] == "benchmark.actor_turn_continuation")
                    self.assertEqual(event["payload"]["completion_guidance"], mode)
                    self.assertEqual(event["payload"]["prompt"], expected)
                    self.assertEqual(event["payload"]["recovery_context"], expected_context)
                    self.assertEqual(event["payload"]["trigger"], "turn.completed")
                    self.assertEqual(event["payload"]["initial_observation_id"], "initial-observation")
                    self.assertEqual(event["payload"]["continuation_kind"],
                                     "home_reminder" if home_check is not None else "generic")
                    if home_check is not None:
                        self.assertEqual(event["payload"]["home_check"]["status"], "away")
                    robot.execute.assert_not_called()
                    robot.reset.assert_not_called()
                    robot.evaluate.assert_not_called()
                finally:
                    await workbench.close()

    async def test_replay_restores_recorded_mode_without_starting_an_actor(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            workbench = Workbench(root, robot=SimpleNamespace(close=Mock()), completion_guidance="both")
            (root / "manifest.json").write_text(json.dumps({**workbench.identity,
                "completion_guidance": {"mode": "both"}}))
            workbench.store.set_state("runtime", "capabilities", {"data_origin": "synthetic_fixture"})
            await workbench.close()
            replay = Workbench(root, robot=SimpleNamespace(close=Mock()), replay=True)
            replay.session.run_turn = Mock()
            try:
                await replay.start("unused")
                self.assertEqual(replay.completion_guidance, "both")
                replay.session.run_turn.assert_not_called()
            finally:
                await replay.close()


class CompletionGuidanceCLICheck(unittest.TestCase):
    def test_default_and_all_explicit_modes_reach_serve(self):
        for mode in (None, *COMPLETION_GUIDANCE_MODES):
            with self.subTest(mode=mode):
                arguments = ["ni-rua", "--benchmark"]
                if mode is not None:
                    arguments += ["--completion-guidance", mode]
                with patch("sys.argv", arguments), patch("ni_rua.apps.gateway.cli.serve", new_callable=AsyncMock) as serve:
                    main()
                serve.assert_awaited_once()
                self.assertEqual(serve.call_args.args[0].completion_guidance, mode or "hook")

    def test_invalid_mode_and_nonbenchmark_use_are_rejected_before_startup(self):
        for arguments in (["--benchmark", "--completion-guidance", "invalid"],
                          ["--completion-guidance", "initial"], ["--completion-guidance", "both"]):
            with self.subTest(arguments=arguments), patch("sys.argv", ["ni-rua", *arguments]), \
                    patch("ni_rua.apps.gateway.cli.serve", new_callable=AsyncMock) as serve, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                serve.assert_not_called()


class CompletionGuidanceObservationCheck(unittest.TestCase):
    def test_hook_reference_can_recover_undelivered_initial_poses_in_all_payload_profiles(self):
        for profile in ("full_v1", "compact_v1", "lean_v1"):
            with self.subTest(profile=profile), tempfile.TemporaryDirectory() as temporary:
                identity = dict(run_id="fixture", episode_id="episode", env_instance_id="env", env_generation=1)
                store = Store(Path(temporary) / "store", identity)
                robot = FixtureRobot(temporary)
                observe = robot.observe
                def bimanual_observation():
                    raw = observe()
                    raw["robot_state"].update(frame_id="env_origin", tcp_config_id="arx-x5-link6-v1",
                                              sim_step=raw["sim_step"], step_domain=raw["step_domain"])
                    raw["robot_state"]["arms"]["left"]["ee_pose"].update(
                        frame_id="env_origin", tcp_config_id="arx-x5-link6-v1")
                    raw["robot_state"]["arms"]["right"] = deepcopy(raw["robot_state"]["arms"]["left"])
                    raw["robot_state"]["arms"]["right"]["ee_pose"]["position_m"] = [-.009, 0, 1]
                    return raw
                robot.observe = bimanual_observation
                config = Config().with_overrides({"features": {"completion_guidance": "hook"}})
                harness = Harness(store, robot, actor_payload_profile=profile,
                                  observation_profile="rgb_proprio_v1", config=config)
                try:
                    initial = harness.start()
                    self.assertFalse(store.records("delivery"))
                    self.assertEqual(harness.home_check.check(identity)["status"], "home")
                    current = robot.observe()
                    for arm in current["robot_state"]["arms"].values():
                        arm["ee_pose"]["position_m"] = [1, 2, 3]
                        arm["ee_pose"]["quaternion_xyzw"] = [0, 0, 1, 0]
                    current_bundle = harness.publish_observation(current)
                    prompt = hook_completion_guidance(initial["observation_id"])
                    self.assertIn(json.dumps({"observation_id": initial["observation_id"]}), prompt)
                    reply = harness.dispatch("history_view_observation", {"observation_id": initial["observation_id"]},
                                             {"tool_call_id": "history-fixture"})
                    self.assertTrue(reply["success"])
                    recovered = json.loads(reply["contentItems"][0]["text"])["robot_state"]["arms"]
                    self.assertEqual(recovered["left"]["ee_pose"]["position_m"], [.009, 0, 1])
                    self.assertEqual(recovered["right"]["ee_pose"]["position_m"], [-.009, 0, 1])
                    for arm in recovered.values():
                        self.assertEqual(arm["ee_pose"]["quaternion_xyzw"], [0, 0, 0, 1])
                    self.assertEqual(harness.latest["observation_id"], current_bundle["observation_id"])
                    checked = harness.home_check.check(identity)
                    self.assertEqual(checked["status"], "away")
                    self.assertEqual(checked["initial_observation_id"], initial["observation_id"])
                    self.assertEqual(checked["current_observation_id"], current_bundle["observation_id"])
                    self.assertEqual(robot.executions, 0)
                finally:
                    store.close()


class HomeCheckLifecycleCheck(unittest.TestCase):
    def setUp(self):
        temporary_dir = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id="fixture", episode_id="episode", env_instance_id="env", env_generation=1)
        self.store = Store(Path(temporary_dir) / "store", identity)
        self.addCleanup(self.store.close)
        self.robot = FixtureRobot(temporary_dir)
        observe = self.robot.observe
        self.offset = 0

        def bimanual_observation():
            raw = observe()
            raw["robot_state"] = measured_state(self.offset)
            raw["robot_state"]["sim_step"] = raw["sim_step"]
            return raw

        self.robot.observe = bimanual_observation
        config = Config().with_overrides({"features": {"completion_guidance": "hook"}})
        self.harness = Harness(self.store, self.robot, observation_profile="rgb_proprio_v1", config=config)
        self.initial = self.harness.start()
        self.check = lambda: self.harness.home_check.check(identity)

    def action(self):
        return dict(schema_version="0.1", **self.store.identity, world_kind="observed", action_id="move",
                    based_on_observation_id=self.harness.latest["observation_id"], frame_id="env_origin",
                    tcp_config_id="arx-x5-link6-v1", mode="absolute", limits_profile="simulation-slow-v1",
                    timeout_s=10, path=dict(kind="backend_free", orientation="backend_free", blend_radius_m=0,
                    position_tolerance_m=.005, waypoints=[dict(left=dict(position_m=[.3, 0, 1],
                    quaternion_xyzw=[0, 0, 0, 1]))]), gripper_events=[])

    def test_progress_is_not_terminal_evidence_and_fresh_observe_recovers_missing_capture(self):
        for terminal_capture in (True, False):
            with self.subTest(terminal_capture=terminal_capture):
                def execute(action, progress_callback=None, cancel_flag=None):
                    self.assertEqual(self.check()["status"], "unknown")
                    self.robot.executions += 1
                    self.offset = .3
                    progress_callback({"phase": "progress", "observation": self.robot.observe()})
                    self.assertEqual(self.check()["status"], "unknown")
                    return {"status": "completed", **({"terminal_observation": self.robot.observe()}
                                                       if terminal_capture else {})}
                self.robot.execute = execute
                result = self.harness.execute({**self.action(), "action_id": f"move-{terminal_capture}"})
                self.assertEqual(result["receipt"]["status"], "completed")
                self.assertEqual(self.check()["status"], "away" if terminal_capture else "unknown")
                if not terminal_capture:
                    # Progress remains historical evidence; it cannot authorize
                    # motion after the terminal capture failed.
                    self.assertIsNone(self.harness.latest)
                    self.assertIsNone(result["receipt"]["result_observation_id"])
                    self.harness.observe()
                    self.assertEqual(self.check()["status"], "away")
                self.assertEqual(self.check()["initial_observation_id"], self.initial["observation_id"])

    def test_rejected_action_preserves_evidence_but_lost_acknowledgement_invalidates_it(self):
        self.assertEqual(self.check()["status"], "home")
        rejected = self.harness.execute({**self.action(), "action_id": "rejected", "based_on_observation_id": "stale"})
        self.assertEqual(rejected["receipt"]["status"], "rejected")
        self.assertEqual(self.check()["status"], "home")
        self.robot.execute = Mock(side_effect=RuntimeError("lost acknowledgement"))
        result = self.harness.execute(self.action())
        self.assertEqual(result["receipt"]["status"], "unknown")
        self.assertEqual(self.check()["status"], "unknown")
        self.assertFalse(self.harness.accepting)

    def test_explicit_reset_records_new_episode_reference_before_further_actions(self):
        old_identity = dict(self.store.identity)
        self.harness.capabilities["reset"] = True
        def reset(seed, layout_id):
            self.assertEqual(self.check()["status"], "unknown")
            self.offset = .7
            return {"observation": self.robot.observe()}
        self.robot.reset = reset
        fresh = self.harness.reset(seed=1, layout_id=2)
        self.assertEqual(self.check()["status"], "home")
        self.assertEqual(self.check()["initial_observation_id"], fresh["observation_id"])
        self.assertNotEqual(fresh["episode_id"], self.initial["episode_id"])
        self.assertEqual(self.harness.home_check.check(old_identity)["status"], "unknown")
        records = [event for event in self.store.events() if event["type"] == "completion_guidance.home_reference"]
        self.assertEqual(len(records), 2)


# --- test_mixed_usage_report.py ---
class MixedUsageReportTest(unittest.TestCase):
    def test_missing_metadata_is_unknown_and_legacy_transport_success_is_not_failure(self):
        card = load_price_card(CONFIGS_ROOT / "pricing/openai-astra-sol-standard-2026-10-02.json")
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            result = summarize_run(root, card)
            self.assertEqual(result["metadata_missing"], ["manifest.json", "summary.json"])
            self.assertEqual(result["primary_outcome"], "unscored_infrastructure")
            self.assertIsNone(result["accounting"]["total"]["rate_card_estimate_by_currency"]["USD"]["total"])
            (root / "summary.json").write_text(json.dumps(dict(status="failed", valid_episode=True, task_success=False)))
            (root / "private").mkdir()
            metrics = root / "private/model-requests-fixture.jsonl"
            metrics.write_text(json.dumps(dict(kind="model_request_finished", http_status=200, transport_complete=True))+"\n")
            result = summarize_run(root, card)
            self.assertEqual(len(result["entries"]), 1)
            self.assertEqual(result["outcome_classification_basis"], "offline_current_protocol")
            self.assertFalse(result["original_native_result"]["task_success"])
            self.assertIsNone(result["task_success"])
            metrics.write_text(json.dumps(dict(kind="model_request_finished", http_status=200))+"\n")
            result = summarize_run(root, card)
            self.assertEqual(result["entries"][-1]["source"], "unattributed_unknown_attempts")
            metrics.unlink()
            (root / "summary.json").write_text(json.dumps(dict(status="failed",
                model_requests=dict(http_attempt_count=0, completed_calls=0))))
            result = summarize_run(root, card)
            self.assertEqual(result["accounting"]["total"]["rate_card_estimate_by_currency"]["USD"]["total"], "0")

    def test_proxy_ledger_excludes_duplicate_actor_snapshot_and_helper_event(self):
        card = load_price_card(CONFIGS_ROOT / "pricing/openai-astra-sol-standard-2026-10-02.json")
        usage = dict(input_tokens=1000, input_tokens_details=dict(cached_tokens=200, cache_write_tokens=0),
                     output_tokens=100, output_tokens_details=dict(reasoning_tokens=60), total_tokens=1100)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            identity = dict(run_id="r", episode_id="e", env_instance_id="env", env_generation=1)
            (root / "manifest.json").write_text(json.dumps(dict(**identity, model="gpt-6-astra", provider="custom")))
            (root / "summary.json").write_text(json.dumps(dict(status="completed", valid_episode=True,
                task_success=True, scored_episode=True, primary_outcome="scored_success",
                token_usage=dict(total=dict(input_tokens=9999999)))))
            store = Store(root, identity)
            store.event("visual.model_reviewed", dict(model="gpt-6.1-sol", usage=usage, request_id="duplicate"))
            store.close()
            (root / "private").mkdir()
            attempts = [dict(kind="model_request_finished", logical_call_id="logical-"+str(i),
                attempt_index=1, usage=value, requested_model=model, served_model=model, source=source, http_status=200)
                for i, (model, source, value) in enumerate((
                    ("gpt-6-astra", "actor", usage), ("gpt-6.1-sol", "sol_visual_review", usage),
                    ("gpt-6-astra", "actor", None)))]
            (root / "private/model-requests-fixture.jsonl").write_text("\n".join(map(json.dumps, attempts))+"\n")
            result = summarize_run(root, card)
            self.assertEqual(len(result["entries"]), 3)
            costs = result["accounting"]["total"]["rate_card_estimate_by_currency"]["USD"]
            self.assertEqual(costs["known_component_subtotal"], "0.01582")
            self.assertIsNone(costs["total"])
            self.assertEqual(costs["unknown_runs"], 1)
            unscored = {**result, "run_id": "unscored", "scored_episode": False, "task_success": None, "entries": []}
            failed = {**result, "run_id": "failure", "scored_episode": True, "task_success": False, "entries": []}
            aggregate = report([result, failed, unscored])
            self.assertEqual(aggregate["outcomes"], dict(episodes=3, scored_episodes=2, scored_successes=1,
                                                       scored_failures=1, unscored_episodes=1))
            per_success = aggregate["cost_per_scored_success"]["by_currency"]["USD"]
            self.assertEqual(per_success["known_component_cost_per_scored_success"], "0.01582")
            self.assertIsNone(per_success["total_cost_per_scored_success"])

    def test_an_unsettled_summary_gives_way_to_the_archives_later_actor_usage(self):
        # Without a proxy ledger the actor's cumulative usage is read: the summary's copy, unless the summary says the
        # relay had not caught up (model_records_settled: false), then the archive's later runtime/token_usage.
        card = load_price_card(CONFIGS_ROOT / "pricing/openai-astra-sol-standard-2026-10-02.json")
        def usage(output):
            return dict(total=dict(input_tokens=100, cached_input_tokens=0, cache_write_input_tokens=0,
                                   output_tokens=output, reasoning_output_tokens=0, total_tokens=100 + output))
        for flag, expected in (("absent", 10), (True, 10), (False, 12)):
            with self.subTest(model_records_settled=flag), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                identity = dict(run_id="r", episode_id="e", env_instance_id="env", env_generation=1)
                (root / "manifest.json").write_text(json.dumps(dict(**identity, model="gpt-6-astra",
                                                                    provider="custom")))
                summary = dict(status="completed", token_usage=usage(10))
                if flag != "absent":
                    summary["model_records_settled"] = flag
                (root / "summary.json").write_text(json.dumps(summary))
                store = Store(root, identity)
                store.set_state("runtime", "token_usage", usage(12))
                store.close()
                actor, = [entry for entry in summarize_run(root, card)["entries"] if entry["source"] == "actor"]
                self.assertEqual(actor["accounting"]["tokens"]["output_tokens"], expected)


# --- test_native_evidence.py ---
IDENTITY = dict(run_id="run", episode_id="ep", env_instance_id="env", env_generation=1)
T0, T1, T2 = "2026-01-01T00:00:01+00:00", "2026-01-01T00:00:02+00:00", "2026-01-01T00:00:03+00:00"


def write_lines(path, records):
    path.write_text("".join(json.dumps(record) + "\n" for record in records))


def image_b64(payload):
    return base64.b64encode(payload).decode()


class StoreHelpersCheck(unittest.TestCase):
    def test_raw_agent_record_shape_and_channel_validation(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, IDENTITY)
            store.raw_agent("sent", {"command": "interrupt"}, channel="control")
            store.raw_agent("received", {"jsonrpc": "2.0"}, channel="mcp")
            store.raw_agent("received", {"type": "assistant"}, channel="native")
            for channel in ("codex", "", None, "MCP"):
                with self.assertRaisesRegex(ValueError, "invalid_agent_raw_channel"):
                    store.raw_agent("sent", {}, channel=channel)
            with self.assertRaises(TypeError):
                store.raw_agent("sent", {})
            lines = (Path(directory) / "agent-raw.jsonl").read_text().splitlines()
            records = [json.loads(line) for line in lines]
            self.assertEqual([r["channel"] for r in records], ["control", "mcp", "native"])
            self.assertEqual(set(records[0]), {"received_at", "direction", "channel", "message"})
            self.assertEqual(records[0]["message"], {"command": "interrupt"})
            self.assertFalse((Path(directory) / "codex-raw.jsonl").exists())
            store.close()

    def test_raw_agent_requires_writer(self):
        with tempfile.TemporaryDirectory() as directory:
            Store(directory, IDENTITY).close()
            reader = Store(directory, IDENTITY, read_only=True)
            with self.assertRaisesRegex(ValueError, "store_is_read_only"):
                reader.raw_agent("sent", {}, channel="control")
            reader.close()

    def test_transcript_helpers_share_the_codex_item_kind(self):
        with tempfile.TemporaryDirectory() as directory:
            store = Store(directory, IDENTITY)
            self.assertEqual(Store.TRANSCRIPT_KIND, "codex_item")
            self.assertIsNone(store.transcript_item("a"))
            self.assertEqual(store.transcript_item("a", {}), {})
            row = dict(id="a", type="agentMessage", text="hi")
            self.assertEqual(store.set_transcript_item("a", row), row)
            self.assertEqual(store.transcript_item("a"), row)
            self.assertEqual(store.read_state("codex_item", "a"), row)
            store.set_state("codex_item", "b", dict(id="b"))
            self.assertEqual(store.transcript_item("b"), dict(id="b"))
            store.close()


class CodexEvidenceCheck(unittest.TestCase):
    def codex_log(self, root):
        png = b"\x89PNGcodex"
        write_lines(root / "codex-raw.jsonl", [
            dict(received_at=T0, direction="sent", message=dict(method="turn/start", params=dict(
                threadId="thr", input=[dict(type="text", text="go"), dict(type="image", url="x")]))),
            dict(received_at=T0, direction="sent", message=dict(method="turn/start", params=dict(
                threadId="other", input=[dict(type="text", text="ignored")]))),
            dict(received_at=T0, direction="received", message=dict(method="turn/started", params=dict(threadId="thr", turn=dict(id="t1")))),
            dict(received_at=T0, direction="received", message=dict(method="item/started", params=dict(item=dict(id="item-1")))),
            dict(received_at=T0, direction="received", message=dict(id=1, result=dict(contentItems=[
                dict(type="inputImage", imageUrl="data:image/png;base64," + image_b64(png)), dict(type="inputText")]))),
            dict(received_at=T1, direction="sent", message=dict(method="turn/interrupt", params=dict(threadId="thr", turnId="t1"))),
            dict(received_at=T2, direction="received", message=dict(method="turn/completed", params=dict(threadId="thr", turn=dict(
                id="t1", status="interrupted", items=[dict(type="agentMessage", text="NI_RUA_CONTINUITY_OK"), dict(type="reasoning")])))),
            dict(received_at=T2, direction="received", message=dict(method="turn/completed", params=dict(threadId="other", turn=dict(id="x", status="completed")))),
        ])
        return hashlib.sha256(png).hexdigest()

    def test_audit_profile_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            digest = self.codex_log(root)
            evidence = load_native_evidence(root, dict(thread_id="thr"), require_texts=True)
            self.assertEqual(evidence.tool_item_ids, {"item-1"})
            self.assertEqual(evidence.sent_image_sha256, {digest})
            self.assertEqual(evidence.turn_terminals, [dict(id="t1", status="interrupted")])
            self.assertEqual(evidence.completed_turn_texts, ["NI_RUA_CONTINUITY_OK"])
            self.assertEqual([r["turn_id"] for r in evidence.interrupt_requests], ["t1"])
            self.assertIsNotNone(evidence.turn_started_at["t1"])
            self.assertEqual(evidence.turn_completed["t1"][0], "interrupted")
            self.assertEqual(evidence.submitted_prompts, [])
            self.assertEqual(load_native_evidence(root, dict(thread_id="thr")).completed_turn_texts, [])

    def test_report_profile_prompts_and_missing_log(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            missing = load_native_evidence(root, dict(thread_id="thr"), profile="report")
            self.assertFalse(missing.log_present)
            self.assertEqual(missing.submitted_prompts, [])
            self.codex_log(root)
            evidence = load_native_evidence(root, dict(thread_id="thr"), profile="report")
            self.assertTrue(evidence.log_present)
            self.assertEqual(evidence.submitted_prompts, [["go"]])
            self.assertEqual(load_native_evidence(root, dict(), profile="report").submitted_prompts, [])
            with self.assertRaises(FileNotFoundError):
                load_native_evidence(root / "nowhere", dict(thread_id="thr"))

    def test_malformed_records_keep_historical_failures(self):
        cases = [
            ("audit", [dict(direction="sent", message=[1])], "invalid_native_rpc_record"),
            ("audit", [dict(direction="sent", message=dict(params=[]))], "invalid_native_rpc_record"),
            ("audit", [[1]], "invalid_native_rpc_record"),
            ("report", [dict(direction="bogus", message={})], "invalid_native_rpc_record"),
            ("report", [dict(direction="sent", message=dict(params=3))], "invalid_native_rpc_record"),
            ("report", [dict(direction="sent", message=dict(method="turn/start", params=dict(threadId="", input=[])))], "invalid_native_turn_start"),
            ("report", [dict(direction="sent", message=dict(method="turn/start", params=dict(threadId="t", input=[dict(type="text")])))], "invalid_native_turn_start"),
        ]
        for profile, records, message in cases:
            with self.subTest(profile=profile, records=records), tempfile.TemporaryDirectory() as directory:
                write_lines(Path(directory) / "codex-raw.jsonl", records)
                with self.assertRaisesRegex(ValueError, message):
                    load_native_evidence(directory, dict(thread_id="t"), profile=profile)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "codex-raw.jsonl").write_text("{not json\n")
            with self.assertRaises(json.JSONDecodeError):
                load_native_evidence(directory, dict(thread_id="t"), profile="report")

    def test_dispatch(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "codex-raw.jsonl").write_text("")
            (root / "agent-raw.jsonl").write_text("")
            for manifest in (dict(), dict(agent_backend=None), dict(agent_backend=dict(kind="codex"))):
                self.assertIsInstance(load_native_evidence(root, manifest), NativeEvidence)
            self.assertTrue(load_native_evidence(root, dict(agent_backend=dict(kind="claude_code"))).log_present)
            with self.assertRaisesRegex(ValueError, "unsupported_agent_backend"):
                load_native_evidence(root, dict(agent_backend=dict(kind="mystery")))
            with self.assertRaisesRegex(ValueError, "unsupported_native_evidence_profile"):
                load_native_evidence(root, dict(), profile="x")
            self.assertTrue(issubclass(CodexRawEvidence, object) and issubclass(AgentRawEvidence, object))


class AgentEvidenceCheck(unittest.TestCase):
    def build(self, directory):
        store = Store(directory, IDENTITY)
        png = b"\x89PNGagent"
        store.raw_agent("sent", dict(command="submit_prompt", thread_id="thr", turn_id="t1",
                                     input=[dict(type="text", text="first"), dict(type="image")]), channel="control")
        store.raw_agent("sent", dict(command="submit_prompt", thread_id="other", turn_id="z",
                                     input=[dict(type="text", text="ignored")]), channel="control")
        store.raw_agent("received", dict(jsonrpc="2.0", id=7, method="tools/call", params=dict(name="observe", _meta=dict(item_id="item-7"))), channel="mcp")
        store.raw_agent("received", dict(jsonrpc="2.0", id=8, method="tools/call", params=dict(name="plan")), channel="mcp")
        store.raw_agent("sent", dict(jsonrpc="2.0", id=7, result=dict(content=[
            dict(type="text", text="ok"), dict(type="image", data=image_b64(png), mimeType="image/png")])), channel="mcp")
        store.raw_agent("sent", dict(jsonrpc="2.0", id=8, result=dict(content=[dict(type="text", text="ok")])), channel="mcp")
        store.raw_agent("sent", dict(jsonrpc="2.0", id=99, result=dict(content=[])), channel="mcp")  # unmatched
        store.raw_agent("received", dict(type="assistant"), channel="native")
        store.raw_agent("sent", dict(command="interrupt", thread_id="thr", turn_id="t1"), channel="control")
        store.event("agent.turn_started", {}, dict(thread_id="thr", turn_id="t1"), source="agent_backend")
        store.event("agent.turn_completed", dict(status="interrupted"), dict(thread_id="thr", turn_id="t1"), source="agent_backend")
        store.event("agent.turn_completed", dict(status="completed"), dict(thread_id="thr", turn_id="t2"), source="agent_backend")
        store.event("agent.turn_completed", dict(status="completed"), dict(thread_id="other", turn_id="o1"), source="agent_backend")
        store.set_transcript_item("m1", dict(id="m1", thread_id="thr", turn_id="t2", type="agentMessage", text="NI_RUA_CONTINUITY_OK"))
        store.set_transcript_item("m2", dict(id="m2", thread_id="thr", turn_id="t9", type="agentMessage", text="not terminal"))
        store.set_transcript_item("m3", dict(id="m3", thread_id="other", turn_id="o1", type="agentMessage", text="other thread"))
        store.set_transcript_item("c1", dict(id="c1", thread_id="thr", turn_id="t2", type="commandExecution"))
        return store, hashlib.sha256(png).hexdigest()

    def test_extraction(self):
        with tempfile.TemporaryDirectory() as directory:
            store, digest = self.build(directory)
            manifest = dict(thread_id="thr", agent_backend=dict(kind="claude_code"))
            evidence = load_native_evidence(directory, manifest, store, require_texts=True)
            self.assertEqual(evidence.tool_item_ids, {"item-7", "8"})
            self.assertEqual(evidence.sent_image_sha256, {digest})
            self.assertEqual(evidence.submitted_prompts, [["first"]])
            self.assertEqual([(r["turn_id"]) for r in evidence.interrupt_requests], ["t1"])
            self.assertIsNotNone(evidence.interrupt_requests[0]["sent_at"])
            self.assertEqual(evidence.turn_terminals, [dict(id="t1", status="interrupted"), dict(id="t2", status="completed")])
            self.assertEqual(set(evidence.interrupted_completion_seq), {"t1"})
            self.assertIsNotNone(evidence.turn_started_at["t1"])
            self.assertEqual(evidence.turn_completed["t2"][0], "completed")
            self.assertEqual(evidence.completed_turn_texts, ["NI_RUA_CONTINUITY_OK"])
            self.assertEqual(load_native_evidence(directory, manifest, store).completed_turn_texts, [])
            without_store = load_native_evidence(directory, manifest)
            self.assertEqual(without_store.turn_terminals, [])
            self.assertEqual(without_store.submitted_prompts, [["first"]])
            store.close()

    def test_report_profile_tolerates_missing_log_audit_does_not(self):
        with tempfile.TemporaryDirectory() as directory:
            manifest = dict(thread_id="thr", agent_backend=dict(kind="opencode"))
            self.assertFalse(load_native_evidence(directory, manifest, profile="report").log_present)
            with self.assertRaises(FileNotFoundError):
                load_native_evidence(directory, manifest)

    def test_malformed_lines_are_rejected(self):
        bad = [
            dict(direction="sent", channel="mcp", message=[]),
            dict(direction="sideways", channel="mcp", message={}),
            dict(direction="sent", channel="codex", message={}),
            dict(direction="sent", message={}),
            [1],
        ]
        manifest = dict(thread_id="thr", agent_backend=dict(kind="claude_code"))
        for record in bad:
            with self.subTest(record=record), tempfile.TemporaryDirectory() as directory:
                write_lines(Path(directory) / "agent-raw.jsonl", [record])
                with self.assertRaisesRegex(ValueError, "invalid_agent_raw_record"):
                    load_native_evidence(directory, manifest)
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory) / "agent-raw.jsonl").write_text("{broken\n")
            with self.assertRaises(json.JSONDecodeError):
                load_native_evidence(directory, manifest)
        with tempfile.TemporaryDirectory() as directory:
            write_lines(Path(directory) / "agent-raw.jsonl", [dict(direction="sent", channel="control",
                message=dict(command="submit_prompt", thread_id="thr", input="oops"))])
            with self.assertRaisesRegex(ValueError, "invalid_agent_raw_record"):
                load_native_evidence(directory, manifest)


# --- test_recovery_gateway_integration.py ---
class RecoveryGatewayIntegrationTests(unittest.IsolatedAsyncioTestCase):
    def make_workbench(self, root, *, bounded=True, evidence=True):
        store = Store(root, dict(run_id="r", episode_id="e", env_instance_id="env", env_generation=1))
        done = threading.Event()
        done.set()
        session = SimpleNamespace(state="completed", turn_done=done, turn_id="turn-0", thread_id="thread",
            model_limit_reached=threading.Event(), model_proxy=SimpleNamespace(limit_notified=False, request_count=1),
            completed_final_answer=Mock(return_value="I cannot continue."))
        harness = SimpleNamespace(episode_done=threading.Event(), episode_terminal=None, accepting=True,
            native_action_count=99, started_at=time.monotonic(), last_native_progress_at=time.monotonic(),
            active_action=None, wait_stop_reason=None)
        workbench = SimpleNamespace(session=session, harness=harness, completion_guidance="none", store=store,
            config=SimpleNamespace(features=SimpleNamespace(bounded_continuation=bounded, recovery_evidence=evidence)),
            identity=dict(episode_id="e"), current_unknown_actions=lambda: False, has_active_programs=lambda: False,
            control_lock=asyncio.Lock())
        def advance(prompt):
            session.turn_id = "turn-" + str(session.run_turn.call_count)
            session.model_proxy.request_count += 1
            self.assertLess(session.run_turn.call_count, 15)
        session.run_turn = Mock(side_effect=advance)
        return workbench

    async def test_callback_during_baseline_evidence_read_precedes_any_continuation(self):
        for condition in ("terminal", "model_limit", "unknown", "action", "program", "not_accepting", "shutdown"):
            with self.subTest(condition=condition), tempfile.TemporaryDirectory() as root:
                workbench = self.make_workbench(Path(root))
                shutdown = asyncio.Event()
                def evidence(*args, **kwargs):
                    result = completed_turn_evidence(*args, **kwargs)
                    if condition == "terminal": workbench.harness.episode_done.set()
                    elif condition == "model_limit": workbench.session.model_proxy.limit_notified = True
                    elif condition == "unknown": workbench.current_unknown_actions = lambda: True
                    elif condition == "action": workbench.harness.active_action = "inflight"
                    elif condition == "program": workbench.has_active_programs = lambda: True
                    elif condition == "not_accepting": workbench.harness.accepting = False
                    else: shutdown.set()
                    return result
                try:
                    with patch("ni_rua.cortex.supervision.recovery_evidence.completed_turn_evidence", side_effect=evidence):
                        result = await asyncio.wait_for(wait_benchmark(workbench, shutdown), 2)
                    self.assertFalse(result)
                    workbench.session.run_turn.assert_not_called()
                    self.assertNotEqual(workbench.harness.wait_stop_reason, "actor_no_progress")
                finally:
                    workbench.store.close()

    async def test_evidence_off_never_reads_new_evidence_and_retains_three_opportunities(self):
        with tempfile.TemporaryDirectory() as root:
            workbench = self.make_workbench(Path(root), evidence=False)
            try:
                with patch("ni_rua.cortex.supervision.recovery_evidence.completed_turn_evidence", side_effect=AssertionError("disabled")):
                    self.assertFalse(await asyncio.wait_for(wait_benchmark(workbench, asyncio.Event()), 2))
                self.assertEqual(workbench.session.run_turn.call_count, 3)
                self.assertEqual(workbench.harness.actor_stall["version"], "bounded_completed_final_progress_v1")
            finally:
                workbench.store.close()

    async def test_terminal_callback_while_recording_stall_retains_native_priority(self):
        with tempfile.TemporaryDirectory() as root:
            workbench = self.make_workbench(Path(root))
            event = workbench.store.event
            def record(kind, payload, *args, **kwargs):
                result = event(kind, payload, *args, **kwargs)
                if kind == "benchmark.actor_progress" and payload.get("triggered"):
                    workbench.harness.episode_done.set()
                return result
            try:
                with patch.object(workbench.store, "event", side_effect=record):
                    self.assertFalse(await asyncio.wait_for(wait_benchmark(workbench, asyncio.Event()), 2))
                self.assertEqual(workbench.harness.wait_stop_reason, "native_episode_ended")
                self.assertEqual(workbench.session.run_turn.call_count, 2)
            finally:
                workbench.store.close()

    async def test_unknown_callback_while_formatting_guidance_prevents_new_turn(self):
        from ni_rua.cortex.supervision.recovery_guidance import recovery_guidance
        with tempfile.TemporaryDirectory() as root:
            workbench = self.make_workbench(Path(root))
            def guidance(*args, **kwargs):
                result = recovery_guidance(*args, **kwargs)
                workbench.current_unknown_actions = lambda: True
                return result
            try:
                with patch("ni_rua.cortex.supervision.recovery_guidance.recovery_guidance", side_effect=guidance):
                    self.assertFalse(await asyncio.wait_for(wait_benchmark(workbench, asyncio.Event()), 2))
                workbench.session.run_turn.assert_not_called()
                self.assertEqual(workbench.harness.wait_stop_reason, "harness_stop_before_continuation")
            finally:
                workbench.store.close()

    async def test_bounded_off_overrides_evidence_on_and_preserves_legacy_inability_policy(self):
        with tempfile.TemporaryDirectory() as root:
            workbench = self.make_workbench(Path(root), bounded=False)
            try:
                with patch("ni_rua.cortex.supervision.recovery_evidence.completed_turn_evidence", side_effect=AssertionError("legacy")):
                    self.assertFalse(await asyncio.wait_for(wait_benchmark(workbench, asyncio.Event()), 2))
                self.assertEqual(workbench.session.run_turn.call_count, 4)
                self.assertEqual(workbench.harness.wait_stop_reason, "actor_repeated_inability")
            finally:
                workbench.store.close()



if __name__ == "__main__":
    unittest.main()
