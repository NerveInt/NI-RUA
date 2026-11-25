"""
- test_task_instruction_overrides.py: Per-task corrections of the actor-facing task text, without a simulator.
- test_task_notes.py: Exercise actual startup delivery, opt-out and historical prompt preservation.
- test_task_planning.py: CPU-only task-plan admission and persistence checks; no model or robot process.
"""
import json
from pathlib import Path
import re
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import AsyncMock, Mock, patch
from ni_rua.core.config import Config
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.cortex.context.public_task_rules import actor_task_rules_text, public_task_rules
from ni_rua.cortex.context.task_instruction_overrides import (
    CATALOG_PATH, apply_task_instruction_override, task_instruction_override,
)
from support.motion_fixtures import saved_observation
from ni_rua.cortex.context.actor_instructions import OPERATION_NOTES
from contextlib import redirect_stderr
import io
from ni_rua.apps.gateway.cli import main
from ni_rua.core.observation_policy import ACTOR_PROFILE
from ni_rua.cortex.context.task_notes import (DEFAULT_TASK_NOTES, LEGACY_DEFAULT_TASK_NOTES,
                                TASK_NOTES_VERSION, resolve_task_notes, version_for as notes_version_for)
from scripts.benchmark_report import summarize
from copy import deepcopy
from unittest.mock import Mock, patch
from ni_rua.runtime.harness import Harness
from ni_rua.core.observation_policy import LEGACY_ACTOR_PROFILE
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.core.storage.store import Store
from ni_rua.cortex.context.task_planning import TaskPlanning, scheduling_failure_advisory
from support.fixture_robot import FixtureRobot
from support.paths import isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_task_instruction_overrides.py ---

ORGANIZE_NATIVE = ("Place the alarm clock on the drawer, put the figurine on the stand, "
                   "place the mouse on the mouse pad, and push the keyboard into the frame.")
LISTED = ("organize_table", "arrange_largest_number", "arrange_largest_number_random")
THRESHOLD = re.compile(r"\d+(\.\d+)? ?(cm|mm|m|deg|degrees|°)\b|\b\d+ ?x ?\d+\b", re.I)


def delivered(task, native="native"):
    return apply_task_instruction_override(native, public_task_rules(task), task_instruction_override(task))


class OverrideCatalogTest(unittest.TestCase):
    def test_every_entry_targets_existing_documented_sections(self):
        catalog = json.loads(CATALOG_PATH.read_text(encoding="utf-8"))
        tasks = set(catalog["tasks"]) | set(catalog["aliases"])
        self.assertEqual(tasks, set(LISTED))
        for task in tasks:
            with self.subTest(task=task):
                self.assertEqual(task_instruction_override(task)["status"], "applied")
                instruction, text = delivered(task)
                # The native first line is kept and the official scoring table stays verbatim.
                self.assertEqual(instruction, "native")
                self.assertTrue(text.endswith(public_task_rules(task)["public_rules"][-1]))
                # Public sources only: no numeric threshold or dimension beyond the official text.
                official = actor_task_rules_text(public_task_rules(task))
                self.assertEqual(sorted(m.group(0) for m in THRESHOLD.finditer(text)),
                                 sorted(m.group(0) for m in THRESHOLD.finditer(official)))
                self.assertNotIn("Scoring details", text)

    def test_organize_table_drops_drawer_clause(self):
        rules = public_task_rules("organize_table")
        self.assertIn("open the drawer", actor_task_rules_text(rules))
        instruction, text = apply_task_instruction_override(
            ORGANIZE_NATIVE, rules, task_instruction_override("organize_table"))
        self.assertEqual(instruction, ORGANIZE_NATIVE)
        self.assertNotIn("open the drawer", text)
        self.assertNotIn("miscellaneous items inside", text)
        self.assertNotIn("three miscellaneous items", text)
        self.assertIn("The scene contains no miscellaneous items and the drawer cannot be opened", text)
        self.assertTrue(text.startswith(
            "Instruction\n\nPlace the mouse on the mouse pad, push the keyboard into the frame, "
            "put the figurine on the stand, and place the alarm clock on the drawer.\n\nDescription\n\n"))

    def test_arrange_comment_uses_table_axes_for_both_layout_variants(self):
        for task in ("arrange_largest_number", "arrange_largest_number_random"):
            with self.subTest(task=task):
                _, text = delivered(task)
                self.assertNotIn("remain within the permitted rotational tolerance relative to its initial pose", text)
                self.assertIn("Each digit's orientation is judged against the fixed table axes, "
                              "not relative to its initial pose.", text)
                self.assertIn("place all the numbers on the pad", text)
                self.assertNotIn("four numbers", text)

    def test_unlisted_task_and_disabled_flag_are_unchanged(self):
        for task, enabled in (("stack_blocks", True), ("insert_tubes", True), ("cover_blocks", True),
                              ("organize_table", False), ("arrange_largest_number", False)):
            with self.subTest(task=task, enabled=enabled):
                rules = public_task_rules(task)
                record = task_instruction_override(task, enabled)
                self.assertEqual(record["status"], "disabled" if not enabled else "not_listed")
                self.assertNotIn("entry", record)
                self.assertEqual(apply_task_instruction_override("native text", rules, record),
                                 ("native text", actor_task_rules_text(rules)))
        rules = public_task_rules("organize_table")
        self.assertEqual(apply_task_instruction_override("native text", rules, None),
                         ("native text", actor_task_rules_text(rules)))

    def test_instruction_append_and_unknown_section_paths(self):
        record = task_instruction_override("organize_table")
        record["entry"] = {"instruction": "Replaced.", "append_sections": ["Note\n\nExtra."]}
        rules = public_task_rules("insert_tubes")
        self.assertEqual(apply_task_instruction_override("native", rules, record),
                         ("Replaced.", actor_task_rules_text(rules) + "\n\nNote\n\nExtra."))
        record["entry"] = {"replace_sections": {"Comment": "x"}}
        with self.assertRaisesRegex(ValueError, "unknown_section"):
            apply_task_instruction_override("native", rules, record)


class GatewayPromptTest(unittest.IsolatedAsyncioTestCase):
    async def initial_prompt(self, task, native, enabled=None):
        features = {} if enabled is None else {"task_instruction_overrides": enabled}
        config = Config().with_overrides({"run": {"task": task}, "features": features})
        with tempfile.TemporaryDirectory() as temporary:
            robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
            workbench = Workbench(Path(temporary) / "run", robot=robot, visual_review_model=None, config=config)
            initial = {**saved_observation(workbench.store, "initial-observation"), "task_instruction": native}
            done = threading.Event()
            done.set()
            harness = SimpleNamespace(
                accepting=True, active_action=None, states={}, action_count=0,
                latest=initial, home_check=None, capabilities={},
                action_budget=lambda: dict(native_steps_used=0, native_steps_limit=None, native_steps_remaining=None),
                episode_done=threading.Event(), episode_terminal=None,
                native_action_count=0, started_at=time.monotonic(), last_native_progress_at=time.monotonic(),
                scene=SimpleNamespace(capabilities=lambda: {}), start=Mock())
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
                model_proxy=SimpleNamespace(limit_notified=False, request_count=1))
            workbench.harness, workbench.session = harness, session
            terminal = SimpleNamespace(start=AsyncMock(), close=AsyncMock(),
                                       session_id="fixture-terminal", process=SimpleNamespace(pid=-1))
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                    await workbench.start("unused", benchmark=True)
                prompt = session.run_turn.call_args.args[0]
                manifest = json.loads((workbench.root / "manifest.json").read_text())
                records = [value for value in workbench.store.records("configuration") if "prompt" in value]
            finally:
                await workbench.close()
        return prompt, manifest, records[0]

    @staticmethod
    def historic_head(native, rules):
        return (f"Complete the following native RoboDojo task: {native}\n"
                "Use permitted RGB, measured state and tools; inspect actual feedback. "
                "The environment ends execution automatically on its terminal condition. "
                "Do not read simulator files or evaluator state.\n" + actor_task_rules_text(rules)
                + "\nCurrent native execution budget: ")

    async def test_default_on_override_reaches_prompt_manifest_and_record(self):
        prompt, manifest, record = await self.initial_prompt("organize_table", ORGANIZE_NATIVE)
        override = task_instruction_override("organize_table")
        expected_instruction, expected_rules = apply_task_instruction_override(
            ORGANIZE_NATIVE, public_task_rules("organize_table"), override)
        self.assertEqual(expected_instruction, ORGANIZE_NATIVE)
        self.assertTrue(prompt.startswith(
            f"Complete the following native RoboDojo task: {expected_instruction}\n"))
        self.assertIn("\n" + expected_rules + "\nCurrent native execution budget: ", prompt)
        self.assertNotIn("open the drawer", prompt)
        self.assertTrue(manifest["effective_config"]["features"]["task_instruction_overrides"])
        self.assertEqual(manifest["task_instruction_override"], override)
        self.assertIn("cortex/context/task_instruction_overrides.json", manifest["source_sha256"])
        self.assertEqual(record["native_instruction"], ORGANIZE_NATIVE)
        self.assertEqual(record["task_instruction_override"]["status"], "applied")

    async def test_disabled_and_unlisted_tasks_keep_historic_prompt(self):
        cases = (("organize_table", ORGANIZE_NATIVE, False, "disabled"),
                 ("insert_tubes", "Insert the three tubes into the rack one by one.", None, "not_listed"),
                 ("stack_blocks", "Stack the blocks.", None, "not_listed"))
        for task, native, enabled, status in cases:
            with self.subTest(task=task, enabled=enabled):
                prompt, manifest, _ = await self.initial_prompt(task, native, enabled)
                self.assertTrue(prompt.startswith(self.historic_head(native, public_task_rules(task))))
                self.assertEqual(manifest["task_instruction_override"]["status"], status)
                self.assertEqual(manifest["effective_config"]["features"]["task_instruction_overrides"],
                                 enabled is not False)



# --- Folded from tests/test_task_notes.py ---

def fixture_workbench(root, **options):
    robot = SimpleNamespace(close=Mock(), execute=Mock(), reset=Mock(), evaluate=Mock())
    # This process-free fixture exercises prompt archival, not the Sol helper.
    options.setdefault("visual_review_model", None)
    workbench = Workbench(root, robot=robot, **options)
    done = threading.Event()
    done.set()
    harness = SimpleNamespace(
        accepting=True, active_action=None, states={}, action_count=0,
        latest={"task_instruction": "Fixture native task.", "observation_id": "initial",
                "observation_profile": ACTOR_PROFILE},
        capabilities={}, episode_done=threading.Event(), episode_terminal=None,
        native_action_count=0, started_at=time.monotonic(), last_native_progress_at=time.monotonic(),
        scene=SimpleNamespace(capabilities=lambda: {}), start=Mock(),
        action_budget=lambda: dict(native_steps_used=0, native_step_limit=700, native_steps_remaining=700),
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
    return workbench, terminal


class TaskNotesStartupTest(unittest.IsolatedAsyncioTestCase):
    async def test_delivery_and_archive_match_for_defaults_custom_opt_out_and_smoke(self):
        for benchmark, enabled, custom in (
                (True, True, None), (True, True, "Public extra rule."),
                (True, False, None), (True, False, "Public extra rule."),
                (False, True, None)):
            with self.subTest(benchmark=benchmark, enabled=enabled, custom=custom), tempfile.TemporaryDirectory() as temporary:
                workbench, terminal = fixture_workbench(Path(temporary) / "run", default_task_notes=enabled)
                try:
                    with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                        await workbench.start("Smoke instruction.", benchmark=benchmark,
                                              task_notes=custom, task_notes_source="public-fixture" if custom else None)
                    workbench.session.run_turn.assert_called_once()
                    prompt = workbench.session.run_turn.call_args.args[0]
                    self.assertEqual(prompt.count(OPERATION_NOTES), int(benchmark and enabled))
                    if custom:
                        self.assertEqual(prompt.count(custom), 1)
                    if not benchmark:
                        self.assertEqual(prompt, "Smoke instruction.")
                    manifest = json.loads((workbench.root / "manifest.json").read_text())
                    saved = next(value for value in workbench.store.records("configuration") if "prompt" in value)
                    report = summarize(workbench.root)
                    self.assertEqual(saved["prompt"], prompt)
                    for key in ("task_notes", "task_notes_source", "task_notes_config"):
                        self.assertEqual(manifest[key], saved[key])
                        self.assertEqual(report["conditions"][key], manifest[key])
                    self.assertEqual(manifest["task_notes_config"]["default_enabled"], benchmark and enabled)
                    self.assertEqual(manifest["task_notes_config"]["custom_text"], custom)
                    if benchmark and enabled:
                        self.assertEqual(manifest["task_notes_config"]["default_version"], "unified_operation_notes_v7")
                        self.assertIn("unified_operation_notes_v7", manifest["task_notes_source"])
                        self.assertIn("| Situation | Visible check | Adjustment when evidence disagrees |", prompt)
                        self.assertIn("reselect", prompt)
                        for consideration in ("Task, roles or geometry are uncertain", "Grasp, transport or cooperation is uncertain",
                                              "Working arm cannot reach or object needs two arms",
                                              "Placement or release is uncertain", "A motion or expected event fails"):
                            self.assertIn("| " + consideration + " |", prompt)
                        self.assertIn("Avoid unnecessary intermediate home returns", prompt)
                        self.assertIn("deeper contact is not automatically better", prompt)
                        self.assertIn("lift clear of the support and neighbors before lateral transport or folding", prompt)
                        self.assertIn("actual_contact_position_m", prompt)
                    if manifest["task_notes"]:
                        self.assertIn(manifest["task_notes"], prompt)
                        self.assertNotIn(manifest["task_notes_source"], prompt)
                        self.assertTrue(prompt.endswith(manifest["task_notes"]))
                    self.assertIn("cortex/context/task_notes.py", manifest["source_sha256"])
                    for name in ("execute", "reset", "evaluate"):
                        getattr(workbench.robot, name).assert_not_called()
                finally:
                    await workbench.close()

    async def test_resume_preserves_saved_text_even_after_default_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            first, terminal = fixture_workbench(root)
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                    await first.start("unused", benchmark=True)
            finally:
                await first.close()
            manifest = json.loads((root / "manifest.json").read_text())
            manifest["task_notes"] = "Archived operation text."
            manifest["task_notes_source"] = "archived-source"
            manifest["task_notes_config"]["default_version"] = "archived-v0"
            (root / "manifest.json").write_text(json.dumps(manifest))
            resumed, terminal = fixture_workbench(root, resume=True, default_task_notes=False)
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal), \
                        patch("ni_rua.cortex.context.task_notes.DEFAULT_TASK_NOTES", "Changed future text."):
                    await resumed.start("unused")
                resumed.session.run_turn.assert_not_called()
                saved = json.loads((root / "manifest.json").read_text())
                for key in ("task_notes", "task_notes_source", "task_notes_config"):
                    self.assertEqual(saved[key], manifest[key])
            finally:
                await resumed.close()

    async def test_legacy_replay_does_not_enable_defaults_or_rewrite_manifest(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            first, _ = fixture_workbench(root)
            original = json.dumps({**first.identity, "task_notes": "Old custom text.", "task_notes_source": "old-public"})
            (root / "manifest.json").write_text(original)
            first.store.set_state("runtime", "capabilities", {"data_origin": "synthetic_fixture"})
            await first.close()
            replay, _ = fixture_workbench(root, replay=True)
            try:
                await replay.start("unused")
                self.assertFalse(replay.default_task_notes)
                replay.session.run_turn.assert_not_called()
                self.assertEqual(replay.archived_task_notes["text"], "Old custom text.")
                self.assertEqual((root / "manifest.json").read_text(), original)
            finally:
                await replay.close()


class TaskNotesCLITest(unittest.TestCase):
    def test_protocol_ablation_selects_exact_v11_text_and_source(self):
        legacy=resolve_task_notes(manipulation_protocol=False)
        current=resolve_task_notes(manipulation_protocol=True)
        self.assertEqual(legacy["text"],LEGACY_DEFAULT_TASK_NOTES)
        self.assertEqual(legacy["config"]["default_version"],"operation-notes-v11")
        self.assertEqual(notes_version_for(False),"operation-notes-v11")
        self.assertIn("operation-notes-v11",legacy["source"])
        self.assertEqual(current["text"],DEFAULT_TASK_NOTES)
        self.assertEqual(current["config"]["default_version"],TASK_NOTES_VERSION)
        self.assertNotIn("operation_context",legacy["text"])
        self.assertIn("operation_context",current["text"])

    def test_default_and_explicit_switches(self):
        for flag, enabled in ((None, True), ("--default-task-notes", True), ("--no-default-task-notes", False)):
            with self.subTest(flag=flag), patch("sys.argv", ["ni_rua", "--benchmark", *([flag] if flag else [])]), \
                    patch("ni_rua.apps.gateway.cli.serve", new_callable=AsyncMock) as serve:
                main()
                self.assertEqual(serve.call_args.args[0].default_task_notes, enabled)

    def test_incomplete_or_empty_custom_notes_rejected(self):
        for flags in (["--task-notes", "missing source"], ["--task-notes-source", "missing text"],
                      ["--task-notes", " ", "--task-notes-source", "public"],
                      ["--task-notes", "x" * 8001, "--task-notes-source", "public"]):
            with self.subTest(flags=flags[:1]), patch("sys.argv", ["ni_rua", "--benchmark", *flags]), \
                    patch("ni_rua.apps.gateway.cli.serve", new_callable=AsyncMock) as serve, redirect_stderr(io.StringIO()):
                with self.assertRaises(SystemExit) as error:
                    main()
                self.assertEqual(error.exception.code, 2)
                serve.assert_not_called()

    def test_custom_limit_is_independent_of_default_text(self):
        record = resolve_task_notes(True, "x" * 8000, "public")
        self.assertEqual(record["config"]["custom_text"], "x" * 8000)
        self.assertIn(DEFAULT_TASK_NOTES, record["text"])
        self.assertIsNone(resolve_task_notes(False)["text"])



# --- Folded from tests/test_task_planning.py ---

def archived_dict_plan_record(identity, observation_id, plan_id="task-plan-legacy-1", **updates):
    plan_doc = dict(
        objective="Observe the demonstration, then reproduce the visible result.",
        stages=[dict(goal="Observe the unknown sequence.",
                     advance_when="The relevant ordering is visible in chronological frames.")],
        constraints=[],
        uncertainties=["The demonstrated ordering is not yet known."],
        completion_criteria=["The visible arrangement matches the observed demonstration."],
    )
    plan_doc.update(updates)
    return dict(
        plan_id=plan_id,
        version="actor_task_plan_v9",
        identity=dict(identity),
        revision=1,
        previous_plan_id=None,
        observation_id=observation_id,
        plan=plan_doc,
        source_kind="actor_judgment",
        tool_choices="advisory",
    )


def payload(result):
    return json.loads(result["contentItems"][0]["text"])


def attention(**updates):
    return dict(dict(concern="The held object may have shifted.",
        check="Compare its visible alignment with the fingers; adjust the placement approach if shifted.",
        when="After the normal transfer action, before placement."), **updates)


def ledger_entry(**updates):
    return dict(dict(id="retention", expected="The object remains aligned with the fingers after the lift.",
                     status="pending", basis="unknown",
                     next_check="Use the next task-progress frame to compare object and fingers."), **updates)


class PlanningRobot(FixtureRobot):
    def start(self):
        return dict(super().start(), path_kinds=["backend_free", "hold"],
                    bounded_wait=dict(max_native_steps=50), reset=True)

    def reset(self, seed=0, layout_id=0):
        return dict(observation=self.observe())


class TaskPlanningCheck(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.identity = dict(run_id="task-planning", episode_id="initial", env_instance_id="fixture", env_generation=1)
        self.store = Store(self.root / "store", self.identity)
        self.addCleanup(self.store.close)
        self.robot = PlanningRobot(self.root)
        self.config = Config.from_dict(dict(features=dict(completion_guidance="none",
            observation_profile=LEGACY_ACTOR_PROFILE, actor_payload_profile="full_v1")))
        self.harness = Harness(self.store, self.robot, config=self.config, enable_wait=True, task_plan_required=True)
        self.harness.start()
        self.calls = 0

    def dispatch(self, name, arguments=None):
        self.calls += 1
        return self.harness.dispatch(name, arguments or {}, dict(tool_call_id=f"call-{self.calls}"))

    def submit(self, value="Observe the demonstration, then reproduce the visible result.", observation_id=None):
        args = dict(
            operation="submit",
            plan=value,
            ledger=[],
            observation_id=observation_id or self.harness.latest["observation_id"],
        )
        payload(self.dispatch("task_plan", args))
        return self.harness.task_planning.current(self.identity)

    def action_args(self):
        return dict(observation_id=self.harness.latest["observation_id"], native_steps=1)

    def test_robot_action_requires_plan_without_consuming_budget(self):
        with patch.object(self.harness, "invalidate_scene") as invalidate:
            result = payload(self.dispatch("robot_wait", self.action_args()))
        self.assertEqual(result["receipt"]["status"], "rejected")
        self.assertIn("task_plan_required", result["receipt"]["reason"])
        invalidate.assert_not_called()
        self.assertEqual(self.robot.executions, 0)
        self.assertEqual(self.harness.action_count, 0)
        self.submit()
        result = payload(self.dispatch("robot_wait", self.action_args()))
        self.assertEqual(result["receipt"]["status"], "completed")
        self.assertEqual(self.robot.executions, 1)

    def test_queries_allowed_and_do_not_unlock_actions(self):
        read = payload(self.dispatch("task_plan", dict(operation="read")))
        self.assertTrue(read["required"])
        self.assertIsNone(read["plan"])
        self.assertEqual(read["tool_choices"], "advisory")
        self.assertTrue(self.dispatch("robot_observe")["success"])
        self.assertTrue(self.dispatch("robot_capabilities")["success"])
        self.assertTrue(self.dispatch("history_view_observation", dict(
            observation_id=self.harness.latest["observation_id"], camera_id="fixture_camera", roi=[0, 0, 1, 1]))["success"])
        self.assertIsNone(self.harness.task_planning.current(self.identity))
        self.assertEqual(self.robot.executions, 0)

    def test_revision_and_tool_suggestions_are_advisory(self):
        first = self.submit("Observe the demonstration before moving.")
        original = deepcopy(first)
        second = self.submit("Move after reviewing evidence; optional tools remain advisory.")
        self.assertEqual(second["revision"], 2)
        self.assertEqual(second["previous_plan_id"], first["plan_id"])
        self.assertEqual(self.store.read("task_plan", first["plan_id"]), original)
        self.assertEqual(payload(self.dispatch("task_plan", dict(operation="read")))["plan"], second["plan"])
        moved = payload(self.dispatch("robot_wait", self.action_args()))
        self.assertEqual(moved["receipt"]["status"], "completed")
        self.assertEqual(self.harness.task_planning.current(self.identity)["plan_id"], second["plan_id"])

    def test_freeform_experience_choices_can_change_without_action_binding(self):
        initial = "Check uncertain geometry with a side view before choosing a grasp."
        first = payload(self.dispatch("task_plan", dict(plan=initial, ledger=[])))
        self.assertEqual(first, dict(plan_id="P1", status="saved"))
        # A provisional consideration does not restrict subsequent tool choices.
        moved = payload(self.dispatch("robot_wait", self.action_args()))
        self.assertEqual(moved["receipt"]["status"], "completed")
        self.assertEqual(self.harness.task_planning.current(self.identity)["plan_id"], "P1")

        revised = "The target stayed at its source; reconsider the moving object and test retention."
        second = payload(self.dispatch("task_plan", dict(operation="update", plan=revised)))
        self.assertEqual(second, dict(plan_id="P2", status="saved"))
        self.assertEqual(self.store.read("task_plan", "P1")["plan"], initial)
        self.assertEqual(payload(self.dispatch("task_plan", dict(operation="read")))["plan"], revised)
        record = self.harness.task_planning.current(self.identity)
        self.assertEqual((record["revision"], record["previous_plan_id"]), (2, "P1"))
        self.assertEqual(record["observation_id"], self.harness.latest["observation_id"])
        self.assertEqual(payload(self.dispatch("robot_wait", self.action_args()))["receipt"]["status"],
                         "completed")

    def test_attention_partial_update_preserves_plan_and_replaces_or_clears_list(self):
        def call(role, arguments):
            return self.harness.invoke("task_plan", arguments, {}, role=role).data
        first = call("actor", dict(plan="Understand the target; transfer it and verify support.", ledger=[]))
        self.assertEqual(first, dict(plan_id="P1", status="saved"))
        self.assertNotIn("attention", call("program", dict(operation="read")))
        self.assertEqual(self.harness.task_planning.decision_context(self.identity)["attention"], [])
        self.harness.observe()
        selected = [attention()]
        second = call("program", dict(operation="update", attention=selected))
        self.assertEqual(second, dict(plan_id="P2", status="saved"))
        saved = self.store.read("task_plan", "P2")
        selected[0]["concern"] = "Caller mutation must not change saved intent."
        self.assertEqual(saved["attention"], [attention()])
        self.assertEqual(saved["plan"], self.store.read("task_plan", "P1")["plan"])
        self.assertEqual((saved["revision"], saved["previous_plan_id"]), (2, "P1"))
        self.assertEqual(saved["observation_id"], self.harness.latest["observation_id"])
        third = call("actor", dict(operation="update", plan="Use the changed approach, then verify support."))
        self.assertEqual(self.store.read("task_plan", third["plan_id"])["attention"], [attention()])
        replacement = [attention(concern="Release might be hidden by the support rim.")]
        call("program", dict(operation="update", attention=replacement))
        self.assertEqual(call("actor", dict(operation="read"))["attention"], replacement)
        cleared = call("actor", dict(operation="update", attention=[]))
        self.assertEqual(cleared, dict(plan_id="P5", status="saved"))
        self.assertEqual(call("program", dict(operation="read"))["attention"], [])
        self.assertEqual(self.store.read("task_plan", "P2"), saved)
        self.assertEqual(self.robot.executions, 0)

    def test_ledger_is_independent_replaceable_and_advisory(self):
        selected = [ledger_entry()]
        first = self.dispatch("task_plan", dict(plan="Transfer and verify support.", attention=[attention()], ledger=selected))
        self.assertEqual(payload(first)["status"], "saved")
        selected[0]["basis"] = "caller mutation"
        saved = self.harness.task_planning.current(self.identity)
        self.assertEqual(saved["ledger"], [ledger_entry()])
        self.assertEqual(self.harness.task_planning.decision_context(self.identity)["ledger"], [ledger_entry()])
        self.assertEqual(payload(self.dispatch("task_plan", dict(operation="read")))["ledger"], [ledger_entry()])

        # A pending row remains advisory: normal task progress is still admitted.
        moved = payload(self.dispatch("robot_wait", self.action_args()))
        self.assertEqual(moved["receipt"]["status"], "completed")

        self.harness.observe()
        supported = ledger_entry(status="supported", basis="RGB shows the object moving with the fingers.")
        updated = payload(self.dispatch("task_plan", dict(operation="update", ledger=[supported])))
        self.assertEqual(updated["status"], "saved")
        record = self.harness.task_planning.current(self.identity)
        self.assertEqual(record["attention"], [attention()])
        self.assertEqual(record["ledger"], [supported])
        self.assertEqual(record["ledger_observation_id"], self.harness.latest["observation_id"])

        self.harness.observe()
        self.dispatch("task_plan", dict(operation="update", ledger=[]))
        self.assertEqual(self.harness.task_planning.decision_context(self.identity)["ledger"], [])

    def test_repeated_update_same_observation_is_a_noop_but_new_observation_is_saved(self):
        self.dispatch("task_plan", dict(plan="Inspect and verify the transfer.", ledger=[]))
        before_records = self.store.records("task_plan")
        before_events = self.store.events_by_type(["task_plan.submitted"])
        repeated = self.dispatch("task_plan", dict(operation="update",
            plan="Inspect and verify the transfer."))
        self.assertEqual(payload(repeated), dict(plan_id="P1", status="unchanged"))
        self.assertEqual(self.store.records("task_plan"), before_records)
        self.assertEqual(self.store.events_by_type(["task_plan.submitted"]), before_events)

        self.harness.observe()
        reconsidered = self.dispatch("task_plan", dict(operation="update",
            plan="Inspect and verify the transfer."))
        self.assertEqual(payload(reconsidered), dict(plan_id="P2", status="saved"))
        self.assertEqual(self.harness.task_planning.current(self.identity)["plan_observation_id"],
                         self.harness.latest["observation_id"])

    def test_initial_freeform_submit_requires_explicit_ledger(self):
        with self.assertRaisesRegex(ValueError, "task_plan_submit_requires_ledger"):
            self.dispatch("task_plan", dict(plan="Observe and verify the transfer."))
        self.assertIsNone(self.harness.task_planning.current(self.identity))
        self.dispatch("task_plan", dict(plan="Observe and verify the transfer.", ledger=[]))
        self.assertEqual(self.harness.task_planning.current(self.identity)["ledger"], [])

    def test_invalid_ledger_updates_are_atomic_and_old_records_project_empty_ledger(self):
        self.dispatch("task_plan", dict(plan="Inspect and transfer.", ledger=[ledger_entry()]))
        before = self.harness.task_planning.current(self.identity)
        records = self.store.records("task_plan")
        events = self.store.events_by_type(["task_plan.submitted"])
        invalid = [None, [{}], [ledger_entry(status="bad")],
                   [ledger_entry(), ledger_entry(id="retention")],
                   [ledger_entry(extra="unexpected")], [ledger_entry(expected=" ")],
                   [ledger_entry()] * 9]
        for value in invalid:
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.dispatch("task_plan", dict(operation="update", ledger=value))
            self.assertEqual(self.store.records("task_plan"), records)
            self.assertEqual(self.harness.task_planning.current(self.identity), before)
            self.assertEqual(self.store.events_by_type(["task_plan.submitted"]), events)

        old = dict(before)
        old.pop("ledger")
        old.pop("ledger_observation_id")
        old["plan_id"] = "P-old"
        old["revision"] = 99
        self.store.save("task_plan", old["plan_id"], old)
        self.store.set_state("runtime", "task_plan", old)
        self.assertEqual(self.harness.task_planning.decision_context(self.identity)["ledger"], [])
        self.assertEqual(payload(self.dispatch("task_plan", dict(operation="read")))["ledger"], [])

    def test_invalid_attention_updates_leave_all_saved_planning_state_unchanged(self):
        long_valid = attention(check="证" * 1600)
        self.dispatch("task_plan", dict(plan="Inspect the first transfer.", attention=[long_valid], ledger=[]))
        before = self.harness.task_planning.current(self.identity)
        records = self.store.records("task_plan")
        events = self.store.events_by_type(["task_plan.submitted"])
        invalid = [None, {}, "concern", [None], [{}], [dict(concern="x", check="y")],
                   [dict(attention(), extra="unexpected")]]
        invalid += [[attention(**{field: value})] for field in ("concern", "check", "when")
                    for value in (" ", "x" * 1601, False)]
        requests = [dict(operation="update", plan="Must not be partially saved.", attention=value)
                    for value in invalid]
        requests += [dict(operation="update"), dict(attention=[]),
                     dict(operation="update", plan=" ", attention=[])]
        for arguments in requests:
            with self.subTest(arguments=arguments), self.assertRaises(ValueError):
                self.dispatch("task_plan", arguments)
            self.assertEqual(self.store.records("task_plan"), records)
            self.assertEqual(self.harness.task_planning.current(self.identity), before)
            self.assertEqual(self.store.events_by_type(["task_plan.submitted"]), events)
        self.assertEqual(before["attention"], [long_valid])

    def test_partial_updates_preserve_each_fields_observation_source(self):
        owner = self.harness.task_planning

        def check_sources(plan_observation, attention_observation):
            saved = owner.current(self.identity)
            context = owner.decision_context(self.identity)
            read = self.harness.invoke("task_plan", dict(operation="read"), {}, role="program").data
            for value in (saved, context, read):
                self.assertEqual(value["plan_observation_id"], plan_observation)
                self.assertEqual(value["attention_observation_id"], attention_observation)
                if value is not saved or "ledger_observation_id" in value:
                    self.assertIn("ledger_observation_id", value)
                self.assertNotIn("based_on_observation_id", value)
            self.assertEqual(saved["observation_id"], self.harness.latest["observation_id"])

        initial = self.harness.latest["observation_id"]
        self.dispatch("task_plan", dict(plan="Transfer then verify support.", attention=[attention()], ledger=[]))
        check_sources(initial, initial)
        self.harness.observe()
        attention_observation = self.harness.latest["observation_id"]
        self.dispatch("task_plan", dict(operation="update", attention=[attention(concern="Check the new alignment.")]))
        check_sources(initial, attention_observation)
        self.assertEqual(owner.current(self.identity)["plan"], "Transfer then verify support.")

        self.harness.observe()
        self.dispatch("task_plan", dict(operation="update", plan="Inspect the changed support first."))
        check_sources(self.harness.latest["observation_id"], attention_observation)
        # Explicitly resubmitting the same text records reconsideration at the new observation.
        self.harness.observe()
        plan_observation = self.harness.latest["observation_id"]
        self.dispatch("task_plan", dict(operation="update", plan=owner.current(self.identity)["plan"]))
        check_sources(plan_observation, attention_observation)

        self.harness.observe()
        self.dispatch("task_plan", dict(operation="update", attention=[]))
        check_sources(plan_observation, self.harness.latest["observation_id"])
        self.assertEqual(owner.decision_context(self.identity)["attention"], [])
        self.harness.observe()
        self.dispatch("task_plan", dict(plan="A new full submission without concerns.", ledger=[]))
        check_sources(self.harness.latest["observation_id"], None)
        self.assertNotIn("attention", owner.current(self.identity))
        self.assertEqual(owner.decision_context(self.identity)["attention"], [])

    def test_old_records_keep_unknown_field_sources_until_explicitly_updated(self):
        owner = self.harness.task_planning
        # Old records have only the observation at save time; it cannot identify
        # which retained field was reconsidered, even when its text is unchanged.
        for revision in (1, 2):
            self.harness.observe()
            record = dict(plan_id=f"P{revision}", version="freeform_task_plan", identity=self.identity,
                revision=revision, previous_plan_id="P1" if revision == 2 else None,
                observation_id=self.harness.latest["observation_id"], plan="Retained old plan.",
                attention=[attention()], source_kind="actor_judgment", tool_choices="advisory")
            self.store.save("task_plan", record["plan_id"], record)
            self.store.set_state("runtime", "task_plan", record)
            for value in (owner.decision_context(self.identity),
                          payload(self.dispatch("task_plan", dict(operation="read")))):
                self.assertEqual(value["plan"], record["plan"])
                self.assertEqual(value["attention"], record["attention"])
                self.assertIsNone(value["plan_observation_id"])
                self.assertIsNone(value["attention_observation_id"])
            self.assertEqual(owner.current(self.identity), record)

        self.harness.observe()
        updated_attention = self.harness.latest["observation_id"]
        self.dispatch("task_plan", dict(operation="update", attention=[attention()]))
        self.assertIsNone(owner.current(self.identity)["plan_observation_id"])
        self.assertEqual(owner.current(self.identity)["attention_observation_id"], updated_attention)
        self.harness.observe()
        self.dispatch("task_plan", dict(operation="update", plan="Reconsidered plan."))
        self.assertEqual(owner.current(self.identity)["plan_observation_id"], self.harness.latest["observation_id"])
        self.assertEqual(owner.current(self.identity)["attention_observation_id"], updated_attention)
        self.assertEqual(self.store.read("task_plan", "P2"), record)

    def test_attention_context_uses_persisted_owner_and_isolates_episode(self):
        self.dispatch("task_plan", dict(plan="Verify support before withdrawing.", attention=[attention()], ledger=[]))
        owner = TaskPlanning(self.store)
        original = owner.current(self.identity)
        expected = dict(plan_id="P1", revision=1, plan=original["plan"], attention=[attention()],
                        ledger=[],
                        plan_observation_id=self.harness.latest["observation_id"],
                        attention_observation_id=self.harness.latest["observation_id"],
                        ledger_observation_id=self.harness.latest["observation_id"])
        self.assertEqual(owner.decision_context(self.identity), expected)
        projected = owner.decision_context(self.identity)
        projected["attention"][0]["check"] = "Do not mutate the owner."
        self.assertEqual(owner.decision_context(self.identity), expected)
        self.store.set_state("runtime", "task_plan", dict(original, attention=[]))
        self.assertIsNone(owner.decision_context(self.identity))
        with self.assertRaisesRegex(ValueError, "task_plan_required"):
            owner.require(self.identity)
        self.store.set_state("runtime", "task_plan", original)
        old_observation = self.harness.latest["observation_id"]
        self.harness.observe()
        with self.assertRaisesRegex(ValueError, "stale_observation"):
            self.dispatch("task_plan", dict(operation="update", attention=[], observation_id=old_observation))
        with self.assertRaisesRegex(ValueError, "task_plan_requires_current_episode_observation"):
            owner.call(dict(operation="update", attention=[]), self.identity,
                       dict(self.harness.latest, episode_id="other-episode"), {})
        self.assertEqual(owner.current(self.identity), original)
        for field in self.identity:
            other = dict(self.identity, **{field: 2 if field == "env_generation" else "other"})
            self.assertIsNone(owner.decision_context(other))
        self.harness.reset()
        self.assertIsNone(owner.decision_context(self.identity))
        with self.assertRaisesRegex(ValueError, "task_plan_not_submitted"):
            self.dispatch("task_plan", dict(operation="update", attention=[]))
        self.assertEqual(self.store.read("task_plan", "P1"), original)
        self.dispatch("task_plan", dict(plan="Inspect the new episode.", ledger=[]))
        context = owner.decision_context(self.identity)
        self.assertEqual((context["revision"], context["attention"]), (1, []))

    def test_legacy_dictionary_plan_never_silently_discards_attention(self):
        legacy = archived_dict_plan_record(self.identity, self.harness.latest["observation_id"])
        self.store.save("task_plan", legacy["plan_id"], legacy)
        self.store.set_state("runtime", "task_plan", legacy)
        owner = self.harness.task_planning
        self.assertEqual(owner.decision_context(self.identity)["plan"], legacy["plan"])
        self.assertEqual(owner.decision_context(self.identity)["attention"], [])
        for arguments in (dict(operation="update", attention=[]),
                          dict(operation="update", plan="New text", attention=[]),
                          dict(operation="submit", observation_id=self.harness.latest["observation_id"],
                               plan=legacy["plan"], attention=[])):
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, "attention_requires"):
                self.dispatch("task_plan", arguments)
            self.assertEqual(owner.current(self.identity), legacy)
        self.assertEqual(payload(self.dispatch("task_plan", dict(operation="read")))["plan"], legacy)
        # Existing freeform replacement of an archived legacy plan remains supported.
        self.dispatch("task_plan", dict(operation="update", plan="A revised ordinary-language plan."))
        self.assertEqual(owner.decision_context(self.identity)["plan"], "A revised ordinary-language plan.")

    def test_stale_and_invalid_submissions_do_not_unlock(self):
        old_observation = self.harness.latest["observation_id"]
        self.harness.observe()
        with self.assertRaisesRegex(ValueError, "stale_observation"):
            self.submit(observation_id=old_observation)
        for value in ("", "  ", "x" * 16001, {}, dict(objective="legacy"), 123):
            with self.subTest(plan=value), self.assertRaisesRegex(ValueError, "invalid_task_plan"):
                self.submit(value)
        self.assertIsNone(self.harness.task_planning.current(self.identity))
        self.assertEqual(self.store.records("task_plan"), [])

    def test_projection_requires_matching_immutable_record(self):
        original = self.submit()
        for changed in (dict(original, plan_id="missing-record"), dict(original, revision=99)):
            with self.subTest(projection=changed):
                self.store.set_state("runtime", "task_plan", changed)
                self.assertIsNone(self.harness.task_planning.current(self.identity))
                with self.assertRaisesRegex(ValueError, "task_plan_required"):
                    self.harness.task_planning.require(self.identity)
        self.store.set_state("runtime", "task_plan", original)
        self.assertEqual(self.harness.task_planning.current(self.identity), original)

    def test_observation_from_other_episode_cannot_anchor_submission(self):
        self.harness.latest = dict(self.harness.latest, episode_id="other-episode")
        with self.assertRaisesRegex(ValueError, "task_plan_requires_current_episode_observation"):
            self.submit()
        self.assertIsNone(self.harness.task_planning.current(self.identity))

    def test_reset_invalidates_plan_and_same_identity_reconstruction_restores_it(self):
        first = self.submit()
        rebuilt = Harness(self.store, self.robot, config=self.config, task_plan_required=True)
        self.assertEqual(rebuilt.task_planning.current(self.identity), first)
        for key, alternate in (("run_id", "other"), ("episode_id", "other"),
                               ("env_instance_id", "other"), ("env_generation", 2)):
            with self.subTest(identity_field=key):
                self.assertIsNone(rebuilt.task_planning.current(dict(self.identity, **{key: alternate})))
        previous_identity = deepcopy(self.identity)
        self.harness.reset()
        self.assertIsNone(self.harness.task_planning.current(self.identity))
        self.assertEqual(self.store.read("task_plan", first["plan_id"])["identity"], previous_identity)
        rejected = payload(self.dispatch("robot_wait", self.action_args()))
        self.assertIn("task_plan_required", rejected["receipt"]["reason"])
        second = self.submit()
        self.assertEqual(second["revision"], 1)
        self.assertIsNone(second["previous_plan_id"])

    def test_rejected_action_id_is_immutable_after_plan_submission(self):
        args = dict(self.action_args(), action_id="before-plan")
        rejected = payload(self.dispatch("robot_wait", args))
        self.submit()
        self.assertEqual(payload(self.dispatch("robot_wait", args)), rejected)
        admitted = payload(self.dispatch("robot_wait", dict(args, action_id="after-plan")))
        self.assertEqual(admitted["receipt"]["status"], "completed")
        self.harness.reset()
        self.assertEqual(self.harness.execute(admitted["request"]),
                         {key: admitted[key] for key in ("request", "receipt")})
        self.assertEqual(self.robot.executions, 1)

    def test_program_dispatch_obeys_gate_and_can_submit_plan(self):
        from support.program_fixture import LegacyProgramRunner
        runner = LegacyProgramRunner(self.harness)
        self.harness.programs = runner
        entry = dict(job_id="fixture-program", cancel=threading.Event(), deadline=time.monotonic() + 600,
                     call_lock=threading.Lock(), calls=0, max_calls=10, correlation={},
                     condition=threading.Condition(), workspace=runner.root,
                     job=dict(job_id='fixture-program', status='running'),
                     action_evidence=[], model_wait_s=0., review_wait_s=0.)
        runner.entries[entry["job_id"]] = entry
        def call(operation, tool, arguments):
            return payload(runner.dispatch(entry, dict(operation_id=operation, tool=tool, arguments=arguments)))
        result = call("1", "robot_wait", self.action_args())
        self.assertIn("task_plan_required", result["receipt"]["reason"])
        call(
            "2",
            "task_plan",
            dict(
                operation="submit",
                plan="Observe and execute bounded transfer.",
                ledger=[],
                observation_id=self.harness.latest["observation_id"],
            ),
        )
        result = call("3", "robot_wait", self.action_args())
        self.assertEqual(result["receipt"]["status"], "completed")
        self.assertEqual(self.robot.executions, 1)

    def test_policy_gate_prevents_inference_and_still_calls_execute_after_submission(self):
        policy = SimpleNamespace(infer=Mock(return_value=dict(actions=[[0.] * 14])))
        self.harness.action_policy = policy
        args = dict(action_id="policy-action", observation_id=self.harness.latest["observation_id"],
                    prompt="Raise the left arm.", prefix=1)
        with self.assertRaisesRegex(ValueError, "task_plan_required"):
            self.harness.policy_step(args, {})
        policy.infer.assert_not_called()
        self.submit()
        with patch.object(self.harness, "execute", return_value={"sentinel": True}) as execute:
            self.assertEqual(self.harness.policy_step(args, {}), {"sentinel": True})
        policy.infer.assert_called_once()
        self.assertEqual(execute.call_args.args[0]["type"], "JointPrefixAction")

    def test_required_tool_cannot_be_disabled_and_default_harness_remains_optional(self):
        config = self.config.with_overrides(dict(features=dict(disabled_tools=["task_plan"])))
        with self.assertRaisesRegex(ValueError, "required_task_plan_tool_cannot_be_disabled"):
            Harness(self.store, self.robot, config=config, task_plan_required=True)
        optional = Harness(self.store, self.robot, config=config)
        optional.start()
        result = payload(optional.dispatch("robot_wait", dict(self.action_args(),
                         observation_id=optional.latest["observation_id"]), dict(tool_call_id="optional")))
        self.assertEqual(result["receipt"]["status"], "completed")

    def test_policy_inference_cannot_carry_old_plan_across_reset(self):
        self.submit()
        def infer(*args, **kwargs):
            self.harness.reset()
            return dict(actions=[[0.] * 14])
        self.harness.action_policy = SimpleNamespace(infer=Mock(side_effect=infer))
        args = dict(action_id="reset-during-policy", observation_id=self.harness.latest["observation_id"],
                    prompt="Raise the arm.", prefix=1)
        result = self.harness.policy_step(args, {})
        self.assertEqual(result["receipt"]["status"], "rejected")
        self.assertIn("task_plan_required", result["receipt"]["reason"])
        self.assertEqual(self.robot.executions, 0)
        self.assertEqual(self.harness.policy_step(args, {}), result)
        self.harness.action_policy.infer.assert_called_once()


class TaskPlanningGatewayCheck(unittest.IsolatedAsyncioTestCase):
    async def test_production_cannot_disable_required_tool(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config.from_dict(dict(features=dict(disabled_tools=["task_plan"])))
            with self.assertRaisesRegex(ValueError, "required_task_plan_tool_cannot_be_disabled"):
                Workbench(Path(temporary) / "disabled-plan", robot=SimpleNamespace(close=lambda: None), config=config)

    async def test_production_contract_is_fixed_and_exported(self):
        with tempfile.TemporaryDirectory() as temporary:
            config = Config.from_dict(dict(features=dict(observation_profile=LEGACY_ACTOR_PROFILE,
                motion_review=False, completion_guidance="none", terminal=False, visual_review_model=None)))
            workbench = Workbench(Path(temporary) / "gateway-plan", robot=SimpleNamespace(close=lambda: None), config=config)
            self.assertTrue(workbench.harness.task_planning.required)
            self.assertIn("task_plan", {tool["name"] for tool in workbench.actor_tools()})
            # Follow gateway lifecycle fixtures: substitute process-free startup dependencies.
            harness = SimpleNamespace(accepting=True, active_action=None, states={}, action_count=0,
                capabilities=dict(data_origin="synthetic_fixture"), thread_id=None,
                scene=SimpleNamespace(capabilities=lambda: dict(physics="unsupported")))
            for field in ('tool_registry', 'config', 'disabled_tools', 'observation_profile',
                          'motion_review', 'enable_wait', 'vision_compare', 'visual_reviewer',
                          'action_policy', 'programs'):
                setattr(harness, field, getattr(workbench.harness, field))
            harness.start = lambda: None
            harness.request_stop = lambda: setattr(harness, "accepting", False)
            session = SimpleNamespace(thread_id="fixture-thread", turn_id=None, state="idle", actor_dir=workbench.root,
                model="fixture-model", provider="fixture-provider", effort="high", price_card=None,
                cli_executable=Path("/fixture/codex"), process=SimpleNamespace(pid=-1), close=lambda: None)
            session.start = Mock(return_value="fixture-thread")
            session.run_turn = lambda prompt: None
            workbench.harness, workbench.session = harness, session
            try:
                await workbench.start("Observe the task, then act.")
                instructions = session.start.call_args.args[1]
                self.assertIn("Before your first physical action", instructions)
                self.assertIn("provisional understanding", instructions)
                manifest = json.loads((workbench.root / "manifest.json").read_text())
                self.assertEqual(manifest["task_planning"], dict(version="freeform_task_plan", required_before_robot_actions=True,
                                 tool_choices="advisory", revisions="actor_discretion"))
                self.assertIn("cortex/context/task_planning.py", manifest["source_sha256"])
                _, source = workbench.store.resolve_artifact(manifest["source_ref"], allowed_scopes=("observer",))
                self.assertIn("cortex/context/task_planning.py", json.loads(source))
            finally:
                await workbench.close()


class TaskSchedulingTest(unittest.TestCase):
    """在线任务调度笔记与失败回执依赖重评估提示契约测试。"""

    def setUp(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        self.identity = dict(run_id="r", episode_id="e", env_instance_id="env", env_generation=1)
        self.store = Store(Path(temp), self.identity)
        self.addCleanup(self.store.close)
        self.observation = dict(self.identity, observation_id="obs", sim_step=0)
        self.store.save("observation", "obs", self.observation)
        self.plan = dict(
            objective="Complete independent visible work.",
            stages=[
                dict(id="a", goal="One object.", advance_when="Placed."),
                dict(id="b", goal="Another object.", advance_when="Placed."),
            ],
            constraints=[],
            uncertainties=[],
            completion_criteria=["Visible objects are placed."],
        )

    def test_task_notes_ablation_and_retired_operation_candidates_compat(self):
        from ni_rua.cortex.context.task_notes import (
            PREVIOUS_DEFAULT_TASK_NOTES,
            TASK_SCHEDULING_GUIDANCE,
            default_notes_for,
            resolve_task_notes,
        )

        flags = dict(task_scheduling=False, operation_candidates=False)
        self.assertEqual(default_notes_for(**flags), PREVIOUS_DEFAULT_TASK_NOTES)
        resolved = resolve_task_notes(**flags)
        self.assertEqual(resolved["config"]["default_version"], "operation-notes-v14")
        self.assertEqual(resolved["config"]["operation_candidates"], False)
        for scheduling in (False, True):
            content = default_notes_for(task_scheduling=scheduling, operation_candidates=True)
            self.assertEqual(TASK_SCHEDULING_GUIDANCE in content, scheduling)
            self.assertEqual(
                resolve_task_notes(task_scheduling=scheduling, operation_candidates=True)["config"]["operation_candidates"],
                False,
            )

    def test_failed_action_advisory_preserves_dependencies_and_stays_bounded(self):
        self.plan["stages"][1]["depends_on"] = ["a"]
        record = dict(plan=self.plan, observation_id="previous-plan-observation")
        before = deepcopy(record)
        result = scheduling_failure_advisory(dict(status="failed", reason="tracking_error"), record)
        self.assertEqual(result["dependency_free_plan_stages_to_reassess"], ["a"])
        self.assertEqual(result["plan_observation_id"], "previous-plan-observation")
        self.assertNotIn("holding_arm", result)
        self.assertEqual(record, before)
        self.assertIsNone(scheduling_failure_advisory(dict(status="failed"), task_scheduling=False))
        for receipt in (None, {}, dict(status="completed"), dict(status="executing"), dict(status="cancelled")):
            self.assertIsNone(scheduling_failure_advisory(receipt))
        self.assertEqual(scheduling_failure_advisory(dict(status="rejected", reason="path_infeasible"))["kind"], "scheduling_review")
        long_record = dict(plan=dict(stages=[dict(id=str(index), goal="long actor text" * 1000) for index in range(12)]))
        bounded = scheduling_failure_advisory(dict(status="failed"), long_record)
        self.assertEqual(bounded["dependency_free_plan_stages_to_reassess"], ["0", "1", "2"])
        self.assertNotIn("long actor text", str(bounded))
        # Freeform prose plans attach plan_observation_id without fabricating stage lists.
        freeform_record = dict(plan="Pick left then right.", observation_id="obs-freeform")
        freeform_adv = scheduling_failure_advisory(dict(status="failed"), freeform_record)
        self.assertEqual(freeform_adv["plan_observation_id"], "obs-freeform")
        self.assertNotIn("dependency_free_plan_stages_to_reassess", freeform_adv)


if __name__ == "__main__":
    unittest.main()
