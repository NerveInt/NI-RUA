"""
- test_operation_memory.py: Operation lessons reuse immutable facts without merging task/shared scopes.
- test_operation_verification.py: Operation review consumes permitted evidence without executing recovery.
- test_robot_calibration_information.py: CPU-only calibration reads from real Harness publication and Store artifacts.
- test_robot_information_integration.py: Initialization and actor retrieval use the same physical knowledge interface.
"""
from copy import deepcopy
from pathlib import Path
import tempfile
import unittest
from ni_rua.core.presentation.actor_payload import lean_v2_display
from unittest.mock import patch
from ni_rua.core.contracts import digest
from ni_rua.cortex.context.operation_memory import OperationMemory
from ni_rua.core.storage.store import Store
from ni_rua.core.tool_errors import ToolInputError, error_payload
from support.program_fixture import program_services
import json
import threading
import time
from types import SimpleNamespace
from unittest.mock import Mock, patch
import cv2
import numpy as np
from jsonschema import ValidationError
from ni_rua.core.contracts import canonical, digest
from ni_rua.core.config import Config
from ni_rua.apps.gateway.workbench import Workbench
from ni_rua.runtime.harness import Harness
from ni_rua.cortex.context.operation_verification import PHASES, TOOL_REQUIRED
from ni_rua.runtime.programs.runner import ProgramRunner, model_wait, program_remaining
from ni_rua.cortex.session.visual_review import MODEL, VisualReviewer
from unittest.mock import Mock
from ni_rua.core.contracts import canonical
from ni_rua.core.observation_policy import ACTOR_PROFILE, LEGACY_ACTOR_PROFILE
from ni_rua.robot.model.robot_calibration_information import robot_calibration_information
from support.fixture_robot import FixtureRobot
import base64
import zipfile
from types import MethodType
from ni_rua.cortex.context.actor_instructions import INSTRUCTIONS, TASK_INSTRUCTIONS
from ni_rua.cortex.context.task_planning import PLANNING_INSTRUCTIONS
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.core.presentation.actor_output import ActorOutput
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.robot.model.robot_information import robot_information, robot_resource_path
from support.agent_fixtures import fixture_workbench
from support.paths import isolate_home


def setUpModule():
    isolate_home()   # a fresh Workbench opens the grasp memory store at its '~' default


NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_operation_memory.py ---

class OperationMemoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.identity = dict(run_id="run", episode_id="task", env_instance_id="env", env_generation=1)
        self.store = Store(Path(self.temp.name) / "task", self.identity)
        self.shared = Store(Path(self.temp.name) / "shared", self.identity)
        self.addCleanup(self.store.close)
        self.addCleanup(self.shared.close)
        self.memory = OperationMemory(self.store, self.shared)
        for number, identifier in enumerate(("before", "after", "unrelated")):
            self.store.save("observation", identifier, dict(self.identity,
                observation_id=identifier, sim_step=number * 10, step_domain="adapter_control_step"))
        self.action = dict(self.identity, action_id="move", type="EEAction", based_on_observation_id="before",
            path=dict(waypoints=[dict(left=dict(position_m=[1, 2, 3]))]),
            gripper_events=[dict(arm="left", phase="after_motion", goal_normalized=1.)], speed_scale=.5)
        self.receipt = dict(status="completed", result_observation_id="after", completed_native_steps=9)
        self.store.save("action_request", "move", self.action)
        self.store.save("action_receipt", "move", self.receipt)
        self.request = dict(kind="record", scope="task", operation="retreat",
            summary="Clear the tall rim vertically before moving sideways.",
            conditions="Object visibly released inside the container.",
            action_ids=["move"], observation_ids=["after"])

    def test_execution_facts_are_extracted_and_completion_does_not_verify_object_success(self):
        before = deepcopy(self.request)
        record = self.memory.call(self.request)["record"]
        self.assertEqual(self.request, before)
        self.assertEqual(record["memory_id"], "task_experience_1")
        self.assertEqual(record["physical_outcome"], "not_verified")
        self.assertEqual(record["facts"][0]["status"], "completed")
        self.assertEqual(record["facts"][0]["completed_native_steps"], 9)
        self.assertEqual(record["facts"][0]["gripper_commands"][0]["goal_normalized"], 1.)
        self.assertEqual(self.store.read("action_request", "move"), self.action)
        self.assertEqual(self.store.read("action_receipt", "move"), self.receipt)
        self.assertNotIn("action_id", self.store.read("action_receipt", "move"))
        self.assertNotIn("evidence", record)
        stored = self.store.read("operation_memory", record["memory_id"])
        self.assertIn("request_digest", stored["evidence"]["actions"][0])
        self.assertEqual(stored["evidence"]["actions"][0]["receipt_digest"], digest(self.receipt))
        self.assertNotIn("position_m", str(record))

    def test_shared_writers_do_not_allocate_ids_from_a_stale_record_count(self):
        request = dict(self.request, scope="shared")
        with patch.object(self.shared, "records", return_value=[]):
            first = self.memory.call(request)["record"]
            second = self.memory.call(request)["record"]
        self.assertNotEqual(first["memory_id"], second["memory_id"])
        self.assertEqual(len(self.shared.records("operation_memory")), 2)

    def test_task_and_shared_queries_remain_separate_and_shared_survives_next_task(self):
        task = self.memory.call(self.request)["record"]
        shared = self.memory.call(dict(self.request, scope="shared"))["record"]
        self.assertEqual(self.memory.call(dict(kind="query", scope="task"))["records"], [task])
        self.assertEqual(self.memory.call(dict(kind="query", scope="shared"))["records"], [shared])
        other_identity = dict(self.identity, episode_id="next", env_generation=2)
        with tempfile.TemporaryDirectory() as directory:
            other_store = Store(directory, other_identity)
            try:
                next_memory = OperationMemory(other_store, self.shared)
                self.assertEqual(next_memory.call(dict(kind="query", scope="task"))["records"], [])
                reused = next_memory.call(dict(kind="query", scope="shared"))["records"][0]
                self.assertEqual(reused["source"]["episode_id"], "task")
                self.assertIn("re-localize", reused["reuse"])
                self.assertNotIn("position_m", str(reused))
            finally:
                other_store.close()
        self.store.identity = other_identity
        self.assertEqual(self.memory.call(dict(kind="query", scope="task"))["records"], [])

    def test_all_operation_categories_use_the_same_history_contract(self):
        for operation in ("approach", "grasp", "transport", "place", "retreat", "wait", "recover"):
            self.memory.call(dict(self.request, operation=operation))
            found = self.memory.call(dict(kind="query", scope="task", operation=operation.upper()))
            self.assertEqual(found["total_matches"], 1)
            self.assertEqual(found["records"][0]["operation"], operation)
        self.assertEqual(self.memory.call(dict(kind="query", scope="task", query="tall rim", limit=2))["total_matches"], 7)
        self.assertEqual(len(self.memory.call(dict(kind="query", scope="task", limit=2))["records"]), 2)

    def test_failed_plan_without_new_observation_can_record_only_mechanical_failure(self):
        action = dict(self.action, action_id="failed")
        self.store.save("action_request", "failed", action)
        self.store.save("action_receipt", "failed", dict(action_id="failed", status="failed", reason="path_infeasible"))
        record = self.memory.call(dict(self.request, action_ids=["failed"], observation_ids=["before"]))["record"]
        self.assertEqual(record["facts"][0]["reason"], "path_infeasible")
        self.assertIsNone(record["facts"][0]["result_observation_id"])
        self.assertEqual(record["physical_outcome"], "not_verified")

    def test_wrong_identity_unrelated_and_missing_result_evidence_rejected(self):
        for observations, error in ((["before"], "requires_action_result"),
                                     (["after", "unrelated"], "unrelated_observation")):
            with self.subTest(observations=observations), self.assertRaisesRegex(ValueError, error):
                self.memory.call(dict(self.request, observation_ids=observations))
        with self.assertRaisesRegex(ToolInputError, "action_request_unavailable"):
            self.memory.call(dict(self.request, action_ids=["not-an-action"]))
        self.store.identity = dict(self.identity, env_generation=2)
        with self.assertRaisesRegex(ValueError, "identity_mismatch"):
            self.memory.call(self.request)

    def test_caller_cannot_supply_fake_facts_or_unbounded_payload(self):
        for update in (dict(facts=[dict(status="success")]), dict(summary="x" * 801),
                       dict(action_ids=["move", "move"]), dict(scope="both")):
            with self.subTest(update=update), self.assertRaises(ValueError):
                self.memory.call(dict(self.request, **update))
        for limit in (0, 31, True):
            with self.assertRaisesRegex(ValueError, "invalid_operation_memory_limit"):
                self.memory.call(dict(kind="query", scope="task", limit=limit))

    def test_shared_conditions_and_dedicated_store_are_required(self):
        request = dict(self.request, scope="shared")
        request.pop("conditions")
        with self.assertRaisesRegex(ValueError, "shared_conditions_required"):
            self.memory.call(request)
        local_only = OperationMemory(self.store)
        self.assertFalse(local_only.call(dict(kind="help"))["shared_available"])
        with self.assertRaisesRegex(ValueError, "shared_store_unavailable"):
            local_only.call(dict(kind="query", scope="shared"))

    def test_explicit_receipt_or_request_ids_cannot_disagree_with_record_key(self):
        for identifier, request_id, receipt_id in (("explicit", "explicit", "explicit"),
                ("wrong-receipt", "wrong-receipt", "another-action"),
                ("wrong-request", "another-action", "wrong-request"),
                ("null-receipt", "null-receipt", None)):
            action = dict(self.action, action_id=request_id)
            receipt = dict(self.receipt, action_id=receipt_id)
            self.store.save("action_request", identifier, action)
            self.store.save("action_receipt", identifier, receipt)
            request = dict(self.request, action_ids=[identifier])
            if identifier == "explicit":
                self.assertEqual(self.memory.call(request)["record"]["facts"][0]["action_id"], identifier)
            else:
                with self.assertRaisesRegex(ToolInputError, "action_id_mismatch") as raised:
                    self.memory.call(request)
                self.assertEqual(raised.exception.field_path, "/request/action_ids/0")
                self.assertEqual(raised.exception.actual, identifier)
            self.assertEqual(self.store.read("action_request", identifier), action)
            self.assertEqual(self.store.read("action_receipt", identifier), receipt)

    def test_missing_and_unlinked_evidence_errors_identify_the_required_ids(self):
        with self.assertRaises(ToolInputError) as raised:
            self.memory.call(dict(self.request, observation_ids=["before"]))
        payload = error_payload(raised.exception, tool="operation_memory", arguments=dict(request=self.request))
        self.assertEqual(payload["field_path"], "/request/observation_ids")
        self.assertEqual(payload["actual"], "move")
        self.assertIn("after", payload["suggestion"])
        with self.assertRaises(ToolInputError) as raised:
            self.memory.call(dict(self.request, observation_ids=["after", "unrelated"]))
        self.assertEqual(raised.exception.field_path, "/request/observation_ids/1")
        self.assertEqual(raised.exception.actual, "unrelated")
        with self.assertRaises(ToolInputError) as raised:
            self.memory.call(dict(self.request, observation_ids=["not-an-observation"]))
        self.assertEqual(raised.exception.code, "operation_memory_observation_unavailable")
        self.assertEqual(self.store.records("operation_memory"), [])

    def test_empty_action_ids_explain_that_preview_notes_need_action_evidence(self):
        with self.assertRaises(ToolInputError) as raised:
            self.memory.call(dict(self.request, action_ids=[]))
        self.assertEqual(raised.exception.field_path, "/request/action_ids")
        self.assertEqual(raised.exception.actual, [])
        self.assertIn("history_list_actions", raised.exception.suggestion)
        self.assertIn("preview-only", raised.exception.suggestion.lower())
        self.assertEqual(self.store.records("operation_memory"), [])

    def test_request_errors_explain_accepted_fields_values_and_limits(self):
        cases = [
            (dict(kind="query", scope="task", extra=True), "invalid_operation_memory_fields", "/request", "Allowed fields:"),
            (dict(self.request, summary="x" * 801), "invalid_operation_memory_summary", "/request/summary", "800 characters"),
            (dict(kind="query", scope="both"), "invalid_operation_memory_scope", "/request/scope", "scope='task'"),
            (dict(kind="lookup"), "unknown_operation_memory_kind", "/request/kind", "kind='query'"),
            (dict(kind="query", scope="task", limit=True), "invalid_operation_memory_limit", "/request/limit", "1 through 30"),
        ]
        for request, code, field_path, advice in cases:
            with self.subTest(code=code), self.assertRaises(ToolInputError) as raised:
                self.memory.call(request)
            self.assertEqual(raised.exception.code, code)
            self.assertEqual(raised.exception.field_path, field_path)
            self.assertIn(advice, raised.exception.suggestion)

    def test_shared_experience_survives_disk_reopen_for_a_different_run(self):
        root = Path(self.temp.name) / "persistent-shared"
        first = Store(root, self.identity)
        try:
            recorded = OperationMemory(self.store, first).call(dict(self.request, scope="shared"))["record"]
        finally:
            first.close()
        next_identity = dict(self.identity, run_id="next-run", episode_id="next-task", env_generation=2)
        next_store = Store(Path(self.temp.name) / "next-task", next_identity)
        reopened = Store(root, next_identity)
        try:
            next_memory = OperationMemory(next_store, reopened)
            self.assertEqual(next_memory.call(dict(kind="query", scope="task"))["records"], [])
            found = next_memory.call(dict(kind="query", scope="shared", query="rim"))["records"]
            self.assertEqual(found, [recorded])
            self.assertEqual(found[0]["source"]["run_id"], self.identity["run_id"])
            self.assertEqual(self.store.read("action_receipt", "move"), self.receipt)
        finally:
            next_store.close()
            reopened.close()


class OperationMemoryIntegrationTests(unittest.TestCase):
    """Harness 回执经短引用与记忆写入保持不可变测试 (迁自 test_operation_memory_integration.py)。"""

    def test_actor_records_the_actual_backend_receipt_without_rewriting_it(self):
        import json
        from ni_rua.robot.backends.robodojo.robodojo import FRAME_ID, TCP_CONFIG_ID
        from ni_rua.runtime.harness import Harness
        from support.fixture_robot import FixtureRobot

        temporary = self.enterContext(tempfile.TemporaryDirectory())
        identity = dict(run_id="memory-integration", episode_id="fixture", env_instance_id="env", env_generation=1)
        store = Store(Path(temporary) / "task", identity)
        self.addCleanup(store.close)
        robot = FixtureRobot(temporary)
        harness = Harness(store, robot)
        harness.start()
        action = dict(
            schema_version="0.1",
            **identity,
            world_kind="observed",
            action_id="action-" + "a" * 32,
            based_on_observation_id=harness.latest["observation_id"],
            frame_id=FRAME_ID,
            tcp_config_id=TCP_CONFIG_ID,
            mode="absolute",
            limits_profile="simulation-slow-v1",
            timeout_s=10,
            path=dict(
                kind="backend_free",
                orientation="backend_free",
                blend_radius_m=0,
                position_tolerance_m=0.005,
                waypoints=[dict(left=dict(position_m=[0.01, 0, 1], quaternion_xyzw=[0, 0, 0, 1]))],
            ),
            gripper_events=[],
        )
        result = harness.execute(action)
        receipt = store.read("action_receipt", action["action_id"])
        self.assertNotIn("action_id", receipt)
        original_digests = (digest(store.read("action_request", action["action_id"])), digest(receipt))
        shown = harness.actor_output.present(dict(
            action_id=action["action_id"],
            observation_id=result["receipt"]["result_observation_id"],
        ))
        self.assertEqual(shown["action_id"], "act_1")
        harness.thread_id = "thread"
        response = harness.handle_tool(dict(
            threadId="thread",
            turnId="turn",
            callId="remember",
            tool="operation_memory",
            arguments=dict(request=dict(
                kind="record",
                scope="task",
                operation="retreat",
                summary="Inspect the measured end state after withdrawing.",
                action_ids=[shown["action_id"]],
                observation_ids=[shown["observation_id"]],
            )),
        ))
        self.assertTrue(response["success"])
        record = json.loads(response["contentItems"][0]["text"])["record"]
        self.assertEqual(record["physical_outcome"], "not_verified")
        self.assertEqual(record["facts"][0]["action_id"], shown["action_id"])
        self.assertEqual(len(store.records("operation_memory")), 1)
        self.assertEqual(
            original_digests,
            (digest(store.read("action_request", action["action_id"])), digest(store.read("action_receipt", action["action_id"]))),
        )
        self.assertEqual(robot.executions, 1)



# --- Folded from tests/test_operation_verification.py ---

def answer(**changes):
    return dict(assessment="met", same_target_identified="supported", visibility_after="visible",
                target_motion="displaced", gripper_relation="held", support_relation="visibly_separated",
                destination_relation="not_applicable", gripper_withdrawal="not_applicable",
                before_evidence="The striped target is visible beside the fingers on the table.",
                after_evidence="The same stripe is visible between the raised fingers above the table.",
                execution_evidence="One completed move has fresh measured arm state; closure is a command only.",
                uncertainties=[], follow_up="none", follow_up_reason="", recovery_feedback="", **changes)


class OperationVerificationTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.identity = dict(run_id="r", episode_id="e", env_instance_id="i", env_generation=1)
        self.store = Store(temporary.name, self.identity)
        self.addCleanup(self.store.close)
        self.reviewer = VisualReviewer(SimpleNamespace(base_url="http://127.0.0.1:1", _token="fixture", model=MODEL))
        self.observation("before", 10, [0, 0, .1])
        self.observation("after", 15, [0, 0, .12])
        self.request = dict(**self.identity, action_id="lift", based_on_observation_id="before", type="EEAction",
                            path=dict(kind="backend_free", waypoints=[dict(left=dict(position_m=[0, 0, .12],
                                      quaternion_xyzw=[0, 0, 0, 1]))]),
                            gripper_events=[dict(arm="left", phase="after_motion", goal_normalized=0)], timeout_s=20)
        self.receipt = dict(status="completed", result_observation_id="after", executed_steps=5,
                            execution_interval=dict(start_sim_step=10, end_sim_step=15, step_domain="fixture"))
        self.store.save("action_request", "lift", self.request)
        self.args = dict(phase="short_lift", target="striped visible block", expected_result="Block visibly moves with gripper off table",
                         action_ids=["lift"], before_observation_id="before", after_observation_id="after",
                         camera_id="head", roi=[1, 1, 8, 8])

    def observation(self, name, step, position, *, scope="actor"):
        pixels = np.zeros((20, 20, 3), np.uint8)
        pixels[3:7, 3:7] = [255, step, 100]
        rgb = self.store.put_artifact(cv2.imencode(".png", pixels)[1].tobytes(), "rgb", "image/png", scope)
        state = dict(arms=dict(left=dict(q_rad=[0]*6,
                     ee_pose=dict(position_m=position, quaternion_xyzw=[0, 0, 0, 1], frame_id="env_origin", tcp_config_id="tcp"),
                     gripper=dict(goal_normalized=0, goal_source_kind="command"))), sim_step=step, step_domain="fixture")
        state_ref = self.store.put_artifact(canonical(state).encode(), "robot_state", "application/json")
        self.store.save("observation", name, dict(**self.identity, observation_id=name, observation_profile="rgb_proprio_v1",
                        sim_step=step, step_domain="fixture", robot_state_ref=state_ref,
                        cameras=[dict(camera_id="head", frame_id=name, rgb_ref=rgb, width=20, height=20)]))

    def raw(self, value, *, model=MODEL, status="completed", tool="record_operation_verification", calls=1):
        result = dict(model=model, status=status, id="response-fixture", usage=dict(output_tokens=80),
                      answer=value, thread_id="fresh-fixture", turn_id="fixture-turn",
                      context_policy="fresh_ephemeral_thread_selected_evidence_only_v1", backend="official_codex_app_server_sdk")
        return result

    def review(self, value=None, **changes):
        with patch.object(self.reviewer, "_request", return_value=self.raw(value or answer())) as request:
            result = self.reviewer.review_operation(self.store, self.identity, **{**self.args, **changes})
        return result, request

    def test_positive_review_consumes_actual_images_receipt_and_measured_state(self):
        self.store.save("action_receipt", "lift", self.receipt)
        cancel = threading.Event()
        result, request = self.review(cancel_event=cancel, deadline=time.monotonic()-10)
        self.assertEqual(result["answer"]["assessment"], "met")
        self.assertEqual(result["execution"]["status"], "completed")
        self.assertEqual(result["assessment_constraints"], [])
        evidence = result["execution"]["actions"][0]
        self.assertEqual(evidence["receipt_sha256"], digest(self.receipt))
        self.assertEqual(evidence["request_sha256"], digest(self.request))
        self.assertEqual(result["execution"]["after_robot_state"]["state"]["arms"]["left"]["ee_pose"]["position_m"], [0, 0, .12])
        self.assertEqual([source["observation_id"] for source in result["sources"]], ["before", "after"])
        self.assertTrue(all(source["sha256"] for source in result["sources"]))
        payload = request.call_args.args[0]
        self.assertEqual(payload["model"], MODEL)
        self.assertEqual(payload["effort"], "high")
        self.assertEqual(sum(content["type"] == "image" for content in payload["input"]), 2)
        prompt = payload["input"][0]["text"]
        self.assertIn('"executed_steps":5', prompt)
        self.assertIn('"position_m":[0,0,0.12]', prompt)
        self.assertIn("closed jaw never proves success", prompt.lower())
        self.assertEqual(request.call_args.kwargs, dict(cancel_event=cancel))
        self.assertNotIn("usage", result)
        self.assertTrue(result["actor_decides_next_action"])
        self.assertEqual(result["task_completion"], "not_assessed")
        event = self.store.events()[-1]
        self.assertEqual(event["type"], "visual.model_reviewed")
        self.assertEqual(event["payload"]["review_kind"], "operation_verify")
        self.assertEqual(event["payload"]["usage"]["output_tokens"], 80)
        self.assertEqual(self.store.artifact(event["payload"]["request_ref"])[0]["access_scope"], "observer")

    def test_unconfirmed_execution_cannot_be_promoted_by_positive_model_answer(self):
        result, _ = self.review()
        self.assertEqual(result["execution"]["status"], "unknown")
        self.assertFalse(result["execution"]["actions"][0]["receipt_recorded"])
        self.assertIsNone(result["execution"]["actions"][0]["receipt_sha256"])
        self.assertEqual(result["answer"]["assessment"], "uncertain")
        self.assertEqual(result["answer"]["follow_up"], "return_to_actor")
        self.assertIn("execution_not_completed:lift", result["execution"]["blockers"])

    def test_partial_execution_is_preserved_and_negative_feedback_is_model_led(self):
        self.store.save("action_receipt", "lift", {**self.receipt, "status": "failed", "reason": "tracking_divergence", "executed_steps": 2})
        value = answer()
        value.update(assessment="not_met", gripper_relation="separated", uncertainties=["Target partly occluded"],
                     follow_up="inspect_fresh_rgb", follow_up_reason="Resolve target location before choosing recovery",
                     recovery_feedback="Review the visible target and measured arm pose before a new approach.")
        result, _ = self.review(value)
        self.assertEqual(result["answer"], value)
        self.assertEqual(result["execution"]["actions"][0]["feedback"]["executed_steps"], 2)
        self.assertEqual(result["execution"]["status"], "incomplete")
        result, _ = self.review()
        self.assertEqual(result["answer"]["assessment"], "met")
        self.assertEqual(result["execution"]["status"], "incomplete")
        self.assertEqual(result["execution"]["actions"][0]["feedback"]["status"], "failed")

    def test_different_result_observation_and_replayed_frame_remain_unconfirmed(self):
        self.observation("newer", 16, [0, 0, .13])
        self.store.save("action_receipt", "lift", self.receipt)
        result, _ = self.review(after_observation_id="newer")
        self.assertEqual(result["answer"]["assessment"], "uncertain")
        self.assertIn("final_action_result_observation_mismatch", result["assessment_constraints"])
        result, _ = self.review(after_observation_id="before")
        self.assertIn("no_new_simulation_step", result["assessment_constraints"])

    def test_phases_are_independent_model_judgments_without_fixed_sequence(self):
        self.store.save("action_receipt", "lift", self.receipt)
        for phase in PHASES:
            with self.subTest(phase=phase):
                value = answer()
                if phase == "approach":
                    value.update(gripper_relation="near", support_relation="visibly_supported")
                if phase in ("place", "release", "retreat"):
                    value.update(support_relation="visibly_supported", destination_relation="visibly_at_destination")
                if phase in ("release", "retreat"):
                    value.update(gripper_relation="separated", target_motion="roughly_stationary")
                if phase == "retreat":
                    value["gripper_withdrawal"] = "visually_supported"
                result, _ = self.review(value, phase=phase, destination="visible tray" if phase in ("place", "release", "retreat") else None)
                self.assertEqual(result["phase"], phase)
                self.assertEqual(result["answer"]["assessment"], "met")
        value = answer()
        value.update(assessment="uncertain", follow_up="inspect_another_camera", follow_up_reason="Support is occluded")
        result, _ = self.review(value)
        self.assertEqual(result["answer"], value)

    def test_multiple_actions_need_linked_observation_chain(self):
        self.observation("middle", 12, [0, 0, .11])
        self.store.save("action_receipt", "lift", {**self.receipt, "result_observation_id": "middle"})
        self.store.save("action_request", "second", {**self.request, "action_id": "second", "based_on_observation_id": "before"})
        self.store.save("action_receipt", "second", self.receipt)
        result, _ = self.review(action_ids=["lift", "second"])
        self.assertEqual(result["answer"]["assessment"], "uncertain")
        self.assertIn("action_observation_chain_mismatch:second", result["assessment_constraints"])

    def test_bad_schema_wrong_model_and_extra_calls_fail_with_private_provenance(self):
        self.store.save("action_receipt", "lift", self.receipt)
        invalid = answer()
        invalid["invented_contact_truth"] = True
        for raw, error in [(self.raw(invalid), ValidationError), (self.raw(answer(), model="substitute"), ValueError),
                           (self.raw(answer(), status="incomplete"), ValueError)]:
            with self.subTest(error=error), patch.object(self.reviewer, "_request", return_value=raw):
                with self.assertRaises(error):
                    self.reviewer.review_operation(self.store, self.identity, **self.args)
                event = self.store.events()[-1]
                self.assertEqual(event["type"], "visual.model_review_failed")
                self.assertTrue(event["payload"]["response_ref"])

    def test_non_actor_images_and_privileged_receipts_never_reach_model(self):
        self.observation("private", 15, [0, 0, .12], scope="observer")
        self.store.save("action_receipt", "lift", {**self.receipt, "contact_points": [[0, 0, 0]]})
        with patch.object(self.reviewer, "_request") as request:
            with self.assertRaisesRegex(ValueError, "access_scope_denied"):
                self.reviewer.review_operation(self.store, self.identity, **{**self.args, "after_observation_id": "private"})
            with self.assertRaisesRegex(ValueError, "privileged_data"):
                self.reviewer.review_operation(self.store, self.identity, **self.args)
            request.assert_not_called()

    def test_wrong_episode_action_is_rejected_before_model_and_cancel_is_recorded(self):
        self.store.save("action_request", "old", {**self.request, "action_id": "old", "env_generation": 2})
        with patch.object(self.reviewer, "_request") as request:
            with self.assertRaisesRegex(ValueError, "action_identity_mismatch"):
                self.reviewer.review_operation(self.store, self.identity, **{**self.args, "action_ids": ["old"]})
            request.assert_not_called()
        cancel = threading.Event()
        cancel.set()
        with patch.object(self.reviewer, "_request", side_effect=TimeoutError("cancelled")):
            with self.assertRaises(TimeoutError):
                self.reviewer.review_operation(self.store, self.identity, **self.args, cancel_event=cancel)
        event = self.store.events()[-1]
        self.assertEqual(event["type"], "visual.model_review_cancelled")
        self.assertIsNone(event["payload"]["usage"])

    def test_invalid_requests_do_not_invoke_model(self):
        for changes in (dict(phase="made_up"), dict(target=" "), dict(action_ids=[]), dict(action_ids=["lift", "lift"])):
            with self.subTest(changes=changes), patch.object(self.reviewer, "_request") as request:
                with self.assertRaises(ValueError):
                    self.reviewer.review_operation(self.store, self.identity, **{**self.args, **changes})
                request.assert_not_called()

    def test_transport_delegates_to_official_agent_without_response_deadline(self):
        self.reviewer.client = SimpleNamespace(model=MODEL, review_evidence=Mock(return_value={"fixture": True}))
        payload = dict(model=MODEL, input=[dict(type="text", text="Evidence only")], outputSchema={})
        cancel = threading.Event()
        self.assertEqual(self.reviewer._request(payload, "fixture", cancel_event=cancel, deadline=time.monotonic()-100), {"fixture": True})
        self.reviewer.client.review_evidence.assert_called_once_with(input_items=payload["input"], output_schema={},
                                                                     request_id="fixture", cancel_event=cancel)

    def harness(self):
        robot = SimpleNamespace(execute=Mock(), observe=Mock())
        config = Config.from_dict(dict(features=dict(observation_profile="rgb_proprio_v1", motion_review=False,
                                                     completion_guidance="none")))
        harness = Harness(self.store, robot, config=config, motion_review=False)
        harness.visual_reviewer = self.reviewer
        return harness, robot

    def test_harness_dispatch_reviews_evidence_without_changing_action_or_receipt(self):
        self.store.save("action_receipt", "lift", self.receipt)
        harness, robot = self.harness()
        with patch.object(self.reviewer, "_request", return_value=self.raw(answer())) as request:
            returned = harness.dispatch("vision_review", dict(self.args, evidence_mode="operation"),
                                        dict(tool_call_id="verification"))
        self.assertTrue(returned["success"])
        result = json.loads(returned["contentItems"][0]["text"])
        self.assertEqual(result["answer"]["assessment"], "met")
        self.assertEqual(result["execution"]["actions"][0]["receipt_sha256"], digest(self.receipt))
        self.assertEqual(result["sources"][1]["observation_id"], "after")
        self.assertEqual(self.store.read("action_request", "lift"), self.request)
        self.assertEqual(self.store.read("action_receipt", "lift"), self.receipt)
        self.assertEqual(harness.states["lift"], self.receipt)
        self.assertEqual(harness.action_count, 0)
        robot.execute.assert_not_called()
        robot.observe.assert_not_called()
        self.assertIs(request.call_args.kwargs["cancel_event"], harness.cancel)
        self.assertNotIn("deadline", request.call_args.kwargs)

    def test_harness_binds_program_cancel_without_model_deadline(self):
        self.store.save("action_receipt", "lift", self.receipt)
        harness, robot = self.harness()
        cancel = threading.Event()
        harness.programs = ProgramRunner(program_services(harness))
        harness.programs.entries['program-fixture'] = dict(cancel=cancel, deadline=time.monotonic()+60)
        with patch.object(self.reviewer, "_request", return_value=self.raw(answer())) as request:
            returned = harness.dispatch("vision_review", dict(self.args, evidence_mode="operation"),
                                        dict(tool_call_id="program-verification", program_id="program-fixture"))
        self.assertTrue(returned["success"])
        self.assertEqual(request.call_args.kwargs, dict(cancel_event=cancel))
        robot.execute.assert_not_called()
        cancel.set()
        with patch.object(self.reviewer, "_request") as request:
            with self.assertRaisesRegex(ValueError, "program_channel_expired"):
                harness.dispatch("vision_review", dict(self.args, evidence_mode="operation"),
                                 dict(tool_call_id="expired-verification", program_id="program-fixture"))
            request.assert_not_called()

    def test_unified_visual_review_routes_operation_and_preserves_full_evidence(self):
        self.store.save("action_receipt", "lift", self.receipt)
        harness, robot = self.harness()
        with patch.object(self.reviewer, "_request", return_value=self.raw(answer())):
            response = harness.invoke("vision_review", dict(self.args, evidence_mode="operation"),
                                        dict(tool_call_id="unified-review"), role="actor")
        raw = response.data
        shown = json.loads(harness.actor_output.present(response)["contentItems"][0]["text"])
        self.assertEqual(shown["answer"]["assessment"], "met")
        self.assertEqual(shown["execution"]["action_ids"], ["lift"])
        self.assertNotIn("actions", shown["execution"])
        self.assertTrue(shown["details_ref"].startswith("artifact:ref_"))
        self.assertIn("actions", raw["execution"])
        self.assertEqual(self.store.read("action_receipt", "lift"), self.receipt)
        robot.execute.assert_not_called()

    def test_program_helper_can_exceed_nominal_budget_and_next_call_still_runs(self):
        self.store.save("action_receipt", "lift", self.receipt)
        harness, robot = self.harness()
        runner = harness.programs = ProgramRunner(program_services(harness))
        self.addCleanup(runner.close)
        runner.run(dict(source="pass", timeout_s=10),
                   dict(thread_id="thread", turn_id="turn", tool_call_id="warmup"))
        source = ("from ni_rua_client import NiRuaClient\nc=NiRuaClient()\n"
                  "print(c.call('vision_review', " + repr(dict(self.args, evidence_mode="operation")) + ")['success'])\n"
                  "print(c.call('robot_action_status', {'action_id':'lift'})['success'])\n")
        real_monotonic = time.monotonic
        offset = [0.0]
        real_remaining = program_remaining
        monitor_checked = threading.Event()
        advanced = False
        def tracked_remaining(entry):
            if advanced:
                monitor_checked.set()
            return real_remaining(entry)
        observed = {}
        def slow_request(*args, **kwargs):
            nonlocal advanced
            with runner.lock:
                entry = next(
                    item for item in runner.entries.values()
                    if item["job"].get("correlation", {}).get("tool_call_id") == "slow-sol"
                )
            observed["remaining_at_start"] = program_remaining(entry)
            observed["wait_started"] = entry["model_wait_started_at"]
            # The program monitor remains active and sees the original deadline
            # pass while only this program's Sol request is in flight.
            offset[0] += 1.25
            advanced = True
            self.assertTrue(monitor_checked.wait(0.5))
            observed["cancelled"] = entry["cancel"].is_set()
            observed["remaining_at_end"] = program_remaining(entry)
            return self.raw(answer())
        with patch("ni_rua.runtime.programs.runner.time.monotonic", side_effect=lambda: real_monotonic() + offset[0]), \
                patch("ni_rua.runtime.programs.runner.program_remaining", side_effect=tracked_remaining), \
                patch.object(self.reviewer, "_request", side_effect=slow_request):
            result = runner.run(dict(source=source, timeout_s=1, max_calls=2),
                                dict(thread_id="thread", turn_id="turn", tool_call_id="slow-sol"))
        self.assertEqual(result["status"], "succeeded", result.get("stderr_text"))
        self.assertEqual(result["calls"], 2)
        self.assertIn("True\nTrue", result["stdout_text"])
        self.assertIsNotNone(observed["wait_started"])
        self.assertFalse(observed["cancelled"])
        self.assertEqual(observed["remaining_at_start"], observed["remaining_at_end"])
        self.assertGreaterEqual(result["model_wait_s"], 1.25)
        self.assertEqual(result["timeout_semantics"], "work_wall_time_excluding_own_model_and_main_review_waits")
        robot.execute.assert_not_called()

    def test_explicit_program_cancel_still_interrupts_waiting_sol(self):
        self.store.save("action_receipt", "lift", self.receipt)
        harness, robot = self.harness()
        runner = harness.programs = ProgramRunner(program_services(harness))
        self.addCleanup(runner.close)
        entered = threading.Event()
        def waiting_request(*args, **kwargs):
            entered.set()
            self.assertTrue(kwargs["cancel_event"].wait(4), "Program cancellation did not reach Sol")
            raise TimeoutError("visual_review_program_cancelled")
        source = ("from ni_rua_client import NiRuaClient\n"
                  "NiRuaClient().call('vision_review', " + repr(dict(self.args, evidence_mode="operation")) + ")\n")
        results = []
        with patch.object(self.reviewer, "_request", side_effect=waiting_request):
            worker = threading.Thread(target=lambda: results.append(runner.run(dict(source=source, timeout_s=10),
                                      dict(thread_id="thread", turn_id="turn", tool_call_id="cancel-sol"))))
            worker.start()
            self.assertTrue(entered.wait(8))
            entry = next(iter(runner.entries.values()))
            runner.cancel(entry["job_id"])
            worker.join(5)
        self.assertFalse(worker.is_alive())
        self.assertEqual(results[0]["status"], "cancelled")
        self.assertTrue(results[0]["stop_confirmed"])
        self.assertIsNone(entry["model_wait_started_at"])
        robot.execute.assert_not_called()

    def test_harness_channel_restores_own_work_time_after_sol_wait(self):
        self.store.save("action_receipt", "lift", self.receipt)
        harness, robot = self.harness()
        clock = [100.0]
        entry = dict(cancel=threading.Event(), deadline=120.0)
        harness.programs = ProgramRunner(program_services(harness))
        harness.programs.entries['budget-fixture'] = entry
        correlation = dict(program_id="budget-fixture")
        with patch("ni_rua.runtime.programs.runner.time.monotonic", side_effect=lambda: clock[0]):
            with model_wait(entry):
                clock[0] += 1000
                self.assertIs(harness._review_controls(correlation)['cancel_event'], entry['cancel'])
            self.assertIs(harness._review_controls(correlation)['cancel_event'], entry['cancel'])
            self.assertEqual(program_remaining(entry), 20.0)
            clock[0] += 21
            with self.assertRaisesRegex(ValueError, "program_channel_expired"):
                harness._review_controls(correlation)

    def test_legacy_vision_review_pauses_only_its_program_request_wait(self):
        self.store.save("action_receipt", "lift", self.receipt)
        harness, robot = self.harness()
        clock = [100.0]
        entry = dict(cancel=threading.Event(), deadline=120.0)
        harness.programs = ProgramRunner(program_services(harness))
        harness.programs.entries['legacy-fixture'] = entry
        value = dict(visibility_after="visible", motion="displaced", after_center_px=[10, 10],
                     retained_with_gripper="visually_supported", evidence="Visible target moved with fingers.")
        raw = dict(model=MODEL, status="completed", answer=value)
        def request(*args, **kwargs):
            clock[0] += 1000
            self.assertEqual(program_remaining(entry), 20.0)
            return raw
        args = {key: self.args[key] for key in ("before_observation_id", "after_observation_id", "camera_id", "roi", "target")}
        with patch("ni_rua.runtime.programs.runner.time.monotonic", side_effect=lambda: clock[0]), \
                patch.object(self.reviewer, "_request", side_effect=request):
            result = harness.dispatch("vision_review", args, dict(program_id="legacy-fixture", tool_call_id="legacy-sol"))
            self.assertEqual(program_remaining(entry), 20.0)
        self.assertTrue(result["success"])
        self.assertEqual(entry["model_wait_s"], 1000.0)
        robot.execute.assert_not_called()

    def test_harness_disabled_or_absent_reviewer_rejects_before_processing(self):
        self.store.save("action_receipt", "lift", self.receipt)
        harness, robot = self.harness()
        with patch.object(self.reviewer, "_request") as request:
            harness.disabled_tools = frozenset({"operation_verify"})
            with self.assertRaisesRegex(ValueError, "tool_disabled_by_config:operation_verify"):
                harness.dispatch("vision_review", dict(self.args, evidence_mode="operation"),
                                 dict(tool_call_id="disabled"))
            harness.disabled_tools = frozenset({"vision_review"})
            with self.assertRaisesRegex(ValueError, "tool_disabled_by_config:vision_review"):
                harness.dispatch("vision_review", dict(self.args, evidence_mode="operation"),
                                 dict(tool_call_id="disabled-unified"))
            harness.disabled_tools = frozenset()
            harness.visual_reviewer = None
            with self.assertRaisesRegex(ValueError, "visual_review_not_enabled_for_run"):
                harness.dispatch("vision_review", dict(self.args, evidence_mode="operation"),
                                 dict(tool_call_id="unavailable"))
            request.assert_not_called()
        robot.execute.assert_not_called()


class OperationToolExposureTest(unittest.IsolatedAsyncioTestCase):
    async def test_actor_tools_follow_default_off_explicit_sol_and_disabled_tool(self):
        for overrides, enabled in (({}, False), ({"visual_review_model": MODEL}, True),
                                   ({"visual_review_model": None}, False),
                                   ({"visual_review_model": MODEL, "disabled_tools": ["operation_verify"]}, False)):
            with self.subTest(overrides=overrides), tempfile.TemporaryDirectory() as temporary:
                config = Config.from_dict(dict(features=dict(motion_review=False, **overrides)))
                workbench = Workbench(Path(temporary) / "run", config=config)
                try:
                    self.assertNotIn("vision_review", {tool["name"] for tool in workbench.actor_tools()})
                    self.assertIsNone(workbench.harness.visual_reviewer)
                    if enabled:
                        workbench.harness.visual_reviewer = Mock()
                    catalog = workbench.actor_tools()
                    names = {tool["name"] for tool in catalog}
                    self.assertNotIn("operation_verify", names)
                    self.assertEqual("vision_review" in names, enabled)
                    if "vision_review" in names:
                        schema = next(tool["inputSchema"] for tool in catalog if tool["name"] == "vision_review")
                        self.assertEqual("oneOf" in schema, enabled)
                    self.assertEqual(workbench.visual_review_model, config.features.visual_review_model)
                    self.assertIsNone(workbench.robot.process)
                    if not overrides:
                        self.assertIsNone(workbench.visual_review_model)
                    self.assertIn("action_ids", TOOL_REQUIRED)
                finally:
                    await workbench.close()



# --- Folded from tests/test_robot_calibration_information.py ---

def translation(x=0., y=0., z=0.):
    return [[1., 0., 0., x], [0., 1., 0., y], [0., 0., 1., z], [0., 0., 0., 1.]]


class CalibratedRobot(FixtureRobot):
    def __init__(self, root):
        super().__init__(root)
        self.observations = 0

    def observe(self):
        self.observations += 1
        raw = super().observe()
        frames, calibration, poses = [], {}, {}
        for name, arm in (("head", None), ("left_wrist", "left"), ("right_wrist", "right")):
            frame_id = f"{name}-{self.observations}"
            frames.append(dict(camera_id=name, frame_id=frame_id, png_path=str(self.path), width=1, height=1))
            calibration[name] = dict(K=[[123.45678901234567, 0., .5], [0., 124.98765432109876, .5], [0., 0., 1.]],
                distortion=[.0123456789012345, 0., 0., 0., 0.], distortion_model="fixture_distortion",
                image_size=[1, 1], optical_axes="+x right,+y down,+z forward")
            poses[name] = dict(frame_id=frame_id, T_world_camera=translation(.12345678901234567, .2, .3),
                               source_kind="robot_kinematics" if arm else "simulator_camera_transform")
            if arm:
                calibration[name].update(arm=arm, link_name="link6", T_link_camera=translation(.0123456789012345),
                                         T_world_base=translation(99.))
                poses[name].update(arm=arm, link_name="link6", T_world_link=translation(.25),
                                   T_world_base=translation(-.345678901234567 if arm == "left" else .345678901234567))
            else:
                calibration[name]["T_world_camera"] = translation(88.)
        raw.update(observation_profile=ACTOR_PROFILE, cameras=frames,
            camera_calibration=dict(reference_frame_id="env_origin", cameras=calibration),
            camera_pose_sample=dict(reference_frame_id="env_origin", sim_step=self.executions,
                                    step_domain=raw["step_domain"], cameras=poses),
            robot_model=dict(base_link="base_link", joints=[]))
        self.last_raw = deepcopy(raw)
        return raw


class RobotCalibrationInformationTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        identity = dict(run_id="calibration", episode_id="episode", env_instance_id="environment", env_generation=2)
        self.store = Store(self.root / "run", identity)
        self.addCleanup(self.store.close)
        self.robot = CalibratedRobot(self.root)
        self.harness = Harness(self.store, self.robot, observation_profile=ACTOR_PROFILE)
        self.harness.start()

    def replace_artifact(self, field, kind, update, *, scope="actor"):
        reference = self.harness.latest["cameras"][0][field]
        _, content = self.store.resolve_artifact(reference)
        value = json.loads(content)
        update(value)
        changed = self.store.put_artifact(canonical(value).encode(), kind, "application/json", scope)
        for frame in self.harness.latest["cameras"]:
            frame[field] = changed

    def test_exact_latest_camera_and_base_values_without_new_capture(self):
        before = self.store.db.total_changes
        calls = self.robot.observations
        result = robot_calibration_information(self.harness)
        self.assertTrue(result["available"], result)
        self.assertEqual(result["observation_id"], self.harness.latest["observation_id"])
        self.assertEqual(result["env_generation"], 2)
        self.assertEqual(result["sim_step"], 0)
        self.assertEqual(result["reference_frame_id"], "env_origin")
        self.assertEqual(result["units"]["transform_translation"], "m")
        raw = self.robot.last_raw
        for name, camera in result["cameras"].items():
            for key in ("K", "distortion", "image_size", "optical_axes"):
                self.assertEqual(camera[key], raw["camera_calibration"]["cameras"][name][key])
            self.assertEqual(camera["T_world_camera"], raw["camera_pose_sample"]["cameras"][name]["T_world_camera"])
            self.assertNotIn("T_world_base", camera)
        for arm in ("left", "right"):
            self.assertEqual(result["bases"][arm]["T_world_base"],
                             raw["camera_pose_sample"]["cameras"][arm + "_wrist"]["T_world_base"])
            self.assertEqual(result["cameras"][arm + "_wrist"]["T_link_camera"],
                             raw["camera_calibration"]["cameras"][arm + "_wrist"]["T_link_camera"])
            self.assertNotIn("K", result["bases"][arm])
        self.assertEqual(self.robot.observations, calls)
        self.assertEqual(self.robot.executions, 0)
        self.assertEqual(self.store.db.total_changes, before)
        # Independent results cannot mutate retained numeric evidence.
        result["cameras"]["left_wrist"]["K"][0][0] = -1
        self.assertEqual(robot_calibration_information(self.harness)["cameras"]["left_wrist"]["K"],
                         raw["camera_calibration"]["cameras"]["left_wrist"]["K"])

    def test_reads_newly_published_anchor_and_never_static_base_fallback(self):
        old_id = self.harness.latest["observation_id"]
        raw = self.robot.observe()
        raw["sim_step"] = 17
        raw["camera_pose_sample"]["sim_step"] = 17
        del raw["camera_pose_sample"]["cameras"]["left_wrist"]["T_world_base"]
        self.harness.publish_observation(raw)
        result = robot_calibration_information(self.harness)
        self.assertTrue(result["available"], result)
        self.assertNotEqual(result["observation_id"], old_id)
        self.assertEqual(result["sim_step"], 17)
        self.assertNotIn("left", result["bases"])
        self.assertIn("bases.left.T_world_base", result["unavailable_fields"])

    def test_profile_rejection_happens_before_any_artifact_read(self):
        for profile in (LEGACY_ACTOR_PROFILE, None, "unknown"):
            store = Mock()
            result = robot_calibration_information(SimpleNamespace(observation_profile=profile, store=store))
            self.assertFalse(result["available"])
            self.assertEqual(result["reason"], "calibration_not_permitted_by_observation_profile")
            store.resolve_artifact.assert_not_called()
        result = robot_calibration_information(SimpleNamespace(observation_profile=ACTOR_PROFILE))
        self.assertFalse(result["available"])
        self.assertEqual(result["reason"], "latest_observation_unavailable")

    def test_wrong_generation_or_profile_cannot_relabel_old_observation(self):
        self.harness.latest["env_generation"] = 1
        self.assertEqual(robot_calibration_information(self.harness)["reason"], "calibration_observation_identity_mismatch")
        self.harness.latest["env_generation"] = 2
        self.harness.latest["observation_profile"] = LEGACY_ACTOR_PROFILE
        self.assertFalse(robot_calibration_information(self.harness)["available"])

    def test_pose_step_and_frame_must_match_published_camera(self):
        for mutation, expected in ((lambda value: value.update(sim_step=99), "calibration_pose_step_mismatch"),
                                  (lambda value: value["cameras"]["head"].update(frame_id="another-frame"), "calibration_pose_frame_mismatch")):
            with self.subTest(expected=expected):
                original = deepcopy(self.harness.latest)
                self.replace_artifact("pose_sample_ref", "camera_pose_sample", mutation)
                self.assertEqual(robot_calibration_information(self.harness)["reason"], expected)
                self.harness.latest = original

    def test_scope_and_privilege_are_checked_before_projection(self):
        for scope, mutation in (("observer", lambda value: None),
                                ("actor", lambda value: value.update(evaluator_state={"answer": True}))):
            with self.subTest(scope=scope):
                original = deepcopy(self.harness.latest)
                self.replace_artifact("calibration_ref", "camera_calibration", mutation, scope=scope)
                result = robot_calibration_information(self.harness)
                self.assertFalse(result["available"])
                self.assertNotIn("cameras", result)
                self.harness.latest = original

    def test_missing_sources_and_wrong_artifact_kind_are_unavailable(self):
        original = deepcopy(self.harness.latest)
        del self.harness.latest["cameras"][0]["pose_sample_ref"]
        self.assertEqual(robot_calibration_information(self.harness)["reason"], "calibration_artifact_reference_missing")
        self.harness.latest = original
        self.replace_artifact("calibration_ref", "computed", lambda value: None)
        self.assertEqual(robot_calibration_information(self.harness)["reason"], "calibration_artifact_kind_mismatch")

    def test_absent_optical_or_hand_eye_fields_are_explicitly_unknown(self):
        def remove_fields(value):
            del value["cameras"]["left_wrist"]["T_link_camera"]
            del value["cameras"]["head"]["distortion"]
        self.replace_artifact("calibration_ref", "camera_calibration", remove_fields)
        result = robot_calibration_information(self.harness)
        self.assertTrue(result["available"], result)
        self.assertNotIn("T_link_camera", result["cameras"]["left_wrist"])
        self.assertNotIn("distortion", result["cameras"]["head"])
        self.assertIn("cameras.left_wrist.T_link_camera", result["unavailable_fields"])
        self.assertIn("cameras.head.distortion", result["unavailable_fields"])



# --- Folded from tests/test_robot_information_integration.py ---

def text_result(response):
    return json.loads(response["contentItems"][0]["text"])


def wire_information(workbench):
    harness = workbench.harness
    harness.capabilities["robot_model_id"] = "arx-x5"
    harness.store = workbench.store
    harness.actor_presentation = ActorPresentation(workbench.store)
    harness.actor_output = ActorOutput(workbench.store, harness.actor_presentation)
    harness._archive_tool_result = MethodType(Harness._archive_tool_result, harness)
    harness.robot_information = MethodType(Harness.robot_information, harness)


class RobotInformationStartupTest(unittest.IsolatedAsyncioTestCase):
    async def test_unregistered_robot_does_not_claim_to_supply_a_urdf(self):
        with tempfile.TemporaryDirectory() as temporary:
            workbench, terminal = fixture_workbench(Path(temporary) / "run")
            wire_information(workbench)
            workbench.harness.capabilities["robot_model_id"] = "unregistered"
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                    await workbench.start("fixture")
                instructions = workbench.session.start.call_args.args[1]
                data = json.loads(instructions.split("\n\nHarness instructions\n", 1)[0]
                                  .removeprefix("Robot physical information\n"))
                self.assertFalse(data["available"])
                self.assertNotIn("urdf", data)
                self.assertNotIn("Robot URDF", instructions)
                self.assertNotIn("The original URDF is supplied above", instructions)
            finally:
                await workbench.close()

    async def test_default_precedes_instructions_and_task_notes_are_last(self):
        with tempfile.TemporaryDirectory() as temporary:
            # The inline form of the URDF (the file form is test_urdf_delivery_file_keeps_the_urdf_out_of_the_startup_context)
            workbench, terminal = fixture_workbench(
                Path(temporary) / "run", default_task_notes=False,
                config=Config.from_dict({"features": {"robot_urdf_delivery": "inline"}}))
            wire_information(workbench)
            robot = CalibratedRobot(Path(temporary))
            published = Harness(workbench.store, robot, observation_profile=ACTOR_PROFILE)
            published.start()
            # Exercise startup with the actual publication path and its exact
            # episode artifacts, while keeping the process-free Workbench shell.
            for field in ("identity", "observation_profile", "latest"):
                setattr(workbench.harness, field, deepcopy(getattr(published, field)))
            workbench.harness.latest["task_instruction"] = "Fixture native task."
            capture_count = robot.observations
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                    await workbench.start("unused", benchmark=True, task_notes="Keep task notes last.",
                                          task_notes_source="https://example.test/unneeded-source")
                instructions = workbench.session.start.call_args.args[1]
                prefix, behavior = instructions.split("\n\nHarness instructions\n", 1)
                metadata, urdf = prefix.removeprefix("Robot physical information\n").split("\n\nRobot URDF\n", 1)
                data = json.loads(metadata)
                self.assertNotIn("urdf", data)
                self.assertNotIn("joints", data["reference"]["kinematics"])
                with zipfile.ZipFile(robot_resource_path(resource_id="geometry_model")) as archive:
                    self.assertEqual(urdf.encode("utf-8"), archive.read("X5A.urdf"))
                self.assertEqual(instructions.count(urdf), 1)
                data["urdf"] = urdf
                self.assertEqual(data, workbench.harness.robot_information())
                self.assertNotIn("dynamics", data["reference"])
                self.assertEqual(data["reference"]["identity"]["native_action_dimensions"], 14)
                self.assertNotIn("planning_collision", data["reference"]["geometry"])
                self.assertRegex(data["reference"]["geometry"]["resource_ref"], r"^artifact:ref_\d+$")
                calibration = data["calibration"]
                self.assertTrue(calibration["available"], calibration)
                self.assertEqual(workbench.harness.actor_output.restore(calibration["observation_id"]),
                                 published.latest["observation_id"])
                for arm in ("left", "right"):
                    camera = arm + "_wrist"
                    self.assertEqual(calibration["bases"][arm]["T_world_base"],
                                     robot.last_raw["camera_pose_sample"]["cameras"][camera]["T_world_base"])
                    self.assertNotEqual(calibration["bases"][arm]["T_world_base"],
                                        robot.last_raw["camera_calibration"]["cameras"][camera]["T_world_base"])
                    self.assertEqual(calibration["cameras"][camera]["K"],
                                     robot.last_raw["camera_calibration"]["cameras"][camera]["K"])
                self.assertNotIn("default_mounts", data["reference"]["kinematics"])
                self.assertEqual(robot.observations, capture_count)
                self.assertEqual(robot.executions, 0)
                self.assertTrue(behavior.startswith(TASK_INSTRUCTIONS))
                expected_execution = INSTRUCTIONS.replace(
                    "Use native shell and file tools for computation, files and memory inside your sandbox.\n"
                    "Read the public Harness implementation under /harness. Install Python packages with python3 -m pip;\n"
                    "packages and working files stay in your sandbox. Web search and general network access are unavailable.\n",
                    "Use native shell and file tools for computation, files and memory in your working directory.\n"
                    f"Read the public Harness implementation under {workbench.root / 'private' / 'actor-source'}.\n"
                    "Code runs without OS sandbox isolation. Web search is disabled.\n",
                )
                self.assertLess(behavior.index(PLANNING_INSTRUCTIONS), behavior.index(expected_execution))
                self.assertEqual(behavior.count(PLANNING_INSTRUCTIONS), 1)
                prompt = workbench.session.run_turn.call_args.args[0]
                self.assertTrue(prompt.endswith("\nTask notes:\nKeep task notes last."))
                self.assertNotIn("https://", prompt)
                self.assertNotIn("unneeded-source", prompt)
                self.assertNotRegex(instructions, r"\b[0-9a-f]{40,}\b|sha256|source_url|source_filename")
                self.assertEqual(workbench.store.read_state("runtime", "robot_information"), data)
                self.assertEqual(sum(e["type"] == "robot_information.initialized" for e in workbench.store.events()), 1)
                for name in ("execute", "reset", "evaluate"):
                    getattr(workbench.robot, name).assert_not_called()
            finally:
                await workbench.close()

    async def start_with(self, features, *, capabilities=None, **options):
        """``(instructions, actor_dir, workbench information, actor tools)`` of a fixture startup under ``features``
        overrides; ``capabilities``: the session's ``AgentCapabilities`` (default: the fixture's Codex host)."""
        directory = self.enterContext(tempfile.TemporaryDirectory())
        config = Config.from_dict({"features": features})
        workbench, terminal = fixture_workbench(Path(directory) / "run", config=config, **options)
        if capabilities is not None:
            workbench.session.capabilities = capabilities
        # The fixture replaces the harness after Workbench.__init__ set this flag from the session's capabilities.
        from ni_rua.cortex.agents.base import session_capabilities
        from ni_rua.cortex.agents.codex.session import CodexSession
        workbench.harness.deferred_tool_loading = bool(
            session_capabilities(workbench.session, CodexSession.capabilities).deferred_tools)
        wire_information(workbench)
        try:
            with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                await workbench.start("fixture")
            return (workbench.session.start.call_args.args[1], Path(workbench.session.actor_dir),
                    workbench.harness.robot_information(), workbench.actor_tools())
        finally:
            await workbench.close()

    async def test_urdf_delivery_file_keeps_the_urdf_out_of_the_startup_context(self):
        urdf = robot_information()["urdf"]
        instructions, actor_dir, information, _ = await self.start_with({"robot_urdf_delivery": "inline"})
        self.assertEqual(instructions.count(urdf), 1)
        self.assertFalse((actor_dir / "robot.urdf").exists())
        # the default is the file
        for features in ({}, {"robot_urdf_delivery": "file"}):
            instructions, actor_dir, information, _ = await self.start_with(features)
            self.assertNotIn(urdf, instructions)
            self.assertEqual((actor_dir / "robot.urdf").read_bytes(), urdf.encode("utf-8"))
            self.assertEqual(information["urdf"], urdf)
            self.assertNotIn("Robot URDF\n", instructions)

    async def test_urdf_file_delivery_without_a_working_directory_stays_inline(self):
        """无 actor 工作目录的托管会话桩回退内联 URDF；external 后端在 robot_urdf_delivery='file' 下改为省略内联大 URDF 并引导按需调 robot_information（零路径泄露）。"""
        from ni_rua.cortex.context.actor_instructions import (
            EXTERNAL_URDF_ON_DEMAND_CLAUSE,
            EXTERNAL_URDF_ON_DEMAND_GUIDANCE,
        )
        from ni_rua.runtime.tools.catalog_guide import REFERENCE_NAME
        urdf = robot_information()["urdf"]
        for kind in ("external", None):
            with self.subTest(session=kind or "no actor_dir"):
                directory = self.enterContext(tempfile.TemporaryDirectory())
                workbench, terminal = fixture_workbench(Path(directory) / "run", config=Config.from_dict(
                    {"features": {"robot_urdf_delivery": "file", "tool_catalog_profile": "full_v1"}}))
                if kind is None:
                    workbench.session.actor_dir = None
                else:
                    workbench.session.backend_kind = kind
                wire_information(workbench)
                try:
                    with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                        await workbench.start("fixture")
                    instructions = workbench.session.start.call_args.args[1]
                    if kind == "external":
                        self.assertNotIn(urdf, instructions)
                        self.assertNotIn("Robot URDF\n", instructions)
                        self.assertIn(EXTERNAL_URDF_ON_DEMAND_CLAUSE, instructions)
                        self.assertIn(EXTERNAL_URDF_ON_DEMAND_GUIDANCE, instructions)
                    else:
                        self.assertEqual(instructions.count(urdf), 1)
                    self.assertNotIn("is the file robot.urdf", instructions)
                    self.assertFalse((Path(directory) / "run" / "robot.urdf").exists())
                    self.assertFalse((Path(directory) / "run" / REFERENCE_NAME).exists())
                finally:
                    await workbench.close()

    async def test_layered_catalogue_publishes_tool_reference_next_to_the_actor(self):
        from ni_rua.runtime.tools.catalog_guide import REFERENCE_NAME, reference_document
        _, actor_dir, _, tools = await self.start_with({"tool_catalog_profile": "layered_v1"})
        expected = reference_document(tools)
        self.assertIsNotNone(expected)
        self.assertEqual((actor_dir / REFERENCE_NAME).read_bytes(), expected.encode("utf-8"))
        for profile in ("layered_v2", "layered_v3"):
            _, actor_dir, _, tools = await self.start_with({"tool_catalog_profile": profile})
            self.assertEqual((actor_dir / REFERENCE_NAME).read_bytes(), reference_document(tools).encode("utf-8"), profile)
        for profile in ("full_v1", "compact_v1"):
            _, actor_dir, _, _ = await self.start_with({"tool_catalog_profile": profile})
            self.assertFalse((actor_dir / REFERENCE_NAME).exists(), profile)

    async def test_layered_v2_names_the_on_demand_tools_only_for_a_host_that_defers_them(self):
        from ni_rua.cortex.agents.base import AgentCapabilities
        from ni_rua.cortex.agents.codex.session import CodexSession
        from ni_rua.runtime.tools.catalog_guide import ON_DEMAND_HEADER, REFERENCE_NAME, on_demand_index, reference_document
        first_line = ON_DEMAND_HEADER.split("\n")[0]
        self.assertTrue(CodexSession.capabilities.deferred_tools)
        instructions, actor_dir, _, tools = await self.start_with({"tool_catalog_profile": "layered_v2"})
        index = on_demand_index(tools)
        self.assertIsNotNone(index)
        self.assertEqual(instructions.count(index), 1)
        self.assertEqual(instructions.count(first_line), 1)
        self.assertEqual((actor_dir / REFERENCE_NAME).read_bytes(), reference_document(tools).encode("utf-8"))
        instructions, actor_dir, _, _ = await self.start_with({"tool_catalog_profile": "layered_v2"},
                                                             capabilities=AgentCapabilities(deferred_tools=False))
        self.assertNotIn(first_line, instructions)
        self.assertTrue((actor_dir / REFERENCE_NAME).is_file())
        for profile in ("full_v1", "compact_v1", "layered_v1", "layered_v3"):
            instructions, _, _, _ = await self.start_with({"tool_catalog_profile": profile})
            self.assertNotIn(first_line, instructions, profile)

    async def test_resume_reinitializes_information_without_starting_an_extra_task_turn(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary) / "run"
            inline = Config.from_dict({"features": {"robot_urdf_delivery": "inline"}})
            first, terminal = fixture_workbench(root, config=inline)
            wire_information(first)
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                    await first.start("fixture")
            finally:
                await first.close()
            resumed, terminal = fixture_workbench(root, resume=True, config=inline)
            wire_information(resumed)
            try:
                with patch("ni_rua.apps.gateway.workbench.Terminal", return_value=terminal):
                    await resumed.start("unused")
                self.assertTrue(resumed.session.start.call_args.args[1].startswith("Robot physical information\n"))
                self.assertEqual(resumed.session.start.call_args.args[1].count(robot_information()["urdf"]), 1)
                resumed.session.run_turn.assert_not_called()
                self.assertEqual(sum(e["type"] == "robot_information.initialized" for e in resumed.store.events()), 2)
            finally:
                await resumed.close()


class RobotInformationToolTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        identity = dict(run_id="physical", episode_id="ep", env_instance_id="env", env_generation=1)
        self.store = Store(self.root / "run", identity)
        self.addCleanup(self.store.close)
        self.robot = FixtureRobot(self.root)
        self.harness = Harness(self.store, self.robot)
        self.harness.start()
        self.harness.capabilities["robot_model_id"] = "arx-x5"
        self.harness.thread_id = "thread"
        self.counter = 0

    def call(self, tool="robot_information", arguments=None):
        self.counter += 1
        return self.harness.handle_tool(dict(threadId="thread", turnId="turn", callId=f"call-{self.counter}",
                                             tool=tool, arguments=arguments or {}))

    def test_tool_and_initial_read_match_including_legacy_static_knowledge(self):
        self.assertEqual(self.harness.observation_profile, LEGACY_ACTOR_PROFILE)
        # The actor reads the default display (lean_v2: floats with six decimals) of the typed information.
        default = text_result(self.call())
        self.assertEqual(default, lean_v2_display(self.harness.robot_information()))
        self.assertFalse(default["calibration"]["available"])
        static = deepcopy(default)
        static.pop("calibration")
        static["reference"]["geometry"].pop("resource_ref")
        self.assertEqual(static, lean_v2_display(robot_information()))
        section = text_result(self.call(arguments=dict(section="kinematics")))
        self.assertEqual(section["reference"]["kinematics"],
                         lean_v2_display(robot_information(section="all")["reference"]["kinematics"]))
        self.assertIn("declared_limits", section["reference"]["kinematics"]["joints"][0])
        self.assertNotIn("joints", default["reference"]["kinematics"])
        self.assertNotIn("briefing", section)
        self.assertFalse(text_result(self.call(arguments=dict(robot_id="another-arm")))["available"])
        self.harness.capabilities["robot_model_id"] = "unregistered"
        self.assertFalse(text_result(self.call())["available"])
        self.assertEqual(self.robot.executions, 0)
        self.assertEqual(self.harness.native_action_count, 0)

    def test_illustration_and_cad_are_only_returned_when_requested(self):
        self.assertEqual(len(self.call()["contentItems"]), 1)
        image = self.call(arguments=dict(section="geometry_image"))
        self.assertEqual(image["contentItems"][1]["type"], "inputImage")
        resources = text_result(self.call(arguments=dict(section="resources")))
        self.assertEqual(len(resources["resources"]), 5)
        resource = text_result(self.call(arguments=dict(section="resource", resource_id="base_step")))
        self.assertRegex(resource["resource_ref"], r"^artifact:ref_\d+$")
        self.assertNotRegex(json.dumps(resource), r'"(?:source|sources|source_url|source_filename|sha256|hash)"|https?://|\b[0-9a-f]{40,}\b')
        raw_ref = self.harness.actor_presentation.restore(resource["resource_ref"])
        self.assertTrue(self.store.resolve_artifact(raw_ref)[1].startswith(b"ISO-10303-21;"))
        self.assertFalse(text_result(self.call(arguments=dict(section="resource", resource_id="../unknown")))["available"])
        self.assertEqual(self.robot.executions, 0)

    def test_invalid_arguments_have_explicit_errors_without_actions(self):
        cases = [
            (dict(section=[]), "invalid_tool_arguments"),
            (dict(section=""), "invalid_tool_arguments"),
            (dict(robot_id=[]), "invalid_tool_arguments"),
            (dict(section="resource"), "invalid_tool_arguments"),
            (dict(resource_id="base_step"), "invalid_tool_arguments"),
            (dict(unregistered_field=True), "invalid_tool_arguments"),
        ]
        for arguments, error in cases:
            with self.subTest(arguments=arguments), self.assertRaisesRegex(ValueError, error):
                self.call(arguments=arguments)
        self.assertEqual(self.robot.executions, 0)

    def test_program_artifact_reads_keep_short_references_and_remove_metadata(self):
        self.harness.observation_profile = ACTOR_PROFILE
        nested = self.store.put_artifact(b"binary", "rgb", "application/octet-stream")
        raw = json.dumps(dict(source_sha256="a"*64, source_url="https://example.test/model",
                              joint_angles=[1.23456789012345], nested_ref=nested)).encode()
        reference = self.store.put_artifact(raw, "robot_model", "application/json")
        short = self.harness.actor_presentation.present(reference)
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+10, call_lock=threading.RLock())
        result = runner.dispatch(entry, dict(artifact_ref=short))
        data = json.loads(base64.b64decode(result["data_base64"]))
        self.assertEqual(data["joint_angles"], [1.23456789012345])
        self.assertNotRegex(json.dumps(result["artifact"])+json.dumps(data), r"sha256|https://|\b[0-9a-f]{40,}\b")
        self.assertRegex(data["nested_ref"], r"^artifact:ref_\d+$")
        self.assertEqual(self.store.resolve_artifact(reference)[1], raw)
        # A public robot knowledge tool must not lift the old profile's exact
        # scene/camera calibration or evaluator-artifact restrictions.
        self.harness.observation_profile = LEGACY_ACTOR_PROFILE
        with self.assertRaisesRegex(ValueError, "calibrated_artifact"):
            runner.dispatch(entry, dict(artifact_ref=short))

    def test_observation_and_capabilities_remove_hashes_but_keep_usable_evidence(self):
        self.harness.capabilities.update(source_sha256="a"*64, source_url="https://example.test/manual")
        observation = text_result(self.call("robot_observe", dict(detail="full")))
        capabilities = text_result(self.call("robot_capabilities", dict(detail="full")))
        self.assertNotRegex(json.dumps([observation, capabilities]),
                            r'"(?:sha256|source_sha256|source_url)"|https?://|\b[0-9a-f]{40,}\b')
        self.assertEqual(observation["robot_state"]["source_kind"], "measurement")
        reference = observation["bundle"]["cameras"][0]["rgb_ref"]
        self.assertRegex(reference, r"^artifact:ref_\d+$")
        original = self.harness.actor_presentation.restore(reference)
        runner = ProgramRunner(program_services(self.harness))
        self.addCleanup(runner.close)
        entry = dict(cancel=threading.Event(), deadline=time.monotonic()+10, call_lock=threading.RLock())
        result = runner.dispatch(entry, dict(artifact_ref=reference))
        self.assertEqual(base64.b64decode(result["data_base64"]), self.store.resolve_artifact(original)[1])
        self.assertEqual(self.robot.executions, 0)

    def test_real_actor_program_can_reread_default_and_follow_an_alias(self):
        runner = ProgramRunner(program_services(self.harness))
        self.harness.programs = runner
        self.addCleanup(runner.close)
        reference = self.store.put_artifact(b'{"sha256":"discard","position_m":[1,2,3]}', "computed", "application/json")
        alias = self.harness.actor_presentation.present(reference)
        source = ("import json, io, zipfile, xml.etree.ElementTree as ET\nfrom ni_rua_client import NiRuaClient\nc=NiRuaClient()\n"
                  "r=c.call('robot_information', {})\np=json.loads(r['contentItems'][0]['text'])\n"
                  "assert p['available'] and p['reference']['identity']['native_action_dimensions']==14\n"
                  "with zipfile.ZipFile(io.BytesIO(c.artifact(p['reference']['geometry']['resource_ref']))) as z:\n"
                  "    assert p['urdf'].encode('utf-8') == z.read(p['reference']['geometry']['assembly_member'])\n"
                  "    assert 'joints' not in p['reference']['kinematics']\n"
                  "    arm=ET.fromstring(z.read(p['reference']['geometry']['assembly_member']))\n"
                  "    assert len(arm.findall('link')) == 11\n"
                  "    assert z.read('meshes/link7.STL')[84:]\n"
                  f"a=json.loads(c.artifact({alias!r}))\n"
                  "assert a == {'position_m':[1,2,3]}\nprint('physical_reference_and_alias_read')\n")
        result = runner.run(dict(source=source, timeout_s=10, max_calls=1),
            dict(thread_id="thread", turn_id="turn", tool_call_id="program", item_id="program"))
        self.assertEqual(result["status"], "succeeded", self.store.resolve_artifact(result["stderr_ref"])[1])
        self.assertIn(b"physical_reference_and_alias_read", self.store.resolve_artifact(result["stdout_ref"])[1])
        self.assertEqual(self.robot.executions, 0)

    def test_default_and_calibration_tool_deliver_current_exact_calibration(self):
        robot = CalibratedRobot(self.root)
        harness = Harness(self.store, robot, observation_profile=ACTOR_PROFILE)
        harness.start()
        harness.capabilities["robot_model_id"] = "arx-x5"
        harness.thread_id = "thread"
        self.harness = harness
        calls = robot.observations
        default = text_result(self.call())
        calibration = text_result(self.call(arguments=dict(section="calibration")))
        self.assertTrue(calibration["available"], calibration)
        self.assertEqual(default["calibration"], calibration["calibration"])
        # default display: six decimals; the typed result keeps every digit of the calibration
        exact_k = robot.last_raw["camera_calibration"]["cameras"]["left_wrist"]["K"]
        self.assertEqual(calibration["calibration"]["cameras"]["left_wrist"]["K"], lean_v2_display(exact_k))
        self.assertEqual(harness.robot_information(section="calibration")["calibration"]["cameras"]["left_wrist"]["K"],
                         exact_k)
        self.assertEqual(harness.actor_output.restore(calibration["calibration"]["observation_id"]),
                         harness.latest["observation_id"])
        self.assertFalse(text_result(self.call(arguments=dict(section="calibration", robot_id="another-arm")))["available"])
        self.assertEqual(robot.observations, calls)
        self.assertEqual(robot.executions, 0)



if __name__ == "__main__":
    unittest.main()
