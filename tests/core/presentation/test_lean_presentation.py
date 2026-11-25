"""Actor payload compact_v1 projection and ActorOutput compact mode contract tests."""
import copy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

from ni_rua.core.config import Config
from ni_rua.core.contracts import canonical
from ni_rua.core.presentation.actor_output import ActorOutput, compact_model_value
from ni_rua.core.presentation.actor_payload import compact_action_result, compact_observation
from ni_rua.core.presentation.actor_presentation import ActorPresentation, metadata_findings
from ni_rua.core.storage.store import Store
from ni_rua.core.tool_output import StructuredText, ToolResponse


NI_RUA_TEST_TIER = "fast"


class CompactPayloadContractTests(unittest.TestCase):
    def test_lean_compact_module_and_flag_are_retired(self):
        self.assertIsNone(importlib.util.find_spec("ni_rua.core.presentation.lean_compact"))
        self.assertFalse(Config().features.lean_compact)
        migrated = Config.from_dict({"version": 1, "features": {"lean_compact": True}})
        self.assertEqual(migrated.version, 2)
        self.assertFalse(migrated.features.lean_compact)
        self.assertNotIn("lean_compact", migrated.to_dict()["features"])

    def test_compact_observation_retains_measurements_clocks_and_unknowns(self):
        raw = {
            "bundle": {
                "schema_version": "0.1",
                "run_id": "r1",
                "episode_id": "ep1",
                "env_instance_id": "env1",
                "observation_id": "obs_1",
                "sim_step": 240,
                "robot_state_ref": "artifact:" + "a" * 64 + "-" + "b" * 16,
                "reconstruction_warning": "depth_missing",
                "cameras": [
                    {
                        "camera_id": "cam_head",
                        "width": 640,
                        "height": 480,
                        "image_encoding": {"format": "jpeg"},
                        "sha256": "0" * 64,
                        "source_sha256": "1" * 64,
                        "capture_time": "2026-11-17T00:00:00Z",
                        "clock_domain": "sim",
                    }
                ],
            },
            "robot_state": {
                "arms": {
                    "left": {
                        "q_rad": [0.123456789, -0.987654321, float("nan")],
                        "gripper": {"goal_normalized": 0.0, "position_m": [0.02551234, 0.02581234]},
                    }
                }
            },
            "image_view": {"camera_id": "cam_head", "roi": [10, 20, 128, 128]},
        }
        original = copy.deepcopy(raw)
        projected = compact_observation(raw)
        self.assertEqual(
            raw["bundle"]["schema_version"], original["bundle"]["schema_version"]
        )  # input is never mutated
        for omitted in ("schema_version", "run_id", "episode_id", "env_instance_id", "robot_state_ref"):
            self.assertNotIn(omitted, projected["bundle"])
        self.assertEqual(projected["bundle"]["observation_id"], "obs_1")
        self.assertEqual(projected["bundle"]["sim_step"], 240)
        self.assertEqual(projected["bundle"]["reconstruction_warning"], "depth_missing")
        cam = projected["bundle"]["cameras"][0]
        self.assertEqual(cam, {"camera_id": "cam_head", "width": 640, "height": 480})
        q = projected["robot_state"]["arms"]["left"]["q_rad"]
        self.assertEqual(q[:2], [0.123457, -0.987654])
        self.assertTrue(math.isnan(q[2]))
        # Gripper command is never promoted to a contact claim; measured jaw positions stay distinct.
        self.assertEqual(
            projected["robot_state"]["arms"]["left"]["gripper"],
            {"goal_normalized": 0.0, "position_m": [0.025512, 0.025812]},
        )
        self.assertEqual(projected["image_view"], {"camera_id": "cam_head", "roi": [10, 20, 128, 128]})

    def test_compact_action_result_keeps_diagnostics_and_reduces_request_echo(self):
        raw = {
            "request": {
                "action_id": "act_1",
                "based_on_observation_id": "obs_1",
                "stages": [{"id": "s1", "kind": "move", "arm": "left"}],
            },
            "receipt": {
                "action_id": "act_1",
                "status": "failed",
                "reason": "tracking_error",
                "executed_native_steps": 14,
                "position_error_m": 0.01843219,
                "contact_block_summary": {"blocked_above_target": True, "approach_shortfall_m": 0.0213456},
                "versions": {"worker": "1"},
                "step_semantics": "native_control_steps",
                "trace_sampling": "boundary",
                "planned_trace_ref": "artifact:1",
                "dispatched_trace_ref": "artifact:2",
                "actual_trace_ref": "artifact:3",
            },
        }
        projected = compact_action_result(raw)
        self.assertEqual(projected["request"], {"action_id": "act_1", "based_on_observation_id": "obs_1"})
        receipt = projected["receipt"]
        for omitted in (
            "versions",
            "step_semantics",
            "trace_sampling",
            "planned_trace_ref",
            "dispatched_trace_ref",
            "actual_trace_ref",
        ):
            self.assertNotIn(omitted, receipt)
        self.assertEqual((receipt["status"], receipt["reason"]), ("failed", "tracking_error"))
        self.assertEqual(receipt["position_error_m"], 0.018432)
        self.assertEqual(
            receipt["contact_block_summary"],
            {"blocked_above_target": True, "approach_shortfall_m": 0.021346},
        )

    def test_actor_output_compact_mode_drops_repeated_semantics_while_keeping_typed_refs(self):
        temp = self.enterContext(tempfile.TemporaryDirectory())
        store = Store(Path(temp) / "run", dict(run_id="cmp", episode_id="ep", env_instance_id="env", env_generation=1))
        self.addCleanup(store.close)
        presentation = ActorPresentation(store)
        out = ActorOutput(store, presentation, payload_profile="compact_v1", compact=True)
        traj_ref = "artifact:" + "c" * 64 + "-" + "d" * 16
        dropped_ref = "artifact:" + "e" * 64 + "-" + "f" * 16
        payload = {
            "status": "completed",
            "residual_m": 0.0045678,
            "unknown_values_are_null": True,
            "execution_semantics": "repeated prose",
            "custom_semantics": "repeated prose",
            "trajectory_feedback_ref": traj_ref,
            "details_ref": dropped_ref,
        }
        response = ToolResponse([StructuredText(payload, "action_plan")], data=dict(payload), kind="action")
        wire = out.present(response, tool="robot_action")
        self.assertTrue(wire["success"])
        text = wire["contentItems"][0]["text"]
        self.assertEqual(metadata_findings(text), [])
        decoded = json.loads(text)
        self.assertEqual(decoded["residual_m"], 0.0046)
        self.assertNotIn("unknown_values_are_null", decoded)
        self.assertNotIn("execution_semantics", decoded)
        self.assertNotIn("custom_semantics", decoded)
        self.assertNotIn("details_ref", decoded)
        self.assertTrue(decoded["trajectory_feedback_ref"].startswith("artifact:ref_"))
        self.assertEqual(canonical(compact_model_value(decoded)), text)


if __name__ == "__main__":
    unittest.main()
