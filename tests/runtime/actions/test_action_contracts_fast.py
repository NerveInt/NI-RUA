"""
- test_action_model.py: The probe preserves state semantics and rejects misbound model responses.
- test_action_policy.py: Native joint policy inputs cannot silently borrow base-model semantics.
- test_stability.py: Bounded waits stop on uncertain evidence, budget, or consecutive quiet samples.
"""
import copy
from types import SimpleNamespace
import unittest
from scripts.probes.action_model_probe import probe_state, validate_actions
from ni_rua.runtime.policy.action_policy import arx_state, validate_joint_chunk
from ni_rua.runtime.actions.action_state_ledger import (
    attach_execution_ledger,
    attach_preview_ledger,
    record_execution_outcome,
)
from unittest.mock import Mock, patch
import numpy as np
from ni_rua.runtime.actions.stability import bounded_settle, validate_options

NI_RUA_TEST_TIER = "fast"


# --- Folded from tests/test_action_model.py ---

class ActionModelProbeCheck(unittest.TestCase):
    def test_state_provenance_and_response_identity(self):
        state = {"arms": {side: dict(source_kind="measurement",
            joint_names=[f"joint{i}" for i in range(1, 7)], q_rad=[offset + i / 10 for i in range(6)],
            gripper=dict(command_source_kind="command", command_normalized=offset,
                         position_source_kind="measurement", position_m=[0.123, 0.456]))
            for side, offset in (("left", 0), ("right", 1))}}
        values, fields = probe_state(state)
        self.assertEqual(values[:14], [0, .1, .2, .3, .4, .5, 0, 1, 1.1, 1.2, 1.3, 1.4, 1.5, 1])
        self.assertEqual(values[14:], [0] * 18)
        self.assertEqual(fields[6], "left/gripper:command_normalized")
        for field, value in (("q_rad", [float("nan")] * 6), ("source_kind", "command"),
                             ("joint_names", list(reversed(state["arms"]["left"]["joint_names"])))):
            bad = copy.deepcopy(state); bad["arms"]["left"][field] = value
            with self.assertRaises(ValueError): probe_state(bad)
        bad = copy.deepcopy(state); bad["arms"]["left"]["gripper"]["command_normalized"] = 2
        with self.assertRaises(ValueError): probe_state(bad)
        rows = [[i / 10] * 32 for i in range(50)]
        actions = [SimpleNamespace(get_action=lambda row=row: SimpleNamespace(tolist=lambda: row),
                    get_timestep=lambda i=i: 100 + i, get_timestamp=lambda i=i: 10.0 + i / 30)
                   for i, row in enumerate(rows)]
        self.assertEqual(validate_actions(actions, 100, 10.0)[0], rows)
        with self.assertRaisesRegex(ValueError, "identity"): validate_actions(actions, 101, 10.0)
        with self.assertRaisesRegex(ValueError, "identity"): validate_actions(actions, 100, 9.0)
        with self.assertRaisesRegex(ValueError, "length"): validate_actions(actions[:-1], 100, 10.0)
        rows[0][0] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"): validate_actions(actions, 100, 10.0)



# --- Folded from tests/test_action_policy.py ---

class ActionPolicyCheck(unittest.TestCase):
    def test_native_joint_order_and_command_provenance(self):
        state = {"arms": {side: dict(source_kind="measurement", q_rad=[i + offset for i in range(6)],
            gripper=dict(command_source_kind="command", command_normalized=offset,
                         position_m=[0.02, -0.02])) for side, offset in (("left", 0), ("right", 1))}}
        self.assertEqual(arx_state(state), [0, 1, 2, 3, 4, 5, 0, 1, 2, 3, 4, 5, 6, 1])
        for field, value in (("source_kind", "command"), ("q_rad", [float("nan")] * 6),
                             ("joint_names", [f"joint{i}" for i in range(6, 0, -1)])):
            bad = copy.deepcopy(state)
            bad["arms"]["left"][field] = value
            with self.assertRaises(ValueError):
                arx_state(bad)
        bad = copy.deepcopy(state)
        bad["arms"]["right"]["gripper"]["command_source_kind"] = "measurement"
        with self.assertRaises(ValueError):
            arx_state(bad)

    def test_no_implicit_base_model_slicing_or_nonfinite_commands(self):
        with self.assertRaisesRegex(ValueError, "50_by_14"):
            validate_joint_chunk([[0] * 32 for _ in range(50)])
        chunk = [[0] * 14 for _ in range(50)]
        self.assertEqual(validate_joint_chunk(chunk), chunk)
        chunk[4][7] = float("inf")
        with self.assertRaisesRegex(ValueError, "finite"):
            validate_joint_chunk(chunk)



# --- Folded from tests/test_action_state_ledger.py ---

class _DummyHarness:
    def __init__(self):
        self.identity = {"run_id": "r1", "episode_id": "ep0", "env_instance_id": "e1", "env_generation": 0}


class ActionStateLedgerTest(unittest.TestCase):
    def test_records_failed_hotspot_and_warns_on_repeated_preview_within_15mm(self):
        harness = _DummyHarness()
        action = {
            "stages": [
                {
                    "id": "descend_grasp",
                    "kind": "move",
                    "arm": "right",
                    "action": {
                        "path": {
                            "waypoints": [
                                {"right": {"position_m": [0.10, -0.30, 0.82], "quaternion_xyzw": [0, 0, 0, 1]}}
                            ]
                        }
                    },
                }
            ]
        }
        receipt = {
            "status": "stopped",
            "batch_status": "stopped",
            "stop_reason": "endpoint_not_reached",
            "failed_stage_id": "descend_grasp",
            "completed_stage_ids": [],
            "executed_native_steps": 18,
        }
        record_execution_outcome(harness, "plan-1", action, receipt)
        record_execution_outcome(harness, "plan-2", action, receipt)

        # Next preview targeting 8 mm away triggers repeated_hotspot warning
        next_action = {
            "stages": [
                {
                    "id": "retry_grasp",
                    "kind": "move",
                    "arm": "right",
                    "action": {
                        "path": {
                            "waypoints": [
                                {"right": {"position_m": [0.105, -0.305, 0.822], "quaternion_xyzw": [0, 0, 0, 1]}}
                            ]
                        }
                    },
                }
            ]
        }
        summary = attach_preview_ledger(harness, next_action, {"status": "planned"})
        self.assertIn("episode_ledger", summary)
        self.assertEqual(summary["episode_ledger"]["actions_executed"], 2)
        self.assertIn("repeat_failure_warning", summary)
        hotspot = summary["repeat_failure_warning"][0]
        self.assertEqual(hotspot["severity"], "repeated_hotspot")
        self.assertEqual(hotspot["prior_failures"], 2)
        self.assertNotIn("guidance", hotspot)
        self.assertEqual(
            hotspot["suggested_adjustments"],
            ["position_m", "grasp_position_m", "angles_deg", "contact_region"],
        )

        res = attach_execution_ledger(harness, {"status": "completed"})
        self.assertEqual(res["episode_ledger"]["actions_executed"], 2)

        # Vertical Z nudge (+3.2 cm at same XY) still matches cylindrical hotspot
        z_nudge_action = {
            "stages": [
                {
                    "id": "retry_z_nudge",
                    "kind": "move",
                    "arm": "right",
                    "action": {
                        "path": {
                            "waypoints": [
                                {"right": {"position_m": [0.102, -0.302, 0.852], "quaternion_xyzw": [0, 0, 0, 1]}}
                            ]
                        }
                    },
                }
            ]
        }
        z_summary = attach_preview_ledger(harness, z_nudge_action, {"status": "planned"})
        self.assertIn("repeat_failure_warning", z_summary)

        # Completing a move at the same XY evicts the hotspot
        completed_receipt = {
            "status": "completed",
            "batch_status": "completed",
            "completed_stage_ids": ["retry_z_nudge"],
            "executed_native_steps": 14,
        }
        record_execution_outcome(harness, "plan-3", z_nudge_action, completed_receipt)
        post_summary = attach_preview_ledger(harness, z_nudge_action, {"status": "planned"})
        self.assertNotIn("repeat_failure_warning", post_summary)

    def test_attribute_contact_block_3d_and_compute_object_table_delta(self):
        from ni_rua.runtime.actions.action_state_ledger import attribute_contact_block_3d, compute_object_table_delta
        harness = _DummyHarness()
        before_table = dict(
            columns=["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.30, 0.785], 0.815, 0.765, 0.050, 0.050, 0.0, None, "table", "ok"],
                ["o2", [-0.15, -0.25, 0.780], 0.800, 0.765, 0.040, 0.040, 0.0, None, "table", "ok"],
            ],
        )
        harness._obs3d_memo = ("obs-1", {"object_table": (before_table, None, {})})
        action = {
            "based_on_observation_id": "obs-1",
            "stages": [
                {
                    "id": "descend_o1",
                    "kind": "move",
                    "arm": "right",
                    "action": {"input_grasp_pose": {"position_m": [0.112, -0.292, 0.790]}},
                }
            ],
        }
        receipt = {
            "status": "failed",
            "batch_status": "failed",
            "reason": "tracking_error",
            "failed_stage_id": "descend_o1",
            "completed_stage_ids": [],
            "executed_native_steps": 22,
            "contact_block_summary": {
                "stage_id": "descend_o1",
                "arm": "right",
                "blocked_above_target": True,
                "actual_contact_position_m": [0.112, -0.292, 0.819],
            },
        }
        record_execution_outcome(harness, "plan-3d", action, receipt)
        attr = receipt["contact_block_summary"]["contact_object_3d"]
        self.assertEqual(attr["id"], "o1")
        self.assertAlmostEqual(attr["height_above_top_m"], 0.004, places=3)
        self.assertEqual(attr["xy_offset_from_centre_m"], [0.012, 0.008])

        # Compute before/after 3D object table delta: o1 lifted by +6 cm, o2 unchanged
        after_table = dict(
            columns=before_table["columns"],
            table_z=0.765,
            rows=[
                ["o1", [0.105, -0.295, 0.845], 0.875, 0.825, 0.050, 0.050, 0.0, None, "table", "ok"],
                ["o2", [-0.150, -0.250, 0.781], 0.801, 0.765, 0.040, 0.040, 0.0, None, "table", "ok"],
                ["o3", [0.050, -0.100, 0.920], 0.950, 0.890, 0.050, 0.030, 0.0, None, "held_left", "held_fk_depth"],
            ],
        )
        delta = compute_object_table_delta(before_table, after_table)
        self.assertEqual(len(delta["moved"]), 1)
        self.assertEqual(delta["moved"][0]["before_id"], "o1")
        self.assertTrue(delta["moved"][0]["lifted"])
        self.assertEqual(delta["unchanged_count"], 1)
        self.assertEqual(delta["vacated"], [])
        self.assertEqual(delta["appeared"], [])

    def test_unknown_status_not_masked_by_batch_status_and_missing_steps_stay_none(self):
        harness = _DummyHarness()
        action = {
            "stages": [
                {
                    "id": "move_unknown",
                    "kind": "move",
                    "arm": "left",
                    "action": {
                        "path": {
                            "waypoints": [
                                {"left": {"position_m": [0.15, -0.20, 0.80], "quaternion_xyzw": [0, 0, 0, 1]}}
                            ]
                        }
                    },
                }
            ]
        }
        unknown_receipt = {
            "status": "unknown",
            "batch_status": "completed",
            "reason": "worker_timeout",
            "completed_stage_ids": [],
        }
        record_execution_outcome(harness, "plan-unknown", action, unknown_receipt)
        res = attach_execution_ledger(harness, {"status": "unknown"})
        outcome = res["episode_ledger"]["recent_outcomes"][-1]
        self.assertEqual(outcome["status"], "unknown")
        self.assertIsNone(outcome["executed_native_steps"])



# --- Folded from tests/test_stability.py ---

class StabilityTests(unittest.TestCase):
    def harness(self, budget=50):
        harness = SimpleNamespace(latest={"observation_id": "o0"}, store=object(), identity={},
                                  enable_wait=True, disabled_tools=set(), used=0)
        harness.action_budget = lambda: dict(native_steps_remaining=budget - harness.used)
        harness._action_from_args = lambda name, args: dict(args, action_id="wait" + str(harness.used))
        def execute(action, correlation):
            harness.used += action["native_steps"]
            harness.latest = dict(observation_id="o" + str(harness.used))
            return dict(receipt=dict(status="completed", completed_native_steps=action["native_steps"]))
        harness.execute = Mock(side_effect=execute)
        return harness

    def comparison(self, motion=0, status="tracked_features"):
        return dict(target=dict(status=status, points_before_px=[[0, 0], [1, 1]],
                                points_after_px=[[motion, 0], [motion+1, 1]]))

    def run_settle(self, harness, evidence=None, **options):
        with patch("ni_rua.runtime.actions.stability._camera_pose", return_value=(np.eye(4), dict(width=100, height=100))), \
             patch("ni_rua.perception.visual_feedback.compare_observations", return_value=evidence or self.comparison()):
            return bounded_settle(harness, {"roi": [0, 0, 50, 50], **options}, {})

    def test_three_consecutive_qualified_samples_and_real_cost(self):
        harness = self.harness()
        result = self.run_settle(harness)
        self.assertEqual(result["status"], "low_visible_motion")
        self.assertEqual(result["executed_native_steps"], 15)
        self.assertEqual(harness.execute.call_count, 3)
        self.assertEqual(result["final_observation_id"], "o15")

    def test_ambiguous_visibility_is_unknown_and_stops(self):
        harness = self.harness()
        result = self.run_settle(harness, self.comparison(status="lost"))
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(harness.execute.call_count, 1)

    def test_budget_and_explicit_limit_bound_actual_waits(self):
        harness = self.harness(budget=7)
        result = self.run_settle(harness, self.comparison(2), max_native_steps=20)
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(harness.used, 7)
        harness = self.harness()
        result = self.run_settle(harness, self.comparison(2), max_native_steps=6)
        self.assertEqual(result["status"], "limit_reached")
        self.assertEqual(harness.used, 6)

    def test_unknown_camera_and_bad_roi_do_not_advance_world(self):
        harness = self.harness()
        with patch("ni_rua.runtime.actions.stability._camera_pose", side_effect=ValueError("camera_unknown")):
            result = bounded_settle(harness, dict(roi=[0, 0, 20, 20]), {})
        self.assertEqual(result["status"], "unknown")
        harness.execute.assert_not_called()
        result = self.run_settle(harness, roi=[90, 0, 50, 50])
        self.assertEqual(result["reason"], "settle_roi_out_of_bounds")
        harness.execute.assert_not_called()

    def test_invalid_request_rejected_before_wait(self):
        for options in ({}, dict(roi=[0, 0, 20, 20], max_native_steps=51),
                        dict(roi=[0, 0, 20, 20], max_motion_px=float("nan"))):
            with self.assertRaises(ValueError):
                validate_options(options)



if __name__ == "__main__":
    unittest.main()
