"""
- test_metadata_delivery.py: Session references require a delivered baseline, never a stored observation.
- test_tool_descriptions.py: Published tools carry typed parameters and complete help for both roles.
- test_tool_registry_prose_bounds.py: Registry lint: schema bounds, enums and defaults must also be stated in the description prose.

Why: OpenCode on the OpenAI Responses wire strips JSON Schema keywords (minimum, maximum, exclusiveMinimum,
exclusiveMaximum, default, minItems, maxItems, minLength, maxLength, ...) from MCP tool schemas before they reach the
model (spike 06 S6), and AGENTS.md requires every parameter's meaning, unit, default and range to live in the contract
text. The catalog is built exactly as an actor sees it (Workbench.actor_tools fallback runtime, pose_skill on, both
optional plugins registered) and every published tool's input schema is walked recursively.

Text visible to the model for a property = the property's OWN `description` plus the TOOL's description. (Ancestor
property descriptions are deliberately not counted: a stripped-schema client may flatten nested objects.)

Matching rules (conservative; a property is checked once per keyword it carries):
* numeric keywords (minimum, maximum, exclusiveMinimum, exclusiveMaximum, minItems, maxItems, minLength, maxLength):
  the text must contain a numeric token whose value equals the bound. Tokens are decimal or scientific literals
  (`0.05`, `-1`, `1e-3`, `1.5E+2`) and thousands-free integers; unit suffixes do not matter (`5 mm` still yields 5),
  but unit conversion is NOT applied (0.005 m is not matched by "5 mm"). Non-negative integers up to 12 may instead
  be written as an English word delimited by non-letters (`Exactly three finite numbers` states minItems=maxItems=3).
* enum: every enum value must occur in the text, case-sensitively, delimited by non-identifier characters
  (so `a` does not match inside `data`). Non-string values are matched through their JSON rendering (`true`, `3`).
* default: the text must contain the word `default` (case-insensitive, also `defaults`/`defaulting`), and, when the
  default is a number or a short string, that value must also be present as a token / delimited word. Container
  defaults (empty or not) only require the word `default`.
"""
from copy import deepcopy
from dataclasses import replace
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from ni_rua.core.presentation.capability_output import CapabilityDelivery
from ni_rua.core.presentation.observation_output import (
    MetadataDelivery,
    observation_metadata,
    same_frame_robot_state,
    saved_observation_state,
)
from ni_rua.core.tool_output import StructuredText, ToolResponse
from ni_rua.runtime.tools.registry import validator_for
from support.motion_fixtures import DualArmHarnessBase
from support.motion_fixtures import _DualRuntime
import numpy as np
from ni_rua.robot.motion.scene_obstacles import (
    merge_and_persist_obstacles,
    plan_move_with_obstacle_clearance,
)
from ni_rua.runtime.tools.builtins import create_registry
from ni_rua.runtime.tools.definitions import (DEPTH_REQUEST_DESCRIPTIONS,
    GEOMETRY_REQUEST_DESCRIPTIONS, TOOLS, public_definitions)
from ni_rua.runtime.actions.robot_action_schema import ROBOT_ACTION_TOOL
import math
import re
from ni_rua.core.config import Config
from support.agent_fixtures import full_registry

NI_RUA_TEST_TIER = "fast"


# --- Folded from tests/test_metadata_delivery.py ---

def observation(identifier="obs-1", generation=1):
    matrix = [[1., 0., 0., .123456789012345], [0., 1., 0., .2],
              [0., 0., 1., .3], [0., 0., 0., 1.]]
    calibration = dict(K=[[397.041259765625, 0., 320.], [0., 397.041259765625, 240.], [0., 0., 1.]],
        distortion=[0., 0., 0., 0., 0.], distortion_model="pinhole_no_distortion", image_size=[640, 480],
        optical_axes="+x right,+y down,+z forward", calibration_source="public sensor calibration")
    arm = dict(joint_names=["joint" + str(i) for i in range(1, 7)], source_kind="measurement",
        q_rad=[.123456789012345] * 6, qdot_rad_s=[.00000123456789] * 6,
        ee_pose=dict(position_m=[.1, .2, .3], quaternion_xyzw=[0., 0., 0., 1.],
            frame_id="env_origin", tcp_config_id="arx-x5-link6-v1", source_kind="measurement"),
        gripper=dict(joint_names=["joint7", "joint8"], position_m=[.012345678901, .02],
            position_source_kind="measurement", goal_normalized=.5, goal_source_kind="command"),
        grasp_pose=dict(position_m=[.1, .2, .4], angles_deg=[0., 0., 90.], frame_id="env_origin",
            source_kind="derived_from_measured_link6", convention="robot_move_grasp", contact_region="pad_center",
            contact_depth_m=0., contact_point_meaning="center_between_fingers_at_distal_15mm_pad_center"))
    value = dict(bundle=dict(observation_id=identifier, run_id="run", episode_id="episode", env_instance_id="env",
        env_generation=generation, sim_step=10, action_budget=dict(native_steps_remaining=700)),
        robot_state=dict(arms=dict(left=deepcopy(arm), right=deepcopy(arm)), sim_step=10,
            frame_id="env_origin", tcp_config_id="arx-x5-link6-v1", units=dict(position="m", joints="rad")),
        camera_calibration=dict(profile_id="public-cal-v1", reference_frame_id="env_origin",
            source_kind="public_calibration", full_ref="artifact:calibration", cameras={
                "head": deepcopy(calibration), "left": deepcopy(calibration), "right": deepcopy(calibration)}),
        camera_pose_sample=dict(sim_step=10, reference_frame_id="env_origin", full_ref="artifact:pose-" + identifier,
            cameras={name: dict(T_world_camera=deepcopy(matrix), frame_id=identifier + "-" + name,
                source_kind="measurement") for name in ("head", "left", "right")}))
    return observation_metadata(value, current_observation_id=identifier,
        current_action_budget=dict(native_steps_remaining=700), compact_geometry=True)


def clocked_observation(identifier="obs-1"):
    value = observation(identifier)
    bundle = value["bundle"]
    bundle.update(schema_version="0.1", robot_model_id="arx-x5",
                  observation_profile="rgb_proprio_calibrated_v1",
                  clock_domain="env/generation-1", step_domain="adapter_control_step",
                  calibration_id="a" * 64, robot_model_ref="artifact:model",
                  data_origin="robodojo_runtime", task_instruction="Keep the task visible.")
    bundle["cameras"] = [dict(camera_id=name, width=640, height=480,
        pixel_transform="identity", control_dt_s=.004, step_domain="adapter_control_step",
        frame_id=identifier + "-" + name, capture_time=identifier + "-time",
        sim_step=10, clock_domain=None, rgb_available=True, calibration_available=True,
        pose_sample_available=True, synchronization="physics paused; camera fence")
        for name in ("head", "left", "right")]
    for budget in (bundle["action_budget"], value["current_action_budget"]):
        budget.update(semantics="official native steps; separate from tool requests and model-call budget; reserves are not included",
                      source="native_execution_accounting", unknown_values_are_null=True,
                      native_steps_limit=1000, native_steps_used=300, per_request_native_limit=250)
    value["robot_state"]["step_domain"] = "adapter_control_step"
    value["camera_pose_sample"]["step_domain"] = "adapter_control_step"
    return value


class MetadataDeliveryTests(unittest.TestCase):
    def setUp(self):
        self.delivery = MetadataDelivery()
        self.epoch = ({"compaction": "epoch-1"}, "app-1", "transport-1", None)

    def prepare(self, value, **options):
        return self.delivery.prepare(value, epoch=options.pop("epoch", self.epoch), **options)

    def establish(self, value=None):
        first = self.prepare(value or observation())
        self.assertTrue(self.delivery.commit(first))
        return first

    def test_first_delivery_is_exact_and_preparation_without_commit_cannot_be_referenced(self):
        first = observation()
        before = deepcopy(first)
        prepared = self.prepare(first)
        self.assertEqual(prepared.metadata, first)
        # A discarded/overwritten wait sample has not entered the actor history.
        second = self.prepare(observation("obs-2"))
        self.assertNotIn("metadata_delivery", second.metadata)
        self.assertIn("K", second.metadata["camera_calibration"]["cameras"]["head"])
        self.assertEqual(first, before)

    def test_references_only_static_fields_and_preserves_live_measurement_precision(self):
        self.establish()
        second = observation("obs-2")
        second["robot_state"]["arms"]["left"]["q_rad"][0] = .987654321098765
        second["camera_pose_sample"]["cameras"]["left"]["T_world_camera"][0][3] = .987654321098765
        second["current_action_budget"]["native_steps_remaining"] = 612
        before = deepcopy(second)
        projected = self.prepare(second).metadata
        self.assertEqual(second, before)
        refs = projected["metadata_delivery"]["static_refs"]
        self.assertEqual(set(refs), {"camera_calibration", "robot_state", "bundle"})
        self.assertTrue(all(ref["observation_id"] == "obs-1" for ref in refs.values()))
        self.assertNotIn("K", projected["camera_calibration"]["cameras"]["head"])
        self.assertEqual(projected["camera_calibration"]["full_ref"], "artifact:calibration")
        self.assertEqual(
            projected["camera_pose_sample"],
            {k: v for k, v in second["camera_pose_sample"].items() if k != "reference_frame_id"},
        )
        state = projected["robot_state"]["arms"]["left"]
        self.assertNotIn("joint_names", state)
        for field in ("q_rad", "qdot_rad_s", "source_kind"):
            self.assertEqual(state[field], second["robot_state"]["arms"]["left"][field])
        self.assertEqual(state["gripper"]["position_m"], second["robot_state"]["arms"]["left"]["gripper"]["position_m"])
        self.assertEqual(state["gripper"]["position_source_kind"], "measurement")
        self.assertEqual(state["gripper"]["goal_source_kind"], "command")
        self.assertEqual(state["grasp_pose"]["source_kind"], "derived_from_measured_link6")
        self.assertEqual(projected["current_action_budget"], second["current_action_budget"])
        self.assertTrue(projected["observation_is_current"])
        self.assertIn("unchanged intrinsics use metadata_delivery.static_refs", projected["geometry_delivery"]["omitted"])

    def test_changed_intrinsics_or_joint_order_are_delivered_inline_before_new_reference(self):
        self.establish()
        changed = observation("obs-2")
        changed["camera_calibration"]["cameras"]["head"]["K"][0][0] = 500.123456789
        changed["robot_state"]["arms"]["left"]["joint_names"].reverse()
        projection = self.prepare(changed)
        self.assertEqual(projection.metadata, changed)
        self.assertTrue(self.delivery.commit(projection))
        third = deepcopy(changed)
        third["bundle"]["observation_id"] = "obs-3"
        refs = self.prepare(third).metadata["metadata_delivery"]["static_refs"]
        self.assertTrue(all(ref["observation_id"] == "obs-2" for ref in refs.values()))

    def test_each_epoch_component_and_environment_generation_invalidates_baseline(self):
        epochs = [(dict(compaction="epoch-2"), "app-1", "transport-1", None),
                  (self.epoch[0], "app-2", "transport-1", None),
                  (self.epoch[0], "app-1", "transport-2", None),
                  (self.epoch[0], "app-1", "transport-1", "compact-call")]
        for epoch in epochs:
            with self.subTest(epoch=epoch):
                self.establish()
                changed = observation("obs-2")
                self.assertEqual(self.prepare(changed, epoch=epoch).metadata, changed)
        self.establish()
        reset = observation("obs-reset", generation=2)
        self.assertEqual(self.prepare(reset).metadata, reset)

    def test_old_pending_delivery_cannot_restore_baseline_after_context_invalidation(self):
        pending = self.prepare(observation())
        self.delivery.invalidate()
        self.assertFalse(self.delivery.commit(pending))
        current = self.prepare(observation("obs-2"), epoch="new-context")
        self.assertNotIn("metadata_delivery", current.metadata)
        self.assertFalse(self.delivery.commit(pending))

    def test_full_exception_delivery_ignores_reference_and_teaches_only_on_commit(self):
        self.establish()
        exception = observation("obs-error")
        exception["robot_state"]["arms"]["left"]["future_warning"] = dict(status="unknown", reason="sensor_fault")
        full = self.prepare(exception, force_full=True)
        self.assertEqual(full.metadata, exception)
        self.assertTrue(self.delivery.commit(full))
        next_value = observation("obs-next")
        refs = self.prepare(next_value).metadata["metadata_delivery"]["static_refs"]
        self.assertTrue(all(ref["observation_id"] == "obs-error" for ref in refs.values()))

    def test_unknown_fields_live_sources_and_historical_budget_never_use_baseline_values(self):
        self.establish()
        history = observation("obs-history")
        history["current_observation_id"] = "obs-current"
        history["observation_is_current"] = False
        history["current_action_budget"] = dict(native_steps_remaining=12)
        history["camera_calibration"]["future_warning"] = dict(status="uncertain")
        history["camera_calibration"]["cameras"]["head"]["future_quality"] = "degraded"
        history["robot_state"]["arms"]["left"]["source_kind"] = "estimate"
        projected = self.prepare(history).metadata
        self.assertEqual(projected["camera_calibration"]["future_warning"], dict(status="uncertain"))
        self.assertEqual(projected["camera_calibration"]["cameras"]["head"]["future_quality"], "degraded")
        self.assertEqual(projected["robot_state"]["arms"]["left"]["source_kind"], "estimate")
        self.assertFalse(projected["observation_is_current"])
        self.assertEqual(projected["current_observation_id"], "obs-current")
        self.assertEqual(projected["bundle"]["action_budget"]["native_steps_remaining"], 700)
        self.assertEqual(projected["current_action_budget"]["native_steps_remaining"], 12)
        self.assertEqual(projected["metadata_delivery"]["read_full"]["observation_id"], "obs-history")

    def test_missing_observation_identity_cannot_establish_reference_and_nonfinite_static_is_inline(self):
        value = observation()
        value["bundle"].pop("observation_id")
        prepared = self.prepare(value)
        self.delivery.commit(prepared)
        self.assertEqual(self.prepare(value).metadata, value)
        value = observation("obs-nan")
        value["camera_calibration"]["cameras"]["head"]["K"][0][0] = float("nan")
        prepared = self.prepare(value)
        self.delivery.commit(prepared)
        next_value = self.prepare(value).metadata
        self.assertIn("K", next_value["camera_calibration"]["cameras"]["head"])

    def test_measured_payload_reduction_does_not_require_float_rounding(self):
        first = self.establish().metadata
        second = self.prepare(observation("obs-2")).metadata
        encode = lambda value: json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
        self.assertLess(len(encode(second)), len(encode(first)) * .85)
        self.assertEqual(second["robot_state"]["arms"]["left"]["q_rad"][0], .123456789012345)

    def test_unknown_context_does_not_reuse_or_preserve_an_older_baseline(self):
        self.establish()
        value = observation("obs-unknown-context")
        unknown = self.prepare(value, epoch=None)
        self.assertEqual(unknown.metadata, value)
        self.assertFalse(self.delivery.commit(unknown))
        restored = self.prepare(observation("obs-restored"))
        self.assertNotIn("metadata_delivery", restored.metadata)

    def test_unknown_static_values_and_unfamiliar_projection_metadata_stay_inline(self):
        self.establish()
        value = observation("obs-unknown-values")
        value["camera_calibration"]["cameras"]["head"]["K"] = None
        value["robot_state"]["arms"]["left"]["joint_names"] = []
        self.delivery.commit(self.prepare(value))
        value["bundle"]["observation_id"] = "obs-next"
        projected = self.prepare(value).metadata
        self.assertIsNone(projected["camera_calibration"]["cameras"]["head"]["K"])
        self.assertEqual(projected["robot_state"]["arms"]["left"]["joint_names"], [])
        value["metadata_delivery"] = dict(future_warning="unknown producer")
        projected = self.prepare(value)
        self.assertEqual(projected.metadata, value)
        self.assertFalse(self.delivery.commit(projected))

    def establish_clocked_conventions(self):
        # Geometry wording and delivery prose change on the second reply.
        # Both strings must actually be delivered before this group can cite them.
        for identifier in ("obs-1", "obs-2"):
            projection = self.prepare(clocked_observation(identifier))
            self.assertTrue(self.delivery.commit(projection))
        return self.prepare(clocked_observation("obs-3"))

    def test_grouped_bundle_prose_and_clock_conventions_preserve_live_evidence(self):
        projection = self.establish_clocked_conventions()
        actual = projection.metadata
        original = clocked_observation("obs-3")
        ref = actual["metadata_delivery"]["static_refs"]["bundle"]
        self.assertEqual(ref["observation_id"], "obs-1")
        for key in ("run_id", "schema_version", "clock_domain", "calibration_id"):
            self.assertNotIn(key, actual["bundle"])
        for key in ("observation_id", "sim_step", "robot_model_ref", "data_origin", "task_instruction"):
            self.assertEqual(actual["bundle"][key], original["bundle"][key])
        for delivered, source in zip(actual["bundle"]["cameras"], original["bundle"]["cameras"]):
            for key in ("camera_id", "capture_time", "frame_id", "sim_step", "clock_domain",
                        "synchronization", "rgb_available", "calibration_available", "pose_sample_available"):
                self.assertEqual(delivered[key], source[key])
            self.assertNotIn("width", delivered)
        for delivered, source in ((actual["bundle"]["action_budget"], original["bundle"]["action_budget"]),
                                  (actual["current_action_budget"], original["current_action_budget"])):
            for key in ("source", "native_steps_remaining", "native_steps_used", "native_steps_limit",
                        "per_request_native_limit"):
                self.assertEqual(delivered[key], source[key])
            self.assertNotIn("semantics", delivered)
        self.assertNotIn("observation_budget_semantics", actual)
        self.assertNotIn("omitted", actual["geometry_delivery"])
        self.assertNotIn("semantics", actual["metadata_delivery"])
        self.assertEqual(actual["camera_pose_sample"]["cameras"], original["camera_pose_sample"]["cameras"])
        self.assertEqual(actual["geometry_delivery"]["read_full"]["observation_id"], "obs-3")
        encode = lambda v: json.dumps(v, sort_keys=True, separators=(",", ":")).encode()
        self.assertLess(len(encode(actual)), len(encode(original)) - 1000)

    def test_camera_conventions_follow_ids_and_keep_original_list_order(self):
        self.establish_clocked_conventions()
        value = clocked_observation("obs-reordered")
        value["bundle"]["cameras"].reverse()
        actual = self.prepare(value).metadata
        self.assertIn("bundle", actual["metadata_delivery"]["static_refs"])
        self.assertEqual([camera["camera_id"] for camera in actual["bundle"]["cameras"]],
                         [camera["camera_id"] for camera in value["bundle"]["cameras"]])
        self.assertTrue(all("width" not in camera for camera in actual["bundle"]["cameras"]))
        value = clocked_observation("obs-resized")
        value["bundle"]["cameras"][0]["width"] = 800
        pending = self.prepare(value)
        self.assertNotIn("bundle", pending.metadata["metadata_delivery"]["static_refs"])
        self.assertEqual(pending.metadata["bundle"]["cameras"], value["bundle"]["cameras"])
        # Discarding the resized sample cannot teach the changed camera convention.
        changed_again = self.prepare(value)
        self.assertNotIn("bundle", changed_again.metadata["metadata_delivery"]["static_refs"])
        self.assertTrue(self.delivery.commit(changed_again))
        next_value = deepcopy(value)
        next_value["bundle"]["observation_id"] = "obs-resized-next"
        self.assertEqual(self.prepare(next_value).metadata["metadata_delivery"]["static_refs"]["bundle"]["observation_id"],
                         "obs-resized")

    def test_unknown_camera_budget_and_prose_fields_remain_inline(self):
        self.establish_clocked_conventions()
        value = clocked_observation("obs-unknown")
        value["bundle"]["future_diagnostic"] = {"uncertain": True}
        value["bundle"]["schema_version"] = "future"
        value["bundle"]["cameras"][0].update(width=None, pixel_transform=[[2, 0], [0, 2]],
                                            control_dt_s=float("nan"), future_fault="unknown")
        value["current_action_budget"]["semantics"] = "unknown budget accounting"
        value["observation_budget_semantics"] = "future historical budget rule"
        value["geometry_delivery"]["omitted"] = {"unknown": "geometry rule"}
        self.assertTrue(self.delivery.commit(self.prepare(value)))
        actual = self.prepare(value).metadata
        # Nonfinite live values make byte comparison conservative, never an error.
        self.assertNotIn("bundle", actual["metadata_delivery"]["static_refs"])
        self.assertEqual(actual["bundle"]["future_diagnostic"], value["bundle"]["future_diagnostic"])
        self.assertEqual(actual["bundle"]["schema_version"], "future")
        self.assertIsNone(actual["bundle"]["cameras"][0]["width"])
        self.assertEqual(actual["bundle"]["cameras"][0]["future_fault"], "unknown")
        self.assertEqual(actual["current_action_budget"]["semantics"], "unknown budget accounting")
        self.assertEqual(actual["geometry_delivery"]["omitted"], value["geometry_delivery"]["omitted"])

    def test_duplicate_camera_ids_and_changed_membership_never_remove_camera_list(self):
        self.establish_clocked_conventions()
        value = clocked_observation("obs-duplicate")
        value["bundle"]["cameras"][1]["camera_id"] = value["bundle"]["cameras"][0]["camera_id"]
        self.assertTrue(self.delivery.commit(self.prepare(value)))
        projected = self.prepare(value).metadata
        self.assertEqual(projected["bundle"]["cameras"], value["bundle"]["cameras"])
        value = clocked_observation("obs-missing-camera")
        value["bundle"]["cameras"].pop()
        projected = self.prepare(value).metadata
        self.assertEqual(projected["bundle"]["cameras"], value["bundle"]["cameras"])

    def test_grouped_conventions_respect_full_historical_and_invalidation_contracts(self):
        pending = self.establish_clocked_conventions()
        value = clocked_observation("obs-history")
        value["current_observation_id"] = "obs-current"
        value["observation_is_current"] = False
        value["current_action_budget"]["native_steps_remaining"] = 12
        actual = self.prepare(value).metadata
        self.assertFalse(actual["observation_is_current"])
        self.assertEqual(actual["current_action_budget"]["native_steps_remaining"], 12)
        self.assertEqual(actual["bundle"]["action_budget"]["native_steps_remaining"], 700)
        self.assertEqual(self.prepare(value, force_full=True).metadata, value)
        self.delivery.invalidate()
        self.assertFalse(self.delivery.commit(pending))
        self.assertEqual(self.prepare(value).metadata, value)

    def test_incomplete_identity_and_unprofitable_small_groups_remain_inline(self):
        value = dict(bundle=dict(observation_id="obs-small", run_id="r", episode_id="e",
                                env_instance_id="i", env_generation=1))
        self.delivery.commit(self.prepare(value))
        self.assertEqual(self.prepare(value).metadata, value)
        self.establish_clocked_conventions()
        value = clocked_observation("obs-missing-identity")
        value["bundle"].pop("episode_id")
        self.assertEqual(self.prepare(value).metadata, value)


class MetadataHarnessTests(unittest.TestCase):
    setUp = DualArmHarnessBase.setUp

    def read(self, **kwargs):
        return json.loads(self.h.observation_content(self.h.latest, {}, **kwargs)[0]["text"])

    def test_overwritten_temporal_metadata_and_program_reads_do_not_consume_baseline(self):
        self.read(metadata_delivery=False)
        self.h.observation_content(self.h.latest, {"program_id": "fixture"})
        self.read(detail="full")
        first = self.read()
        self.assertNotIn("metadata_delivery", first)
        repeated = self.read()
        self.assertIn("metadata_delivery", repeated)
        self.assertEqual(
            repeated["current_action_budget"],
            {k: v for k, v in first["current_action_budget"].items() if k not in ("semantics", "unknown_values_are_null")},
        )
        self.assertEqual(repeated["camera_pose_sample"], first["camera_pose_sample"])

    def test_model_context_and_environment_resets_restore_inline_conventions(self):
        self.read()
        self.assertIn("metadata_delivery", self.read())
        for reset in (lambda: self.store.set_state("runtime", "image_context_epoch", "compacted"),
                      lambda: setattr(self.h, "thread_id", "replacement"),
                      lambda: self.h.identity.update(env_generation=2)):
            reset()
            self.assertNotIn("metadata_delivery", self.read())
            self.assertIn("metadata_delivery", self.read())

    def test_context_loss_and_structured_projection_errors_fall_back_to_inline(self):
        self.read()
        with patch.object(self.h._metadata_delivery, "commit", return_value=False):
            parts = self.h.observation_content(self.h.latest, {}, structured=True)
        self.assertIsInstance(parts[0], StructuredText)
        self.assertNotIn("metadata_delivery", parts[0].value)
        shown = self.h.actor_output.present(ToolResponse(parts))
        self.assertNotIn("metadata_delivery", json.loads(shown["contentItems"][0]["text"]))

        epoch = self.h._model_context_epoch()
        with patch.object(self.h, "_model_context_epoch", side_effect=[epoch, (*epoch, "compaction")]):
            self.assertNotIn("metadata_delivery", self.read())

        self.read()
        owner = self.h._metadata_delivery
        prepare = owner.prepare
        with patch.object(owner, "prepare", side_effect=lambda *a, **kw: replace(prepare(*a, **kw), metadata={"bad": object()})), \
                patch.object(owner, "commit") as commit:
            parts = self.h.observation_content(self.h.latest, {}, structured=True)
        commit.assert_not_called()
        self.assertNotIn("metadata_delivery", parts[0].value)
        self.assertIn("robot_state", parts[0].value)
        self.assertIsNot(owner, self.h._metadata_delivery)

        for operation in ("prepare", "commit", "serialization"):
            with self.subTest(operation=operation):
                self.read()
                cur_owner = self.h._metadata_delivery
                options = (dict(return_value=SimpleNamespace(metadata={"bad": object()}))
                           if operation == "serialization" else dict(side_effect=RuntimeError("optional_fault")))
                with patch.object(cur_owner, "prepare" if operation == "serialization" else operation, **options):
                    value = self.read()
                self.assertNotIn("metadata_delivery", value)
                self.assertIn("robot_state", value)
                self.assertIsNot(cur_owner, self.h._metadata_delivery)


class ObservationProjectionAndDeliveryTests(unittest.TestCase):
    setUp = DualArmHarnessBase.setUp

    def test_reconstructive_matrices_omitted_and_missing_transform_keeps_calibration(self):
        matrix = [[1, 0, 0, .123456789], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]
        value = dict(bundle=dict(observation_id="old", action_budget=dict(native_steps_remaining=700),
            cameras=[dict(camera_id="wrist", calibration_ref="artifact:cal", pose_sample_ref="artifact:pose")]),
            camera_calibration=dict(cameras=dict(wrist=dict(K=matrix[:3], T_world_base=matrix,
                T_link_camera=matrix, distortion=[0, 0, 0, 0, 0], source_kind="calibration", future_warning="keep"))),
            camera_pose_sample=dict(cameras=dict(wrist=dict(T_world_camera=matrix, T_world_base=matrix,
                T_world_link=matrix, source_kind="robot_kinematics", frame_id="wrist-frame")), sim_step=30))
        original = deepcopy(value)
        result = observation_metadata(value, current_observation_id="now",
            current_action_budget=dict(native_steps_remaining=12), compact_geometry=True)
        self.assertEqual(value, original)
        self.assertFalse(result["observation_is_current"])
        self.assertEqual(result["bundle"]["action_budget"]["native_steps_remaining"], 700)
        self.assertEqual(result["current_action_budget"]["native_steps_remaining"], 12)
        self.assertNotIn("T_world_link", result["camera_pose_sample"]["cameras"]["wrist"])
        self.assertNotIn("T_link_camera", result["camera_calibration"]["cameras"]["wrist"])
        self.assertEqual(result["camera_calibration"]["full_ref"], "artifact:cal")

        missing = dict(bundle=dict(observation_id="obs"), camera_calibration=dict(cameras={
            "head": dict(T_world_camera=[[1]], T_link_camera=[[2]], unknown=True)}),
            camera_pose_sample=dict(cameras={"head": {"unavailable_reason": "no pose"}}))
        res_missing = observation_metadata(missing, current_observation_id=None, current_action_budget={}, compact_geometry=True)
        self.assertEqual(res_missing["camera_calibration"], missing["camera_calibration"])

    def test_historical_snapshot_full_paths_and_saved_frame_state_binding(self):
        self.h.native_budget_known, self.h.native_budget_limit, self.h.native_action_count = True, 700, 0
        old = self.h.observe()
        self.h.native_action_count = 688
        latest = self.h.observe()
        response = self.h.dispatch("history_view_observation", dict(observation_id=old["observation_id"], detail="summary"), {})
        value = json.loads(response["contentItems"][0]["text"])
        self.assertFalse(value["observation_is_current"])
        self.assertEqual(value["current_observation_id"], latest["observation_id"])
        self.assertEqual(value["current_action_budget"]["native_steps_remaining"], 12)
        self.assertEqual(value["bundle"]["action_budget"]["native_steps_remaining"], 700)

        full = self.h.observation_content(latest, {}, detail="full")
        internal = self.h.observation_content(latest, {"program_id": "test"})
        for field in ("camera_calibration", "camera_pose_sample", "robot_state"):
            self.assertEqual(json.loads(full[0]["text"]).get(field), json.loads(internal[0]["text"]).get(field))
        with self.assertRaisesRegex(ValueError, "invalid_output_detail"):
            self.h.observation_content(latest, {}, detail="invalid")

        raw = self.robot.observe()
        raw["robot_state"].update(sim_step=raw["sim_step"], step_domain=raw["step_domain"])
        published = self.h.publish_observation(raw)
        evidence = saved_observation_state(self.store, published["observation_id"])
        state = json.loads(self.store.resolve_artifact(published["robot_state_ref"], ("actor",))[1])
        self.assertEqual(evidence["status"], "available")
        self.assertEqual(evidence["robot_state"], state)
        wrong = deepcopy(state)
        wrong["sim_step"] += 1
        self.assertEqual(same_frame_robot_state(published, wrong)["status"], "unavailable")
        self.assertEqual(saved_observation_state(self.store, "missing")["status"], "unavailable")


class CapabilityDeliveryAndHarnessTests(unittest.TestCase):
    setUp = DualArmHarnessBase.setUp

    def read_caps(self, args=None, correlation=None):
        result = self.h.dispatch("robot_capabilities", args or {}, correlation or {})
        return json.loads(result["contentItems"][0]["text"])

    def test_capability_projection_context_reset_and_native_pose_guidance(self):
        delivery = CapabilityDelivery()
        caps = dict(frame_id="env_origin", dual_arm=True, warnings=["fixture"])
        epoch = dict(thread="a", compact=0)
        first = delivery.project(caps, epoch=epoch, current_action_budget={"remaining": 10})
        second = delivery.project(caps, epoch=epoch, current_action_budget={"remaining": 3})
        self.assertEqual(second["capabilities_delivery"]["status"], "unchanged")
        self.assertEqual(second["current_action_budget"], {"remaining": 3})
        epoch["compact"] = 1
        self.assertEqual(delivery.project(caps, epoch=epoch, current_action_budget={})["frame_id"], "env_origin")
        caps["dual_arm"] = False
        self.assertFalse(delivery.project(caps, epoch=epoch, current_action_budget={})["dual_arm"])
        first["warnings"].append("mutated")
        self.assertEqual(caps["warnings"], ["fixture"])

        self.assertEqual(self.read_caps(correlation={"program_id": "fixture"}), self.h.capabilities)
        self.assertEqual(self.read_caps({"detail": "full"}), self.h.capabilities)
        self.assertEqual(self.read_caps()["capabilities_delivery"]["status"], "inline")
        self.assertEqual(self.read_caps()["capabilities_delivery"]["status"], "unchanged")

        self.h.native_budget_known, self.h.native_budget_limit, self.h.native_action_count = True, 100, 99
        self.assertEqual(self.read_caps()["current_action_budget"]["native_steps_remaining"], 1)
        self.store.set_state("runtime", "image_context_epoch", "new")
        self.assertEqual(self.read_caps()["capabilities_delivery"]["status"], "inline")

        self.h.capabilities = _DualRuntime().adapter.capabilities()
        public = {item["name"]: item for item in self.h.tool_registry.catalog(self.h, role="actor")}
        schema = public["robot_action"]["inputSchema"]
        stage_schema = next(part["then"] for part in schema["properties"]["stages"]["items"]["allOf"]
                            if part["if"]["properties"]["kind"]["const"] == "move")
        for detail in ("summary", "full"):
            reply = self.h.invoke("robot_capabilities", {"detail": detail}, {}, role="actor")
            text = json.loads(self.h.actor_output.present(reply)["contentItems"][0]["text"])["tool_geometry"]["conversion"]
            self.assertIn("kind='move'", text)
            for field in ("id", "arm", "position_m", "quaternion_xyzw", "grasp_position_m", "angles_deg"):
                self.assertIn(field, text)
                self.assertIn(field, stage_schema["properties"])
        validator_for(schema).validate(dict(mode="preview", observation_id="current-observation",
            stages=[dict(id="left_move", kind="move", arm="left", grasp_position_m=[-.3, 0, 1], angles_deg=[0, 0, 0])]))



# --- Folded from tests/test_scene_obstacles.py ---

class _DummyHarness:
    def __init__(self):
        self.identity = {"run_id": "r1", "episode_id": "ep0", "env_instance_id": "e1", "env_generation": 0}


class _DummyAdapter:
    def __init__(self):
        self.calls = []

    def _plan_segment(self, robot, target, diagnostics, index, **options):
        self.calls.append((index, list(target["position_m"])))
        z = target["position_m"][2]
        q0 = np.zeros(6)
        q1 = np.array([0.1, 0.2, z, 0.0, 0.0, 0.0])
        return np.vstack([q0, q1]), 1

    def _resample_action_plan(self, positions, arm, own, *, start_q=None):
        return np.asarray(positions, dtype=float)


class SceneObstaclesTest(unittest.TestCase):
    def test_merge_and_persist_obstacles_across_calls(self):
        from ni_rua.robot.motion.proxy_contract import validate_proxies
        harness = _DummyHarness()
        args1 = {
            "obstacles": [
                {
                    "id": "tower_base",
                    "min_m": [-0.05, -0.35, 0.75],
                    "max_m": [0.05, -0.25, 0.85],
                    "evidence": "cam_head obs-1",
                },
            ]
        }
        out1 = merge_and_persist_obstacles(harness, args1)
        self.assertEqual(len(out1["obstacles"]), 1)
        validated1 = validate_proxies({"obstacles": out1["obstacles"]})
        self.assertEqual(len(validated1["obstacles"]), 1)

        args2 = {"mode": "preview"}
        out2 = merge_and_persist_obstacles(harness, args2)
        self.assertEqual(len(out2["obstacles"]), 1)
        self.assertEqual(out2["obstacles"][0]["id"], "tower_base")
        validated2 = validate_proxies({"obstacles": out2["obstacles"]})
        self.assertEqual(validated2["obstacles"][0]["id"], "tower_base")

        # Episode switch resets persistent obstacles
        harness.identity["episode_id"] = "ep1"
        out3 = merge_and_persist_obstacles(harness, {"mode": "preview"})
        self.assertNotIn("obstacles", out3)

    def test_plan_move_with_obstacle_clearance_builds_two_segment_arc(self):
        adapter = _DummyAdapter()
        obstacles = [
            {"id": "holder", "min_m": [-0.03, -0.32, 0.75], "max_m": [0.03, -0.28, 0.86], "margin_m": 0.005},
        ]
        target = {"position_m": [0.12, -0.30, 0.90], "quaternion_xyzw": [0.0, 0.7071, 0.0, 0.7071]}
        plan, resampled = plan_move_with_obstacle_clearance(
            adapter,
            None,
            "right",
            {"path": {"position_tolerance_m": 0.005}},
            target,
            [-0.12, -0.30, 0.90],
            [0.0] * 6,
            obstacles,
            None,
            [],
            cap_remaining=100,
        )
        self.assertIsNotNone(plan)
        self.assertIsNotNone(resampled)
        self.assertEqual(len(adapter.calls), 3)
        # Second call is the raised midpoint above holder_top + 0.13
        self.assertGreater(adapter.calls[1][1][2], 0.98)

    def test_merge_and_persist_exempts_placed_obstacles_at_grasp_position_m_stacking_target(self):
        harness = _DummyHarness()
        harness._persistent_obstacles_identity = dict(harness.identity)
        harness._persistent_obstacles = {
            "placed_left_set1": {
                "id": "placed_left_set1",
                "min_m": [0.076, -0.224, 0.75],
                "max_m": [0.124, -0.176, 0.80],
                "margin_m": 0.003,
                "evidence": "auto_recorded_release",
            },
            "placed_right_other": {
                "id": "placed_right_other",
                "min_m": [-0.224, -0.224, 0.75],
                "max_m": [-0.176, -0.176, 0.80],
                "margin_m": 0.003,
                "evidence": "auto_recorded_release",
            },
        }
        args = {
            "stages": [
                {"id": "stack2", "kind": "move", "arm": "left", "grasp_position_m": [0.10, -0.20, 0.83]},
            ]
        }
        out = merge_and_persist_obstacles(harness, args)
        self.assertEqual([b["id"] for b in out["obstacles"]], ["placed_right_other"])

    def test_multistage_batch_preserves_placed_obstacle_for_transit_arc(self):
        harness = _DummyHarness()
        harness._persistent_obstacles_identity = dict(harness.identity)
        harness._persistent_obstacles = {
            "placed_left_doll1": {
                "id": "placed_left_doll1",
                "min_m": [-0.02, -0.32, 0.75],
                "max_m": [0.02, -0.28, 0.88],
                "margin_m": 0.003,
                "evidence": "auto_recorded_release",
            },
        }
        args = {
            "stages": [
                {"id": "touch_doll1", "kind": "move", "arm": "left", "grasp_position_m": [0.0, -0.30, 0.80]},
                {"id": "carry_across", "kind": "move", "arm": "left", "grasp_position_m": [0.12, -0.30, 0.82]},
            ]
        }
        out = merge_and_persist_obstacles(harness, args)
        self.assertEqual([b["id"] for b in out["obstacles"]], ["placed_left_doll1"])
        adapter = _DummyAdapter()
        # Stage 1 targets doll1 directly -> bypassed (no arc)
        plan1, arc1 = plan_move_with_obstacle_clearance(
            adapter, None, "left", {"path": {"position_tolerance_m": 0.005}, "input_grasp_pose": {"position_m": [0.0, -0.30, 0.80]}},
            {"position_m": [0.0, -0.30, 0.80], "quaternion_xyzw": [0, 0.7071, 0, 0.7071]},
            [-0.12, -0.30, 0.82], [0.0] * 6, out["obstacles"], None, [], cap_remaining=100,
        )
        self.assertIsNone(arc1)
        # Stage 2 transits from [-0.12, -0.30] to [0.12, -0.30] across doll1 -> arcs over doll1
        adapter2 = _DummyAdapter()
        plan2, arc2 = plan_move_with_obstacle_clearance(
            adapter2, None, "left", {"path": {"position_tolerance_m": 0.005}, "input_grasp_pose": {"position_m": [0.12, -0.30, 0.82]}},
            {"position_m": [0.12, -0.30, 0.82], "quaternion_xyzw": [0, 0.7071, 0, 0.7071]},
            [-0.12, -0.30, 0.82], [0.0] * 6, out["obstacles"], None, [], cap_remaining=100,
        )
        self.assertIsNotNone(arc2)

    def test_record_placed_obstacles_does_not_fabricate_implicit_boxes(self):
        from ni_rua.robot.motion.scene_obstacles import record_placed_obstacles
        harness = _DummyHarness()
        place_action = {
            "stages": [
                {"id": "place_down", "kind": "move", "arm": "right", "action": {"input_grasp_pose": {"position_m": [0.05, 0.02, 0.82]}}},
                {"id": "release", "kind": "gripper", "arm": "right", "action": {"gripper_events": [{"goal_normalized": 1.0}]}},
                {"id": "withdraw", "kind": "move", "arm": "right", "action": {"input_grasp_pose": {"position_m": [0.05, 0.02, 0.94]}}},
            ]
        }
        record_placed_obstacles(harness, place_action, {
            "status": "completed",
            "completed_stage_ids": ["place_down", "release", "withdraw"],
            "stage_results": {},
        })
        self.assertEqual(getattr(harness, "_persistent_obstacles", {}), {})

    def test_obstacles_from_object_table_auto_bridges_and_exempts_target(self):
        from ni_rua.robot.motion.scene_obstacles import filter_review_obstacles, obstacles_from_object_table
        harness = _DummyHarness()
        table = dict(
            columns=["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
            table_z=0.765,
            rows=[
                ["o1", [0.00, -0.30, 0.810], 0.860, 0.765, 0.060, 0.060, 0.0, None, "table", "ok"],
                ["o2", [0.12, -0.30, 0.790], 0.820, 0.765, 0.050, 0.050, 0.0, None, "table", "ok"],
                ["o3", [-0.15, -0.20, 0.780], 0.790, 0.765, 0.040, 0.040, 0.0, None, "table", "low:depth"],
            ],
        )
        harness._obs3d_memo = ("obs-1", {"object_table": (table, None, {})})
        auto_boxes = obstacles_from_object_table(harness, "obs-1")
        # o3 is skipped because quality starts with low:
        self.assertEqual([b["id"] for b in auto_boxes], ["vggt_auto_o1", "vggt_auto_o2"])
        # Without explicit obstacle_object_ids, merge_and_persist_obstacles does not inject unrequested vggt_auto_* boxes
        args_no_explicit = {
            "observation_id": "obs-1",
            "stages": [
                {"id": "grasp_o2", "kind": "move", "arm": "right", "grasp_position_m": [0.12, -0.30, 0.80]},
            ],
        }
        self.assertNotIn("obstacles", merge_and_persist_obstacles(harness, args_no_explicit))
        # With explicit obstacle_object_ids=['o1'], o1 is converted into an obstacle box
        args_explicit = dict(args_no_explicit, obstacle_object_ids=["o1"])
        merged = merge_and_persist_obstacles(harness, args_explicit)
        self.assertEqual([b["id"] for b in merged["obstacles"]], ["o1"])
        explicit = obstacles_from_object_table(harness, "obs-1", object_ids=["o1"])
        self.assertEqual([b["id"] for b in filter_review_obstacles(explicit, {})], ["o1"])

    def test_robot_action_input_schema_accepts_obstacle_and_compliance_controls(self):
        from ni_rua.runtime.tools.registry import validator_for
        from ni_rua.runtime.actions.robot_action_schema import ROBOT_ACTION_TOOL

        validator = validator_for(ROBOT_ACTION_TOOL["inputSchema"])
        payload = {
            "mode": "preview",
            "observation_id": "obs-1",
            "obstacle_object_ids": ["o1", "o2"],
            "obstacle_policy": "stateless",
            "ignore_obstacle_ids": ["o3"],
            "target_exemption_preset": "normal",
            "clearance_preset": "high",
            "max_upward_dz_m": 0.04,
            "min_aperture_m": 0.002,
            "max_aperture_m": 0.080,
            "max_joint_travel_rad": 4.5,
            "max_backtracking_rad": 2.0,
            "max_tracking_divergence_m": 0.025,
            "stages": [
                {
                    "id": "transit",
                    "kind": "move",
                    "arm": "right",
                    "grasp_position_m": [0.12, -0.30, 0.85],
                    "angles_deg": [0.0, 0.0, 0.0],
                    "clearance_height_m": 0.16,
                    "max_upward_dz_m": 0.03,
                }
            ],
        }
        errors = list(validator.iter_errors(payload))
        self.assertEqual(errors, [])

    def test_stateless_obstacle_policy_applies_only_to_current_call_without_persisting(self):
        harness = _DummyHarness()
        table = dict(
            columns=["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
            table_z=0.765,
            rows=[
                ["o1", [0.00, -0.30, 0.810], 0.860, 0.765, 0.060, 0.060, 0.0, None, "table", "ok"],
                ["o2", [0.12, -0.30, 0.790], 0.820, 0.765, 0.050, 0.050, 0.0, None, "table", "ok"],
            ],
        )
        harness._obs3d_memo = ("obs-1", {"object_table": (table, None, {})})
        args = {
            "observation_id": "obs-1",
            "obstacle_policy": "stateless",
            "obstacle_object_ids": ["o1", "o2"],
            "ignore_obstacle_ids": ["o2"],
            "target_exemption_preset": "tight",
            "stages": [
                {"id": "move1", "kind": "move", "arm": "right", "grasp_position_m": [0.15, -0.25, 0.85]},
            ],
        }
        compiled_stages = [{"id": "move1", "kind": "move", "arm": "right", "action": {"path": {}}}]
        merged = merge_and_persist_obstacles(harness, args, compiled_stages)
        self.assertEqual([b["id"] for b in merged["obstacles"]], ["o1"])
        self.assertAlmostEqual(compiled_stages[0]["action"]["path"]["target_exemption_radius_m"], 0.030)
        # Registry remains empty because policy == 'stateless'
        self.assertEqual(getattr(harness, "_persistent_obstacles", {}), {})

    def test_observation_scoped_vggt_obstacles_evict_on_new_observation_while_manual_boxes_persist(self):
        harness = _DummyHarness()
        table_obs1 = dict(
            columns=["id", "centre_m", "top_z", "bottom_z", "length", "width", "yaw_deg", "tilt_deg", "support", "quality"],
            table_z=0.765,
            rows=[
                ["o1", [0.02, -0.28, 0.84], 0.92, 0.765, 0.08, 0.08, 0.0, None, "table", "ok"],
                ["o2", [-0.14, -0.24, 0.80], 0.84, 0.765, 0.05, 0.05, 0.0, None, "table", "ok"],
            ],
        )
        harness._obs3d_memo = ("obs-1", {"object_table": (table_obs1, None, {})})
        call_obs1 = {
            "observation_id": "obs-1",
            "obstacle_object_ids": ["o1"],
            "obstacles": [
                {
                    "id": "manual_guard",
                    "min_m": [-0.30, -0.35, 0.765],
                    "max_m": [-0.24, -0.29, 0.880],
                    "evidence": "calibrated_world_fixture",
                }
            ],
            "stages": [{"id": "move_obs1", "kind": "move", "arm": "right", "grasp_position_m": [0.18, -0.22, 0.86]}],
        }
        merged_obs1 = merge_and_persist_obstacles(harness, call_obs1)
        self.assertEqual({b["id"] for b in merged_obs1["obstacles"]}, {"manual_guard", "o1"})

        merged_same_obs1 = merge_and_persist_obstacles(
            harness,
            {"observation_id": "obs-1", "stages": [{"id": "m_retry", "kind": "move", "arm": "right", "grasp_position_m": [0.18, -0.22, 0.86]}]},
        )
        self.assertEqual({b["id"] for b in merged_same_obs1["obstacles"]}, {"manual_guard", "o1"})

        merged_obs2 = merge_and_persist_obstacles(
            harness,
            {"observation_id": "obs-2", "stages": [{"id": "m_obs2", "kind": "move", "arm": "right", "grasp_position_m": [0.02, -0.28, 0.84]}]},
        )
        self.assertEqual([b["id"] for b in merged_obs2["obstacles"]], ["manual_guard"])
        self.assertNotIn("o1", harness._persistent_obstacles)

        cleared = merge_and_persist_obstacles(
            harness,
            {"observation_id": "obs-2", "obstacle_policy": "clear", "stages": [{"id": "m", "kind": "move", "arm": "right", "grasp_position_m": [0.1, -0.2, 0.8]}]},
        )
        self.assertNotIn("obstacles", cleared)
        self.assertEqual(harness._persistent_obstacles, {})



# --- Folded from tests/test_tool_descriptions.py ---

def _so101_profile(empty_close_aperture_m=None):
    import json
    from pathlib import Path
    from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment
    from support.paths import CONFIGS_ROOT
    configs = CONFIGS_ROOT
    document = json.loads((configs / 'so101.example.json').read_text())
    document['gripper']['empty_close_aperture_m'] = empty_close_aperture_m
    return embodiment_profile(validate_deployment(document, base=configs))


EMBODIMENTS = {'arx_x5_default': None, 'so101_unmeasured_jaw': _so101_profile(),
               'so101_measured_jaw': _so101_profile(0.0045)}


PUBLIC_NAMES = {
    'task_plan', 'robot_observe', 'robot_capabilities', 'robot_information',
    'robot_contact_overlay', 'robot_action_status', 'robot_wait',
    'history_list_actions', 'history_view_observation', 'history_read_trace',
    'history_annotate_action', 'geometry_query', 'program_run', 'program_status',
    'vision_compare', 'vision_review', 'robot_policy_step', 'depth_query',
    'robot_action', 'operation_memory',
}


def schema_nodes(value, path=()):
    """Walk real input schemas, excluding examples and conditional selectors."""
    if isinstance(value, dict):
        yield path, value
        for key, child in value.items():
            if key not in {'description', 'examples', 'default', 'enum', 'const', 'if', 'not'}:
                yield from schema_nodes(child, path + (key,))
    elif isinstance(value, list):
        for index, child in enumerate(value):
            yield from schema_nodes(child, path + (index,))


def validation_schema(value):
    if isinstance(value, dict):
        return {key: validation_schema(item) for key, item in value.items()
                if key not in {'description', 'examples'}}
    if isinstance(value, list):
        return [validation_schema(item) for item in value]
    return value


class ToolDescriptionTests(unittest.TestCase):
    def test_all_public_input_fields_have_help(self):
        definitions = public_definitions()
        self.assertEqual({item['name'] for item in definitions}, PUBLIC_NAMES)
        for definition in definitions:
            self.assertTrue(definition['description'].strip(), definition['name'])
            for path, node in schema_nodes(definition['inputSchema']):
                for field, field_schema in node.get('properties', {}).items():
                    with self.subTest(tool=definition['name'], path=path, field=field):
                        self.assertTrue(field_schema.get('description', '').strip())

    def test_every_embodiment_catalogue_has_help_and_valid_examples_for_both_roles(self):
        """The profile-rendered catalogues (design 13 WP2): every field documented, every example admitted."""
        for name, profile in EMBODIMENTS.items():
            registry = create_registry(runtime=SimpleNamespace(embodiment=profile))
            definitions = public_definitions(profile=profile)
            self.assertEqual({item['name'] for item in definitions}, PUBLIC_NAMES)
            for definition in definitions:
                for path, node in schema_nodes(definition['inputSchema']):
                    for field, field_schema in node.get('properties', {}).items():
                        with self.subTest(embodiment=name, tool=definition['name'], path=path, field=field):
                            self.assertTrue(field_schema.get('description', '').strip())
                for example in definition['inputSchema']['examples']:
                    for role in registry.spec(definition['name'], role='actor').roles & {'actor', 'program'}:
                        with self.subTest(embodiment=name, tool=definition['name'], role=role, example=example):
                            self.assertEqual(registry.validate(definition['name'], example, role=role).definition,
                                             definition)

    def test_every_published_example_validates_through_both_allowed_roles(self):
        registry = create_registry()
        for definition in public_definitions():
            schema = definition['inputSchema']
            self.assertTrue(schema.get('examples'), definition['name'])
            for path, node in schema_nodes(schema):
                for example in node.get('examples', []):
                    with self.subTest(tool=definition['name'], path=path, example=example):
                        self.assertEqual(list(validator_for(node).iter_errors(example)), [])
            for example in schema['examples']:
                for role in ('actor', 'program'):
                    actor_spec = registry.spec(definition['name'], role='actor')
                    if role in actor_spec.roles:
                        with self.subTest(tool=definition['name'], role=role, example=example):
                            spec = registry.validate(definition['name'], example, role=role)
                            self.assertEqual(spec.definition, definition)

    def test_task_attention_schema_matches_partial_updates_and_text_limits_for_both_roles(self):
        registry = create_registry()
        entry = dict(concern='Current uncertainty', check='Use the next visible endpoint; adjust if it differs.',
                     when='Before placement')
        ledger = dict(id='retention', expected='The object remains aligned after lift.', status='pending',
                      basis='unknown', next_check='Compare the next task-progress frame.')
        valid = [dict(plan='Brief task understanding and approach.', ledger=[]),
                 dict(plan='Brief plan.', attention=[], ledger=[]), dict(operation='read'),
                 dict(operation='update', plan='Changed understanding.'),
                 dict(operation='update', attention=[]),
                 dict(operation='update', ledger=[ledger]),
                 dict(operation='update', attention=[entry] * 3),
                 dict(operation='update', attention=[dict(entry, check='证' * 1600)])]
        invalid = [{}, dict(plan='Missing the required initial ledger.'), dict(attention=[]), dict(operation='update'),
                   dict(operation='read', attention=[]), dict(operation='read', plan='x'),
                   dict(plan='x', attention=None), dict(plan='x', attention=[dict(entry, when=' ')]),
                   dict(plan='x', attention=[dict(entry, check='x' * 1601)]),
                   dict(plan='x', attention=[dict(entry, extra='unknown')]),
                   dict(operation='update', attention=[entry] * 4)]
        invalid += [dict(plan='x', ledger=[dict(ledger, status='bad')]),
                    dict(plan='x', ledger=[dict(ledger, id=' ')]),
                    dict(plan='x', ledger=[dict(ledger, extra='unknown')]),
                    dict(plan='x', ledger=[ledger] * 4)]
        for role in ('actor', 'program'):
            schema = registry.spec('task_plan', role=role).definition['inputSchema']
            validator = validator_for(schema)
            for arguments in valid:
                with self.subTest(role=role, arguments=arguments):
                    self.assertTrue(validator.is_valid(arguments))
                    registry.validate('task_plan', arguments, role=role)
            for arguments in invalid:
                with self.subTest(role=role, arguments=arguments):
                    self.assertFalse(validator.is_valid(arguments))
            self.assertEqual(schema['properties']['attention']['maxItems'], 3)
            self.assertNotIn('minItems', schema['properties']['attention'])
        definition = registry.spec('task_plan', role='actor').definition
        self.assertEqual(definition, registry.spec('task_plan', role='program').definition)
        self.assertIn('[] clears it', definition['description'])
        self.assertIn('normal task-progress', definition['description'])
        self.assertIn('evidence ledger', definition['description'])
        self.assertIn('one to three short ledger rows', definition['description'])
        self.assertIn('change the contact, route or target mechanism', definition['description'])

    def test_publishing_and_editing_a_catalog_do_not_mutate_source_contracts(self):
        legacy_before = deepcopy(TOOLS)
        action_before = deepcopy(ROBOT_ACTION_TOOL)
        catalog = public_definitions()
        self.assertEqual(TOOLS, legacy_before)
        self.assertEqual(ROBOT_ACTION_TOOL, action_before)
        for definition in catalog:
            definition['description'] = 'local display edit'
            definition['inputSchema'].clear()
        self.assertEqual(TOOLS, legacy_before)
        self.assertEqual(ROBOT_ACTION_TOOL, action_before)
        action = next(item for item in public_definitions() if item['name'] == 'robot_action')
        self.assertEqual(action['inputSchema'], action_before['inputSchema'])
        self.assertTrue(action['description'].startswith(action_before['description']))

    def test_typed_geometry_depth_requests_preserve_distinct_legal_branches(self):
        registry = create_registry()
        geometry = [dict(kind='help'),
            dict(kind='pixel_plane', observation_id='obs', camera_id='head', pixel=[320, 240],
                 plane_origin_m=[0, 0, 2], plane_normal=[0, 0, 1], plane_origin_alt_m=[0, 0, 3]),
            dict(kind='project_geometry', observation_id='obs', camera_id='head', points_m=[[0, 0, 2]]),
            dict(kind='sparse_track', observation_ids=['before', 'after'], camera_id='head',
                 roi=[100.5, 80.5, 400., 320.]),
            dict(kind='alignment', observation_ids=['heldout1', 'heldout2', 'heldout3'], camera_id='head', scene_version=1),
            dict(kind='preview_ee', observation_id='obs', camera_id='head', axis_length_m=.03,
                 action=dict(frame_id='env_origin', mode='absolute', path=dict(waypoints=[
                     dict(left=dict(position_m=[0, 0, 2], quaternion_xyzw=[0, 0, 0, 1]))])))]
        depth = [dict(kind='help'), dict(kind='skill_help'), dict(kind='target_status'),
            dict(kind='region_geometry', target_id='target'),
            dict(kind='grasp_candidates', target_id='target', count=3),
            dict(kind='place_candidates', target_id='target', held_dimensions_m=[.1, .2, .3], grasp_offset_m=[0, 0, .02]),
            dict(kind='refresh_target', target_id='target', observation_id='obs', roi=[0, 0, 1, 1]),
            dict(kind='depth_points', observation_id='obs', camera_id='head', pixels=[[1., 2.]], radius_px=0)]
        for name, requests in [('geometry_query', geometry + depth), ('depth_query', depth)]:
            for role in ('actor', 'program'):
                for request in requests:
                    with self.subTest(tool=name, role=role, request=request):
                        registry.validate(name, dict(request=request), role=role)
                if name == 'geometry_query':
                    # The compatibility route preserves owner validation;
                    # test_depth_integration exercises its real rejection path.
                    with self.assertRaises(ValueError):
                        registry.validate(name, dict(request=dict(kind='unknown_depth_kind')), role=role)
                    continue
                for request in [dict(kind='region_geometry', target_id='target', observation_id='obs'),
                        dict(kind='refresh_target', target_id='target', observation_id='obs', camera_id='head', roi=[0, 0, 4, 4]),
                        dict(kind='depth_points', observation_id='obs', camera_id='head', pixels=[[1, 2]], radius_px=5)]:
                    with self.subTest(tool=name, role=role, invalid=request), self.assertRaises(ValueError):
                        registry.validate(name, dict(request=request), role=role)

    def test_request_types_and_distinct_field_meanings_survive_tool_publication(self):
        catalog = {item['name']: item for item in public_definitions()}
        for name, fields in (('geometry_query', GEOMETRY_REQUEST_DESCRIPTIONS),
                             ('depth_query', DEPTH_REQUEST_DESCRIPTIONS)):
            definition = catalog[name]
            request = definition['inputSchema']['properties']['request']
            self.assertIn("tool description's input field guide", request['description'])
            self.assertNotIn('fields below', request['description'])
            self.assertNotIn('properties below', definition['description'])
            if name == 'geometry_query':
                self.assertIn('Use depth_query for the complete field contract', definition['description'])
                self.assertNotIn(DEPTH_REQUEST_DESCRIPTIONS['depth_tolerance_m'], definition['description'])
            for field, text in fields.items():
                with self.subTest(tool=name, field=field):
                    matches = [node['properties'][field] for _, node in schema_nodes(request)
                               if field in node.get('properties', {})]
                    self.assertTrue(matches)
                    self.assertTrue(any(value.get('description') == text for value in matches))
                    self.assertEqual(definition['description'].count(text), 1)
            for path, node in schema_nodes(request):
                for field, child in node.get('properties', {}).items():
                    with self.subTest(tool=name, path=path, field=field):
                        self.assertIn('type', child)
                        self.assertIn(child['description'], definition['description'])

    def test_all_tool_help_preserves_bounds_defaults_and_branch_logic_without_size_heuristics(self):
        catalog = {item['name']: item for item in public_definitions()}
        help_text = catalog['robot_action']['description']
        for required in ('stages[].id:', 'Maximum characters: 80', 'Maximum characters: 1000',
                         'Unique items: true', 'stages[].angles_deg[1]:',
                         'Minimum (inclusive): -90', 'Maximum (inclusive): 90',
                         'if (kind equals "move"', 'requires id, kind, arm'):
            self.assertIn(required, help_text)
        field_help = catalog['robot_action']['inputSchema']['properties']['stages']['items']['properties']
        for name in ('position_m', 'grasp_position_m', 'goal_normalized'):
            self.assertEqual(help_text.count('stages[].' + name + ': ' + field_help[name]['description']), 1)
        self.assertIn('Maximum characters: 200', catalog['vision_review']['description'])
        observe = catalog['robot_observe']
        self.assertIn('Input field guide', observe['description'])
        for text in catalog['robot_wait']['inputSchema']['properties'].values():
            self.assertIn(text['description'], catalog['robot_wait']['description'])
        for required in ('at most 1 fields', 'otherwise', 'at least one of (requires plan | requires attention | requires ledger)'):
            self.assertIn(required, catalog['task_plan']['description'])
        for required in ('if pixels_a is present, requires pixels_b', 'if pixels_b is present, requires pixels_a'):
            self.assertIn(required, catalog['geometry_query']['description'])
        depth = catalog['depth_query']['description']
        for required in ('exactly one of', 'must not satisfy', 'requires target_id',
                         'Minimum (inclusive): 0.0005', 'Maximum (inclusive): 0.15', 'Default: 0.015'):
            self.assertIn(required, depth)

    def test_core_and_frequently_used_tools_are_loaded_initially(self):
        core = {'task_plan', 'robot_observe', 'robot_action', 'program_run', 'operation_memory',
                'geometry_query', 'history_view_observation', 'robot_wait',
                'robot_contact_overlay', 'history_list_actions',
                # replies name these two as the read of what they left out; a host without working tool search
                # cannot load a deferred tool
                'robot_action_status', 'program_status'}
        for item in public_definitions():
            self.assertEqual(item.get('deferLoading', False), item['name'] not in core, item['name'])

    def test_program_help_explains_complete_text_and_repeated_delivery(self):
        catalog = {item['name']: item for item in public_definitions()}
        run_help = catalog['program_run']['description']
        for required in ('stdout', 'stderr', 'all available saved log text', 'middle diagnostics',
                         'source_complete', 'program_status', "detail='full'",
                         'client.artifact', 'exact saved bytes', "*_delivery.status='unchanged'", 'Inline image data URLs',
                         'automatically forward', 'inputImage', 'without re-encoding',
                         'visual_feedback', 'previous_tool_call_id'):
            with self.subTest(tool='program_run', required=required):
                self.assertIn(required, run_help)
        status_help = catalog['program_status']['description']
        self.assertIn('program_run', status_help)
        for name in ('program_run', 'program_status'):
            self.assertIn('all saved stdout/stderr text', catalog[name]['inputSchema']['properties']['detail']['description'])

    def test_shared_action_shapes_retain_each_fields_distinct_meaning(self):
        action = next(item for item in public_definitions() if item['name'] == 'robot_action')
        properties = action['inputSchema']['properties']
        stage = properties['stages']['items']
        stage_properties = stage['properties']
        self.assertIn('center between fingers', stage_properties['grasp_position_m']['description'])
        self.assertIn('link6/flange', stage_properties['position_m']['description'])
        self.assertNotEqual(stage_properties['grasp_position_m']['description'],
                            stage_properties['position_m']['description'])
        self.assertIn('Move only', stage_properties['gripper_goal_normalized']['description'])
        self.assertIn('Gripper stage only', stage_properties['goal_normalized']['description'])
        move = next(branch['then'] for branch in stage['allOf']
                    if branch['if']['properties']['kind']['const'] == 'move')
        for field in ('grasp_position_m', 'position_m'):
            self.assertEqual(stage_properties[field]['description'],
                             move['properties'][field]['description'])
        boxes = [properties['obstacles']['items'], properties['held_object'],
                 *properties['held_objects']['properties'].values()]
        for box in boxes:
            with self.subTest(box=box):
                self.assertIn('lower', box['properties']['min_m']['description'])
                self.assertIn('upper', box['properties']['max_m']['description'])
                self.assertNotEqual(box['properties']['min_m']['description'],
                                    box['properties']['max_m']['description'])

    def test_geometry_help_identifies_unapplied_scene_proposals(self):
        geometry = next(item for item in public_definitions() if item['name'] == 'geometry_query')
        for required in ('proposed patch', 'without applying it', 'existing saved version'):
            self.assertIn(required, geometry['description'])
        branches = geometry['inputSchema']['properties']['request']['oneOf']
        fields = next(part['properties'] for part in branches if part['properties']['kind']['const'] == 'build_scene')
        self.assertIn('does not apply', fields['base_scene_version']['description'])
        self.assertIn('at least 6 accepted', fields['primitive']['description'])



# --- Folded from tests/test_tool_registry_prose_bounds.py ---

NUMERIC_KEYWORDS = ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
                    'minItems', 'maxItems', 'minLength', 'maxLength')
NUMBER = re.compile(r'(?<![\w.])[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?')


def actor_catalog():
    config = Config.from_dict({'features': {'pose_skill': True}})
    runtime = SimpleNamespace(config=config, disabled_tools=config.features.disabled_tools,
        observation_profile=config.features.observation_profile, motion_review=config.features.motion_review,
        enable_wait=True, vision_compare=config.features.vision_compare, visual_reviewer=object(),
        action_policy=True, capabilities=None, programs=True)
    return full_registry().catalog(runtime, role='actor')


def numbers_in(text):
    values = []
    for token in NUMBER.findall(text):
        try:
            values.append(float(token))
        except ValueError:
            pass
    return values


NUMBER_WORDS = ('zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine', 'ten', 'eleven',
                'twelve')


def has_number(text, value):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return False
    if any(math.isclose(found, float(value), rel_tol=1e-9, abs_tol=1e-12) for found in numbers_in(text)):
        return True
    return float(value) in range(len(NUMBER_WORDS)) and re.search(
        r'(?<![A-Za-z])' + NUMBER_WORDS[int(value)] + r'(?![A-Za-z])', text, re.I) is not None


def has_word(text, word):
    return re.search(r'(?<![\w])' + re.escape(word) + r'(?![\w])', text) is not None


def has_value(text, value):
    if isinstance(value, bool) or value is None:
        return has_word(text, json.dumps(value))
    if isinstance(value, (int, float)):
        return has_number(text, value)
    return has_word(text, str(value))


def walk(schema, path, root):
    """Yield (json_path, subschema) for every schema node below and including `schema`."""
    if not isinstance(schema, dict):
        return
    ref = schema.get('$ref')
    if ref and ref.startswith('#/'):
        target = root
        for part in ref[2:].split('/'):
            target = target[part]
        yield from walk(target, path, root)
    yield path, schema
    for key, child in (schema.get('properties') or {}).items():
        yield from walk(child, path + '/properties/' + key, root)
    for key in ('items', 'additionalProperties', 'not', 'if', 'then', 'else'):
        if isinstance(schema.get(key), dict):
            yield from walk(schema[key], path + '/' + key, root)
    for key in ('anyOf', 'oneOf', 'allOf', 'prefixItems'):
        for index, child in enumerate(schema.get(key) or []):
            yield from walk(child, path + '/%s/%d' % (key, index), root)
    for key in ('$defs', 'definitions'):
        for name, child in (schema.get(key) or {}).items():
            yield from walk(child, path + '/%s/%s' % (key, name), root)


def gaps(tool):
    """Return (path, keyword, detail) for every schema constraint missing from the visible prose."""
    found = []
    schema = tool['inputSchema']
    for path, node in walk(schema, '', schema):
        text = (node.get('description') or '') + '\n' + (tool.get('description') or '')
        for keyword in NUMERIC_KEYWORDS:
            if keyword in node and not has_number(text, node[keyword]):
                found.append((path, keyword, node[keyword]))
        if 'enum' in node:
            for value in node['enum']:
                if not has_value(text, value):
                    found.append((path, 'enum', value))
        if 'default' in node:
            value = node['default']
            scalar = isinstance(value, (bool, int, float)) or value is None or (isinstance(value, str) and value)
            if not re.search(r'defaults?\b|defaulting', text, re.I) or (scalar and not has_value(text, value)):
                found.append((path, 'default', value))
    return found


class MatcherTest(unittest.TestCase):
    def test_numbers(self):
        self.assertTrue(has_number('range 0.05 to 1.5 m', 0.05))
        self.assertTrue(has_number('at most 1e-3 rad', 0.001))
        self.assertTrue(has_number('between -1 and 1', -1))
        self.assertTrue(has_number('limit 12 items', 12))
        self.assertFalse(has_number('limit 120 items', 12))
        self.assertFalse(has_number('about 5 mm', 0.005))
        self.assertFalse(has_number('v1.25', 1.2))
        self.assertTrue(has_number('Exactly three finite numbers', 3))
        self.assertFalse(has_number('Exactly three finite numbers', 2))
        self.assertFalse(has_number('someone', 1))

    def test_words(self):
        self.assertTrue(has_word('mode a or b', 'a'))
        self.assertFalse(has_word('some data', 'a'))
        self.assertTrue(has_value('flag is true', True))
        self.assertTrue(has_value('pick contact_stop', 'contact_stop'))

    def test_gap_detection_and_walk(self):
        tool = dict(name='t', description='Tool. Up to 5 things.', inputSchema=dict(type='object', properties=dict(
            a=dict(type='number', minimum=0.1, description='x'),
            b=dict(type='array', maxItems=5, items=dict(type='string', enum=['p', 'q'], description='p only'))),
            required=[]))
        self.assertEqual([(p, k) for p, k, _ in gaps(tool)],
                         [('/properties/a', 'minimum'), ('/properties/b/items', 'enum')])


class RegistryProseBoundsTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.catalog = actor_catalog()

    def test_catalog_contains_both_plugins_and_pose_skill(self):
        names = {tool['name'] for tool in self.catalog}
        self.assertTrue({'robotic_view', 'stereo_guide', 'pose_plan', 'robot_action', 'program_run'} <= names)

    def test_every_bound_enum_and_default_is_stated_in_prose(self):
        problems = []
        for tool in self.catalog:
            problems += ['%s%s %s=%r' % (tool['name'], path, keyword, value) for path, keyword, value in gaps(tool)]
        self.assertEqual(problems, [], '\n' + '\n'.join(problems))

    def test_top_level_combinator_rules_are_stated_in_prose(self):
        # MCP publishes inputSchema without top-level anyOf/oneOf/allOf/if/then/else (mcp_server.portable_input_schema;
        # Claude Code drops such tools). The generated "input: ... Object rules: ..." clause must carry them.
        from ni_rua.cortex.agents.mcp_server import TOP_LEVEL_COMBINATORS
        phrases = {'anyOf': 'at least one of (', 'oneOf': 'exactly one of (', 'allOf': 'all of (', 'if': 'if ('}
        problems = []
        for tool in self.catalog:
            present = [key for key in TOP_LEVEL_COMBINATORS if key in tool['inputSchema']]
            guide = [line for line in tool['description'].splitlines() if line.startswith('input: ')]
            for key in present:
                if key in phrases and not any('Object rules:' in line and phrases[key] in line for line in guide):
                    problems.append('%s top-level %s is not stated in the description' % (tool['name'], key))
            if any(key in ('then', 'else') for key in present) and not any('if (' in line for line in guide):
                problems.append('%s top-level then/else is not stated in the description' % tool['name'])
        self.assertEqual(problems, [])



# --- Catalogue profiles (features.tool_catalog_profile) ---

CATALOG_VARIANTS = {
    'arx_x5_default': {},
    'so101_unmeasured_jaw': {'profile': EMBODIMENTS['so101_unmeasured_jaw']},
    'perception_enabled': dict(vggt_height='optional', object_table=True, scene3d=True,
                               observe3d=(True, True, False)),
    'pose_skill': dict(pose_skill=True),
}
SO101 = 'so101_unmeasured_jaw'
BOUND_KEYS = ('minimum', 'maximum', 'exclusiveMinimum', 'exclusiveMaximum',
              'minItems', 'maxItems', 'minLength', 'maxLength')


def robot_of(options):
    profile = options.get('profile')
    return profile.robot_model_id if profile is not None else 'arx-x5'


def guide_walk(node, path=''):
    """``(path, schema node)`` of every node of an input schema, paths in the guide's notation (``stages[].id``;
    branches of a union and conditional ``then``/``else`` share the path of the object that declares them)."""
    if not isinstance(node, dict):
        return
    yield (path or 'input'), node
    for field, child in (node.get('properties') or {}).items():
        yield from guide_walk(child, path + '.' + field if path else field)
    if isinstance(node.get('items'), dict):
        yield from guide_walk(node['items'], path + '[]')
    for index, child in enumerate(node.get('prefixItems', [])):
        yield from guide_walk(child, f'{path}[{index}]')
    for key in ('oneOf', 'anyOf', 'allOf'):
        for part in node.get(key, []):
            yield from guide_walk(part, path)
    for branch_key in ('then', 'else'):
        if isinstance(node.get(branch_key), dict):
            yield from guide_walk(node[branch_key], path)


def property_paths(schema):
    return {path for path, _ in guide_walk(schema) if path != 'input' and not path.endswith(']')}


def line_path(line):
    """Path a guide line is about: ``path: ...`` or ``  path when <selector>: ...``."""
    head = line.strip().split(':', 1)[0]
    return head.split(' when ', 1)[0]


def lines_of(lines, path):
    return [line for line in lines if line_path(line) == path]


def descriptions_of(schema):
    return sorted({node['description'] for _, node in guide_walk(schema) if isinstance(node.get('description'), str)},
                  key=len, reverse=True)


def names_in_rules(node):
    """Field names the structural rules of one schema node mention (required, dependentRequired, if/then/else chains)."""
    names = set(node.get('required', []))
    for field, needs in (node.get('dependentRequired') or {}).items():
        names |= {field, *needs}
    condition = node.get('if', {})
    own = (set(condition) == {'required'} and isinstance(node.get('then'), dict) and set(node['then']) == {'properties'}
           and set(node['then']['properties']) <= set(condition['required']) and 'else' not in node)
    if 'if' in node and not own:  # "if X is given then X satisfies ..." is a plain declaration of X, not a rule
        names |= set(condition.get('properties', {})) | set(condition.get('required', []))
        for branch in ('then', 'else'):
            target = node.get(branch)
            if isinstance(target, dict):
                names |= set(target.get('required', [])) | set(names_in_rules(target))
    return names


def real_catalogs(profile):
    return {name: public_definitions(**options, catalog_profile=profile) for name, options in CATALOG_VARIANTS.items()}


ARX_VARIANTS = {name: options for name, options in CATALOG_VARIANTS.items() if robot_of(options) == 'arx-x5'}


def catalog_runtime(profile, **features):
    """A runtime namespace for ``create_registry(...).catalog`` under ``features.tool_catalog_profile=profile``."""
    config = Config.from_dict({'features': dict(tool_catalog_profile=profile, **features)})
    return SimpleNamespace(disabled_tools=(), observation_profile='rgb_proprio_calibrated_v1', motion_review=True,
                           enable_wait=True, vision_compare=True, visual_reviewer=True, action_policy=None,
                           capabilities=None, programs=True, config=config, features=config.features)


PERCEPTION_FEATURES = dict(vggt_height=True, object_table=True, scene3d=True, object_table_observe=True,
                           scene3d_observe=True)


def published_catalog(profile, **features):
    runtime = catalog_runtime(profile, **features)
    return create_registry(runtime=runtime).catalog(runtime, role='actor')


def field_line(description, path):
    """The first guide line of ``path`` (``path: ...``, not a form line) of a tool description, or None."""
    return next((line for line in description.split('\n') if line.startswith(path + ':') and not line.startswith(' ')), None)


def reference_of(line):
    """``(path, tool)`` when ``line`` is a shared-field reference (``catalog_guide.SHARED_TEXT``), else None."""
    from ni_rua.runtime.tools.catalog_guide import SHARED_TEXT
    head, tail = SHARED_TEXT.split('{tool}')
    path, mark, text = line.partition(': ')
    if mark and not path.startswith(' ') and text.startswith(head) and text.endswith(tail) and len(text) > len(head + tail):
        return path, text[len(head):len(text) - len(tail)]
    return None


class ToolCatalogProfileTests(unittest.TestCase):
    def test_profiles_change_descriptions_and_deferral_only(self):
        from ni_rua.runtime.tools.catalog_guide import LAYERS, PROFILES, PROSE_PROFILES, cold_tools
        self.assertEqual(PROFILES, ('full_v1', 'guide_only_v1', 'compact_v1', 'layered_v1', 'layered_v2', 'layered_v3'))
        self.assertEqual(PROSE_PROFILES, ('full_v1', 'guide_only_v1'))
        self.assertEqual(tuple(LAYERS), ('layered_v1', 'layered_v2', 'layered_v3'))
        for name, options in CATALOG_VARIANTS.items():
            base = public_definitions(**options)
            full = public_definitions(**options, catalog_profile='full_v1')
            self.assertEqual(json.dumps(base), json.dumps(full), name)
            deferred = {item['name'] for item in full if item.get('deferLoading')}
            for profile in PROFILES:
                catalog = public_definitions(**options, catalog_profile=profile)
                with self.subTest(embodiment=name, profile=profile):
                    self.assertEqual([item['name'] for item in catalog], [item['name'] for item in base])
                    for item, original in zip(catalog, base):
                        self.assertEqual(json.dumps(item['inputSchema']), json.dumps(original['inputSchema']), item['name'])
                        self.assertEqual({k for k in item if k not in ('description', 'deferLoading')},
                                         {k for k in original if k not in ('description', 'deferLoading')})
                    expected = deferred | (cold_tools(robot_of(options), profile) & {item['name'] for item in catalog}) \
                        if profile in LAYERS else deferred
                    self.assertEqual({item['name'] for item in catalog if item.get('deferLoading')}, expected)
        with self.assertRaises(ValueError):
            public_definitions(catalog_profile='bogus')

    def test_cold_selection_applies_only_to_the_robot_with_evidence(self):
        from ni_rua.runtime.tools.catalog_guide import (COLD, COLD_TOOLS, EVIDENCE_ROBOT, LAYERS, PROFILES, cold_fields,
                                                          cold_tools, reference_document)
        self.assertEqual(EVIDENCE_ROBOT, 'arx-x5')
        self.assertEqual(cold_tools('arx-x5'), COLD_TOOLS)
        self.assertEqual(cold_tools('so101'), frozenset())
        for profile in PROFILES:   # only a layered profile that defers publishes a tool on demand
            self.assertEqual(cold_tools('arx-x5', profile), COLD_TOOLS if LAYERS.get(profile, {}).get('defer') else frozenset())
        for tool in COLD:
            self.assertIs(cold_fields(tool, 'arx-x5'), COLD[tool])
            self.assertIsNone(cold_fields(tool, 'so101'))
        self.assertIsNone(cold_fields('no_such_tool', 'arx-x5'))
        full = public_definitions(catalog_profile='full_v1')
        so101 = CATALOG_VARIANTS[SO101]
        for profile in LAYERS:
            with self.subTest(profile=profile):
                arx = public_definitions(catalog_profile=profile)
                self.assertEqual({i['name'] for i in arx if i.get('deferLoading')},
                                 {i['name'] for i in full if i.get('deferLoading')} | cold_tools('arx-x5', profile))
                self.assertIsNotNone(reference_document(arx))
                layered = public_definitions(**so101, catalog_profile=profile)
                self.assertEqual(json.dumps(layered), json.dumps(public_definitions(**so101, catalog_profile='compact_v1')))
                self.assertIsNone(reference_document(layered, so101['profile'].robot_model_id))
                self.assertIsNone(reference_document(layered, 'so101'))

    def test_cold_lists_are_whitelists_that_track_the_schema(self):
        from ni_rua.runtime.tools.catalog_guide import COLD, _COLD_GEOMETRY_KINDS, _COLD_ROBOT_ACTION
        by_name = {item['name']: item for item in public_definitions()}
        action = by_name['robot_action']['inputSchema']
        paths = property_paths(action)
        self.assertFalse(COLD['robot_action']('stages[].a_field_added_later', set()))
        self.assertFalse(COLD['robot_action']('a_field_added_later', set()))
        self.assertEqual(sorted(name for name in _COLD_ROBOT_ACTION if name not in paths), [])
        geometry = [next(i for i in public_definitions(**CATALOG_VARIANTS[v]) if i['name'] == 'geometry_query')
                    for v in ('arx_x5_default', 'perception_enabled')]  # scene3d is published with the perception flags
        kinds = {value for item in geometry for _, node in guide_walk(item['inputSchema'])
                 for value in (node.get('properties', {}).get('kind', {}).get('enum', [])
                               + [node.get('properties', {}).get('kind', {}).get('const')])}
        self.assertEqual(sorted(_COLD_GEOMETRY_KINDS - kinds), [])
        required = {name for path, node in guide_walk(action) if path == 'stages[]' for name in node.get('required', [])}
        self.assertTrue(required)
        for name in sorted(required):
            self.assertFalse(COLD['robot_action'](f'stages[].{name}', set()), name)

    def test_no_real_tool_needs_the_prose_fallback(self):
        from ni_rua.runtime.tools.catalog_guide import guide_lines
        for name, catalog in real_catalogs('compact_v1').items():
            for item in catalog:
                with self.subTest(embodiment=name, tool=item['name']):
                    guide_lines(item['inputSchema'])

    def test_unsupported_schemas_are_published_with_the_prose_guide(self):
        from ni_rua.runtime.tools.catalog_guide import UnsupportedSchema, guide_lines
        from ni_rua.runtime.tools.definitions import _publish
        base = dict(type='object', additionalProperties=False, required=['a'])
        schemas = {
            'unknown keyword': dict(base, properties=dict(a=dict(type='string', description='A.')),
                                    patternProperties={'^x': dict(type='number')}),
            'alternative declaring fields': dict(base, properties=dict(a=dict(type='string', description='A.')),
                                                 anyOf=[dict(properties=dict(b=dict(type='number', maximum=1)))]),
        }
        for label, schema in schemas.items():
            item = dict(type='function', name='synthetic', description='Synthetic tool.', inputSchema=schema)
            with self.subTest(label):
                with self.assertRaises(UnsupportedSchema):
                    guide_lines(schema)
                self.assertIsInstance(UnsupportedSchema('x'), ValueError)
                compact = _publish(deepcopy(item), 'compact_v1', 'arx-x5')
                self.assertEqual(compact['description'], _publish(deepcopy(item), 'full_v1')['description'])
                self.assertNotIn('deferLoading', compact)

    def test_compact_catalogue_is_lossless(self):
        from ni_rua.runtime.tools.catalog_guide import guide_lines
        from ni_rua.core.contracts import canonical

        def token(value):
            return canonical(int(value) if isinstance(value, float) and value.is_integer() else value)

        def implied(key, value, node, siblings):
            """Bounds another part of the signature already states: a length bound every sibling enum/const member
            satisfies, and an item count equal to the number of elements of a fixed tuple."""
            if key in ('minLength', 'maxLength'):
                members = [m for other in siblings for m in ([other['const']] if 'const' in other else other.get('enum', []))]
                test = (lambda m: isinstance(m, str) and len(m) >= value) if key == 'minLength' else (
                    lambda m: isinstance(m, str) and len(m) <= value)
                return bool(members) and all(test(m) for m in members)
            prefix = node.get('prefixItems')
            return bool(prefix) and key in ('minItems', 'maxItems') and len(prefix) == value \
                and node.get('minItems') == node.get('maxItems') == len(prefix)

        for name, catalog in real_catalogs('compact_v1').items():
            for item in catalog:
                lines, cold = guide_lines(item['inputSchema'])
                self.assertEqual(cold, [])
                tool = item['name']
                nodes_at = {}
                for path, node in guide_walk(item['inputSchema']):
                    nodes_at.setdefault(path, []).append(node)
                for path, node in guide_walk(item['inputSchema']):
                    if isinstance(node.get('description'), str):
                        with self.subTest(embodiment=name, tool=tool, path=path, what='description'):
                            self.assertIn(node['description'], item['description'])
                    own = path
                    while own.endswith(']') and not lines_of(lines, own):  # a folded item node is stated by its container
                        own = own[:own.rfind('[')]
                    text = '\n'.join(lines_of(lines, own))
                    values = [token(value) for value in ([node['const']] if 'const' in node else node.get('enum', []))]
                    values += [token(node[key]) for key in BOUND_KEYS
                               if key in node and not implied(key, node[key], node, nodes_at[path])]
                    for value in values:
                        with self.subTest(embodiment=name, tool=tool, path=path, value=value):
                            self.assertRegex(text, r'(?<!\d)(?<!\d\.)' + re.escape(value) + r'(?!\d)')

    def test_compact_catalogue_states_every_structural_rule(self):
        """Every field a schema's rules (required, dependentRequired, if/then/else chains) name, and every closed or
        size-limited object, is stated on the rule or form lines of that object's path."""
        from ni_rua.runtime.tools.catalog_guide import guide_lines
        for name, catalog in real_catalogs('compact_v1').items():
            for item in catalog:
                schema = item['inputSchema']
                lines, _ = guide_lines(schema)
                text_at = {}
                for path in {path for path, _ in guide_walk(schema)}:
                    text = '\n'.join(lines_of(lines, path))
                    for description in descriptions_of(schema):
                        text = text.replace(description, '')
                    text_at[path] = text
                for path, node in guide_walk(schema):
                    text = text_at[path]
                    if not text and path.endswith(']'):
                        text = text_at.get(path[:path.rfind('[')], '')
                    for field in sorted(names_in_rules(node)):
                        with self.subTest(embodiment=name, tool=item['name'], path=path, field=field):
                            self.assertRegex(text, r'(?<![\w.])' + re.escape(field) + r'(?![\w])')
                    if node.get('additionalProperties') is False and 'properties' in node:
                        with self.subTest(embodiment=name, tool=item['name'], path=path, rule='closed'):
                            self.assertIn('closed', text)
                    for key in ('minProperties', 'maxProperties'):
                        if key in node:
                            with self.subTest(embodiment=name, tool=item['name'], path=path, rule=key):
                                self.assertIn(str(node[key]), text)

    def test_task_plan_else_chain_fields_are_in_the_root_rules(self):
        from ni_rua.runtime.tools.catalog_guide import guide_lines
        item = next(i for i in public_definitions() if i['name'] == 'task_plan')
        lines, _ = guide_lines(item['inputSchema'])
        text = '\n'.join(lines_of(lines, 'input'))
        for description in descriptions_of(item['inputSchema']):
            text = text.replace(description, '')
        for field in ('plan', 'attention', 'ledger'):
            self.assertRegex(text, r'(?<![\w.])' + field + r'(?![\w])', field)

    def test_discriminated_unions_state_their_forms_and_selectors(self):
        from ni_rua.core.contracts import canonical
        from ni_rua.runtime.tools.catalog_guide import guide_lines
        checked = set()
        for name, catalog in real_catalogs('compact_v1').items():
            for item in catalog:
                schema = item['inputSchema']
                lines, _ = guide_lines(schema)
                for path, node in guide_walk(schema):
                    branches = node.get('oneOf')
                    if not branches:
                        continue
                    constants = [{f: ([v['const']] if 'const' in v else v['enum']) for f, v in b.get('properties', {}).items()
                                  if 'const' in v or 'enum' in v} for b in branches]
                    shared = [f for f in constants[0] if all(f in c for c in constants)]
                    if not shared:
                        continue
                    checked.add((item['name'], path))
                    parent = [l for l in lines_of(lines, path) if not l.startswith(' ')]
                    forms = [l for l in lines_of(lines, path) if l.startswith(' ')]
                    with self.subTest(embodiment=name, tool=item['name'], path=path):
                        self.assertEqual(len(parent), 1)
                        self.assertEqual(len(forms), len(branches))
                        self.assertIn('Rules:', parent[0])
                        for field in shared:
                            for branch, line, found in zip(branches, forms, constants):
                                selector = line.split(' when ', 1)[1].split(':', 1)[0]
                                minimal = f'{field}=' + '|'.join(canonical(v) for v in found[field])
                                if field in branch.get('required', []):
                                    self.assertEqual(selector.split(', ')[0], minimal)
                                else:
                                    self.assertNotEqual(selector.split(', ')[0], minimal)
                                    self.assertTrue(selector.startswith(minimal))
        self.assertLessEqual({('geometry_query', 'request'), ('depth_query', 'request'), ('vision_review', 'input')}, checked)

    def test_layered_catalogue_hides_nothing_and_publishes_the_reference(self):
        from ni_rua.runtime.tools.catalog_guide import cold_fields, guide_lines, reference_document
        for name, options in CATALOG_VARIANTS.items():
            if robot_of(options) != 'arx-x5':
                continue
            catalog = public_definitions(**options, catalog_profile='layered_v1')
            document = reference_document(catalog)
            self.assertIsNotNone(document, name)
            document_lines = set(document.splitlines())
            with_cold = 0
            for item in catalog:
                tool, schema = item['name'], item['inputSchema']
                lines, cold_items = guide_lines(schema, cold_fields(tool, 'arx-x5'))
                with_cold += bool(cold_items)
                described = {line_path(line) for line in lines}
                cold_paths = {entry.split(':', 1)[0] for entry in cold_items}
                self.assertFalse(described & cold_paths, tool)
                for path in property_paths(schema):
                    with self.subTest(embodiment=name, tool=tool, path=path):
                        self.assertIn(path, described | cold_paths)
                full_lines, _ = guide_lines(schema)
                for line in full_lines:
                    if line_path(line) in cold_paths:
                        self.assertIn(line, document_lines, (tool, line_path(line)))
                self.assertEqual('## ' + tool in document, bool(cold_items), tool)
            self.assertGreater(with_cold, 0, name)
            self.assertIsNone(reference_document([i for i in catalog if i['name'] not in ('robot_action', 'geometry_query')]))

    def test_layered_v2_hides_nothing_and_publishes_the_reference(self):
        from ni_rua.runtime.tools.catalog_guide import (LAYERS, _listed, _render, cold_fields, cold_forms,
                                                          guide_lines, reference_document)
        with_cold = 0
        for name, options in ARX_VARIANTS.items():
            catalog = public_definitions(**options, catalog_profile='layered_v2')
            document_lines = set(reference_document(catalog).splitlines())
            for item in catalog:
                tool, schema = item['name'], item['inputSchema']
                cold = cold_fields(tool, 'arx-x5')
                v1_paths = [entry.split(':', 1)[0] for entry in guide_lines(schema, cold)[1]]
                entries = _render(schema, cold, cold_forms(tool, 'arx-x5'), LAYERS['layered_v2'])[1]
                listed = {path for path, _, _ in entries}
                with_cold += bool(listed)
                described = [line.split(':', 1)[0] for line in item['description'].split('\n') if not line.startswith(' ')]
                full_lines, _ = guide_lines(schema)
                for path in listed:
                    self.assertIn(path, described, (name, tool, path))
                for path in v1_paths:
                    with self.subTest(embodiment=name, tool=tool, path=path):
                        self.assertTrue(path in listed or _listed(path, listed))
                        for line in full_lines:
                            if line_path(line) == path:
                                self.assertIn(line, document_lines)
        self.assertGreater(with_cold, 0)

    def test_layered_v2_cold_fields_keep_the_start_of_their_help(self):
        from ni_rua.runtime.tools.catalog_guide import GIST_LIMIT, LAYERS, _render, cold_fields, cold_forms, gist
        short, long_text = 'word ' * 20, 'word ' * 60
        self.assertEqual(gist(short.strip()), (short.strip(), False))
        self.assertEqual(gist('x' * GIST_LIMIT), ('x' * GIST_LIMIT, False))
        cut, flag = gist(long_text.strip())
        self.assertTrue(flag and cut.endswith('...') and long_text.startswith(cut[:-4]) and len(cut) <= GIST_LIMIT + 4)
        seen = dict(complete=0, cut=0, bare=0)
        for name, options in ARX_VARIANTS.items():
            for item in public_definitions(**options, catalog_profile='layered_v2'):
                tool = item['name']
                lines = set(item['description'].split('\n'))
                entries = _render(item['inputSchema'], cold_fields(tool, 'arx-x5'), cold_forms(tool, 'arx-x5'),
                                  LAYERS['layered_v2'])[1]
                for path, signature, help_text in entries:
                    with self.subTest(embodiment=name, tool=tool, path=path):
                        head = path + ':' + (' ' + signature if signature else '')
                        if not help_text:
                            seen['bare'] += 1
                            self.assertIn(head, lines)
                            continue
                        text, was_cut = gist(help_text)
                        self.assertIn(head + ' -- ' + text, lines)
                        if len(help_text) <= GIST_LIMIT:
                            seen['complete'] += 1
                            self.assertEqual((text, was_cut), (help_text, False))
                        else:
                            seen['cut'] += 1
                            self.assertTrue(was_cut and text.endswith('...') and len(text) <= GIST_LIMIT + 4)
                            self.assertTrue(help_text.startswith(text[:-4]))
        self.assertGreater(seen['cut'], 0)
        self.assertGreater(seen['complete'], 0)

    def test_layered_v2_names_the_rarely_used_union_forms_without_their_rules(self):
        from ni_rua.runtime.tools.catalog_guide import (COLD_FORMS, COLD_FORMS_HEADER, LAYERS, REFERENCE_NAME, _render,
                                                          cold_forms, reference_document)
        omitted_total = 0
        for name, options in ARX_VARIANTS.items():
            compact = next(i for i in public_definitions(**options, catalog_profile='compact_v1') if i['name'] == 'geometry_query')
            v1 = next(i for i in public_definitions(**options, catalog_profile='layered_v1') if i['name'] == 'geometry_query')
            catalog = public_definitions(**options, catalog_profile='layered_v2')
            v2 = next(i for i in catalog if i['name'] == 'geometry_query')
            document_lines = set(reference_document(catalog).splitlines())
            items = _render(v2['inputSchema'], None, cold_forms('geometry_query', 'arx-x5'), LAYERS['layered_v2'])[2]
            selectors = [selector for path, selector in items if path == 'request']
            self.assertEqual(len(selectors), len(items))
            prefix = '  request when '
            compact_forms = [line for line in compact['description'].split('\n') if line.startswith(prefix)]
            v2_lines = v2['description'].split('\n')
            with self.subTest(embodiment=name):
                omitted = [line for line in compact_forms if any(line.startswith(prefix + selector + ':') for selector in selectors)]
                self.assertEqual(len(omitted), len(selectors))
                self.assertGreater(len(compact_forms), len(omitted))
                omitted_total += len(omitted)
                for line in compact_forms:
                    if line in omitted:
                        self.assertNotIn(line, v2_lines)
                        self.assertIn(line, document_lines)
                    else:
                        self.assertIn(line, v2_lines)
                for selector in selectors:
                    self.assertFalse(any(line.startswith(prefix + selector + ':') for line in v2_lines), selector)
                self.assertIn(COLD_FORMS_HEADER.format(name=REFERENCE_NAME, path='request', forms='; '.join(selectors)),
                              v2['description'])
                v1_lines = v1['description'].split('\n')
                for line in compact_forms:
                    self.assertIn(line, v1_lines)
        self.assertGreater(omitted_total, 0)
        self.assertEqual(set(COLD_FORMS), {'geometry_query'})

    def test_share_lines_references_only_earlier_loaded_tools_with_identical_lines(self):
        from ni_rua.runtime.tools.catalog_guide import share_lines
        references = 0
        for name, options in ARX_VARIANTS.items():
            catalog = public_definitions(**options, catalog_profile='layered_v2')
            shared = share_lines(catalog)
            references += self.check_shared(catalog, shared, name)
            self.assertEqual(share_lines(shared), shared, name)
            for tool in {tool for item in shared for line in item['description'].split('\n')
                         for found in [reference_of(line)] if found for tool in [found[1]]}:
                rest = share_lines([item for item in catalog if item['name'] != tool])
                for item in rest:
                    self.assertFalse(any((reference_of(line) or (None, None))[1] == tool
                                         for line in item['description'].split('\n')), (name, tool, item['name']))
        self.assertGreater(references, 0)
        perception = share_lines(public_definitions(**ARX_VARIANTS['perception_enabled'], catalog_profile='layered_v2'))
        self.assertTrue(any(reference_of(line) for item in perception for line in item['description'].split('\n')))

    def check_shared(self, before, after, label):
        """Contract of ``share_lines(before) == after``; returns the number of references."""
        from ni_rua.runtime.tools.catalog_guide import SHARED_MIN
        order = [item['name'] for item in after]
        self.assertEqual(order, [item['name'] for item in before], label)
        by_name = {item['name']: item for item in after}
        count = 0
        for old, new in zip(before, after):
            self.assertEqual({k: v for k, v in new.items() if k != 'description'},
                             {k: v for k, v in old.items() if k != 'description'}, (label, old['name']))
            if old.get('deferLoading'):
                self.assertEqual(new, old, (label, old['name']))
                continue
            old_lines, new_lines = old['description'].split('\n'), new['description'].split('\n')
            self.assertEqual(len(old_lines), len(new_lines), (label, old['name']))
            for was, now in zip(old_lines, new_lines):
                found = reference_of(now)
                if found is None:
                    self.assertEqual(now, was, (label, old['name']))
                    continue
                count += 1
                path, tool = found
                self.assertTrue(was.startswith(path + ': ') and len(was) - len(path + ': ') >= SHARED_MIN, (label, now))
                self.assertIn(tool, by_name, (label, now))
                self.assertFalse(by_name[tool].get('deferLoading'), (label, now))
                self.assertLess(order.index(tool), order.index(old['name']), (label, now))
                self.assertEqual(field_line(by_name[tool]['description'], path), was, (label, old['name'], path))
        return count

    def test_share_lines_leaves_short_cold_and_deferred_lines_alone(self):
        from ni_rua.runtime.tools.catalog_guide import COLD_GIST_HEADER, SHARED_MIN, SHARED_TEXT, share_lines
        intro = next(item for item in public_definitions(catalog_profile='layered_v2'))['description']
        intro = intro[intro.index('Input field guide'):].split('\n')[0]
        long_text, cold_header = 'x' * SHARED_MIN, COLD_GIST_HEADER.format(name='ref.md', example='cold')

        def tool(name, *rows, deferred=False):
            item = dict(name=name, description='Head.\n' + intro + '\n' + '\n'.join(rows))
            return dict(item, deferLoading=True) if deferred else item

        guide = ('a: ' + long_text, 'short: ' + long_text[:-1], cold_header, 'cold: ' + long_text)
        first, second = tool('first', *guide), tool('second', *guide)
        result = share_lines([first, second])
        self.assertEqual(result[0], first)
        rows = result[1]['description'].split('\n')
        self.assertEqual(rows[2], 'a: ' + SHARED_TEXT.format(tool='first'))
        self.assertEqual(rows[3:], ['short: ' + long_text[:-1], cold_header, 'cold: ' + long_text])
        for pair in ([tool('first', *guide, deferred=True), second], [first, tool('second', *guide, deferred=True)]):
            self.assertEqual(share_lines(pair), pair)
        self.assertEqual(share_lines([second, dict(name='plain', description='No guide.')])[1]['description'], 'No guide.')

    def test_only_layered_v2_publishes_shared_references_in_the_registry_catalogue(self):
        from ni_rua.runtime.tools.catalog_guide import share_lines
        for profile in ('full_v1', 'compact_v1', 'layered_v1'):
            catalog = published_catalog(profile, **PERCEPTION_FEATURES)
            self.assertFalse([item['name'] for item in catalog
                              for line in item['description'].split('\n') if reference_of(line)], profile)
        for profile in ('layered_v2', 'layered_v3'):
            shared = published_catalog(profile, **PERCEPTION_FEATURES)
            with patch('ni_rua.runtime.tools.catalog_guide.share_lines', lambda tools: tools):
                unshared = published_catalog(profile, **PERCEPTION_FEATURES)
            self.assertGreater(self.check_shared(unshared, shared, 'registry ' + profile), 0)
            self.assertEqual(json.dumps(shared), json.dumps(share_lines(unshared)))

    def test_layered_v3_is_layered_v2_with_every_tool_loaded_and_no_shell_command(self):
        from ni_rua.runtime.tools.catalog_guide import (COLD_GIST_HEADER, COLD_LINE_HEADER, COLD_TOOLS, LAYERS,
                                                          REFERENCE_NAME, _grep_example, _render, cold_fields, cold_forms)
        self.assertEqual({key: LAYERS['layered_v3'][key] for key in ('defer', 'index', 'command')},
                         dict(defer=False, index=False, command=False))
        self.assertEqual({key: value for key, value in LAYERS['layered_v3'].items() if key not in ('defer', 'index', 'command')},
                         {key: value for key, value in LAYERS['layered_v2'].items() if key not in ('defer', 'index', 'command')})
        changed = 0
        for name, options in CATALOG_VARIANTS.items():
            full = public_definitions(**options)
            v2 = public_definitions(**options, catalog_profile='layered_v2')
            v3 = public_definitions(**options, catalog_profile='layered_v3')
            with self.subTest(embodiment=name):
                # No tool is published for on-demand discovery that the default catalogue loads.
                self.assertEqual([item.get('deferLoading') for item in v3], [item.get('deferLoading') for item in full])
                for earlier, item in zip(v2, v3):
                    cold = cold_fields(item['name'], robot_of(options))
                    entries = _render(item['inputSchema'], cold, cold_forms(item['name'], robot_of(options)),
                                      LAYERS['layered_v3'])[1] if cold else []
                    expected = earlier['description']
                    if entries:   # the same text, with the header that names the line instead of a command
                        command = COLD_GIST_HEADER.format(name=REFERENCE_NAME, example=_grep_example(entries[0][0]))
                        self.assertEqual(expected.count(command), 1, item['name'])
                        expected = expected.replace(command, COLD_LINE_HEADER.format(name=REFERENCE_NAME))
                        changed += 1
                    self.assertEqual(item['description'], expected, item['name'])
                    self.assertNotIn(item['name'], COLD_TOOLS if item.get('deferLoading') else ())
        self.assertGreater(changed, 0)

    def test_on_demand_index_names_every_deferred_tool_once_in_catalogue_order(self):
        from ni_rua.runtime.tools.catalog_guide import ON_DEMAND_HEADER, on_demand_index
        from cortex.agents.test_agent_prompt_parity import BACKEND_WORDS
        counted = 0
        for name, options in CATALOG_VARIANTS.items():
            for profile in ('full_v1', 'compact_v1', 'layered_v1', 'layered_v2', 'layered_v3'):
                catalog = public_definitions(**options, catalog_profile=profile)
                with self.subTest(embodiment=name, profile=profile):
                    self.assertIsNone(on_demand_index([item for item in catalog if not item.get('deferLoading')]))
                    index = on_demand_index(catalog)
                    deferred = [item['name'] for item in catalog if item.get('deferLoading')]
                    self.assertTrue(deferred)
                    self.assertTrue(index.startswith(ON_DEMAND_HEADER + '\n'))
                    rows = index[len(ON_DEMAND_HEADER) + 1:].splitlines()
                    self.assertEqual(len(rows), len(deferred))
                    for row, tool in zip(rows, deferred):
                        self.assertTrue(row.startswith(f'- {tool}:'), (row, tool))
                    counted += len(rows)
                    for word in BACKEND_WORDS:
                        self.assertNotIn(word, index)
        self.assertGreater(counted, 0)

    def test_guide_header_names_only_the_notation_in_use(self):
        from ni_rua.runtime.tools.catalog_guide import _NOTATION
        catalog = {item['name']: item for item in public_definitions(catalog_profile='compact_v1')}

        def present(tool):
            return {key for key, text in _NOTATION if text in catalog[tool]['description']}

        def anywhere(tool, test):
            return any(test(node) for _, node in guide_walk(catalog[tool]['inputSchema']))

        self.assertLess(present('robot_wait'), {key for key, _ in _NOTATION})
        for tool in catalog:
            used = present(tool)
            with self.subTest(tool=tool):
                if not anywhere(tool, lambda n: 'dependentRequired' in n):
                    self.assertNotIn('needs', used)
                if not anywhere(tool, lambda n: n.get('type') == 'array'):
                    self.assertNotIn('list', used)
                    self.assertNotIn('count', used)
                if not anywhere(tool, lambda n: 'default' in n):
                    self.assertNotIn('default', used)
                if not anywhere(tool, lambda n: n.get('additionalProperties') is False and 'properties' in n):
                    self.assertNotIn('closed', used)
                if not anywhere(tool, lambda n: 'oneOf' in n or 'if' in n or 'allOf' in n):
                    self.assertNotIn('forms', used)
        self.assertTrue(present('robot_action') > present('robot_wait'))

    def test_external_backend_demotes_layered_to_compact(self):
        from ni_rua.runtime.tools.builtins import catalog_profile_of

        def config(profile, backend):
            return SimpleNamespace(features=SimpleNamespace(tool_catalog_profile=profile),
                                   agent=SimpleNamespace(backend=backend))

        for layered in ('layered_v1', 'layered_v2', 'layered_v3'):
            self.assertEqual(catalog_profile_of(config(layered, 'external')), 'compact_v1')
        for profile in ('full_v1', 'compact_v1', 'layered_v1', 'layered_v2', 'layered_v3'):
            for backend in ('codex', 'claude', None):
                self.assertEqual(catalog_profile_of(config(profile, backend)), profile)
            if not profile.startswith('layered'):
                self.assertEqual(catalog_profile_of(config(profile, 'external')), profile)
        self.assertEqual(catalog_profile_of(None), 'full_v1')
        self.assertEqual(catalog_profile_of(SimpleNamespace()), 'full_v1')

    def test_every_agent_backend_renders_the_default_catalogue_form(self):
        """The default ``layered_v3`` reaches the catalogue of Codex, Claude Code and OpenCode; the external backend
        has no working directory for the reference file and gets ``compact_v1`` (its tool descriptions state every
        field themselves). The configured profile is never replaced by anything else."""
        from ni_rua.apps.gateway.workbench import Workbench
        from ni_rua.core.contracts import canonical
        from ni_rua.runtime.tools.builtins import catalog_profile_of

        def catalogue(backend, *, context_delivery=None, **features):
            agent = {'backend': backend}
            if context_delivery is not None:
                agent[backend] = {'context_delivery': context_delivery}
            config = Config.from_dict({'agent': agent, 'features': features})
            return catalog_profile_of(config), Workbench.actor_tools(SimpleNamespace(config=config))

        profiles = {backend: catalogue(backend)[0] for backend in ('codex', 'claude_code', 'opencode', 'external')}
        self.assertEqual(profiles, dict(codex='layered_v3', claude_code='layered_v3', opencode='layered_v3',
                                        external='compact_v1'))
        layered = canonical(catalogue('codex', tool_catalog_profile='layered_v3')[1])
        compact = canonical(catalogue('codex', tool_catalog_profile='compact_v1')[1])
        self.assertNotEqual(layered, compact)
        self.assertNotEqual(canonical(catalogue('external')[1]), layered)
        self.assertEqual(canonical(catalogue('codex')[1]), layered)
        self.assertEqual(canonical(catalogue('external', tool_catalog_profile='compact_v1')[1]),
                         canonical(catalogue('external')[1]))
        # 四后端默认均为 delta：claude_code 与 opencode 默认目录与 codex 字节一致；
        # 显式切回 context_delivery='full' 仅影响 robot_observe，不改变目录形态。
        for backend in ('claude_code', 'opencode'):
            changed = {new['name'] for old, new in zip(catalogue('codex')[1], catalogue(backend)[1], strict=True)
                       if canonical(old) != canonical(new)}
            self.assertEqual(changed, set(), backend)
            full_changed = {new['name'] for old, new in zip(
                catalogue('codex')[1], catalogue(backend, context_delivery='full')[1], strict=True
            ) if canonical(old) != canonical(new)}
            self.assertEqual(full_changed, {'robot_observe'}, backend)

    def test_selecting_field_of_conditional_forms_keeps_every_declared_value(self):
        # Forms select only "a"; the field's own declaration allows three values and a default.
        from ni_rua.runtime.tools.catalog_guide import guide_lines, signature
        declared = {'type': 'string', 'enum': ['a', 'b', 'c'], 'default': 'a'}
        schema = {'type': 'object', 'required': ['kind'], 'properties': {'kind': declared, 'p': {'type': 'number'}},
                  'allOf': [{'if': {'properties': {'kind': {'const': 'a'}}, 'required': ['kind']},
                             'then': {'properties': {'kind': {'const': 'a'}}, 'required': ['p']}}]}
        lines, _ = guide_lines(schema)
        self.assertIn('kind: ' + signature(declared), lines)

    def test_form_only_narrowing_of_a_field_declared_outside_the_forms_is_qualified(self):
        from ni_rua.runtime.tools.catalog_guide import guide_lines, signature
        outside, narrowed = {'type': 'number', 'minimum': 0}, {'type': 'number', 'minimum': 0, 'maximum': 5}
        schema = {'type': 'object', 'required': ['kind'],
                  'properties': {'kind': {'enum': ['a', 'b']}, 'v': outside},
                  'allOf': [{'if': {'properties': {'kind': {'const': 'a'}}, 'required': ['kind']},
                             'then': {'properties': {'v': {'maximum': 5}}}}]}
        lines, _ = guide_lines(schema)
        alternatives = next(line for line in lines if line.startswith('v: '))[len('v: '):].split(' / ')
        # The declaration outside the forms stands alone; the narrowed one names its form before the signature.
        self.assertEqual(alternatives[0], signature(outside))
        self.assertEqual(len(alternatives), 2)
        self.assertTrue(alternatives[1].endswith(signature(narrowed)))
        self.assertNotEqual(alternatives[1], signature(narrowed))

    def test_tuple_signatures_state_their_item_count(self):
        from ni_rua.runtime.tools.catalog_guide import signature
        text = {'type': 'string'}
        pair = [text, {'type': 'number'}]
        one = signature({'type': 'array', 'prefixItems': [text], 'minItems': 1, 'maxItems': 1})
        self.assertNotEqual(one, signature({'type': 'array', 'items': text}))
        exact = signature({'type': 'array', 'prefixItems': pair, 'minItems': 2, 'maxItems': 2})
        self.assertNotEqual(signature({'type': 'array', 'prefixItems': pair, 'items': False}), exact)
        self.assertNotEqual(signature({'type': 'array', 'prefixItems': pair, 'minItems': 2}), exact)

    def test_signatures_refuse_what_they_cannot_state(self):
        from ni_rua.runtime.tools.catalog_guide import UnsupportedSchema, guide_lines, signature
        for node in ({'enum': ['', 'a'], 'minLength': 1},      # the value set says more than the schema allows
                     {'const': 7, 'maximum': 5},
                     {'minimum': 0, 'maximum': 1},              # bounds without a type constrain numbers only
                     {'maxLength': 5}):
            with self.subTest(node=node), self.assertRaises(UnsupportedSchema):
                signature(node)
        for node in ({'enum': ['a', 'b'], 'minLength': 1}, {'type': 'string', 'enum': ['a']}, {'const': 3, 'minimum': 1}):
            with self.subTest(node=node):
                self.assertEqual(signature(node), signature({key: value for key, value in node.items()
                                                             if key in ('enum', 'const')}))
        # A oneOf union whose selecting field is also declared outside the forms has no single value list.
        union = {'type': 'object', 'properties': {'kind': {'enum': ['a', 'b', 'c']}},
                 'oneOf': [{'properties': {'kind': {'const': 'a'}}, 'required': ['kind']},
                           {'properties': {'kind': {'const': 'b'}}, 'required': ['kind']}]}
        with self.assertRaises(UnsupportedSchema):
            guide_lines(union)

    def test_registry_verdicts_do_not_depend_on_the_catalogue_profile(self):
        from ni_rua.core.tool_errors import ToolInputError
        example = next(item for item in public_definitions() if item['name'] == 'robot_action')['inputSchema']['examples'][0]
        invalid = (dict(example, mode='no_such_mode'), {'timeout_s': float('nan')}, {'stages': 'not-a-list'})
        verdicts = {}
        for profile in ('full_v1', 'compact_v1', 'layered_v1', 'layered_v2', 'layered_v3'):
            registry = create_registry(runtime=SimpleNamespace(
                config=Config.from_dict({'features': {'tool_catalog_profile': profile}})))
            for role in ('actor', 'program'):
                for index, arguments in enumerate((example, *invalid)):
                    try:
                        registry.validate('robot_action', arguments, role=role)
                        verdict = 'valid'
                    except ToolInputError:
                        verdict = 'invalid'
                    verdicts.setdefault((role, index), set()).add(verdict)
        self.assertEqual({key: sorted(value) for key, value in verdicts.items() if key[1] == 0},
                         {('actor', 0): ['valid'], ('program', 0): ['valid']})
        self.assertTrue(all(value == {'invalid'} for key, value in verdicts.items() if key[1] > 0), verdicts)

    def test_profile_option_is_a_validated_choice(self):
        from ni_rua.core.config import ConfigError
        for profile in ('full_v1', 'compact_v1', 'layered_v1', 'layered_v2', 'layered_v3'):
            self.assertEqual(Config.from_dict({'features': {'tool_catalog_profile': profile}}).features.tool_catalog_profile, profile)
        self.assertEqual(Config().features.tool_catalog_profile, 'layered_v3')
        self.assertEqual(Config().features.robot_urdf_delivery, 'file')
        with self.assertRaises(ConfigError):
            Config.from_dict({'features': {'tool_catalog_profile': 'bogus'}})
        with self.assertRaises(ConfigError):
            Config.from_dict({'features': {'robot_urdf_delivery': 'bogus'}})


if __name__ == "__main__":
    unittest.main()
