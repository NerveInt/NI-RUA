"""
- test_home_check.py: CPU-only home geometry and observation-lifecycle checks; no robot imports.
- test_robot_action_reach.py: robot_action mode=reach and the split orientation bounds (approach_tolerance_rad / roll_tolerance_rad).

* Schema gating: ``robot_action_schema.build(profile)`` publishes mode=reach only with the profile's ``reach`` tag and
  the two split fields (stage and batch level, with unit, range, default, example and receipt fields in their help)
  only with ``approach_roll``; the ARX-X5 contract carries neither (its golden catalogue stays byte-identical:
  ``test_embodiment_x5_golden``).
* Dispatch: a reach request is compiled by the preview compiler, validated against the current observation under the
  preview's idle rule and answered by ``robot.reach_batch``; it writes no plan, moves nothing and keeps the
  observation current. Non-move stages and every field outside the reach set are refused before anything runs, with
  a model-facing hint pointing at the offending field. A failure of the whole query reaches the model as a tool error
  naming its reason (``ik_service_unreachable`` has its own hint).
* Effect ``read``; the typed ``data`` result matches the robot_action output contract (inspect results still do).
* Split fields resolve into each compiled move path (stage value, else the batch value) and survive continuation
  (``stage_overrides``, inherited batch defaults, batch repair).

No model, network, GPU, robot or simulator: the fake SO-101 (``support.embodiment_golden``) with ``reach_batch``
replaced by a recording fake, plus one query of the real SO-101 worker's local IK over a JSON round trip.
Run: ``PYTHONPATH=tests mamba run -n ni-rua python -m unittest test_robot_action_reach -v``.
"""
from copy import deepcopy
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from ni_rua.runtime.actions.home_check import HOME_CHECK_POLICY, HomePoseCheck
from ni_rua.core.storage.store import Store
from types import SimpleNamespace
from unittest import mock
from support import embodiment_golden as golden
from ni_rua.core.tool_errors import error_payload
from ni_rua.runtime.actions import robot_action_schema
from ni_rua.runtime.actions.robot_action_schema import build
from ni_rua.runtime.tools.builtins import action_effect, output_schemas
from ni_rua.runtime.tools.registry import validator_for
from ni_rua.runtime.actions.strategy_checkpoint import (adaptive_action_checkpoint,
    checkpoint_triggered, mechanism_family)

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_home_check.py ---

class HomePoseCheckTest(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.identity = dict(run_id="home-test", episode_id="episode-1", env_instance_id="env", env_generation=1)
        self.store = Store(Path(self.directory.name), self.identity)
        self.addCleanup(self.store.close)
        self.checker = HomePoseCheck(self.store)
        self.sequence = 0

    def state(self, step=0):
        pose = dict(position_m=[0, 0, 0], quaternion_xyzw=[0, 0, 0, 1],
                    frame_id="env_origin", tcp_config_id="arx-x5-link6-v1", source_kind="measurement")
        return dict(arms={name: dict(ee_pose=deepcopy(pose)) for name in ("left", "right")},
                    frame_id=pose["frame_id"], tcp_config_id=pose["tcp_config_id"],
                    sim_step=step, step_domain="adapter_control_step")

    def bundle(self, state=None, **updates):
        state = self.state() if state is None else state
        self.sequence += 1
        reference = self.store.put_artifact(json.dumps(state).encode(), "robot_state", "application/json")
        return dict(schema_version="0.1", **self.identity, world_kind="observed",
                    observation_id=f"obs-{self.sequence}", robot_state_ref=reference,
                    sim_step=state.get("sim_step"), step_domain=state.get("step_domain"), **updates)

    def initial(self, state=None):
        initial = self.bundle(state)
        self.checker.capture_initial(initial)
        return initial

    def result(self, state):
        return self.checker.update_current(self.bundle(state))

    def assert_status(self, status):
        result = self.checker.check(dict(self.identity))
        self.assertEqual(result["status"], status, result)
        json.dumps(result, allow_nan=False)
        return result

    @staticmethod
    def rotation(degrees):
        radians = math.radians(degrees) / 2
        return [0, 0, math.sin(radians), math.cos(radians)]

    def test_no_implicit_initial_reference_from_current(self):
        self.assert_status("unknown")
        bundle = self.bundle()
        self.checker.update_current(bundle)
        self.assertEqual(self.assert_status("unknown")["reason"], "missing_initial_observation")
        self.assertEqual(self.checker.capture_initial(bundle)["status"], "available")
        self.assert_status("home")

    def test_initial_and_returned_records_are_immutable(self):
        initial = self.initial()
        reference = self.checker.reference_record()
        state = self.state(1)
        state["arms"]["left"]["ee_pose"]["position_m"] = [0.2, 0, 0]
        current = self.bundle(state)
        self.checker.update_current(current)
        self.assert_status("away")
        rejected = self.checker.capture_initial(current)
        self.assertFalse(rejected["capture_accepted"])
        self.assertEqual(rejected["capture_reason"], "initial_reference_immutable")
        self.assertEqual(self.checker.reference_record(), reference)
        reference["arms"]["left"]["position_m"][0] = 100
        rejected["policy"]["arms"].clear()
        initial["observation_id"] = "tampered"
        current["sim_step"] = -1
        result = self.assert_status("away")
        result["arms"]["left"]["position_error_m"][0] = 0
        self.assertEqual(self.assert_status("away")["arms"]["left"]["position_error_m"][0], .2)
        self.assertEqual(self.checker.reference_record()["arms"]["left"]["position_m"], [0, 0, 0])

    def test_both_arms_required_and_either_can_be_away(self):
        self.initial()
        for name in ("left", "right"):
            with self.subTest(arm=name):
                state = self.state(1)
                state["arms"][name]["ee_pose"]["position_m"][1] = -.151
                result = self.result(state)
                self.assertEqual(result["status"], "away")
                self.assertEqual(result["arms"][name]["position_error_m"], [0, .151, 0])
                state["arms"].pop(name)
                self.assertEqual(self.result(state)["status"], "unknown")

    def test_axis_tolerance_is_not_euclidean(self):
        self.initial()
        state = self.state(1)
        state["arms"]["left"]["ee_pose"]["position_m"] = [.14, -.14, .14]
        self.assertGreater(math.hypot(.14, .14, .14), .15)
        result = self.result(state)
        self.assertEqual(result["status"], "home")
        self.assertEqual(result["arms"]["left"]["position_error_m"], [.14, .14, .14])

    def test_orientation_alone_and_arbitrary_initial_orientation(self):
        initial = self.state()
        initial["arms"]["left"]["ee_pose"]["quaternion_xyzw"] = self.rotation(67)
        self.initial(initial)
        current = deepcopy(initial)
        current["sim_step"] = 1
        current["arms"]["left"]["ee_pose"]["quaternion_xyzw"] = self.rotation(88)
        result = self.result(current)
        self.assertEqual(result["status"], "away")
        self.assertTrue(result["arms"]["left"]["position_within_tolerance"])
        self.assertAlmostEqual(result["arms"]["left"]["rotation_error_deg"], 21)

    def test_quaternion_sign_scale_and_extreme_finite_norms(self):
        self.initial()
        for quaternion in ([0, 0, 0, -2], [0, 0, 0, 1e300], [0, 0, 0, 1e-300]):
            with self.subTest(quaternion=quaternion):
                state = self.state(1)
                state["arms"]["left"]["ee_pose"]["quaternion_xyzw"] = quaternion
                self.assertEqual(self.result(state)["status"], "home")
        state["arms"]["left"]["ee_pose"]["quaternion_xyzw"] = [0, 0, 1, 0]
        self.assertEqual(self.result(state)["arms"]["left"]["rotation_error_deg"], 180)

    def test_exact_inclusive_boundaries_and_small_real_excess(self):
        initial = self.state()
        initial["arms"]["left"]["ee_pose"]["position_m"] = [.3, -.3, .6]
        self.initial(initial)
        state = self.state(1)
        state["arms"]["left"]["ee_pose"]["position_m"] = [.45, -.45, .75]
        state["arms"]["left"]["ee_pose"]["quaternion_xyzw"] = self.rotation(20)
        self.assertEqual(self.result(state)["status"], "home")
        state["arms"]["left"]["ee_pose"]["position_m"][0] += 1e-12
        self.assertEqual(self.result(state)["status"], "away")
        state["arms"]["left"]["ee_pose"]["position_m"][0] = .45
        state["arms"]["left"]["ee_pose"]["quaternion_xyzw"] = self.rotation(20 + 1e-10)
        self.assertEqual(self.result(state)["status"], "away")

    def test_joints_gripper_and_other_state_do_not_decide_home(self):
        self.initial()
        state = self.state(1)
        state["arms"]["right"].update(q_rad=[2 * math.pi], gripper={"goal_normalized": 0})
        self.assertEqual(self.result(state)["status"], "home")

    def test_invalid_measured_poses_fail_closed(self):
        self.initial()
        for field, value in [("position_m", [math.nan, 0, 0]), ("position_m", [math.inf, 0, 0]),
                             ("position_m", [True, 0, 0]), ("position_m", [0, 0]),
                             ("quaternion_xyzw", [0, 0, 0, 0]),
                             ("quaternion_xyzw", [0, 0, math.inf, 1]),
                             ("quaternion_xyzw", [0, 0, math.nan, 1]),
                             ("quaternion_xyzw", [0, 0, 1]),
                             ("source_kind", "command"), ("frame_id", "world"),
                             ("tcp_config_id", "grasp-center"), ("frame_id", None)]:
            with self.subTest(field=field, value=value):
                state = self.state(1)
                state["arms"]["left"]["ee_pose"][field] = value
                self.assertEqual(self.result(state)["status"], "unknown")
                self.assert_status("unknown")

    def test_invalid_initial_cannot_be_replaced_by_later_current(self):
        state = self.state()
        state["arms"].pop("right")
        record = self.checker.capture_initial(self.bundle(state))
        self.assertEqual(record["status"], "unknown")
        later = self.bundle(self.state(1))
        self.checker.update_current(later)
        self.checker.capture_initial(later)
        self.assert_status("unknown")
        self.assertEqual(self.checker.reference_record(), record)

    def test_artifact_missing_corrupt_non_actor_or_wrong_metadata(self):
        self.initial()
        for scope in ("internal", "evaluator", "observer"):
            with self.subTest(scope=scope):
                bundle = self.bundle(self.state(1))
                bundle["robot_state_ref"] = self.store.put_artifact(json.dumps(self.state(1)).encode(), "robot_state", "application/json", scope)
                self.assertEqual(self.checker.update_current(bundle)["status"], "unknown")
        for data, kind, mime in [(b"not json", "robot_state", "application/json"),
                                 (b"[]", "robot_state", "application/json"),
                                 (json.dumps(self.state(1)).encode(), "evaluation", "application/json"),
                                 (json.dumps(self.state(1)).encode(), "robot_state", "text/plain")]:
            bundle = self.bundle(self.state(1))
            bundle["robot_state_ref"] = self.store.put_artifact(data, kind, mime)
            self.assertEqual(self.checker.update_current(bundle)["status"], "unknown")
        bundle = self.bundle(self.state(2))
        path = self.store.artifact(bundle["robot_state_ref"])[1]
        path.write_bytes(b"corrupt")
        self.assertEqual(self.checker.update_current(bundle)["status"], "unknown")
        path.unlink()
        self.assertEqual(self.checker.update_current(bundle)["status"], "unknown")
        bundle["robot_state_ref"] = "artifact:missing"
        self.assertEqual(self.checker.update_current(bundle)["status"], "unknown")

    def test_only_actor_scope_is_requested(self):
        bundle = self.bundle()
        with patch.object(self.store, "resolve_artifact", wraps=self.store.resolve_artifact) as resolve:
            self.checker.capture_initial(bundle)
        resolve.assert_called_once_with(bundle["robot_state_ref"], allowed_scopes=("actor",))

    def test_missing_and_mismatched_metadata(self):
        self.initial()
        for key, value in [("frame_id", "world"), ("tcp_config_id", "other"),
                           ("sim_step", None), ("step_domain", None)]:
            with self.subTest(state_key=key):
                state = self.state(1)
                state[key] = value
                self.assertEqual(self.result(state)["status"], "unknown")
        for key, value in [("world_kind", "predicted"), ("schema_version", "other"),
                           ("observation_id", None), ("robot_state_ref", None),
                           ("sim_step", -1), ("sim_step", 1.0), ("sim_step", True),
                           ("sim_step", 20), ("step_domain", "native_step"),
                           ("run_id", "other"), ("episode_id", "old"),
                           ("env_instance_id", "other"), ("env_generation", 2)]:
            with self.subTest(bundle_key=key, value=value):
                bundle = self.bundle(self.state(1))
                bundle[key] = value
                self.assertEqual(self.checker.update_current(bundle)["status"], "unknown")
        self.assertEqual(self.checker.update_current(None)["status"], "unknown")

    def test_invalidation_needs_new_observation_but_not_step_advance(self):
        initial = self.initial()
        for _ in range(3):
            self.assert_status("home")
        result = self.checker.invalidate_current("action_in_progress")
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "action_in_progress")
        self.assertEqual(self.checker.update_current(initial)["reason"], "observation_invalidated")
        terminal = self.bundle()  # A rejected action can leave the native step unchanged.
        self.assertEqual(self.checker.update_current(terminal)["status"], "home")
        self.assertEqual(self.checker.update_current(terminal)["status"], "home")

    def test_regression_domain_change_and_same_id_conflict_do_not_lower_watermark(self):
        self.initial()
        current = self.bundle(self.state(10))
        self.checker.update_current(current)
        self.checker.invalidate_current("action_in_progress")
        self.assertEqual(self.result(self.state(9))["reason"], "observation_step_regressed")
        self.assertEqual(self.result(self.state(8))["reason"], "observation_step_regressed")
        state = self.state(10)
        state["step_domain"] = "different_domain"
        self.assertEqual(self.result(state)["reason"], "step_domain_changed")
        state = self.state(11)
        changed = self.bundle(state)
        changed["observation_id"] = current["observation_id"]
        self.assertEqual(self.checker.update_current(changed)["reason"], "observation_id_content_changed")
        self.assertEqual(self.result(self.state(10))["status"], "home")

    def test_reset_requires_matching_new_reference_and_rejects_old_episode(self):
        previous = dict(self.identity)
        old = self.initial()
        self.checker.invalidate_current("environment_reset")
        self.identity.update(episode_id="episode-2", env_generation=2)
        new_state = self.state()
        new_state["arms"]["left"]["ee_pose"]["position_m"] = [1, 2, 3]
        first = self.bundle(new_state)
        self.assertEqual(self.checker.update_current(first)["reason"], "initial_identity_mismatch")
        self.checker.capture_initial(first)
        self.assert_status("home")
        self.assertEqual(self.checker.check(previous)["status"], "unknown")
        self.assertEqual(self.checker.capture_initial(old)["status"], "unknown")
        self.assertEqual(self.checker.reference_record()["initial_observation_id"], first["observation_id"])
        self.assertEqual(self.checker.update_current(old)["status"], "unknown")
        self.assertEqual(self.checker.update_current(first)["status"], "home")

    def test_nonfinite_arithmetic_stays_json_serializable(self):
        initial = self.state()
        initial["arms"]["left"]["ee_pose"]["position_m"][0] = 1e308
        self.initial(initial)
        current = self.state(1)
        current["arms"]["left"]["ee_pose"]["position_m"][0] = -1e308
        result = self.result(current)
        self.assertEqual(result["status"], "unknown")
        self.assertEqual(result["reason"], "position_difference_not_finite")
        json.dumps(result, allow_nan=False)
        json.dumps(self.checker.reference_record(), allow_nan=False)
        json.dumps(HOME_CHECK_POLICY, allow_nan=False)



# --- Folded from tests/test_robot_action_reach.py ---

SPLIT = ("approach_tolerance_rad", "roll_tolerance_rad")
DOWN = [1.0, 0.0, 0.0, 0.0]  # TCP +z (approach axis) pointing at world -z


def _so101_profile(ik_service=False):
    from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment
    document = json.loads((golden.REPOSITORY / "configs" / "so101.example.json").read_text())
    if ik_service:
        document["ik_service"] = {"solver": "curobo", "url": "http://127.0.0.1:9"}
    return embodiment_profile(validate_deployment(document, base=golden.REPOSITORY / "configs"))


def _so101_harness(root, *, ik_service=False):
    """``golden.compile_harness("so101")``, optionally with an ``ik_service`` section (never contacted: the tests
    replace ``reach_batch``)."""
    from ni_rua.core.storage.store import Store
    from ni_rua.runtime.harness import Harness
    root = Path(root)
    deployment = golden.so101_deployment(root / "deployment", empty_close_aperture_m=golden.SO101_EMPTY_CLOSE_M)
    if ik_service:
        document = json.loads(deployment.read_text())
        document["ik_service"] = {"solver": "curobo", "url": "http://127.0.0.1:9"}
        deployment.write_text(json.dumps(document, indent=2))
    config = golden.load_config(golden.SO101_CONFIGS["so101_gateway"], overrides={"backend": {"deployment": str(deployment)}})
    robot = golden._so101_robot_class()(config=config, run_dir=root / "robot")
    features = config.features
    harness = Harness(Store(root / "store", dict(golden.IDENTITY)), robot,
                      actor_payload_profile=features.actor_payload_profile, enable_wait=features.enable_wait,
                      observation_profile=features.observation_profile, motion_review=features.motion_review,
                      vision_compare=features.vision_compare, config=config, task_plan_required=True)
    harness.start()
    return harness


def _close(harness):
    harness.store.close()
    close = getattr(harness.robot, "close", None)
    if callable(close):
        close()


def _verdicts(action, unreachable=()):
    """A ``reach_batch`` answer in the backend shape for every stage of the compiled ``action``."""
    results = {}
    for stage in action["stages"]:
        path = stage["action"]["path"]
        ok = stage["id"] not in unreachable
        ik = dict(ok=ok, position_error_m=0.0004 if ok else 0.081, orientation_error_rad=0.02 if ok else 0.9,
                  orientation_checked=True, solver_converged=ok)
        if not ok:
            ik["position_priority"] = dict(quaternion_xyzw=[0.0, 1.0, 0.0, 0.0], position_error_m=0.0009,
                                           orientation_error_rad=0.7)
        entry = dict(ok=ok, ik=ik, tolerances={key: path[key] for key in ("position_tolerance_m",
                                                                           "orientation_tolerance_rad")},
                     reached=dict(position_m=[0.2, 0.0, 0.05], quaternion_xyzw=DOWN, approach_axis=[0.0, 0.0, -1.0]))
        if ok:
            entry.update(q_rad=[0.0, -0.5, 0.6, 1.2, 0.0], joint_travel_rad=[0.0, 0.1, 0.1, 0.2, 0.0])
        results[stage["id"]] = entry
    return dict(status="solved", stage_count=len(results),
                reachable_stage_count=sum(1 for entry in results.values() if entry["ok"]),
                stage_results=results, physics_advanced=False, executed_native_steps=0)


# ---- schema gating ----------------------------------------------------------------------------------------------
class ReachSchemaGatingTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.x5, cls.so101, cls.so101_ik = build(None), build(_so101_profile()), build(_so101_profile(ik_service=True))

    def text(self, contract):
        return json.dumps(contract.tool)

    def test_x5_publishes_neither_reach_nor_split_bounds(self):
        properties = self.x5.tool["inputSchema"]["properties"]
        self.assertNotIn("reach", properties["mode"]["enum"])
        for field in SPLIT:
            self.assertNotIn(field, properties)
            self.assertNotIn(field, self.x5.stage_schemas["move"]["properties"])
            self.assertNotIn(field, self.text(self.x5))
        self.assertNotIn("mode=reach", self.text(self.x5))
        self.assertNotIn("Preview/direct/reach", self.text(self.x5))
        self.assertNotIn("reach", [example.get("mode") for example in self.x5.tool["inputSchema"]["examples"]])
        # The default-robot module views are the X5 rendering.
        self.assertEqual(robot_action_schema.ROBOT_ACTION_TOOL, self.x5.tool)

    def test_so101_publishes_reach_and_split_bounds_only_with_their_tags(self):
        self.assertIn("reach", _so101_profile().supports)
        self.assertNotIn("approach_roll", _so101_profile().supports)
        for contract, split in ((self.so101, False), (self.so101_ik, True)):
            properties = contract.tool["inputSchema"]["properties"]
            self.assertEqual(properties["mode"]["enum"][-1], "reach")
            self.assertIn("mode=reach takes the current observation_id and 1..16 move stages", contract.tool["description"])
            self.assertIn("reach solves IK of each move stage pose", properties["mode"]["description"])
            self.assertIn("preview/preflight/execute_if_clean/direct/reach", properties["observation_id"]["description"])
            self.assertIn("mode=reach accepts move stages only", properties["stages"]["description"])
            self.assertTrue(properties["timeout_s"]["description"].startswith("Preview/direct/reach only:"))
            for field in SPLIT:
                with self.subTest(split=split, field=field):
                    self.assertEqual(field in properties, split)
                    self.assertEqual(field in contract.stage_schemas["move"]["properties"], split)
                    self.assertEqual(field in contract.stage["properties"], split)
                    self.assertEqual(field in self.text(contract), split)
            # A reach example is published and admitted by the schema it documents.
            example = contract.tool["inputSchema"]["examples"][-1]
            self.assertEqual(example["mode"], "reach")
            self.assertEqual(all(field in example["stages"][1] for field in SPLIT), split)
            self.assertTrue(validator_for(contract.tool["inputSchema"]).is_valid(example))
        # The X5 absent-feature sentence names neither tag; SO-101's sentence does not name them either.
        self.assertNotIn("Not available on this robot", self.x5.tool["description"])
        self.assertNotIn("approach", self.so101.tool["description"].split("Not available on this robot")[1])

    def test_split_fields_carry_unit_range_default_example_and_receipt_fields(self):
        stage = self.so101_ik.stage_schemas["move"]["properties"]
        batch = self.so101_ik.tool["inputSchema"]["properties"]
        for field in SPLIT:
            for where, schema in (("stage", stage[field]), ("batch", batch[field])):
                with self.subTest(field=field, where=where):
                    self.assertEqual((schema["type"], schema["exclusiveMinimum"], schema["maximum"]),
                                     ("number", 0, math.pi))
                    text = schema["description"]
                    for phrase in ("radians", "(0, pi]", "3.14159", "default unset" if where == "batch" else "Default unset"):
                        self.assertIn(phrase, text)
        approach, roll = stage["approach_tolerance_rad"]["description"], stage["roll_tolerance_rad"]["description"]
        self.assertIn("gripper_frame_link local +z, outward along the jaws", approach)
        self.assertIn("approach_tolerance_rad=0.35 (accepts up to ~20 deg tilt) and roll_tolerance_rad=0.1", approach)
        for text in (approach, roll):
            self.assertIn("unset component uses the stage's resolved orientation_tolerance_rad", text)
            self.assertIn("approach_error_rad and roll_error_rad", text)
        self.assertIn("jaw-direction error seen along the approach", roll)
        self.assertIn("approach_tolerance_rad/roll_tolerance_rad bound the approach-axis tilt",
                      self.so101_ik.tool["description"])
        # Every move-stage bound is enforced by the schema at the limit and just beyond it.
        validator = validator_for(self.so101_ik.stage_schemas["move"])
        base = dict(id="m", kind="move", arm="right", position_m=[0.2, 0.0, 0.05], quaternion_xyzw=DOWN)
        for field in SPLIT:
            self.assertTrue(validator.is_valid(dict(base, **{field: math.pi})))
            self.assertTrue(validator.is_valid(dict(base, **{field: 1e-6})))
            self.assertFalse(validator.is_valid(dict(base, **{field: 0})))
            self.assertFalse(validator.is_valid(dict(base, **{field: math.pi + 1e-6})))


# ---- effect and output contract ---------------------------------------------------------------------------------
class ReachEffectAndContractTests(unittest.TestCase):
    def test_reach_is_a_read(self):
        self.assertEqual(action_effect({"mode": "reach"}), "read")
        self.assertEqual(action_effect({"mode": "preview"}), "plan")
        self.assertEqual(action_effect({"mode": "inspect"}), "read")
        self.assertEqual(action_effect({"mode": "execute"}), "physical")

    def test_data_contract_admits_reach_verdicts_and_inspect_results(self):
        validator = validator_for(output_schemas("robot_action")["data"])
        action = dict(stages=[dict(id="a", action=dict(path=dict(position_tolerance_m=0.01, orientation_tolerance_rad=0.15))),
                              dict(id="b", action=dict(path=dict(position_tolerance_m=0.01, orientation_tolerance_rad=0.15)))])
        reach = dict(_verdicts(action, unreachable={"b"}), observation_id="obs-1")
        self.assertEqual(list(validator.iter_errors(reach)), [])
        # Existing inspect results stay valid: display-only (null) and the missing-feedback record.
        self.assertTrue(validator.is_valid(None))
        self.assertTrue(validator.is_valid(dict(action_id="a1", available=False, trajectory_feedback={})))
        # A reach result without its verdicts, or with a malformed stage, is a contract defect.
        for broken in ({key: value for key, value in reach.items() if key != "stage_results"},
                       dict(reach, stage_results=dict(a=dict(ik={}))),
                       dict(reach, stage_results=dict(a=dict(reach["stage_results"]["a"], ok="yes")))):
            self.assertFalse(validator.is_valid(broken))


# ---- dispatch -------------------------------------------------------------------------------------------------
class _HarnessCase(unittest.TestCase):
    ik_service = False

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="ni-rua-reach-")
        cls.harness = _so101_harness(cls.directory.name, ik_service=cls.ik_service)
        cls.arm = cls.harness.embodiment.arms[0]

    @classmethod
    def tearDownClass(cls):
        _close(cls.harness)
        cls.directory.cleanup()

    def setUp(self):
        self.calls = []

    def fake_reach(self, unreachable=(), result=None):
        def reach_batch(action):
            self.calls.append(deepcopy(action))
            return deepcopy(result) if result is not None else _verdicts(action, unreachable)
        return mock.patch.object(self.harness.robot, "reach_batch", reach_batch)

    def move(self, identifier, **fields):
        return {**dict(id=identifier, kind="move", arm=self.arm, position_m=[0.2, 0.0, 0.05], quaternion_xyzw=DOWN),
                **fields}

    def request(self, *stages, **batch):
        return dict(mode="reach", observation_id=self.harness.current_observation()["observation_id"],
                    stages=list(stages) or [self.move("c1")], **batch)

    def invoke(self, args, role="actor", call="reach"):
        return self.harness.invoke("robot_action", deepcopy(args), dict(tool_call_id=call), role=role)

    def refused(self, args, code, field_path=None):
        """``args`` is refused with ``code`` before the backend is asked; returns the model-facing error payload."""
        with self.fake_reach():
            with self.assertRaises(ValueError) as caught:
                self.invoke(args, call="refused-" + code)
        self.assertEqual(str(caught.exception), code)
        self.assertEqual(self.calls, [])
        payload = error_payload(caught.exception, tool="robot_action", arguments=args)
        self.assertEqual(payload["code"], code)
        if field_path is not None:
            self.assertEqual(payload["field_path"], field_path)
        return payload


class ReachDispatchTests(_HarnessCase):
    def test_reach_answers_every_stage_read_only_through_both_roles(self):
        harness = self.harness
        before = harness.current_observation()["observation_id"]
        plans_before = len(harness.store.records("unified_action_plan"))
        requests_before = len(harness.store.records("action_request"))
        args = self.request(self.move("near"), self.move("far", position_m=[0.38, 0.2, 0.3], tolerance_preset="coarse"),
                            position_tolerance_m=0.008, timeout_s=20)
        for role in ("actor", "program"):
            with self.subTest(role=role), self.fake_reach(unreachable={"far"}):
                response = self.invoke(args, role=role, call="reach-" + role)
                self.assertIsNone(response.contract_error)
                self.assertEqual((response.kind, response.success), ("data", True))
                data = response.data
                self.assertEqual((data["status"], data["stage_count"], data["reachable_stage_count"]), ("solved", 2, 1))
                self.assertEqual(data["observation_id"], before)
                self.assertEqual((data["physics_advanced"], data["executed_native_steps"]), (False, 0))
                self.assertEqual(set(data["stage_results"]), {"near", "far"})
                self.assertIn("position_priority", data["stage_results"]["far"]["ik"])
                # The model sees the same verdicts in the first display part.
                self.assertEqual(json.loads(json.dumps(response.content[0].value)), data)
        # The backend got the preview compilation: move stages only, resolved tolerances, the current observation.
        self.assertEqual(len(self.calls), 2)
        action = self.calls[0]
        self.assertEqual((action["type"], action["based_on_observation_id"], action["timeout_s"]), ("BatchAction", before, 20))
        self.assertEqual([stage["kind"] for stage in action["stages"]], ["move", "move"])
        near, far = (stage["action"]["path"] for stage in action["stages"])
        self.assertEqual(near["position_tolerance_m"], 0.008)
        self.assertEqual(far["tolerance_preset"], "coarse")
        self.assertEqual(far["position_tolerance_m"], harness.embodiment.tolerance_presets["coarse"].position_m)
        self.assertEqual(near["waypoints"][0][self.arm]["position_m"], [0.2, 0.0, 0.05])
        # Nothing was planned, recorded as an action, executed or invalidated.
        self.assertEqual(getattr(harness, "unified_action_plans", {}), {})
        self.assertEqual(len(harness.store.records("unified_action_plan")), plans_before)
        self.assertEqual(len(harness.store.records("action_request")), requests_before)
        self.assertEqual(harness.current_observation()["observation_id"], before)
        self.assertIsNone(harness.active_action)
        self.assertEqual(harness.tool_registry.spec("robot_action", role="actor").effect_for(args), "read")

    def test_non_move_stages_are_refused_with_the_offending_stage(self):
        gripper = dict(id="open", kind="gripper", arm=self.arm, goal_normalized=1)
        observe = dict(id="look", kind="observe", reason="inspect the cube")
        for index, stage in ((1, gripper), (0, observe)):
            stages = [self.move("c1"), stage] if index else [stage, self.move("c1")]
            with self.subTest(kind=stage["kind"]):
                payload = self.refused(self.request(*stages), "robot_action_reach_accepts_move_poses_only",
                                       f"/stages/{index}/kind")
                self.assertIn("kind=move", payload["suggestion"])

    def test_extra_stage_fields_are_refused_not_ignored(self):
        for field, value in (("speed_scale", 0.5), ("gripper_goal_normalized", 0.0), ("stall_guard_preset", "off"),
                             ("min_aperture_m", 0.01), ("depends_on", ["c0"])):
            with self.subTest(field=field):
                stages = [self.move("c0"), self.move("c1", **{field: value})]
                self.refused(self.request(*stages), "robot_action_reach_accepts_move_poses_only", f"/stages/1/{field}")

    def test_command_and_plan_fields_are_refused_before_they_act(self):
        task_plan = getattr(self.harness, "task_planning", None)
        before = task_plan.current(self.harness.identity) if task_plan is not None else None
        for field, value in (("plan_update", {"plan": "grasp the cube"}), ("auto_execute_if_clean", False),
                             ("action_id", "a-reach"), ("source_plan_id", "p1"), ("settle", {}),
                             ("checkpoint_after_gripper", True), ("stall_guard_preset", "off")):
            with self.subTest(field=field):
                payload = self.refused(self.request(**{field: value}), "robot_action_reach_has_unsupported_fields",
                                       "/" + field)
                self.assertIn("observation_id, stages, timeout_s", payload["suggestion"])
        if task_plan is not None:
            self.assertEqual(task_plan.current(self.harness.identity), before)

    def test_reach_needs_the_current_observation_and_an_idle_run(self):
        args = self.request()
        self.refused({key: value for key, value in args.items() if key != "observation_id"},
                     "robot_action_requires_observation_id", "/observation_id")
        self.refused(dict(args, observation_id="obs-stale"), "stale_observation", "/observation_id")
        self.harness.active_action = "a-in-flight"
        try:
            payload = self.refused(args, "robot_action_reach_requires_idle_active_run")
        finally:
            self.harness.active_action = None
        self.assertIn("idle active run", payload["suggestion"])

    def test_backend_without_reach_batch_is_refused(self):
        with mock.patch.object(type(self.harness.robot), "reach_batch", None, create=True):
            self.assertFalse(callable(getattr(self.harness.robot, "reach_batch", None)))
            with self.assertRaises(ValueError) as caught:
                self.invoke(self.request(), call="no-reach")
        self.assertEqual(str(caught.exception), "robot_action_reach_unavailable")
        self.assertIn("Preview the candidate stages", error_payload(caught.exception, tool="robot_action",
                                                                    arguments=self.request())["suggestion"])

    def test_failure_of_the_whole_query_is_a_tool_error_naming_its_reason(self):
        failed = dict(status="failed", reason="ik_service_unreachable", physics_advanced=False, executed_native_steps=0)
        with self.fake_reach(result=failed):
            with self.assertRaises(ValueError) as caught:
                self.invoke(self.request(), call="ik-down")
        self.assertEqual(len(self.calls), 1)
        payload = error_payload(caught.exception, tool="robot_action", arguments=self.request())
        self.assertEqual(payload["code"], "ik_service_unreachable")
        self.assertIn("tell the operator", payload["suggestion"])
        self.assertIn("Retry at most once", payload["suggestion"])
        with self.fake_reach(result=dict(status="failed", physics_advanced=False, executed_native_steps=0)):
            with self.assertRaises(ValueError) as caught:
                self.invoke(self.request(), call="no-reason")
        self.assertEqual(str(caught.exception), "robot_action_reach_failed")

    def test_split_bounds_need_the_approach_roll_tag(self):
        from ni_rua.runtime.actions.unified_actions import compile_request
        stage_level = self.request(self.move("c1", approach_tolerance_rad=0.35))
        batch_level = self.request(roll_tolerance_rad=0.1)
        for args, code in ((stage_level, "unsupported_robot_action_stage_fields"),
                           (batch_level, "unsupported_capability")):
            with self.subTest(code=code):
                with self.fake_reach(), self.assertRaises(ValueError):
                    self.invoke(args, call="split-without-tag-" + code)  # the run's schema has no such field
                self.assertEqual(self.calls, [])
                with self.assertRaisesRegex(ValueError, code):  # nor does the compiler admit it
                    compile_request(self.harness, args)


class ReachRealWorkerTests(_HarnessCase):
    """One query of the real ``SO101Adapter.reach_batch`` (packaged local IK) over the worker's JSON round trip."""

    def test_reachable_and_unreachable_poses_are_both_answered(self):
        from ni_rua.robot.backends.so101.ik_plan import matrix_to_quaternion
        adapter = self.harness.robot.adapter
        T = adapter.tcp(adapter.read()["q_rad"])
        home = dict(position_m=[float(v) for v in T[:3, 3]], quaternion_xyzw=matrix_to_quaternion(T[:3, :3]))

        def reach_batch(action):
            self.harness.robot.validate_action(action)
            return adapter.reach_batch(json.loads(json.dumps(action)))
        args = self.request(dict(id="home", kind="move", arm=self.arm, **home),
                            dict(id="far", kind="move", arm=self.arm, position_m=[0.44, 0.3, 0.3],
                                 quaternion_xyzw=home["quaternion_xyzw"]))
        with mock.patch.object(self.harness.robot, "reach_batch", reach_batch):
            response = self.invoke(args, call="real-worker")
        self.assertIsNone(response.contract_error)
        data = response.data
        self.assertEqual((data["status"], data["stage_count"], data["reachable_stage_count"]), ("solved", 2, 1))
        self.assertTrue(data["stage_results"]["home"]["ok"])
        self.assertEqual(len(data["stage_results"]["home"]["q_rad"]), 5)
        self.assertFalse(data["stage_results"]["far"]["ok"])
        self.assertNotIn("q_rad", data["stage_results"]["far"])
        # A reach never moves: its tolerances are the IK acceptance bounds only (no arrival-correction bounds).
        for result in data["stage_results"].values():
            self.assertFalse({"arrival_correction_rounds", "arrival_correction_max_offset_rad"} & set(result["tolerances"]))


class SplitBoundsCompileTests(_HarnessCase):
    ik_service = True

    def test_split_bounds_resolve_into_each_compiled_path(self):
        from ni_rua.runtime.actions.unified_actions import compile_request
        self.assertIn("approach_roll", self.harness.embodiment.supports)
        request = dict(observation_id=self.harness.current_observation()["observation_id"],
                       stages=[self.move("tilt", approach_tolerance_rad=0.35),
                               self.move("strict", roll_tolerance_rad=0.05),
                               self.move("free", tolerance_preset="coarse")],
                       roll_tolerance_rad=0.1)
        self.harness.tool_registry.validate("robot_action", deepcopy(request), role="actor")
        action = compile_request(self.harness, deepcopy(request))
        paths = {stage["id"]: stage["action"]["path"] for stage in action["stages"]}
        self.assertEqual((paths["tilt"]["approach_tolerance_rad"], paths["tilt"]["roll_tolerance_rad"]), (0.35, 0.1))
        self.assertEqual(paths["strict"]["roll_tolerance_rad"], 0.05)
        self.assertNotIn("approach_tolerance_rad", paths["strict"])
        self.assertEqual(paths["free"]["roll_tolerance_rad"], 0.1)
        self.assertEqual(action["roll_tolerance_rad"], 0.1)
        # Out of range is refused by the schema and the compiler alike.
        for value in (0, math.pi + 1e-3):
            bad = dict(request, stages=[self.move("tilt", approach_tolerance_rad=value)])
            with self.assertRaises(ValueError):
                self.harness.tool_registry.validate("robot_action", deepcopy(bad), role="actor")
            with self.assertRaises(ValueError):
                compile_request(self.harness, deepcopy(bad))

    def test_reach_sends_the_split_bounds_to_the_backend(self):
        with self.fake_reach():
            response = self.invoke(self.request(self.move("c1", approach_tolerance_rad=0.35, roll_tolerance_rad=0.1)),
                                   call="split-reach")
        self.assertIsNone(response.contract_error)
        path = self.calls[0]["stages"][0]["action"]["path"]
        self.assertEqual((path["approach_tolerance_rad"], path["roll_tolerance_rad"]), (0.35, 0.1))


# ---- X5 -----------------------------------------------------------------------------------------------------------
class X5ReachTests(unittest.TestCase):
    def test_x5_schema_refuses_mode_reach_and_split_fields(self):
        from ni_rua.core.tool_errors import ToolInputError
        with tempfile.TemporaryDirectory(prefix="ni-rua-reach-x5-") as directory:
            harness = golden.compile_harness("x5", directory)
            try:
                observation = harness.current_observation()["observation_id"]
                move = dict(golden.MOVE)
                for args in (dict(mode="reach", observation_id=observation, stages=[move]),
                             dict(observation_id=observation, stages=[dict(move, approach_tolerance_rad=0.3)]),
                             dict(observation_id=observation, stages=[move], roll_tolerance_rad=0.3)):
                    with self.subTest(args=args), self.assertRaises(ToolInputError):
                        harness.tool_registry.validate("robot_action", args, role="actor")
            finally:
                _close(harness)


# ---- continuation and repair --------------------------------------------------------------------------------------
class SplitBoundsContinuationTests(unittest.TestCase):
    def test_stage_overrides_set_and_presets_clear_split_bounds(self):
        from ni_rua.runtime.actions.plan_continuation import apply_stage_override, extract_and_override_stages
        stages = [dict(id="a", kind="move", arm="right", position_m=[0.2, 0, 0.05], quaternion_xyzw=DOWN),
                  dict(id="b", kind="move", arm="right", position_m=[0.2, 0, 0.05], quaternion_xyzw=DOWN,
                       approach_tolerance_rad=0.3, roll_tolerance_rad=0.1)]
        remaining, overridden = extract_and_override_stages(stages, "a", {
            "a": {"approach_tolerance_rad": 0.35, "roll_tolerance_rad": 0.05}, "b": {"tolerance_preset": "coarse"}})
        self.assertEqual(overridden, ["a", "b"])
        self.assertEqual((remaining[0]["approach_tolerance_rad"], remaining[0]["roll_tolerance_rad"]), (0.35, 0.05))
        self.assertNotIn("approach_tolerance_rad", remaining[1])  # a preset sets no split bound
        self.assertNotIn("roll_tolerance_rad", remaining[1])
        kept = apply_stage_override(deepcopy(stages[1]), {"tolerance_preset": "coarse", "roll_tolerance_rad": 0.2})
        self.assertEqual(kept["roll_tolerance_rad"], 0.2)
        self.assertNotIn("approach_tolerance_rad", kept)
        # A compiled (backend continuation) stage receives them in its path.
        compiled = dict(id="b", arm="right", kind="move", action=dict(path=dict(
            waypoints=[{"right": dict(position_m=[0.2, 0, 0.05], quaternion_xyzw=DOWN)}],
            tolerance_preset="normal", position_tolerance_m=0.01, orientation_tolerance_rad=0.15,
            approach_tolerance_rad=0.3)))
        apply_stage_override(compiled, {"roll_tolerance_rad": 0.1})
        self.assertEqual((compiled["action"]["path"]["approach_tolerance_rad"], compiled["action"]["path"]["roll_tolerance_rad"]),
                         (0.3, 0.1))
        self.assertNotIn("roll_tolerance_rad", compiled)
        apply_stage_override(compiled, {"tolerance_preset": "coarse"})
        self.assertNotIn("approach_tolerance_rad", compiled["action"]["path"])
        self.assertNotIn("roll_tolerance_rad", compiled["action"]["path"])

    def record(self):
        identity = dict(golden.IDENTITY)
        stages = [dict(id="a", kind="move", arm="right", position_m=[0.2, 0, 0.05], quaternion_xyzw=DOWN),
                  dict(id="b", kind="move", arm="right", position_m=[0.25, 0, 0.05], quaternion_xyzw=DOWN)]
        source_args = dict(observation_id="obs-1", stages=stages, approach_tolerance_rad=0.35, roll_tolerance_rad=0.1)
        return identity, dict(identity=identity, action=dict(stages=[]), source_args=source_args,
                              source_stages=deepcopy(stages), raw_stages=deepcopy(stages), completed_stage_ids=["a"],
                              summary=dict(completed_stage_ids=["a"], failed_stage_id="b"))

    def test_continuation_inherits_batch_split_bounds(self):
        from ni_rua.runtime.actions.plan_continuation import resolve_continuation_args
        identity, record = self.record()
        harness = SimpleNamespace(identity=identity, unified_action_plans={"p1": record})
        resolved, meta = resolve_continuation_args(harness, dict(observation_id="obs-2", source_plan_id="p1",
                                                                 start_from_stage_id="b"))
        self.assertEqual((resolved["approach_tolerance_rad"], resolved["roll_tolerance_rad"]), (0.35, 0.1))
        self.assertEqual(meta["remaining_stage_ids"], ["b"])
        resolved, _ = resolve_continuation_args(harness, dict(observation_id="obs-2", source_plan_id="p1",
                                                              start_from_stage_id="b", roll_tolerance_rad=0.2))
        self.assertEqual((resolved["approach_tolerance_rad"], resolved["roll_tolerance_rad"]), (0.35, 0.2))

    def test_batch_repair_inherits_split_bounds_with_the_tolerance_group(self):
        from ni_rua.runtime.actions.batch_repair import build_repaired_args
        _, record = self.record()
        lookup = lambda harness, identifier: record
        repaired = build_repaired_args(None, dict(observation_id="obs-2", plan_id="p1"), lookup)
        self.assertEqual([stage["id"] for stage in repaired["stages"]], ["b"])
        self.assertEqual((repaired["approach_tolerance_rad"], repaired["roll_tolerance_rad"]), (0.35, 0.1))
        # An explicit tolerance in the repair request replaces the whole inherited tolerance group.
        repaired = build_repaired_args(None, dict(observation_id="obs-2", plan_id="p1", position_tolerance_m=0.02), lookup)
        self.assertNotIn("approach_tolerance_rad", repaired)



# --- Folded from tests/test_strategy_checkpoint.py ---

class StrategyCheckpointTest(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.identity = dict(run_id="checkpoint", episode_id="episode", env_instance_id="env", env_generation=1)
        self.store = Store(Path(self.tmp.name), self.identity)
        self.addCleanup(self.store.close)
        self.addCleanup(self.tmp.cleanup)

    def action(self, action_id, *, contact="pad_center", kind="BatchAction"):
        return dict(schema_version="0.1", **self.identity, world_kind="observed", type=kind,
            action_id=action_id, based_on_observation_id="obs", stages=[dict(
                id="move", arm="left", kind="move", action=dict(type="move", arm="left",
                    path=dict(contact_region=contact, waypoints=[[0, 0, 0]])))])

    def fail(self, action_id, reason="path_infeasible"):
        action = self.action(action_id)
        self.store.event("action.requested", {"request": action}, {"action_id": action_id})
        self.store.event("action.failed", {"status": "failed", "reason": reason}, {"action_id": action_id})

    def terminal(self, action_id, event_type, *, status="stopped", reason=None):
        action = self.action(action_id)
        self.store.event("action.requested", {"request": action}, {"action_id": action_id})
        payload = {"status": status}
        if reason is not None:
            payload["reason"] = reason
        self.store.event(event_type, payload, {"action_id": action_id})

    def test_repeated_failed_family_returns_short_warning_without_identifiers(self):
        self.fail("a1")
        self.fail("a2")
        checkpoint = adaptive_action_checkpoint(self.store, self.identity)
        self.assertEqual(checkpoint["repeated_failed_mechanism_count"], 2)
        self.assertTrue(checkpoint_triggered(checkpoint))
        text = str(checkpoint)
        self.assertNotIn("a1", text)
        self.assertNotIn("sha256", text)
        self.assertNotIn("episode", text)

    def test_changed_family_and_completed_action_clear_warning(self):
        self.fail("a1")
        self.fail("a2")
        self.assertIsNotNone(adaptive_action_checkpoint(self.store, self.identity))
        action = self.action("a3", contact="fingertip")
        self.store.event("action.requested", {"request": action}, {"action_id": "a3"})
        self.store.event("action.completed", {"status": "completed"}, {"action_id": "a3"})
        self.assertIsNone(adaptive_action_checkpoint(self.store, self.identity))

    def test_pending_plan_or_different_episode_does_not_block_or_warn(self):
        self.fail("a1")
        self.assertIsNone(adaptive_action_checkpoint(self.store, self.identity))
        other = dict(self.identity, episode_id="other")
        self.store.event("action.requested", {"request": dict(self.action("a2"), **other)}, {"action_id": "a2"})
        self.store.event("action.failed", {"status": "failed"}, {"action_id": "a2"})
        self.assertIsNone(adaptive_action_checkpoint(self.store, self.identity))

    def test_observation_checkpoints_are_not_failures(self):
        self.terminal("a1", "action.stopped", reason="observation_checkpoint")
        self.terminal("a2", "action.stopped", reason="observation_checkpoint")
        self.assertIsNone(adaptive_action_checkpoint(self.store, self.identity))

    def test_episode_stop_and_cancellation_are_not_failures(self):
        self.terminal("a1", "action.stopped", reason="episode_ended")
        self.terminal("a2", "action.stopped", reason="episode_ended")
        self.assertIsNone(adaptive_action_checkpoint(self.store, self.identity))
        self.terminal("a3", "action.cancelled", status="cancelled", reason="user_stop")
        self.assertIsNone(adaptive_action_checkpoint(self.store, self.identity))

    def test_abnormal_stop_still_warns(self):
        self.terminal("a1", "action.stopped", reason="tracking_error")
        self.terminal("a2", "action.stopped", reason="tracking_error")
        checkpoint = adaptive_action_checkpoint(self.store, self.identity)
        self.assertEqual(checkpoint["repeated_failed_mechanism_count"], 2)

        self.fail("a3")
        continued = adaptive_action_checkpoint(self.store, self.identity)
        self.assertFalse(checkpoint_triggered(continued))

    def test_checkpoint_reads_latest_tail_after_a_long_history(self):
        for index in range(1001):
            self.terminal(f"old-{index}", "action.completed", status="completed")
        self.fail("latest-a")
        self.fail("latest-b")
        checkpoint = adaptive_action_checkpoint(self.store, self.identity)
        self.assertEqual(checkpoint["repeated_failed_mechanism_count"], 2)



if __name__ == "__main__":
    unittest.main()
