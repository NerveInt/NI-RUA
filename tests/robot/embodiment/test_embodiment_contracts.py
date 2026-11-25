"""Shared validators and preset tables read the run's EmbodimentProfile (design 13, work package WP5).

Findings C1 (grip guard of the run's gripper), C6 (speed range of the embodiment), C13 (arm ids of the embodiment),
C14 / C15 (batch frame, limits, timeouts, stage counts and leaf tolerances from the profile), F2 (no import-time
preset copies; ``features.*`` overrides take effect, decision Q2) and F3 / V-E (the seven unread threshold fields are
removed; old documents still load). X5 byte identity at default thresholds is tests/test_embodiment_x5_golden.py
(the V-E allowance of tests/support/embodiment_golden.py is checked here as well).
"""

NI_RUA_TEST_TIER = "fast"

from copy import deepcopy
import dataclasses
import json
import math
from pathlib import Path
from support.paths import REPO_ROOT
import tempfile
from types import SimpleNamespace
import unittest

from support import embodiment_golden as golden

from ni_rua.core import action_validation, config as config_module
from ni_rua.core.config import (FEATURE_FIELDS, MOTION_THRESHOLD_FIELDS, RETIRED_FEATURE_FIELDS, Config, ConfigError,
                                motion_thresholds)
from ni_rua.core.observation_policy import ACTOR_PROFILE, actor_robot_state
from ni_rua.robot.model import embodiments as e
from ni_rua.robot.motion import batch_contract, grip_guard, motion_safety, motion_speed, motion_tolerance

REPOSITORY = REPO_ROOT
RETIRED = {"soft_position_tolerance_m": 0.01, "grip_thin_max_aperture_m": 0.065, "grip_rigid_max_aperture_m": 0.087,
           "grip_wide_max_aperture_m": 0.0875, "max_min_grip_m": 0.088, "closed_jaw_goal_max": 0.05,
           "checkpoint_empty_jaw_measured_max": 0.03}
RETIRED_BLOAT = {"self_briefing": False, "operation_candidates": False, "lean_projection": False,
                 "lean_projection_k": 1, "lean_projection_block": 8, "lean_compact": False, "lean_request_dump": 0}


def so101_profile(**gripper):
    """The SO-101 profile of configs/so101.example.json (CAD full opening 0.1372 m), with ``gripper`` overrides."""
    from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment
    document = json.loads((REPOSITORY / "configs" / "so101.example.json").read_text())
    document["gripper"].update(gripper)
    return embodiment_profile(validate_deployment(document, base=REPOSITORY / "configs"))


def overridden_x5(**features):
    """(Config, ARX-X5 profile) for a configuration with non-default ``features`` thresholds."""
    config = Config.from_dict({"features": features})
    return config, e.arx_x5_profile(thresholds=config.features)


class NoImportTimeCopiesTests(unittest.TestCase):
    """F2: the shared modules hold no preset table built at import; old names are views of the default robot."""

    def test_modules_define_no_preset_tables_or_threshold_snapshots(self):
        for module, names in ((grip_guard, ("GRIP_GUARD_PRESETS", "DEFAULT_MAX_APERTURE_M",
                                            "DEFAULT_EMPTY_CLOSE_APERTURE_M")),
                              (motion_tolerance, ("TOLERANCE_PRESETS", "STALL_GUARD_PRESETS", "MAX_POSITION_TOLERANCE_M",
                                                  "MAX_ORIENTATION_TOLERANCE_RAD", "MIN_TRACKING_DIVERGENCE_M",
                                                  "MAX_TRACKING_DIVERGENCE_M", "DEFAULT_TOLERANCE_PRESET")),
                              (motion_safety, ("JOINT_TRAVEL_PRESETS", "MIN_TRAVEL_RAD", "MIN_EXCESS_RAD")),
                              (action_validation, ("DEFAULT_ARMS",)), (batch_contract, ("ARMS",))):
            with self.subTest(module=module.__name__):
                self.assertNotIn("_BASE_THRESHOLDS", vars(module))
                for name in names:
                    self.assertNotIn(name, vars(module), name)
                    getattr(module, name)  # still importable for old callers
                with self.assertRaises(AttributeError):
                    getattr(module, "NO_SUCH_NAME")
        # One definition of the shared tables and of the free-orientation threshold.
        self.assertIs(motion_speed.ARM_PRESETS, e.SPEED_PRESETS)
        self.assertIs(motion_speed.GRIPPER_SPEED_PRESETS, e.GRIPPER_SPEED_PRESETS)
        self.assertIs(motion_tolerance.ORIENTATION_FREE_TOLERANCE_RAD, e.ORIENTATION_FREE_TOLERANCE_RAD)

    def test_old_names_are_the_default_robot_profile(self):
        profile = e.ARX_X5_PROFILE
        self.assertEqual(grip_guard.DEFAULT_MAX_APERTURE_M, profile.gripper.full_open_aperture_m)
        self.assertEqual(grip_guard.DEFAULT_EMPTY_CLOSE_APERTURE_M, profile.gripper.empty_close_aperture_m)
        self.assertEqual(grip_guard.GRIP_GUARD_PRESETS, grip_guard.grip_guard_presets(profile))
        self.assertEqual(motion_tolerance.TOLERANCE_PRESETS, motion_tolerance.tolerance_preset_table(profile))
        self.assertEqual(motion_tolerance.STALL_GUARD_PRESETS, dict(profile.stall_guard_presets_m))
        self.assertEqual(batch_contract.ARMS, profile.arms)
        self.assertEqual(action_validation.DEFAULT_ARMS, profile.arms)
        self.assertEqual(motion_safety.JOINT_TRAVEL_PRESETS, motion_safety.joint_travel_presets(None))
        self.assertEqual(motion_safety.MIN_TRAVEL_RAD, math.pi)


class ThresholdOverridesTakeEffectTests(unittest.TestCase):
    """Decision Q2: a non-default features.* threshold changes the preset, the validators and the evaluation."""

    def test_grip_rigid_min_aperture_changes_preset_validator_and_evaluation(self):
        config, profile = overridden_x5(grip_rigid_min_aperture_m=0.004)
        self.assertEqual(config.features.grip_rigid_min_aperture_m, 0.004)
        self.assertEqual(profile.gripper.preset_margins_m["rigid_object"], (0.004, None))
        self.assertEqual(grip_guard.preset_aperture_bounds("rigid_object"), (0.002, None))
        self.assertEqual(grip_guard.preset_aperture_bounds("rigid_object", profile=profile), (0.004, None))
        guard = {"grip_guard_preset": "rigid_object"}
        self.assertEqual(grip_guard.validate_grip_guard_fields(guard)["min_aperture_m"], 0.002)
        self.assertEqual(grip_guard.validate_grip_guard_fields(guard, profile=profile)["min_aperture_m"], 0.004)
        facts = {"measured_aperture_m": 0.003}
        self.assertFalse(grip_guard.evaluate_grip_guard(facts, guard)["triggered"])
        evaluation = grip_guard.evaluate_grip_guard(facts, guard, profile=profile)
        self.assertEqual((evaluation["triggered"], evaluation["reason"], evaluation["min_aperture_m"]),
                         (True, "aperture_below_min_m", 0.004))
        self.assertIn("min_aperture_m=0.0040m", evaluation["message"])
        # The batch-level rule shared by every executor resolves with the same profile.
        resolved = grip_guard.stage_grip_guard({}, guard, 0.0, profile=profile)
        self.assertEqual(resolved["min_aperture_m"], 0.004)

    def test_full_opening_override_moves_the_explicit_bound(self):
        _, profile = overridden_x5(max_gripper_aperture_m=0.09)
        self.assertEqual(profile.gripper.full_open_aperture_m, 0.09)
        with self.assertRaisesRegex(ValueError, "invalid_max_aperture_m"):
            grip_guard.validate_grip_guard_fields({"max_aperture_m": 0.089})
        self.assertEqual(grip_guard.validate_grip_guard_fields({"max_aperture_m": 0.089}, profile=profile)
                         ["max_aperture_m"], 0.089)
        # The ARX-X5 worker maps normalized jaw openings with its run threshold (V-D: the one full opening).
        adapter = SimpleNamespace(motion_thresholds=motion_thresholds({"max_gripper_aperture_m": 0.09}))
        self.assertAlmostEqual(grip_guard._aperture_from_jaw(0.5, adapter=adapter), 0.045)
        self.assertAlmostEqual(grip_guard._aperture_from_jaw(0.5), 0.044)

    def test_tolerance_and_stall_overrides(self):
        _, profile = overridden_x5(tight_position_tolerance_m=0.002, max_position_tolerance_m=0.12,
                                   stall_guard_strict_divergence_m=0.012, stall_guard_max_divergence_m=0.2)
        tight = {"tolerance_preset": "tight"}
        self.assertEqual(motion_tolerance.resolve_stage_tolerance(tight)["position_tolerance_m"], 0.003)
        self.assertEqual(motion_tolerance.resolve_stage_tolerance(tight, profile=profile)["position_tolerance_m"], 0.002)
        with self.assertRaisesRegex(ValueError, "invalid_position_tolerance_m"):
            motion_tolerance.validate_tolerance_fields({"position_tolerance_m": 0.11})
        motion_tolerance.validate_tolerance_fields({"position_tolerance_m": 0.11}, profile=profile)
        strict = {"stall_guard_preset": "strict"}
        self.assertEqual(motion_tolerance.resolve_stall_guard(strict)["max_tracking_divergence_m"], 0.01)
        self.assertEqual(motion_tolerance.resolve_stall_guard(strict, profile=profile)["max_tracking_divergence_m"],
                         0.012)
        with self.assertRaisesRegex(ValueError, "invalid_max_tracking_divergence_m"):
            motion_tolerance.validate_stall_guard_fields({"max_tracking_divergence_m": 0.18})
        motion_tolerance.validate_stall_guard_fields({"max_tracking_divergence_m": 0.18}, profile=profile)

    def test_joint_travel_override_reaches_the_worker_admission(self):
        config, _ = overridden_x5(joint_travel_extended_max_rad=6.0)
        self.assertEqual(motion_safety.resolve_joint_travel_thresholds("extended")[0], 1.5 * math.pi)
        self.assertEqual(motion_safety.resolve_joint_travel_thresholds("extended", thresholds=config.features)[0], 6.0)
        names = [f"joint{index}" for index in range(1, 7)]
        start, path = [0.0] * 6, [[5.5, 0, 0, 0, 0, 0], [0.0] * 6]  # 11 rad of travel, all of it backtracking
        kwargs = dict(joint_names=names, joint_types=["revolute"] * 6, joint_travel_preset="extended")
        self.assertEqual(motion_safety.assess_motion(start, path, **kwargs)["status"], "reject")
        assessment = motion_safety.assess_motion(start, path, **kwargs, thresholds=config.features)
        self.assertEqual((assessment["status"], assessment["thresholds"]["min_total_travel_rad"]), ("reject", 6.0))
        quiet = [[2.6, 0, 0, 0, 0, 0], [0.0] * 6]  # 5.2 rad: above the default 1.5 pi, below the configured 6 rad
        self.assertEqual(motion_safety.assess_motion(start, quiet, **kwargs)["status"], "reject")
        self.assertEqual(motion_safety.assess_motion(start, quiet, **kwargs, thresholds=config.features)["status"], "pass")
        # adapter_assessment reads the adapter's run thresholds (the X5 worker receives NI_RUA_MOTION_THRESHOLDS).
        model = {"joints": [{"name": name, "type": "revolute", "limits": {"lower": -10.0, "upper": 10.0}}
                            for name in names]}
        before = {"arms": {"left": {"q_rad": start, "joint_names": names}}}
        for thresholds, status, minimum in ((None, "reject", 1.5 * math.pi), (config.features, "pass", 6.0)):
            adapter = SimpleNamespace(robot_model=model, motion_safety_guard_enabled=True)
            if thresholds is not None:
                adapter.motion_thresholds = motion_thresholds(thresholds)
            result = motion_safety.adapter_assessment(adapter, before, {"left": quiet}, native_steps=2,
                                                      request_remaining=None, joint_travel_preset="extended")
            self.assertEqual((result["status"], result["arms"]["left"]["thresholds"]["min_total_travel_rad"]),
                             (status, minimum))

    def test_run_harness_profile_carries_the_override_into_batch_validation(self):
        """End to end at the shared boundary: the Harness of an X5 run with an override validates a compiled batch
        with ``harness.embodiment`` (the call sites pass it; see the WP5 report for the plumbing diffs)."""
        from ni_rua.core.contracts import validate_action
        from ni_rua.runtime.actions.unified_actions import compile_request
        with tempfile.TemporaryDirectory() as directory, golden.repository_cwd():
            harness = golden.compile_harness("x5", directory)
            try:
                default = harness.embodiment
                request = dict(observation_id=harness.latest["observation_id"], action_id="wp5-q2",
                               stages=[golden.MOVE, golden.JAW])
                action = compile_request(harness, deepcopy(request))
            finally:
                harness.store.close()
        overridden = e.arx_x5_profile(binding=e.get_robodojo_robot(),
                                      thresholds={"max_gripper_aperture_m": 0.09, "grip_rigid_min_aperture_m": 0.004})
        self.assertEqual(dataclasses.replace(overridden, gripper=default.gripper), default)
        batch = dict(action, max_aperture_m=0.089, grip_guard_preset="rigid_object")
        with self.assertRaisesRegex(ValueError, "invalid_max_aperture_m"):
            batch_contract.validate_batch_action(deepcopy(batch), harness.capabilities, profile=default)
        batch_contract.validate_batch_action(deepcopy(batch), harness.capabilities, profile=overridden)
        identity = {key: action[key] for key in ("run_id", "episode_id", "env_instance_id", "env_generation")}
        validate_action(deepcopy(batch), identity, action["based_on_observation_id"], harness.capabilities,
                        profile=overridden)


class EmbodimentGripperTests(unittest.TestCase):
    """C1 / V-D: the shared grip-guard validation uses the run's gripper (zero point and single full opening)."""

    def test_so101_profile_accepts_0_10_and_rejects_0_14(self):
        profile = so101_profile()
        self.assertEqual(profile.gripper.full_open_aperture_m, 0.1372)
        for field in ("min_aperture_m", "max_aperture_m"):
            with self.subTest(field=field):
                self.assertEqual(grip_guard.validate_grip_guard_fields({field: 0.10}, profile=profile)[field], 0.10)
                with self.assertRaisesRegex(ValueError, "invalid_" + field):
                    grip_guard.validate_grip_guard_fields({field: 0.14}, profile=profile)
                with self.assertRaisesRegex(ValueError, "invalid_" + field):  # the default robot: X5 0.088 m
                    grip_guard.validate_grip_guard_fields({field: 0.10})

    def test_presets_are_margins_above_the_profile_zero_point(self):
        measured = so101_profile(empty_close_aperture_m=0.0045)
        self.assertEqual(grip_guard.preset_aperture_bounds("rigid_object", profile=measured), (0.0065, None))
        self.assertEqual(grip_guard.validate_grip_guard_fields({"grip_guard_preset": "non_empty"}, profile=measured),
                         {"grip_guard_preset": "non_empty", "min_aperture_m": 0.006, "max_aperture_m": None})
        # The SO-101 profile offers the four schema presets only; legacy X5 aliases are not presets there.
        with self.assertRaisesRegex(ValueError, "invalid_grip_guard_preset"):
            grip_guard.validate_grip_guard_fields({"grip_guard_preset": "solid"}, profile=measured)
        # An explicit zero point (backend argument) still overrides the profile's.
        self.assertEqual(grip_guard.preset_aperture_bounds("non_empty", profile=measured, empty_close_aperture_m=0.0),
                         (0.0015, None))

    def test_unmeasured_zero_point_resolves_no_preset_and_never_fires_the_slip_recheck(self):
        unmeasured = so101_profile()
        self.assertIsNone(unmeasured.gripper.empty_close_aperture_m)
        with self.assertRaisesRegex(ValueError, "measured empty-close"):
            grip_guard.validate_grip_guard_fields({"grip_guard_preset": "non_empty"}, profile=unmeasured)
        self.assertEqual(grip_guard.preset_aperture_bounds("none", profile=unmeasured), (None, None))
        explicit = grip_guard.validate_grip_guard_fields({"min_aperture_m": 0.01}, profile=unmeasured)
        self.assertEqual(explicit["min_aperture_m"], 0.01)
        armed = {"right": {"grip_guard_preset": None, "min_aperture_m": 0.01, "max_aperture_m": None}}
        facts = {"jaw_delta": -0.5, "measured_aperture_m": 0.0}
        self.assertEqual(grip_guard.after_stage_grip_guard(
            dict(armed), "right", "move", None, None, facts, has_remaining_stages=True, has_arm_remaining=True,
            is_downward_move=False, profile=unmeasured), (None, False))
        evaluation, triggered = grip_guard.after_stage_grip_guard(
            dict(armed), "right", "move", None, None, facts, has_remaining_stages=True, has_arm_remaining=True,
            is_downward_move=False, profile=so101_profile(empty_close_aperture_m=0.0))
        self.assertTrue(triggered)
        self.assertEqual(evaluation["reason"], "aperture_below_min_m")


class EmbodimentSpeedTests(unittest.TestCase):
    """C6: the speed range and presets are the embodiment's (SO-101 [0.25, 1.0])."""

    def test_so101_range_and_presets(self):
        profile = so101_profile()
        move = {"path": {"kind": "backend_free", "waypoints": [{}]}}
        self.assertEqual(motion_speed.resolve_action_speed(dict(move, speed_scale=1.5))["speed_scale"], 1.5)
        with self.assertRaisesRegex(ValueError, r"invalid_speed_scale: expected finite number in \[0.25, 1.0\]"):
            motion_speed.resolve_action_speed(dict(move, speed_scale=1.5), profile=profile)
        with self.assertRaisesRegex(ValueError, "invalid_speed_preset: expected one of slower, slow, careful, normal$"):
            motion_speed.resolve_action_speed(dict(move, speed_preset="fast"), profile=profile)
        self.assertEqual(motion_speed.resolve_action_speed(dict(move, speed_preset="careful"), profile=profile)
                         ["speed_scale"], 0.9)
        with self.assertRaisesRegex(ValueError, r"invalid_default_motion_speed_scale: expected finite number in "
                                                r"\[0.25, 1.0\]"):
            motion_speed.resolve_action_speed(dict(move), 1.25, profile=profile)
        self.assertEqual(motion_speed.resolve_action_speed(dict(move), 1.25)["speed_scale"], 1.25)
        with self.assertRaisesRegex(ValueError, r"\[0.25, 1.0\]"):
            action_validation.validate_action_speed(dict(move, speed_scale=1.01), profile=profile)


class EmbodimentArmsTests(unittest.TestCase):
    """C13: actor_robot_state accepts the embodiment's arm ids, and only those."""

    @staticmethod
    def state(arm):
        return {"arms": {arm: {"q_rad": [0.0] * 5, "ee_pose": {"position_m": [0.1, 0.0, 0.2],
                                                              "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}}},
                "frame_id": "base"}

    def test_arm0_passes_with_its_embodiment_arms(self):
        projected = actor_robot_state(self.state("arm0"), ACTOR_PROFILE, arms=("arm0",))
        self.assertEqual(list(projected["arms"]), ["arm0"])
        with self.assertRaisesRegex(ValueError, "invalid_actor_robot_arms"):
            actor_robot_state(self.state("arm0"), ACTOR_PROFILE)  # default robot: left / right
        with self.assertRaisesRegex(ValueError, "invalid_actor_robot_arms"):
            actor_robot_state(self.state("left"), ACTOR_PROFILE, arms=("arm0",))
        self.assertIn("right", actor_robot_state(self.state("right"), ACTOR_PROFILE)["arms"])
        profile = dataclasses.replace(
            e.ARX_X5_PROFILE, arms=("arm0",), max_simultaneous_arms=1,
            supports=e.ARX_X5_PROFILE.supports - {"dual_arm"},
            arm_joint_names=e.FrozenDict(arm0=("j1",)), jaw_joint_names=e.FrozenDict(arm0=("j7",)),
            wrist_cameras=e.FrozenDict(), home_check=None)
        self.assertEqual(action_validation.capability_arms(None, profile), ("arm0",))
        self.assertEqual(action_validation.capability_arms({"arms": ["right"]}, profile), ("right",))


class EmbodimentBatchLimitsTests(unittest.TestCase):
    """C14 / C15: frame, TCP, limits profiles, deadline maximum and stage limits come from the profile."""

    @classmethod
    def setUpClass(cls):
        from ni_rua.runtime.actions.unified_actions import compile_request
        with tempfile.TemporaryDirectory() as directory, golden.repository_cwd():
            harness = golden.compile_harness("x5", directory)
            try:
                cls.capabilities = deepcopy(harness.capabilities)
                observation = harness.latest["observation_id"]
                cls.action = compile_request(harness, dict(observation_id=observation, action_id="wp5-limits",
                                                           stages=[golden.MOVE, golden.JAW]))
                cls.three = compile_request(harness, dict(
                    observation_id=observation, action_id="wp5-three",
                    stages=[golden.MOVE, dict(golden.JAW, id="j1"), dict(golden.MOVE, id="m2")]))
            finally:
                harness.store.close()

    def test_deadline_and_stage_limits(self):
        base = e.ARX_X5_PROFILE
        batch_contract.validate_batch_action(deepcopy(self.action), self.capabilities)
        short = dataclasses.replace(base, max_timeout_s=100.0, default_action_timeout_s=60.0)
        with self.assertRaisesRegex(ValueError, "invalid_batch_timeout"):
            batch_contract.validate_batch_action(deepcopy(self.action), self.capabilities, profile=short)
        batch_contract.validate_batch_action(dict(deepcopy(self.action), timeout_s=100.0), self.capabilities,
                                             profile=short)
        small = dataclasses.replace(base, stages_per_arm=2, max_stages=4)
        batch_contract.validate_batch_action(deepcopy(self.three), self.capabilities)
        with self.assertRaisesRegex(ValueError, "batch_exceeds_sixteen_stages_per_arm"):
            batch_contract.validate_batch_action(deepcopy(self.three), self.capabilities, profile=small)
        tiny = dataclasses.replace(base, stages_per_arm=1, max_stages=1)
        with self.assertRaisesRegex(ValueError, "batch_requires_one_to_thirty_two_stages"):
            batch_contract.validate_batch_action(deepcopy(self.action), self.capabilities, profile=tiny)

    def test_arx_frame_tcp_and_limits_are_the_profile_identity(self):
        profile = e.ARX_X5_PROFILE
        self.assertEqual((self.action["frame_id"], self.action["tcp_config_id"]),
                         (profile.frame_id, profile.tcp_config_id))
        self.assertIn(self.action["limits_profile"], profile.limits_profiles)
        for key, value in (("frame_id", "so101_base"), ("tcp_config_id", "so101-gripper-frame-v1")):
            with self.assertRaisesRegex(ValueError, "unsupported_batch_frame_or_tcp"):
                batch_contract.validate_batch_action(dict(deepcopy(self.action), **{key: value}), self.capabilities)
        with self.assertRaisesRegex(ValueError, "unsupported_batch_limits"):
            batch_contract.validate_batch_action(dict(deepcopy(self.action), limits_profile="so101-deployment-v1"),
                                                 self.capabilities)

    def test_backend_compiled_leaf_tolerances_use_the_profile(self):
        """A non-ARX backend's leaf without explicit tolerances is checked with the profile default, and explicit
        values with the profile maximum (no separate .005 / .1 / .10 numbers)."""
        from ni_rua.robot.backends.so101 import contract
        profile = so101_profile()
        capabilities = dict(arms=["right"], frame_id=profile.frame_id, tcp_config_id=profile.tcp_config_id,
                            limits_profile=profile.limits_profiles[0], path_kinds=["backend_free", "hold"],
                            max_waypoints=1, max_timeout_s=300.0, dual_arm=False, gripper_after_motion=True,
                            motion_tolerance_defaults={})
        identity = dict(golden.IDENTITY, schema_version="0.1")
        leaf = contract.compile_action("move", dict(arm="right", observation_id="obs", position_m=[0.2, 0.0, 0.1],
                                                    quaternion_xyzw=[0.0, 0.0, 0.0, 1.0], timeout_s=30.0),
                                       {key: identity[key] for key in golden.IDENTITY}, capabilities)
        for key in ("position_tolerance_m", "orientation_tolerance_rad"):
            leaf["path"].pop(key, None)
        batch = dict(schema_version="0.1", **golden.IDENTITY, world_kind="observed", type="BatchAction",
                     action_id="wp5-leaf", based_on_observation_id="obs", mode="absolute", frame_id=profile.frame_id,
                     tcp_config_id=profile.tcp_config_id, limits_profile=profile.limits_profiles[0], timeout_s=30.0,
                     stages=[dict(id="m", arm="right", kind="move", action=leaf)])
        batch_contract.validate_batch_action(deepcopy(batch), capabilities, profile=profile)
        narrow = dataclasses.replace(profile, max_position_tolerance_m=0.008,
                                     tolerance_presets=e.FrozenDict(normal=e.Tolerance(0.005, 0.15)),
                                     default_tolerance=e.Tolerance(0.005, 0.15))
        batch_contract.validate_batch_action(deepcopy(batch), capabilities, profile=narrow)
        wide = deepcopy(batch)
        wide["stages"][0]["action"]["path"]["position_tolerance_m"] = 0.009
        batch_contract.validate_batch_action(deepcopy(wide), capabilities, profile=profile)
        with self.assertRaisesRegex(ValueError, "invalid_position_tolerance_m"):
            batch_contract.validate_batch_action(deepcopy(wide), capabilities, profile=narrow)


class EmbodimentDefaultToleranceTests(unittest.TestCase):
    """The tolerance applied when a stage names none is the profile's default_tolerance (SO-101: deployment)."""

    def test_default_tolerance_and_its_label(self):
        self.assertEqual(motion_tolerance.resolve_stage_tolerance({}), {
            "tolerance_preset": "normal", "position_tolerance_m": 0.01, "orientation_tolerance_rad": 0.15})
        profile = dataclasses.replace(so101_profile(), default_tolerance=e.Tolerance(0.02, 0.2))
        self.assertEqual(motion_tolerance.resolve_stage_tolerance({}, profile=profile), {
            "tolerance_preset": None, "position_tolerance_m": 0.02, "orientation_tolerance_rad": 0.2})
        self.assertEqual(motion_tolerance.resolve_stage_tolerance({"tolerance_preset": "normal"}, profile=profile)
                         ["position_tolerance_m"], 0.01)
        self.assertEqual(motion_tolerance.resolve_stage_tolerance({}, {"position_tolerance_m": 0.004}, profile=profile)
                         ["position_tolerance_m"], 0.004)
        with self.assertRaisesRegex(ValueError, "invalid_tolerance_preset"):  # not an SO-101 preset
            motion_tolerance.validate_tolerance_fields({"tolerance_preset": "transit"}, profile=profile)


class RetiredThresholdFieldsTests(unittest.TestCase):
    """V-E: the seven unread fields are gone from the configuration surface; old documents still load."""

    def test_fields_are_removed(self):
        all_retired = {**RETIRED, **RETIRED_BLOAT}
        self.assertEqual(set(all_retired), set(RETIRED_FEATURE_FIELDS))
        features = Config().to_dict()["features"]
        for name in all_retired:
            with self.subTest(name=name):
                self.assertNotIn(name, features)
                self.assertNotIn(name, MOTION_THRESHOLD_FIELDS)
                self.assertNotIn(name, FEATURE_FIELDS)
                self.assertNotIn(name, motion_thresholds())
                self.assertNotIn(name, motion_thresholds(dict(all_retired)))
                self.assertFalse(any(name in str(value) for value in vars(config_module.FeaturesConfig).values()
                                     if isinstance(value, str)))

    def test_old_documents_load_and_drop_the_fields(self):
        expected = Config.from_dict({}).to_dict()
        self.assertEqual(Config.from_dict({"features": dict(RETIRED, **RETIRED_BLOAT)}).to_dict(), expected)
        # No validation of a field that has no effect (the old range of closed_jaw_goal_max was [0, 1]).
        self.assertEqual(Config.from_dict({"features": {"closed_jaw_goal_max": 7}}).to_dict(), expected)
        self.assertEqual(Config().with_overrides({"features": {"max_min_grip_m": 0.05}}).to_dict(), expected)
        with self.assertRaisesRegex(ConfigError, "unknown_config_keys"):
            Config.from_dict({"features": {"soft_position_tolerance_mm": 0.01}})
        # An archived v1 effective_config.json written at 5fd93c6 (before V-E and v2) loads cleanly via migrate_v1_to_v2:
        # all retired keys are dropped and the 44 motion thresholds move from features to embodiment.thresholds.
        _, recorded = golden.load_fixture("x5", "config")
        archived = recorded["effective_config/ni_rua_benchmark"]
        self.assertTrue(set(RETIRED) <= set(archived["features"]))
        loaded_doc = Config.from_dict(archived).to_dict()
        loaded = loaded_doc["features"]
        threshold_set = set(MOTION_THRESHOLD_FIELDS)
        kept = {
            key: value
            for key, value in archived["features"].items()
            if key not in RETIRED_FEATURE_FIELDS and key not in threshold_set
        }
        self.assertFalse(set(RETIRED_FEATURE_FIELDS) & set(loaded))
        self.assertFalse(threshold_set & set(loaded))
        self.assertEqual(
            loaded_doc["embodiment"]["thresholds"],
            {key: archived["features"][key] for key in MOTION_THRESHOLD_FIELDS if key in archived["features"]},
        )
        self.assertEqual([key for key in loaded if key in kept], list(kept))
        self.assertLessEqual(set(loaded) - set(kept),
                             {"startup_inline_grounding", "startup_grounding_views",
                              "persistent_object_tracking", "multiview_scene3d", "budgeted_action_reply",
                              "reply_same_as", "reply_float32", "reply_static_once",
                              "action_reply_profile", "reply_compact_encoding", "robot_urdf_delivery"})
        # Path settings are resolved on load (the fixture stores them redacted); every other kept value is unchanged.
        self.assertEqual({key: value for key, value in loaded.items() if key in kept and key != "grasp_memory_dir"},
                         {key: value for key, value in kept.items() if key != "grasp_memory_dir"})

    def test_old_files_load(self):
        # The shipped configs no longer carry the retired fields; an older file that still does loads (below).
        for name in ("ni-rua.json", "ni-rua.example.json"):
            shipped = json.loads((REPOSITORY / "configs" / name).read_text())["features"]
            self.assertFalse(set(RETIRED_FEATURE_FIELDS) & set(shipped), name)
            self.assertFalse(set(MOTION_THRESHOLD_FIELDS) & set(shipped), name)
        self.assertNotIn("soft_position_tolerance_m", golden.load_config().to_dict()["features"])
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "old.json"
            path.write_text(json.dumps({"features": dict(RETIRED, default_position_tolerance_m=0.006)}))
            config = golden.load_config(path)
        self.assertEqual(config.embodiment.thresholds["default_position_tolerance_m"], 0.006)
        self.assertEqual(config.features.default_position_tolerance_m, 0.006)
        self.assertNotIn("max_min_grip_m", config.to_dict()["features"])

    def test_grip_margins_are_bounded_by_the_full_opening(self):
        with self.assertRaisesRegex(ConfigError, "config_above_maximum:EmbodimentConfig.thresholds.grip_rigid_min_aperture_m"):
            Config.from_dict({"features": {"grip_rigid_min_aperture_m": 0.0881}})
        with self.assertRaisesRegex(ConfigError, "config_above_maximum:EmbodimentConfig.thresholds.grip_rigid_min_aperture_m"):
            Config.from_dict({"embodiment": {"thresholds": {"grip_rigid_min_aperture_m": 0.0881}}})
        config = Config.from_dict({"embodiment": {"thresholds": {"grip_rigid_min_aperture_m": 0.0881, "max_gripper_aperture_m": 0.1}}})
        self.assertEqual(config.embodiment.thresholds.grip_rigid_min_aperture_m, 0.0881)
        self.assertEqual(config.features.grip_rigid_min_aperture_m, 0.0881)


# WP12: features.vggt_support_z_m defaults to "profile" (one more value in the same config items).
F7_ALLOWANCE = "design-13-F7-support-height-from-profile"
STAGE0B_CONFIG_ALLOWANCE = "stage-0b-ni-rua-main-config"
P2_W2_CONFIG_ALLOWANCE = "phase2-wave2-config-schema-v2-and-bloat-retirement"
P2_MODEL_DEFAULT_ALLOWANCE = "phase2-config-default-model-sol"
URDF_DELIVERY_ALLOWANCE = "catalogue-profiles-robot-urdf-delivery-key"
# The shipped defaults layered_v3 / file in the seven config items that show defaults (tests/support/embodiment_golden.py).
CATALOGUE_DEFAULTS_ALLOWANCE = "catalogue-defaults-layered-v3-file-config"
# The reply-delivery defaults (tests/support/embodiment_golden.py): config keys, catalogue notes, lean_v2 startup.
REPLY_DELIVERY_ALLOWANCE = "reply-delivery-defaults-config"


def golden_catalogue_defaults():
    """The shipped values of the two settings ``golden.RECORDED_CATALOGUE_FEATURES`` pins to the recorded form."""
    from support.reply_defaults import DEFAULT_REPLY_FEATURES
    return {name: DEFAULT_REPLY_FEATURES[name] for name in golden.RECORDED_CATALOGUE_FEATURES}


class ReplyDeliveryDefaultsTests(unittest.TestCase):
    """The reply-delivery defaults at a real Workbench start (fixture robot, FakeAgentSession): what the session is
    handed by default, that the earlier form is restored exactly by configuration, and that every setting on its own
    changes only its own surface."""

    NOTED = dict(reply_static_once=True, action_reply_profile="summary_first", reply_compact_encoding=True,
                 reply_same_as=True)
    REPLY_ONLY = dict(budgeted_action_reply=True, reply_float32=True)

    def start(self, **features):
        from support.prompt_capture import _fixture_robot
        with golden.repository_cwd(), tempfile.TemporaryDirectory(prefix="ni-rua-reply-startup-") as directory:
            root = Path(directory).resolve()
            # the shipped defaults of tool_catalog_profile / robot_urdf_delivery unless ``features`` states them
            config = golden._scenario_config(golden.PROMPT_CAPTURE, {"features": features}, root, recorded_catalogue=False)
            captured = golden._workbench_start(config, _fixture_robot(root / "robot"), root, True)
            redact = captured.pop("redactor")
            started = golden.normalize(redact(golden.normalize(dict(
                instructions=captured["instructions"], tools=captured["tools"],
                features=captured["effective_config"]["features"]))))
            # the files the session was given in its working directory (the tool reference, robot.urdf)
            started["actor_files"] = sorted(path.name for path in (root / "run" / "actor").glob("*"))
            return started

    @staticmethod
    def notes(started, collision=False):
        from ni_rua.core.presentation import reply_budget, reply_compaction, reply_encoding
        fixed = reply_compaction.static_once_plan().note
        if not collision:
            fixed = fixed.replace("collision_scope: " + reply_compaction._COLLISION_NOTE + ". ", "")
        return dict(action_reply_profile=reply_budget.SUMMARY_FIRST_CATALOGUE_NOTE, reply_static_once=fixed,
                    reply_compact_encoding=reply_encoding.COMPACT_ENCODING_CATALOGUE_NOTE,
                    reply_same_as=reply_compaction.SAME_AS_CATALOGUE_NOTE)

    def test_default_start_carries_the_new_form_and_the_legacy_settings_restore_the_recorded_one_exactly(self):
        from support.reply_defaults import DEFAULT_REPLY_FEATURES, LEGACY_REPLY_FEATURES
        # the golden scenarios state the recorded catalogue form (full_v1, URDF inline) and the shipped reply defaults
        default = golden.run_startup("x5", "prompt_capture_fixture")
        reply_defaults = {name: value for name, value in DEFAULT_REPLY_FEATURES.items()
                          if name not in golden.RECORDED_CATALOGUE_FEATURES}
        self.assertEqual({name: default["effective_config"]["features"][name] for name in DEFAULT_REPLY_FEATURES},
                         dict(reply_defaults, **golden.RECORDED_CATALOGUE_FEATURES))
        action = next(tool for tool in default["tools"] if tool["name"] == "robot_action")
        notes = self.notes(default)
        self.assertEqual([action["description"].count(note) for note in notes.values()], [1, 1, 1, 1])
        guide = action["description"].index("\nInput field guide")
        self.assertEqual(action["description"][:guide].endswith("".join(notes.values())), True)
        # Every reply switch off and lean_v1: instructions and catalogue are the recorded ones with the allowances
        # that predate the reply-delivery defaults, byte for byte; the one later allowance that applies to it is
        # "status-tools-loaded-up-front" (robot_action_status and program_status no longer carry deferLoading).
        legacy = self.start(**LEGACY_REPLY_FEATURES)
        self.assertEqual({name: legacy["features"][name] for name in LEGACY_REPLY_FEATURES}, LEGACY_REPLY_FEATURES)
        earlier = [item for item in golden.ALLOWANCES if not item["id"].startswith("reply-delivery-defaults")]
        self.assertEqual(len(golden.ALLOWANCES) - len(earlier), 3)
        with unittest.mock.patch.object(golden, "ALLOWANCES", earlier):
            catalogue = golden.compare("x5", "catalogue", {"actor/prompt_capture_fixture": legacy["tools"]})
            startup = golden.compare("x5", "startup", {"prompt_capture_fixture/instructions": legacy["instructions"]})
            # ... and without the reply-delivery allowances the default form is a change
            changed = golden.compare("x5", "catalogue", {"actor/prompt_capture_fixture": default["tools"]})
        self.assertIn(catalogue["states"]["actor/prompt_capture_fixture"], ("equal", "corrected"))
        self.assertEqual(startup["states"]["prompt_capture_fixture/instructions"], "equal")
        self.assertEqual(changed["states"]["actor/prompt_capture_fixture"], "changed")
        legacy_action = next(tool for tool in legacy["tools"] if tool["name"] == "robot_action")
        for note in self.notes(legacy, collision=True).values():
            self.assertNotIn(note, legacy_action["description"])

    def test_default_start_equals_the_explicit_new_settings_and_the_old_ones_reproduce_the_recorded_form(self):
        """tool_catalog_profile layered_v3 + robot_urdf_delivery file are the defaults: a start without settings is
        the start with them stated, and a start with full_v1 + inline is the recorded prose catalogue and inline URDF
        byte for byte (the legacy settings, whose catalogue / startup comparison with the goldens is above)."""
        from ni_rua.runtime.tools.catalog_guide import REFERENCE_NAME
        from support.reply_defaults import DEFAULT_REPLY_FEATURES, LEGACY_REPLY_FEATURES
        default, explicit = self.start(), self.start(**DEFAULT_REPLY_FEATURES)
        self.assertEqual(default, explicit)
        self.assertEqual({name: default["features"][name] for name in DEFAULT_REPLY_FEATURES}, DEFAULT_REPLY_FEATURES)
        legacy = self.start(**LEGACY_REPLY_FEATURES)
        # layered_v3: typed field guide and the reference file; every tool loaded; nothing else of the catalogue changes
        self.assertEqual([tool["name"] for tool in default["tools"]], [tool["name"] for tool in legacy["tools"]])
        self.assertEqual(default["actor_files"], sorted([REFERENCE_NAME, "robot.urdf"]))
        self.assertEqual(legacy["actor_files"], [])
        # layered_v3 publishes no additional tool for on-demand discovery: deferLoading is what full_v1 has
        self.assertEqual([tool.get("deferLoading") for tool in default["tools"]],
                         [tool.get("deferLoading") for tool in legacy["tools"]])
        for tool in default["tools"]:
            old = next(item for item in legacy["tools"] if item["name"] == tool["name"])
            self.assertEqual(tool["inputSchema"], old["inputSchema"], tool["name"])
            if tool["inputSchema"]["properties"]:   # the field guide is stated in the other (typed) form
                self.assertNotEqual(tool["description"], old["description"], tool["name"])
        self.assertLess(sum(len(tool["description"]) for tool in default["tools"]),
                        sum(len(tool["description"]) for tool in legacy["tools"]))
        # the URDF text is the file robot.urdf instead of a section of the startup instructions
        self.assertEqual((legacy["instructions"].count("Robot URDF\n"), default["instructions"].count("Robot URDF\n")), (1, 0))
        self.assertLess(len(default["instructions"]), len(legacy["instructions"]))

    def test_so101_starts_under_the_defaults_and_publishes_the_compact_text_without_a_reference_file(self):
        """``catalog_guide.EVIDENCE_ROBOT`` is the ARX-X5: any other robot gets the ``compact_v1`` text under a layered
        profile and no reference document (nothing for the descriptions to name); its URDF is delivered as the file."""
        from ni_rua.apps.gateway import workbench as workbench_module
        from ni_rua.runtime.tools.catalog_guide import REFERENCE_NAME
        with golden.repository_cwd(), tempfile.TemporaryDirectory(prefix="ni-rua-so101-defaults-") as directory:
            root = Path(directory).resolve()
            deployment = golden.so101_deployment(root / "deployment")
            overrides = {"backend": {"deployment": str(deployment)}}
            started = {}
            for name, features in (("default", {}), ("compact_v1", {"tool_catalog_profile": "compact_v1"}),
                                   ("recorded", golden.RECORDED_CATALOGUE_FEATURES)):
                where = root / name
                where.mkdir()
                config = golden._scenario_config("configs/so101_gateway.json", {**overrides, "features": features},
                                                 where, recorded_catalogue=False)
                robot = golden._so101_robot_class()(config=config, run_dir=where / "run" / "private" / "robot")
                captured = golden._workbench_start(config, robot, where, False)
                captured.pop("redactor")
                started[name] = dict(captured, actor_files=sorted(path.name for path in (where / "run" / "actor").glob("*")))
        self.assertEqual({key: started["default"]["effective_config"]["features"][key]
                          for key in golden.RECORDED_CATALOGUE_FEATURES}, golden_catalogue_defaults())
        # the compact_v1 text, with the field lines two loaded tools share stated once (every layered profile does that)
        from ni_rua.runtime.tools.catalog_guide import share_lines
        self.assertEqual(json.dumps(started["default"]["tools"]), json.dumps(share_lines(started["compact_v1"]["tools"])))
        self.assertNotEqual(json.dumps(started["default"]["tools"]), json.dumps(started["recorded"]["tools"]))
        self.assertNotIn(REFERENCE_NAME, started["default"]["actor_files"])
        self.assertEqual(started["recorded"]["actor_files"], [])
        self.assertIn("Robot URDF\n", started["recorded"]["instructions"])
        self.assertEqual(("Robot URDF\n" in started["default"]["instructions"], "robot.urdf" in started["default"]["actor_files"]),
                         (False, True))

    def test_catalogue_and_urdf_settings_each_change_only_their_own_surface(self):
        from ni_rua.runtime.tools.catalog_guide import REFERENCE_NAME
        from support.reply_defaults import LEGACY_REPLY_FEATURES
        off = self.start(**LEGACY_REPLY_FEATURES)
        layered = self.start(**dict(LEGACY_REPLY_FEATURES, tool_catalog_profile="layered_v3"))
        self.assertEqual(layered["instructions"], off["instructions"])
        self.assertEqual(layered["actor_files"], [REFERENCE_NAME])
        self.assertNotEqual(layered["tools"], off["tools"])
        urdf_file = self.start(**dict(LEGACY_REPLY_FEATURES, robot_urdf_delivery="file"))
        self.assertEqual(urdf_file["tools"], off["tools"])
        self.assertEqual(urdf_file["actor_files"], ["robot.urdf"])
        self.assertNotEqual(urdf_file["instructions"], off["instructions"])
        # every other setting stays what the legacy form states
        for started in (layered, urdf_file):
            self.assertEqual({name: started["features"][name] for name in LEGACY_REPLY_FEATURES
                              if name not in ("tool_catalog_profile", "robot_urdf_delivery")},
                             {name: value for name, value in LEGACY_REPLY_FEATURES.items()
                              if name not in ("tool_catalog_profile", "robot_urdf_delivery")})

    def test_each_setting_alone_and_every_group_starts_and_changes_only_its_own_surface(self):
        from support.reply_defaults import DEFAULT_REPLY_FEATURES, LEGACY_REPLY_FEATURES
        off = self.start(**LEGACY_REPLY_FEATURES)
        tools = {tool["name"]: tool for tool in off["tools"]}
        notes = self.notes(off)
        settings = {**self.REPLY_ONLY, **self.NOTED}
        cases = [*({name: value} for name, value in settings.items()), self.REPLY_ONLY, self.NOTED, settings,
                 dict(settings, actor_payload_profile="lean_v2"), dict(tool_catalog_profile="guide_only_v1")]
        self.assertEqual(dict(settings, actor_payload_profile="lean_v2", **golden_catalogue_defaults()), DEFAULT_REPLY_FEATURES)
        for enabled in cases:
            with self.subTest(enabled=enabled):
                started = self.start(**{**LEGACY_REPLY_FEATURES, **enabled})
                self.assertEqual({name: started["features"][name] for name in enabled}, enabled)
                self.assertEqual([tool["name"] for tool in started["tools"]], list(tools))
                changed = {tool["name"] for tool in started["tools"] if tool != tools[tool["name"]]}
                noted = [name for name in self.NOTED if name in enabled]
                action = next(tool for tool in started["tools"] if tool["name"] == "robot_action")
                # No reply setting touches the instructions (lean_v2 prints the capability floats in them with six
                # decimals, which changes nothing for this fixture robot; the X5 goldens record the two numbers it
                # shortens). tool_catalog_profile is applied by the Codex session, not in the registry catalogue.
                self.assertEqual(started["instructions"], off["instructions"])
                self.assertEqual(changed, {"robot_action"} if noted else set())
                text = action["description"]
                for name in noted:
                    self.assertEqual(text.count(notes[name]), 1)
                    text = text.replace(notes[name], "", 1)
                self.assertEqual(dict(action, description=text), tools["robot_action"])
                for name, note in notes.items():
                    self.assertEqual(note in action["description"], name in enabled)


class V_E_AllowanceTests(unittest.TestCase):
    """The WP5 golden allowance is exact: its declared sites, old values and count match the recorded fixture, and
    it is required (a tree that kept the fields would differ from the expectation)."""

    @classmethod
    def setUpClass(cls):
        cls.allowance = next(item for item in golden.ALLOWANCES if item["id"] == "design-13-V-E-unread-threshold-fields")
        _, cls.recorded = golden.load_fixture("x5", "config")

    def test_sites_old_values_and_count(self):
        allowance = golden.check_allowance(self.allowance)
        self.assertEqual((allowance["package"], allowance["embodiment"], allowance["family"]), ("WP5", "x5", "config"))
        self.assertEqual(allowance["occurrences"], 147)
        self.assertEqual(len(allowance["edits"]), 21)
        carrying = {name for name, value in self.recorded.items() if "soft_position_tolerance_m" in json.dumps(value)}
        self.assertEqual(set(allowance["edits"]), carrying)
        for name, edits in allowance["edits"].items():
            self.assertEqual({pointer.rsplit("/", 1)[1]: old for pointer, old, new in edits}, RETIRED)
            self.assertTrue(all(new is golden.ABSENT for _, _, new in edits))
            _, ids = golden.apply_allowances("x5", "config", name, self.recorded[name])
            # WP12's F7 allowance, Stage 0B, and Phase 2 Wave 2 v2 allowance also edit Config / load / effective items.
            self.assertEqual([item for item in ids if item not in (F7_ALLOWANCE, STAGE0B_CONFIG_ALLOWANCE, P2_W2_CONFIG_ALLOWANCE,
                                                              P2_MODEL_DEFAULT_ALLOWANCE, REPLY_DELIVERY_ALLOWANCE, URDF_DELIVERY_ALLOWANCE,
                                                              CATALOGUE_DEFAULTS_ALLOWANCE)],
                             [allowance["id"]])
            with unittest.mock.patch.object(golden, "ALLOWANCES", [allowance]):
                ve_only, _ = golden.apply_allowances("x5", "config", name, self.recorded[name])
            removed = len(golden.json_lines(self.recorded[name])) - len(golden.json_lines(ve_only))
            self.assertEqual(removed, 7, name)

    def test_allowance_is_required_and_exact(self):
        name = "Config()"
        unchanged = golden.compare("x5", "config", {name: self.recorded[name]})
        self.assertEqual(unchanged["states"][name], "changed")
        expected, _ = golden.apply_allowances("x5", "config", name, self.recorded[name])
        self.assertEqual(golden.compare("x5", "config", {name: expected})["allowed"][name],
                         [self.allowance["id"], F7_ALLOWANCE, STAGE0B_CONFIG_ALLOWANCE, P2_W2_CONFIG_ALLOWANCE,
                          P2_MODEL_DEFAULT_ALLOWANCE, REPLY_DELIVERY_ALLOWANCE, URDF_DELIVERY_ALLOWANCE,
                          CATALOGUE_DEFAULTS_ALLOWANCE])
        tampered = deepcopy(expected)
        tampered["features"]["default_motion_speed_scale"] = 1.1
        self.assertEqual(golden.compare("x5", "config", {name: tampered})["states"][name], "changed")
        wrong = deepcopy(self.recorded[name])
        wrong["features"]["closed_jaw_goal_max"] = 0.06
        with self.assertRaisesRegex(ValueError, "allowance_old_value_mismatch"):
            golden.apply_allowances("x5", "config", name, wrong)
        self.assertEqual(golden.compare("x5", "config", {name: self.recorded[name]}, allow_allowances=False)
                         ["states"][name], "equal")


if __name__ == "__main__":
    unittest.main()
