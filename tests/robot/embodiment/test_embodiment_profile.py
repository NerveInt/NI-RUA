"""EmbodimentProfile (design 13 section 5.1, work package WP1): builders, backend hooks, Harness plumbing, archive.

* X5: the profile equals the facts every shared module still types at its point of use (until the owners of those
  modules read the profile instead), the RoboDojo hook passes the run thresholds (decision Q2) and the motion mode.
* SO-101: the profile built from ``configs/so101.example.json`` and from a deployment with the measured gripper
  values equals the expected facts; deployment thresholds load with units and defaults and are validated (Q4).
* Agreement: profile and the backend's worker ``capabilities()`` never disagree, key by key, for both robots.
* Harness: the profile is resolved before the tool registry, reachable as ``harness.embodiment``; C5/C7/C8/C9 read it.
* Archive: one ``embodiment_profile-<digest>`` configuration record per started run, absent from every model-visible
  input and actor-scope artifact; an archive without the record still resumes.

No model, network, GPU, robot or native simulator (the WP0 fixture robots: ``support.embodiment_golden``).
"""

NI_RUA_TEST_TIER = "component"

import asyncio
import copy
import dataclasses
import json
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

from support import embodiment_golden as golden

from ni_rua.core.action_validation import capability_arms
from ni_rua.core.config import Config, motion_thresholds
from ni_rua.core.contracts import digest
from ni_rua.robot.backends.so101 import deployment as so101_deployment
from ni_rua.robot.model import embodiments as e

EXAMPLE = golden.REPOSITORY / "configs" / "so101.example.json"
MEASURED = dict(empty_close_aperture_m=0.0051, full_open_aperture_m=0.1361)  # development unit, 2026-11-10..13


def x5_config(root, features=None):
    return golden.load_config(overrides={"features": dict(grasp_memory_dir=str(Path(root) / "grasp-memory"),
                                                          self_briefing=False, terminal=False, **(features or {}))})


def x5_robot(root, features=None):
    config = x5_config(root, features)
    return golden._x5_robot_class()(Path(root) / "robot", config), config


def so101_deployment_file(root, *, gripper=None, motion=None, cameras=None):
    """``configs/so101.example.json`` as the WP0 fixture writes it, with optional section edits."""
    path = golden.so101_deployment(Path(root) / "deployment")
    document = json.loads(path.read_text())
    document["gripper"].update(gripper or {})
    document["motion"].update(motion or {})
    if cameras is not None:
        document["cameras"] = {name: document["cameras"][name] for name in cameras}
    path.write_text(json.dumps(document, indent=2))
    return path


def so101_robot(root, *, features=None, **edits):
    path = so101_deployment_file(root, **edits)
    overrides = {"backend": {"deployment": str(path)},
                 "features": dict(grasp_memory_dir=str(Path(root) / "grasp-memory"), self_briefing=False,
                                  terminal=False, **(features or {}))}
    config = golden.load_config(golden.SO101_CONFIGS["so101_gateway"], overrides=overrides)
    return golden._so101_robot_class()(config=config, run_dir=Path(root) / "robot"), config


def started_harness(robot, config, root):
    from ni_rua.core.storage.store import Store
    from ni_rua.runtime.harness import Harness
    store = Store(Path(root) / "store", dict(golden.IDENTITY))
    harness = Harness(store, robot, config=config, observation_profile=config.features.observation_profile,
                      motion_review=False, task_plan_required=True)
    harness.start()
    return harness


class TemporaryRoot(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="ni-rua-embodiment-profile-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name).resolve()


# ---- the data type ------------------------------------------------------------------------------------------------
class ProfileTypeTests(unittest.TestCase):
    def test_profile_is_frozen_hashable_and_round_trips(self):
        profile = e.ARX_X5_PROFILE
        with self.assertRaises(dataclasses.FrozenInstanceError):
            profile.arms = ("left",)
        with self.assertRaises(dataclasses.FrozenInstanceError):
            profile.gripper.full_open_aperture_m = 0.1
        for mapping in (profile.speed_presets, profile.tolerance_presets, profile.wrist_cameras,
                        profile.gripper.preset_margins_m, profile.arm_joint_names):
            with self.assertRaises(TypeError):
                mapping["extra"] = 1
            with self.assertRaises(TypeError):
                mapping.update(extra=1)
            with self.assertRaises(TypeError):
                mapping.pop(next(iter(mapping)))
        self.assertEqual(hash(profile), hash(e.arx_x5_profile()))
        self.assertEqual(copy.deepcopy(profile), profile)
        self.assertEqual(pickle.loads(pickle.dumps(profile)), profile)
        self.assertEqual(dict(profile.speed_presets), {"slower": .5, "slow": .8, "careful": .9, "normal": 1.,
                                                       "fast": 1.25, "faster": 1.5})
        self.assertEqual((profile.robot_model_id, profile.frame_id, profile.tcp_config_id),
                         (e.ARX_X5.robot_model_id, e.ARX_X5.frame_id, e.ARX_X5.tcp_config_id))

    def test_construction_refuses_inconsistent_profiles(self):
        profile = e.ARX_X5_PROFILE
        cases = {
            "arms": dict(arms=()),
            "max_simultaneous_arms": dict(max_simultaneous_arms=3),
            "supports": dict(supports=profile.supports | {"teleport"}),
            "supports.dual_arm": dict(max_simultaneous_arms=1),
            "pose_inputs": dict(pose_inputs=frozenset()),
            "clock": dict(clock="wall"),
            "control_hz": dict(control_hz=0.0),
            "timeouts": dict(default_action_timeout_s=301.0),
            "per_request_step_limit": dict(per_request_step_limit=0),
            "overview_camera": dict(overview_camera="cam_top"),
            "speed_presets": dict(speed_presets=e.FrozenDict(warp=3.0)),
            "default_tolerance_preset": dict(default_tolerance_preset="snug"),
            "gripper.empty_close_aperture_m": dict(gripper=dataclasses.replace(
                profile.gripper, empty_close_aperture_m=profile.gripper.full_open_aperture_m)),
            "gripper.grasp_reference": dict(gripper=dataclasses.replace(profile.gripper, grasp_reference_offset_m=None)),
            "tolerance.default_tolerance": dict(default_tolerance=e.Tolerance(0.01, 0.15, 0.2)),
            "tolerance.compliant": dict(tolerance_presets=e.FrozenDict(
                dict(profile.tolerance_presets, compliant=e.Tolerance(0.015, 0.2, -0.01)))),
            "tolerance.seating": dict(tolerance_presets=e.FrozenDict(
                dict(profile.tolerance_presets, seating=e.Tolerance(0.015, 0.2, True)))),
        }
        for name, change in cases.items():
            with self.subTest(name=name), self.assertRaisesRegex(ValueError, "^invalid_embodiment_profile:" + name):
                dataclasses.replace(profile, **change)

    def test_record_is_plain_deterministic_json(self):
        record = e.profile_record(e.ARX_X5_PROFILE)
        self.assertEqual(record["schema"], e.PROFILE_RECORD_SCHEMA)
        self.assertEqual(json.loads(json.dumps(record, allow_nan=False)), record)
        self.assertEqual(record, e.profile_record(e.arx_x5_profile()))
        body = record["profile"]
        self.assertEqual(body["robot"], {"robot_model_id": "arx-x5", "frame_id": "env_origin",
                                         "tcp_config_id": "arx-x5-link6-v1"})
        self.assertEqual(body["supports"], sorted(e.ARX_X5_PROFILE.supports))
        self.assertEqual(body["gripper"]["full_open_aperture_m"], 0.088)
        self.assertEqual(body["tolerance_presets"]["normal"], {"position_m": 0.01, "orientation_rad": 0.15})
        self.assertEqual(body["tolerance_presets"]["compliant"],
                         {"position_m": 0.015, "orientation_rad": 0.2, "max_upward_dz_m": 0.04})
        self.assertEqual(body["tolerance_presets"]["seating"],
                         {"position_m": 0.015, "orientation_rad": 0.2, "max_upward_dz_m": 0.04})
        self.assertEqual(e.profile_from_record(record), e.ARX_X5_PROFILE)
        with self.assertRaises(TypeError):
            e.profile_record({"robot_model_id": "arx-x5"})

    def test_pre_ad01185_arx_x5_v1_record_restores_compliant_and_seating_upward_dz(self):
        from ni_rua.robot.motion.motion_tolerance import resolve_stage_tolerance

        legacy_x5 = copy.deepcopy(e.profile_record(e.ARX_X5_PROFILE))
        for preset in legacy_x5["profile"]["tolerance_presets"].values():
            preset.pop("max_upward_dz_m", None)
        restored_x5 = e.profile_from_record(legacy_x5)
        self.assertEqual(restored_x5.tolerance_presets["compliant"].max_upward_dz_m, 0.04)
        self.assertEqual(restored_x5.tolerance_presets["seating"].max_upward_dz_m, 0.04)
        self.assertIsNone(restored_x5.tolerance_presets["normal"].max_upward_dz_m)
        self.assertEqual(restored_x5, e.ARX_X5_PROFILE)
        self.assertEqual(
            resolve_stage_tolerance({"tolerance_preset": "compliant"}, profile=restored_x5),
            {
                "tolerance_preset": "compliant",
                "position_tolerance_m": 0.015,
                "orientation_tolerance_rad": 0.2,
                "max_upward_dz_m": 0.04,
            },
        )

        # Explicit max_upward_dz_m keys (including None or custom floats) are honored verbatim.
        explicit_none = copy.deepcopy(legacy_x5)
        explicit_none["profile"]["tolerance_presets"]["compliant"]["max_upward_dz_m"] = None
        restored_explicit = e.profile_from_record(explicit_none)
        self.assertIsNone(restored_explicit.tolerance_presets["compliant"].max_upward_dz_m)
        self.assertIsNone(restored_explicit.tolerance_presets["seating"].max_upward_dz_m)

        # SO-101 v1 records without max_upward_dz_m remain None across all presets.
        deployment = so101_deployment.validate_deployment(json.loads(EXAMPLE.read_text()), base=EXAMPLE.parent)
        so101_prof = so101_deployment.embodiment_profile(deployment)
        so101_rec = e.profile_record(so101_prof)
        restored_so101 = e.profile_from_record(so101_rec)
        self.assertIsNone(restored_so101.tolerance_presets["compliant"].max_upward_dz_m)
        self.assertTrue(all(tol.max_upward_dz_m is None for tol in restored_so101.tolerance_presets.values()))

    def test_importing_the_module_does_not_build_or_import_numpy(self):
        import subprocess
        import sys
        code = ("import sys; import ni_rua.robot.model.embodiments as e; import ni_rua.robot.backends.so101.deployment;"
                " assert 'numpy' not in sys.modules, 'numpy imported'; assert e._default_profile.cache_info().currsize == 0")
        subprocess.run([sys.executable, "-c", code], check=True, cwd=golden.REPOSITORY)


# ---- X5 facts: the profile equals what shared modules type today ---------------------------------------------------
class X5ProfileFactsTests(TemporaryRoot):
    """Until WP2/WP4/WP5/WP7 read the profile, every shared copy of an X5 fact must equal it (no silent drift)."""

    def test_presets_equal_the_shared_module_tables_in_order(self):
        from ni_rua.robot.motion import grip_guard, motion_speed, motion_tolerance
        profile = e.ARX_X5_PROFILE
        self.assertEqual(list(e.SPEED_PRESETS.items()), list(motion_speed.ARM_PRESETS.items()))
        self.assertEqual(list(e.GRIPPER_SPEED_PRESETS.items()), list(motion_speed.GRIPPER_SPEED_PRESETS.items()))
        self.assertEqual(list(profile.tolerance_presets.items()), [
            (name, e.Tolerance(value["position_tolerance_m"], value["orientation_tolerance_rad"],
                               value.get("max_upward_dz_m")))
            for name, value in motion_tolerance.TOLERANCE_PRESETS.items()])
        self.assertEqual(profile.tolerance_presets["compliant"].max_upward_dz_m, 0.04)
        self.assertEqual(profile.tolerance_presets["seating"].max_upward_dz_m, 0.04)
        self.assertIsNone(profile.tolerance_presets["normal"].max_upward_dz_m)
        self.assertEqual(profile.default_tolerance_preset, motion_tolerance.DEFAULT_TOLERANCE_PRESET)
        self.assertEqual(list(profile.gripper.preset_margins_m.items()), [
            (name, (value["min_aperture_m"], value["max_aperture_m"]))
            for name, value in grip_guard.GRIP_GUARD_PRESETS.items()])
        self.assertEqual(list(profile.stall_guard_presets_m.items()), list(motion_tolerance.STALL_GUARD_PRESETS.items()))
        self.assertEqual(e.ORIENTATION_FREE_TOLERANCE_RAD, motion_tolerance.ORIENTATION_FREE_TOLERANCE_RAD)
        self.assertEqual((profile.max_position_tolerance_m, profile.max_orientation_tolerance_rad),
                         (motion_tolerance.MAX_POSITION_TOLERANCE_M, motion_tolerance.MAX_ORIENTATION_TOLERANCE_RAD))
        self.assertEqual(profile.tracking_divergence_range_m,
                         (motion_tolerance.MIN_TRACKING_DIVERGENCE_M, motion_tolerance.MAX_TRACKING_DIVERGENCE_M))

    def test_single_full_opening_value_v_d(self):
        """Decision V-D: one X5 full opening, 0.088 m: the finger joint travel (joint7 + joint8), the guard maximum,
        the configured default and the 'full open 88mm, closed 0m' schema text (design 13 section 6: correct)."""
        from ni_rua.robot.motion import grip_guard
        from ni_rua.runtime.actions import robot_action_schema
        gripper = e.ARX_X5_PROFILE.gripper
        joints = {joint["name"]: joint for joint in e.physical_facts("arx_x5")["kinematics"]["joints"]}
        travel = joints["joint7"]["declared_limits"]["upper"] + joints["joint8"]["declared_limits"]["upper"]
        self.assertEqual(gripper.full_open_aperture_m, 0.088)
        self.assertEqual(gripper.full_open_aperture_m, travel)
        self.assertEqual(gripper.full_open_aperture_m, grip_guard.DEFAULT_MAX_APERTURE_M)
        self.assertEqual(gripper.full_open_aperture_m, Config().features.max_gripper_aperture_m)
        self.assertEqual(gripper.empty_close_aperture_m, grip_guard.DEFAULT_EMPTY_CLOSE_APERTURE_M)
        self.assertIn("[0, 0.088] (full open 88mm, closed 0m", robot_action_schema._STAGE_FIELD_HELP["min_aperture_m"])
        # The 0.0888 / 0.088807166 values are the distal pad surface gap (0.8 mm wider), which V-D replaces.
        pad_gap = e.physical_facts("arx_x5")["geometry"]["distal_flat_pad_gap"]["gap_q7_q8_0_044_m"]
        self.assertAlmostEqual(pad_gap - gripper.full_open_aperture_m, 0.000807, places=6)

    def test_other_x5_facts_equal_their_current_sources(self):
        from ni_rua.perception.external import object_table, pose_skill
        from ni_rua.robot.model import grasp_contact, roboprobe_pose
        from ni_rua.robot.motion import arx_actions
        from ni_rua.runtime.actions import home_check, robot_action_schema
        profile, features = e.ARX_X5_PROFILE, Config().features
        self.assertEqual(profile.gripper.grasp_reference_offset_m, roboprobe_pose.GRASP_POINT_OFFSET_M)
        self.assertEqual(profile.gripper.grasp_reference_offset_m,
                         e.physical_facts("arx_x5")["kinematics"]["pose_interface"]["pad_center_offset_m"])
        self.assertEqual(profile.gripper.fingertip_beyond_reference_m, pose_skill.FINGERTIP_BELOW_PAD_M)
        self.assertEqual(dict(profile.gripper.contact_sections), grasp_contact.CONTACT_DEPTH_PRESETS_M)
        self.assertEqual(profile.gripper.mask_segment, e.LinkSegment("link6", (1.0, 0.0, 0.0), object_table.GRIPPER_REACH_M))
        self.assertEqual((profile.default_action_timeout_s, profile.default_wait_timeout_s),
                         (arx_actions.DEFAULT_ACTION_TIMEOUT_S, arx_actions.DEFAULT_WAIT_TIMEOUT_S))
        policy = home_check.HOME_CHECK_POLICY
        self.assertEqual(profile.home_check, e.HomeCheckProfile(policy["version"], policy["frame_id"],
                                                                policy["tcp_config_id"], tuple(policy["arms"])))
        self.assertEqual(profile.overview_camera, features.vggt_camera_id)
        self.assertEqual(profile.overview_camera, object_table.DEFAULT_CAMERA)
        # features.vggt_support_z_m defaults to "profile" (WP12, F7): the configured height is the profile's.
        self.assertEqual(object_table.effective_support_z_m(features, None, profile), profile.support_surface_z_m)
        self.assertEqual(profile.arms, capability_arms(None))
        self.assertIn(profile.tcp_label, robot_action_schema._STAGE_FIELD_HELP["position_m"])
        self.assertIn(profile.stall_guard_semantics, robot_action_schema._STAGE_FIELD_HELP["stall_guard_preset"])
        self.assertEqual(profile.stages_per_arm * len(profile.arms), profile.max_stages)

    def test_run_thresholds_reach_the_robodojo_profile_q2(self):
        robot, config = x5_robot(self.root, dict(tight_position_tolerance_m=0.004, grip_rigid_min_aperture_m=0.004,
                                                 stall_guard_strict_divergence_m=0.012, max_gripper_aperture_m=0.08))
        profile = robot.embodiment_profile()
        self.assertEqual(profile.tolerance_presets["tight"], e.Tolerance(0.004, 0.05))
        self.assertEqual(profile.gripper.preset_margins_m["rigid_object"], (0.004, None))
        self.assertEqual(profile.stall_guard_presets_m["strict"], 0.012)
        self.assertEqual(profile.gripper.full_open_aperture_m, 0.08)
        self.assertEqual(profile, e.arx_x5_profile(binding=robot.robot_binding, thresholds=config.features))
        default, _ = x5_robot(self.root / "default")
        self.assertEqual(dataclasses.replace(default.embodiment_profile(), per_request_step_limit=None),
                         e.ARX_X5_PROFILE)

    def test_feedback_mode_is_one_arm_without_a_jaw_ramp(self):
        robot, _ = x5_robot(self.root, dict(motion_execution_mode="feedback", motion_review=False))
        profile = robot.embodiment_profile()
        self.assertEqual(profile.max_simultaneous_arms, 1)
        self.assertNotIn("dual_arm", profile.supports)
        self.assertNotIn("gripper_motion", profile.supports)
        self.assertEqual(profile.arms, ("left", "right"))
        with self.assertRaisesRegex(ValueError, "invalid_motion_execution_mode"):
            e.arx_x5_profile(motion_execution_mode="teleop")
        with self.assertRaisesRegex(ValueError, "unsupported_embodiment_binding"):
            e.arx_x5_profile(binding=dataclasses.replace(e.get_robodojo_robot(), robot=e.SO101))

    def test_hook_works_on_an_uninitialised_backend(self):
        from ni_rua.robot.backends.robodojo.robodojo import RobotBackend
        profile = object.__new__(RobotBackend).embodiment_profile()
        self.assertEqual(profile.per_request_step_limit, 250)
        self.assertEqual(dataclasses.replace(profile, per_request_step_limit=None), e.ARX_X5_PROFILE)


# ---- SO-101 facts and deployment thresholds -----------------------------------------------------------------------
def expected_so101(**gripper):
    """The SO-101 facts every deployment derived from the example shares (one arm, 5 joints + jaw, real time)."""
    return dict(
        robot_model_id="so101", frame_id="so101_base", tcp_config_id="so101-gripper-frame-v1",
        tcp_link="gripper_frame_link", tcp_label="gripper_frame_link", limits_profiles=("so101-deployment-v1",),
        arms=("right",), max_simultaneous_arms=1,
        arm_joint_names={"right": ("shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll")},
        jaw_joint_names={"right": ("gripper",)}, stages_per_arm=16, max_stages=32,
        # jaw_grasp_centre: the grasp_centre_m + object_width_m move input (gripper.jaw_grasp carries its CAD bounds).
        pose_inputs=frozenset({"tcp_quaternion", "jaw_grasp_centre"}),
        # reach: robot_action mode=reach on every SO-101 deployment; approach_roll needs the cuRobo ik_service.
        # robot_occlusion / contact_top: the default-on perception reports (deployment perception section absent).
        supports=frozenset({"stall_guard", "settle", "execute_if_clean", "gripper_motion", "max_upward_dz", "reach",
                            "held_perception", "cross_view", "arrival_correction", "robot_occlusion", "contact_top"}),
        speed_scale=e.ScaleRange(0.25, 1.0, 1.0), gripper_speed_scale=e.ScaleRange(0.25, 1.0, 1.0),
        speed_presets={"slower": .5, "slow": .8, "careful": .9, "normal": 1.},
        gripper_speed_presets={"slower": .5, "slow": .8, "normal": 1.},
        default_tolerance_preset="normal", default_tolerance=e.Tolerance(0.01, 0.15),
        max_position_tolerance_m=0.1, max_orientation_tolerance_rad=math.pi,
        clock="real_time", control_hz=20.0, step_domain="so101_control_step", per_request_step_limit=None,
        default_action_timeout_s=60.0, default_wait_timeout_s=30.0, max_timeout_s=300.0,
        cameras=("fixed", "wrist"), overview_camera="fixed", wrist_cameras={"right": "wrist"},
        camera_examples=("fixed", "wrist"), support_surface_z_m=None, handoff=False, policy_step=False,
        home_check=e.HomeCheckProfile("measured-so101-gripper-frame-home-v1", "so101_base", "so101-gripper-frame-v1",
                                      ("right",)),
        gripper=dict(dict(empty_close_aperture_m=None, full_open_aperture_m=0.1372,
                          full_open_source="CAD opening at gripper.jaw_open_rad",
                          preset_margins_m={"none": (None, None), "non_empty": (0.0015, None),
                                            "thin_object": (0.0005, None), "rigid_object": (0.002, None),
                                            "wide_object": (0.003, None)},
                          fingertip_beyond_reference_m=0.0081, grasp_reference="jaw_fingertip_midpoint",
                          grasp_reference_offset_m=None, approach_axis_local=(0.0, 0.0, 1.0),
                          jaw_opening_axis_local=(-1.0, 0.0, 0.0), mask_segment=None, contact_sections=None,
                          # CAD numbers of the grasp-centre pose input (so101_gripper_geometry.jaw_grasp_facts).
                          jaw_grasp=e.JawGraspGeometry(fixed_face_x_m=0.0, min_grasp_depth_m=-0.0053,
                                                       max_grasp_depth_m=0.0611, tcp_plane_gap_m=0.0618,
                                                       max_gap_m=0.1101)),
                     **gripper))


class So101ProfileTests(TemporaryRoot):
    def check(self, profile, expected):
        gripper = expected.pop("gripper")
        for name, value in expected.items():
            with self.subTest(field=name):
                self.assertEqual(getattr(profile, name), value)
        for name, value in gripper.items():
            with self.subTest(field="gripper." + name):
                self.assertEqual(getattr(profile.gripper, name), value)

    def test_profile_from_the_example_deployment(self):
        deployment = so101_deployment.validate_deployment(json.loads(EXAMPLE.read_text()), base=EXAMPLE.parent)
        profile = so101_deployment.embodiment_profile(deployment)
        self.check(profile, expected_so101())
        self.assertEqual(profile.tolerance_presets["unconstrained_orientation"], e.Tolerance(0.015, 3.14159))
        self.assertEqual(list(profile.tolerance_presets), list(so101_deployment.TOLERANCE_PRESET_NAMES))
        self.assertEqual(profile.stall_guard_presets_m, e.stall_guard_presets(motion_thresholds()))

    def test_profile_from_a_deployment_with_the_measured_gripper(self):
        robot, _ = so101_robot(self.root, gripper=MEASURED)
        profile = robot.embodiment_profile()
        self.check(profile, expected_so101(**MEASURED, full_open_source="deployment gripper.full_open_aperture_m"))
        self.assertEqual(profile, so101_deployment.embodiment_profile(robot.deployment))

    def test_deployment_thresholds_reach_the_profile_q4(self):
        robot, _ = so101_robot(self.root, gripper=dict(MEASURED, preset_margins_m=dict(
            so101_deployment.FIELDS["gripper"]["preset_margins_m"]["default"], rigid_object=[0.004, 0.09])),
            motion=dict(default_position_tolerance_m=0.02, default_wait_timeout_s=20.0, command_rate_hz=25.0,
                        max_command_delta_rad=0.035, tolerance_presets=dict(
                            so101_deployment.FIELDS["motion"]["tolerance_presets"]["default"], normal=[0.012, 0.2])),
            cameras=("wrist",))
        profile = robot.embodiment_profile()
        self.assertEqual(profile.gripper.preset_margins_m["rigid_object"], (0.004, 0.09))
        self.assertEqual(profile.tolerance_presets["normal"], e.Tolerance(0.012, 0.2))
        self.assertEqual(profile.default_tolerance, e.Tolerance(0.02, 0.15))
        self.assertEqual((profile.default_wait_timeout_s, profile.control_hz), (20.0, 25.0))
        self.assertEqual((profile.cameras, profile.overview_camera, dict(profile.wrist_cameras), profile.camera_examples),
                         (("wrist",), None, {"right": "wrist"}, ("wrist",)))

    def test_features_thresholds_are_not_read_for_so101_q4(self):
        robot, _ = so101_robot(self.root, features=dict(grip_rigid_min_aperture_m=0.004, max_gripper_aperture_m=0.05,
                                                        default_position_tolerance_m=0.02))
        profile = robot.embodiment_profile()
        self.assertEqual(profile.gripper.preset_margins_m["rigid_object"], (0.002, None))
        self.assertEqual(profile.gripper.full_open_aperture_m, 0.1372)
        self.assertEqual(profile.default_tolerance, e.Tolerance(0.01, 0.15))

    def test_profile_agrees_with_the_so101_contract_module(self):
        from ni_rua.robot.backends.so101 import contract, worker
        deployment = so101_deployment.validate_deployment(json.loads(EXAMPLE.read_text()), base=EXAMPLE.parent)
        profile = so101_deployment.embodiment_profile(deployment)
        self.assertEqual(profile.arms, contract.ARMS)
        self.assertEqual(profile.limits_profiles, (contract.LIMITS_PROFILE,))
        self.assertEqual((profile.frame_id, profile.tcp_config_id), (contract.FRAME_ID, contract.TCP_CONFIG_ID))
        self.assertEqual(profile.step_domain, worker.STEP_DOMAIN)
        self.assertEqual(profile.arm_joint_names["right"], so101_deployment.ARM_JOINTS)
        self.assertEqual(tuple(profile.gripper.preset_margins_m)[1:], contract.SCHEMA_GRIP_GUARD_PRESETS)
        gripper = contract.grip_guard_gripper(deployment)
        self.assertEqual((profile.gripper.empty_close_aperture_m, profile.gripper.full_open_aperture_m,
                          profile.gripper.full_open_source),
                         (gripper["empty_close_aperture_m"], gripper["full_open_aperture_m"], gripper["full_open_source"]))
        # Every bound the profile states is the bound the SO-101 validator enforces: at the limit accepted, beyond refused.
        point = {"right": {"position_m": [0.2, 0.0, 0.1], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}}

        def leaf(**changes):
            path = dict(kind="backend_free", orientation="backend_free", blend_radius_m=0.0, waypoints=[point],
                        position_tolerance_m=0.01, orientation_tolerance_rad=0.15)
            path.update(changes.pop("path", {}))
            return dict(dict(type="EEAction", mode="absolute", frame_id=profile.frame_id,
                             tcp_config_id=profile.tcp_config_id, limits_profile=profile.limits_profiles[0],
                             timeout_s=profile.default_action_timeout_s, path=path, gripper_events=[]), **changes)
        probes = [(dict(speed_scale=profile.speed_scale.max), dict(speed_scale=profile.speed_scale.max + 1e-9)),
                  (dict(gripper_speed_scale=profile.gripper_speed_scale.max),
                   dict(gripper_speed_scale=profile.gripper_speed_scale.max + 1e-9)),
                  (dict(timeout_s=profile.max_timeout_s), dict(timeout_s=profile.max_timeout_s + 1e-9)),
                  (dict(path=dict(position_tolerance_m=profile.max_position_tolerance_m)),
                   dict(path=dict(position_tolerance_m=profile.max_position_tolerance_m + 1e-9))),
                  (dict(path=dict(orientation_tolerance_rad=profile.max_orientation_tolerance_rad)),
                   dict(path=dict(orientation_tolerance_rad=profile.max_orientation_tolerance_rad + 1e-9)))]
        for accepted, refused in probes:
            with self.subTest(accepted=accepted):
                self.assertEqual(contract.validate_leaf(leaf(**accepted), deployment), "right")
                with self.assertRaisesRegex(ValueError, "invalid_action"):
                    contract.validate_leaf(leaf(**refused), deployment)


class So101DeploymentThresholdTests(unittest.TestCase):
    """Q4: SO-101 thresholds live in the deployment file, with units, defaults and validation."""

    def setUp(self):
        self.document = json.loads(EXAMPLE.read_text())

    def load(self, document=None):
        return so101_deployment.validate_deployment(document or self.document, base=EXAMPLE.parent)

    def test_example_carries_the_declared_defaults(self):
        for section, key in (("gripper", "preset_margins_m"), ("motion", "tolerance_presets"),
                             ("motion", "default_wait_timeout_s")):
            with self.subTest(field=f"{section}.{key}"):
                spec = so101_deployment.FIELDS[section][key]
                self.assertEqual(self.document[section][key], spec["default"])
                self.assertTrue(spec["unit"])
                self.assertIn("default", spec["doc"].lower())
                self.assertIn(key, so101_deployment.documentation()[section])
        shared = motion_thresholds()
        self.assertEqual({name: tuple(row) for name, row in self.document["gripper"]["preset_margins_m"].items()},
                         {name: e.grip_guard_preset_margins(shared)[name] for name in so101_deployment.GRIP_GUARD_PRESET_NAMES})
        self.assertEqual({name: e.Tolerance(*row) for name, row in self.document["motion"]["tolerance_presets"].items()},
                         {name: e.tolerance_presets(shared)[name] for name in so101_deployment.TOLERANCE_PRESET_NAMES})

    def test_deployment_without_the_new_fields_loads_with_defaults(self):
        for section, key in (("gripper", "preset_margins_m"), ("motion", "tolerance_presets"),
                             ("motion", "default_wait_timeout_s")):
            del self.document[section][key]
        before = copy.deepcopy(self.document)
        loaded = self.load()
        self.assertEqual(self.document, before)
        self.assertEqual(loaded["gripper"]["preset_margins_m"]["non_empty"], [0.0015, None])
        self.assertEqual(loaded["motion"]["tolerance_presets"]["coarse"], [0.025, 0.35])
        self.assertEqual(loaded["motion"]["default_wait_timeout_s"], 30.0)

    def test_explicit_values_are_preserved_exactly(self):
        self.document["gripper"]["preset_margins_m"]["wide_object"] = [0.006, 0.12]
        self.document["motion"]["tolerance_presets"]["tight"] = [0.002, 0.04]
        self.document["motion"]["default_wait_timeout_s"] = 12.5
        loaded = self.load()
        self.assertEqual(loaded["gripper"]["preset_margins_m"]["wide_object"], [0.006, 0.12])
        self.assertEqual(loaded["motion"]["tolerance_presets"]["tight"], [0.002, 0.04])
        self.assertEqual(loaded["motion"]["default_wait_timeout_s"], 12.5)

    def test_invalid_values_are_refused_before_a_worker_starts(self):
        def margins(**rows):
            return dict(self.document["gripper"]["preset_margins_m"], **rows)

        def tolerances(**rows):
            return dict(self.document["motion"]["tolerance_presets"], **rows)
        cases = [
            ("gripper", "preset_margins_m", {"non_empty": [0.0015, None]}, "exactly the keys"),
            ("gripper", "preset_margins_m", margins(extra=[0.001, None]), "exactly the keys"),
            ("gripper", "preset_margins_m", margins(non_empty=[0.0015]), "list"),
            ("gripper", "preset_margins_m", margins(non_empty=0.0015), "list"),
            ("gripper", "preset_margins_m", margins(non_empty=[-0.001, None]), "out_of_range"),
            ("gripper", "preset_margins_m", margins(non_empty=[0.051, None]), "out_of_range"),
            ("gripper", "preset_margins_m", margins(non_empty=[None, None]), "finite number"),
            ("gripper", "preset_margins_m", margins(non_empty=[float("nan"), None]), "finite number"),
            ("gripper", "preset_margins_m", margins(non_empty=[True, None]), "finite number"),
            ("gripper", "preset_margins_m", margins(non_empty=["0.001", None]), "finite number"),
            ("gripper", "preset_margins_m", margins(non_empty=[0.003, 0.003]), "max_margin must exceed"),
            ("gripper", "preset_margins_m", margins(non_empty=[0.003, 0.201]), "out_of_range"),
            ("motion", "tolerance_presets", tolerances(tight=[0.0, 0.05]), "out_of_range"),
            ("motion", "tolerance_presets", tolerances(tight=[0.101, 0.05]), "out_of_range"),
            ("motion", "tolerance_presets", tolerances(tight=[0.003, math.pi + 1e-6]), "out_of_range"),
            ("motion", "tolerance_presets", tolerances(tight=[0.003, None]), "finite number"),
            ("motion", "tolerance_presets", {name: row for name, row in tolerances().items() if name != "coarse"},
             "exactly the keys"),
            ("motion", "default_wait_timeout_s", 0.05, "out_of_range"),
            ("motion", "default_wait_timeout_s", "30", "finite number"),
        ]
        for section, key, value, message in cases:
            with self.subTest(field=f"{section}.{key}", value=value):
                document = copy.deepcopy(self.document)
                document[section][key] = value
                with self.assertRaisesRegex(ValueError, message):
                    self.load(document)
        document = copy.deepcopy(self.document)
        document["motion"].update(default_wait_timeout_s=200.0, max_timeout_s=100.0)
        with self.assertRaisesRegex(ValueError, "default_wait_timeout_s exceeds max_timeout_s"):
            self.load(document)
        document = copy.deepcopy(self.document)
        document["gripper"].update(empty_close_aperture_m=0.0051, full_open_aperture_m=0.04,
                                   preset_margins_m=margins(wide_object=[0.035, None]))
        with self.assertRaisesRegex(ValueError, "must stay below full_open_aperture_m"):
            self.load(document)


# ---- agreement: profile vs the worker's capabilities, key by key ----------------------------------------------------
def capability_facts(capabilities):
    speed, unified = capabilities["action_speed"], capabilities["unified_action"]
    return dict(
        robot_model_id=capabilities["robot_model_id"], frame_id=capabilities["frame_id"],
        tcp_config_id=capabilities["tcp_config_id"], arms=capability_arms(capabilities),
        max_simultaneous_arms=capabilities["max_simultaneous_arms"], dual_arm=capabilities["dual_arm"],
        arm_joint_names={arm: tuple(names) for arm, names in capabilities["arm_joint_names"].items()},
        limits_profile=capabilities["limits_profile"], control_hz=capabilities["native_control_hz"],
        gripper_motion="during_motion" in capabilities["gripper_phases"],
        max_stages=unified["max_stages"], stages_per_arm=unified["max_stages_per_arm"],
        per_request_step_limit=unified.get("max_native_steps"),
        speed_scale=tuple(speed["speed_scale"][key] for key in ("min", "max", "default")),
        gripper_speed_scale=tuple(speed["gripper_speed_scale"][key] for key in ("min", "max", "default")),
        clock="real_time" if capabilities["clock_policy"].startswith("real time") else "simulated_steps")


def profile_facts(profile):
    return dict(
        robot_model_id=profile.robot_model_id, frame_id=profile.frame_id, tcp_config_id=profile.tcp_config_id,
        arms=profile.arms, max_simultaneous_arms=profile.max_simultaneous_arms, dual_arm="dual_arm" in profile.supports,
        arm_joint_names=dict(profile.arm_joint_names), limits_profile=profile.limits_profiles[0],
        control_hz=profile.control_hz, gripper_motion="gripper_motion" in profile.supports,
        max_stages=profile.max_stages, stages_per_arm=profile.stages_per_arm,
        per_request_step_limit=profile.per_request_step_limit,
        speed_scale=dataclasses.astuple(profile.speed_scale),
        gripper_speed_scale=dataclasses.astuple(profile.gripper_speed_scale), clock=profile.clock)


class CapabilityAgreementTests(TemporaryRoot):
    """The profile and the backend's ``capabilities()`` never disagree (design 13 section 5.1)."""

    def assert_agree(self, profile, capabilities):
        expected, actual = profile_facts(profile), capability_facts(capabilities)
        for key in expected:
            with self.subTest(key=key):
                self.assertEqual(actual[key], expected[key])

    def test_x5_worker_capabilities(self):
        for features in ({}, {"motion_execution_mode": "feedback", "motion_review": False},
                         {"observation_profile": "rgb_proprio_v1", "motion_review": False}):
            with self.subTest(features=features):
                robot, _ = x5_robot(self.root / str(len(features)) / json.dumps(features).replace("/", ""), features)
                profile, capabilities = robot.embodiment_profile(), robot.start()
                self.assert_agree(profile, capabilities)
                geometry = capabilities["tool_geometry"]
                self.assertEqual(geometry["link6_to_grasp_center_local_m"],
                                 [profile.gripper.grasp_reference_offset_m, 0, 0])
                self.assertEqual(tuple(geometry["approach_axis_local"]), profile.gripper.approach_axis_local)
                self.assertEqual(tuple(geometry["jaw_opening_axis_local"]), profile.gripper.jaw_opening_axis_local)
                self.assertEqual({name: region["contact_depth_m"]
                                  for name, region in geometry["contact_selection"]["regions"].items()},
                                 dict(profile.gripper.contact_sections))
                self.assertEqual(math.ceil(capabilities["motion_duration_limit_s"] * profile.control_hz),
                                 profile.per_request_step_limit)
                self.assertEqual(capabilities["motion_duration_limit_s"], e.ROBODOJO_REQUEST_DURATION_LIMIT_S)

    def test_so101_worker_capabilities(self):
        for name, gripper in (("example", {}), ("measured", MEASURED)):
            with self.subTest(deployment=name):
                robot, _ = so101_robot(self.root / name, gripper=gripper)
                profile, capabilities = robot.embodiment_profile(), robot.start()
                self.assert_agree(profile, capabilities)
                self.assertEqual(tuple(capabilities["cameras"]), profile.cameras)
                self.assertEqual((capabilities["default_timeout_s"], capabilities["max_timeout_s"]),
                                 (profile.default_action_timeout_s, profile.max_timeout_s))
                self.assertEqual(capabilities["motion_tolerance_defaults"],
                                 {"position_tolerance_m": profile.default_tolerance.position_m,
                                  "orientation_tolerance_rad": profile.default_tolerance.orientation_rad})
                guard = capabilities["grip_guard"]
                self.assertEqual((guard["empty_close_aperture_m"], guard["full_open_aperture_m"], guard["full_open_source"]),
                                 (profile.gripper.empty_close_aperture_m, profile.gripper.full_open_aperture_m,
                                  profile.gripper.full_open_source))
                zero = profile.gripper.empty_close_aperture_m
                self.assertEqual(guard["presets_available"], zero is not None)
                if zero is not None:
                    self.assertEqual({preset: bounds["min_aperture_m"] for preset, bounds in guard["presets"].items()},
                                     {preset: round(zero + profile.gripper.preset_margins_m[preset][0], 4)
                                      for preset in guard["presets"]})
                geometry = capabilities["tool_geometry"]
                self.assertEqual(geometry["tcp_link"], profile.tcp_link)
                self.assertEqual(tuple(geometry["gripper_geometry"]["approach_axis_tcp"]),
                                 profile.gripper.approach_axis_local)
                self.assertEqual(geometry["gripper_geometry"]["protrusion_beyond_tcp_m"]["max"],
                                 profile.gripper.fingertip_beyond_reference_m)
                self.assertEqual(capabilities["bounded_wait"]["native_control_hz"], profile.control_hz)
                self.assertIn(profile.step_domain, capabilities["step_semantics"])
                robot.close()


# ---- Harness plumbing ----------------------------------------------------------------------------------------------
class HarnessPlumbingTests(TemporaryRoot):
    def test_backend_hook_profile_is_the_run_profile(self):
        robot, config = x5_robot(self.root / "x5")
        harness = started_harness(robot, config, self.root / "x5")
        self.assertEqual(harness.embodiment, robot.embodiment_profile())
        self.assertEqual(harness.action_budget()["per_request_native_limit"], 250)
        self.assertEqual(list(harness.capabilities["action_speed"]["speed_presets"]), list(e.SPEED_PRESETS))
        self.assertIn("handoff_policy", harness.capabilities)
        robot, config = so101_robot(self.root / "so101", gripper=MEASURED)
        harness = started_harness(robot, config, self.root / "so101")
        self.assertEqual(harness.embodiment, so101_deployment.embodiment_profile(robot.deployment))
        self.assertIsNone(harness.action_budget()["per_request_native_limit"])
        speed = harness.capabilities["action_speed"]
        self.assertEqual(speed["speed_presets"], {"slower": .5, "slow": .8, "careful": .9, "normal": 1.})  # C5
        self.assertLessEqual(max(speed["speed_presets"].values()), speed["speed_scale"]["max"])
        self.assertNotIn("handoff_policy", harness.capabilities)  # no second arm to hand over to
        self.assertEqual(harness.latest["robot_model_id"], "so101")
        robot.close()

    def test_backend_without_a_hook_gets_the_default_robot(self):
        from ni_rua.core.storage.store import Store
        from ni_rua.runtime.harness import Harness, resolve_embodiment_profile
        for robot in (None, SimpleNamespace(), mock.Mock(), mock.MagicMock()):
            with self.subTest(robot=type(robot).__name__):
                profile = resolve_embodiment_profile(robot, Config())
                self.assertEqual(profile, e.ARX_X5_PROFILE)
                self.assertIsNone(profile.per_request_step_limit)
        tuned = resolve_embodiment_profile(SimpleNamespace(), Config.from_dict({"features": {"grip_wide_min_aperture_m": 0.005}}))
        self.assertEqual(tuned.gripper.preset_margins_m["wide_object"], (0.005, None))
        harness = Harness(Store(self.root / "store", dict(golden.IDENTITY)), None, config=Config())
        self.assertEqual(harness.embodiment, e.ARX_X5_PROFILE)
        self.assertIsNone(harness.action_budget()["per_request_native_limit"])
        self.assertIsNone(harness.embodiment_record_id)

    def test_hook_returning_something_else_is_refused(self):
        from ni_rua.runtime.harness import resolve_embodiment_profile
        for value in (None, {}, e.profile_record(e.ARX_X5_PROFILE)):
            robot = SimpleNamespace(embodiment_profile=lambda value=value: value)
            with self.subTest(value=type(value).__name__), self.assertRaisesRegex(ValueError, "invalid_embodiment_profile"):
                resolve_embodiment_profile(robot, Config())

    def test_profile_exists_before_the_tool_registry(self):
        from ni_rua.core.storage.store import Store
        from ni_rua.runtime.harness import Harness
        from ni_rua.runtime.tools import builtins
        seen = []
        original = builtins.create_registry

        def recording(specs=(), *args, runtime=None, **kwargs):
            seen.append(getattr(runtime, "embodiment", None))
            return original(specs, *args, runtime=runtime, **kwargs)
        robot, config = x5_robot(self.root)
        with mock.patch.object(builtins, "create_registry", side_effect=recording):
            Harness(Store(self.root / "store", dict(golden.IDENTITY)), robot, config=config)
        self.assertEqual(seen, [robot.embodiment_profile()])

    def test_observation_robot_model_id_and_grasp_pose_follow_the_profile(self):
        X5Robot = golden._x5_robot_class()
        custom = dataclasses.replace(e.ARX_X5_PROFILE, robot=dataclasses.replace(e.ARX_X5, robot_model_id="custom-arm"),
                                     pose_inputs=frozenset({"tcp_quaternion"}))

        class CustomRobot(X5Robot):
            def start(self):
                capabilities = super().start()
                capabilities.pop("robot_model_id")
                return capabilities

            def embodiment_profile(self):
                return custom
        config = x5_config(self.root)
        harness = started_harness(CustomRobot(self.root / "robot", config), config, self.root)
        self.assertEqual(harness.latest["robot_model_id"], "custom-arm")  # C8: no "arx-x5" literal fallback
        state = harness.observation_content(harness.latest, {}, structured=True)[0].value["robot_state"]
        self.assertTrue(state["arms"])
        self.assertFalse(any("grasp_pose" in arm for arm in state["arms"].values()))  # C9: no angles_deg pose input
        x5, x5_config_ = x5_robot(self.root / "x5")
        harness = started_harness(x5, x5_config_, self.root / "x5")
        state = harness.observation_content(harness.latest, {}, structured=True)[0].value["robot_state"]
        self.assertTrue(all("grasp_pose" in arm for arm in state["arms"].values()))


# ---- archive record ------------------------------------------------------------------------------------------------
class Session:
    """``FakeAgentSession`` factory for ``workbench.create_session`` that records the session inputs."""

    def __init__(self):
        self.inputs, self.created = {}, []

    def __call__(self, config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
        from support.fake_agent import FakeAgentSession
        session = FakeAgentSession(run_dir, store, tool_handler, price_card=price_card)
        start, run_turn = session.start, session.run_turn

        def recording_start(tools, instructions, *args, **kwargs):
            self.inputs.update(tools=copy.deepcopy(tools), instructions=instructions)
            return start(tools, instructions, *args, **kwargs)

        def recording_turn(prompt, *args, **kwargs):
            self.inputs.setdefault("prompt", prompt)
            return run_turn(prompt, *args, **kwargs)
        session.start, session.run_turn = recording_start, recording_turn
        self.created.append(session)
        return session


def run_workbench(run, robot, config, *, resume=False):
    """Start and close a real ``Workbench`` (FakeAgentSession, no binary); returns what the store and the session
    held: ``records`` (configuration), ``actor_artifacts`` (bytes of every actor-scope artifact), ``observations``,
    ``capabilities``, ``record_id`` and the session ``inputs``."""
    from ni_rua.apps.gateway import workbench as workbench_module
    from ni_rua.apps.gateway.workbench import DEFAULT_PROMPT, Workbench
    factory = Session()
    with mock.patch.object(workbench_module, "create_session", side_effect=factory):
        workbench = (Workbench(run, robot=robot, visual_review_model=None, config=config, resume=True) if resume
                     else Workbench(run, robot=robot, visual_review_model=None, config=config))

    async def go():
        try:
            await workbench.start(DEFAULT_PROMPT, benchmark=False)
            store = workbench.store
            return dict(records=store.record_items("configuration"), observations=store.records("observation"),
                        actor_artifacts=[store.resolve_artifact("artifact:" + item["artifact_id"], ("actor",))[1]
                                         for item in store.artifacts() if item["access_scope"] == "actor"],
                        capabilities=copy.deepcopy(workbench.harness.capabilities),
                        record_id=workbench.harness.embodiment_record_id, identity=dict(workbench.identity))
        finally:
            await workbench.close()
    result = asyncio.run(go())
    result["inputs"] = factory.inputs
    return result


class ArchiveRecordTests(TemporaryRoot):
    def test_harness_writes_one_profile_record_per_start(self):
        robot, config = so101_robot(self.root, gripper=MEASURED)
        harness = started_harness(robot, config, self.root)
        records = {name: value for name, value in harness.store.record_items("configuration").items()
                   if name.startswith(e.EMBODIMENT_RECORD_PREFIX)}
        record = e.profile_record(harness.embodiment)
        self.assertEqual(records, {e.EMBODIMENT_RECORD_PREFIX + digest(record): record})
        self.assertEqual(harness.embodiment_record_id, next(iter(records)))
        self.assertEqual(record["profile"]["gripper"]["full_open_aperture_m"], 0.1361)
        self.assertEqual(record["profile"]["robot"]["robot_model_id"], "so101")
        robot.close()

    def test_record_never_reaches_model_context_or_actor_artifacts(self):
        for name, build in (("x5", lambda root: x5_robot(root)), ("so101", lambda root: so101_robot(root, gripper=MEASURED))):
            with self.subTest(embodiment=name):
                root = self.root / name
                robot, config = build(root)
                result = run_workbench(root / "run", robot, config)
                record_id = result["record_id"]
                self.assertIn(record_id, result["records"])
                needles = (record_id, record_id.removeprefix(e.EMBODIMENT_RECORD_PREFIX), e.EMBODIMENT_RECORD_PREFIX,
                           e.PROFILE_RECORD_SCHEMA)
                visible = [result["inputs"]["instructions"], result["inputs"]["prompt"],
                           json.dumps(result["inputs"]["tools"]), json.dumps(result["capabilities"]),
                           json.dumps(result["observations"])]
                visible += [data.decode("utf-8", "replace") for data in result["actor_artifacts"]]
                self.assertTrue(result["actor_artifacts"])
                for needle in needles:
                    for index, text in enumerate(visible):
                        with self.subTest(needle=needle, item=index):
                            self.assertNotIn(needle, text)

    def test_archive_without_a_profile_record_still_resumes(self):
        from ni_rua.core.storage.store import Store
        robot, config = x5_robot(self.root)
        run = self.root / "run"
        first = run_workbench(run, robot, config)
        # Make it an archive from before the profile existed: no profile record (and no agent identity, so the
        # resumed fixture session starts a fresh thread).
        store = Store(run, first["identity"])
        with store.lock, store.db:
            store.db.execute("DELETE FROM records WHERE kind='configuration' AND id LIKE ?",
                             (e.EMBODIMENT_RECORD_PREFIX + "%",))
        store.set_state("runtime", "agent", None)
        self.assertFalse(any(name.startswith(e.EMBODIMENT_RECORD_PREFIX) for name in store.record_items("configuration")))
        kept = {name: value for name, value in store.record_items("configuration").items()}
        store.close()
        robot, _ = x5_robot(self.root / "resumed")
        resumed = run_workbench(run, robot, config, resume=True)
        self.assertEqual(resumed["identity"]["env_generation"], first["identity"]["env_generation"] + 1)
        for name, value in kept.items():
            self.assertEqual(resumed["records"][name], value)
        self.assertEqual(resumed["record_id"], first["record_id"])
        self.assertIn(resumed["record_id"], resumed["records"])


if __name__ == "__main__":
    unittest.main()
