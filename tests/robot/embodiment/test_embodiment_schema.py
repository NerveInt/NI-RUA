"""robot_action schema and tool definitions rendered from the run's EmbodimentProfile (design 13 section 5.2, WP2).

* ARX-X5: the published catalogue differs from the 5fd93c6 fixture by exactly the approved section 6 correction and
  the approved V-F allowance (``support.embodiment_golden`` ``# --- WP2 allowances ---``); the test prints the diff.
* SO-101 (fake arm, ``configs/so101_gateway.json`` + ``configs/so101.example.json``): the catalogue a real Workbench
  start publishes carries no ARX-X5 fact and none of the fields SO-101 cannot honour (decision Q3: absent, one
  generated sentence names the features); every numeric bound of robot_action is the profile's and is enforced at the
  limit (accepted at it, rejected just beyond) by the run's schema and validators; dispatch validates against the
  run's schema; omitted deadlines and default cameras come from the profile.
* Native payload: the generated SO-101 robot_action text reaches the upstream model request of the pinned Codex,
  Claude Code and (with ``NI_RUA_OPENCODE_EXECUTABLE``) OpenCode binaries unchanged (fake loopback model, no network).

Run: ``PYTHONPATH=tests mamba run -n ni-rua python -m unittest tests.test_embodiment_schema -v``.
"""

NI_RUA_TEST_TIER = "fast"

from collections import Counter
from copy import deepcopy
import json
import os
from pathlib import Path
import re
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest import mock

from support import embodiment_golden as golden

from ni_rua.robot.model import embodiments
from ni_rua.runtime.actions import robot_action_schema
from ni_rua.runtime.actions.robot_action_schema import _metres, build

DELTA = 1e-6  # "just beyond" a bound, in the field's unit (m, rad, s or dimensionless)


def _so101_profile(**gripper):
    from ni_rua.robot.backends.so101.deployment import embodiment_profile, validate_deployment
    document = json.loads((golden.REPOSITORY / "configs" / "so101.example.json").read_text())
    document["gripper"].update(gripper)
    return embodiment_profile(validate_deployment(document, base=golden.REPOSITORY / "configs"))


def _text(value):
    return json.dumps(value, ensure_ascii=False)


# ---- ARX-X5: exact catalogue difference ------------------------------------------------------------------------
def _leaf_changes(before, after, pairs):
    """``(explained, unexplained)``: for every string leaf that differs, the ``(pointer, old, new, count)`` of each
    replacement pair that turns ``before`` into ``after``; leaves no pair explains (and non-string differences)."""
    old, new = dict(golden.string_leaves(before)), dict(golden.string_leaves(after))
    explained, unexplained = [], []
    for pointer in sorted(set(old) | set(new)):
        if old.get(pointer) == new.get(pointer):
            continue
        if pointer not in old or pointer not in new:
            unexplained.append((pointer, old.get(pointer), new.get(pointer)))
            continue
        text, used = old[pointer], []
        for first, second in pairs:
            if first in text:
                used.append((pointer, first, second, text.count(first)))
                text = text.replace(first, second)
        if text == new[pointer]:
            explained += used
        else:
            unexplained.append((pointer, old[pointer], new[pointer]))
    if golden.render([v for _, v in sorted(_scalars(before))]) != golden.render([v for _, v in sorted(_scalars(after))]):
        unexplained.append(("non-string values", None, None))
    return explained, unexplained


def _scalars(value, pointer=""):
    if isinstance(value, dict):
        for key, item in value.items():
            yield from _scalars(item, f"{pointer}/{key}")
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from _scalars(item, f"{pointer}/{index}")
    elif not isinstance(value, str):
        yield pointer, value


class X5CatalogueDifferenceTests(unittest.TestCase):
    """The X5 catalogue = fixture + approved allowances + section 6 correction, nothing else."""

    maxDiff = None

    def test_x5_catalogue_differs_by_exactly_the_section_6_correction_and_v_f(self):
        _, recorded = golden.load_fixture("x5", "catalogue")
        current = golden.capture("x5", "catalogue", recorded=recorded)
        v_f = next(item for item in golden.ALLOWANCES if item["id"] == "design-13-V-F-max-upward-dz-help")
        golden.check_allowance(v_f)
        pairs = list(golden.APPROVED_CORRECTION["replacements"])
        found, unexplained = Counter(), []
        for name, before in recorded.items():
            base, _ = golden.apply_allowances("x5", "catalogue", name, before)
            tools = {tool["name"]: index for index, tool in enumerate(base)}
            self.assertEqual([tool["name"] for tool in current[name]], [tool["name"] for tool in base], name)
            for tool, index in tools.items():
                explained, rest = _leaf_changes(base[index], current[name][index], pairs)
                found.update((tool,) + change for change in explained)
                unexplained += [(name, tool) + change for change in rest]
        carrying = [name for name, value in recorded.items() if golden.correction_applies("catalogue", value)]
        expected = Counter()
        for _ in carrying:
            for site, counts in golden.APPROVED_CORRECTION["sites"].items():
                expected.update(("robot_action", site, old, new, counts[old])
                                for old, new in golden.APPROVED_CORRECTION["replacements"] if old in counts)
        print(f"\nX5 catalogue difference ({len(recorded)} items, {len(carrying)} carry robot_action):")
        for (tool, site, old, new, count), items in sorted(found.items()):
            print(f"  {tool}{site}: x{count} in {items} items\n    - {old}\n    + {new}")
        self.assertEqual(unexplained, [])
        self.assertEqual(len(carrying), 23)
        self.assertEqual(found, expected)
        self.assertEqual(sum(change[-1] * items for change, items in found.items()), 23 * 21)


# ---- SO-101: the published catalogue ---------------------------------------------------------------------------
X5_TERMS = (r"\bleft\b", r"env_origin", r"link6", r"cam_head", r"88", r"25 Hz", r"simulat", r"native budget",
            r"RoboDojo", r"\bX5\b", r"both arms")


class So101CatalogueTests(unittest.TestCase):
    """The catalogue of a real SO-101 Workbench start (fake arm and cameras, FakeAgentSession)."""

    @classmethod
    def setUpClass(cls):
        startup = golden.run_startup("so101", "so101_gateway")
        cls.catalogues = {"actor": startup["tools"], "program": startup["program_tools"]}
        cls.profile = _so101_profile()
        cls.x5, cls.so101 = build(None), build(cls.profile)

    def fields(self, contract):
        properties = contract.tool["inputSchema"]["properties"]
        stage = {field for schema in contract.stage_schemas.values() for field in schema["properties"]}
        return set(properties) | stage

    def test_catalogue_carries_no_x5_fact(self):
        for role, tools in self.catalogues.items():
            for tool in tools:
                text = _text(tool)
                for term in X5_TERMS:
                    with self.subTest(role=role, tool=tool["name"], term=term):
                        self.assertIsNone(re.search(term, text))

    def test_fields_so101_cannot_honour_are_absent_and_named_once_in_words(self):
        absent = self.fields(self.x5) - self.fields(self.so101)
        self.assertEqual(absent, {
            "grasp_position_m", "angles_deg", "contact_region", "contact_depth_m",
            "allow_jaw_symmetry", "allow_approach_tilt", "max_downward_dz_m",
            "grip_guard_preset", "joint_travel_preset", "max_joint_travel_rad", "max_backtracking_rad",
            "clearance_preset", "clearance_height_m", "obstacles", "obstacle_object_ids",
            "obstacle_policy", "ignore_obstacle_ids", "target_exemption_preset", "target_exemption_radius_m",
            "held_objects", "held_object", "contact_entity_id", "initial_contact", "placement_contact"})
        for role, tools in self.catalogues.items():
            text = _text(tools)
            for field in absent:
                with self.subTest(role=role, field=field):
                    self.assertIsNone(re.search(r"\b" + field + r"\b", text))
            action = next(tool for tool in tools if tool["name"] == "robot_action")
            self.assertEqual(action["description"].count("Not available on this robot, so absent from this schema:"), 1)
            self.assertIn("joint-travel admission limits", action["description"])
            # WP9: the SO-101 profile supports gripper_motion and max_upward_dz, so both fields are published (stage and
            # batch level) and the absent-feature sentence does not name them.
            self.assertNotIn("a vertical arrival window", action["description"])
            self.assertNotIn("jaw motion during a move", action["description"])
            move = action["inputSchema"]["properties"]["stages"]["items"]["properties"]
            self.assertIn("gripper_motion", move)
            self.assertIn("max_upward_dz_m", move)
            self.assertIn("max_upward_dz_m", action["inputSchema"]["properties"])
            # speed presets above the SO-101 range are not offered
            self.assertNotIn("fast", _text(action["inputSchema"]["properties"]["stages"]))
        # The X5 catalogue names no absent feature.
        self.assertNotIn("Not available on this robot", self.x5.tool["description"])

    def test_published_robot_action_is_the_profile_build(self):
        for role, tools in self.catalogues.items():
            action = next(tool for tool in tools if tool["name"] == "robot_action")
            self.assertEqual(action["inputSchema"], self.so101.tool["inputSchema"], role)
            self.assertTrue(action["description"].startswith(self.so101.tool["description"]))

    def test_numbers_in_the_text_are_the_profile_values(self):
        profile, contract = self.profile, self.so101
        gripper, help_ = profile.gripper, contract.stage_field_help
        full = gripper.full_open_aperture_m
        self.assertEqual(full, 0.1372)  # CAD opening at jaw_open_rad (example deployment: no measured value)
        properties = contract.tool["inputSchema"]["properties"]
        stage = contract.stage_schemas["move"]["properties"]
        self.assertEqual(stage["arm"]["enum"], list(profile.arms))
        for field in ("min_aperture_m", "max_aperture_m"):
            self.assertEqual((stage[field]["maximum"], properties[field]["maximum"]), (full, full))
            self.assertIn(f"[0, {_metres(full)}]", help_[field])
        self.assertIn(f"[0..{_metres(full)}m]", contract.tool["description"])
        self.assertEqual((stage["speed_scale"]["minimum"], stage["speed_scale"]["maximum"]), (0.25, 1))
        self.assertEqual(stage["speed_preset"]["enum"], list(profile.speed_presets))
        self.assertIn("[0.25,1]", help_["speed_scale"])
        self.assertEqual(properties["timeout_s"]["maximum"], profile.max_timeout_s)
        self.assertIn(f"default {profile.default_action_timeout_s:g}, range 0.1..{profile.max_timeout_s:g}",
                      contract.action_field_help["timeout_s"])
        self.assertIn(f"default {profile.overview_camera},", contract.action_field_help["camera_id"])
        self.assertIn(profile.stall_guard_semantics, help_["stall_guard_preset"])
        # The vertical window help is this robot's rule (default 0 = no window), not the X5 executor's default table.
        self.assertEqual(help_["max_upward_dz_m"], profile.vertical_window_semantics)
        self.assertIn("Default 0 m (no window) when omitted", help_["max_upward_dz_m"])
        self.assertNotIn("Defaults when omitted: max(pos_tol", _text(self.catalogues["actor"]))
        self.assertIn(f"target {profile.tcp_label} [x,y,z] in {profile.frame_id} metres", help_["position_m"])
        self.assertNotIn("Shared rows cost one native step", contract.tool["description"])
        for name, tolerance in profile.tolerance_presets.items():
            if name in contract.stage_schemas["move"]["properties"]["tolerance_preset"]["enum"]:
                self.assertIn(f"{tolerance.position_m:.3f} m, ", help_["tolerance_preset"], name)
        self.assertIn(f"default normal={profile.default_tolerance.position_m:.3f} m", help_["position_tolerance_m"])
        divergence = profile.tracking_divergence_range_m
        self.assertIn(f"[{divergence[0]:g}, {divergence[1]:g}]", help_["max_tracking_divergence_m"])
        tools = {tool["name"]: tool for tool in self.catalogues["actor"]}
        wait = tools["robot_wait"]
        hz = profile.control_hz
        self.assertIn(f"({hz:g} Hz, real time)", wait["description"])
        self.assertIn(f"0.1..{profile.max_timeout_s:g}, default {profile.default_wait_timeout_s:g}.",
                      wait["inputSchema"]["properties"]["timeout_s"]["description"])
        self.assertEqual(wait["inputSchema"]["properties"]["timeout_s"]["maximum"], profile.max_timeout_s)

    def test_measured_zero_point_publishes_presets_with_the_enforced_bounds(self):
        from ni_rua.robot.motion.grip_guard import preset_aperture_bounds
        profile = _so101_profile(empty_close_aperture_m=0.0045)
        contract = build(profile)
        self.assertEqual(contract.stage_schemas["gripper"]["properties"]["grip_guard_preset"]["enum"],
                         ["none", "non_empty", "thin_object", "rigid_object", "wide_object"])
        for name in ("non_empty", "thin_object", "rigid_object", "wide_object"):
            low, _ = preset_aperture_bounds(name, profile=profile)
            self.assertIn(f"{name} [min {_metres(low)}m]", contract.stage_field_help["grip_guard_preset"])
            self.assertIn(f"{name} [>={low * 1000:g}mm]", contract.tool["description"])
        self.assertIn("closed 0.0045m", contract.stage_field_help["min_aperture_m"])
        self.assertIn("can read 0.0045m when firmly held", contract.stage_field_help["min_aperture_m"])
        self.assertNotIn("named grip-guard presets", contract.tool["description"])


# ---- every bound enforced at the limit -------------------------------------------------------------------------
class _BoundProbe:
    """Mixin: one started Harness of ``embodiment``; every numeric robot_action bound of its profile is accepted at
    the limit (schema admission and ``compile_request`` with the backend validator) and rejected just beyond."""

    embodiment = None
    move = None

    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory(prefix="ni-rua-embodiment-schema-")
        cls.harness = golden.compile_harness(cls.embodiment, cls.directory.name)
        cls.profile = cls.harness.embodiment
        cls.arm = cls.profile.arms[0]

    @classmethod
    def tearDownClass(cls):
        cls.harness.store.close()
        close = getattr(cls.harness.robot, "close", None)
        if callable(close):
            close()
        cls.directory.cleanup()

    def request(self, stage=None, **batch):
        base = dict(self.move, arm=self.arm)
        return dict(observation_id=golden.OBSERVATION_ID, stages=[dict(base, **(stage or {}))], **batch)

    def admitted(self, request):
        self.harness.tool_registry.validate("robot_action", deepcopy(request), role="actor")

    def compiled(self, request):
        from ni_rua.runtime.actions.unified_actions import compile_request
        return compile_request(self.harness, deepcopy(request))

    def bounds(self):
        """``[(label, where, field, low, low_inclusive, high, extra stage fields)]`` from the profile."""
        p = self.profile
        full = p.gripper.full_open_aperture_m
        jaw = {"gripper_goal_normalized": 0.0}
        return [
            ("speed_scale", "stage", "speed_scale", p.speed_scale.min, True, p.speed_scale.max, {}),
            ("gripper_speed_scale", "stage", "gripper_speed_scale", p.gripper_speed_scale.min, True,
             p.gripper_speed_scale.max, jaw),
            ("position_tolerance_m", "stage", "position_tolerance_m", 0.0, False, p.max_position_tolerance_m, {}),
            ("orientation_tolerance_rad", "stage", "orientation_tolerance_rad", 0.0, False,
             p.max_orientation_tolerance_rad, {}),
            ("max_tracking_divergence_m", "stage", "max_tracking_divergence_m", *p.tracking_divergence_range_m[:1],
             True, p.tracking_divergence_range_m[1], {}),
            ("stage min_aperture_m", "stage", "min_aperture_m", 0.0, True, full, jaw),
            ("stage max_aperture_m", "stage", "max_aperture_m", 0.0, True, full, jaw),
            ("batch min_aperture_m", "batch", "min_aperture_m", 0.0, True, full, {}),
            ("batch position_tolerance_m", "batch", "position_tolerance_m", 0.0, False, p.max_position_tolerance_m, {}),
            ("timeout_s", "batch", "timeout_s", 0.1, True, p.max_timeout_s, {}),
        ]

    def probe(self, where, field, value, extra):
        return (self.request({**extra, field: value}) if where == "stage"
                else self.request(extra or None, **{field: value}))

    def test_every_bound_is_enforced_at_the_limit(self):
        for label, where, field, low, low_inclusive, high, extra in self.bounds():
            schema = (self.harness.tool_registry.spec("robot_action", role="actor").definition["inputSchema"]
                      ["properties"])
            published = schema[field] if where == "batch" else schema["stages"]["items"]["allOf"][0]["then"]["properties"][field]
            with self.subTest(label, check="published bound is the profile value"):
                self.assertEqual(published["maximum"], high)
                self.assertEqual(published.get("minimum" if low_inclusive else "exclusiveMinimum"), low)
            for value in (high, *((low,) if low_inclusive else ())):
                with self.subTest(label, value=value, check="accepted at the limit"):
                    request = self.probe(where, field, value, extra)
                    self.admitted(request)
                    self.compiled(request)
            for value in (high + DELTA, low - DELTA if low_inclusive else low):
                with self.subTest(label, value=value, check="rejected just beyond"):
                    with self.assertRaises(ValueError):
                        self.admitted(self.probe(where, field, value, extra))

    def test_dispatch_validates_against_the_run_schema(self):
        from jsonschema import ValidationError
        from ni_rua.runtime.actions.unified_actions import dispatch
        contract = build(self.profile)
        self.assertEqual(self.harness.tool_registry.spec("robot_action", role="actor").definition["inputSchema"],
                         contract.tool["inputSchema"])
        stage_fields = set(build(None).stage_schemas["move"]["properties"]) - set(contract.stage_schemas["move"]["properties"])
        batch_fields = set(build(None).tool["inputSchema"]["properties"]) - set(contract.tool["inputSchema"]["properties"])
        samples = dict(angles_deg=[0, 0, 0], grasp_position_m=[0.1, 0.0, 0.1], max_upward_dz_m=0.01,
                       joint_travel_preset="standard", clearance_height_m=0.1, obstacle_object_ids=["o1"])
        for field, value in samples.items():
            if field in stage_fields:
                request = self.request({field: value})
            elif field in batch_fields:
                request = self.request(**{field: value})
            else:
                continue
            with self.subTest(field=field), self.assertRaises(ValidationError):
                dispatch(self.harness, request, {"tool_call_id": "probe-" + field})


class X5BoundProbeTests(_BoundProbe, unittest.TestCase):
    embodiment = "x5"
    move = golden.MOVE


class So101BoundProbeTests(_BoundProbe, unittest.TestCase):
    embodiment = "so101"
    move = {"id": "m", "kind": "move", "position_m": [0.2, 0.0, 0.15], "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0]}

    def test_explicit_aperture_bound_is_the_backend_limit(self):
        """R4: the published aperture maximum is the bound the SO-101 contract enforces (the jaw's full opening)."""
        from ni_rua.robot.backends.so101 import contract
        self.assertEqual(contract.explicit_aperture_limit_m(self.harness.robot.deployment),
                         self.profile.gripper.full_open_aperture_m)

    def test_omitted_deadline_is_the_deployment_default(self):
        action = self.compiled(self.request())
        self.assertEqual(self.profile.default_action_timeout_s, 60.0)
        self.assertEqual(action["timeout_s"], 60.0)
        self.assertEqual({stage["action"]["timeout_s"] for stage in action["stages"]}, {60.0})

    def test_presets_published_are_the_presets_validated(self):
        contract = build(self.profile)
        gripper = contract.stage_schemas["gripper"]["properties"]
        self.assertEqual(gripper["grip_guard_preset"]["enum"][1:], ["non_empty", "thin_object", "rigid_object",
                                                                    "wide_object"])
        jaw = {"id": "j", "kind": "gripper", "arm": self.arm, "goal_normalized": 0.0}
        for preset in gripper["grip_guard_preset"]["enum"]:
            request = dict(observation_id=golden.OBSERVATION_ID, stages=[dict(jaw, grip_guard_preset=preset)])
            with self.subTest(preset=preset):
                self.admitted(request)
                self.compiled(request)
        for field in ("speed_preset", "tolerance_preset", "stall_guard_preset"):
            for preset in contract.stage_schemas["move"]["properties"][field]["enum"]:
                with self.subTest(field=field, preset=preset):
                    self.compiled(self.request({field: preset}))


# ---- other profile-driven defaults -----------------------------------------------------------------------------
class ProfileDefaultsTests(unittest.TestCase):
    def test_settle_defaults_to_the_overview_camera(self):
        from ni_rua.runtime.actions.stability import validate_options
        self.assertEqual(validate_options({"roi": [0, 0, 4, 4]})["camera_id"], "cam_head")
        self.assertEqual(validate_options({"roi": [0, 0, 4, 4]}, _so101_profile())["camera_id"], "fixed")
        self.assertEqual(validate_options({"roi": [0, 0, 4, 4], "camera_id": "wrist"}, _so101_profile())["camera_id"],
                         "wrist")

    def test_program_deadlines_and_wait_review_follow_the_profile(self):
        from ni_rua.runtime.programs.runner import ProgramRunner
        for profile, action, wait, words in ((None, 120.0, 30.0, "native action budget"),
                                             (_so101_profile(), 60.0, 30.0, "the world keeps moving")):
            runner = object.__new__(ProgramRunner)
            runner.services = SimpleNamespace(embodiment=profile)
            entry = dict(deadline=time.monotonic() + 250, cancel=threading.Event(), budget_lock=threading.RLock())
            with self.subTest(profile=getattr(profile, "robot_model_id", "default")):
                self.assertEqual(runner._action_timeout(dict(entry), "robot_action", {"mode": "preview"})["timeout_s"],
                                 action)
                self.assertEqual(runner._action_timeout(dict(entry), "robot_wait", {})["timeout_s"], wait)
                self.assertEqual(runner._action_timeout(dict(entry), "robot_wait", {"timeout_s": 7}), {"timeout_s": 7})
                captured = {}
                runner._await_review = lambda entry, gate, correlation: captured.update(gate=gate)
                runner.store = SimpleNamespace(put_artifact=lambda *args: "artifact:preview")
                entry["job_id"] = "job"
                with mock.patch("ni_rua.runtime.programs.runner.program_remaining", return_value=0):
                    with self.assertRaises(ValueError):
                        runner._review_wait_action(entry, {"observation_id": "o", "native_steps": 5}, {}, "c1", {})
                self.assertIn(words, captured["gate"]["preview"]["semantics"])

    def test_closed_jaw_downgrade_needs_the_gripper_zero_point(self):
        from ni_rua.runtime.actions.unified_actions import _closed_jaw_before_grasp_arms, _collect_plan_risks_and_notes
        stages = [{"id": "close", "kind": "gripper", "arm": "right", "goal_normalized": 0.0}]
        summary = {"status": "planned"}

        def harness(profile, aperture):
            return SimpleNamespace(embodiment=profile, capabilities={"arms": ["right"]}, config=None,
                                   latest={"grip_facts": {"right": {"measured_aperture_m": aperture}}})
        x5 = embodiments.arx_x5_profile()
        self.assertEqual(_closed_jaw_before_grasp_arms({"stages": stages}, harness(x5, 0.002)), ["right"])
        _, notes = _collect_plan_risks_and_notes(summary, {"stages": stages}, harness(x5, 0.002))
        self.assertEqual([n["code"] for n in notes], ["closed_jaw_before_grasp"])
        self.assertEqual(_closed_jaw_before_grasp_arms({"stages": stages}, harness(x5, 0.004)), [])
        unmeasured = _so101_profile()
        self.assertEqual(_closed_jaw_before_grasp_arms({"stages": stages}, harness(unmeasured, 0.004)), [])
        measured = _so101_profile(empty_close_aperture_m=0.0045)
        # 0.0045 m empty close + 3 mm: a 6 mm reading is closed on SO-101, open on X5.
        self.assertEqual(_closed_jaw_before_grasp_arms({"stages": stages}, harness(measured, 0.006)), ["right"])
        self.assertEqual(_closed_jaw_before_grasp_arms({"stages": stages}, harness(x5, 0.006)), [])

    def test_hotspot_advice_names_only_pose_fields_of_the_run_schema(self):
        """G3 (design 13 section 3.8): repeated-hotspot preview warnings carry typed suggested_adjustments (never
        natural-language preaching guidance) naming only pose fields offered by the embodiment's move schema."""
        from ni_rua.runtime.actions.action_state_ledger import attach_preview_ledger, record_execution_outcome

        def stage(position):
            return {"id": "m", "kind": "move", "arm": "right", "action": {"path": {"waypoints": [
                {"right": {"position_m": position, "quaternion_xyzw": [0, 0, 0, 1]}}]}}}
        failed = {"status": "stopped", "batch_status": "stopped", "stop_reason": "endpoint_not_reached",
                  "failed_stage_id": "m", "completed_stage_ids": []}
        x5_fields = ["position_m", "grasp_position_m", "angles_deg", "contact_region"]
        so101_fields = ["position_m", "quaternion_xyzw", "grasp_centre_m", "fixed_jaw_yaw_deg"]
        for profile, expected in ((None, x5_fields), (embodiments.arx_x5_profile(), x5_fields), (_so101_profile(), so101_fields)):
            harness = SimpleNamespace(identity={"run_id": "r", "episode_id": "e", "env_instance_id": "i",
                                                "env_generation": 0})
            if profile is not None:
                harness.embodiment = profile
            record_execution_outcome(harness, "plan-1", {"stages": [stage([0.10, -0.30, 0.82])]}, failed)
            summary = attach_preview_ledger(harness, {"stages": [stage([0.105, -0.305, 0.822])]}, {"status": "planned"})
            hotspot = summary["repeat_failure_warning"][0]
            move_props = set(build(profile).stage_schemas["move"]["properties"])
            with self.subTest(profile=getattr(profile, "robot_model_id", "no profile")):
                self.assertNotIn("guidance", hotspot)
                self.assertEqual(hotspot["suggested_adjustments"], expected)
                self.assertLessEqual(set(hotspot["suggested_adjustments"]), move_props)
                if profile is not None and "grasp_angles" not in profile.pose_inputs:
                    self.assertNotIn("angles_deg", hotspot["suggested_adjustments"])
                    self.assertNotIn("contact_region", hotspot["suggested_adjustments"])
                self.assertIsNone(summary["episode_ledger"]["recent_outcomes"][-1]["executed_native_steps"])

    def test_default_robot_names_stay_the_x5_rendering(self):
        from ni_rua.runtime.tools import definitions
        self.assertIs(robot_action_schema.ROBOT_ACTION_TOOL, build(None).tool)
        self.assertIs(robot_action_schema._STAGE_SCHEMAS, build(embodiments.ARX_X5_PROFILE).stage_schemas)
        self.assertEqual(definitions.GEOMETRY_REQUEST_DESCRIPTIONS["frame_id"].count("env_origin"), 1)
        self.assertEqual(definitions.PUBLIC_EXAMPLES["robot_contact_overlay"][0]["arms"], ["left"])
        so101 = definitions.public_examples(_so101_profile())["robot_contact_overlay"][0]
        self.assertEqual(so101, {"observation_id": "obs_1", "camera_id": "fixed", "arms": ["right"]})


# ---- the generated text reaches the native request -------------------------------------------------------------
def _so101_capture(backend):
    """``support.prompt_capture.capture`` of a SO-101 run: the fake SO-101 parent backend (``embodiment_golden``)
    instead of the X5 fixture robot, the shipped SO-101 gateway features, the backend's own agent configuration."""
    from ni_rua.core import config as config_module
    from support import prompt_capture
    directory = tempfile.TemporaryDirectory(prefix="ni-rua-so101-native-")
    root = Path(directory.name).resolve()
    deployment = golden.so101_deployment(root / "deployment")
    so101 = {"backend": {"environment": "so101_real", "robot_model_id": "so101", "deployment": str(deployment)},
             "features": {"observation_profile": "rgb_proprio_calibrated_v1", "motion_review": False,
                          "held_perception": False, "external_perception": False}}
    original = config_module.Config.from_dict.__func__

    def from_dict(cls, data, *args, **kwargs):
        return original(cls, golden._merge(data, so101), *args, **kwargs)

    class Robot(golden._so101_robot_class()):
        def observe(self):  # the capture starts a benchmark episode, which needs the public task text
            return dict(super().observe(), instruction=prompt_capture.TASK)

    def robot(path):
        config = golden.load_config(golden.SO101_CONFIGS["so101_gateway"],
                                    overrides={"backend": {"deployment": str(deployment)}})
        return Robot(config=config, run_dir=Path(path))
    try:
        with mock.patch.object(config_module.Config, "from_dict", classmethod(from_dict)), \
                mock.patch.object(prompt_capture, "_fixture_robot", robot):
            return prompt_capture.capture(backend)
    finally:
        directory.cleanup()


class _So101NativePayload:
    backend = None

    @classmethod
    def setUpClass(cls):
        from support.prompt_capture import analyze, published_tools
        cls.captured = _so101_capture(cls.backend)
        cls.facts = analyze(cls.captured)
        prefix = cls.captured.tool_prefix
        cls.published = {tool["name"]: tool for tool in published_tools(cls.captured)}
        cls.action = cls.published[prefix + "robot_action"]

    def test_registry_catalogue_is_the_so101_rendering(self):
        action = next(tool for tool in self.captured.tools if tool["name"] == "robot_action")
        self.assertEqual(action["inputSchema"], build(_so101_profile()).tool["inputSchema"])

    def test_generated_text_reaches_the_model_request(self):
        for tool in self.facts["tools"]:
            if tool["present"]:
                with self.subTest(tool=tool["name"]):
                    self.assertTrue(tool["description_identical"], tool)
        description = self.action["description"]
        for fragment in ("gripper_frame_link", "so101_base", '"arm":"right"', "[0..0.1372m]",
                         "Not available on this robot, so absent from this schema:"):
            self.assertIn(fragment, description)
        for term in X5_TERMS:
            self.assertIsNone(re.search(term, description), term)


def _requirement(name):
    from support import requirements
    return getattr(requirements, name)


@_requirement("requires_codex_binary")
class So101CodexNativePayloadTests(_So101NativePayload, unittest.TestCase):
    backend = "codex"


@_requirement("requires_claude_cli")
class So101ClaudeCodeNativePayloadTests(_So101NativePayload, unittest.TestCase):
    backend = "claude_code"


def _opencode_problem():
    import shutil
    from support.requirements import opencode_problem
    return opencode_problem() or (None if shutil.which("rg") else "ripgrep (rg) is not installed")


@unittest.skipIf(_opencode_problem() is not None, f"pinned OpenCode runtime unavailable: {_opencode_problem()}")
class So101OpenCodeNativePayloadTests(_So101NativePayload, unittest.TestCase):
    backend = "opencode_anthropic"


if __name__ == "__main__":
    unittest.main()
