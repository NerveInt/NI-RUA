"""Scene contracts, pose-axis overlays, scene store invariants, and adaptive view selection."""
NI_RUA_TEST_TIER = "fast"

import unittest
import numpy as np

from ni_rua.perception.overlays.pose_axis_overlay import draw_pose_axis_frame
from ni_rua.perception.overlays.trajectory_overlay import render_trajectory_overlay


class PoseAxisOverlayTest(unittest.TestCase):
    def test_trajectory_overlay_renders_pose_axis_frame_while_keeping_panel_meta_scalar(self):
        observation = dict(world_kind='observed', observation_id='obs-1', sim_step=10, step_domain='control_step')
        frame = dict(camera_id='cam_head', frame_id='head', sim_step=10, step_domain='control_step',
                     rgb_ref='rgb-1', pixel_transform='identity')
        camera = dict(
            observation=observation,
            frame=frame,
            width=320,
            height=240,
            K=np.array([[250.0, 0.0, 160.0], [0.0, 250.0, 120.0], [0.0, 0.0, 1.0]]),
            D=np.zeros(5),
            T=np.eye(4),
            reference_frame_id='env_origin',
            pose_source='fixed_calibration',
        )
        image = np.zeros((240, 320, 3), dtype=np.uint8)
        arm_data = dict(
            phase='move',
            link_positions_m=dict(link5=[0.0, -0.1, 1.5], link6=[0.0, 0.0, 1.5]),
            link6_pose=dict(
                position_m=[0.0, 0.0, 1.5],
                rotation_matrix=np.eye(3).tolist(),
            ),
        )
        pose_frame = draw_pose_axis_frame(image.copy(), camera, 'left', arm_data, (255, 160, 48), 320, 240)
        self.assertEqual(pose_frame['grasp_center_m'], [0.1226, 0.0, 1.5])
        self.assertEqual(pose_frame['pad_center_m'], [0.1501, 0.0, 1.5])
        self.assertEqual(pose_frame['fingertip_m'], [0.1576, 0.0, 1.5])
        self.assertEqual(pose_frame['approach_axis_world'], [1.0, 0.0, 0.0])
        self.assertEqual(pose_frame['jaw_closing_axis_world'], [0.0, 1.0, 0.0])
        trajectory = dict(
            source_kind='candidate_command_trajectory_not_measurement',
            frame_id='env_origin',
            label='plan-1',
            robot_model=dict(joints=[dict(parent='link5', child='link6')]),
            samples=[dict(native_step=0, arms=dict(left=arm_data))],
        )
        rendered, meta = render_trajectory_overlay(camera, image, trajectory, native_steps=[0])
        self.assertEqual(rendered.shape, (240, 320, 3))
        arm_meta = meta['panels'][0]['arms']['left']
        self.assertNotIn('pose_axis_frame', arm_meta)


from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from ni_rua.core.contracts import digest, validate_action
from ni_rua.perception.scene.store import SceneStore


from support.paths import REPO_ROOT

FIXTURE_ROOT = REPO_ROOT / "contracts" / "fixtures"


class SceneCheck(unittest.TestCase):
    def setUp(self):
        self.directory = self.enterContext(tempfile.TemporaryDirectory())
        self.store = SceneStore(self.directory)
        self.fixture = json.loads((FIXTURE_ROOT / "scene_plane_box.json").read_text())
        self.identity = self.fixture["observations"][0]
        for observation in self.fixture["observations"]:
            self.store.import_observation(observation)
        self.scene = self.store.apply_patch(self.fixture["patches"][0])

    def test_analytic_fixture_and_unknown_space(self):
        fixture = deepcopy(self.fixture)
        claimed = fixture.pop("fixture_digest")
        self.assertEqual(claimed, digest(fixture))
        for artifact in fixture["artifacts"]:
            content = (FIXTURE_ROOT / artifact["storage_key"]).read_bytes()
            self.assertEqual(artifact["sha256"], hashlib.sha256(content).hexdigest())
            self.assertEqual(artifact["byte_size"], len(content))
        results = [self.store.query(query, 1, query) for query in fixture["queries"]]
        self.assertEqual([item["status"] for item in results], fixture["expected"]["query_statuses"])
        self.assertEqual(list(results[0]["value"]["point_m"]), fixture["expected"]["ray_point_m"])
        self.assertAlmostEqual(results[1]["value"]["distance_m"], fixture["expected"]["box_distance_m"])
        self.assertTrue(all(item["world_kind"] == "internal" and item["data_origin"] == "synthetic_fixture" for item in results))
        self.assertIsNone(results[2]["value"])
        self.assertTrue(results[2]["unknown_regions"])
        parallel = dict(kind="ray_surface", entity_id="table", origin_m=[0, 0, 1], direction=[1, 0, 0])
        self.assertEqual(self.store.query(self.identity, 1, parallel)["status"], "unknown")
        box = dict(kind="ray_surface", entity_id="box", origin_m=[0.3, 0.3, 1], direction=[0, 0, -2])
        self.assertAlmostEqual(self.store.query(self.identity, 1, box)["value"]["point_m"][2], 0.2)

    def test_snapshot_stays_frozen_and_cas_conflicts(self):
        before = self.store.snapshot(self.identity, 1, "obs-scene-0")
        frozen_digest = digest(before)
        patch = deepcopy(self.fixture["patches"][0])
        patch["base_scene_version"] = 1
        patch["observation_ids"] = ["obs-scene-1"]
        patch["operations"] = [patch["operations"][1]]
        patch["operations"][0]["geometry"] = dict(kind="box", min_m=[0.6, 0.2, 0], max_m=[0.8, 0.4, 0.2])
        self.store.apply_patch(patch)
        after = self.store.snapshot(self.identity, 1, "obs-scene-0")
        self.assertEqual(digest(after), frozen_digest)
        self.assertEqual(after["scene"]["entities"]["box"]["geometry"]["min_m"][0], 0.2)
        self.assertEqual(self.store.head(self.identity)["scene_version"], 2)
        with self.assertRaisesRegex(ValueError, "scene_version_conflict"):
            self.store.apply_patch(patch)
        changed = deepcopy(self.identity)
        changed["sim_step"] = 2
        with self.assertRaisesRegex(ValueError, "immutable_content_conflict"):
            self.store.import_observation(changed)
        reloaded = SceneStore(self.directory)
        self.assertEqual(reloaded.version(self.identity, 1), self.scene)

    def test_lost_entities_and_bad_queries(self):
        patch = deepcopy(self.fixture["patches"][0])
        patch["base_scene_version"] = 1
        patch["operations"] = [dict(op="mark_lost", entity_id="box", source_kind="observation_inference", reason="Occluded; current location unknown.")]
        self.store.apply_patch(patch)
        self.assertEqual(self.store.query(self.identity, 2, dict(kind="distance", entity_id="box", point_m=[0, 0, 0]))["status"], "unknown")
        bad = dict(kind="ray_surface", entity_id="table", origin_m=[0, 0, 1], direction=[0, 0, 0])
        self.assertEqual(self.store.query(self.identity, 1, bad)["status"], "invalid_input")
        wrong_frame = dict(kind="distance", entity_id="box", point_m=[0, 0, 0], frame_id="camera_optical")
        self.assertEqual(self.store.query(self.identity, 1, wrong_frame)["status"], "invalid_input")
        self.assertEqual(self.store.query(self.identity, 1, dict(kind="preview_ee"))["reason"], "unsupported_capability")
        missing_pt = dict(kind="distance", entity_id="box")
        self.assertEqual(self.store.query(self.identity, 1, missing_pt)["status"], "invalid_input")
        with self.assertRaisesRegex(ValueError, "unknown_snapshot"):
            self.store.read_snapshot(self.identity, "snapshot-" + ("a" * 64))
        from unittest import mock
        with mock.patch.object(
            self.store,
            "version",
            return_value={
                **self.scene,
                "entities": {"box": {**self.scene["entities"]["box"], "geometry": {"kind": "box"}}},
            },
        ):
            with self.assertRaises(KeyError):
                self.store.query(self.identity, 1, dict(kind="distance", entity_id="box", point_m=[0, 0, 0]))

    def test_entity_evidence_subset_owns_its_timestamp(self):
        patch = deepcopy(self.fixture["patches"][0])
        patch.update(base_scene_version=1, observation_ids=["obs-scene-0", "obs-scene-1"])
        patch["operations"] = [operation for operation in patch["operations"] if operation["op"] == "upsert_geometry"]
        for operation in patch["operations"]:
            operation["observation_ids"] = ["obs-scene-0" if operation["entity_id"] == "box" else "obs-scene-1"]
        scene = self.store.apply_patch(patch)
        for name, entity in scene["entities"].items():
            observation_id = "obs-scene-0" if name == "box" else "obs-scene-1"
            observation = self.store.observation(self.identity, observation_id)
            self.assertEqual(entity["observation_ids"], [observation_id])
            self.assertEqual(entity["last_observation_id"], observation_id)
            for field in ("step_domain", "clock_domain"):
                self.assertEqual(entity[field], observation[field])
            self.assertEqual(entity["last_sim_step"], observation["sim_step"])

    def test_newer_patch_observation_cannot_revive_entity_from_stale_evidence(self):
        patch = deepcopy(self.fixture["patches"][0])
        patch.update(base_scene_version=1, observation_ids=["obs-scene-1"], operations=[dict(
            op="mark_dirty", entity_id="box", source_kind="observation_inference", reason="Object moved")])
        self.store.apply_patch(patch)
        operation = deepcopy(next(item for item in self.fixture["patches"][0]["operations"]
            if item.get("entity_id") == "box"))
        operation.update(observation_ids=["obs-scene-0"], source_kind="observation_inference", assumptions=[])
        patch.update(base_scene_version=2, observation_ids=["obs-scene-0", "obs-scene-1"], operations=[operation])
        with self.assertRaisesRegex(ValueError, "late_observation_cannot_overwrite_entity"):
            self.store.apply_patch(patch)
        self.assertEqual(self.store.head(self.identity)["scene_version"], 2)
        self.assertEqual(self.store.query(self.identity, 2, dict(kind="entity_pose", entity_id="box"))["status"], "unknown")

    def test_entity_evidence_requires_its_own_comparable_dated_clock(self):
        for change, reason in ((dict(sim_step=None), "undated_observation_cannot_overwrite_entity"),
                (dict(clock_domain="other-clock"), "incomparable_entity_clock"),
                (dict(step_domain="physics_step"), "incomparable_entity_clock")):
            with self.subTest(change=change):
                observation = dict(self.fixture["observations"][0], **change,
                    observation_id="different-" + next(iter(change)))
                self.store.import_observation(observation)
                patch = deepcopy(self.fixture["patches"][0])
                operation = next(item for item in patch["operations"] if item.get("entity_id") == "box")
                operation["observation_ids"] = [observation["observation_id"]]
                patch.update(base_scene_version=1, observation_ids=[observation["observation_id"], "obs-scene-1"],
                    operations=[operation])
                with self.assertRaisesRegex(ValueError, reason):
                    self.store.apply_patch(patch)
                if observation["sim_step"] is not None:
                    operation["observation_ids"].append("obs-scene-1")
                    with self.assertRaisesRegex(ValueError, "incomparable_entity_clock"):
                        self.store.apply_patch(patch)
        self.assertEqual(self.store.head(self.identity)["scene_version"], 1)

    def test_internal_results_have_no_observed_execution_permission(self):
        result = self.store.rollout(dict(snapshot_id="fixture-snapshot", world_kind="internal"))
        self.assertEqual(result["status"], "unsupported")
        self.assertFalse(result["executed"])
        with self.assertRaisesRegex(ValueError, "no observed-world execution permission"):
            self.store.rollout(dict(world_kind="observed"))
        with self.assertRaisesRegex(ValueError, "no observed-world execution permission"):
            self.store.rollout(dict(submit_to_observed=True))
        # The shared gateway validator rejects an internal output before motion validation.
        candidate = self.store.query(self.identity, 1, self.fixture["queries"][0])
        with self.assertRaisesRegex(ValueError, "internal_action_cannot_execute_observed_world"):
            validate_action(candidate, self.identity, "obs-scene-0", {})
        capabilities = self.store.capabilities()
        self.assertFalse(capabilities["worker_ready"])
        self.assertEqual(capabilities["physics"], "unsupported")


from copy import deepcopy
import json
from pathlib import Path
import unittest

from jsonschema import Draft202012Validator

from ni_rua.perception.scene.scene_contracts import (SCHEMAS, GET_CONTRACTS, POST_CONTRACTS,
    GEOMETRY_KINDS, openapi, validate_request, validate_response)
from ni_rua.core.tool_errors import ToolInputError
from support.geometry_fixture import IDENTITY


class SceneContractsCheck(unittest.TestCase):
    def test_all_routes_publish_real_schemas_and_resolvable_references(self):
        document = openapi()
        for name, schema in SCHEMAS.items():
            with self.subTest(schema=name):
                Draft202012Validator.check_schema(schema)
        def check_refs(value):
            if isinstance(value, dict):
                if "$ref" in value:
                    self.assertTrue(value["$ref"].startswith("#/components/schemas/"))
                    self.assertIn(value["$ref"].split("/")[-1], document["components"]["schemas"])
                for part in value.values(): check_refs(part)
            elif isinstance(value, list):
                for part in value: check_refs(part)
        check_refs(document)
        for route in GET_CONTRACTS:
            self.assertIn("content", document["paths"]["/v1/" + route]["get"]["responses"]["200"])
        for route in POST_CONTRACTS:
            operation = document["paths"]["/v1/" + route]["post"]
            self.assertIn("schema", operation["requestBody"]["content"]["application/json"])
        self.assertIn("scene_version", SCHEMAS["SceneQuery"]["oneOf"][0]["required"])
        self.assertIn("Snapshot", document["components"]["schemas"])

    def test_actual_geometry_fields_and_mutually_exclusive_requests(self):
        base = dict(IDENTITY, camera_id="head", observation_id="obs")
        values = dict(fk={}, camera_pose={}, pixel_ray=dict(pixel=[2, 3]),
            pixel_plane=dict(pixel=[2, 3], plane_origin_m=[0, 0, 1], plane_normal=[0, 0, 1],
                             plane_origin_alt_m=[0, 0, 2]),
            ray_surface=dict(pixel=[2, 3], scene_version=1, entity_id="table"),
            project_points=dict(points_m=[[0, 0, 1]]), project_geometry=dict(scene_version=1),
            preview_ee=dict(action=dict(frame_id="env_origin", mode="absolute", path=dict(
                waypoints=[dict(left=dict(position_m=[0, 0, 1], quaternion_xyzw=[0, 0, 0, 1]))]))),
            sparse_track={}, triangulate={}, build_scene=dict(base_scene_version=0),
            alignment=dict(scene_version=1))
        self.assertEqual(set(values) | {"help"}, set(GEOMETRY_KINDS))
        for kind, fields in values.items():
            with self.subTest(kind=kind):
                validate_request("geometry/query", dict(base, kind=kind, **fields))
        validate_request("geometry/query", dict(IDENTITY, kind="help"))
        for changes in (dict(pixel=[True, 3]), dict(pixel=[float("nan"), 2]),
                        dict(pixel=[2, 3], typo=True), dict(pixel=[2, 3], observation_ids=["obs"])):
            with self.subTest(changes=changes), self.assertRaises(ToolInputError):
                validate_request("geometry/query", dict(base, kind="pixel_ray", **changes))
        with self.assertRaises(ToolInputError) as error:
            validate_request("scene/query", dict(IDENTITY, kind="distance", entity_id="box", point_m=[0, 0, 0]))
        self.assertEqual(error.exception.field_path, "/scene_version")

    def test_fixture_requests_and_historical_response_extensions(self):
        fixture = json.loads((FIXTURE_ROOT / "scene_plane_box.json").read_text())
        for observation in fixture["observations"]:
            validate_request("observations", observation)
        for patch in fixture["patches"]:
            validate_request("scene/patches", patch)
        for query in fixture["queries"]:
            validate_request("scene/query", query)
        # Existing snapshot/cancel callers omit schema_version; published schema
        # explicitly permits that omission rather than inventing a new client rule.
        validate_request("snapshots", dict(IDENTITY, scene_version=1, anchor_observation_id="obs"))
        validate_request("jobs/{identifier}/cancel", dict(IDENTITY, reason="test"))
        head = dict(IDENTITY, scene_version=0, world_kind="internal", future_evidence={"exact": .123456789})
        original = deepcopy(head)
        validate_response("scene/head", head)
        self.assertEqual(head, original)


from copy import deepcopy
import io
import json
import unittest

import numpy as np
from PIL import Image

from ni_rua.perception.view_selection import baseline_record, select_observation_views


CAMERAS = ("cam_head", "cam_left_wrist", "cam_right_wrist")


def encoded(array):
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return buffer.getvalue()


def fixture(size=(96, 72)):
    bundle = dict(run_id="run", episode_id="episode", env_instance_id="env", env_generation=1,
                  world_kind="observed", observation_id="obs-before", calibration_id="calibration",
                  cameras=[dict(camera_id=camera, rgb_ref="rgb-" + camera, width=size[0], height=size[1],
                                calibration_ref="calibration-ref", pixel_transform="identity") for camera in CAMERAS])
    image = encoded(np.full((size[1], size[0], 3), 100, dtype=np.uint8))
    images = {camera: image for camera in CAMERAS}
    baselines = {camera: baseline_record(bundle, camera, image) for camera in CAMERAS}
    bundle["observation_id"] = "obs-after"
    return bundle, images, baselines


def select(bundle, images, baselines, **kwargs):
    options = dict(active_arms=["left"], receipt_status="completed",
                   calibration_valid={camera: True for camera in CAMERAS})
    options.update(kwargs)
    return select_observation_views(bundle, images, baselines, **options)


def right_decision(result):
    return next(item for item in result["decisions"] if item["camera_id"] == "cam_right_wrist")


class ViewSelectionTests(unittest.TestCase):
    def test_all_unchanged_views_are_retrievable_with_original_camera_order(self):
        bundle, images, baselines = fixture()
        bundle["cameras"] = [bundle["cameras"][2], bundle["cameras"][0], bundle["cameras"][1]]
        result = select(bundle, images, baselines)
        self.assertEqual(result["selected_camera_ids"], [])
        self.assertEqual(result["skipped_camera_ids"], ["cam_right_wrist", "cam_head", "cam_left_wrist"])
        self.assertEqual([item["camera_id"] for item in result["decisions"]], [camera["camera_id"] for camera in bundle["cameras"]])
        decision = right_decision(result)
        self.assertEqual(decision["observation_id"], "obs-after")
        self.assertEqual(decision["baseline_observation_id"], "obs-before")
        self.assertEqual(decision["reason"], "view_below_change_thresholds")
        self.assertEqual(decision["retrieval"], dict(tool="history_view_observation",
            arguments=dict(observation_id="obs-after", camera_id="cam_right_wrist")))
        self.assertEqual(decision["retrieval_instruction"],
            'history_view_observation({"observation_id":"obs-after","camera_id":"cam_right_wrist"})')
        self.assertEqual(result["settings"]["thumbnail_size"], [96, 72])
        json.dumps(result, allow_nan=False)

    def test_participation_is_telemetry_and_all_seen_unchanged_views_can_be_omitted(self):
        bundle, images, baselines = fixture()
        for arms in (["right"], ["right", "left"], None, [], "left", ["new_arm"], [None]):
            with self.subTest(arms=arms):
                result = select(bundle, images, baselines, active_arms=arms)
                self.assertEqual(result["selected_camera_ids"], [])
                self.assertEqual(result["skipped_camera_ids"], list(CAMERAS))
                self.assertTrue(all(item["reason"] == "view_below_change_thresholds"
                                    for item in result["decisions"]))

    def test_changed_head_and_participating_wrist_are_included_independently(self):
        for camera in CAMERAS:
            bundle, images, baselines = fixture()
            images[camera] = encoded(np.full((72, 96, 3), 101, dtype=np.uint8))
            with self.subTest(camera=camera):
                result = select(bundle, images, baselines, active_arms=["left", "right"])
                self.assertEqual(result["selected_camera_ids"], [camera])
                self.assertEqual(result["skipped_camera_ids"], [other for other in CAMERAS if other != camera])

    def test_successful_ordinary_template_is_eligible_but_handoff_is_full(self):
        bundle, images, baselines = fixture()
        result = select(bundle, images, baselines, context="template", active_arms=None)
        self.assertTrue(result["eligible"])
        self.assertEqual(result["selected_camera_ids"], [])
        result = select(bundle, images, baselines, context="template", force_full=True,
                        full_reason="handoff_verification")
        self.assertEqual(result["selected_camera_ids"], list(CAMERAS))
        self.assertEqual(result["force_full_reason"], "handoff_verification")

    def test_unknown_camera_role_is_included(self):
        bundle, images, baselines = fixture()
        bundle["cameras"].append({**bundle["cameras"][0], "camera_id": "new_camera"})
        result = select(bundle, images, baselines)
        self.assertIn("new_camera", result["selected_camera_ids"])
        self.assertEqual(result["decisions"][-1]["reason"], "unknown_camera_role")

    def test_initial_explicit_history_wait_checkpoint_and_unknown_contexts_are_full(self):
        bundle, images, baselines = fixture()
        for context in ("initial", "explicit_observe", "history", "wait", "observation_checkpoint",
                        "handover", "template_boundary", "unknown", None):
            with self.subTest(context=context):
                result = select(bundle, images, baselines, context=context)
                self.assertEqual(result["selected_camera_ids"], list(CAMERAS))
                self.assertFalse(result["eligible"])
        result = select(bundle, images, baselines, force_full=True)
        self.assertEqual(result["force_full_reason"], "forced_full_context")
        self.assertEqual(result["skipped_camera_ids"], [])

    def test_failure_unknown_status_and_reason_are_full(self):
        bundle, images, baselines = fixture()
        variants = [dict(receipt_status=status) for status in (None, "failed", "stopped", "unknown", "rejected")]
        variants += [dict(receipt_reason=reason) for reason in ("observation_checkpoint", "tracking_error", "new_reason")]
        for options in variants:
            with self.subTest(options=options):
                result = select(bundle, images, baselines, **options)
                self.assertEqual(result["selected_camera_ids"], list(CAMERAS))
                self.assertEqual(result["skipped_camera_ids"], [])

    def test_global_change_in_inactive_view_is_included(self):
        bundle, images, baselines = fixture()
        images["cam_right_wrist"] = encoded(np.full((72, 96, 3), 101, dtype=np.uint8))
        result = select(bundle, images, baselines)
        self.assertEqual(result["selected_camera_ids"], ["cam_right_wrist"])
        self.assertEqual(right_decision(result)["comparison"]["material_change_triggers"], ["mean_abs_rgb_delta"])

    def test_small_changed_area_in_inactive_view_is_included(self):
        bundle, images, baselines = fixture()
        pixels = np.full((72, 96, 3), 100, dtype=np.uint8)
        pixels[0, :10] = 113
        images["cam_right_wrist"] = encoded(pixels)
        decision = right_decision(select(bundle, images, baselines))
        self.assertTrue(decision["include"])
        self.assertEqual(decision["comparison"]["material_change_triggers"], ["changed_area_fraction"])

    def test_one_high_contrast_original_pixel_survives_thumbnail_averaging(self):
        bundle, images, baselines = fixture(size=(640, 480))
        pixels = np.full((480, 640, 3), 100, dtype=np.uint8)
        pixels[201, 300, 0] = 132
        images["cam_right_wrist"] = encoded(pixels)
        decision = right_decision(select(bundle, images, baselines))
        self.assertTrue(decision["include"])
        self.assertEqual(decision["comparison"]["material_change_triggers"], ["max_channel_delta"])
        self.assertEqual(decision["comparison"]["max_channel_delta_resolution"], "original")

    def test_accumulated_changes_compare_to_last_delivered_not_skipped_capture(self):
        bundle, images, baselines = fixture()
        original = deepcopy(baselines)
        for value in range(1, 5):
            bundle["observation_id"] = "capture-" + str(value)
            pixels = np.full((72, 96, 3), 100, dtype=np.uint8)
            pixels[:18] += value
            images["cam_right_wrist"] = encoded(pixels)
            decision = right_decision(select(bundle, images, baselines))
            self.assertEqual(decision["include"], value == 4)
            self.assertEqual(decision["baseline_observation_id"], "obs-before")
            self.assertEqual(decision["comparison"]["mean_abs_rgb_delta"], value / 4)
            self.assertEqual(baselines, original)
        # Only after the last result is actually delivered does the caller advance it.
        baselines["cam_right_wrist"] = baseline_record(bundle, "cam_right_wrist", images["cam_right_wrist"])
        bundle["observation_id"] = "capture-5"
        decision = right_decision(select(bundle, images, baselines))
        self.assertFalse(decision["include"])
        self.assertEqual(decision["baseline_observation_id"], "capture-4")

    def test_missing_corrupt_or_wrong_size_images_always_include_even_if_identical(self):
        for data in (None, b"", b"not-an-image", encoded(np.zeros((12, 10, 3), dtype=np.uint8))):
            for corrupt_baseline in (False, True):
                bundle, images, baselines = fixture()
                images["cam_right_wrist"] = data
                if corrupt_baseline:
                    baselines["cam_right_wrist"]["rgb_bytes"] = data
                with self.subTest(data_type=type(data).__name__, corrupt_baseline=corrupt_baseline):
                    decision = right_decision(select(bundle, images, baselines))
                    self.assertTrue(decision["include"])
                    self.assertEqual(decision["reason"], "image_comparison_unavailable")

    def test_missing_changed_or_unvalidated_calibration_includes(self):
        for validity in (None, True, False, {}, {"cam_right_wrist": False}, {"cam_right_wrist": 1}):
            bundle, images, baselines = fixture()
            with self.subTest(validity=validity):
                decision = right_decision(select(bundle, images, baselines, calibration_valid=validity))
                self.assertTrue(decision["include"])
                self.assertEqual(decision["reason"], "calibration_not_validated")
        for target, key, value in (("bundle", "calibration_id", None), ("camera", "calibration_ref", None),
                                   ("bundle", "calibration_id", "new"), ("camera", "calibration_ref", "new")):
            bundle, images, baselines = fixture()
            (bundle if target == "bundle" else bundle["cameras"][2])[key] = value
            with self.subTest(target=target, key=key, value=value):
                decision = right_decision(select(bundle, images, baselines))
                self.assertTrue(decision["include"])
                self.assertEqual(decision["reason"], "calibration_missing" if value is None else "calibration_changed")

    def test_missing_delivery_identity_provenance_or_original_transform_includes(self):
        variants = [("kind", "actor_program"), ("camera_id", "cam_left_wrist"),
                    ("observation_id", None), ("rgb_ref", None), ("world_kind", "internal"),
                    ("pixel_transform", "crop_resize_lanczos"), ("identity", {}), ("width", 97)]
        for key, value in variants:
            bundle, images, baselines = fixture()
            baselines["cam_right_wrist"][key] = value
            with self.subTest(key=key):
                self.assertTrue(right_decision(select(bundle, images, baselines))["include"])
        for key, value in (("run_id", "different"), ("episode_id", "different"),
                           ("env_instance_id", "different"), ("env_generation", 2)):
            bundle, images, baselines = fixture()
            baselines["cam_right_wrist"]["identity"][key] = value
            self.assertEqual(right_decision(select(bundle, images, baselines))["reason"], "baseline_identity_changed")
        bundle, images, baselines = fixture()
        self.assertEqual(right_decision(select(bundle, images, {}))["reason"], "no_model_delivery_baseline")
        bundle["cameras"][2]["pixel_transform"] = "crop"
        self.assertTrue(right_decision(select(bundle, images, baselines))["include"])

    def test_unknown_observation_identity_or_internal_observation_is_full(self):
        for key, value in (("env_generation", None), ("run_id", ""), ("observation_id", None), ("world_kind", "internal")):
            bundle, images, baselines = fixture()
            bundle[key] = value
            result = select(bundle, images, baselines)
            self.assertFalse(result["eligible"])
            self.assertEqual(result["selected_camera_ids"], list(CAMERAS))

    def test_invalid_settings_include_and_valid_settings_are_explicit(self):
        bundle, images, baselines = fixture()
        for settings in ({"unknown": 1}, {"thumbnail_size": [0, 72]},
                         {"changed_area_fraction_threshold": float("nan")},
                         {"max_channel_delta_threshold": float("inf")},
                         {"mean_abs_rgb_threshold": True}, {"mean_abs_rgb_threshold": 0}):
            with self.subTest(settings=settings):
                result = select(bundle, images, baselines, settings=settings)
                self.assertEqual(result["selected_camera_ids"], list(CAMERAS))
                self.assertEqual(result["force_full_reason"], "invalid_settings")
                json.dumps(result, allow_nan=False)
        settings = dict(thumbnail_size=(48, 36), max_channel_delta_threshold=20)
        result = select(bundle, images, baselines, settings=settings)
        self.assertEqual(result["settings"]["thumbnail_size"], [48, 36])
        self.assertEqual(result["settings"]["max_channel_delta_threshold"], 20)
        self.assertEqual(settings["thumbnail_size"], (48, 36))

    def test_selector_does_not_mutate_inputs_and_metadata_contains_no_bytes(self):
        bundle, images, baselines = fixture()
        before = deepcopy((bundle, images, baselines))
        first = select(bundle, images, baselines)
        second = select(bundle, images, baselines)
        self.assertEqual(first, second)
        self.assertEqual((bundle, images, baselines), before)
        encoded_metadata = json.dumps(first, allow_nan=False)
        self.assertNotIn("rgb_bytes", encoded_metadata)
        first["settings"]["thumbnail_size"][0] = 1
        self.assertEqual(select(bundle, images, baselines)["settings"]["thumbnail_size"], [96, 72])


if __name__ == "__main__":
    unittest.main()
