"""Control-information preservation for the opt-in actor JSON projection."""

from copy import deepcopy
import json
import math
import re
import struct
from types import SimpleNamespace
import unittest
import unittest.mock

from ni_rua.core.contracts import INSPECT_STEP_LIMIT, canonical
from ni_rua.core.presentation.actor_payload import compact_action_result, compact_observation
from ni_rua.core.presentation.reply_budget import fit_reply, fit_reply_report, reply_cost_tokens, text_cost_tokens
from ni_rua.core.presentation import reply_budget, reply_compaction, reply_encoding
from support.reply_defaults import (
    compact_encoding_loss, decode_compact_value, float32 as _float32, resolve_same_as as _resolve_same_as)
from ni_rua.core.presentation.reply_encoding import (
    DERIVED_PARENT, DERIVED_QUANTITIES, STATIONARY_RESOLUTION, UNCHANGED_ARMS_KEY, encode_compact_items,
    encode_compact_value)
from ni_rua.core.presentation.reply_compaction import (
    DETAIL_KEY, SAME_AS_KEY, SAME_AS_PREFIX, STATIC_ONCE_CATALOGUE_NOTE, STATIC_ONCE_LISTS, STATIC_ONCE_STRINGS,
    compact_content_items, compact_reply_text, dangling_same_as, drop_static_once, drop_static_once_report,
    reports_exception, shorten_float32_numbers, static_once_catalogue, unify_detail_entries,
    unify_detail_entries_report)

NI_RUA_TEST_TIER = "fast"


class ActorPayloadTests(unittest.TestCase):
    def observation(self):
        return dict(
            bundle=dict(
                observation_id="obs-current", last_action_id="action-last", sim_step=123456789,
                sim_time_s=7.123456789, step_domain="adapter_control_step",
                task_instruction="Place the object inside the box, front facing left.",
                run_id="run", episode_id="ep", env_instance_id="env", env_generation=3,
                schema_version="0.1", robot_state_ref="artifact:state", world_kind="observed",
                cameras=[dict(camera_id="cam_head", frame_id="frame-current", rgb_ref="artifact:rgb",
                              width=640, height=480, pixel_transform="identity", sim_step=123456789,
                              capture_time="2026-10-03T00:00:00Z", clock_domain=None,
                              image_encoding=dict(quality=85, sha256="encoding-hash"),
                              reconstruction=dict(source_sha256="source-hash", warning="decoded from video"),
                              future_camera_warning="image may be stale")],
                future_bundle_warning="synchronization unknown"),
            robot_state=dict(frame_id="env_origin", tcp_config_id="arx-x5-link6-v1",
                             sim_step=123456789, arms=dict(left=dict(
                                 q_rad=[-1.234567891, 0.00000049], source_kind="measurement",
                                 ee_pose=dict(position_m=[0.123456789, -0.11111119, 0.9],
                                              quaternion_xyzw=[-0.5, 0.5, 0.5, 0.5], frame_id="env_origin",
                                              tcp_config_id="arx-x5-link6-v1", source_kind="measurement"),
                                 grasp_pose=dict(position_m=[0.123456789, -0.11111119, 0.7499],
                                                 angles_deg=[0.0, 0.0, 89.123456789], frame_id="env_origin",
                                                 source_kind="derived_from_measured_link6"),
                                 gripper=dict(command_normalized=1.0, command_source_kind="command",
                                              contact=None, retained_grasp=None)))),
            image_view=dict(source_roi=[10, 20, 30, 40], display_scale=4, width=120, height=160,
                            pixel_transform=[[4, 0, -41.5], [0, 4, -81.5], [0, 0, 1]],
                            original_rgb_ref="artifact:original", rgb_ref="artifact:crop",
                            reconstruction=dict(source_sha256="original-hash", warning="reconstructed")),
        )

    def test_observation_keeps_control_state_crop_geometry_and_unknown_fields(self):
        original = self.observation()
        before = deepcopy(original)
        result = compact_observation(original)
        self.assertEqual(original, before)
        self.assertEqual(set(result), set(original))
        for key in ("observation_id", "last_action_id", "sim_step", "task_instruction", "world_kind",
                    "env_generation", "future_bundle_warning"):
            self.assertEqual(result["bundle"][key], original["bundle"][key])
        self.assertEqual(result["bundle"]["sim_time_s"], 7.123457)
        camera = result["bundle"]["cameras"][0]
        for key in ("camera_id", "frame_id", "rgb_ref", "width", "height", "pixel_transform", "sim_step",
                    "reconstruction", "future_camera_warning"):
            self.assertEqual(camera[key], original["bundle"]["cameras"][0][key])
        self.assertNotIn("image_encoding", camera)
        self.assertNotIn("capture_time", camera)
        self.assertNotIn("robot_state_ref", result["bundle"])
        self.assertEqual(result["image_view"], original["image_view"])
        left = result["robot_state"]["arms"]["left"]
        self.assertEqual(left["gripper"], original["robot_state"]["arms"]["left"]["gripper"])
        self.assertEqual(left["ee_pose"]["source_kind"], "measurement")
        self.assertEqual(left["grasp_pose"]["source_kind"], "derived_from_measured_link6")
        self.assertEqual(left["ee_pose"]["frame_id"], "env_origin")
        self.assertEqual(left["ee_pose"]["tcp_config_id"], "arx-x5-link6-v1")
        self.assertEqual(left["q_rad"], [-1.234568, 0.0])
        result["image_view"]["source_roi"][0] = 999
        result["robot_state"]["arms"]["left"]["gripper"]["contact"] = True
        self.assertEqual(original, before)

    def test_action_keeps_partial_execution_failure_and_future_diagnostics(self):
        receipt = dict(status="unknown", reason="worker_disconnected_after_dispatch",
                       executed_steps=3, executed_native_steps=3, completed_native_steps=2,
                       executed_control_steps=27, execution_interval=dict(start_sim_step=100, end_sim_step=127),
                       execution_start_observation_id="obs-start", result_observation_id=None,
                       planner_failure_stage="execution", failed_waypoint_index=1,
                       position_error_m=0.004567891, orientation_error_rad=0.019876543,
                       waypoint_results=[dict(index=0, status="completed", position_error_m=0.000123456),
                                         dict(index=1, status="unknown", error="disconnect")],
                       future_diagnostic=dict(unknown=True, measured_force=None, phase="partial"),
                       collision_scope="No task-object collision proof", versions=dict(adapter="abc"),
                       step_semantics=dict(executed_steps="native count"), trace_sampling=dict(every=1),
                       planned_trace_ref="artifact:planned", dispatched_trace_ref="artifact:dispatched",
                       actual_trace_ref="artifact:actual", before_state_ref="artifact:before",
                       future_diagnostic_ref="artifact:future")
        original = dict(receipt=receipt, request=dict(action_id="action-a", based_on_observation_id="obs-a",
                                                      path=dict(waypoints=[dict(left=dict(position_m=[1, 2, 3]))])),
                        future_top_level=dict(error="preserved"))
        before = deepcopy(original)
        result = compact_action_result(original)
        self.assertEqual(original, before)
        self.assertEqual(result["request"], dict(action_id="action-a", based_on_observation_id="obs-a"))
        omitted = {"versions", "step_semantics", "trace_sampling", "planned_trace_ref", "dispatched_trace_ref",
                   "actual_trace_ref"}
        self.assertEqual(set(result["receipt"]), set(receipt) - omitted)
        for key in set(receipt) - omitted - {"position_error_m", "orientation_error_rad", "waypoint_results"}:
            self.assertEqual(result["receipt"][key], receipt[key])
        self.assertEqual(result["receipt"]["position_error_m"], 0.004568)
        self.assertEqual(result["receipt"]["orientation_error_rad"], 0.019877)
        self.assertEqual(result["receipt"]["waypoint_results"][1], receipt["waypoint_results"][1])
        self.assertEqual(result["future_top_level"], original["future_top_level"])
        result["receipt"]["future_diagnostic"]["unknown"] = False
        self.assertEqual(original, before)

    def test_only_finite_float_values_are_rounded(self):
        value = dict(extra=[True, False, 9876543210123456789, None, "1.23456789", 1.23456789,
                           float("nan"), float("inf"), float("-inf")])
        for project in (compact_observation, compact_action_result):
            result = project(value)["extra"]
            self.assertIs(result[0], True)
            self.assertIs(result[1], False)
            self.assertEqual(result[2:5], value["extra"][2:5])
            self.assertEqual(result[5], 1.234568)
            self.assertTrue(math.isnan(result[6]))
            self.assertEqual(result[7:], value["extra"][7:])
            self.assertLessEqual(abs(result[5] - value["extra"][5]), .5e-6)

    def test_missing_or_unknown_shapes_are_retained_without_invented_values(self):
        for project in (compact_observation, compact_action_result):
            for value in ({}, dict(receipt=None, request=None, bundle=None), [None, True, 3], None):
                self.assertEqual(project(value), value)
        action = dict(receipt=dict(status="failed", reason=None), request=dict(action_id="a"))
        result = compact_action_result(action)
        self.assertEqual(result["receipt"], action["receipt"])
        self.assertNotIn("based_on_observation_id", result["request"])


class ActionReplyBudgetTests(unittest.TestCase):
    """``fit_reply``: layout by decision value, the host budget rule, and what may be replaced by a reference."""

    @staticmethod
    def text(value):
        return dict(type="inputText", text=canonical(value))

    @staticmethod
    def image(name):
        return dict(type="inputImage", imageUrl="data:image/png;base64," + name)

    def reply(self, *, status="completed", reason=None, failed=(), stage_bytes=3000, sample_bytes=3000,
              reference="artifact:ref_7", action_id="a1"):
        stage = lambda name: dict(status="failed" if name in failed else "completed",
                                  reason="tracking_error" if name in failed else None,
                                  endpoint_residual_m=dict(position_error_m=0.00041), pad="s" * stage_bytes)
        rows = [dict(stage_id=name, status="reject" if name in failed else "pass", pad="r" * 600)
                for name in ("reach", "close", "lift")]
        trajectory = dict(available=True, measured_command_error=dict(max_abs_joint_error_rad=0.01),
                          measured=dict(total_samples=40, samples=[dict(native_step=0, q="m" * sample_bytes)]),
                          planned=dict(total_samples=40, samples=[dict(native_step=0, q="p" * sample_bytes)]),
                          compact_3d_trajectory=dict(unit="m", stages=["c" * sample_bytes]))
        if reference:
            trajectory["trajectory_feedback_ref"] = reference
        result = dict(action_id=action_id, plan_id="p1", action_budget=dict(native_steps_remaining=900),
                      receipt=dict(status=status, reason=reason, batch_status=status,
                                   endpoint_residual_m=dict(position_error_m=0.00041), motion_safety_checks=rows,
                                   effective_tolerances=dict(position_tolerance_m=0.005)),
                      stages={name: stage(name) for name in ("reach", "close", "lift")},
                      trajectory_feedback=trajectory)
        observation = dict(bundle=dict(observation_id="obs_3", cameras=[dict(camera_id="cam_head"),
                                                                         dict(camera_id="cam_left_wrist")]),
                           robot_state=dict(arms=dict(left=dict(q_rad=[0.1] * 6))))
        scene = dict(observation_id="obs_3", object_state_delta=dict(moved=[dict(id="o1", shift_m=[0.02, 0.0, 0.0])]),
                     object_table=dict(rows=[["o1", [0.1, 0.2, 0.78]]]))
        return [self.text(result), self.image("overlay"), self.text(observation), self.image("head"),
                self.image("wrist"), self.text(scene)]

    @staticmethod
    def parsed(items):
        return [json.loads(item["text"]) if item["type"] in ("inputText", "input_text") else item for item in items]

    def test_host_cost_counts_started_four_byte_groups_of_utf8_text_and_ignores_images(self):
        self.assertEqual([text_cost_tokens(text) for text in ("", "a", "abcd", "abcde", "\u6c49" * 4)], [0, 1, 1, 2, 3])
        self.assertEqual(reply_cost_tokens([dict(type="inputText", text="abcde"), self.image("x" * 4000),
                                            dict(type="input_text", text="abcd")]), 3)

    def test_no_budget_or_no_execution_result_returns_the_items_unchanged(self):
        items = self.reply()
        before = deepcopy(items)
        self.assertEqual(fit_reply(items, None), before)
        observation_only = items[2:5]
        fitted, report = fit_reply_report(observation_only, 8)
        self.assertEqual(fitted, observation_only)
        self.assertFalse(report["applied"])
        self.assertEqual(items, before)
        for invalid in (-1, 4.0, True, "48000"):
            with self.assertRaises(ValueError):
                fit_reply(items, invalid)

    def test_layout_is_result_scene_observation_stages_trajectory_and_images_follow_their_text(self):
        items = self.reply()
        fitted, report = fit_reply_report(items, 4 * reply_cost_tokens(items) + 400)
        self.assertEqual((report["fits"], report["by_reference"]), (True, []))
        head, scene, observation, head_image, wrist_image, stages, trajectory, overlay = self.parsed(fitted)
        self.assertEqual(set(head), {"action_id", "plan_id", "action_budget", "receipt"})
        self.assertEqual(set(scene), {"observation_id", "object_state_delta", "object_table"})
        self.assertEqual(observation["bundle"]["observation_id"], "obs_3")
        self.assertEqual((head_image, wrist_image, overlay), (items[3], items[4], items[1]))
        self.assertEqual(set(stages), {"action_id", "stages"})
        self.assertEqual(set(trajectory), {"action_id", "trajectory_feedback"})
        original = json.loads(items[0]["text"])
        # Every value of the original result is still delivered, exactly.
        self.assertEqual(dict(head, stages=stages["stages"], trajectory_feedback=trajectory["trajectory_feedback"]),
                         original)
        self.assertEqual([fitted[1], fitted[2]], [items[5], items[2]])
        self.assertEqual(fit_reply(fitted, 4 * reply_cost_tokens(items) + 400), fitted)

    def test_over_budget_replaces_least_valuable_blocks_first_and_stays_within_budget(self):
        items = self.reply()
        full = reply_cost_tokens(fit_reply(items, 10 ** 6))
        orders = []
        for budget_tokens in range(full, 0, -150):
            fitted, report = fit_reply_report(items, budget_tokens * 4)
            if not report["fits"]:
                break
            self.assertLessEqual(reply_cost_tokens(fitted), budget_tokens)
            self.assertEqual(sum(item["type"] == "inputImage" for item in fitted), 3)
            head, scene, observation = self.parsed(fitted)[:3]
            # The result summary, the scene change and the post-action observation are never reduced.
            self.assertEqual(head["receipt"]["endpoint_residual_m"], dict(position_error_m=0.00041))
            self.assertEqual(head["receipt"]["effective_tolerances"], dict(position_tolerance_m=0.005))
            self.assertEqual([fitted[1], fitted[2]], [items[5], items[2]])
            if report["by_reference"] not in orders:
                orders.append(report["by_reference"])
        blocks = ["motion_safety_pass_rows", "planned_samples", "measured_samples", "compact_3d_trajectory", "stages"]
        self.assertEqual(orders[0], [])
        self.assertEqual(orders[-1], blocks)
        for replaced in orders:
            self.assertEqual(replaced, [name for name in blocks if name in replaced])
        # Per-stage residuals of a clean action outlive every trajectory block: at the smallest budget that still
        # carries them inline, all four other blocks are references.
        low, high = 1, full
        while low < high:
            middle = (low + high) // 2
            if "stages" in fit_reply_report(items, middle * 4)[1]["by_reference"]:
                low = middle + 1
            else:
                high = middle
        self.assertEqual(fit_reply_report(items, low * 4)[1]["by_reference"], blocks[:-1])
        fitted, report = fit_reply_report(items, 40)
        self.assertEqual((report["fits"], report["by_reference"]), (False, blocks))
        self.assertEqual(self.parsed(fitted)[0]["receipt"]["status"], "completed")

    def test_replaced_blocks_name_the_call_that_returns_them_and_keep_their_neighbours(self):
        items = self.reply()
        fitted, report = fit_reply_report(items, 40)
        head, _, _, _, _, stages, trajectory, _ = self.parsed(fitted)
        omitted = self.parsed(items)[0]["trajectory_feedback"]
        status = dict(tool="robot_action_status", action_id="a1", detail="full")
        self.assertEqual(head["receipt"]["motion_safety_checks_by_reference"],
                         dict(omitted="by_reference", pass_rows=3, read=status, at="receipt.motion_safety_checks"))
        self.assertNotIn("motion_safety_checks", head["receipt"])
        self.assertEqual(stages, dict(action_id="a1", stages_by_reference=dict(
            omitted="by_reference", stage_ids=["close", "lift", "reach"], read=status, at="receipt.stage_results")))
        feedback = trajectory["trajectory_feedback"]
        for path_kind in ("measured", "planned"):
            # the call names the native steps of the rows it leaves out: without a selector inspect returns three rows
            self.assertEqual(feedback[path_kind], dict(total_samples=40, samples=dict(omitted="by_reference", read=dict(
                tool="robot_action", mode="inspect", action_id="a1", path_kind=path_kind,
                native_steps=[sample["native_step"] for sample in omitted[path_kind]["samples"]]))))
        self.assertEqual(feedback["compact_3d_trajectory"], dict(
            omitted="by_reference", at="summary.compact_3d_trajectory",
            read=dict(program_run="NiRuaClient.artifact", reference="artifact:ref_7")))
        self.assertEqual(feedback["measured_command_error"], dict(max_abs_joint_error_rad=0.01))
        self.assertEqual(feedback["trajectory_feedback_ref"], "artifact:ref_7")

    def test_sample_reads_name_the_omitted_steps_and_page_when_there_are_more_than_one_call_can_name(self):
        rows = lambda steps: [dict(native_step=step) for step in steps]
        base = dict(tool="robot_action", mode="inspect", action_id="a1", path_kind="measured")
        read = reply_budget._samples_read
        self.assertEqual(read("a1", "measured", rows([0, 8, 40])), dict(base, native_steps=[0, 8, 40]))
        at_limit = list(range(0, 2 * INSPECT_STEP_LIMIT, 2))
        self.assertEqual(read("a1", "measured", rows(at_limit)), dict(base, native_steps=at_limit))
        over = list(range(3, 3 + INSPECT_STEP_LIMIT + 1))
        self.assertEqual(read("a1", "measured", rows(over)),
                         dict(base, start_native_step=3, max_samples=INSPECT_STEP_LIMIT))
        # rows that cannot be named (none, no integer step, a repeated step) leave the plain inspect call
        for unnameable in ([], rows([None]), rows([1, 1]), None):
            self.assertEqual(read("a1", "measured", unnameable), base)

    def test_a_block_without_a_reading_entry_is_never_replaced(self):
        fitted, report = fit_reply_report(self.reply(reference=None), 40)
        self.assertNotIn("compact_3d_trajectory", report["by_reference"])
        self.assertEqual(self.parsed(fitted)[6]["trajectory_feedback"]["compact_3d_trajectory"]["unit"], "m")
        items = self.reply()
        result = json.loads(items[0]["text"])
        result.pop("action_id")
        items[0] = self.text(result)
        fitted, report = fit_reply_report(items, 40)
        self.assertEqual((report["fits"], report["by_reference"]), (False, []))
        parts = self.parsed(fitted)
        self.assertEqual(dict(parts[0], stages=parts[5]["stages"],
                              trajectory_feedback=parts[6]["trajectory_feedback"]), result)

    def test_failed_action_keeps_its_diagnostics_and_gives_up_completed_stage_detail_before_the_measured_path(self):
        items = self.reply(status="failed", reason="tracking_error", failed=("lift",))
        fitted, report = fit_reply_report(items, 40)
        self.assertEqual(report["by_reference"], ["motion_safety_pass_rows", "planned_samples", "stages",
                                                  "measured_samples", "compact_3d_trajectory"])
        head, stages = self.parsed(fitted)[0], self.parsed(fitted)[5]
        original = json.loads(items[0]["text"])
        self.assertEqual(head["stages"], dict(lift=original["stages"]["lift"]))
        self.assertEqual(head["receipt"]["motion_safety_checks"], [original["receipt"]["motion_safety_checks"][2]])
        self.assertEqual(head["receipt"]["motion_safety_checks_by_reference"]["pass_rows"], 2)
        self.assertEqual(stages["stages_by_reference"]["stage_ids"], ["close", "reach"])
        # With room for one more block, the measured path returns before the completed stages' detail.
        full = reply_cost_tokens(fit_reply(items, 10 ** 6))
        seen = []
        for budget_tokens in range(full, 0, -150):
            replaced = fit_reply_report(items, budget_tokens * 4)[1]["by_reference"]
            if replaced not in seen:
                seen.append(replaced)
        self.assertIn(["motion_safety_pass_rows", "planned_samples", "stages"], seen)
        self.assertFalse(any("measured_samples" in replaced and "stages" not in replaced for replaced in seen))

    def test_rollout_spelling_is_accepted_and_kept(self):
        items = [dict(type="input_text", text=item["text"]) if item["type"] == "inputText"
                 else dict(type="input_image", image_url=item["imageUrl"]) for item in self.reply()]
        fitted = fit_reply(items, 40)
        self.assertEqual({item["type"] for item in fitted}, {"input_text", "input_image"})
        self.assertEqual([item["text"] for item in fitted if item["type"] == "input_text"],
                         [item["text"] for item in fit_reply(self.reply(), 40) if item["type"] == "inputText"])


class ReplyCompactionTests(unittest.TestCase):
    """features.reply_same_as / features.reply_float32: lossless, and inert when off."""

    def reply(self):
        budget = dict(native_steps_limit=1050, native_steps_remaining=728, native_steps_used=322,
                      semantics="official native steps; separate from tool requests and model-call budget")
        note = "Positive approach_shortfall_m means the selected reference remains behind its requested final target."
        widened = _float32(1.0667834)  # a float32 measurement printed as float64: 1.0667834281921387
        return dict(
            action_budget=budget,
            receipt=dict(action_budget=dict(budget), status="completed", endpoint_m=[widened, -0.25, 3.141592653589793]),
            stages={"a/b": dict(interpretation=note, q_rad=[widened] * 6), "c~d": dict(interpretation=note)},
            text='number inside a string stays: 1.0667834281921387 and "0.30000001192092896"',
            short=dict(unit="m"), other=dict(unit="m"),
        )

    def test_switches_off_return_the_same_items(self):
        items = [dict(type="inputText", text=canonical(self.reply())), dict(type="inputImage", imageUrl="data:x")]
        self.assertIs(compact_content_items(items), items)
        self.assertEqual(compact_reply_text(items[0]["text"]), items[0]["text"])

    def test_same_as_points_to_the_first_copy_and_expands_back_exactly(self):
        original = self.reply()
        compacted = json.loads(compact_reply_text(canonical(original), same_as=True))
        self.assertEqual(compacted["receipt"]["action_budget"], {SAME_AS_KEY: "/action_budget"})
        self.assertEqual(compacted["stages"]["c~d"]["interpretation"], SAME_AS_PREFIX + "/stages/a~1b/interpretation")
        # copies too short to pay for a pointer stay inline
        self.assertEqual(compacted["other"], dict(unit="m"))
        self.assertEqual(compacted["stages"]["a/b"]["q_rad"], original["stages"]["a/b"]["q_rad"])
        self.assertEqual(_resolve_same_as(compacted), original)
        self.assertLess(len(canonical(compacted)), len(canonical(original)))

    def test_float32_keeps_every_number_at_its_source_precision(self):
        original = self.reply()
        text = compact_reply_text(canonical(original), float32=True)
        compacted = json.loads(text)
        self.assertEqual(compacted["receipt"]["endpoint_m"], [1.0667834, -0.25, 3.141592653589793])
        self.assertEqual(_float32(compacted["receipt"]["endpoint_m"][0]), original["receipt"]["endpoint_m"][0])
        self.assertEqual(compacted["text"], original["text"])
        self.assertEqual(compacted["action_budget"], original["action_budget"])
        for token in ("0.1", "-0.0", "1e-05", "5e-324", "1e+30", "3.141592653589793", "-0.9999999962363862", "12",
                      "0.004", "6.66081190072365e-09"):
            self.assertEqual(shorten_float32_numbers(token), token)
        for token, short in (("0.30000001192092896", "0.3"), ("-0.4099999964237213", "-0.41"),
                             ("1.0667834281921387", "1.0667834")):
            self.assertEqual(shorten_float32_numbers(token), short)
            self.assertEqual(_float32(float(short)), float(token))

    def test_images_and_text_that_is_not_canonical_json_pass_through(self):
        image = dict(type="inputImage", imageUrl="data:image/webp;base64,AAAA")
        spaced = json.dumps(self.reply(), indent=1)
        items = [dict(type="inputText", text="plain 1.0667834281921387"), dict(type="inputText", text=spaced), image,
                 dict(type="inputText", text=canonical(self.reply()))]
        result = compact_content_items(items, same_as=True, float32=True)
        self.assertEqual(result[:3], items[:3])
        self.assertIs(result[2], image)
        self.assertEqual(_resolve_same_as(json.loads(compact_reply_text(items[3]["text"], same_as=True))), self.reply())
        self.assertLess(len(result[3]["text"]), len(items[3]["text"]))
        # pointers never cross text items: the second item repeats the first one in full
        twice = compact_content_items([items[3], items[3]], same_as=True)
        self.assertEqual(twice[0], twice[1])


class ReplyShapingOrderTests(unittest.TestCase):
    """The order of the reply switches: fixed notes out, then layout and fit, with detail entries and same_as / float32
    applied to the text items of the layout. A same_as pointer always names text of its own item."""

    POLICY = dict(policy="joint-travel-admission-v1", thresholds=dict(min_total_travel_rad=3.141592653589793),
                  semantics="p" * 90)
    NOTE = "n" * 120
    ROW = dict(native_step=0, q="q" * 200)

    def reply(self, **kwargs):
        items = ActionReplyBudgetTests().reply(**kwargs)
        result = json.loads(items[0]["text"])
        receipt, feedback = result["receipt"], result["trajectory_feedback"]
        # first copy in a block that is replaced first, second copy in the execution result that always stays
        receipt["motion_safety_checks"][0]["definition"] = self.POLICY
        receipt["policy_definition"] = self.POLICY
        # first copy in the execution result, second copy in the per-stage detail (another text item after layout)
        receipt["zz_note"] = self.NOTE
        result["stages"]["close"]["note"] = self.NOTE
        # first copy in the measured samples (replaceable), second copy in the part of the feedback that stays
        feedback["measured"]["samples"].append(self.ROW)
        feedback["zz_last_row"] = self.ROW
        items[0] = dict(items[0], text=canonical(result))
        return items

    @staticmethod
    def compact(items):
        return compact_content_items(items, same_as=True, float32=True)

    @staticmethod
    def values(items):
        return [_resolve_same_as(json.loads(item["text"])) for item in items if item["type"] == "inputText"]

    def test_compacting_before_the_layout_would_leave_pointers_into_other_items_and_replaced_blocks(self):
        items = self.reply()
        compacted = self.compact(items)
        self.assertEqual(dangling_same_as(compacted), [])
        for budget in (10 ** 6, 40):
            late = fit_reply(compacted, budget)
            self.assertTrue(dangling_same_as(late), budget)

    def test_fit_with_compaction_never_leaves_a_dangling_pointer_and_loses_no_value(self):
        items = self.reply()
        full = reply_cost_tokens(fit_reply(items, 10 ** 6))
        seen = set()
        for budget_tokens in list(range(full, 0, -100)) + [10]:
            fitted, report = fit_reply_report(items, budget_tokens * 4, compact=self.compact)
            self.assertEqual(dangling_same_as(fitted), [], report)
            self.assertEqual(report["cost_tokens_after"], reply_cost_tokens(fitted))
            self.assertEqual(report["fits"], reply_cost_tokens(fitted) <= budget_tokens)
            self.assertEqual(sum(item["type"] == "inputImage" for item in fitted), 3)
            replaced = tuple(report["by_reference"])
            if replaced in seen:
                continue
            seen.add(replaced)
            # Same blocks replaced without compaction: every remaining value is identical once pointers are expanded.
            plain = next(fit for fit, plain_report in (fit_reply_report(items, tokens * 4)
                                                       for tokens in list(range(full, 0, -20)) + [10])
                         if tuple(plain_report["by_reference"]) == replaced)
            self.assertEqual(self.values(fitted), [json.loads(item["text"]) for item in plain
                                                   if item["type"] == "inputText"])
        self.assertIn((), seen)
        self.assertIn(("motion_safety_pass_rows", "planned_samples", "measured_samples", "compact_3d_trajectory",
                       "stages"), seen)

    def test_a_copy_whose_first_occurrence_was_replaced_or_moved_is_printed_in_full(self):
        items = self.reply()
        whole, _ = fit_reply_report(items, 10 ** 6, compact=self.compact)
        head, _, _, stages, feedback = (json.loads(item["text"]) for item in whole if item["type"] == "inputText")
        # inside one item the later copy points at the earlier one ...
        self.assertEqual(head["receipt"]["policy_definition"], {SAME_AS_KEY: "/receipt/motion_safety_checks/0/definition"})
        self.assertEqual(feedback["trajectory_feedback"]["zz_last_row"],
                         {SAME_AS_KEY: "/trajectory_feedback/measured/samples/1"})
        # ... and a copy whose first occurrence is in another item is printed where it is
        self.assertEqual(stages["stages"]["close"]["note"], self.NOTE)
        fitted, report = fit_reply_report(items, 40, compact=self.compact)
        self.assertIn("measured_samples", report["by_reference"])
        head, _, _, _, feedback = (json.loads(item["text"]) for item in fitted if item["type"] == "inputText")
        self.assertEqual(head["receipt"]["policy_definition"], self.POLICY)
        self.assertEqual(feedback["trajectory_feedback"]["zz_last_row"], self.ROW)
        self.assertEqual(feedback["trajectory_feedback"]["measured"]["samples"]["omitted"], "by_reference")

    def test_the_budget_is_spent_on_the_compacted_text(self):
        items = self.reply()
        compacted_cost = reply_cost_tokens(fit_reply(items, 10 ** 6, compact=self.compact))
        self.assertLess(compacted_cost, reply_cost_tokens(fit_reply(items, 10 ** 6)))
        with_hook = fit_reply_report(items, compacted_cost * 4, compact=self.compact)[1]
        without = fit_reply_report(items, compacted_cost * 4)[1]
        self.assertEqual((with_hook["fits"], with_hook["by_reference"]), (True, []))
        self.assertTrue(without["by_reference"])

    def test_detail_entries_are_rewritten_before_same_as_so_no_pointer_runs_through_a_renamed_field(self):
        items = self.reply()
        result = json.loads(items[0]["text"])
        lookup = dict(tool="robot_action_status", action_id="a1", detail="full", pad="x" * 90)
        result["receipt"]["motion_safety_check_summary"] = dict(pass_count=3, full_result=dict(lookup, inner=self.POLICY))
        result["receipt"]["zz_policy"] = self.POLICY
        items[0] = dict(items[0], text=canonical(result))

        def finish(content):
            return self.compact(unify_detail_entries(content, None))

        for budget in (10 ** 6, 40):
            fitted = fit_reply(items, budget, compact=finish)
            self.assertEqual(dangling_same_as(fitted), [])
            self.assertNotIn('"full_result"', "".join(item.get("text", "") for item in fitted))
        # the other order renames a field a pointer runs through
        late = unify_detail_entries(self.compact(fit_reply(items, 10 ** 6)), None)
        self.assertTrue(dangling_same_as(late))


class ReplyStaticOnceTests(unittest.TestCase):
    """features.reply_static_once: listed fixed notes leave routine replies; everything else stays."""

    BUDGET = "official native steps; separate from tool requests and model-call budget; reserves are not included"
    COLLISION = "native robot self-collision and configured static planner table; no task-object or other-arm collision proof"
    SOURCES = list(next(iter(STATIC_ONCE_LISTS))[1])

    @staticmethod
    def text(value):
        return dict(type="inputText", text=canonical(value))

    def result(self, status="completed", reason=None):
        budget = dict(native_steps_limit=1300, native_steps_remaining=952, semantics=self.BUDGET,
                      source="native_execution_accounting", unknown_values_are_null=True)
        return dict(
            action_id="a10", plan_id="p3", action_budget=budget, next_action=dict(mode="inspect", plan_id="p3"),
            receipt=dict(
                status=status, reason=reason, batch_status=status, completed_native_steps=42, collision_scope=self.COLLISION,
                action_budget=dict(budget), endpoint_residual_m=dict(position_error_m=3e-05),
                step_semantics=dict(completed_native_steps="native take_action calls that returned successfully",
                                    executed_control_steps="physics steps within official action interpolation",
                                    executed_steps="official native action count delta; includes a partially executed faulting call"),
                motion_safety_check_summary=dict(count=2, pass_count=2, last_status="pass", full_result=dict(
                    tool="robot_action_status", action_id="a10", detail="full")),
                motion_safety_checks=[
                    dict(stage_id="reach", status="pass", information_sources=list(self.SOURCES)),
                    dict(stage_id="lift", status="pass", information_sources=[
                        "preceding planned named joint endpoint, not future measurement", *self.SOURCES[1:]])]),
            stages=dict(reach=dict(status="completed", completed_steps=11,
                                   endpoint_error=dict(effective_tolerances=dict(position_tolerance_m=0.025),
                                                       position_error_m=0.00011646229619681309),
                                   grasp_contact_feedback=dict(
                                       approach_shortfall_m=4.0171455136239785e-05,
                                       interpretation=next(text for key, text in STATIC_ONCE_STRINGS
                                                           if key == "interpretation")))),
            trajectory_feedback=dict(
                available=True, trajectory_feedback_ref="artifact:ref_49",
                detail_access="trajectory_feedback_ref contains every exact row; read with NiRuaClient.artifact.",
                limitations=["Some initial jaw rows were not retained; no jaw command or missing row was synthesized.",
                             next(text for key, text in STATIC_ONCE_STRINGS if key == "limitations")],
                geometry=dict(status="available", jaw_scope="Measured jaws included only where retained; command geometry stops at link6."),
                measured_command_error=dict(max_abs_joint_error_rad=0.0123, semantics=(
                    "Encoder minus stored target at completed native boundaries; no contact or task-success inference."))))

    def checkpoint(self, code, **receipt):
        """A batch that stopped at a checkpoint: same status and reason whoever asked for the stop; the code tells."""
        result = self.result(status="stopped", reason="observation_checkpoint")
        result["receipt"].update(dict(
            batch_status="awaiting_observation", paused_at_checkpoint=True, checkpoint_stage_ids=["check"],
            checkpoint_reasons=dict(check="Inspect the grasp."), pending_stage_ids=["lift"],
            stop_confirmation=dict(clock="paused", settling_steps=0)), **receipt)
        if code is not None:
            result["receipt"]["checkpoint_reason_code"] = code
        result["stages"]["lift"] = dict(status="pending", completed_steps=0, kind="move", arm="left")
        if code not in (None, "observation_checkpoint", "jaw_checkpoint"):
            result["stages"]["reach"].update(status="paused_at_checkpoint", checkpoint_reason_code=code)
        return result

    def test_a_stop_the_model_asked_for_is_routine_and_a_guard_stop_is_not(self):
        from ni_rua.core.presentation.feedback_delivery import (
            MODEL_REQUESTED_CHECKPOINT_CODES, is_requested_checkpoint_receipt)
        self.assertEqual(MODEL_REQUESTED_CHECKPOINT_CODES, {"observation_checkpoint", "jaw_checkpoint"})
        for code in sorted(MODEL_REQUESTED_CHECKPOINT_CODES):
            with self.subTest(code):
                value = self.checkpoint(code)
                self.assertTrue(is_requested_checkpoint_receipt(value["receipt"]))
                self.assertFalse(reports_exception(value))
                shaped, report = drop_static_once_report([self.text(value), self.text(self.observation())])
                self.assertFalse(report["exception"])
                receipt = json.loads(shaped[0]["text"])["receipt"]
                self.assertNotIn("collision_scope", receipt)
                # what the stop is about stays: status, the typed code, the stage ids and the model's own reason
                for key in ("status", "reason", "checkpoint_reason_code", "checkpoint_stage_ids", "checkpoint_reasons",
                            "pending_stage_ids", "stop_confirmation"):
                    self.assertEqual(receipt[key], value["receipt"][key])
                # summary_first: the fixed routine shape, pending stages inline in the execution result
                fitted, shape = fit_reply_report([self.text(value)], None, profile="summary_first")
                self.assertTrue(shape["routine"])
                self.assertEqual(json.loads(fitted[0]["text"])["stages"], dict(lift=value["stages"]["lift"]))
                # the full profile's replacement order still treats a stop as an exception
                self.assertFalse(reply_budget._routine(value))
        for code in ("aperture_below_min_m", "aperture_above_max_m", "carry_slip_aperture_below_min_m", "new_guard", None):
            with self.subTest(code):
                value = self.checkpoint(code)
                self.assertFalse(is_requested_checkpoint_receipt(value["receipt"]))
                self.assertTrue(reports_exception(value))
                self.assertFalse(fit_reply_report([self.text(value)], None, profile="summary_first")[1]["routine"])

    def observation(self):
        return dict(bundle=dict(observation_id="obs_11", cameras=[dict(
            camera_id="cam_head", sim_step=3480, synchronization="physics paused; native zero-delay camera delivery fence")]),
            metadata_delivery=dict(mode="unchanged_static_fields_referenced", read_full=dict(
                tool="history_view_observation", observation_id="obs_11", detail="full")),
            robot_state=dict(arms=dict(left=dict(q_rad=[0.1] * 6))))

    def test_listed_notes_leave_a_routine_reply_and_every_other_value_stays(self):
        image = dict(type="inputImage", imageUrl="data:image/webp;base64,AAAA")
        items = [self.text(self.result()), image, self.text(self.observation()), dict(type="inputText", text="plain semantics")]
        before = deepcopy(items)
        shaped, report = drop_static_once_report(items)
        self.assertEqual(items, before)
        self.assertEqual(report, dict(exception=False, dropped=dict(
            semantics=3, unknown_values_are_null=2, collision_scope=1, completed_native_steps=1, executed_control_steps=1,
            executed_steps=1, information_sources=1, interpretation=1, limitations=1, jaw_scope=1)))
        self.assertIs(shaped[1], image)
        self.assertEqual(shaped[3], items[3])
        result, observation = json.loads(shaped[0]["text"]), json.loads(shaped[2]["text"])
        original = self.result()
        receipt = result["receipt"]
        self.assertEqual(result["action_budget"], dict(native_steps_limit=1300, native_steps_remaining=952,
                                                       source="native_execution_accounting"))
        self.assertNotIn("collision_scope", receipt)
        self.assertNotIn("step_semantics", receipt)  # held nothing but fixed notes
        # a row whose sources are the fixed list loses the key, a row that names another source keeps its whole list
        self.assertEqual(receipt["motion_safety_checks"], [
            dict(stage_id="reach", status="pass"), original["receipt"]["motion_safety_checks"][1]])
        self.assertEqual(result["stages"]["reach"]["grasp_contact_feedback"], dict(approach_shortfall_m=4.0171455136239785e-05))
        self.assertEqual(result["stages"]["reach"]["endpoint_error"], original["stages"]["reach"]["endpoint_error"])
        feedback = result["trajectory_feedback"]
        self.assertEqual(feedback["limitations"], original["trajectory_feedback"]["limitations"][:1])
        self.assertEqual(feedback["geometry"], dict(status="available"))
        self.assertEqual(feedback["measured_command_error"], dict(max_abs_joint_error_rad=0.0123))
        # a string under a listed key with another text is a diagnostic, not a fixed note
        self.assertEqual(feedback["detail_access"], original["trajectory_feedback"]["detail_access"])
        # a string the robot backend produces per camera is not a fixed note
        self.assertEqual(observation["bundle"]["cameras"], self.observation()["bundle"]["cameras"])
        self.assertEqual(observation["robot_state"], self.observation()["robot_state"])
        self.assertEqual(drop_static_once(shaped), shaped)

    def test_a_reply_that_reports_an_exception_keeps_every_note(self):
        routine = [self.text(self.result()), self.text(self.observation())]
        self.assertFalse(any(reports_exception(json.loads(item["text"])) for item in routine))
        cases = dict(
            failed=self.result(status="failed", reason="tracking_error"),
            stopped=self.result(status="stopped", reason="observation_checkpoint"),
            unknown=self.result(status="unknown"),
            stage=dict(self.result(), stages=dict(reach=dict(status="failed", completed_steps=0))),
            guard=self.checkpoint("aperture_below_min_m"),
            carry_slip=self.checkpoint("carry_slip_aperture_below_min_m", slip_detected_at=dict(stage_id="lift", step=1)),
            unknown_code=self.checkpoint("some_future_guard"),
            no_code=self.checkpoint(None),
            unconfirmed=self.checkpoint("observation_checkpoint", stop_confirmation=dict(confirmed=False)),
            checkpoint_with_slip=dict(self.checkpoint("observation_checkpoint"),
                                      trajectory_feedback=dict(available=True, anomaly_reason="slip_detected")),
            checkpoint_resume_invalid=self.checkpoint("jaw_checkpoint", resume_validation=dict(status="invalid")),
            anomaly=dict(self.result(), trajectory_feedback=dict(available=True, anomaly_reason="slip_detected")),
            unavailable=dict(self.result(), trajectory_feedback=dict(available=False, reason="trajectory_feedback_failed")),
            refused=dict(status="planned", plan_id="p4", blocking_risks=[dict(code="path_infeasible")],
                         action_budget=self.result()["action_budget"]),
            failure=dict(failure_feedback=dict(collision_scope=self.COLLISION), action_budget=self.result()["action_budget"]),
            program=dict(status="awaiting_review", result_semantics=next(
                text for key, text in STATIC_ONCE_STRINGS if key == "result_semantics"), pending_review=dict(review_id="r1")),
        )
        for name, value in cases.items():
            with self.subTest(name):
                items = [self.text(value), self.text(self.observation())]
                shaped, report = drop_static_once_report(items)
                self.assertIs(shaped, items)
                self.assertEqual(report, dict(exception=True, dropped={}))
        program = dict(status="succeeded", returncode=0, stdout_text="ok", result_semantics=cases["program"]["result_semantics"])
        self.assertEqual(json.loads(drop_static_once([self.text(program)])[0]["text"]),
                         dict(status="succeeded", returncode=0, stdout_text="ok"))

    def test_detail_entries_take_one_form_and_name_only_loaded_tools(self):
        loaded = frozenset({"robot_action", "program_run", "robot_action_status", "program_status",
                            "history_view_observation", "history_list_actions"})
        items = [self.text(self.result()), self.text(self.observation())]
        shaped, counts = unify_detail_entries_report(items, loaded)
        self.assertEqual(counts, dict(next_action=1, full_result=1, detail_access=1, read_full=1))
        result, observation = (json.loads(item["text"]) for item in shaped)
        self.assertEqual(result[DETAIL_KEY], dict(tool="robot_action", arguments=dict(mode="inspect", plan_id="p3")))
        self.assertNotIn("next_action", result)
        # detail_access says the artifact holds every exact row: the entry reads that artifact (inspect would return
        # three rows without a selector), here the whole document, through program_run
        reading = result["trajectory_feedback"][DETAIL_KEY]
        self.assertEqual(reading, reply_compaction._artifact_call("artifact:ref_49", None, loaded))
        self.assertEqual(reading["tool"], "program_run")
        self.assertEqual(result["trajectory_feedback"]["trajectory_feedback_ref"], "artifact:ref_49")
        document = dict(paths=dict(measured=dict(samples=[dict(native_step=step) for step in range(40)])))
        artifact_reads, shown = [], []

        class ArtifactClient:
            def artifact(self, reference):
                artifact_reads.append(reference)
                return json.dumps(document)
        with unittest.mock.patch.dict("sys.modules", ni_rua_client=type("ni_rua_client", (), dict(NiRuaClient=ArtifactClient))):
            exec(reading["arguments"]["source"], dict(print=shown.append))
        self.assertEqual((artifact_reads, json.loads(shown[0])), (["artifact:ref_49"], document))
        # without program_run the entry is the inspect call that returns the rows in pages of the inspect limit
        inspect_only = json.loads(unify_detail_entries(items, frozenset({"robot_action"}))[0]["text"])
        self.assertEqual(inspect_only["trajectory_feedback"][DETAIL_KEY], dict(tool="robot_action", arguments=dict(
            mode="inspect", trajectory_ref="artifact:ref_49", path_kind="measured", start_native_step=0,
            max_samples=INSPECT_STEP_LIMIT)))
        self.assertEqual(observation["metadata_delivery"][DETAIL_KEY], dict(
            tool="history_view_observation", arguments=dict(observation_id="obs_11", detail="full")))
        # robot_action_status is loaded up front: the entry names it, with the complete arguments of the read
        summary = result["receipt"]["motion_safety_check_summary"]
        self.assertEqual(set(summary), {"count", "pass_count", "last_status", DETAIL_KEY})
        self.assertEqual(summary[DETAIL_KEY], dict(tool="robot_action_status", arguments=dict(action_id="a10", detail="full")))
        # program_status likewise: the full_result of a program reply becomes the same form
        program = json.loads(unify_detail_entries([self.text(dict(
            status="succeeded", job_id="job_3", full_result=dict(tool="program_status", job_id="job_3", detail="full")))],
            loaded)[0]["text"])
        self.assertEqual(program[DETAIL_KEY], dict(tool="program_status", arguments=dict(job_id="job_3", detail="full")))
        self.assertNotIn("full_result", program)
        # every tool loaded (direct_tools=None): the lookup is named as it is
        direct = json.loads(unify_detail_entries(items)[0]["text"])["receipt"]["motion_safety_check_summary"][DETAIL_KEY]
        self.assertEqual(direct, dict(tool="robot_action_status", arguments=dict(action_id="a10", detail="full")))
        self.assertEqual(unify_detail_entries(shaped, loaded), shaped)

    def test_entries_without_a_loaded_reader_and_commands_stay_as_they_are(self):
        loaded = frozenset({"robot_action", "program_run"})  # a catalogue without the status tools (not the default)
        program = dict(status="succeeded", job_id="job_3", full_result=dict(tool="program_status", job_id="job_3", detail="full"))
        preview = dict(status="planned", plan_id="p4", next_action=dict(mode="execute", plan_id="p4", expected_cursor=0))
        paused = dict(action_id="a2", receipt=dict(status="stopped"), next_action=dict(
            mode="inspect", plan_id="p4", continuation_example=dict(mode="preview")))
        summary = dict(recent_actions=[dict(status="completed", next_action=dict(mode="inspect", plan_id="p12"))])
        self.assertEqual(json.loads(unify_detail_entries([self.text(summary)], loaded)[0]["text"])["recent_actions"],
                         [dict(status="completed", detail=dict(tool="robot_action", arguments=dict(mode="inspect", plan_id="p12")))])
        named = dict(detail="passed_check_summary", full_result=dict(tool="robot_action", mode="inspect", plan_id="p4"))
        items = [self.text(value) for value in (program, preview, paused, named)]
        shaped, counts = unify_detail_entries_report(items, loaded)
        self.assertIs(shaped, items)
        self.assertEqual(counts, {})
        self.assertIs(unify_detail_entries(items, frozenset()), items)

    def test_by_reference_stand_ins_of_the_budget_get_the_same_form(self):
        loaded = frozenset({"robot_action", "program_run", "robot_action_status"})
        items = ActionReplyBudgetTests().reply()
        fitted = fit_reply(items, 40)
        shaped, counts = unify_detail_entries_report(fitted, loaded)
        self.assertEqual(counts, dict(read=5))
        head, _, _, stages, trajectory = (json.loads(item["text"]) for item in shaped if item["type"] == "inputText")
        note = head["receipt"]["motion_safety_checks_by_reference"]
        self.assertEqual({key: note[key] for key in ("omitted", "pass_rows")}, dict(omitted="by_reference", pass_rows=3))
        # a read of robot_action_status names the tool itself; ``at`` is the part of its data the entry stands for
        self.assertEqual(set(note), {"omitted", "pass_rows", DETAIL_KEY, "at"})
        self.assertEqual((note[DETAIL_KEY]["tool"], note["at"]), ("robot_action_status", "receipt.motion_safety_checks"))
        self.assertEqual((stages["stages_by_reference"][DETAIL_KEY]["tool"], stages["stages_by_reference"]["at"]),
                         ("robot_action_status", "receipt.stage_results"))
        feedback = trajectory["trajectory_feedback"]
        self.assertEqual(feedback["planned"]["samples"], dict(omitted="by_reference", detail=dict(
            tool="robot_action", arguments=dict(mode="inspect", action_id="a1", path_kind="planned", native_steps=[
                sample["native_step"] for sample in json.loads(items[0]["text"])["trajectory_feedback"]["planned"]["samples"]]))))
        artifact = feedback["compact_3d_trajectory"]
        self.assertEqual(set(artifact), {"omitted", DETAIL_KEY})
        self.assertIn("NiRuaClient().artifact('artifact:ref_7')", artifact[DETAIL_KEY]["arguments"]["source"])
        self.assertIn("['summary']['compact_3d_trajectory']", artifact[DETAIL_KEY]["arguments"]["source"])

    def test_catalogue_note_is_added_to_robot_action_only(self):
        guide = "\nInput field guide (use only fields applicable to the selected mode/kind):\ninput: Type: object."
        tools = [dict(name="robot_observe", description="Observe." + guide, inputSchema={}),
                 dict(name="robot_action", description="Act." + guide, inputSchema={}),
                 dict(name="robot_wait", description="Wait.", inputSchema={})]
        before = deepcopy(tools)
        noted = static_once_catalogue(tools)
        self.assertEqual(tools, before)
        self.assertEqual([noted[0], noted[2]], [tools[0], tools[2]])
        self.assertEqual(noted[1], dict(tools[1], description="Act." + STATIC_ONCE_CATALOGUE_NOTE + guide))
        bare = static_once_catalogue([dict(name="robot_action", description="Act.")])
        self.assertEqual(bare[0]["description"], "Act." + STATIC_ONCE_CATALOGUE_NOTE)
        self.assertEqual(static_once_catalogue(before[:1]), before[:1])

    def test_only_the_notes_that_apply_to_the_run_s_robot_are_stated_and_left_out(self):
        from ni_rua.core.presentation.reply_compaction import static_once_plan
        full = static_once_plan()
        self.assertEqual(full.segments, ("budget", "collision", "steps", "grasp_contact", "joint_travel", "trajectory",
                                         "trajectory_link6", "program"))
        self.assertEqual((full.note, full.strings, full.lists), (STATIC_ONCE_CATALOGUE_NOTE, STATIC_ONCE_STRINGS, STATIC_ONCE_LISTS))
        x5 = [dict(name="robot_action", description="Shared rows cost one native step. contact_region joint_travel_preset "
                                                    "link6"),
              dict(name="program_run", description="Process exit 0 or tool success does not prove robot/task success.")]
        scope = "planner self-collision only"
        plan = static_once_plan(x5, dict(collision_scope=scope))
        self.assertEqual(plan.segments, full.segments)
        self.assertIn(("collision_scope", scope), plan.strings)
        self.assertNotIn(("collision_scope", self.COLLISION), plan.strings)
        self.assertIn("collision_scope: " + scope + ". ", plan.note)
        # no published collision scope: the group is neither stated nor left out
        self.assertNotIn("collision", static_once_plan(x5, {}).segments)
        # a robot whose catalogue has no native step budget, joint-travel admission, grasp-contact fields or link6
        other = [dict(name="robot_action", description="Move the arm.")]
        self.assertEqual(static_once_plan(other, dict(collision_scope=scope),
                                          SimpleNamespace(trajectory_preview=False)).segments, ())
        plan = static_once_plan(other, dict(collision_scope=scope), SimpleNamespace(trajectory_preview=True))
        self.assertEqual(plan.segments, ("trajectory",))
        for word in ("native step", "joint_travel_preset", "link6", "approach_shortfall_m", "collision_scope", "action_budget"):
            self.assertNotIn(word, plan.note)
        items = [self.text(self.result()), self.text(self.observation())]
        shaped, report = drop_static_once_report(items, plan=plan)
        self.assertEqual(report["dropped"], dict(semantics=1, limitations=1))
        kept = json.loads(shaped[0]["text"])
        original = self.result()
        self.assertEqual(kept["action_budget"], original["action_budget"])
        for key in ("collision_scope", "step_semantics", "motion_safety_checks"):
            self.assertEqual(kept["receipt"][key], original["receipt"][key])
        self.assertEqual(kept["stages"], original["stages"])
        self.assertEqual(kept["trajectory_feedback"]["geometry"], original["trajectory_feedback"]["geometry"])
        # nothing in the catalogue: only the reading rule of the detail entry is stated, nothing is left out
        empty = static_once_plan([], None)
        self.assertEqual((empty.segments, empty.strings, empty.lists, empty.flags), ((), frozenset(), frozenset(), frozenset()))
        self.assertTrue(empty.note.startswith(' detail: {"tool", "arguments"}'))
        self.assertIs(drop_static_once(items, empty), items)
        # the layout note names the object table only when the catalogue does
        features = SimpleNamespace(action_reply_profile="summary_first", reply_static_once=False, reply_compact_encoding=True)
        notes = reply_compaction.reply_catalogue_notes(features, other, None)
        self.assertEqual(notes, [reply_budget.summary_first_catalogue_note(object_table=False, stage_detail=False),
                                 reply_encoding.COMPACT_ENCODING_CATALOGUE_NOTE])
        self.assertNotIn("grasp_contact_feedback", "".join(notes))
        self.assertNotIn("object_table", "".join(notes))
        self.assertIn("object_table", reply_budget.SUMMARY_FIRST_CATALOGUE_NOTE)
        for note in (STATIC_ONCE_CATALOGUE_NOTE, reply_budget.SUMMARY_FIRST_CATALOGUE_NOTE,
                     reply_encoding.COMPACT_ENCODING_CATALOGUE_NOTE):
            self.assertIsNone(re.search(r"\bleft\b", note))


class ReplyCompactEncodingTests(unittest.TestCase):
    """features.reply_compact_encoding: tables, an unchanged arm stated once, derived quantities at 4 digits."""

    @staticmethod
    def arm(q, *, velocity=0.0, jaw_velocity=0.0, first=False):
        entry = dict(jaw_command_normalized=1.0, jaw_goal_normalized=1.0, phase="measured",
                     measured_ee_pose=dict(frame_id="env_origin", position_m=[q, -0.35229986906051636, 0.921500027179718],
                                           source_kind="measurement", tcp_config_id="arx-x5-link6-v1"),
                     q_rad=[q, 0.3047229051589966, 0.0], qdot_rad_s=[velocity, 0.0, 1.0430727005004883 * velocity])
        if not first:
            entry[DERIVED_PARENT] = dict(
                source_kind="finite_difference_at_measurement_boundaries", interval_s=0.04, interval_sim_steps=[110, 120],
                joint_acceleration_source="measured_joint_velocity_difference", ee_frame_id="env_origin",
                joint_acceleration_rad_s2=[0.907532125711441 * velocity, 0.0, 2.465023100376129 * velocity],
                ee_linear_velocity_m_s=[-0.060508400201797485 * velocity, 0.0, 0.3224000334739685 * velocity],
                jaw_velocity_m_s=[jaw_velocity, jaw_velocity])
        return entry

    def measured(self, *, left=(0.0, 1e-9, 2e-9), left_jaw_velocity=0.0):
        right = (0.12201002985239029, 0.25, 0.4)
        return dict(total_samples=43, timing="recorded", samples=[
            dict(native_step=12 * index, sim_step=120 * index, elapsed_time_s=0.48 * index, arms=dict(
                left=self.arm(left[index], first=index == 0, jaw_velocity=left_jaw_velocity),
                right=self.arm(right[index], first=index == 0, velocity=1.0)))
            for index in range(3)])

    def test_listed_structures_become_tables_and_decode_back(self):
        rows = [dict(stage_id="reach", status="completed", arm="left", kind="move", end_step=11, start_m=[0.1, 0.2, 0.3],
                     peak=dict(step=6, divergence_m=0.0035)),
                dict(stage_id="close", status="completed", arm="left", kind="gripper", end_step=14, start_m=[0.1, 0.2, 0.3],
                     peak=dict(step=12, divergence_m=0.0)),
                dict(stage_id="lift", status="completed", arm="left", kind="move", end_step=30, start_m=[0.1, 0.2, 0.5],
                     peak=dict(step=20, divergence_m=0.0011), note=dict(code="warn", text="x"))]
        value = dict(compact_3d_trajectory=dict(unit="m", stages=rows))
        encoded = encode_compact_value(value)
        self.assertEqual(encoded["compact_3d_trajectory"]["stages"], dict(
            columns=["end_step", "kind", "note.code", "peak.divergence_m", "peak.step", "stage_id", "start_m"],
            rows=[[11, "move", None, 0.0035, 6, "reach", [0.1, 0.2, 0.3]],
                  [14, "gripper", None, 0.0, 12, "close", [0.1, 0.2, 0.3]],
                  [30, "move", "warn", 0.0011, 20, "lift", [0.1, 0.2, 0.5]]],
            common={"arm": "left", "status": "completed", "note.text": "x"}, optional=["note"]))
        self.assertEqual(decode_compact_value(encoded), value)
        self.assertLess(len(canonical(encoded)), len(canonical(value)))
        self.assertEqual(encode_compact_value(deepcopy(value)), encoded)
        # an encoded item is returned as it is, so encoding twice cannot stack
        self.assertEqual(encode_compact_value(encoded), encoded)
        # the path decides, wherever the structure is: a program's action summary, a history entry
        nested = dict(action_summary=dict(recent_actions=[dict(action_id="a1", object_state_delta=dict(
            moved=[dict(after_id="o1", before_id="o1", displacement_m=0.0153, lifted=False, delta_m=[-0.001, -0.006, 0.014]),
                   dict(after_id="o7", before_id="o3", displacement_m=0.0408, lifted=True, delta_m=[-0.028, -0.029, -0.006])],
            appeared=[dict(after_id="o8", centre_m=[0.138, -0.055, 0.973])], vacated=[]))]))
        encoded = encode_compact_value(nested)
        delta = encoded["action_summary"]["recent_actions"][0]["object_state_delta"]
        self.assertEqual(delta["moved"]["columns"], ["after_id", "before_id", "delta_m", "displacement_m", "lifted"])
        self.assertEqual((delta["appeared"], delta["vacated"]), (nested["action_summary"]["recent_actions"][0]
                                                                 ["object_state_delta"]["appeared"], []))
        self.assertIsInstance(encoded["action_summary"]["recent_actions"], list)
        self.assertEqual(decode_compact_value(encoded), nested)

    def test_every_structure_that_is_not_listed_keeps_its_json_form(self):
        long = "x" * 40
        same = [dict(id=index, text=long, value=1.5, status="completed") for index in range(4)]
        kept = dict(
            ledger=deepcopy(same), attention=deepcopy(same), advisory_notes=deepcopy(same), policy_definitions=deepcopy(same),
            receipt=dict(motion_safety_checks=deepcopy(same)), bundle=dict(cameras=deepcopy(same)), source_views=deepcopy(same),
            failure_feedback=dict(candidates=deepcopy(same)), batch_results=deepcopy(same), actions=deepcopy(same),
            episode_ledger=dict(recent_outcomes=deepcopy(same), last_object_delta=dict(moved=deepcopy(same))),
            stages={name: dict(status="completed", completed_steps=3, text=long) for name in ("a", "b", "c")},
            stage_results={name: dict(status="completed", completed_steps=3, text=long) for name in ("a", "b", "c")},
            arms=dict(left=dict(q=long, n=1), right=dict(q=long, n=2)), overlays=dict(a=dict(r=long), b=dict(r=long)),
            static_refs=dict(a=dict(r=long), b=dict(r=long)), grasp_hints=dict(o1=dict(r=long), o2=dict(r=long)),
            samples=deepcopy(same), measured=dict(samples=deepcopy(same)), moved=deepcopy(same), rows=deepcopy(same),
            # a flat tolerance object under the listed name is not a map of objects
            endpoint_error=dict(effective_tolerances=dict(max_upward_dz_m=0.0, orientation_tolerance_rad=0.35,
                                                          position_tolerance_m=0.025)))
        self.assertEqual(encode_compact_value(kept), kept)
        self.assertEqual({len(suffix) for suffix in reply_encoding.TABLE_STRUCTURES}, {1, 2, 3})
        self.assertEqual({kind for kind, _ in reply_encoding.TABLE_STRUCTURES.values()}, {"list", "list_after_first", "map"})

    def test_a_listed_structure_that_is_not_same_shaped_or_would_not_shrink_stays(self):
        long = "x" * 40
        for kept in ([dict(a=long, b=1), "text", dict(a=long, b=2)],
                     [{"a.b": long, "c": 1}, {"a.b": long, "c": 2}],
                     [dict(a=long, reason=None), dict(a=long)],
                     [dict(a=1), dict(a=2)],
                     [dict(a=long, b=long)],
                     [[1, 2], [3, 4]]):
            self.assertEqual(encode_compact_value(dict(periodic_checks=kept)), dict(periodic_checks=kept))
        # a real null is fine when every row has the field
        nulls = dict(periodic_checks=[dict(a=long, reason=None, n=index) for index in range(3)])
        encoded = encode_compact_value(nulls)
        self.assertEqual(encoded["periodic_checks"]["rows"], [[0, None], [1, None], [2, None]])
        self.assertEqual(decode_compact_value(encoded), nulls)
        # a lifted string below an optional prefix applies to the rows that have that prefix
        partial = dict(periodic_checks=[dict(n=0), dict(n=1, extra=dict(source=long, value=1.5)),
                                        dict(n=2, extra=dict(source=long, value=2.5))])
        encoded = encode_compact_value(partial)
        self.assertEqual((encoded["periodic_checks"]["common"], encoded["periodic_checks"]["optional"]),
                         ({"extra.source": long}, ["extra"]))
        self.assertEqual(decode_compact_value(encoded), partial)
        # ... and stays in the rows when no printed field of that prefix tells which rows have it
        alone = dict(periodic_checks=[dict(n=0, pad=long), dict(n=1, pad=long, extra=dict(source=long)),
                                      dict(n=2, pad=long, extra=dict(source=long))])
        encoded = encode_compact_value(alone)
        self.assertEqual(encoded["periodic_checks"]["common"], dict(pad=long))
        self.assertEqual(decode_compact_value(encoded), alone)

    def test_stage_maps_become_tables_keyed_by_id_and_planned_samples_keep_their_first_entry_whole(self):
        tolerances = {name: dict(max_upward_dz_m=0.0, orientation_tolerance_rad=0.15, position_tolerance_m=tolerance,
                                 stall_guard_preset="off", tolerance_preset="normal")
                      for name, tolerance in (("clear_top", 0.01), ("side_approach", 0.01), ("pick/motion", 0.003))}
        costs = {name: dict(arm="right", kind="move", native_steps=steps, start_native_step=0, end_native_step=steps)
                 for name, steps in (("a", 106), ("b", 25), ("c", 4))}
        value = dict(effective_tolerances=tolerances, stage_costs=costs)
        encoded = encode_compact_value(value)
        self.assertEqual(encoded["effective_tolerances"], dict(
            keyed_by="id", columns=["id", "max_upward_dz_m", "orientation_tolerance_rad", "position_tolerance_m"],
            rows=[["clear_top", 0.0, 0.15, 0.01], ["side_approach", 0.0, 0.15, 0.01], ["pick/motion", 0.0, 0.15, 0.003]],
            common=dict(stall_guard_preset="off", tolerance_preset="normal")))
        self.assertEqual(encoded["stage_costs"]["keyed_by"], "id")
        self.assertEqual(decode_compact_value(encoded), value)
        # a map whose entries already have an "id" field, or with one entry, stays
        for kept in ({"a": dict(id="x", n=1, pad="p" * 40), "b": dict(id="y", n=2, pad="p" * 40)},
                     {"a": dict(n=1, pad="p" * 40)}):
            self.assertEqual(encode_compact_value(dict(stage_costs=kept)), dict(stage_costs=kept))
        # planned samples: the measured start state has another shape than the command rows
        start = dict(native_step=0, execution_state="initial_measurement", arms=dict(
            left=self.arm(0.0, first=True), right=self.arm(0.12201002985239029, first=True)))
        commands = [dict(native_step=12 * index, execution_state="completed_command", arms=dict(
            left=dict(jaw_goal_normalized=1.0, phase="hold", q_rad=[0.0, 0.3047229051589966, 0.0]),
            right=dict(jaw_goal_normalized=1.0, phase="move", q_rad=[0.125 * index, 0.3124925792217253, 0.0])))
            for index in range(1, 5)]
        planned = dict(trajectory_feedback=dict(planned=dict(total_samples=43, samples=[start, *commands])))
        table = encode_compact_value(planned)["trajectory_feedback"]["planned"]["samples"]
        self.assertEqual(table["first"], start)
        self.assertEqual(table["columns"], ["arms.left.jaw_goal_normalized", "arms.left.q_rad",
                                            "arms.right.jaw_goal_normalized", "arms.right.q_rad", "native_step"])
        self.assertEqual(len(table["rows"]), 4)
        self.assertNotIn("optional", table)
        self.assertEqual(table["common"]["execution_state"], "completed_command")
        self.assertEqual(decode_compact_value(encode_compact_value(planned)), planned)
        # with two command rows the list stays (a table needs two rows after the first entry) ...
        short = dict(trajectory_feedback=dict(planned=dict(samples=[start, commands[0]])))
        self.assertEqual(encode_compact_value(short), short)
        # ... and a same-shaped list becomes a plain table without "first"
        uniform = dict(trajectory_feedback=dict(planned=dict(samples=commands)))
        self.assertNotIn("first", encode_compact_value(uniform)["trajectory_feedback"]["planned"]["samples"])
        # (its resting left arm is stated once, as in measured samples)
        self.assertEqual(compact_encoding_loss(uniform, decode_compact_value(encode_compact_value(uniform)))["unchanged_arms"], 4)

    def test_an_arm_that_did_not_move_is_stated_once_and_a_moving_jaw_or_joint_keeps_it_in_the_samples(self):
        value = dict(trajectory_feedback=dict(measured=self.measured()))
        encoded = encode_compact_value(value)["trajectory_feedback"]["measured"]
        self.assertEqual(set(encoded[UNCHANGED_ARMS_KEY]), {"left"})
        self.assertEqual(encoded[UNCHANGED_ARMS_KEY]["left"], dict(self.arm(0.0, first=True), unchanged=True,
                                                                  max_abs_joint_change_rad=2e-9))
        self.assertFalse(any(column.startswith("arms.left") for column in encoded["samples"]["columns"]))
        self.assertIn("arms.right.q_rad", encoded["samples"]["columns"])
        self.assertEqual(encoded["samples"]["optional"], ["arms.right.derived_motion"])
        self.assertEqual(encoded["samples"]["common"]["arms.right.derived_motion.source_kind"],
                         "finite_difference_at_measurement_boundaries")
        loss = compact_encoding_loss(value, decode_compact_value(encode_compact_value(value)))
        self.assertEqual((loss["unchanged_arms"], loss["unchanged_abs"], loss["unchanged_derived_abs"]), (3, 2e-9, 0.0))
        for name, changed in dict(
                joint=self.measured(left=(0.0, 0.0, 4 * STATIONARY_RESOLUTION)),
                jaw=self.measured(left_jaw_velocity=-4.023313522338867e-05)).items():
            with self.subTest(name):
                whole = dict(trajectory_feedback=dict(measured=changed))
                moved = encode_compact_value(whole)["trajectory_feedback"]["measured"]
                self.assertNotIn(UNCHANGED_ARMS_KEY, moved)
                self.assertIn("arms.left.q_rad", moved["samples"]["columns"])
                self.assertEqual(compact_encoding_loss(whole, decode_compact_value(encode_compact_value(whole)))
                                 ["unchanged_arms"], 0)

    def test_only_derived_quantities_are_rounded_and_by_at_most_half_a_unit_of_the_fourth_digit(self):
        value = dict(trajectory_feedback=dict(measured=self.measured(left=(0.0, 0.3, 0.6))))
        encoded = encode_compact_value(value)
        decoded = decode_compact_value(encoded)
        loss = compact_encoding_loss(value, decoded)
        self.assertEqual(loss["unchanged_arms"], 0)
        self.assertTrue(0 < loss["derived_relative"] <= 5e-4, loss)
        right = decoded["trajectory_feedback"]["measured"]["samples"][1]["arms"]["right"]
        original = value["trajectory_feedback"]["measured"]["samples"][1]["arms"]["right"]
        # rounding does not depend on the table: the same samples under a path that is not listed are rounded too
        elsewhere = encode_compact_value(dict(other=value["trajectory_feedback"]["measured"]))["other"]
        self.assertIsInstance(elsewhere["samples"], list)
        self.assertEqual(elsewhere["samples"][1]["arms"]["right"][DERIVED_PARENT]["joint_acceleration_rad_s2"], [0.9075, 0.0, 2.465])
        self.assertEqual(right[DERIVED_PARENT]["joint_acceleration_rad_s2"], [0.9075, 0.0, 2.465])
        self.assertEqual(right[DERIVED_PARENT]["ee_linear_velocity_m_s"], [-0.06051, 0.0, 0.3224])
        # measurements, including the measured joint velocity and the interval, keep every digit
        for key in ("q_rad", "qdot_rad_s", "measured_ee_pose"):
            self.assertEqual(right[key], original[key])
        self.assertEqual(right[DERIVED_PARENT]["interval_s"], 0.04)
        # a field of the same name outside derived_motion is a measurement
        outside = dict(joint_acceleration_rad_s2=[0.907532125711441], ee_linear_velocity_m_s=0.060508400201797485)
        self.assertEqual(encode_compact_value(outside), outside)
        self.assertEqual(set(DERIVED_QUANTITIES.values()), {"velocity", "acceleration"})

    def test_with_same_as_and_float32_the_result_is_deterministic_and_has_no_dangling_pointer(self):
        items = ReplyShapingOrderTests().reply()
        result = json.loads(items[0]["text"])
        result["trajectory_feedback"]["measured"] = self.measured()
        result["trajectory_feedback"]["planned"] = dict(samples=deepcopy(result["trajectory_feedback"]["measured"]["samples"]))
        items[0] = dict(items[0], text=canonical(result))

        def finish(content):
            return compact_content_items(encode_compact_items(content), same_as=True, float32=True)

        for profile in reply_budget.ACTION_REPLY_PROFILES:
            for budget in (None, 10 ** 6, 2000, 40):
                if budget is None and profile == "full":
                    continue
                fitted, report = fit_reply_report(items, budget, compact=finish, profile=profile)
                self.assertEqual(dangling_same_as(fitted), [], (profile, budget))
                self.assertEqual(fit_reply_report(deepcopy(items), budget, compact=finish, profile=profile)[0], fitted)
                self.assertEqual(sum(item["type"] == "inputImage" for item in fitted), 3)
        whole = finish(fit_reply(items, 10 ** 6))
        self.assertEqual(whole, fit_reply(items, 10 ** 6, compact=finish))
        # the other order (same_as first, then the encoding) moves the values pointers name
        block = dict(a="y" * 90, i=0)
        repeated = [dict(type="inputText", text=canonical(dict(
            periodic_checks=[dict(n=index, name="row", block=dict(block, i=index)) for index in range(3)], zz=block)))]
        self.assertEqual(json.loads(compact_content_items(repeated, same_as=True)[0]["text"])["zz"],
                         {SAME_AS_KEY: "/periodic_checks/0/block"})
        self.assertTrue(dangling_same_as(encode_compact_items(compact_content_items(repeated, same_as=True))))
        self.assertEqual(dangling_same_as(finish(repeated)), [])
        # in the right order a pointer names a value of the encoded structure (here the table's common string)
        self.assertEqual(_resolve_same_as(json.loads(finish(repeated)[0]["text"]))["zz"], block)
        images = [item for item in items if item["type"] == "inputImage"]
        self.assertTrue(all(any(item is image for item in encode_compact_items(items)) for image in images))
        plain = [dict(type="inputText", text="not json"), dict(type="inputText", text=json.dumps(result, indent=1))]
        self.assertEqual(encode_compact_items(plain), plain)


class SummaryFirstReplyTests(unittest.TestCase):
    """features.action_reply_profile="summary_first": one fixed shape, more inline after an exception."""

    def reply(self, **kwargs):
        items = ActionReplyBudgetTests().reply(**kwargs)
        result = json.loads(items[0]["text"])
        steps = dict(reach=(0, 10), close=(10, 20), lift=(20, 30))
        for name, entry in result["stages"].items():
            entry.update(kind="move", arm="left", grip_facts=dict(measured_aperture_m=0.0481), completed_steps=10,
                         endpoint_error=dict(effective_tolerances=dict(position_tolerance_m=0.01), position_error_m=0.00011646229619681309),
                         grasp_contact_feedback=dict(approach_shortfall_m=4.0171455136239785e-05),
                         robot_specific_report=dict(max_below_path_m=0.0))
        feedback = result["trajectory_feedback"]
        feedback["compact_3d_trajectory"] = dict(unit="m", stages=[
            dict(stage_id=name + "/motion", start_step=first, end_step=last, status=result["stages"][name]["status"])
            for name, (first, last) in steps.items()])
        for path_kind in ("measured", "planned"):
            feedback[path_kind]["samples"] = [dict(native_step=step, q=path_kind[0] * 300) for step in (0, 8, 16, 24, 32, 40)]
        items[0] = dict(items[0], text=canonical(result))
        return items, result

    @staticmethod
    def parts(items):
        return [json.loads(item["text"]) for item in items if item["type"] == "inputText"]

    def test_routine_reply_has_the_fixed_shape_with_or_without_a_budget(self):
        items, original = self.reply()
        fitted, report = fit_reply_report(items, None, profile="summary_first")
        self.assertEqual(report, dict(applied=True, budget_tokens=None, cost_tokens_before=reply_cost_tokens(items),
                                      cost_tokens_after=reply_cost_tokens(fitted), fits=True, profile="summary_first",
                                      routine=True, fixed=list(reply_budget._SUMMARY_ROUTINE),
                                      by_reference=list(reply_budget._SUMMARY_ROUTINE)))
        self.assertEqual(fit_reply_report(items, 10 ** 6, profile="summary_first")[0], fitted)
        head, scene, observation, stages, trajectory = self.parts(fitted)
        self.assertEqual([fitted[1], fitted[2]], [items[5], items[2]])
        self.assertEqual(sum(item["type"] == "inputImage" for item in fitted), 3)
        receipt = dict(original["receipt"])
        receipt.pop("motion_safety_checks")
        self.assertEqual({key: head["receipt"][key] for key in receipt}, receipt)
        self.assertEqual(head["receipt"]["motion_safety_checks_by_reference"]["pass_rows"], 3)
        # a cleanly completed stage keeps every field but grasp_contact_feedback: outcome, enforced tolerances with
        # the exact residual, the measured grip, step counts and any field the profile does not know
        self.assertEqual(reply_budget.STAGE_DETAIL_BY_REFERENCE, ("grasp_contact_feedback",))
        self.assertEqual(stages["stages"], {name: {key: item for key, item in entry.items() if key != "grasp_contact_feedback"}
                                             for name, entry in original["stages"].items()})
        self.assertEqual(stages["stages"]["lift"]["endpoint_error"]["effective_tolerances"], dict(position_tolerance_m=0.01))
        self.assertEqual(stages["stages"]["lift"]["robot_specific_report"], dict(max_below_path_m=0.0))
        self.assertEqual(stages["stages_by_reference"], dict(
            omitted="by_reference", stage_ids=["close", "lift", "reach"], fields=["grasp_contact_feedback"],
            read=dict(tool="robot_action_status", action_id="a1", detail="full"), at="receipt.stage_results"))
        # a robot whose stages have no such field: nothing of a stage is given by reference
        plain = deepcopy(original)
        for entry in plain["stages"].values():
            entry.pop("grasp_contact_feedback")
        bare, bare_report = fit_reply_report([dict(items[0], text=canonical(plain)), *items[1:]], None,
                                             profile="summary_first")
        self.assertNotIn("stage_detail", bare_report["by_reference"])
        self.assertEqual(next(part for part in self.parts(bare) if "stages" in part)["stages"], plain["stages"])
        feedback = trajectory["trajectory_feedback"]
        self.assertEqual(feedback["compact_3d_trajectory"], original["trajectory_feedback"]["compact_3d_trajectory"])
        self.assertEqual(feedback["measured_command_error"], original["trajectory_feedback"]["measured_command_error"])
        for path_kind in ("measured", "planned"):
            self.assertEqual(feedback[path_kind]["samples"]["omitted"], "by_reference")
        # the full profile with room to spare keeps all of it inline, and without a budget does nothing
        self.assertEqual(fit_reply_report(items, 10 ** 6)[1]["by_reference"], [])
        self.assertEqual(fit_reply_report(items, None)[0], items)
        with self.assertRaises(ValueError):
            fit_reply(items, None, profile="minimal")

    def test_exception_reply_keeps_the_measured_samples_around_the_affected_stage_inline(self):
        cases = dict(
            failed=dict(status="failed", reason="tracking_error", failed=("lift",)),
            paused=dict(status="stopped", reason="observation_checkpoint"),
        )
        for name, kwargs in cases.items():
            with self.subTest(name):
                items, original = self.reply(**kwargs)
                if name == "paused":
                    result = deepcopy(original)
                    result["receipt"]["checkpoint_stage_ids"] = ["close"]
                    items[0] = dict(items[0], text=canonical(result))
                fitted, report = fit_reply_report(items, None, profile="summary_first")
                self.assertEqual((report["routine"], report["fixed"]), (False, list(reply_budget._SUMMARY_EXCEPTION)))
                parts = self.parts(fitted)
                head, measured = parts[0], parts[-1]["trajectory_feedback"]["measured"]
                window = dict(failed=[20, 30], paused=[10, 20])[name]
                # every sample inside the affected steps and the nearest one on each side
                self.assertEqual([sample["native_step"] for sample in measured["samples"]],
                                 dict(failed=[16, 24, 32], paused=[8, 16, 24])[name])
                kept = [sample["native_step"] for sample in measured["samples"]]
                outside = [sample["native_step"] for sample in original["trajectory_feedback"]["measured"]["samples"]
                           if sample["native_step"] not in kept]
                # the stand-in names exactly the rows outside the window: without a selector inspect returns three
                self.assertEqual(measured["samples_by_reference"], dict(
                    omitted="by_reference", samples_outside_native_steps=window, count=3,
                    read=dict(tool="robot_action", mode="inspect", action_id="a1", path_kind="measured",
                              native_steps=outside)))
                self.assertEqual(len(outside), 3)
                self.assertEqual(parts[-1]["trajectory_feedback"]["planned"]["samples"]["omitted"], "by_reference")
                if name == "failed":
                    self.assertEqual(head["stages"], dict(lift=original["stages"]["lift"]))
                    self.assertEqual(head["receipt"]["motion_safety_checks"], [original["receipt"]["motion_safety_checks"][2]])
                    self.assertEqual(set(parts[-2]["stages"]), {"reach", "close"})

    def test_exception_without_a_step_window_keeps_all_measured_samples_and_slip_adds_its_onset(self):
        items, original = self.reply(status="failed", reason="tracking_error", failed=("lift",))
        result = deepcopy(original)
        result["trajectory_feedback"]["compact_3d_trajectory"]["stages"] = []
        items[0] = dict(items[0], text=canonical(result))
        fitted, report = fit_reply_report(items, None, profile="summary_first")
        self.assertEqual(report["by_reference"], ["motion_safety_pass_rows", "planned_samples", "stage_detail"])
        self.assertEqual(self.parts(fitted)[-1]["trajectory_feedback"]["measured"], original["trajectory_feedback"]["measured"])
        # a stop at an observe checkpoint has no stage row: the window is the step execution stopped at
        result["receipt"].update(status="stopped", reason="observation_checkpoint", completed_native_steps=40)
        result["stages"]["lift"].update(status="completed", reason=None)
        items[0] = dict(items[0], text=canonical(result))
        fitted, report = fit_reply_report(items, None, profile="summary_first")
        measured = self.parts(fitted)[-1]["trajectory_feedback"]["measured"]
        self.assertEqual([sample["native_step"] for sample in measured["samples"]], [32, 40])
        self.assertEqual((measured["samples_by_reference"]["samples_outside_native_steps"], measured["samples_by_reference"]["count"]),
                         ([40, 40], 4))
        items, original = self.reply()
        result = deepcopy(original)
        result["trajectory_feedback"]["anomaly_reason"] = "slip_detected"
        result["trajectory_feedback"]["compact_3d_trajectory"]["process_trajectory"] = dict(
            co_motion=dict(left=dict(slip_detected=True, slip_onset_step=33)))
        items[0] = dict(items[0], text=canonical(result))
        fitted, report = fit_reply_report(items, None, profile="summary_first")
        self.assertFalse(report["routine"])
        self.assertEqual([sample["native_step"] for sample in self.parts(fitted)[-1]["trajectory_feedback"]["measured"]["samples"]],
                         [32, 40])

    def test_the_host_budget_still_applies_and_continues_in_the_usual_order(self):
        for kwargs, tail in ((dict(), ["compact_3d_trajectory", "stages"]),
                             (dict(status="failed", reason="tracking_error", failed=("lift",)),
                              ["stages", "measured_samples", "compact_3d_trajectory"])):
            items, _ = self.reply(**kwargs)
            shape, fixed_report = fit_reply_report(items, None, profile="summary_first")
            cost = reply_cost_tokens(shape)
            self.assertEqual(fit_reply_report(items, cost * 4, profile="summary_first")[0], shape)
            seen = []
            for budget_tokens in range(cost, 0, -10):
                fitted, report = fit_reply_report(items, budget_tokens * 4, profile="summary_first")
                self.assertEqual(report["by_reference"][:len(report["fixed"])], fixed_report["fixed"])
                extra = report["by_reference"][len(report["fixed"]):]
                self.assertEqual(report["fits"], reply_cost_tokens(fitted) <= budget_tokens)
                if extra not in seen:
                    seen.append(extra)
            self.assertEqual(seen[0], [])
            self.assertEqual(seen[-1], tail)
            for extra in seen:
                self.assertEqual(extra, [name for name in tail if name in extra])
            last = self.parts(fit_reply(items, 40, profile="summary_first"))[-1]["trajectory_feedback"]
            self.assertEqual(set(last["measured"]["samples"]), {"omitted", "read"})
            self.assertNotIn("samples_by_reference", last["measured"])

    def test_detail_entries_and_the_catalogue_note_follow_the_profile(self):
        items, original = self.reply(status="failed", reason="tracking_error", failed=("lift",))
        loaded = frozenset({"robot_action", "program_run", "robot_action_status"})
        fitted = fit_reply(items, None, compact=lambda content: unify_detail_entries(content, loaded), profile="summary_first")
        parts = self.parts(fitted)
        measured = parts[-1]["trajectory_feedback"]["measured"]
        kept = [sample["native_step"] for sample in measured["samples"]]
        outside = [sample["native_step"] for sample in original["trajectory_feedback"]["measured"]["samples"]
                   if sample["native_step"] not in kept]
        self.assertEqual(measured["samples_by_reference"], dict(
            omitted="by_reference", samples_outside_native_steps=[20, 30], count=3,
            detail=dict(tool="robot_action", arguments=dict(mode="inspect", action_id="a1", path_kind="measured",
                                                             native_steps=outside))))
        self.assertEqual(parts[-2]["stages_by_reference"][DETAIL_KEY]["tool"], "robot_action_status")
        self.assertNotIn('"read":', "".join(item.get("text", "") for item in fitted))
        notes = SimpleNamespace(action_reply_profile="summary_first", reply_static_once=True, reply_compact_encoding=True,
                                reply_same_as=True)
        self.assertEqual(reply_compaction.reply_catalogue_notes(notes), [
            reply_budget.SUMMARY_FIRST_CATALOGUE_NOTE, STATIC_ONCE_CATALOGUE_NOTE, reply_encoding.COMPACT_ENCODING_CATALOGUE_NOTE,
            reply_compaction.SAME_AS_CATALOGUE_NOTE])
        self.assertEqual(reply_compaction.reply_catalogue_notes(SimpleNamespace(reply_same_as=True)),
                         [reply_compaction.SAME_AS_CATALOGUE_NOTE])
        self.assertEqual(reply_compaction.reply_catalogue_notes(SimpleNamespace(reply_same_as=False)), [])
        self.assertEqual(reply_compaction.reply_catalogue_notes(SimpleNamespace()), [])
        tools = [dict(name="robot_action", description="Act.\nInput field guide: x")]
        self.assertIs(reply_compaction.catalogue_with_reply_notes(tools, []), tools)
        for key in reply_budget.STAGE_DETAIL_BY_REFERENCE:
            self.assertEqual(reply_budget.SUMMARY_FIRST_CATALOGUE_NOTE.count(key), 2)
            self.assertNotIn(key, reply_budget.summary_first_catalogue_note(stage_detail=False))
        for key in DERIVED_QUANTITIES:
            self.assertIn(key, reply_encoding.COMPACT_ENCODING_CATALOGUE_NOTE)


class LeanV2WithReplySettingsTests(unittest.TestCase):
    """The default payload profile (lean_v2: six-decimal display, no raw frame ids) under the default reply settings:
    one order, the digits each class of number ends with, and no conflict over the program notes."""

    RAW_FRAME = "raw-1d316f67-9940-4e95-8f3e-c8621e4082cb-cam_head"

    def reply(self):
        fixture = ReplyCompactEncodingTests()
        measured = fixture.measured(left=(0.0, 0.3, 0.6))
        measured["samples"][1]["arms"]["right"][DERIVED_PARENT]["jaw_velocity_m_s"] = [-6.835907697677612e-05, 0.0]
        result = dict(
            action_id="a1", action_budget=ReplyStaticOnceTests().result()["action_budget"],
            receipt=dict(status="completed", batch_status="completed", completed_native_steps=42,
                         action_budget=ReplyStaticOnceTests().result()["action_budget"],
                         endpoint_residual_m=dict(position_error_m=0.00011646229619681309, offset_m=0.12260000000000001,
                                                  widened_float32=_float32(1.0667834)),
                         effective_tolerances=dict(position_tolerance_m=0.01)),
            trajectory_feedback=dict(available=True, measured=measured))
        observation = dict(bundle=dict(observation_id="obs_3", cameras=[dict(camera_id="cam_head", frame_id=self.RAW_FRAME,
                                                                             sim_step=3480)]),
                           robot_state=dict(arms=dict(left=dict(q_rad=[0.12201002985239029] * 6))))
        text = lambda value: dict(type="inputText", text=canonical(value))
        return [text(result), text(observation)]

    @staticmethod
    def deliver(items, tool="robot_action"):
        """What Harness.handle_tool does by default: lean_v2 display at presentation, then the reply shaping."""
        from ni_rua.core.presentation.actor_payload import lean_v2_display
        shown = [dict(item, text=canonical(lean_v2_display(json.loads(item["text"]), tool=tool))) for item in items]
        finish = lambda content: compact_content_items(
            encode_compact_items(unify_detail_entries(content, None)), same_as=True, float32=True)
        return fit_reply(drop_static_once(shown), 48000, compact=finish, profile="summary_first", layout=True)

    def test_one_order_gives_one_result_without_dangling_pointers(self):
        items = self.reply()
        delivered = self.deliver(items)
        self.assertEqual(self.deliver(deepcopy(items)), delivered)
        self.assertEqual(dangling_same_as(delivered), [])
        self.assertNotIn(self.RAW_FRAME, "".join(item["text"] for item in delivered))
        self.assertIn('"same_as"', delivered[0]["text"])          # receipt.action_budget -> /action_budget

    def test_digits_of_each_class_of_number(self):
        from support.reply_defaults import action_result, reply_values
        delivered = self.deliver(self.reply())
        result = action_result(delivered)
        residual = result["receipt"]["endpoint_residual_m"]
        # measurements: six decimals (lean_v2); float32 shortening finds nothing shorter afterwards
        self.assertEqual(residual, dict(position_error_m=0.000116, offset_m=0.1226, widened_float32=1.066783))
        self.assertEqual(result["receipt"]["effective_tolerances"], dict(position_tolerance_m=0.01))
        self.assertEqual(reply_values(delivered)[1]["robot_state"]["arms"]["left"]["q_rad"], [0.12201] * 6)
        # with lean_v1 the same measurement keeps its source digits (float32 prints the shortest equal float32)
        legacy = fit_reply(self.reply(), 48000, compact=lambda content: compact_content_items(
            encode_compact_items(content), same_as=True, float32=True), profile="summary_first", layout=True)
        self.assertEqual(action_result(legacy)["receipt"]["endpoint_residual_m"],
                         dict(position_error_m=0.00011646229619681309, offset_m=0.12260000000000001, widened_float32=1.0667834))
        # derived quantities (inline here: unknown outcome keeps the samples): six decimals, then four significant
        # digits, so a small value keeps the digits six decimals leave
        items = self.reply()
        value = json.loads(items[0]["text"])
        value["receipt"]["status"] = "unknown"
        items[0] = dict(items[0], text=canonical(value))
        derived = action_result(self.deliver(items))["trajectory_feedback"]["measured"]["samples"][1]["arms"]["right"][DERIVED_PARENT]
        self.assertEqual(derived["joint_acceleration_rad_s2"], [0.9075, 0.0, 2.465])
        self.assertEqual(derived["ee_linear_velocity_m_s"], [-0.06051, 0.0, 0.3224])
        self.assertEqual(derived["jaw_velocity_m_s"], [-6.8e-05, 0.0])
        self.assertEqual(derived["interval_s"], 0.04)

    def test_program_notes_are_removed_once_and_by_the_profile_first(self):
        from ni_rua.core.presentation.actor_payload import lean_v2_display
        notes = {key: text for key, text in STATIC_ONCE_STRINGS if key in ("result_semantics",)}
        program = dict(status="failed", job_id="job_3", returncode=1, stdout_text="x", result_semantics=notes["result_semantics"],
                       timeout_semantics="work_wall_time", action_summary=dict(action_calls=0, semantics=next(
                           text for key, text in STATIC_ONCE_STRINGS if key == "semantics" and text.startswith("Recorded action"))))
        text = lambda value: dict(type="inputText", text=canonical(value))
        shown = lean_v2_display(program, tool="program_run")
        # lean_v2 leaves the two description sentences out of every program reply, also a failed one ...
        self.assertEqual(set(program) - set(shown), {"result_semantics", "timeout_semantics"})
        # ... and reply_static_once then finds nothing of them; it keeps the remaining note of a failed program
        kept, report = drop_static_once_report([text(shown)])
        self.assertEqual((report["exception"], json.loads(kept[0]["text"])), (True, shown))
        routine = lean_v2_display(dict(program, status="succeeded", returncode=0), tool="program_run")
        dropped, report = drop_static_once_report([text(routine)])
        self.assertEqual(report["dropped"], dict(semantics=1))
        self.assertEqual(json.loads(dropped[0]["text"]), dict(status="succeeded", job_id="job_3", returncode=0,
                                                              stdout_text="x", action_summary=dict(action_calls=0)))


if __name__ == "__main__":
    unittest.main()
