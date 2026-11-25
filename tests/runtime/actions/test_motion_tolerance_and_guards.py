"""
- test_grip_guard_and_trajectory.py: 测试夹爪本体事实/条件门禁、重采样统一与关节行程准入参数化。
- test_motion_tolerance_and_stall.py: 显式容差预设、接触受阻早停与连坐 Bug 解绑回归测试。
"""
import math
import threading
import unittest
import numpy as np
from ni_rua.robot.motion.grip_guard import (
    DEFAULT_MAX_APERTURE_M,
    GRIP_GUARD_PRESETS,
    compute_grip_facts,
    evaluate_grip_guard,
    resolve_grip_guard,
)
from ni_rua.robot.motion.motion_safety import JOINT_TRAVEL_PRESETS, assess_motion
from support.motion_fixtures import batch, jaw, jaws, move
from support.motion_fixtures import preview_adapter
from copy import deepcopy
from ni_rua.robot.motion.motion_tolerance import (
    MAX_DOWNWARD_DZ_M,
    MAX_ORIENTATION_TOLERANCE_RAD,
    ORIENTATION_FREE_TOLERANCE_RAD,
    TOLERANCE_FIELDS,
    TOLERANCE_PRESETS,
    compute_endpoint_residual,
    orientation_is_free,
    resolve_stage_tolerance,
    resolve_stall_guard,
    validate_tolerance_fields,
)
from ni_rua.runtime.actions.unified_actions import _summarize_effective_tolerances, compile_request
from support.motion_fixtures import DualArmHarnessBase
from support.motion_fixtures import periodic_pose, wait_action

NI_RUA_TEST_TIER = "fast"


# --- Folded from tests/test_grip_guard_and_trajectory.py ---

class GripGuardAndTrajectoryTests(unittest.TestCase):
    def execute(self, adapter, action, plan):
        return adapter.execute(
            dict(action, batch_plan_id=plan["plan_id"]),
            lambda update: None,
            threading.Event(),
        )

    def test_compute_grip_facts_returns_objective_measurements_without_empty_grasp_guess(self):
        before = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]
        after = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.25]
        facts = compute_grip_facts(before, after)
        self.assertEqual(facts["measured_jaw_before"], 1.0)
        self.assertEqual(facts["measured_jaw_after"], 0.25)
        self.assertEqual(facts["jaw_delta"], -0.75)
        self.assertAlmostEqual(facts["measured_aperture_m"], 0.25 * DEFAULT_MAX_APERTURE_M, places=6)
        self.assertAlmostEqual(facts["aperture_delta_m"], -0.75 * DEFAULT_MAX_APERTURE_M, places=6)

        adapter, _, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals["left"] = 1.0
        action = batch(
            [jaw(identifier="close", goal=0.0, checkpoint_after_gripper=True), move("lift")],
        )
        plan = adapter.preview_batch(action)
        result = self.execute(adapter, action, plan)
        self.assertEqual(result["status"], "stopped")
        self.assertEqual(result["reason"], "observation_checkpoint")
        self.assertNotIn("likely empty grasp", result["checkpoint_reasons"]["close"])
        stage_facts = result["stage_results"]["close"]["grip_facts"]
        self.assertEqual(stage_facts["measured_jaw_before"], 1.0)
        self.assertEqual(stage_facts["measured_jaw_after"], 0.0)
        self.assertEqual(stage_facts["measured_aperture_m"], 0.0)

    def test_grip_guard_triggers_below_min_above_max_and_passes_within_bounds(self):
        # 1) 低于下界触发 aperture_below_min_m
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals["left"] = 1.0
        action_below = batch([
            jaw(identifier="close", goal=0.0, grip_guard_preset="rigid_object"),
            move("lift"),
        ])
        plan_below = adapter.preview_batch(action_below)
        res_below = self.execute(adapter, action_below, plan_below)
        self.assertEqual(res_below["status"], "stopped")
        self.assertTrue(res_below["paused_at_checkpoint"])
        self.assertEqual(res_below["checkpoint_reason"], "aperture_below_min_m")
        # 护栏触发的暂停与模型自设检查点的 status/reason 相同，靠 checkpoint_reason_code 区分：回复整形按异常处理
        from ni_rua.core.presentation.feedback_delivery import is_requested_checkpoint_receipt
        self.assertEqual((res_below["status"], res_below["reason"], res_below["checkpoint_reason_code"]),
                         ("stopped", "observation_checkpoint", "aperture_below_min_m"))
        self.assertFalse(is_requested_checkpoint_receipt(res_below))
        self.assertEqual(res_below["stage_results"]["close"]["status"], "paused_at_checkpoint")
        self.assertEqual(res_below["stage_results"]["close"]["checkpoint_reason"], "aperture_below_min_m")
        self.assertEqual(res_below["stage_results"]["lift"]["completed_steps"], 0)

        # 2) 显式指定 max_aperture_m 时，高于上界触发 aperture_above_max_m
        adapter2, _, _ = preview_adapter()
        jaws(adapter2)
        adapter2.last_dispatched_goals["left"] = 1.0
        action_above = batch([
            jaw(identifier="close", goal=0.8, grip_guard_preset="thin_object", max_aperture_m=0.025),
            move("lift"),
        ])
        plan_above = adapter2.preview_batch(action_above)
        res_above = self.execute(adapter2, action_above, plan_above)
        self.assertEqual(res_above["status"], "stopped")
        self.assertTrue(res_above["paused_at_checkpoint"])
        self.assertEqual(res_above["checkpoint_reason"], "aperture_above_max_m")
        self.assertEqual(res_above["stage_results"]["close"]["status"], "paused_at_checkpoint")
        self.assertEqual(res_above["stage_results"]["close"]["checkpoint_reason"], "aperture_above_max_m")

        # 3) 处于 [min_aperture_m, max_aperture_m] 区间内连续执行完成
        adapter3, _, _ = preview_adapter()
        jaws(adapter3)
        adapter3.last_dispatched_goals["left"] = 1.0
        action_ok = batch([
            jaw(identifier="close", goal=0.25, min_aperture_m=0.010, max_aperture_m=0.035),
            move("lift"),
        ])
        plan_ok = adapter3.preview_batch(action_ok)
        res_ok = self.execute(adapter3, action_ok, plan_ok)
        self.assertEqual(res_ok["status"], "completed")
        self.assertAlmostEqual(
            res_ok["stage_results"]["close"]["grip_facts"]["measured_aperture_m"],
            0.25 * DEFAULT_MAX_APERTURE_M,
            places=6,
        )

    def test_explicit_checkpoint_after_gripper_does_not_skip_opening(self):
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals["left"] = 0.0
        action = batch([
            jaw(identifier="open", goal=1.0, checkpoint_after_gripper=True),
            move("retract"),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(len(plan["checkpoints"]), 1)
        self.assertEqual(plan["checkpoints"][0]["stage_ids"], ["open"])
        res = self.execute(adapter, action, plan)
        self.assertEqual(res["status"], "stopped")
        self.assertEqual(res["checkpoint_stage_ids"], ["open"])

    def test_resample_action_plan_scale_one_preserves_time_domain_scurve_deceleration(self):
        adapter, _, _ = preview_adapter()
        # 构造带 S-curve 末端减速的原规划轨迹：在直线关节路径上，末端步长小于中间巡航步长
        t = np.linspace(0.0, 1.0, 11)
        s = 3.0 * t**2 - 2.0 * t**3
        scurve = np.zeros((11, 6))
        scurve[:, 0] = 0.40 * s
        rows = adapter._resample_action_plan(scurve, "left", {"speed_scale": 1.0}, start_q=np.zeros(6))
        np.testing.assert_allclose(rows[-1], scurve[-1])
        deltas = np.abs(np.diff(np.vstack([np.zeros(6), rows])[:, 0]))
        # 保留原规划时间域 S-curve 减速特征：最后一步增量显著小于峰值巡航增量
        self.assertLess(deltas[-1], deltas.max() * 0.6)

    def test_joint_travel_presets_and_numeric_overrides(self):
        names = [f"joint{i}" for i in range(1, 7)]
        types = ["revolute"] * 6
        limits = [[-6.28, 6.28]] * 6
        start = np.zeros(6)
        # 构造总行程 4.0 rad (> pi, <= 1.5*pi)，回撤 2.0 rad (> pi/2, <= pi) 的轨迹
        positions = np.array([
            [2.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        # 默认 standard 下被拒绝
        std = assess_motion(start, positions, joint_names=names, joint_types=types, joint_limits=limits)
        self.assertEqual(std["status"], "reject")
        self.assertIn("abnormal_joint_travel", std["reasons"])

        # extended 预设下通过
        ext = assess_motion(
            start, positions, joint_names=names, joint_types=types, joint_limits=limits,
            joint_travel_preset="extended",
        )
        self.assertEqual(ext["status"], "pass")
        self.assertEqual(ext["thresholds"]["min_total_travel_rad"], JOINT_TRAVEL_PRESETS["extended"]["min_total_travel_rad"])

        # 显式数值覆盖通过
        custom = assess_motion(
            start, positions, joint_names=names, joint_types=types, joint_limits=limits,
            max_joint_travel_rad=4.5, max_backtracking_rad=2.5,
        )
        self.assertEqual(custom["status"], "pass")
        self.assertEqual(custom["thresholds"]["min_total_travel_rad"], 4.5)
        self.assertEqual(custom["thresholds"]["min_periodic_or_backtracking_excess_rad"], 2.5)

        # unconstrained 跳过行程检查但仍保留步数预算检查
        huge = np.array([
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ])
        unc = assess_motion(
            start, huge, joint_names=names, joint_types=types, joint_limits=limits,
            joint_travel_preset="unconstrained", native_steps=10, request_remaining=5,
        )
        self.assertEqual(unc["status"], "reject")
        self.assertEqual(unc["reasons"], ["insufficient_remaining_native_budget"])
        self.assertEqual(unc["witnesses"], [])

    def test_unified_actions_compile_request_propagates_grip_guard_and_joint_travel(self):
        from types import SimpleNamespace
        from ni_rua.runtime.actions.unified_actions import compile_request
        harness = SimpleNamespace(
            identity={
                "run_id": "r1",
                "episode_id": "e1",
                "env_instance_id": "i1",
                "env_generation": 1,
            },
            capabilities={
                "frame_id": "env_origin",
                "tcp_config_id": "arx-x5-link6-v1",
                "limits_profile": "simulation-slow-v1",
                "motion_execution_mode": "resampled",
                "dual_arm": True,
            },
            config=SimpleNamespace(features=SimpleNamespace(default_motion_speed_scale=1.0)),
            lock=threading.Lock(),
            store=SimpleNamespace(record_sequence=lambda kind, prefix: 0),
        )
        compiled = compile_request(
            harness,
            {
                "observation_id": "obs-1",
                "grip_guard_preset": "rigid_object",
                "joint_travel_preset": "extended",
                "stages": [
                    {
                        "id": "grasp_it",
                        "kind": "move",
                        "arm": "left",
                        "grasp_position_m": [0.3, 0.1, 0.8],
                        "angles_deg": [0.0, 0.0, 0.0],
                        "gripper_goal_normalized": 0.0,
                        "min_aperture_m": 0.005,
                        "max_aperture_m": 0.060,
                        "max_joint_travel_rad": 5.0,
                        "max_backtracking_rad": 2.5,
                    }
                ],
            },
        )
        self.assertEqual(compiled["grip_guard_preset"], "rigid_object")
        self.assertEqual(compiled["joint_travel_preset"], "extended")
        self.assertEqual(len(compiled["stages"]), 2)
        motion_stage, jaw_stage = compiled["stages"]
        self.assertEqual(motion_stage["id"], "grasp_it/motion")
        self.assertEqual(motion_stage["max_joint_travel_rad"], 5.0)
        self.assertEqual(motion_stage["max_backtracking_rad"], 2.5)
        self.assertEqual(jaw_stage["id"], "grasp_it")
        self.assertEqual(jaw_stage["min_aperture_m"], 0.005)
        self.assertEqual(jaw_stage["max_aperture_m"], 0.060)

    def test_physical_jaw_ignores_negative_overdrive_gripper_scale(self):
        """回归测试：当 robot.gripper_scale=[-0.01, 0.044] 含负向过压区间时，空抓 position_m=[0.0, 0.0] 必须报告 0.0m 而非 16.3mm。"""
        adapter, _, _ = preview_adapter()
        robot = adapter.robot.robot_list[0]
        robot.gripper_joints_name = ["joint7", "joint8"]
        robot.gripper_move = dict(base="joint7", sign=1)
        robot.gripper_scale = [-0.010, 0.044]
        original = adapter.state

        def state():
            result = original()
            for arm in ("left", "right"):
                goal = adapter.last_dispatched_goals[arm]
                phys_pos = 0.044 * float(goal)
                result["arms"][arm]["gripper"] = dict(
                    position_source_kind="measurement",
                    position_m=[phys_pos, phys_pos],
                    joint_names=["joint7", "joint8"],
                )
            return result

        adapter.state = state
        adapter.last_dispatched_goals["left"] = 1.0
        action_empty = batch([
            jaw(identifier="close", goal=0.0, grip_guard_preset="rigid_object"),
            move("lift"),
        ])
        plan = adapter.preview_batch(action_empty)
        res = self.execute(adapter, action_empty, plan)
        self.assertEqual(res["status"], "stopped")
        self.assertEqual(res["checkpoint_reason"], "aperture_below_min_m")
        facts = res["stage_results"]["close"]["grip_facts"]
        self.assertEqual(facts["measured_jaw_before"], 1.0)
        self.assertEqual(facts["measured_jaw_after"], 0.0)
        self.assertEqual(facts["measured_aperture_m"], 0.0)

    def test_mid_move_in_hand_slip_triggers_grip_guard_checkpoint_before_carry(self):
        """回归测试：close 阶段结束时夹住锥面 (25mm >= min_aperture_m)，但随后 lift 阶段中物体滑脱闭合至 0mm 时，必须在 lift 结束时暂停拦截后续 carry。"""
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals["left"] = 1.0
        action_slip = batch([
            jaw(identifier="close", goal=0.0, grip_guard_preset="rigid_object"),
            move("lift", angle=-0.125),
            move("carry", angle=-0.25),
        ])
        plan = adapter.preview_batch(action_slip)
        close_end = plan["stage_costs"]["close"]["end_native_step"]
        lift_end = plan["stage_costs"]["lift"]["end_native_step"]
        original = adapter.state

        def slip_during_lift_state():
            st = original()
            step = adapter.env.take_action_cnt[0]
            if 1 <= step <= close_end:
                st["arms"]["left"]["gripper"]["position_m"] = [0.30]
            elif step >= lift_end:
                st["arms"]["left"]["gripper"]["position_m"] = [0.0]
            return st

        adapter.state = slip_during_lift_state
        res = self.execute(adapter, action_slip, plan)
        self.assertEqual(res["status"], "stopped")
        self.assertEqual(res["reason"], "observation_checkpoint")
        self.assertTrue(res["paused_at_checkpoint"])
        self.assertEqual(res["checkpoint_reason"], "aperture_below_min_m")
        self.assertEqual(res["checkpoint_stage_ids"], ["lift"])
        self.assertEqual(res["stage_results"]["close"]["status"], "completed")
        self.assertEqual(res["stage_results"]["lift"]["status"], "paused_at_checkpoint")
        self.assertEqual(res["stage_results"]["carry"]["completed_steps"], 0)

        # 2. 单次待定门禁 (pending_guard) 仅在闭爪后紧邻的第一段 move (lift) 生效，后续 carry/place 阶段不再常驻拦截薄物
        adapter2, _, _ = preview_adapter()
        jaws(adapter2)
        adapter2.last_dispatched_goals["left"] = 1.0
        action_carry_slip = batch([
            jaw(identifier="close", goal=0.0, grip_guard_preset="non_empty"),
            move("lift", angle=-0.125),
            move("carry", angle=-0.25),
            move("place", angle=-0.375),
        ])
        plan2 = adapter2.preview_batch(action_carry_slip)
        lift2_end = plan2["stage_costs"]["lift"]["end_native_step"]
        carry2_end = plan2["stage_costs"]["carry"]["end_native_step"]
        original2 = adapter2.state

        def slip_during_carry_state():
            st = original2()
            step = adapter2.env.take_action_cnt[0]
            if 1 <= step <= lift2_end:
                st["arms"]["left"]["gripper"]["position_m"] = [0.059]
            elif step >= carry2_end:
                st["arms"]["left"]["gripper"]["position_m"] = [0.0]
            return st

        adapter2.state = slip_during_carry_state
        res2 = self.execute(adapter2, action_carry_slip, plan2)
        self.assertEqual(res2["status"], "completed")
        self.assertEqual(res2["completed_stage_ids"], ["close", "lift", "carry", "place"])

    def test_carry_slip_guard_validation_and_default_invariance(self):
        from ni_rua.robot.motion.batch_contract import validate_batch_action
        from ni_rua.robot.motion.grip_guard import validate_grip_guard_fields

        # Default invariance: 3-key dict when no carry_slip fields are passed
        plain = validate_grip_guard_fields({"grip_guard_preset": "rigid_object"})
        self.assertEqual(set(plain.keys()), {"grip_guard_preset", "min_aperture_m", "max_aperture_m"})

        # Explicit carry_slip_guard=False returns 6-key dict and preserves nominal runtime behavior
        disabled = validate_grip_guard_fields({"grip_guard_preset": "rigid_object", "carry_slip_guard": False})
        self.assertEqual(disabled["carry_slip_guard"], False)
        self.assertEqual(disabled["carry_slip_scope"], "next_move")
        self.assertEqual(disabled["slip_consecutive_steps"], 1)

        # Validation rejections with invalid_carry_slip_* codes
        bad_mappings = (
            ({"grip_guard_preset": "rigid_object", "carry_slip_guard": "yes"}, "invalid_carry_slip_guard"),
            ({"grip_guard_preset": "rigid_object", "carry_slip_scope": "next_move"}, "invalid_carry_slip_guard_disabled"),
            ({"grip_guard_preset": "rigid_object", "carry_slip_guard": False, "slip_consecutive_steps": 2}, "invalid_carry_slip_guard_disabled"),
            ({"grip_guard_preset": "none", "carry_slip_guard": True}, "invalid_carry_slip_requires_min_aperture"),
            ({"carry_slip_guard": True}, "invalid_carry_slip_requires_min_aperture"),
            ({"grip_guard_preset": "rigid_object", "carry_slip_guard": True, "carry_slip_scope": "forever"}, "invalid_carry_slip_scope"),
            ({"grip_guard_preset": "rigid_object", "carry_slip_guard": True, "slip_consecutive_steps": 0}, "invalid_carry_slip_consecutive_steps"),
            ({"grip_guard_preset": "rigid_object", "carry_slip_guard": True, "slip_consecutive_steps": 51}, "invalid_carry_slip_consecutive_steps"),
            ({"grip_guard_preset": "rigid_object", "carry_slip_guard": True, "slip_consecutive_steps": True}, "invalid_carry_slip_consecutive_steps"),
        )
        for mapping, expected_code in bad_mappings:
            with self.subTest(mapping=mapping), self.assertRaisesRegex(ValueError, expected_code):
                validate_grip_guard_fields(mapping)

        # Batch-level & stage-level rejections via validate_batch_action
        with self.assertRaisesRegex(ValueError, "batch_observe_requires_reason"):
            validate_batch_action(batch([dict(id="obs", kind="observe", reason="look", carry_slip_guard=True)]))
        with self.assertRaisesRegex(ValueError, "invalid_carry_slip_requires_jaw_stage"):
            validate_batch_action(batch([move("m1", carry_slip_guard=True, min_aperture_m=0.01)]))
        with self.assertRaisesRegex(ValueError, "invalid_carry_slip_requires_min_aperture"):
            validate_batch_action(batch([jaw("close", goal=0.0, grip_guard_preset="none")], carry_slip_guard=True))

    def test_carry_slip_guard_per_step_pause_scopes_streak_and_release_disarm(self):
        # 1) Mid-transport drop on step k=1 of multi-step lift (scope="next_move") stops immediately before remaining lift steps
        adapter, _, dispatched = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals["left"] = 1.0
        action = batch([
            jaw("close", goal=0.0, grip_guard_preset="rigid_object", carry_slip_guard=True),
            move("lift", angle=-0.25),
            move("carry", angle=-0.50),
        ])
        plan = adapter.preview_batch(action)
        close_end = plan["stage_costs"]["close"]["end_native_step"]
        lift_steps = plan["stage_costs"]["lift"]["native_steps"]
        self.assertGreaterEqual(lift_steps, 4)
        orig_state = adapter.state

        def drop_on_lift_step_1():
            st = orig_state()
            step = adapter.env.take_action_cnt[0]
            if 1 <= step <= close_end + 1:
                st["arms"]["left"]["gripper"]["position_m"] = [0.30]
            elif step > close_end + 1:
                st["arms"]["left"]["gripper"]["position_m"] = [0.0]
            return st

        adapter.state = drop_on_lift_step_1
        res = self.execute(adapter, action, plan)
        self.assertEqual(res["status"], "stopped")
        self.assertEqual(res["reason"], "observation_checkpoint")
        self.assertEqual(res["checkpoint_stage_id"], "lift")
        self.assertEqual(res["checkpoint_reason"], "carry_slip_aperture_below_min_m")
        self.assertEqual(res["checkpoint_reason_code"], "carry_slip_aperture_below_min_m")
        self.assertEqual(res["executed_native_steps"], close_end + 2)
        self.assertEqual(len(dispatched), close_end + 2)
        self.assertEqual(res["stage_results"]["lift"]["status"], "paused_at_checkpoint")
        self.assertEqual(res["stage_results"]["lift"]["completed_steps"], 2)
        self.assertEqual(res["stage_results"]["carry"]["status"], "pending")
        self.assertEqual(res["stage_results"]["carry"]["completed_steps"], 0)
        self.assertEqual(res["effective_grip_guard"]["carry_slip_guard"], True)
        self.assertEqual(res["effective_grip_guard"]["carry_slip_scope"], "next_move")
        self.assertEqual(
            res["slip_detected_at"],
            {
                "stage_id": "lift",
                "arm": "left",
                "step_index": 1,
                "measured_aperture_m": 0.0,
                "min_aperture_m": 0.002,
                "consecutive_steps": 1,
            },
        )

        # 2) Scope "next_move" vs "until_release": drop during 2nd move ("carry") is ignored by "next_move" but caught by "until_release"
        for scope, expected_status in (("next_move", "completed"), ("until_release", "stopped")):
            with self.subTest(scope=scope):
                ad, _, _ = preview_adapter()
                jaws(ad)
                ad.last_dispatched_goals["left"] = 1.0
                act = batch([
                    jaw("close", goal=0.0, grip_guard_preset="rigid_object", carry_slip_guard=True, carry_slip_scope=scope),
                    move("lift", angle=-0.125),
                    move("carry", angle=-0.375),
                ])
                pl = ad.preview_batch(act)
                lift_end = pl["stage_costs"]["lift"]["end_native_step"]
                base_st = ad.state

                def drop_in_carry():
                    s = base_st()
                    step = ad.env.take_action_cnt[0]
                    if 1 <= step <= lift_end + 1:
                        s["arms"]["left"]["gripper"]["position_m"] = [0.30]
                    elif step > lift_end + 1:
                        s["arms"]["left"]["gripper"]["position_m"] = [0.0]
                    return s

                ad.state = drop_in_carry
                out = self.execute(ad, act, pl)
                self.assertEqual(out["status"], expected_status)
                if expected_status == "stopped":
                    self.assertEqual(out["checkpoint_stage_id"], "carry")
                    self.assertEqual(out["checkpoint_reason_code"], "carry_slip_aperture_below_min_m")
                    self.assertEqual(out["slip_detected_at"]["step_index"], 1)

        # 3) Transient 1-step dip absorbed when slip_consecutive_steps=2; 2 consecutive dips trigger
        for two_dips, expected_status in ((False, "completed"), (True, "stopped")):
            with self.subTest(two_dips=two_dips):
                ad, _, _ = preview_adapter()
                jaws(ad)
                ad.last_dispatched_goals["left"] = 1.0
                act = batch([
                    jaw("close", goal=0.0, grip_guard_preset="rigid_object", carry_slip_guard=True, slip_consecutive_steps=2),
                    move("lift", angle=-0.25),
                ])
                pl = ad.preview_batch(act)
                c_end = pl["stage_costs"]["close"]["end_native_step"]
                base_st = ad.state

                def dip_state():
                    s = base_st()
                    step = ad.env.take_action_cnt[0]
                    if step >= 1:
                        dip_steps = {c_end + 1, c_end + 2} if two_dips else {c_end + 1}
                        s["arms"]["left"]["gripper"]["position_m"] = [0.0 if step in dip_steps else 0.30]
                    return s

                ad.state = dip_state
                out = self.execute(ad, act, pl)
                self.assertEqual(out["status"], expected_status)
                if expected_status == "stopped":
                    self.assertEqual(out["slip_detected_at"]["consecutive_steps"], 2)
                    self.assertEqual(out["slip_detected_at"]["step_index"], 1)

        # 4) Disarm on release (goal >= 0.5): both standalone jaw release and during_motion jaw opening disarm until_release
        ad_rel, _, _ = preview_adapter()
        jaws(ad_rel)
        ad_rel.last_dispatched_goals["left"] = 1.0
        during_open = move("open_retract", angle=-0.375)
        during_open["action"]["gripper_events"] = [
            dict(arm="left", phase="during_motion", goal_normalized=0.8, start_fraction=0.0, end_fraction=1.0)
        ]
        act_rel = batch([
            jaw("close", goal=0.0, grip_guard_preset="rigid_object", carry_slip_guard=True, carry_slip_scope="until_release"),
            move("lift", angle=-0.125),
            during_open,
            move("retreat", angle=-0.50),
        ])
        pl_rel = ad_rel.preview_batch(act_rel)
        lift_rel_end = pl_rel["stage_costs"]["lift"]["end_native_step"]
        base_rel_st = ad_rel.state

        def open_then_zero_aperture():
            s = base_rel_st()
            step = ad_rel.env.take_action_cnt[0]
            if step >= 1:
                s["arms"]["left"]["gripper"]["position_m"] = [0.30 if step <= lift_rel_end else 0.0]
            return s

        ad_rel.state = open_then_zero_aperture
        out_rel = self.execute(ad_rel, act_rel, pl_rel)
        self.assertEqual(out_rel["status"], "completed")
        self.assertEqual(out_rel["completed_stage_ids"], ["close", "lift", "open_retract", "retreat"])



# --- Folded from tests/test_motion_tolerance_and_stall.py ---

class MotionToleranceAndStallTests(unittest.TestCase):
    setUp = DualArmHarnessBase.setUp

    def execute_batch(self, adapter, action, plan):
        return adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda update: None, threading.Event())

    def test_tolerance_preset_and_priority_resolution(self):
        # 默认回退为 normal (0.010m, 0.15rad)
        default_tol = resolve_stage_tolerance({})
        self.assertEqual(default_tol, {
            'tolerance_preset': 'normal',
            'position_tolerance_m': 0.010,
            'orientation_tolerance_rad': 0.15,
        })
        # 各预设值核对（含 tight=3mm 与 compliant/seating 向上顺应 40mm）
        self.assertAlmostEqual(TOLERANCE_PRESETS['tight']['position_tolerance_m'], 0.003, places=5)
        self.assertAlmostEqual(TOLERANCE_PRESETS['tight']['orientation_tolerance_rad'], 0.05, places=5)
        self.assertAlmostEqual(TOLERANCE_PRESETS['compliant']['max_upward_dz_m'], 0.040, places=5)
        self.assertAlmostEqual(TOLERANCE_PRESETS['seating']['max_upward_dz_m'], 0.040, places=5)
        for name, expected in TOLERANCE_PRESETS.items():
            resolved = resolve_stage_tolerance({'tolerance_preset': name})
            self.assertEqual(resolved['tolerance_preset'], name)
            self.assertAlmostEqual(resolved['position_tolerance_m'], expected['position_tolerance_m'], places=5)
            self.assertAlmostEqual(resolved['orientation_tolerance_rad'], expected['orientation_tolerance_rad'], places=5)
            if 'max_upward_dz_m' in expected:
                self.assertAlmostEqual(resolved['max_upward_dz_m'], expected['max_upward_dz_m'], places=5)
            else:
                self.assertNotIn('max_upward_dz_m', resolved)

        # 优先级链：stage 显式数值 > stage preset > batch 显式数值 > batch preset > 默认
        batch_defaults = {
            'tolerance_preset': 'tight',
            'position_tolerance_m': 0.018,
        }
        # 继承 batch 预设 + batch 显式覆盖
        inh = resolve_stage_tolerance({}, batch_defaults)
        self.assertEqual(inh['tolerance_preset'], 'tight')
        self.assertAlmostEqual(inh['position_tolerance_m'], 0.018)
        self.assertAlmostEqual(inh['orientation_tolerance_rad'], 0.05)
        self.assertNotIn('max_upward_dz_m', inh)

        # stage preset 覆盖 batch
        stage_p = resolve_stage_tolerance({'tolerance_preset': 'coarse'}, batch_defaults)
        self.assertEqual(stage_p['tolerance_preset'], 'coarse')
        self.assertAlmostEqual(stage_p['position_tolerance_m'], 0.025)
        self.assertAlmostEqual(stage_p['orientation_tolerance_rad'], 0.35)

        # stage 显式数值只覆盖对应维度，另一维度保留 stage/batch preset
        stage_num = resolve_stage_tolerance(
            {'tolerance_preset': 'coarse', 'position_tolerance_m': 0.04},
            batch_defaults,
        )
        self.assertAlmostEqual(stage_num['position_tolerance_m'], 0.04)
        self.assertAlmostEqual(stage_num['orientation_tolerance_rad'], 0.35)

        # max_upward_dz_m 完整四级优先级链与非顺应 stage preset 重置语义
        inh_comp = resolve_stage_tolerance({}, {'tolerance_preset': 'compliant'})
        self.assertAlmostEqual(inh_comp['max_upward_dz_m'], 0.040, places=5)
        inh_comp_num = resolve_stage_tolerance({}, {'tolerance_preset': 'compliant', 'max_upward_dz_m': 0.032})
        self.assertAlmostEqual(inh_comp_num['max_upward_dz_m'], 0.032, places=5)
        reset_tight = resolve_stage_tolerance({'tolerance_preset': 'tight'}, {'tolerance_preset': 'compliant'})
        self.assertNotIn('max_upward_dz_m', reset_tight)
        reset_normal = resolve_stage_tolerance(
            {'tolerance_preset': 'normal'},
            {'tolerance_preset': 'compliant', 'max_upward_dz_m': 0.035},
        )
        self.assertNotIn('max_upward_dz_m', reset_normal)
        stage_seating_over_batch = resolve_stage_tolerance(
            {'tolerance_preset': 'seating'},
            {'tolerance_preset': 'tight', 'max_upward_dz_m': 0.012},
        )
        self.assertAlmostEqual(stage_seating_over_batch['max_upward_dz_m'], 0.040, places=5)
        stage_explicit_dz = resolve_stage_tolerance(
            {'tolerance_preset': 'tight', 'max_upward_dz_m': 0.025},
            {'tolerance_preset': 'compliant'},
        )
        self.assertAlmostEqual(stage_explicit_dz['max_upward_dz_m'], 0.025, places=5)

    def test_free_orientation_threshold_matches_schema_maximum_and_preset(self):
        # Regression: position-only IK required >= math.pi, above both the schema maximum and this preset.
        preset = TOLERANCE_PRESETS['unconstrained_orientation']['orientation_tolerance_rad']
        self.assertEqual(ORIENTATION_FREE_TOLERANCE_RAD, 3.14159)
        self.assertLessEqual(ORIENTATION_FREE_TOLERANCE_RAD, MAX_ORIENTATION_TOLERANCE_RAD)
        for value in (preset, MAX_ORIENTATION_TOLERANCE_RAD, 3.14159265, math.pi):
            self.assertTrue(orientation_is_free(value), value)
        for value in (0.15, 0.6, 3.0, 3.14158):
            self.assertFalse(orientation_is_free(value), value)
        resolved = resolve_stage_tolerance({'tolerance_preset': 'unconstrained_orientation'})
        self.assertTrue(orientation_is_free(resolved['orientation_tolerance_rad']))

    def test_stall_guard_preset_and_priority_resolution(self):
        self.assertEqual(resolve_stall_guard({}), {
            'stall_guard_preset': 'off',
            'max_tracking_divergence_m': None,
        })
        self.assertEqual(resolve_stall_guard({'stall_guard_preset': 'contact_stop'}), {
            'stall_guard_preset': 'contact_stop',
            'max_tracking_divergence_m': 0.020,
        })
        self.assertEqual(resolve_stall_guard({'stall_guard_preset': 'strict'}), {
            'stall_guard_preset': 'strict',
            'max_tracking_divergence_m': 0.010,
        })
        # 优先级：stage 显式 > stage preset > batch 显式 > batch preset
        batch_def = {'stall_guard_preset': 'contact_stop', 'max_tracking_divergence_m': 0.03}
        self.assertEqual(resolve_stall_guard({}, batch_def)['max_tracking_divergence_m'], 0.03)
        self.assertEqual(resolve_stall_guard({'stall_guard_preset': 'strict'}, batch_def)['max_tracking_divergence_m'], 0.010)
        self.assertEqual(resolve_stall_guard({'stall_guard_preset': 'off'}, batch_def)['max_tracking_divergence_m'], None)
        self.assertEqual(
            resolve_stall_guard({'stall_guard_preset': 'strict', 'max_tracking_divergence_m': 0.04}, batch_def)['max_tracking_divergence_m'],
            0.04,
        )

    def test_compute_endpoint_residual_signed_components(self):
        target = {'position_m': [0.20, -0.10, 0.80], 'quaternion_xyzw': [0.0, 0.0, 0.0, 1.0]}
        measured = {'position_m': [0.212, -0.105, 0.785], 'quaternion_xyzw': [0.0, 0.0, math.sin(0.05), math.cos(0.05)]}
        res = compute_endpoint_residual(target, measured)
        self.assertAlmostEqual(res['dx_m'], 0.012, places=5)
        self.assertAlmostEqual(res['dy_m'], -0.005, places=5)
        self.assertAlmostEqual(res['dz_m'], -0.015, places=5)
        self.assertAlmostEqual(res['position_error_m'], round(math.sqrt(0.012**2 + 0.005**2 + 0.015**2), 5), places=5)
        self.assertAlmostEqual(res['orientation_error_rad'], 0.10, places=4)

    def test_coarse_tolerance_preset_allows_larger_arrival_error(self):
        adapter, _, _ = preview_adapter()
        action = batch([move('transit', arm='left', angle=-0.25, tolerance_preset='coarse')])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        original_state = adapter.state

        # 在终点注入 12mm 位置偏差与 0.20 rad 姿态偏差（超过 normal 5mm/0.1rad，但在 coarse 15mm/0.25rad 内）
        def perturbed_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= plan['planned_native_steps']:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [pose['position_m'][0] + 0.012, pose['position_m'][1], pose['position_m'][2]]
                angle = -0.25 + 0.20
                pose['quaternion_xyzw'] = [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)]
            return st

        adapter.state = perturbed_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        residual = result['stage_results']['transit']['endpoint_residual_m']
        self.assertAlmostEqual(residual['dx_m'], 0.012, places=4)
        self.assertAlmostEqual(residual['orientation_error_rad'], 0.20, places=4)
        self.assertEqual(result['endpoint_residual_m'], residual)

    def test_stall_guard_contact_stop_triggers_early_before_stage_completion(self):
        adapter, _, dispatched = preview_adapter()
        # angle=-0.5 规划 10 步
        action = batch([move('descent', arm='left', angle=-0.5, stall_guard_preset='contact_stop')])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['planned_native_steps'], 10)

        original_state = adapter.state
        frozen_pose = deepcopy(original_state()['arms']['left']['ee_pose'])

        # 模拟接触受阻：机械臂末端位置卡在初始位置不动，随规划推进与规划轨迹偏离超过 25mm
        def blocked_state():
            st = original_state()
            st['arms']['left']['ee_pose'] = deepcopy(frozen_pose)
            return st

        adapter.state = blocked_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'tracking_error')
        self.assertEqual(result['failed_phase'], 'stall_guard_stopped')
        self.assertEqual(result['failed_stage_id'], 'descent')
        # 连续 2 步超过 0.025m 即停机，远早于 10 步
        self.assertLess(len(dispatched), 5)
        self.assertEqual(result['executed_native_steps'], len(dispatched))
        self.assertIn('first_divergence', result)
        self.assertIn('first_divergence', result['stage_results']['descent'])
        self.assertIn('endpoint_residual_m', result)
        self.assertEqual(result['stage_results']['descent']['failed_phase'], 'stall_guard_stopped')

    def test_stall_guard_ignores_moving_pd_lag_and_contact_stop_accepts_compliant_endpoint(self):
        adapter, _, dispatched = preview_adapter()
        action = batch([move('sweep', arm='left', angle=-0.5, stall_guard_preset='contact_stop', position_tolerance_m=0.020)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        total_steps = plan['planned_native_steps']
        original_state = adapter.state

        # 模拟运动中的 PD 跟随滞后（滞后 22mm > 20mm，但每步仍在持续推进 > 1.5mm），以及终点 18mm 接触顺应偏差（<= 20mm）
        def lagging_but_moving_state():
            st = original_state()
            step = adapter.env.take_action_cnt[0]
            pos = list(st['arms']['left']['ee_pose']['position_m'])
            if 0 < step < total_steps:
                pos[0] -= 0.022
            elif step >= total_steps:
                pos[0] -= 0.018
            st['arms']['left']['ee_pose']['position_m'] = pos
            return st

        adapter.state = lagging_but_moving_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(dispatched), total_steps)
        self.assertAlmostEqual(result['stage_results']['sweep']['endpoint_error']['position_error_m'], 0.018, places=4)

    def test_pure_gripper_stage_ignores_ee_compliance_shift_and_records_metric(self):
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        action = batch([jaw(identifier='close_jaw', arm='left', goal=0.0)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')

        original_state = adapter.state

        # 模拟闭合夹爪时因接触物体产生的 15mm 末端顺应偏移（超过默认 5mm/10mm 位置容差）
        def compliant_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= 1:
                pos = st['arms']['left']['ee_pose']['position_m']
                st['arms']['left']['ee_pose']['position_m'] = [pos[0] + 0.015, pos[1], pos[2]]
            return st

        adapter.state = compliant_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertIsNone(result['reason'])
        self.assertAlmostEqual(result['stage_results']['close_jaw']['compliance_shift_m'], 0.015, places=5)

    def test_passive_arm_drift_does_not_fail_final_held_on_single_arm_batch(self):
        adapter, _, _ = preview_adapter()
        # 仅左臂运动，右臂在本批次无任何阶段
        action = batch([move('left_only', arm='left', angle=-0.125)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')

        original_state = adapter.state

        # 模拟被动右臂在左臂运动期间发生 8mm 控制器漂移（超过 5mm）
        def passive_drift_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= 1:
                pos = st['arms']['right']['ee_pose']['position_m']
                st['arms']['right']['ee_pose']['position_m'] = [pos[0], pos[1] - 0.008, pos[2]]
            return st

        adapter.state = passive_drift_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertIsNone(result['reason'])
        self.assertAlmostEqual(result['passive_arm_drift_m'], 0.008, places=5)
        self.assertAlmostEqual(result['passive_arm_drifts_m']['right'], 0.008, places=5)

    def test_unified_action_compiles_batch_and_stage_tolerance_and_stall_guard(self):
        args = {
            'observation_id': self.h.latest['observation_id'],
            'tolerance_preset': 'coarse',
            'stall_guard_preset': 'contact_stop',
            'stages': [
                {
                    'id': 'transit',
                    'kind': 'move',
                    'arm': 'left',
                    'grasp_position_m': [0.2, -0.1, 0.8],
                    'angles_deg': [0, 0, 0],
                },
                {
                    'id': 'insert',
                    'kind': 'move',
                    'arm': 'left',
                    'grasp_position_m': [0.2, -0.1, 0.75],
                    'angles_deg': [0, 0, 0],
                    'tolerance_preset': 'tight',
                    'orientation_tolerance_rad': 0.12,
                    'stall_guard_preset': 'strict',
                },
            ],
        }
        for role in ('actor', 'program'):
            self.h.tool_registry.validate('robot_action', args, role=role)
        compiled = compile_request(self.h, args)
        p0 = compiled['stages'][0]['action']['path']
        p1 = compiled['stages'][1]['action']['path']
        self.assertEqual(p0['tolerance_preset'], 'coarse')
        self.assertAlmostEqual(p0['position_tolerance_m'], 0.025)
        self.assertAlmostEqual(p0['orientation_tolerance_rad'], 0.35)
        self.assertEqual(p0['stall_guard_preset'], 'contact_stop')
        self.assertAlmostEqual(p0['max_tracking_divergence_m'], 0.020)

        self.assertEqual(p1['tolerance_preset'], 'tight')
        self.assertAlmostEqual(p1['position_tolerance_m'], 0.003)
        self.assertAlmostEqual(p1['orientation_tolerance_rad'], 0.12)
        self.assertEqual(p1['stall_guard_preset'], 'strict')
        self.assertAlmostEqual(p1['max_tracking_divergence_m'], 0.010)

    def test_nonfinite_physx_state_raises_clean_runtime_error_without_poisoning_receipt_json(self):
        import io
        import json
        from types import SimpleNamespace
        from unittest.mock import patch
        from ni_rua.robot.backends.robodojo.robot_worker import RoboDojoAdapter

        # 1. 直接验证 RoboDojoAdapter.state 在 PhysX 返回 NaN 位姿或零范数四元数时抛出 RuntimeError
        raw_adapter = object.__new__(RoboDojoAdapter)
        raw_adapter.np = np
        raw_adapter.control_step = 0
        raw_adapter.goals = {'left': 1.0}
        raw_adapter.robot_description = SimpleNamespace(frame_id='env_origin', tcp_config_id='arx-x5-link6-v1')
        target_robot = SimpleNamespace(
            type='target', arm_name='left_arm', gripper_name='left_gripper',
            gripper_scale=[0.0, 1.0], gripper_move={'sign': 1},
            arm_joints_name=[f'j{i}' for i in range(6)], arm_joint_indices=list(range(6)),
            gripper_joints_name=['jaw'],
        )
        class _DummyVel:
            def __getitem__(self, idx):
                return SimpleNamespace(cpu=lambda: np.zeros(6))

        key = SimpleNamespace(data=SimpleNamespace(joint_vel=_DummyVel()))
        raw_adapter.robot = SimpleNamespace(
            robot_list=[target_robot],
            robot_key=[key],
            get_real_endpose=lambda r, idx: np.array([[float('nan')] * 7]),
            get_joint=lambda r, idx: np.zeros((1, 6)),
            get_end_effector_real_val=lambda r, idx: np.zeros((1, 1)),
            process_name=lambda n: n,
            control_manager=SimpleNamespace(prev_control=[{'left_gripper': {'position': [0.5]}}]),
        )
        with self.assertRaisesRegex(RuntimeError, 'simulator_nonfinite_robot_state:left'):
            RoboDojoAdapter.state(raw_adapter)

        # 2. 验证在 execute 期间当 state() 抛出 simulator_nonfinite_robot_state 时，回执保持严格 JSON 合规（不含 NaN）
        adapter, _, _ = preview_adapter()
        action = batch([move('one', arm='left', angle=-0.125)])
        plan = adapter.preview_batch(action)
        original_state = adapter.state

        def nan_state():
            if adapter.env.take_action_cnt[0] >= 1:
                raise RuntimeError('simulator_nonfinite_robot_state:left')
            return original_state()

        adapter.state = nan_state
        with patch('sys.stderr', io.StringIO()):
            receipt = self.execute_batch(adapter, action, plan)
        serialized = json.dumps(receipt, allow_nan=False)
        self.assertIn('simulator_nonfinite_robot_state:left', serialized)

    def test_normal_soft_tolerance_and_persistent_obstacles_and_segment_start_drift(self):
        # 1. normal 预设下 5mm ~ 10mm 终点偏差按 1cm 软容差通过并标记 within_1cm_soft_tolerance
        adapter, _, _ = preview_adapter()
        action = batch([move('approach', arm='left', angle=-0.25)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        original_state = adapter.state

        def soft_perturbed_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= plan['planned_native_steps']:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [pose['position_m'][0] + 0.0082, pose['position_m'][1], pose['position_m'][2]]
            return st

        adapter.state = soft_perturbed_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertTrue(result['stage_results']['approach']['endpoint_error'].get('within_1cm_soft_tolerance'))
        self.assertEqual(result['stage_results']['approach']['completion_quality'], 'within_1cm_tolerance')

        # 2. 会话级持久化障碍物默认合并到同 episode 后续 compile_request，stateless 模式下不注入
        args1 = {
            'observation_id': self.h.latest['observation_id'],
            'obstacles': [{
                'id': 'box_wall',
                'min_m': [0.10, -0.20, 0.70],
                'max_m': [0.30, -0.05, 0.88],
                'evidence': 'measured front box wall from depth_query',
            }],
            'stages': [{
                'id': 'm1', 'kind': 'move', 'arm': 'left',
                'grasp_position_m': [0.20, -0.10, 0.92], 'angles_deg': [0, 0, 0],
            }],
        }
        c1 = compile_request(self.h, args1)
        self.assertEqual(len(c1['review_proxies']['obstacles']), 1)
        self.assertEqual(c1['review_proxies']['obstacles'][0]['id'], 'box_wall')
        args2 = {
            'observation_id': self.h.latest['observation_id'],
            'stages': [{
                'id': 'm2', 'kind': 'move', 'arm': 'left',
                'grasp_position_m': [0.20, 0.10, 0.92], 'angles_deg': [0, 0, 0],
            }],
        }
        c2 = compile_request(self.h, args2)
        self.assertEqual(len(c2['review_proxies']['obstacles']), 1)
        self.assertEqual(c2['review_proxies']['obstacles'][0]['id'], 'box_wall')
        c3 = compile_request(self.h, dict(args2, obstacle_policy='stateless'))
        self.assertNotIn('review_proxies', c3)

    def test_stage_tolerances_do_not_ratchet_and_preserve_normal_default(self):
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        action = batch([
            move('transit', arm='left', angle=-0.125, tolerance_preset='coarse'),
            move('place', arm='left', angle=-0.25),
            jaw(identifier='open_jaw', arm='left', goal=1.0),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        cache = adapter._batch_plans[plan['plan_id']]
        # place 未显式传容差时保持 normal 默认 10mm，不做隐式 5mm 覆写，也不被阶段 1 的 coarse (25mm) 污染
        self.assertAlmostEqual(cache['stages']['place']['position_tolerance_m'], 0.010, places=5)
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertAlmostEqual(cache['tolerances']['left'], 0.010, places=5)

    def test_vertical_compliance_rejects_undeclared_downward_sag_and_xy_miss(self):
        # A 32 mm downward sag exceeds the declared 10 mm tolerance even with XY aligned.
        # Reject it and do not execute the subsequent release.
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        action = batch([
            move('carry_sag', arm='left', angle=-0.25),
            jaw(identifier='release_jaw', arm='left', goal=1.0),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        move_steps = plan['stage_costs']['carry_sag']['end_native_step']
        original_state = adapter.state

        def vertical_compliant_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= move_steps:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [
                    pose['position_m'][0] + 0.004,
                    pose['position_m'][1] - 0.003,
                    pose['position_m'][2] - 0.032,
                ]
            return st

        adapter.state = vertical_compliant_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(result['reason'], 'tracking_error')
        self.assertEqual(result['failed_stage_id'], 'carry_sag')
        self.assertEqual(result['stage_results']['release_jaw']['completed_steps'], 0)
        descent_res = result['stage_results']['carry_sag']
        self.assertFalse(descent_res['endpoint_error'].get('vertical_compliance_accepted', False))
        self.assertAlmostEqual(math.hypot(descent_res['endpoint_residual_m']['dx_m'],
                                         descent_res['endpoint_residual_m']['dy_m']), 0.005, places=4)
        self.assertAlmostEqual(descent_res['endpoint_residual_m']['dz_m'], -0.032, places=4)

        # 2. 未声明 contact_stop 的开爪抓取下探若被挡在目标上方 +32mm (> 12mm)，拒绝悬空闭爪
        adapter_air, _, _ = preview_adapter()
        jaws(adapter_air)
        adapter_air.last_dispatched_goals['left'] = 1.0
        action_air = batch([
            move('blocked_descent', arm='left', angle=-0.25),
            jaw(identifier='air_close', arm='left', goal=0.0),
        ])
        plan_air = adapter_air.preview_batch(action_air)
        orig_air = adapter_air.state

        def blocked_high_state():
            st = orig_air()
            if adapter_air.env.take_action_cnt[0] >= plan_air['stage_costs']['blocked_descent']['end_native_step']:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [pose['position_m'][0], pose['position_m'][1], pose['position_m'][2] + 0.032]
            return st

        adapter_air.state = blocked_high_state
        res_air = self.execute_batch(adapter_air, action_air, plan_air)
        self.assertEqual(res_air['status'], 'failed')
        self.assertEqual(res_air['reason'], 'tracking_error')
        self.assertEqual(res_air['failed_stage_id'], 'blocked_descent')
        self.assertEqual(res_air['stage_results']['air_close']['completed_steps'], 0)

        # 3. 水平 (dx, dy) 偏离 15mm (> 10mm normal 容差)，即使 Z 误差为 0 也严格拒绝
        adapter2, _, _ = preview_adapter()
        action2 = batch([move('miss_xy', arm='left', angle=-0.25)])
        plan2 = adapter2.preview_batch(action2)
        orig2 = adapter2.state

        def xy_miss_state():
            st = orig2()
            if adapter2.env.take_action_cnt[0] >= plan2['planned_native_steps']:
                pose = st['arms']['left']['ee_pose']
                pose['position_m'] = [pose['position_m'][0] + 0.015, pose['position_m'][1], pose['position_m'][2]]
            return st

        adapter2.state = xy_miss_state
        result2 = self.execute_batch(adapter2, action2, plan2)
        self.assertEqual(result2['status'], 'failed')
        self.assertEqual(result2['reason'], 'tracking_error')
        self.assertEqual(result2['failed_stage_id'], 'miss_xy')

    def test_contact_stop_mid_trajectory_vertical_touchdown_clamps_and_executes_attached_gripper(self):
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        action = batch([
            move('touchdown', arm='left', angle=-0.5, stall_guard_preset='contact_stop'),
            jaw(identifier='release', arm='left', goal=1.0),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        target_pose = deepcopy(adapter._batch_plans[plan['plan_id']]['stages']['touchdown']['target'])
        original_state = adapter.state

        # 模拟下探在中途接触台面：水平 (x, y) 已对准目标 (dx=3mm, dy=2mm)，但高度停在目标上方 +30mm 不再下降
        def vertical_touchdown_state():
            st = original_state()
            if adapter.env.take_action_cnt[0] >= 2:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_pose['position_m'][0] + 0.003,
                        target_pose['position_m'][1] + 0.002,
                        target_pose['position_m'][2] + 0.030,
                    ],
                    'quaternion_xyzw': list(target_pose['quaternion_xyzw']),
                }
            return st

        adapter.state = vertical_touchdown_state
        result = self.execute_batch(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['completed_stage_ids'], ['touchdown', 'release'])
        self.assertTrue(result['stage_results']['touchdown'].get('contact_stopped_early'))
        self.assertAlmostEqual(result['stage_results']['touchdown']['endpoint_residual_m']['dz_m'], 0.030, places=4)

        # 拆除 _is_open_jaw_pre_grasp 隐藏缩窄后：contact_stop 默认统一遵循公开契约 max(pos_tol, 0.040m)；显式设置 max_upward_dz_m=0.015 时 +28mm 严格报 tracking_error
        adapter_grasp, _, _ = preview_adapter()
        jaws(adapter_grasp)
        adapter_grasp.last_dispatched_goals['left'] = 1.0
        action_grasp = batch([
            move('grasp_descent', arm='left', angle=-0.5, stall_guard_preset='contact_stop', max_upward_dz_m=0.015),
            jaw(identifier='grasp_close', arm='left', goal=0.0),
        ])
        plan_grasp = adapter_grasp.preview_batch(action_grasp)
        target_grasp = deepcopy(adapter_grasp._batch_plans[plan_grasp['plan_id']]['stages']['grasp_descent']['target'])
        orig_grasp_state = adapter_grasp.state

        def blocked_grasp_state():
            st = orig_grasp_state()
            if adapter_grasp.env.take_action_cnt[0] >= 2:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_grasp['position_m'][0] + 0.002,
                        target_grasp['position_m'][1] + 0.001,
                        target_grasp['position_m'][2] + 0.028,
                    ],
                    'quaternion_xyzw': list(target_grasp['quaternion_xyzw']),
                }
            return st

        adapter_grasp.state = blocked_grasp_state
        res_grasp = self.execute_batch(adapter_grasp, action_grasp, plan_grasp)
        self.assertEqual(res_grasp['status'], 'failed')
        self.assertEqual(res_grasp['reason'], 'tracking_error')
        self.assertEqual(res_grasp['failed_stage_id'], 'grasp_descent')
        self.assertEqual(res_grasp['stage_results']['grasp_close']['completed_steps'], 0)

    def test_planning_failure_preserves_explicit_grasp_orientation(self):
        adapter, _, _ = preview_adapter()
        adapter.joint_replan_enabled = True
        stage = move('place', arm='left', angle=-0.125)
        stage['action']['input_grasp_pose'] = {'left': {
            'position_m': [0.25, -0.20, 0.88], 'angles_deg': [0., 0., 0.], 'offset_m': 0.15007}}
        original = deepcopy(stage)
        calls = []
        original_plan = adapter._plan_segment
        def planner(robot, target, diagnostics, waypoint_index=0, **kwargs):
            calls.append(deepcopy(target))
            if len(calls) == 1:
                diagnostics.append({'stage': 'ik', 'feasible_seed_count': 0})
                return None
            return original_plan(robot, target, diagnostics, waypoint_index, **kwargs)
        adapter._plan_segment = planner
        plan = adapter.preview_batch(batch([stage]))
        self.assertEqual(plan['status'], 'failed')
        self.assertEqual(plan['executed_native_steps'], 0)
        self.assertEqual(len(calls), 1)
        self.assertEqual(stage, original)

    def test_vertical_compliance_respects_rotation_and_downward_tolerances(self):
        from ni_rua.robot.motion.batch_motion import _check_pose_arrival
        target = {'position_m': [0., 0., 0.8], 'quaternion_xyzw': [0., 0., 0., 1.]}
        for dz, angle, expected in [(0.001, 0.20, False), (-0.035, 0., False),
                                     (0.015, 0.02, True), (-0.002, 0.02, True)]:
            measured = {'position_m': [0., 0., 0.8 + dz],
                        'quaternion_xyzw': [0., 0., math.sin(angle/2), math.cos(angle/2)]}
            result = _check_pose_arrival(measured, target, .003, .05,
                                        allow_vertical_compliance=True, max_upward_dz_m=.02)
            self.assertEqual(result[0], expected, (dz, angle, result))

    def test_explicit_max_upward_dz_m_and_effective_tolerances_reported(self):
        # 模型显式传入阶段级 max_upward_dz_m 时，preview 与 execute 均透明返回 effective_tolerances 且严格按模型设置执行
        adapter_rel, _, _ = preview_adapter()
        jaws(adapter_rel)
        adapter_rel.last_dispatched_goals['left'] = 0.0
        action_rel = batch([
            jaw(identifier='close_first', arm='left', goal=0.0),
            move('place_descent', arm='left', angle=-0.5, stall_guard_preset='contact_stop', max_upward_dz_m=0.075, position_tolerance_m=0.020),
            jaw(identifier='place_open', arm='left', goal=1.0),
        ])
        plan_rel = adapter_rel.preview_batch(action_rel)
        self.assertEqual(adapter_rel._batch_plans[plan_rel['plan_id']]['stages']['place_descent']['max_upward_dz_m'], 0.075)
        place_end = plan_rel['stage_costs']['place_descent']['end_native_step']
        target_rel = deepcopy(adapter_rel._batch_plans[plan_rel['plan_id']]['stages']['place_descent']['target'])
        orig_rel_state = adapter_rel.state

        def stack_contact_state():
            st = orig_rel_state()
            if adapter_rel.env.take_action_cnt[0] >= place_end:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_rel['position_m'][0] + 0.010,
                        target_rel['position_m'][1] - 0.010,
                        target_rel['position_m'][2] + 0.061,
                    ],
                    'quaternion_xyzw': list(target_rel['quaternion_xyzw']),
                }
            return st

        adapter_rel.state = stack_contact_state
        res_rel = self.execute_batch(adapter_rel, action_rel, plan_rel)
        self.assertEqual(res_rel['status'], 'completed', res_rel)
        self.assertEqual(res_rel['completed_stage_ids'], ['close_first', 'place_descent', 'place_open'])
        eff_tol = res_rel['stage_results']['place_descent']['endpoint_error']['effective_tolerances']
        self.assertEqual(eff_tol['max_upward_dz_m'], 0.075)

        # 3. 开爪接触下探未显式传 max_upward_dz_m 时遵循公开 40mm 契约（+28mm 放行，+48mm 报 tracking_error 并返回 contact_block_summary）
        adapter_ep_ok, _, _ = preview_adapter()
        jaws(adapter_ep_ok)
        adapter_ep_ok.last_dispatched_goals['left'] = 1.0
        action_ep_ok = batch([
            move('grasp_ep_ok', arm='left', angle=-0.25, stall_guard_preset='contact_stop'),
            jaw(identifier='close_ep_ok', arm='left', goal=0.0),
        ])
        plan_ep_ok = adapter_ep_ok.preview_batch(action_ep_ok)
        ep_ok_end = plan_ep_ok['stage_costs']['grasp_ep_ok']['end_native_step']
        target_ep_ok = deepcopy(adapter_ep_ok._batch_plans[plan_ep_ok['plan_id']]['stages']['grasp_ep_ok']['target'])
        orig_ep_ok_state = adapter_ep_ok.state

        def endpoint_28mm_grasp_state():
            st = orig_ep_ok_state()
            if adapter_ep_ok.env.take_action_cnt[0] >= ep_ok_end:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_ep_ok['position_m'][0] + 0.002,
                        target_ep_ok['position_m'][1] + 0.001,
                        target_ep_ok['position_m'][2] + 0.028,
                    ],
                    'quaternion_xyzw': list(target_ep_ok['quaternion_xyzw']),
                }
            return st

        adapter_ep_ok.state = endpoint_28mm_grasp_state
        res_ep_ok = self.execute_batch(adapter_ep_ok, action_ep_ok, plan_ep_ok)
        self.assertEqual(res_ep_ok['status'], 'completed', res_ep_ok)

        adapter_ep, _, _ = preview_adapter()
        jaws(adapter_ep)
        adapter_ep.last_dispatched_goals['left'] = 1.0
        action_ep = batch([
            move('grasp_ep', arm='left', angle=-0.25, stall_guard_preset='contact_stop'),
            jaw(identifier='close_ep', arm='left', goal=0.0),
        ])
        plan_ep = adapter_ep.preview_batch(action_ep)
        ep_end = plan_ep['stage_costs']['grasp_ep']['end_native_step']
        target_ep = deepcopy(adapter_ep._batch_plans[plan_ep['plan_id']]['stages']['grasp_ep']['target'])
        orig_ep_state = adapter_ep.state

        def endpoint_blocked_grasp_state():
            st = orig_ep_state()
            if adapter_ep.env.take_action_cnt[0] >= ep_end:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_ep['position_m'][0] + 0.002,
                        target_ep['position_m'][1] + 0.001,
                        target_ep['position_m'][2] + 0.048,
                    ],
                    'quaternion_xyzw': list(target_ep['quaternion_xyzw']),
                }
            return st

        adapter_ep.state = endpoint_blocked_grasp_state
        res_ep = self.execute_batch(adapter_ep, action_ep, plan_ep)
        self.assertEqual(res_ep['status'], 'failed')
        self.assertEqual(res_ep['reason'], 'tracking_error')
        self.assertEqual(res_ep['failed_stage_id'], 'grasp_ep')
        self.assertEqual(res_ep['stage_results']['close_ep']['completed_steps'], 0)
        self.assertIn('contact_block_summary', res_ep)

        # 4. 闭爪接触下探后原地旋转的 tight 阶段显式指定 max_upward_dz_m=0.02 时允许 +16mm 顺应高度
        adapter_turn, _, _ = preview_adapter()
        jaws(adapter_turn)
        adapter_turn.last_dispatched_goals['left'] = 0.0
        action_turn = batch([
            jaw(identifier='hold_nut', arm='left', goal=0.0),
            move('shallow_insert', arm='left', angle=-0.25, stall_guard_preset='contact_stop'),
            move('turn_quarter', arm='left', angle=-0.25, tolerance_preset='tight', max_upward_dz_m=0.02),
        ])
        plan_turn = adapter_turn.preview_batch(action_turn)
        insert_end = plan_turn['stage_costs']['shallow_insert']['end_native_step']
        target_turn = deepcopy(adapter_turn._batch_plans[plan_turn['plan_id']]['stages']['shallow_insert']['target'])
        orig_turn_state = adapter_turn.state

        def compliant_support_turn_state():
            st = orig_turn_state()
            if adapter_turn.env.take_action_cnt[0] >= insert_end:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_turn['position_m'][0] + 0.002,
                        target_turn['position_m'][1] - 0.001,
                        target_turn['position_m'][2] + 0.016,
                    ],
                    'quaternion_xyzw': list(target_turn['quaternion_xyzw']),
                }
            return st

        adapter_turn.state = compliant_support_turn_state
        res_turn = self.execute_batch(adapter_turn, action_turn, plan_turn)
        self.assertEqual(res_turn['status'], 'completed', res_turn)
        self.assertEqual(res_turn['completed_stage_ids'], ['hold_nut', 'shallow_insert', 'turn_quarter'])

        # 5. 同时设置 tolerance_preset='tight' 与 stall_guard_preset='contact_stop' 的阶段，在 contact_stop 接触停住后 endpoint() 不得将 allow_vertical 置为 False
        adapter_tc, _, _ = preview_adapter()
        jaws(adapter_tc)
        adapter_tc.last_dispatched_goals['left'] = 0.0
        action_tc = batch([
            jaw(identifier='hold_slice', arm='left', goal=0.0),
            move('short_insert', arm='left', angle=-0.5, tolerance_preset='tight', stall_guard_preset='contact_stop'),
            jaw(identifier='release_slice', arm='left', goal=1.0),
        ])
        plan_tc = adapter_tc.preview_batch(action_tc)
        target_tc = deepcopy(adapter_tc._batch_plans[plan_tc['plan_id']]['stages']['short_insert']['target'])
        orig_tc_state = adapter_tc.state

        def tight_contact_stop_state():
            st = orig_tc_state()
            if adapter_tc.env.take_action_cnt[0] >= 2:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_tc['position_m'][0] + 0.0005,
                        target_tc['position_m'][1] - 0.0005,
                        target_tc['position_m'][2] + 0.025,
                    ],
                    'quaternion_xyzw': list(target_tc['quaternion_xyzw']),
                }
            return st

        adapter_tc.state = tight_contact_stop_state
        res_tc = self.execute_batch(adapter_tc, action_tc, plan_tc)
        self.assertEqual(res_tc['status'], 'completed', res_tc)
        self.assertEqual(res_tc['completed_stage_ids'], ['hold_slice', 'short_insert', 'release_slice'])
        self.assertEqual(
            res_tc['stage_results']['short_insert']['endpoint_error']['effective_tolerances']['max_upward_dz_m'],
            0.04,
        )

        # 6. 显式传入严格 position_tolerance_m < 0.009（且无 contact_stop 与显式 max_upward_dz_m）时，不应放行 +20mm 的垂直偏差
        adapter_strict, _, _ = preview_adapter()
        jaws(adapter_strict)
        action_strict = batch([
            move('precise_insert', arm='left', angle=-0.5, position_tolerance_m=0.005),
        ])
        plan_strict = adapter_strict.preview_batch(action_strict)
        target_strict = deepcopy(adapter_strict._batch_plans[plan_strict['plan_id']]['stages']['precise_insert']['target'])
        orig_strict_state = adapter_strict.state

        def strict_pos_tol_state():
            st = orig_strict_state()
            if adapter_strict.env.take_action_cnt[0] >= 1:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_strict['position_m'][0] + 0.001,
                        target_strict['position_m'][1] - 0.001,
                        target_strict['position_m'][2] + 0.016,
                    ],
                    'quaternion_xyzw': list(target_strict['quaternion_xyzw']),
                }
            return st

        adapter_strict.state = strict_pos_tol_state
        res_strict = self.execute_batch(adapter_strict, action_strict, plan_strict)
        self.assertEqual(res_strict['status'], 'failed', res_strict)

    def test_removed_black_boxes_5mm_open_jaw_and_sub_5_5mm_enforced_honestly(self):
        """验证已彻底切除三处静默改参黑盒：0.005m 不被吞参、非 contact_stop 默认 max_upward_dz_m==0、亚 5.5mm 残差不偷放行。"""
        # 1) 显式传 position_tolerance_m=0.005（未传 tolerance_preset）必须严格保持 5mm，7mm 水平残差如实报 tracking_error
        adapter_5mm, _, _ = preview_adapter()
        jaws(adapter_5mm)
        action_5mm = batch([move('exact_5mm', arm='left', angle=-0.25, position_tolerance_m=0.005)])
        plan_5mm = adapter_5mm.preview_batch(action_5mm)
        target_5mm = deepcopy(adapter_5mm._batch_plans[plan_5mm['plan_id']]['stages']['exact_5mm']['target'])
        orig_5mm_state = adapter_5mm.state

        def err_7mm_state():
            st = orig_5mm_state()
            if adapter_5mm.env.take_action_cnt[0] >= 1:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [
                        target_5mm['position_m'][0] + 0.007,
                        target_5mm['position_m'][1],
                        target_5mm['position_m'][2],
                    ],
                    'quaternion_xyzw': list(target_5mm['quaternion_xyzw']),
                }
            return st

        adapter_5mm.state = err_7mm_state
        res_5mm = self.execute_batch(adapter_5mm, action_5mm, plan_5mm)
        self.assertEqual(res_5mm['status'], 'failed')
        self.assertEqual(res_5mm['reason'], 'tracking_error')
        eff_5mm = res_5mm['stage_results']['exact_5mm']['endpoint_error']['effective_tolerances']
        self.assertAlmostEqual(eff_5mm['position_tolerance_m'], 0.005, places=5)
        self.assertAlmostEqual(eff_5mm['max_upward_dz_m'], 0.0, places=5)

        # 2) 普通非接触移动（无论开爪或闭爪、未传 max_upward_dz_m 且非 contact_stop）严格为 max_upward_dz_m=0.0m
        compiled_preview = compile_request(
            self.h,
            {
                'observation_id': self.h.latest['observation_id'],
                'stages': [
                    {
                        'id': 'open_approach',
                        'kind': 'move',
                        'arm': 'left',
                        'grasp_position_m': [0.20, 0.15, 0.80],
                        'angles_deg': [0.0, 0.0, 0.0],
                    },
                    {'id': 'close_jaw', 'kind': 'gripper', 'arm': 'left', 'goal_normalized': 0.0},
                    {
                        'id': 'closed_lift',
                        'kind': 'move',
                        'arm': 'left',
                        'grasp_position_m': [0.20, 0.15, 0.90],
                        'angles_deg': [0.0, 0.0, 0.0],
                    },
                ],
            },
        )
        from ni_rua.runtime.actions.unified_actions import _summarize_effective_tolerances
        tol_summary = _summarize_effective_tolerances(self.h, compiled_preview)
        self.assertEqual(tol_summary['open_approach']['max_upward_dz_m'], 0.0)
        self.assertNotIn('max_downward_dz_m', tol_summary['open_approach'])
        self.assertEqual(tol_summary['closed_lift']['max_upward_dz_m'], 0.0)
        self.assertNotIn('max_downward_dz_m', tol_summary['closed_lift'])

        # 3) position_tolerance_m=0.003 在实测误差 4.5mm 时（即便 planned_positions_m[-1] 与实测重合）绝不偷放宽到 5.5mm
        adapter_3mm, _, _ = preview_adapter()
        jaws(adapter_3mm)
        action_3mm = batch([move('tight_3mm', arm='left', angle=-0.25, position_tolerance_m=0.003)])
        plan_3mm = adapter_3mm.preview_batch(action_3mm)
        stored_stage = adapter_3mm._batch_plans[plan_3mm['plan_id']]['stages']['tight_3mm']
        target_3mm = deepcopy(stored_stage['target'])
        drifted_pos = [
            target_3mm['position_m'][0] + 0.0045,
            target_3mm['position_m'][1],
            target_3mm['position_m'][2],
        ]
        # 模拟插值末端 planned_positions_m[-1] 与实测重合，检验是否仍有 <5.5mm 强判 ok=True 后门
        stored_stage['planned_positions_m'][-1] = list(drifted_pos)
        orig_3mm_state = adapter_3mm.state

        def err_4_5mm_state():
            st = orig_3mm_state()
            if adapter_3mm.env.take_action_cnt[0] >= 1:
                st['arms']['left']['ee_pose'] = {
                    'position_m': list(drifted_pos),
                    'quaternion_xyzw': list(target_3mm['quaternion_xyzw']),
                }
            return st

        adapter_3mm.state = err_4_5mm_state
        res_3mm = self.execute_batch(adapter_3mm, action_3mm, plan_3mm)
        self.assertEqual(res_3mm['status'], 'failed')
        self.assertEqual(res_3mm['reason'], 'tracking_error')

    def test_u1_tight_5mm_compliant_upward_dz_and_planned_ik_residual(self):
        """验证 tight=5mm 默认与显式 3mm 不放宽、compliant 40mm 向上搁放顺应与重置、以及 planned_ik_residual_m 透明汇报。"""
        from unittest.mock import patch
        from ni_rua.robot.motion import batch_motion as bm
        from ni_rua.runtime.actions.unified_actions import _summarize_effective_tolerances

        # 1) batch 设 compliant 时，stage 继承 40mm 向上顺应；显式设 tight/normal 的 stage 重置为 0.0m
        compiled = compile_request(
            self.h,
            {
                'observation_id': self.h.latest['observation_id'],
                'tolerance_preset': 'compliant',
                'stages': [
                    {
                        'id': 'place_shelf',
                        'kind': 'move',
                        'arm': 'left',
                        'grasp_position_m': [0.22, 0.12, 0.80],
                        'angles_deg': [0.0, 0.0, 0.0],
                    },
                    {
                        'id': 'align_tight',
                        'kind': 'move',
                        'arm': 'left',
                        'grasp_position_m': [0.22, 0.12, 0.82],
                        'angles_deg': [0.0, 0.0, 0.0],
                        'tolerance_preset': 'tight',
                    },
                    {
                        'id': 'retract_normal',
                        'kind': 'move',
                        'arm': 'left',
                        'grasp_position_m': [0.22, 0.12, 0.88],
                        'angles_deg': [0.0, 0.0, 0.0],
                        'tolerance_preset': 'normal',
                    },
                ],
            },
        )
        eff = _summarize_effective_tolerances(self.h, compiled)
        self.assertEqual(eff['place_shelf']['position_tolerance_m'], 0.015)
        self.assertEqual(eff['place_shelf']['max_upward_dz_m'], 0.040)
        self.assertEqual(eff['align_tight']['position_tolerance_m'], 0.003)
        self.assertEqual(eff['align_tight']['max_upward_dz_m'], 0.0)
        self.assertEqual(eff['retract_normal']['position_tolerance_m'], 0.010)
        self.assertEqual(eff['retract_normal']['max_upward_dz_m'], 0.0)

        # 2) tight 预设（3mm）在 2.4mm 误差下通过、在 4.2mm 误差下严格失败；显式传入 position_tolerance_m=0.005 在 4.2mm 下通过
        for explicit_tol, err_x, expected_status in ((None, 0.0024, 'completed'), (None, 0.0042, 'failed'), (0.005, 0.0042, 'completed')):
            adapter_t, _, _ = preview_adapter()
            jaws(adapter_t)
            kw = {'tolerance_preset': 'tight'}
            if explicit_tol is not None:
                kw['position_tolerance_m'] = explicit_tol
            act_t = batch([move('pin_align', arm='left', angle=-0.25, **kw)])
            plan_t = adapter_t.preview_batch(act_t)
            target_t = deepcopy(adapter_t._batch_plans[plan_t['plan_id']]['stages']['pin_align']['target'])
            orig_t_state = adapter_t.state

            def state_with_err(orig=orig_t_state, adp=adapter_t, tgt=target_t, dx=err_x):
                st = orig()
                if adp.env.take_action_cnt[0] >= plan_t['planned_native_steps']:
                    st['arms']['left']['ee_pose'] = {
                        'position_m': [tgt['position_m'][0] + dx, tgt['position_m'][1], tgt['position_m'][2]],
                        'quaternion_xyzw': list(tgt['quaternion_xyzw']),
                    }
                return st

            adapter_t.state = state_with_err
            res_t = self.execute_batch(adapter_t, act_t, plan_t)
            self.assertEqual(res_t['status'], expected_status)
            self.assertAlmostEqual(
                res_t['stage_results']['pin_align']['endpoint_error']['effective_tolerances']['position_tolerance_m'],
                0.003 if explicit_tol is None else explicit_tol,
                places=5,
            )

        # 3) compliant 预设无需 contact_stop 即可顺应 +28mm 向上支撑搁放并顺利衔接开爪；向下 -28mm 仍如实判定失败
        for dz_offset, expected_status in ((0.028, 'completed'), (-0.028, 'failed')):
            adapter_c, _, _ = preview_adapter()
            jaws(adapter_c)
            act_c = batch([
                move('seat_on_hook', arm='left', angle=-0.25, tolerance_preset='compliant'),
                jaw('open_after_seat', arm='left', goal=1.0),
            ])
            plan_c = adapter_c.preview_batch(act_c)
            target_c = deepcopy(adapter_c._batch_plans[plan_c['plan_id']]['stages']['seat_on_hook']['target'])
            move_end_step = plan_c['stage_costs']['seat_on_hook']['end_native_step']
            orig_c_state = adapter_c.state

            def state_dz(orig=orig_c_state, adp=adapter_c, tgt=target_c, end_step=move_end_step, dz=dz_offset):
                st = orig()
                if adp.env.take_action_cnt[0] >= end_step:
                    st['arms']['left']['ee_pose'] = {
                        'position_m': [tgt['position_m'][0] + 0.004, tgt['position_m'][1], tgt['position_m'][2] + dz],
                        'quaternion_xyzw': list(tgt['quaternion_xyzw']),
                    }
                return st

            adapter_c.state = state_dz
            res_c = self.execute_batch(adapter_c, act_c, plan_c)
            self.assertEqual(res_c['status'], expected_status)
            self.assertAlmostEqual(
                res_c['stage_results']['seat_on_hook']['endpoint_error']['effective_tolerances']['max_upward_dz_m'],
                0.040,
                places=5,
            )
            if expected_status == 'completed':
                self.assertTrue(res_c['stage_results']['seat_on_hook']['endpoint_error'].get('vertical_compliance_accepted'))
                self.assertIn('open_after_seat', res_c['completed_stage_ids'])

        # 4) planned_ik_residual_m 在 preview stage_costs 与 execute endpoint_error 中精确透明汇报，且绝不篡改 ok 判定
        adapter_ik, _, _ = preview_adapter()
        jaws(adapter_ik)
        orig_populate = bm._populate_planned_positions

        def populate_with_ik_gap(adp, before, records, full):
            orig_populate(adp, before, records, full)
            for rec in records.values():
                if rec['kind'] == 'move' and rec.get('planned_positions_m'):
                    tgt = rec['target']['position_m']
                    rec['planned_positions_m'][-1] = [tgt[0] + 0.003, tgt[1] - 0.004, tgt[2]]

        act_ik = batch([move('ik_stage', arm='left', angle=-0.25, position_tolerance_m=0.003)])
        with patch.object(bm, '_populate_planned_positions', side_effect=populate_with_ik_gap):
            plan_ik = adapter_ik.preview_batch(act_ik)
        self.assertEqual(plan_ik['status'], 'planned')
        self.assertAlmostEqual(plan_ik['stage_costs']['ik_stage']['planned_ik_residual_m'], 0.005, places=5)

        target_ik = deepcopy(adapter_ik._batch_plans[plan_ik['plan_id']]['stages']['ik_stage']['target'])
        orig_ik_state = adapter_ik.state

        def state_at_ik_endpoint():
            st = orig_ik_state()
            if adapter_ik.env.take_action_cnt[0] >= 1:
                st['arms']['left']['ee_pose'] = {
                    'position_m': [target_ik['position_m'][0] + 0.003, target_ik['position_m'][1] - 0.004, target_ik['position_m'][2]],
                    'quaternion_xyzw': list(target_ik['quaternion_xyzw']),
                }
            return st

        adapter_ik.state = state_at_ik_endpoint
        res_ik = self.execute_batch(adapter_ik, act_ik, plan_ik)
        # 5mm 实测误差超出显式 3mm 容差：planned_ik_residual_m=0.005 如实报告，但绝不放宽 ok 判定
        self.assertEqual(res_ik['status'], 'failed')
        self.assertEqual(res_ik['reason'], 'tracking_error')
        self.assertAlmostEqual(res_ik['stage_results']['ik_stage']['endpoint_error']['planned_ik_residual_m'], 0.005, places=5)

    def test_max_downward_dz_payload_droop_and_one_sided_compliance_half_space(self):
        self.assertIn('max_downward_dz_m', TOLERANCE_FIELDS)
        self.assertEqual(MAX_DOWNWARD_DZ_M, 0.15)
        self.assertEqual(
            resolve_stage_tolerance({}, {'tolerance_preset': 'tight', 'max_downward_dz_m': 0.018}),
            {'tolerance_preset': 'tight', 'position_tolerance_m': 0.003, 'orientation_tolerance_rad': 0.05, 'max_downward_dz_m': 0.018},
        )
        self.assertNotIn('max_downward_dz_m', resolve_stage_tolerance({'tolerance_preset': 'tight'}, {'max_downward_dz_m': 0.018}))
        self.assertEqual(
            resolve_stage_tolerance(
                {'tolerance_preset': 'tight', 'max_downward_dz_m': 0.022},
                {'max_downward_dz_m': 0.010},
            )['max_downward_dz_m'],
            0.022,
        )
        validate_tolerance_fields({'max_downward_dz_m': 0.0})
        validate_tolerance_fields({'max_downward_dz_m': 0.15})
        for invalid in (-0.001, 0.151, True, float('nan'), float('inf'), '0.02'):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(ValueError, 'invalid_max_downward_dz_m'):
                validate_tolerance_fields({'max_downward_dz_m': invalid})

        eff = _summarize_effective_tolerances(
            None,
            {'stages': [{'id': 'insert_pin/motion', 'kind': 'move', 'arm': 'left', 'action': {'path': {'tolerance_preset': 'tight', 'position_tolerance_m': 0.005, 'orientation_tolerance_rad': 0.05, 'max_downward_dz_m': 0.02, 'stall_guard_preset': 'off'}}}]},
        )
        self.assertEqual(eff['insert_pin']['max_downward_dz_m'], 0.02)
        self.assertEqual(eff['insert_pin']['max_upward_dz_m'], 0.0)

        def move_xyz(identifier, *, arm, x_m, y_m, z_m, **values):
            own = wait_action(timeout_s=30)
            del own['hold_native_steps']
            own['path'] = dict(kind='backend_free', waypoints=[{arm: dict(position_m=[float(x_m), float(y_m), float(z_m)], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0])}])
            return dict(id=identifier, kind='move', arm=arm, action=own, **values)

        def droop_adapter(left_loaded_offset_xyz=(0.0, 0.0, 0.0), unload_on_open=False):
            adapter, planner, dispatched = preview_adapter()
            jaws(adapter)
            cmd_q = {'left': [0.0] * 6, 'right': [0.0] * 6}
            jaw_cmd = {'left': 0.10, 'right': 1.0}
            adapter.last_dispatched_goals = {'left': 0.10, 'right': 1.0}

            def custom_plan_path(current, target, origin):
                goal = np.array(current, copy=True)
                goal[0] = (float(target[0]) - 0.30) / 0.10
                goal[1] = (float(target[1]) - 0.00) / 0.10
                goal[2] = (float(target[2]) - 0.90) / 0.10
                return {'status': 'Success', 'position': np.stack([current, goal])}

            planner.plan_path = custom_plan_path
            robot = adapter.robot.robot_list[0]

            def state():
                arms = {}
                for side in ('left', 'right'):
                    q_nom = list(cmd_q[side])
                    dx, dy, dz = left_loaded_offset_xyz if (side == 'left' and not (unload_on_open and jaw_cmd['left'] >= 0.80)) else (0.0, 0.0, 0.0)
                    q_meas = [q_nom[0] + dx / 0.10, q_nom[1] + dy / 0.10, q_nom[2] + dz / 0.10, *q_nom[3:]]
                    pose = dict(position_m=[0.30 + 0.10 * q_meas[0], 0.10 * q_meas[1], 0.90 + 0.10 * q_meas[2]], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0], source_kind='measurement', frame_id='env_origin', tcp_config_id='arx-x5-link6-v1')
                    arms[side] = dict(q_rad=q_meas, joint_names=list(robot.arm_joints_name), source_kind='measurement', ee_pose=pose, gripper=dict(position_source_kind='measurement', position_m=[jaw_cmd[side] * 0.044], joint_names=['jaw']))
                return {'arms': arms}

            adapter.state = state
            raw_take = adapter.env.take_action

            def take_action(act):
                raw_take(act)
                for side in ('left', 'right'):
                    cmd_q[side] = list(dispatched[-1][f'{side}_arm_joint_state'])
                    jaw_cmd[side] = float(adapter.last_dispatched_goals[side])

            adapter.env.take_action = take_action
            return adapter, dispatched

        adapter, dispatched = droop_adapter((0.00036, -0.00048, -0.012), unload_on_open=True)
        action = batch([
            move_xyz('align_loaded', arm='left', x_m=0.32, y_m=-0.02, z_m=0.88, tolerance_preset='tight', position_tolerance_m=0.005, orientation_tolerance_rad=0.05, max_downward_dz_m=0.020),
            jaw('release_payload', arm='left', goal=1.0),
            move_xyz('retract_unloaded', arm='left', x_m=0.30, y_m=0.00, z_m=0.92),
        ])
        plan = adapter.preview_batch(action)
        receipt = self.execute_batch(adapter, action, plan)
        self.assertEqual(receipt['status'], 'completed', receipt)
        self.assertEqual(receipt['stage_results']['align_loaded']['completion_quality'], 'vertical_compliance_within_xy_tolerance')
        rel_cost = plan['stage_costs']['release_payload']
        rel_q2 = [row['left_arm_joint_state'][2] for row in dispatched[rel_cost['start_native_step']:rel_cost['end_native_step']]]
        self.assertTrue(all(abs(q2 - (-0.20)) < 1e-6 for q2 in rel_q2), rel_q2)

        for label, offset_xyz in (('xy_misaligned_7mm', (0.0050, 0.0050, -0.010)), ('unrequested_upward_12mm', (0.0002, 0.0002, 0.012)), ('excess_downward_28mm', (0.0002, 0.0002, -0.028))):
            with self.subTest(case=label):
                adp_bad, _ = droop_adapter(offset_xyz, unload_on_open=False)
                act_bad = batch([move_xyz('strict_check', arm='left', x_m=0.32, y_m=0.00, z_m=0.88, tolerance_preset='tight', position_tolerance_m=0.005, orientation_tolerance_rad=0.05, max_downward_dz_m=0.020)])
                rec_bad = self.execute_batch(adp_bad, act_bad, adp_bad.preview_batch(act_bad))
                self.assertEqual((rec_bad['status'], rec_bad['reason']), ('failed', 'tracking_error'))

        for dx_m, dz_m, expected_status, expect_vert in ((0.008, -0.028, 'completed', True), (0.006, 0.006, 'completed', False), (0.008, 0.008, 'failed', False)):
            adp_h, _, _ = preview_adapter()
            jaws(adp_h)
            act_h = batch([jaw(identifier='hold_item', arm='left', goal=0.0), move('loaded_move', arm='left', angle=-0.25, max_downward_dz_m=0.040)])
            plan_h = adp_h.preview_batch(act_h)
            move_end = plan_h['stage_costs']['loaded_move']['end_native_step']
            tgt = deepcopy(adp_h._batch_plans[plan_h['plan_id']]['stages']['loaded_move']['target'])
            orig_st = adp_h.state

            def loaded_state(orig=orig_st, adp=adp_h, stop_step=move_end, target=tgt, ox=dx_m, oz=dz_m):
                st = orig()
                if adp.env.take_action_cnt[0] >= stop_step:
                    st['arms']['left']['ee_pose'] = {'position_m': [target['position_m'][0] + ox, target['position_m'][1], target['position_m'][2] + oz], 'quaternion_xyzw': list(target['quaternion_xyzw'])}
                return st

            adp_h.state = loaded_state
            res_h = self.execute_batch(adp_h, act_h, plan_h)
            self.assertEqual(res_h['status'], expected_status)
            preview_eff = plan_h['stage_costs']['loaded_move']['effective_tolerances']
            exec_err = res_h['stage_results']['loaded_move']['endpoint_error']
            self.assertEqual(preview_eff['max_upward_dz_m'], 0.0)
            self.assertEqual(preview_eff['max_downward_dz_m'], 0.040)
            self.assertEqual(exec_err['effective_tolerances']['max_upward_dz_m'], 0.0)
            self.assertEqual(exec_err['effective_tolerances']['max_downward_dz_m'], 0.040)
            self.assertEqual(bool(exec_err.get('vertical_compliance_accepted')), expect_vert)
            if expected_status == 'completed':
                self.assertEqual(adp_h._batch_plans[plan_h['plan_id']]['downward_dz_tolerances']['left'], 0.040)

        for dx_m, dz_m, expected_status in ((0.010, 0.028, 'completed'), (0.012, -0.012, 'failed')):
            adp_c, _, _ = preview_adapter()
            jaws(adp_c)
            act_c = batch([move('seat_move', arm='left', angle=-0.25, tolerance_preset='compliant')])
            plan_c = adp_c.preview_batch(act_c)
            end_step = plan_c['stage_costs']['seat_move']['end_native_step']
            tgt_c = deepcopy(adp_c._batch_plans[plan_c['plan_id']]['stages']['seat_move']['target'])
            orig_c = adp_c.state

            def seat_state(orig=orig_c, adp=adp_c, stop_step=end_step, target=tgt_c, ox=dx_m, oz=dz_m):
                st = orig()
                if adp.env.take_action_cnt[0] >= stop_step:
                    st['arms']['left']['ee_pose'] = {'position_m': [target['position_m'][0] + ox, target['position_m'][1], target['position_m'][2] + oz], 'quaternion_xyzw': list(target['quaternion_xyzw'])}
                return st

            adp_c.state = seat_state
            res_c = self.execute_batch(adp_c, act_c, plan_c)
            self.assertEqual(res_c['status'], expected_status)



if __name__ == "__main__":
    unittest.main()
