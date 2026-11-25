"""接触即停（contact_stop）阶段驻停后的起点平滑重锚定与安全门控回归测试。"""

NI_RUA_TEST_TIER = "component"

from copy import deepcopy
import math
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.core.observation_policy import reject_privileged
from ni_rua.core.presentation.actor_presentation import ActorPresentation, metadata_findings
from ni_rua.core.storage.store import Store
from ni_rua.core.tool_output import validate_display
from ni_rua.core.contracts import canonical
from ni_rua.perception.external import ik_prefilter
from ni_rua.robot.motion.contact_reanchor import reanchor_move_stage_from_contact
from ni_rua.robot.motion.endpoint_ik_polish import polish_trajectory_endpoint
from support.motion_fixtures import batch, jaw, jaws
from support.motion_fixtures import replan_adapter, wait_action
from support.motion_fixtures import preview_adapter


def _pose_3d(angle_xy: float, z_m: float) -> dict:
    """构造水平圆周角与独立高度 Z 的三维末端位姿。"""
    return dict(
        position_m=[math.cos(angle_xy), math.sin(angle_xy), float(z_m)],
        quaternion_xyzw=[0.0, 0.0, math.sin(angle_xy / 2.0), math.cos(angle_xy / 2.0)],
    )


def _move_3d(identifier: str, *, arm: str = "left", angle_xy: float = 0.0, z_m: float = 1.0, **values) -> dict:
    """构造显式指定三维末端目标的 move 阶段字典。"""
    own = wait_action(timeout_s=30)
    del own["hold_native_steps"]
    own["path"] = dict(kind="backend_free", waypoints=[{arm: _pose_3d(angle_xy, z_m)}])
    return dict(id=identifier, kind="move", arm=arm, action=own, **values)


def _contact_3d_adapter(*, z_scale_m_per_rad: float = 0.10, left_q2_floor_rad: float | None = None):
    """构造具备真实关节步进与接触高度下限截断的三维双臂测试适配器。

    运动学映射：
    - `q[0]` 控制水平方位角 `(cos(q[0]), sin(q[0]))`；
    - `q[2]` 控制垂直高度 `z = 1.0 + z_scale_m_per_rad * q[2]`；
    - 当 `left_q2_floor_rad` 非空时，物理环境在 `take_action` 期间将左臂 `q[2]` 截断在 `>= left_q2_floor_rad`，
      真实复现接触台面/物体表面后关节与末端同步驻停的物理过程。
    """
    adapter, planner, dispatched = preview_adapter()
    jaws(adapter)
    measured = {"left": [0.0] * 6, "right": [0.0] * 6}

    def custom_plan_path(current, target, origin):
        planner.pose_calls.append((np.array(current, copy=True), deepcopy(target), origin))
        goal = np.array(current, copy=True)
        goal[0] = math.atan2(target[1], target[0])
        goal[2] = (float(target[2]) - 1.0) / z_scale_m_per_rad
        original = np.stack([current, goal])
        planner.originals.append(original.copy())
        return {"status": "Success", "position": original}

    planner.plan_path = custom_plan_path
    robot = adapter.robot.robot_list[0]

    def state():
        arms = {}
        for side in ("left", "right"):
            q = list(measured[side])
            pose = _pose_3d(q[0], 1.0 + z_scale_m_per_rad * q[2])
            pose.update(source_kind="measurement", frame_id="env_origin", tcp_config_id="arx-x5-link6-v1")
            arms[side] = dict(
                q_rad=q,
                joint_names=list(robot.arm_joints_name),
                source_kind="measurement",
                ee_pose=pose,
                gripper=dict(
                    position_source_kind="measurement",
                    position_m=[adapter.last_dispatched_goals[side]],
                    joint_names=["jaw"],
                ),
            )
        return {"arms": arms}

    adapter.state = state
    raw_take_action = adapter.env.take_action

    def take_action(action):
        raw_take_action(action)
        for side in ("left", "right"):
            q_next = list(dispatched[-1][f"{side}_arm_joint_state"])
            if side == "left" and left_q2_floor_rad is not None and q_next[2] < left_q2_floor_rad:
                q_next[2] = float(left_q2_floor_rad)
            measured[side] = q_next

    adapter.env.take_action = take_action
    return adapter, planner, dispatched, measured


class BatchContactReanchorTests(unittest.TestCase):
    """验证 contact_stop 早停或顺应到位后后续阶段平滑重锚定与安全门控。"""

    def _execute(self, adapter, action: dict, plan: dict, **extra) -> dict:
        req = dict(action, batch_plan_id=plan["plan_id"], **extra)
        return adapter.execute(req, lambda update: None, threading.Event())

    def test_contact_stop_early_touchdown_gripper_and_lift_reanchors_smoothly(self):
        # 复现原 Bug：contact_stop 下探在 q2=-0.20rad (z=0.98m) 提前触停，与原定目标 q2=-0.60rad (z=0.94m) 相差 0.40rad (>0.10rad 阈值)
        adapter, _, dispatched, _ = _contact_3d_adapter(z_scale_m_per_rad=0.10, left_q2_floor_rad=-0.20)
        adapter.last_dispatched_goals["left"] = 1.0
        action = batch([
            _move_3d("descend", arm="left", z_m=0.940, stall_guard_preset="contact_stop", max_upward_dz_m=0.045),
            jaw(identifier="close_jaw", arm="left", goal=0.0),
            _move_3d("lift", arm="left", z_m=1.000),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan["status"], "planned")

        with (
            patch.object(adapter, "_plan_segment", side_effect=AssertionError("execute 严禁调用 _plan_segment")),
            patch.object(adapter, "_resample_action_plan", side_effect=AssertionError("execute 严禁调用 _resample_action_plan")),
        ):
            receipt = self._execute(adapter, action, plan)

        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertEqual(receipt["completed_stage_ids"], ["descend", "close_jaw", "lift"])
        self.assertEqual(
            receipt.get("preview_execution_semantics"),
            "cached native rows with contact_stop hold/start-residual reanchoring; no replanning or jaw rebase",
        )

        descend_res = receipt["stage_results"]["descend"]
        self.assertTrue(descend_res.get("contact_stopped_early"))
        self.assertEqual(descend_res.get("actual_contact_position_m"), [1.0, 0.0, 0.98])

        lift_res = receipt["stage_results"]["lift"]
        self.assertTrue(lift_res.get("reanchored_from_contact"))
        reanchor = lift_res["contact_reanchor"]
        self.assertEqual(reanchor["status"], "reanchored")
        self.assertEqual(reanchor["source_stage_id"], "descend")
        self.assertAlmostEqual(reanchor["nominal_start_q_rad"][2], -0.60, places=5)
        self.assertAlmostEqual(reanchor["actual_start_q_rad"][2], -0.20, places=5)
        self.assertAlmostEqual(reanchor["max_start_joint_delta_rad"], 0.40, places=5)

        # 验证实际下发的关节指令：close_jaw 全程驻停于 -0.20rad，lift 从 -0.20rad 单调平滑回升至 0.0rad，零反向硬拽阶跃
        close_cost = plan["stage_costs"]["close_jaw"]
        lift_cost = plan["stage_costs"]["lift"]
        close_q2 = [dispatched[i]["left_arm_joint_state"][2] for i in range(close_cost["start_native_step"], close_cost["end_native_step"])]
        self.assertTrue(all(abs(v - (-0.20)) < 1e-9 for v in close_q2), close_q2)

        lift_q2 = [dispatched[i]["left_arm_joint_state"][2] for i in range(lift_cost["start_native_step"], lift_cost["end_native_step"])]
        prev_q2 = -0.20
        for step_q2 in lift_q2:
            dq = step_q2 - prev_q2
            self.assertGreater(dq, 0.0)
            self.assertLessEqual(dq, 0.05 + 1e-9)
            prev_q2 = step_q2
        self.assertAlmostEqual(lift_q2[-1], 0.0, places=6)

    def test_small_contact_offset_eliminates_downward_step_jump_on_retract(self):
        # 接触早停关节差 0.08rad (<0.10rad 起点漂移阈值)：旧逻辑虽不报 batch_segment_start_changed，但首帧会向 -0.60rad 反向硬拽
        adapter, _, dispatched, _ = _contact_3d_adapter(z_scale_m_per_rad=0.40, left_q2_floor_rad=-0.52)
        adapter.last_dispatched_goals["left"] = 0.0
        action = batch([
            _move_3d("place_down", arm="left", z_m=0.760, stall_guard_preset="contact_stop", max_upward_dz_m=0.040),
            jaw(identifier="open_jaw", arm="left", goal=1.0),
            _move_3d("retract_up", arm="left", z_m=1.000),
        ])
        plan = adapter.preview_batch(action)
        receipt = self._execute(adapter, action, plan)

        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertTrue(receipt["stage_results"]["retract_up"].get("reanchored_from_contact"))
        retract_cost = plan["stage_costs"]["retract_up"]
        retract_q2 = [dispatched[i]["left_arm_joint_state"][2] for i in range(retract_cost["start_native_step"], retract_cost["end_native_step"])]
        self.assertGreater(retract_q2[0], -0.52)
        for prev_v, next_v in zip([-0.52, *retract_q2[:-1]], retract_q2):
            self.assertGreaterEqual(next_v - prev_v, 0.0)

    def test_endpoint_vertical_compliance_contact_stop_reanchors_next_move(self):
        # 开爪预抓取使用纯默认 contact_stop（不覆盖 max_upward_dz_m），在阶段终点以 +18mm 垂直顺应完成时同样重锚定后续 lift
        adapter, _, dispatched, _ = _contact_3d_adapter(z_scale_m_per_rad=0.05, left_q2_floor_rad=-0.24)
        adapter.last_dispatched_goals["left"] = 1.0
        action = batch([
            _move_3d("grasp_down", arm="left", z_m=0.970, stall_guard_preset="contact_stop"),
            jaw(identifier="grasp_close", arm="left", goal=0.0),
            _move_3d("grasp_lift", arm="left", z_m=1.000),
        ])
        plan = adapter.preview_batch(action)
        receipt = self._execute(adapter, action, plan)

        self.assertEqual(receipt["status"], "completed", receipt)
        self.assertTrue(receipt["stage_results"]["grasp_lift"].get("reanchored_from_contact"))
        lift_cost = plan["stage_costs"]["grasp_lift"]
        first_lift_q2 = dispatched[lift_cost["start_native_step"]]["left_arm_joint_state"][2]
        self.assertGreater(first_lift_q2, -0.24)

    def test_checkpoint_after_gripper_pause_and_resume_preserves_reanchor(self):
        # 验证带 checkpoint_after_gripper=True 暂停后恢复执行：不触发首步跳变错误，且恢复回执完整保留 reanchored_from_contact
        adapter, _, _, _ = _contact_3d_adapter(z_scale_m_per_rad=0.10, left_q2_floor_rad=-0.20)
        adapter.last_dispatched_goals["left"] = 1.0
        action = batch(
            [
                _move_3d("descend", arm="left", z_m=0.940, stall_guard_preset="contact_stop", max_upward_dz_m=0.045),
                jaw(identifier="close_jaw", arm="left", goal=0.0),
                _move_3d("lift", arm="left", z_m=1.000),
            ],
            checkpoint_after_gripper=True,
        )
        plan = adapter.preview_batch(action)
        paused = self._execute(adapter, action, plan)
        self.assertEqual(paused["status"], "stopped")
        self.assertEqual(paused["reason"], "observation_checkpoint")
        self.assertEqual(paused["completed_stage_ids"], ["descend", "close_jaw"])
        self.assertTrue(paused["stage_results"]["lift"].get("reanchored_from_contact"))

        resumed = self._execute(
            adapter,
            action,
            plan,
            batch_expected_cursor=paused["batch_cursor"],
            batch_expected_revision=paused["batch_revision"],
        )
        self.assertEqual(resumed["status"], "completed", resumed)
        self.assertEqual(resumed["completed_stage_ids"], ["descend", "close_jaw", "lift"])
        self.assertTrue(resumed["stage_results"]["lift"].get("reanchored_from_contact"))

    def test_step_budget_exceeded_rejects_honestly_without_joint_jump(self):
        # 下一 move 阶段原计划仅从 -0.60rad 微调至 -0.64rad（仅分配 1 步窗口），接触早停于 -0.20rad 后需 0.44rad 跨度，诚实拒绝跳变
        adapter, _, _, _ = _contact_3d_adapter(z_scale_m_per_rad=0.10, left_q2_floor_rad=-0.20)
        adapter.last_dispatched_goals["left"] = 1.0
        action = batch([
            _move_3d("descend", arm="left", z_m=0.940, stall_guard_preset="contact_stop", max_upward_dz_m=0.045),
            jaw(identifier="close_jaw", arm="left", goal=0.0),
            _move_3d("short_press", arm="left", z_m=0.936),
        ])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan["stage_costs"]["short_press"]["native_steps"], 1)

        receipt = self._execute(adapter, action, plan)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["reason"], "contact_reanchor_step_budget_exceeded")
        self.assertEqual(receipt["failed_stage_id"], "short_press")
        self.assertEqual(receipt["completed_stage_ids"], ["descend", "close_jaw"])
        self.assertEqual(receipt["stage_results"]["short_press"]["completed_steps"], 0)
        self.assertFalse(receipt["stage_results"]["short_press"]["reanchored_from_contact"])
        self.assertGreater(receipt["contact_reanchor"]["required_min_steps"], 1)

    def test_elevated_contact_lateral_retract_reanchors_smoothly_without_blind_geometry_gate(self):
        # 拆除盲目防斜切否决门控后：前置接触停在高处 (z=0.98m) 时，只要预分配时间窗口满足单步关节速度上限，后续横移+高度变化平滑重锚定完成
        adapter, _, _, _ = _contact_3d_adapter(z_scale_m_per_rad=0.10, left_q2_floor_rad=-0.20)
        adapter.last_dispatched_goals["left"] = 0.0
        action_ok = batch([
            _move_3d("insert_try", arm="left", angle_xy=0.0, z_m=0.940, stall_guard_preset="contact_stop"),
            _move_3d("slide_retract", arm="left", angle_xy=-0.50, z_m=0.960, max_upward_dz_m=0.030),
        ])
        plan_ok = adapter.preview_batch(action_ok)
        receipt_ok = self._execute(adapter, action_ok, plan_ok)
        self.assertEqual(receipt_ok["status"], "completed", receipt_ok)
        self.assertEqual(receipt_ok["completed_stage_ids"], ["insert_try", "slide_retract"])
        self.assertTrue(receipt_ok["stage_results"]["slide_retract"]["reanchored_from_contact"])

        # 若后续阶段时间窗口过短不足以容纳关节跨度，仅由关节步长预算门控诚实拦截（绝不触发已删除的 unsafe_lateral_descent）
        adapter_short, _, _, _ = _contact_3d_adapter(z_scale_m_per_rad=0.10, left_q2_floor_rad=-0.20)
        adapter_short.last_dispatched_goals["left"] = 0.0
        action_short = batch([
            _move_3d("insert_try", arm="left", angle_xy=0.0, z_m=0.940, stall_guard_preset="contact_stop"),
            _move_3d("slide_low", arm="left", angle_xy=-0.25, z_m=0.940),
        ])
        plan_short = adapter_short.preview_batch(action_short)
        receipt_short = self._execute(adapter_short, action_short, plan_short)
        self.assertEqual(receipt_short["status"], "failed")
        self.assertEqual(receipt_short["reason"], "contact_reanchor_step_budget_exceeded")

    def test_non_contact_stop_joint_drift_still_triggers_segment_start_changed(self):
        # 未开启 contact_stop 的阶段若后续 move 起点发生 >0.10rad 关节偏差，必须继续严格报 batch_segment_start_changed
        adapter, _, _, measured = _contact_3d_adapter(z_scale_m_per_rad=0.10, left_q2_floor_rad=None)
        action = batch([
            _move_3d("free_move", arm="left", z_m=0.980),
            jaw(identifier="close_jaw", arm="left", goal=0.0),
            _move_3d("lift", arm="left", z_m=1.000),
        ])
        plan = adapter.preview_batch(action)
        lift_start = plan["stage_costs"]["lift"]["start_native_step"]
        orig_take = adapter.env.take_action

        def perturb_at_lift_start(act):
            orig_take(act)
            if adapter.env.take_action_cnt[0] == lift_start:
                measured["left"][2] = 0.25

        adapter.env.take_action = perturb_at_lift_start
        receipt = self._execute(adapter, action, plan)
        self.assertEqual(receipt["status"], "failed")
        self.assertEqual(receipt["reason"], "batch_segment_start_changed")
        self.assertEqual(receipt["failed_stage_id"], "lift")
        self.assertNotIn("reanchored_from_contact", receipt["stage_results"]["lift"])

    def test_dual_arm_concurrent_peer_trajectory_untouched_and_actor_scrubber_clean(self):
        # 左臂接触早停并重锚定期间，右臂并发多阶段轨迹必须逐帧保持原样；同时验证回执满足脱敏与公开显示契约
        adapter, _, dispatched, _ = _contact_3d_adapter(z_scale_m_per_rad=0.10, left_q2_floor_rad=-0.20)
        adapter.last_dispatched_goals["left"] = 1.0
        action = batch([
            _move_3d("l_down", arm="left", z_m=0.940, stall_guard_preset="contact_stop", max_upward_dz_m=0.045),
            jaw(identifier="l_close", arm="left", goal=0.0),
            _move_3d("l_lift", arm="left", z_m=1.000),
            _move_3d("r_out", arm="right", angle_xy=-0.30, z_m=1.000),
            _move_3d("r_back", arm="right", angle_xy=0.0, z_m=1.000),
        ])
        plan = adapter.preview_batch(action)
        planned_right_rows = [list(row[7:13]) for row in adapter._batch_plans[plan["plan_id"]]["rows"]]

        receipt = self._execute(adapter, action, plan)
        self.assertEqual(receipt["status"], "completed", receipt)
        for step_idx, expected_right_q in enumerate(planned_right_rows):
            np.testing.assert_array_equal(expected_right_q, dispatched[step_idx]["right_arm_joint_state"])

        stage_ids = [s["id"] for s in action["stages"]]
        reject_privileged(receipt, stage_ids=stage_ids)
        validate_display(receipt)
        with tempfile.TemporaryDirectory() as tmp:
            store = Store(
                Path(tmp) / "run",
                dict(run_id="reanchor", episode_id="ep", env_instance_id="env", env_generation=1),
            )
            try:
                presented = ActorPresentation(store).present_data(receipt)
                validate_display(presented)
                self.assertEqual(metadata_findings(canonical(presented)), [])
            finally:
                store.close()

    def test_contact_reanchor_preserves_scurve_deceleration_and_curved_arch(self):
        class _StepLimitAdapter:
            def _waypoint_speed_scale(self, action_dict: dict, idx: int) -> float:
                return float(action_dict.get("speed_scale", 1.0))

            def _speed_step_limits(self, arm: str, speed_scale: float) -> np.ndarray:
                return np.full(6, 0.04 * float(speed_scale), dtype=float)

        adapter = _StepLimitAdapter()
        step_profile = np.array([0.040, 0.075, 0.076, 0.076, 0.072, 0.055, 0.032, 0.018, 0.012, 0.004], dtype=float)
        orig_j0 = np.cumsum(step_profile)
        rows = [[float(orig_j0[i])] + [0.0] * 13 for i in range(len(step_profile))]
        next_rec = dict(start=0, end=len(step_profile), expected_start_q=[0.0] * 6, rows=[list(r) for r in rows], action=dict(speed_scale=1.0))
        held_q = [-0.065, 0.0, 0.0, 0.0, 0.0, 0.0]
        info = reanchor_move_stage_from_contact(adapter, "left", "seat", dict(contact_stopped_early=True, target=dict(position_m=[0.20, 0.10, 0.78])), next_rec, rows, held_q, [0.20, 0.10, 0.792])
        self.assertEqual(info["status"], "reanchored")
        reanchored_j0 = np.array([r[0] for r in next_rec["rows"]], dtype=float)
        deltas_j0 = np.abs(np.diff(np.concatenate([[held_q[0]], reanchored_j0])))
        self.assertLessEqual(float(np.max(deltas_j0)), 0.08 + 1e-9)
        self.assertAlmostEqual(float(reanchored_j0[-1]), float(orig_j0[-1]), places=7)
        self.assertLess(float(deltas_j0[-1]), 0.020)

        n_curve = 8
        nominal_start_q = np.array([0.10, -0.40, 0.50, -0.20, 0.30, 0.00], dtype=float)
        target_q = np.array([0.30, -0.10, 0.20, 0.10, 0.10, 0.20], dtype=float)
        alphas = np.arange(1, n_curve + 1, dtype=float)[:, None] / n_curve
        arch = np.sin(math.pi * alphas) * np.array([[0.0, 0.12, -0.10, 0.05, 0.0, 0.0]])
        orig_q = nominal_start_q[None, :] + alphas * (target_q - nominal_start_q)[None, :] + arch
        curve_rows = [orig_q[i].tolist() + [0.0] * 8 for i in range(n_curve)]
        planned_arch = [[0.20 + 0.10 * ((i + 1) / n_curve), 0.10, 0.78 + 0.09 * math.sin(math.pi * ((i + 1) / n_curve))] for i in range(n_curve)]
        curve_rec = dict(
            start=0,
            end=n_curve,
            expected_start_q=nominal_start_q.tolist(),
            rows=[list(r) for r in curve_rows],
            planned_positions_m=planned_arch,
            action=dict(speed_scale=1.0),
        )
        curve_held_q = (nominal_start_q + np.array([0.0056, -0.0032, 0.0021, 0.0, 0.0, 0.0])).tolist()
        curve_info = reanchor_move_stage_from_contact(
            adapter,
            "left",
            "grasp_down",
            dict(contact_stopped_early=True, target=dict(position_m=[0.20, 0.10, 0.78])),
            curve_rec,
            curve_rows,
            curve_held_q,
            [0.201, 0.099, 0.795],
        )
        self.assertEqual(curve_info["status"], "reanchored")
        self.assertGreater(curve_rec["planned_positions_m"][n_curve // 2 - 1][2], 0.85)
        reanchored = np.array([r[:6] for r in curve_rows], dtype=float)
        np.testing.assert_allclose(reanchored[-1], target_q, atol=1e-9)
        step_diffs = np.abs(np.diff(np.vstack([curve_held_q, reanchored]), axis=0))
        orig_step_diffs = np.abs(np.diff(np.vstack([nominal_start_q, orig_q]), axis=0))
        self.assertLessEqual(float(np.max(step_diffs)), max(0.04, float(np.max(orig_step_diffs))) + 1e-9)
        u_j0 = (reanchored[:, 0] - curve_held_q[0]) / (target_q[0] - curve_held_q[0])
        chord_j1 = curve_held_q[1] + u_j0 * (target_q[1] - curve_held_q[1])
        self.assertGreater(float(np.max(reanchored[:, 1] - chord_j1)), 0.10)

        bad_held_q = (nominal_start_q + np.array([0.60, -0.55, 0.0, 0.0, 0.0, 0.0])).tolist()
        err = reanchor_move_stage_from_contact(
            adapter,
            "left",
            "grasp_down",
            dict(contact_stopped_early=True, target=dict(position_m=[0.20, 0.10, 0.78])),
            curve_rec,
            curve_rows,
            bad_held_q,
            [0.201, 0.099, 0.795],
        )
        self.assertEqual(err["status"], "rejected")
        self.assertEqual(err["reason"], "contact_reanchor_step_budget_exceeded")

        # 双关节错峰运动步长预算样本（修复跨关节逐帧最大值虚高累加拒绝）
        n_stag = 10
        j0_stag = np.cumsum(np.array([0.065] * 5 + [0.005] * 5, dtype=float))
        j1_stag = np.cumsum(np.array([0.005] * 5 + [0.065] * 5, dtype=float))
        stag_rows = [[float(j0_stag[i]), float(j1_stag[i])] + [0.0] * 12 for i in range(n_stag)]
        stag_rec = dict(
            start=0,
            end=n_stag,
            expected_start_q=[0.0] * 6,
            rows=[list(r) for r in stag_rows],
            action=dict(speed_scale=1.0),
        )
        stag_info = reanchor_move_stage_from_contact(
            adapter,
            "left",
            "staggered_down",
            dict(contact_stopped_early=True, target=dict(position_m=[0.20, 0.10, 0.78])),
            stag_rec,
            stag_rows,
            [-0.030, -0.025, 0.0, 0.0, 0.0, 0.0],
            [0.20, 0.10, 0.795],
        )
        self.assertIsNotNone(stag_info)
        self.assertEqual(stag_info["status"], "reanchored")
        self.assertLessEqual(stag_info["required_min_steps"], n_stag)

        # 未提前接触停止且仅 1e-7 rad 数值噪声：返回 None 无操作且原行完全不变
        noop_rows = [[0.05 * (i + 1)] + [0.0] * 13 for i in range(4)]
        noop_orig = [list(r) for r in noop_rows]
        noop_rec = dict(
            start=0,
            end=4,
            expected_start_q=[0.0] * 6,
            rows=[list(r) for r in noop_rows],
            action=dict(speed_scale=1.0),
        )
        noop_res = reanchor_move_stage_from_contact(
            adapter,
            "left",
            "touch",
            dict(contact_stopped_early=False),
            noop_rec,
            noop_rows,
            [1e-7, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.20, 0.10, 0.78],
        )
        self.assertIsNone(noop_res)
        self.assertEqual(noop_rows, noop_orig)
        self.assertEqual(noop_rec["rows"], noop_orig)

    def test_endpoint_ik_polish_reduces_4_8mm_slop_to_sub_0_1mm_and_wires_into_robot_worker(self):
        from scipy.spatial.transform import Rotation

        chain = ik_prefilter.resolve_6r_chain_spec("right")
        t_wb = ik_prefilter.resolve_arm_base_transform("right")
        limits = chain.joint_limits_rad.copy()

        def world_link6(q_rad: np.ndarray) -> np.ndarray:
            pos_b, rot_b, _, _ = ik_prefilter.fk_batch_base(np.asarray(q_rad, dtype=np.float64).reshape(1, 6), chain)
            mat = np.eye(4, dtype=np.float64)
            mat[:3, :3] = t_wb[:3, :3] @ rot_b[0]
            mat[:3, 3] = t_wb[:3, :3] @ pos_b[0] + t_wb[:3, 3]
            return mat

        def quat_xyzw(rot: np.ndarray) -> list[float]:
            return [float(v) for v in Rotation.from_matrix(rot).as_quat()]

        def make_pose_error(target: dict):
            t_pos = np.asarray(target["position_m"], dtype=np.float64)
            t_quat = np.asarray(target["quaternion_xyzw"], dtype=np.float64)
            t_quat /= float(np.linalg.norm(t_quat))

            def pose_error(q_rad: np.ndarray) -> tuple[float, float]:
                mat = world_link6(q_rad)
                pos_err = float(np.linalg.norm(mat[:3, 3] - t_pos))
                cur_quat = np.asarray(quat_xyzw(mat[:3, :3]), dtype=np.float64)
                return pos_err, float(2.0 * math.acos(min(1.0, abs(float(np.dot(cur_quat, t_quat))))))

            pose_error.world_link6 = world_link6
            return pose_error

        q_exact = np.array([0.15, 0.72, 0.68, -0.45, 0.22, -0.60], dtype=float)
        mat_goal = world_link6(q_exact)
        target = dict(position_m=[float(v) for v in mat_goal[:3, 3]], quaternion_xyzw=quat_xyzw(mat_goal[:3, :3]))
        pose_error = make_pose_error(target)
        q_slop = q_exact + np.array([0.013, -0.011, 0.010, -0.014, 0.015, -0.018], dtype=float)
        init_pos_err, _ = pose_error(q_slop)
        self.assertGreater(init_pos_err, 0.0040)
        self.assertLess(init_pos_err, 0.0060)
        q_start = np.array([0.0, 0.35, 0.35, -0.20, 0.0, 0.0], dtype=float)
        s_grid = np.linspace(0.0, 1.0, 12, dtype=float)[:, None]
        raw_traj = (1.0 - s_grid) * q_start[None, :] + s_grid * q_slop[None, :]

        diag: dict = {}
        polished = polish_trajectory_endpoint(raw_traj, limits=limits, target=target, pose_error=pose_error, world_link6=world_link6, diagnostic=diag)
        self.assertTrue(diag.get("selected"))
        self.assertEqual(diag.get("reason"), "polished_endpoint_ik_residual")
        after_pos_err, after_ori_err = pose_error(polished[-1])
        self.assertLess(after_pos_err, 0.0001)
        self.assertLess(after_ori_err, 0.0010)
        np.testing.assert_array_equal(polished[0], raw_traj[0])
        self.assertLessEqual(float(np.max(np.abs(polished[-1] - raw_traj[-1]))), 0.045 + 1e-9)
        self.assertTrue(np.all(polished >= limits[:, 0] - 1e-9))
        self.assertTrue(np.all(polished <= limits[:, 1] + 1e-9))

        exact_traj = (1.0 - s_grid) * q_start[None, :] + s_grid * q_exact[None, :]
        self.assertIs(polish_trajectory_endpoint(exact_traj, limits=limits, target=target, pose_error=pose_error, world_link6=world_link6), exact_traj)

        q_far = q_exact + np.array([0.06, -0.05, 0.05, 0.0, 0.0, 0.0], dtype=float)
        self.assertGreater(pose_error(q_far)[0], 0.010)
        far_traj = (1.0 - s_grid) * q_start[None, :] + s_grid * q_far[None, :]
        untouched_far = polish_trajectory_endpoint(far_traj, limits=limits, target=target, pose_error=pose_error, world_link6=world_link6)
        self.assertIs(untouched_far, far_traj)

        adapter, robot, planner, _ = replan_adapter("right", enabled=True, wound=False)
        planner.plan_path = lambda cur, tgt, orig: {"status": "Success", "position": raw_traj.copy()}
        adapter._joint_replan_geometry = lambda r, p, t: (["revolute"] * 6, lambda q: {"base": np.eye(4), "tool": world_link6(q)}, make_pose_error(t))
        planning_diags: list[dict] = []
        seg_positions, _ = adapter._plan_segment(robot, target, planning_diags, waypoint_index=0, joint_replan_context={"deadline": 1e12})
        self.assertLess(pose_error(seg_positions[-1])[0], 0.0001)
        self.assertTrue(any(d.get("stage") == "endpoint_ik_polish" and d.get("selected") for d in planning_diags))

    def test_endpoint_ik_polish_bimanual_chain_and_obstacle_distance_under_pink_vbool_and_coal_fallbacks(self):
        import sys
        from contextlib import ExitStack
        from unittest.mock import patch
        from scipy.spatial.transform import Rotation
        from ni_rua.robot.model import kinematics as kin
        from ni_rua.robot.motion.scene_obstacles import _segment_xy_distance_to_box

        orig_cache = kin._PINK_DAQP_CACHE
        self.addCleanup(setattr, kin, "_PINK_DAQP_CACHE", orig_cache)

        vbool_err = TypeError(
            "No to_python (by-value) converter found for C++ type: std::vector<bool, std::allocator<bool> >"
        )

        for arm in ("left", "right"):
            spec = ik_prefilter.resolve_6r_chain_spec(arm)
            k_chain = spec.to_kinematic_chain()
            t_wb = ik_prefilter.resolve_arm_base_transform(arm)
            limits = k_chain.joint_limits_rad.copy()
            q_exact = np.array([0.14, 0.70, 0.66, -0.42, 0.24, -0.52], dtype=float)
            if arm == "left":
                q_exact[0] = -0.14
            mat_goal = kin.fk_links(k_chain, q_exact, T_world_base=t_wb)[k_chain.tip_link]
            quat = [float(v) for v in Rotation.from_matrix(mat_goal[:3, :3]).as_quat()]
            target = dict(position_m=[float(v) for v in mat_goal[:3, 3]], quaternion_xyzw=quat)

            def _world_fk(q_vec, _chain=k_chain, _wb=t_wb):
                return kin.fk_links(_chain, q_vec, T_world_base=_wb)[_chain.tip_link]

            def _pose_err(q_vec, _goal=mat_goal, _wfk=_world_fk):
                return kin.pose_error(_wfk(q_vec), _goal)

            _pose_err.world_link6 = _world_fk  # type: ignore[attr-defined]
            _pose_err.chain = k_chain  # type: ignore[attr-defined]
            _pose_err.T_world_base = t_wb  # type: ignore[attr-defined]

            q_slop = q_exact + np.array([0.012, -0.010, 0.009, -0.013, 0.014, -0.016], dtype=float)
            s_grid = np.linspace(0.0, 1.0, 12, dtype=float)[:, None]
            q_start = np.array([0.0, 0.35, 0.35, -0.20, 0.0, 0.0], dtype=float)
            raw_traj = (1.0 - s_grid) * q_start[None, :] + s_grid * q_slop[None, :]

            for mode_name, patches in [
                ("pink_daqp", (patch.object(kin, "_PINK_DAQP_CACHE", True),)),
                ("probe_false", (patch("ni_rua.robot.model.kinematics._pink_daqp_usable", return_value=False),)),
                ("pink_config_vbool", (patch.object(kin, "_PINK_DAQP_CACHE", True), patch("pink.Configuration", side_effect=vbool_err))),
                ("pink_limit_vbool", (patch.object(kin, "_PINK_DAQP_CACHE", True), patch("pink.limits.ConfigurationLimit", side_effect=vbool_err))),
                ("solve_ik_direct_vbool", (patch("ni_rua.robot.motion.endpoint_ik_polish.solve_ik", side_effect=vbool_err),)),
            ]:
                with self.subTest(arm=arm, mode=mode_name), ExitStack() as stack:
                    for p in patches:
                        stack.enter_context(p)
                    diag: dict = {}
                    polished = polish_trajectory_endpoint(
                        raw_traj,
                        limits=limits,
                        target=target,
                        pose_error=_pose_err,
                        diagnostic=diag,
                    )
                    self.assertTrue(diag.get("selected"), (arm, mode_name, diag))
                    self.assertEqual(diag.get("reason"), "polished_endpoint_ik_residual")
                    after_pe, after_oe = _pose_err(polished[-1])
                    self.assertLess(after_pe, 1e-4)
                    self.assertLess(after_oe, 1e-3)
                    np.testing.assert_array_equal(polished[0], raw_traj[0])

            with patch("ni_rua.robot.motion.endpoint_ik_polish.solve_ik", side_effect=TypeError("unrelated_polish_bug")):
                with self.assertRaisesRegex(TypeError, "unrelated_polish_bug"):
                    polish_trajectory_endpoint(raw_traj, limits=limits, target=target, pose_error=_pose_err)

        # Verify _segment_xy_distance_to_box across midpoint-hit, edge-sep, corner-nearest, endpoint-nearest, diagonal-hit, and degenerate point.
        xy_cases = [
            ([0.0, 0.0], [1.0, 0.0], [0.030, -0.001], [0.032, 0.001], 0.0),
            ([0.0, 0.05], [1.0, 0.05], [0.2, -0.02], [0.3, 0.01], 0.04),
            ([-0.1, 0.1], [0.1, -0.1], [0.02, 0.02], [0.10, 0.10], math.sqrt(0.0008)),
            ([0.05, 0.05], [0.20, 0.20], [-0.10, -0.10], [0.0, 0.0], math.hypot(0.05, 0.05)),
            ([-0.05, -0.05], [0.05, 0.05], [-0.01, -0.01], [0.01, 0.01], 0.0),
            ([0.12, 0.0], [0.12, 0.0], [0.0, -0.05], [0.10, 0.05], 0.02),
        ]
        for p0, p1, lo_xy, hi_xy, expected_d in xy_cases:
            d_coal = _segment_xy_distance_to_box(p0, p1, lo_xy, hi_xy)
            with patch.dict(sys.modules, {"coal": None}):
                d_fb = _segment_xy_distance_to_box(p0, p1, lo_xy, hi_xy)
            self.assertAlmostEqual(d_coal, expected_d, places=7)
            self.assertAlmostEqual(d_fb, expected_d, places=9)


if __name__ == "__main__":
    unittest.main()
