"""Phase 1-1 单元测试：无图数值 mode='preflight'、grasp_preview_3d 软硬风险解耦与 execute_if_clean 单回合直通。"""
from copy import deepcopy
import json
import unittest
from unittest.mock import Mock, patch
from ni_rua.cortex.context.actor_instructions import (
    CALIBRATED_OBSERVATION_INSTRUCTIONS,
    INSTRUCTIONS,
    OPERATION_NOTES,
    instructions,
)
from ni_rua.core.tool_output import wire_response
from ni_rua.robot.motion.batch_contract import IMAGE_DELIVERY_MODES, validate_batch_action
from ni_rua.runtime.actions.grasp_preview_3d import audit_batch_grasp_3d
from ni_rua.runtime.actions.robot_action_schema import ROBOT_ACTION_TOOL
from ni_rua.runtime.actions.unified_actions import compile_request, dispatch as typed_dispatch
from ni_rua.runtime.programs.runner import ProgramRunner
from ni_rua.runtime.tools.builtins import action_effect, create_registry
from support.host_capabilities import expect, program_isolation
from support.motion_fixtures import DualArmHarnessBase

NI_RUA_TEST_TIER = "component"


COLUMNS = [
    "id",
    "centre_m",
    "top_z",
    "bottom_z",
    "length",
    "width",
    "yaw_deg",
    "tilt_deg",
    "support",
    "quality",
]


class PreflightAndExecuteIfCleanTests(unittest.TestCase):
    """验证 mode='preflight'、软硬风险解耦、execute_if_clean 直通与 image_delivery 透传契约。"""

    setUp = DualArmHarnessBase.setUp

    def set_preview_stub(self):
        """绑定包含起/中/末采样点、关节行程与仿真耗时的真实结构化 preview_batch 桩。"""
        def preview_side_effect(action):
            samples = [
                dict(
                    native_step=0,
                    arms=dict(
                        left=dict(
                            q_rad=[0.0, 0.1, -0.1, 0.0, 0.0, 0.0],
                            link6_pose=dict(position_m=[0.18, 0.15, 0.92], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0]),
                            jaw_goal_normalized=1.0,
                            phase="initial",
                        ),
                        right=dict(
                            q_rad=[0.0, -0.1, 0.1, 0.0, 0.0, 0.0],
                            link6_pose=dict(position_m=[0.18, -0.15, 0.92], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0]),
                            jaw_goal_normalized=1.0,
                            phase="initial",
                        ),
                    ),
                ),
                dict(
                    native_step=4,
                    arms=dict(
                        left=dict(
                            q_rad=[0.05, 0.15, -0.12, 0.0, 0.0, 0.0],
                            link6_pose=dict(position_m=[0.14, 0.15, 0.85], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0]),
                            jaw_goal_normalized=1.0,
                            phase="move",
                        ),
                        right=dict(
                            q_rad=[0.12, -0.22, 0.18, 0.05, -0.04, 0.2],
                            link6_pose=dict(position_m=[0.12, -0.18, 0.84], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0]),
                            jaw_goal_normalized=1.0,
                            phase="move",
                        ),
                    ),
                ),
                dict(
                    native_step=8,
                    arms=dict(
                        left=dict(
                            q_rad=[0.05, 0.15, -0.12, 0.0, 0.0, 0.0],
                            link6_pose=dict(position_m=[0.14, 0.15, 0.85], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0]),
                            jaw_goal_normalized=1.0,
                            phase="hold",
                        ),
                        right=dict(
                            q_rad=[0.24, -0.35, 0.28, 0.10, -0.08, 0.42],
                            link6_pose=dict(position_m=[0.10, -0.20, 0.78], quaternion_xyzw=[0.0, 0.0, 0.0, 1.0]),
                            jaw_goal_normalized=0.0,
                            phase="gripper",
                        ),
                    ),
                ),
            ]
            full = dict(
                schema_version="candidate-trajectory-v1",
                source_kind="candidate_command_trajectory_not_measurement",
                frame_id="env_origin",
                tcp_config_id="arx-x5-link6-v1",
                joint_names={"left": [f"lj{i}" for i in range(6)], "right": [f"rj{i}" for i in range(6)]},
                samples=samples,
                geometry=dict(status="unknown"),
                inter_arm_proximity=dict(status="unknown"),
                execution_binding="exact_cached_rows; no replanning or resampling",
            )
            stage_costs = {
                st["id"]: dict(
                    arm=st.get("arm"),
                    kind=st.get("kind"),
                    start_native_step=0,
                    end_native_step=4,
                    native_steps=4,
                )
                for st in action.get("stages", [])
            }
            arm_costs = dict(
                left=dict(
                    motion_native_steps=4,
                    gripper_native_steps=0,
                    hold_native_steps=4,
                    joint_travel_rad=[0.05, 0.05, 0.02, 0.0, 0.0, 0.0],
                    sum_joint_travel_rad=0.12,
                ),
                right=dict(
                    motion_native_steps=6,
                    gripper_native_steps=2,
                    hold_native_steps=0,
                    joint_travel_rad=[0.24, 0.25, 0.18, 0.10, 0.08, 0.42],
                    sum_joint_travel_rad=1.27,
                ),
            )
            return dict(
                status="planned",
                plan_id="native-plan-" + str(self.robot.preview_batch.call_count),
                planned_native_steps=8,
                nominal_sim_time_s=0.4,
                arm_costs=arm_costs,
                stage_costs=stage_costs,
                trajectory_data=full,
                trajectory_evidence=dict(available=True),
                motion_safety_checks=[
                    dict(
                        stage_id=st["id"],
                        enabled=True,
                        status="pass",
                        start_state_source="initial_measurement",
                        arms={},
                    )
                    for st in action.get("stages", [])
                    if st.get("kind") != "observe"
                ],
                geometry_review=_stub_geometry_review(action),
            )

        def _stub_geometry_review(action):
            proxies = action.get("review_proxies") or {}
            obs_boxes = proxies.get("obstacles") or []
            contact_entity_id = proxies.get("contact_entity_id")
            risks = list(getattr(self, "_extra_geometry_risks", None) or [])
            for obs in obs_boxes:
                cx = 0.5 * (obs["min_m"][0] + obs["max_m"][0])
                cy = 0.5 * (obs["min_m"][1] + obs["max_m"][1])
                for st in action.get("stages", []):
                    if st.get("kind") != "move":
                        continue
                    gp = (st.get("action") or {}).get("input_grasp_pose") or {}
                    if st.get("arm") in gp:
                        gp = gp[st["arm"]]
                    pos = gp.get("position_m")
                    if isinstance(pos, list) and ((pos[0] - cx) ** 2 + (pos[1] - cy) ** 2) ** 0.5 < 0.08:
                        item = dict(
                            kind="scene_proxy_intersection",
                            moving_part=f"{st.get('arm')}/link6",
                            obstacle=obs["id"],
                            obstacle_id=obs["id"],
                            stage_id=st["id"],
                            native_step=1,
                            estimated_clearance_m=-0.012,
                            min_clearance_m=-0.012,
                        )
                        if obs["id"] == contact_entity_id:
                            item["contact_declaration"] = "caller_selected_contact_entity; overlap retained"
                        risks.append(item)
            return dict(status="risks_detected" if risks else "conditional", risks=risks)

        self.robot.preview_batch = Mock(side_effect=preview_side_effect)
        self.robot.execute = Mock(
            side_effect=lambda action, *args, **kwargs: dict(
                status="completed",
                reason=None,
                executed_native_steps=8,
                completed_native_steps=8,
                batch_status="completed",
                batch_cursor=8,
                completed_stage_ids=[s["id"] for s in action.get("stages", [])],
                pending_stage_ids=[],
                stage_results={},
                terminal_observation=self.robot.observe(),
            )
        )

    def test_preflight_skips_overlay_rendering_and_caches_executable_plan(self):
        """mode='preflight' 完全跳过图像渲染（0 张图片），透明返回数值摘要并缓存可直接 execute 的 plan_id。"""
        self.set_preview_stub()
        self.h.thread_id = "thread-preflight"
        obs_id = self.h.latest["observation_id"]
        table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[["o1", [0.10, -0.20, 0.780], 0.792, 0.765, 0.140, 0.025, 30.0, 88.0, "table", "ok"]],
        )
        self.h._obs3d_memo = (obs_id, {"object_table": (table, None, {})})

        args = dict(
            mode="preflight",
            observation_id=obs_id,
            stages=[
                dict(
                    id="grasp_o1",
                    kind="move",
                    arm="right",
                    grasp_position_m=[0.10, -0.20, 0.780],
                    angles_deg=[0.0, 0.0, -60.0],
                    gripper_goal_normalized=0.0,
                    position_tolerance_m=0.008,
                    orientation_tolerance_rad=0.12,
                    max_upward_dz_m=0.030,
                    stall_guard_preset="contact_stop",
                )
            ],
        )
        self.assertEqual(action_effect(args), "plan")
        self.assertEqual(action_effect(dict(args, auto_execute_if_clean=True)), "physical")

        with patch("ni_rua.runtime.actions.inspection._trajectory_image") as overlay_fn:
            reply = self.h.handle_tool(
                dict(threadId="thread-preflight", turnId="turn-1", callId="pf-1", tool="robot_action", arguments=args)
            )
            overlay_fn.assert_not_called()

        images = [item for item in reply["contentItems"] if item.get("type") == "inputImage"]
        self.assertEqual(images, [])
        payload = json.loads(reply["contentItems"][0]["text"])
        self.assertEqual(payload["status"], "planned")
        self.assertTrue(payload["plan_id"].startswith("p"))
        self.assertEqual(payload["trajectory_overlay"]["status"], "skipped")
        self.assertEqual(payload["blocking_risks"], [])
        self.assertEqual(payload["advisory_notes"], [])

        # 验证显式传入的容差与柔顺参数 100% 原样生效并透明回传
        tol = payload["effective_tolerances"]["grasp_o1"]
        self.assertEqual(tol["position_tolerance_m"], 0.008)
        self.assertEqual(tol["orientation_tolerance_rad"], 0.12)
        self.assertEqual(tol["max_upward_dz_m"], 0.03)
        self.assertEqual(tol["stall_guard_preset"], "contact_stop")
        self.assertEqual(tol["max_tracking_divergence_m"], 0.02)

        # 验证纯数值关节与末端轨迹摘要齐全
        traj_sum = payload["trajectory_summary"]
        self.assertEqual(traj_sum["planned_native_steps"], 8)
        self.assertEqual(traj_sum["nominal_sim_time_s"], 0.4)
        self.assertEqual(traj_sum["active_arms"], ["right"])
        self.assertEqual(traj_sum["arms"]["right"]["end_link6_position_m"], [0.1, -0.2, 0.78])
        self.assertEqual(traj_sum["arms"]["right"]["end_jaw_goal_normalized"], 0.0)
        self.assertEqual(traj_sum["arms"]["right"]["sum_joint_travel_rad"], 1.27)
        self.assertEqual(traj_sum["stages"]["grasp_o1"]["jaw_goal_normalized"], 0.0)
        self.assertEqual(traj_sum["stages"]["grasp_o1"]["native_steps"], 8)

        # 生成的 plan_id 可直接传给 mode='execute' 执行，且仍支持 mode='inspect' 读取缓存轨迹
        insp = wire_response(typed_dispatch(self.h, dict(mode="inspect", plan_id=payload["plan_id"]), {"tool_call_id": "insp-pf"}))
        self.assertGreaterEqual(len(insp["contentItems"]), 2)

        exec_reply = self.h.handle_tool(
            dict(
                threadId="thread-preflight",
                turnId="turn-2",
                callId="exec-pf-1",
                tool="robot_action",
                arguments=dict(mode="execute", plan_id=payload["plan_id"]),
            )
        )
        exec_payload = json.loads(exec_reply["contentItems"][0]["text"])
        self.assertEqual(exec_payload["receipt"]["status"], "completed")
        self.assertEqual(self.robot.execute.call_count, 1)

    def test_program_run_loops_preflight_candidates_and_executes_winner(self):
        """在 program_run 沙箱中毫秒级循环调用 mode='preflight' 筛选候选位姿并直通 execute。"""
        self.set_preview_stub()
        obs_id = self.h.latest["observation_id"]
        # o1 旁边紧贴邻物 o2：yaw=30 时张爪与 o2 发生硬干涉碰撞，yaw=-60 时无碰撞
        crowded_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.140, 0.026, 30.0, 88.0, "table", "ok"],
                ["o2", [0.10 + 0.035 * 0.5, -0.20 - 0.035 * 0.866, 0.780], 0.815, 0.765, 0.032, 0.032, 0.0, None, "table", "ok"],
            ],
        )
        # 精确放置邻物到 yaw=-60 闭合方向手指处以制造碰撞对比：
        # angles_deg=[0,0,-60] 时闭合轴与 o1(30°) 垂直；把 o2 放在 [0.120, -0.235, 0.780] 让其中一组候选触发碰撞
        crowded_table["rows"][1][1] = [0.118, -0.231, 0.780]
        self.h._obs3d_memo = (obs_id, {"object_table": (crowded_table, None, {})})

        runner = ProgramRunner(
            __import__("dataclasses").replace(
                self.h.program_services(),
                sandbox_enabled=expect(program_isolation, "program_run runs under the Landlock + seccomp sandbox"),
            )
        )
        self.h.programs = runner
        self.addCleanup(runner.close)

        # 1) allow_direct_actions=False 时循环 preflight 属于 'plan' 效果，不挂起等待审批、不产生图片
        readonly_source = (
            "from ni_rua_client import NiRuaClient\n"
            "client = NiRuaClient()\n"
            "plans = []\n"
            "for idx, z in enumerate([0.780, 0.785, 0.790]):\n"
            f"    res = client.call('robot_action', {{'mode': 'preflight', 'observation_id': {obs_id!r}, "
            "'stages': [{'id': f'cand_{idx}', 'kind': 'move', 'arm': 'right', "
            "'grasp_position_m': [0.10, -0.20, z], 'angles_deg': [0.0, 0.0, -60.0]}]})\n"
            "    assert res['kind'] == 'plan' and res['success'] is True\n"
            "    assert 'effective_tolerances' in res['data'] and 'trajectory_summary' in res['data']\n"
            "    plans.append(res['data']['plan_id'])\n"
            "print('SCREENED:', len(plans))\n"
        )
        with patch("ni_rua.runtime.actions.inspection._trajectory_image") as overlay_fn:
            job_ro = runner.run(
                dict(source=readonly_source, timeout_s=60, allow_direct_actions=False),
                dict(thread_id="thread", turn_id="turn-ro", tool_call_id="prog-ro"),
            )
            overlay_fn.assert_not_called()

        self.assertEqual(job_ro["status"], "succeeded", job_ro.get("stderr_text"))
        self.assertIn("SCREENED: 3", job_ro.get("stdout_text", ""))
        self.assertEqual(job_ro.get("visual_feedback", []), [])
        self.assertEqual(self.robot.execute.call_count, 0)

        # 2) allow_direct_actions=True 时在沙箱内用 preflight + 显式 obstacle_object_ids=['o2'] 排除碰撞候选后直接 execute 胜出 plan_id
        # 构造两个目标：第一个与显式障碍 o2 重叠触发 geometry_review_risk 硬阻断（同时 advisory_notes 客观记录 warn_finger_collides_neighbor），第二个在远端 o3 无风险
        crowded_table["rows"].append(
            ["o3", [-0.15, 0.18, 0.780], 0.795, 0.765, 0.050, 0.030, 0.0, 88.0, "table", "ok"]
        )
        direct_source = (
            "from ni_rua_client import NiRuaClient\n"
            "client = NiRuaClient()\n"
            "candidates = [\n"
            "    ([0.10, -0.20, 0.780], [0.0, 0.0, -60.0]),\n"
            "    ([-0.15, 0.18, 0.780], [0.0, 0.0, 90.0]),\n"
            "]\n"
            "chosen = None\n"
            "rejected = 0\n"
            "for idx, (pos, ang) in enumerate(candidates):\n"
            f"    pf = client.call('robot_action', {{'mode': 'preflight', 'observation_id': {obs_id!r}, "
            "'obstacle_object_ids': ['o2'], 'obstacle_policy': 'stateless', "
            "'stages': [{'id': f'g_{idx}', 'kind': 'move', 'arm': 'right', "
            "'grasp_position_m': pos, 'angles_deg': ang, 'gripper_goal_normalized': 0.0}]})\n"
            "    if pf['data']['blocking_risks']:\n"
            "        rejected += 1\n"
            "        continue\n"
            "    chosen = pf['data']['plan_id']\n"
            "    break\n"
            "assert rejected == 1 and chosen is not None\n"
            "act = client.call('robot_action', {'mode': 'execute', 'plan_id': chosen})\n"
            "assert act['kind'] == 'action' and act['data']['receipt']['status'] == 'completed'\n"
            "print('EXECUTED_PLAN:', chosen)\n"
        )
        job_exec = runner.run(
            dict(source=direct_source, timeout_s=60, allow_direct_actions=True),
            dict(thread_id="thread", turn_id="turn-exec", tool_call_id="prog-exec"),
        )
        self.assertEqual(job_exec["status"], "succeeded", job_exec.get("stderr_text"))
        self.assertIn("EXECUTED_PLAN:", job_exec.get("stdout_text", ""))
        self.assertEqual(self.robot.execute.call_count, 1)

    def test_soft_advisory_notes_do_not_block_execute_if_clean_single_turn(self):
        """软几何提示（斜抓夹角偏大、薄物体 contact_stop 指尖低于支撑面、门禁偏宽）透明展示但不阻断 execute_if_clean。"""
        self.set_preview_stub()
        obs_id = self.h.latest["observation_id"]

        # 场景 A：细长物体（60mm×22mm，在 88.8mm 开度内无下探撞指硬碰撞）斜抓（夹角 30° < 45°）+ min_aperture_m 偏大 -> 仅产生软提示，单回合直通执行
        elongated_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[["o1", [0.10, -0.20, 0.780], 0.792, 0.765, 0.060, 0.022, 30.0, 88.0, "table", "ok"]],
        )
        self.h._obs3d_memo = (obs_id, {"object_table": (elongated_table, None, {})})
        oblique_args = dict(
            mode="execute_if_clean",
            observation_id=obs_id,
            plan_update=dict(
                plan="Single-turn oblique grasp with soft advisory.",
                ledger=[dict(id="L1", expected="Tube grasped cleanly", status="pending", basis="obs_1", next_check="grip_facts")],
            ),
            stages=[
                dict(
                    id="oblique_grasp",
                    kind="move",
                    arm="right",
                    grasp_position_m=[0.10, -0.20, 0.780],
                    angles_deg=[0.0, 0.0, 0.0],
                    gripper_goal_normalized=0.0,
                    min_aperture_m=0.035,
                )
            ],
        )
        with patch("ni_rua.runtime.actions.inspection._trajectory_image") as overlay_fn:
            res_oblique = typed_dispatch(self.h, oblique_args, {"tool_call_id": "exec-clean-oblique"})
            overlay_fn.assert_not_called()

        self.assertEqual(res_oblique.kind, "action")
        self.assertEqual(self.robot.execute.call_count, 1)
        self.assertIsNotNone(self.h.task_planning.current(self.h.identity))
        advisory_codes = {n["code"] for n in res_oblique.data.get("advisory_notes", [])}
        self.assertIn("warn_along_long_axis", advisory_codes)
        self.assertIn("warn_guard_exceeds_object_width", advisory_codes)

        # 场景 B：2mm 超薄物体贴桌抓取 + stall_guard_preset='contact_stop'（理论指尖低于桌面约 6.5mm）-> 直接放行物理执行
        obs_id_2 = self.h.latest["observation_id"]
        thin_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[["o_card", [0.12, -0.18, 0.766], 0.767, 0.765, 0.080, 0.050, 0.0, 89.0, "table", "ok"]],
        )
        self.h._obs3d_memo = (obs_id_2, {"object_table": (thin_table, None, {})})
        thin_args = dict(
            mode="execute_if_clean",
            observation_id=obs_id_2,
            stages=[
                dict(
                    id="thin_grasp",
                    kind="move",
                    arm="right",
                    grasp_position_m=[0.12, -0.18, 0.766],
                    angles_deg=[0.0, 0.0, 90.0],
                    gripper_goal_normalized=0.0,
                    stall_guard_preset="contact_stop",
                    grip_guard_preset="thin_object",
                )
            ],
        )
        res_thin = typed_dispatch(self.h, thin_args, {"tool_call_id": "exec-clean-thin"})
        self.assertEqual(res_thin.kind, "action")
        self.assertEqual(self.robot.execute.call_count, 2)
        thin_codes = {n["code"] for n in res_thin.data.get("advisory_notes", [])}
        self.assertIn("warn_fingertip_below_support", thin_codes)

    def test_hard_blocking_risks_intercept_execute_if_clean_with_zero_images_by_default(self):
        """环境猜测风险统一降为中性 advisory_notes 直通执行；仅显式 obstacle_object_ids 或本体 motion_safety_reject 硬阻断 execute_if_clean。"""
        self.set_preview_stub()
        obs_id = self.h.latest["observation_id"]

        # 1) 未显式传 obstacle_object_ids 时，邻物干涉、低空横移与已闭爪下探全部记入中性 advisory_notes，直通执行不误拦
        neighbor_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.140, 0.026, 30.0, 88.0, "table", "ok"],
                ["o2", [0.118, -0.231, 0.780], 0.815, 0.765, 0.032, 0.032, 0.0, None, "table", "ok"],
            ],
        )
        self.h._obs3d_memo = (obs_id, {"object_table": (neighbor_table, None, {})})
        collide_args = dict(
            mode="execute_if_clean",
            observation_id=obs_id,
            stages=[
                dict(
                    id="grasp_collide",
                    kind="move",
                    arm="right",
                    grasp_position_m=[0.10, -0.20, 0.780],
                    angles_deg=[0.0, 0.0, -60.0],
                    gripper_goal_normalized=0.0,
                )
            ],
        )
        res_unblocked_neighbor = typed_dispatch(self.h, collide_args, {"tool_call_id": "pass-neighbor"})
        self.assertEqual(res_unblocked_neighbor.kind, "action")
        self.assertEqual(res_unblocked_neighbor.data.get("blocking_risks", []), [])
        self.assertIn(
            "warn_finger_collides_neighbor",
            {n["code"] for n in res_unblocked_neighbor.data.get("advisory_notes", [])},
        )
        self.assertEqual(self.robot.execute.call_count, 1)

        # 低空搬运与闭爪下探同样进入中性 advisory_notes，不含指令式说教，单回合直通执行
        wall_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[
                ["o_pick", [0.10, -0.28, 0.780], 0.795, 0.765, 0.040, 0.030, 0.0, 88.0, "table", "ok"],
                ["o_wall", [0.10, -0.14, 0.830], 0.890, 0.765, 0.120, 0.040, 0.0, 88.0, "table", "ok"],
            ],
        )
        obs_id_2 = self.h.latest["observation_id"]
        self.h._obs3d_memo = (obs_id_2, {"object_table": (wall_table, None, {})})
        transport_args = dict(
            mode="execute_if_clean",
            observation_id=obs_id_2,
            stages=[
                dict(
                    id="pick",
                    kind="move",
                    arm="right",
                    grasp_position_m=[0.10, -0.28, 0.780],
                    angles_deg=[0.0, 0.0, 90.0],
                    gripper_goal_normalized=0.0,
                ),
                dict(
                    id="low_carry_pass",
                    kind="move",
                    arm="right",
                    grasp_position_m=[0.10, 0.0, 0.795],
                    angles_deg=[0.0, 0.0, 90.0],
                ),
            ],
        )
        res_transport = typed_dispatch(self.h, transport_args, {"tool_call_id": "pass-transport"})
        self.assertEqual(res_transport.kind, "action")
        note_codes = {n["code"] for n in res_transport.data.get("advisory_notes", [])}
        self.assertIn("warn_transport_collision", note_codes)
        for note in res_transport.data.get("advisory_notes", []):
            self.assertNotIn("raise transit z", note["message"].lower())
        self.assertEqual(self.robot.execute.call_count, 2)

        obs_id_closed = self.h.latest["observation_id"]
        clean_table = dict(
            columns=COLUMNS,
            table_z=0.765,
            rows=[["o1", [0.10, -0.20, 0.780], 0.795, 0.765, 0.050, 0.030, 0.0, 88.0, "table", "ok"]],
        )
        self.h._obs3d_memo = (obs_id_closed, {"object_table": (clean_table, None, {})})
        self.h.latest["grip_facts"] = {"right": {"measured_aperture_m": 0.0, "measured_jaw_after": 0.0}}
        closed_grasp_args = dict(
            mode="execute_if_clean",
            observation_id=obs_id_closed,
            stages=[
                dict(
                    id="grasp_while_closed",
                    kind="move",
                    arm="right",
                    grasp_position_m=[0.10, -0.20, 0.780],
                    angles_deg=[0.0, 0.0, 90.0],
                    gripper_goal_normalized=0.0,
                )
            ],
        )
        res_closed = typed_dispatch(self.h, closed_grasp_args, {"tool_call_id": "pass-closed"})
        self.assertEqual(res_closed.kind, "action")
        closed_codes = {n["code"] for n in res_closed.data.get("advisory_notes", [])}
        self.assertTrue(closed_codes & {"warn_closed_jaw_descent", "closed_jaw_before_grasp"})
        for note in res_closed.data.get("advisory_notes", []):
            self.assertNotIn("open the jaw", note["message"].lower())
        self.assertEqual(self.robot.execute.call_count, 3)

        # 2) 模型显式传入 obstacle_object_ids=['o2'] 时触发 geometry_review_risk 硬阻断 -> 退回 plan_id 且默认 0 张图
        obs_id_3 = self.h.latest["observation_id"]
        self.h._obs3d_memo = (obs_id_3, {"object_table": (neighbor_table, None, {})})
        explicit_obs_args = dict(
            collide_args,
            observation_id=obs_id_3,
            obstacle_object_ids=["o2"],
            obstacle_policy="stateless",
        )
        with patch(
            "ni_rua.runtime.actions.inspection._trajectory_image",
            return_value=(dict(status="available"), [dict(type="inputImage", imageUrl="data:image/jpeg;base64,AA==")]),
        ) as overlay_fn:
            blocked_default = wire_response(typed_dispatch(self.h, explicit_obs_args, {"tool_call_id": "blk-default"}))
            overlay_fn.assert_not_called()
            self.assertEqual([c for c in blocked_default["contentItems"] if c["type"] == "inputImage"], [])
            blocked_summary = json.loads(blocked_default["contentItems"][0]["text"])
            self.assertTrue(blocked_summary["downgraded_from_execute_if_clean"])
            self.assertIn("geometry_review_risks", blocked_summary["downgrade_reasons"])
            self.assertEqual(blocked_summary["blocking_risks"][0]["code"], "geometry_review_risk")
            self.assertEqual(self.robot.execute.call_count, 3)

            # 显式传 image_delivery='all' 时按需渲染并返回轨迹叠加图
            blocked_with_img = wire_response(
                typed_dispatch(self.h, dict(explicit_obs_args, image_delivery="all"), {"tool_call_id": "blk-all"})
            )
            overlay_fn.assert_called_once()
            self.assertEqual(len([c for c in blocked_with_img["contentItems"] if c["type"] == "inputImage"]), 1)

        # 3) 机器人本体 motion_safety_checks 拒绝（如关节行程超限或自碰）时触发硬阻断，绝不执行
        orig_preview = self.robot.preview_batch.side_effect

        def reject_motion_safety(action):
            plan = orig_preview(action)
            plan["motion_safety_checks"] = [
                dict(
                    stage_id=action["stages"][0]["id"],
                    enabled=True,
                    status="reject",
                    reason="joint_travel_limit_exceeded",
                    start_state_source="initial_measurement",
                    arms={},
                )
            ]
            return plan

        self.robot.preview_batch.side_effect = reject_motion_safety
        res_safety_blk = typed_dispatch(
            self.h,
            dict(collide_args, observation_id=obs_id_3),
            {"tool_call_id": "blk-motion-safety"},
        )
        self.assertEqual(res_safety_blk.kind, "plan")
        self.assertTrue(res_safety_blk.data["downgraded_from_execute_if_clean"])
        self.assertIn("motion_safety_reject", res_safety_blk.data["downgrade_reasons"])
        self.assertEqual(res_safety_blk.data["blocking_risks"][0]["code"], "motion_safety_reject")
        self.assertEqual(self.robot.execute.call_count, 3)

    def test_image_delivery_schema_validation_and_passthrough_without_silent_override(self):
        """验证 image_delivery 枚举契约校验、向下游 BatchAction 精确透传且不静默覆盖默认策略。"""
        self.set_preview_stub()
        registry = create_registry()
        obs_id = self.h.latest["observation_id"]
        base_stage = [
            dict(
                id="move_1",
                kind="move",
                arm="left",
                grasp_position_m=[0.20, 0.15, 0.82],
                angles_deg=[0.0, 0.0, 0.0],
            )
        ]

        # 1) 全部 5 种合法 image_delivery 通过 actor 与 program 双角色校验，非法枚举报错
        for delivery in IMAGE_DELIVERY_MODES:
            for role in ("actor", "program"):
                registry.validate(
                    "robot_action",
                    dict(mode="preflight", image_delivery=delivery, observation_id=obs_id, stages=base_stage),
                    role=role,
                )
        for role in ("actor", "program"):
            with self.assertRaises(ValueError):
                registry.validate(
                    "robot_action",
                    dict(mode="preflight", image_delivery="invalid_policy", observation_id=obs_id, stages=base_stage),
                    role=role,
                )

        compiled = compile_request(
            self.h, dict(observation_id=obs_id, image_delivery="on_anomaly", stages=base_stage)
        )
        self.assertEqual(compiled["image_delivery"], "on_anomaly")
        bad_action = deepcopy(compiled)
        bad_action["image_delivery"] = "bogus"
        with self.assertRaisesRegex(ValueError, "invalid_image_delivery"):
            validate_batch_action(bad_action, self.h.capabilities)

        # 2) preflight 未显式传 image_delivery 时，后续 execute 的 action_request 不被静默塞入 image_delivery='none'
        pf_res = typed_dispatch(
            self.h,
            dict(mode="preflight", observation_id=obs_id, stages=base_stage),
            {"tool_call_id": "pf-no-img-override"},
        )
        exec_default = typed_dispatch(
            self.h,
            dict(mode="execute", plan_id=pf_res.data["plan_id"], action_id="act-default-delivery"),
            {"tool_call_id": "exec-default-delivery"},
        )
        self.assertEqual(exec_default.kind, "action")
        stored_default_req = self.store.read("action_request", "act-default-delivery")
        self.assertNotIn("image_delivery", stored_default_req)

        # 3) execute_if_clean 或 execute 显式传入 image_delivery 时，精确透传进 action_request 与 observation_content
        captured_feedback_actions = []
        orig_obs_content = self.h.observation_content

        def spy_obs_content(bundle, correlation, *, feedback_action=None, **kwargs):
            captured_feedback_actions.append(deepcopy(feedback_action))
            return orig_obs_content(bundle, correlation, feedback_action=feedback_action, **kwargs)

        with patch.object(self.h, "observation_content", side_effect=spy_obs_content):
            clean_res = typed_dispatch(
                self.h,
                dict(
                    mode="execute_if_clean",
                    image_delivery="focus_roi",
                    action_id="act-focus-roi",
                    observation_id=self.h.latest["observation_id"],
                    stages=base_stage,
                ),
                {"tool_call_id": "exec-focus-roi"},
            )
        self.assertEqual(clean_res.kind, "action")
        self.assertEqual(self.store.read("action_request", "act-focus-roi")["image_delivery"], "focus_roi")
        self.assertEqual(captured_feedback_actions[-1]["image_delivery"], "focus_roi")

        # 4) mode='preview' 配合 image_delivery='none' 跳过轨迹渲染，默认 mode='preview' 保持渲染兼容
        with patch(
            "ni_rua.runtime.actions.inspection._trajectory_image",
            return_value=(dict(status="available"), [dict(type="inputImage", imageUrl="data:image/jpeg;base64,AA==")]),
        ) as overlay_fn:
            pv_none = wire_response(
                typed_dispatch(
                    self.h,
                    dict(mode="preview", image_delivery="none", observation_id=self.h.latest["observation_id"], stages=base_stage),
                    {"tool_call_id": "pv-none"},
                )
            )
            overlay_fn.assert_not_called()
            self.assertEqual([c for c in pv_none["contentItems"] if c["type"] == "inputImage"], [])

            pv_default = wire_response(
                typed_dispatch(
                    self.h,
                    dict(mode="preview", observation_id=self.h.latest["observation_id"], stages=base_stage),
                    {"tool_call_id": "pv-default"},
                )
            )
            overlay_fn.assert_called_once()
            self.assertEqual(len([c for c in pv_default["contentItems"] if c["type"] == "inputImage"]), 1)

    def test_actor_instructions_promote_single_turn_execute_if_clean_and_zero_image_preflight(self):
        """系统指令与 OPERATION_NOTES 对齐 execute_if_clean 单回合直通、黄金基线视觉闭环与 focus_3d 1:1 像素换算指引。"""
        rendered = instructions(CALIBRATED_OBSERVATION_INSTRUCTIONS, "Autonomous run.")
        for expected in (
            "mode='execute_if_clean'",
            "plan_update=",
            "blocking_risks",
            "advisory_notes",
            "mode='preflight'",
            "RGB images",
            "object_state_delta",
            "robot_observe(focus_3d=",
            "history_view_observation(..., focus_3d={...})",
            "full_uv = [crop_xywh_px[0] + local_u, crop_xywh_px[1] + local_v]",
        ):
            self.assertIn(expected, INSTRUCTIONS)
            self.assertIn(expected, rendered)

        for forbidden in (
            "0-image closed loop",
            "s3.virtual_field",
            "do not reflexively",
            "Do not pass image_delivery='head_only'",
        ):
            self.assertNotIn(forbidden, INSTRUCTIONS)
            self.assertNotIn(forbidden, OPERATION_NOTES)
            self.assertNotIn(forbidden, ROBOT_ACTION_TOOL["description"])

        # 验证 OPERATION_NOTES 恢复高信噪比视觉与 3D 联合核查，并保留 focus_3d 像素换算公式
        for expected_note in (
            "RGB visual feedback",
            "object_state_delta",
            "checkpoint_after_gripper=true",
            "focus_3d",
            "history_view_observation",
            "grip_facts (measured_aperture_m)",
            "full_uv = [crop_xywh_px[0] + local_u, crop_xywh_px[1] + local_v]",
        ):
            self.assertIn(expected_note, OPERATION_NOTES)
        self.assertNotIn("check both RGB and grip_facts", OPERATION_NOTES)

        # 确保沙箱路径替换与禁用记忆替换锚点完整保留
        no_sandbox = instructions(
            CALIBRATED_OBSERVATION_INSTRUCTIONS,
            "Autonomous run.",
            source_root="/custom/ni-rua",
            operation_memory=False,
        )
        self.assertIn("/custom/ni-rua", no_sandbox)
        self.assertNotIn("operation_memory separates current-task records", no_sandbox)

    def test_inter_arm_and_declared_contact_proxies_are_advisory_without_blocking_execute_if_clean(self):
        """粗膨胀球 inter_arm_proxy_intersection 与显式声明接触的代理重叠归入 advisory_notes，不阻断 execute_if_clean；未声明显式障碍继续阻断。"""
        from ni_rua.runtime.actions.unified_actions import _is_blocking_geometry_risk

        self.assertTrue(_is_blocking_geometry_risk({"kind": "scene_proxy_intersection", "obstacle": "box1"}))
        self.assertFalse(
            _is_blocking_geometry_risk(
                {
                    "kind": "scene_proxy_intersection",
                    "obstacle": "box1",
                    "contact_declaration": "caller_selected_contact_entity; overlap retained",
                }
            )
        )
        self.assertFalse(_is_blocking_geometry_risk({"kind": "inter_arm_proxy_intersection", "obstacle": "right/link5"}))
        self.assertFalse(_is_blocking_geometry_risk("not_a_dict"))

        self.set_preview_stub()
        self.h.thread_id = "thread-proxy-advisory"
        obs_id = self.h.latest["observation_id"]

        # 1) 双臂膨胀球重叠 inter_arm_proxy_intersection -> advisory_notes，execute_if_clean 单回合直通执行
        self._extra_geometry_risks = [
            dict(
                kind="inter_arm_proxy_intersection",
                moving_part="left/link4",
                obstacle="right/link5",
                native_step=4,
                estimated_clearance_m=-0.006,
            )
        ]
        try:
            pf_resp = typed_dispatch(
                self.h,
                dict(
                    mode="preflight",
                    observation_id=obs_id,
                    stages=[
                        dict(
                            id="coop_left",
                            kind="move",
                            arm="left",
                            grasp_position_m=[-0.04, 0.12, 0.82],
                            angles_deg=[0.0, 0.0, 0.0],
                        ),
                        dict(
                            id="coop_right",
                            kind="move",
                            arm="right",
                            grasp_position_m=[0.04, -0.12, 0.82],
                            angles_deg=[0.0, 0.0, 0.0],
                        ),
                    ],
                ),
                {"tool_call_id": "pf-inter-arm"},
            )
            self.assertEqual(pf_resp.data["blocking_risks"], [])
            self.assertTrue(any(n.get("code") == "inter_arm_proxy_advisory" for n in pf_resp.data["advisory_notes"]))

            exec_resp = typed_dispatch(
                self.h,
                dict(
                    mode="execute_if_clean",
                    observation_id=obs_id,
                    stages=[
                        dict(
                            id="coop_left",
                            kind="move",
                            arm="left",
                            grasp_position_m=[-0.04, 0.12, 0.82],
                            angles_deg=[0.0, 0.0, 0.0],
                        ),
                        dict(
                            id="coop_right",
                            kind="move",
                            arm="right",
                            grasp_position_m=[0.04, -0.12, 0.82],
                            angles_deg=[0.0, 0.0, 0.0],
                        ),
                    ],
                ),
                {"tool_call_id": "eic-inter-arm"},
            )
            self.assertEqual(exec_resp.kind, "action")
            self.assertEqual(exec_resp.data["receipt"]["status"], "completed")
            self.assertEqual(exec_resp.data.get("blocking_risks", []), [])
            self.assertTrue(any(n.get("code") == "inter_arm_proxy_advisory" for n in exec_resp.data.get("advisory_notes", [])))
            self.assertEqual(self.robot.execute.call_count, 1)
        finally:
            self._extra_geometry_risks = []

        # 2) 已显式声明 contact_entity_id 的障碍接触 -> declared_contact_proxy_advisory，不阻断 execute_if_clean
        self.robot.execute.reset_mock()
        obs_id2 = self.h.latest["observation_id"]
        declared_resp = typed_dispatch(
            self.h,
            dict(
                mode="execute_if_clean",
                observation_id=obs_id2,
                contact_entity_id="rack_hook",
                obstacles=[dict(id="rack_hook", min_m=[0.08, -0.22, 0.76], max_m=[0.12, -0.18, 0.86], margin_m=0.0, evidence="visible rack hook")],
                stages=[
                    dict(
                        id="hang_mug",
                        kind="move",
                        arm="right",
                        grasp_position_m=[0.10, -0.20, 0.80],
                        angles_deg=[0.0, 0.0, 0.0],
                        tolerance_preset="compliant",
                    )
                ],
            ),
            {"tool_call_id": "eic-declared-contact"},
        )
        self.assertEqual(declared_resp.kind, "action")
        self.assertEqual(declared_resp.data["receipt"]["status"], "completed")
        self.assertEqual(declared_resp.data.get("blocking_risks", []), [])
        self.assertEqual(self.robot.execute.call_count, 1)
        self.assertTrue(
            any(
                n.get("code") == "declared_contact_proxy_advisory"
                for n in declared_resp.data.get("advisory_notes", [])
            )
        )

        # 3) 未声明接触的显式障碍碰撞 -> geometry_review_risk 硬阻断 execute_if_clean
        self.robot.execute.reset_mock()
        obs_id3 = self.h.latest["observation_id"]
        blocked_resp = typed_dispatch(
            self.h,
            dict(
                mode="execute_if_clean",
                observation_id=obs_id3,
                obstacles=[dict(id="wall_obs", min_m=[0.08, -0.22, 0.76], max_m=[0.12, -0.18, 0.86], margin_m=0.0, evidence="visible wall obstacle")],
                stages=[
                    dict(
                        id="hit_wall",
                        kind="move",
                        arm="right",
                        grasp_position_m=[0.10, -0.20, 0.80],
                        angles_deg=[0.0, 0.0, 0.0],
                    )
                ],
            ),
            {"tool_call_id": "eic-undeclared-obs"},
        )
        self.assertEqual(blocked_resp.kind, "plan")
        self.assertTrue(blocked_resp.data["downgraded_from_execute_if_clean"])
        self.assertIn("geometry_review_risks", blocked_resp.data["downgrade_reasons"])
        self.assertTrue(any(r.get("code") == "geometry_review_risk" for r in blocked_resp.data["blocking_risks"]))
        self.assertEqual(self.robot.execute.call_count, 0)

    def _load_so101_profile(self, *, empty_close_aperture_m=None):
        """加载仓库示例配置构建的真实 SO-101 EmbodimentProfile。"""
        from pathlib import Path
        from ni_rua.robot.backends.so101 import deployment

        repo_root = Path(__file__).resolve().parents[3]
        doc = json.loads((repo_root / "configs" / "so101.example.json").read_text())
        if empty_close_aperture_m is not None:
            doc["gripper"]["empty_close_aperture_m"] = empty_close_aperture_m
        return deployment.embodiment_profile(deployment.validate_deployment(doc, base=repo_root / "configs"))

    def test_carry_slip_guard_per_key_inheritance_and_repair_reexecuted_stages(self):
        """验证续跑/修补/阶段覆盖四处防滑脱三键逐键继承、显式 False 清从属键、缺阈值诚实报错与重跑阶段标注。"""
        from types import SimpleNamespace
        import ni_rua.runtime.actions.batch_repair as batch_repair_mod
        from ni_rua.robot.motion.batch_continuation import (
            _apply_compiled_stage_override,
            apply_stage_override,
            resolve_backend_continuation,
        )
        from ni_rua.robot.motion.grip_guard import validate_carry_slip_syntax
        from ni_rua.runtime.actions.batch_repair import build_repaired_args
        from ni_rua.runtime.actions.plan_continuation import resolve_continuation_args

        self.assertFalse(hasattr(batch_repair_mod, "_INHERITED_BATCH_FIELDS"))
        self.set_preview_stub()
        obs_id = self.h.latest["observation_id"]

        # 1) 创建带批次级防滑脱三键的源计划
        src_resp = typed_dispatch(
            self.h,
            dict(
                mode="preflight",
                observation_id=obs_id,
                grip_guard_preset="rigid_object",
                carry_slip_guard=True,
                carry_slip_scope="until_release",
                slip_consecutive_steps=3,
                stages=[
                    dict(
                        id="grasp",
                        kind="move",
                        arm="left",
                        grasp_position_m=[0.20, 0.15, 0.80],
                        angles_deg=[0.0, 0.0, 0.0],
                        gripper_goal_normalized=0.0,
                        carry_slip_guard=True,
                        carry_slip_scope="until_release",
                        slip_consecutive_steps=2,
                    ),
                    dict(
                        id="lift",
                        kind="move",
                        arm="left",
                        grasp_position_m=[0.20, 0.15, 0.90],
                        angles_deg=[0.0, 0.0, 0.0],
                    ),
                    dict(
                        id="place",
                        kind="move",
                        arm="left",
                        grasp_position_m=[0.14, 0.15, 0.84],
                        angles_deg=[0.0, 0.0, 0.0],
                    ),
                ],
            ),
            {"tool_call_id": "cs-src-1"},
        )
        src_id = src_resp.data["plan_id"]

        # (a) plan_continuation：缺省继承全部三键；单改 slip_consecutive_steps=5 不丢 carry_slip_guard=True；显式 False 清从属键
        cont_full, _ = resolve_continuation_args(
            self.h,
            dict(observation_id=obs_id, source_plan_id=src_id, start_from_stage_id="grasp"),
        )
        self.assertIs(cont_full["carry_slip_guard"], True)
        self.assertEqual(cont_full["carry_slip_scope"], "until_release")
        self.assertEqual(cont_full["slip_consecutive_steps"], 3)

        cont_step_only, _ = resolve_continuation_args(
            self.h,
            dict(
                observation_id=obs_id,
                source_plan_id=src_id,
                start_from_stage_id="grasp",
                slip_consecutive_steps=5,
            ),
        )
        self.assertIs(cont_step_only["carry_slip_guard"], True)
        self.assertEqual(cont_step_only["carry_slip_scope"], "until_release")
        self.assertEqual(cont_step_only["slip_consecutive_steps"], 5)
        compiled_step_only = compile_request(self.h, cont_step_only)
        self.assertEqual(compiled_step_only["slip_consecutive_steps"], 5)

        cont_disabled, _ = resolve_continuation_args(
            self.h,
            dict(
                observation_id=obs_id,
                source_plan_id=src_id,
                start_from_stage_id="lift",
                carry_slip_guard=False,
            ),
        )
        self.assertIs(cont_disabled["carry_slip_guard"], False)
        self.assertNotIn("carry_slip_scope", cont_disabled)
        self.assertNotIn("slip_consecutive_steps", cont_disabled)
        compile_request(self.h, cont_disabled)

        # 从 lift 续跑（无闭爪阶段）时计划摘要诚实反映 carry_slip_guard_armed=False
        cont_lift_resp = typed_dispatch(
            self.h,
            dict(
                mode="preflight",
                observation_id=obs_id,
                source_plan_id=src_id,
                start_from_stage_id="lift",
            ),
            {"tool_call_id": "cs-cont-lift"},
        )
        self.assertEqual(cont_lift_resp.data["status"], "planned")
        self.assertIs(cont_lift_resp.data["carry_slip_guard_armed"], False)
        self.assertIs(cont_lift_resp.data["continuation_of"]["carry_slip_guard_armed"], False)
        self.assertTrue(
            any(
                n.get("code") == "carry_slip_guard_not_armed_without_closing_jaw"
                for n in cont_lift_resp.data.get("advisory_notes", [])
            )
        )

        # (b) 源计划仅在已完成的闭爪阶段设了 min_aperture_m，从 lift 续跑时诚实报 invalid_carry_slip_requires_min_aperture
        stage_only_min_resp = typed_dispatch(
            self.h,
            dict(
                mode="preflight",
                observation_id=obs_id,
                carry_slip_guard=True,
                stages=[
                    dict(
                        id="close",
                        kind="gripper",
                        arm="left",
                        goal_normalized=0.0,
                        min_aperture_m=0.015,
                    ),
                    dict(
                        id="lift_after",
                        kind="move",
                        arm="left",
                        grasp_position_m=[0.20, 0.15, 0.90],
                        angles_deg=[0.0, 0.0, 0.0],
                    ),
                ],
            ),
            {"tool_call_id": "cs-stage-min-only"},
        )
        with self.assertRaisesRegex(ValueError, "invalid_carry_slip_requires_min_aperture"):
            typed_dispatch(
                self.h,
                dict(
                    mode="preflight",
                    observation_id=obs_id,
                    source_plan_id=stage_only_min_resp.data["plan_id"],
                    start_from_stage_id="lift_after",
                ),
                {"tool_call_id": "cs-missing-min-err"},
            )

        # (c) batch_repair：/motion 完成后残余夹爪白名单保留三键、逐键继承、显式 False 清从属键、重跑已完成阶段标注
        rec = deepcopy(self.h.unified_action_plans[src_id])
        rec["raw_stages"][0]["grip_check_preset"] = "legacy_dead"
        rec["raw_stages"][0]["min_grip_m"] = 0.01
        rec["summary"]["completed_stage_ids"] = ["grasp/motion"]
        rec["summary"]["failed_stage_id"] = "lift"
        self.h.unified_action_plans["plan-cs-repair"] = rec

        repaired_jaw = build_repaired_args(
            self.h,
            dict(observation_id=obs_id, plan_id="plan-cs-repair", slip_consecutive_steps=4),
            lambda _h, pid: self.h.unified_action_plans[pid],
        )
        self.assertIs(repaired_jaw["carry_slip_guard"], True)
        self.assertEqual(repaired_jaw["carry_slip_scope"], "until_release")
        self.assertEqual(repaired_jaw["slip_consecutive_steps"], 4)
        split_gripper = repaired_jaw["stages"][0]
        self.assertEqual(split_gripper["kind"], "gripper")
        self.assertIs(split_gripper["carry_slip_guard"], True)
        self.assertEqual(split_gripper["carry_slip_scope"], "until_release")
        self.assertEqual(split_gripper["slip_consecutive_steps"], 2)
        self.assertNotIn("grip_check_preset", split_gripper)
        self.assertNotIn("min_grip_m", split_gripper)

        repaired_off = build_repaired_args(
            self.h,
            dict(observation_id=obs_id, plan_id="plan-cs-repair", carry_slip_guard=False),
            lambda _h, pid: self.h.unified_action_plans[pid],
        )
        self.assertIs(repaired_off["carry_slip_guard"], False)
        self.assertNotIn("carry_slip_scope", repaired_off)
        self.assertNotIn("slip_consecutive_steps", repaired_off)

        rec2 = deepcopy(self.h.unified_action_plans[src_id])
        rec2["summary"]["completed_stage_ids"] = ["grasp/motion", "grasp", "lift"]
        rec2["summary"]["failed_stage_id"] = "place"
        self.h.unified_action_plans["plan-cs-rerun"] = rec2
        rerun_resp = typed_dispatch(
            self.h,
            dict(
                mode="preflight",
                observation_id=obs_id,
                plan_id="plan-cs-rerun",
                stages=[
                    dict(
                        id="lift",
                        kind="move",
                        arm="left",
                        grasp_position_m=[0.20, 0.15, 0.92],
                        angles_deg=[0.0, 0.0, 0.0],
                    )
                ],
            ),
            {"tool_call_id": "cs-repair-rerun"},
        )
        self.assertEqual(rerun_resp.data["re_executed_stage_ids"], ["lift"])

        # (d) batch_continuation：apply_stage_override、_apply_compiled_stage_override 与 resolve_backend_continuation
        raw_st = dict(
            id="close",
            kind="gripper",
            arm="left",
            goal_normalized=0.0,
            carry_slip_guard=True,
            carry_slip_scope="until_release",
            slip_consecutive_steps=3,
        )
        over_step = apply_stage_override(deepcopy(raw_st), dict(slip_consecutive_steps=2))
        self.assertIs(over_step["carry_slip_guard"], True)
        self.assertEqual(over_step["carry_slip_scope"], "until_release")
        self.assertEqual(over_step["slip_consecutive_steps"], 2)

        over_off = apply_stage_override(deepcopy(raw_st), dict(carry_slip_guard=False))
        self.assertIs(over_off["carry_slip_guard"], False)
        self.assertNotIn("carry_slip_scope", over_off)
        self.assertNotIn("slip_consecutive_steps", over_off)
        validate_carry_slip_syntax(over_off)

        over_conflict = apply_stage_override(
            deepcopy(raw_st), dict(carry_slip_guard=False, carry_slip_scope="until_release")
        )
        with self.assertRaisesRegex(ValueError, "invalid_carry_slip_guard_disabled"):
            validate_carry_slip_syntax(over_conflict)

        compiled_st = dict(
            id="close",
            kind="gripper",
            arm="left",
            action=dict(gripper_events=[dict(goal_normalized=0.0)]),
            carry_slip_guard=True,
            carry_slip_scope="until_release",
            slip_consecutive_steps=3,
        )
        _apply_compiled_stage_override(compiled_st, dict(carry_slip_guard=False))
        self.assertIs(compiled_st["carry_slip_guard"], False)
        self.assertNotIn("carry_slip_scope", compiled_st)
        self.assertNotIn("slip_consecutive_steps", compiled_st)

        fake_adapter = SimpleNamespace(
            _batch_plans={
                "bp1": dict(
                    source_stages=[
                        dict(
                            deepcopy(compiled_st),
                            carry_slip_guard=True,
                            carry_slip_scope="until_release",
                            slip_consecutive_steps=2,
                        )
                    ],
                    source_action=dict(carry_slip_guard=True, carry_slip_scope="until_release", slip_consecutive_steps=2),
                    completed_stage_ids=[],
                )
            }
        )
        backend_resolved, _ = resolve_backend_continuation(
            fake_adapter,
            dict(source_plan_id="bp1", start_from_stage_id="close", carry_slip_guard=False),
        )
        self.assertIs(backend_resolved["carry_slip_guard"], False)
        self.assertNotIn("carry_slip_scope", backend_resolved)
        self.assertNotIn("slip_consecutive_steps", backend_resolved)

    def test_execute_if_clean_cached_plan_and_continuation_expanded_cleanliness(self):
        """验证 execute_if_clean+plan_id 脏计划降级零执行且不污染缓存、检查点正常恢复、以及 source_plan_id 展开检出闭爪。"""
        from types import SimpleNamespace
        from ni_rua.runtime.actions.unified_actions import preview_downgrade_reasons_for_args

        self.set_preview_stub()
        self.robot.execute.reset_mock()
        obs_id = self.h.latest["observation_id"]

        # 1) 先用 preflight 生成含未声明障碍碰撞（geometry_review_risks）的缓存计划
        dirty_pf = typed_dispatch(
            self.h,
            dict(
                mode="preflight",
                observation_id=obs_id,
                obstacles=[
                    dict(
                        id="obs_box",
                        min_m=[0.08, -0.22, 0.76],
                        max_m=[0.12, -0.18, 0.86],
                        margin_m=0.0,
                        evidence="visible obstacle",
                    )
                ],
                stages=[
                    dict(
                        id="move_into_box",
                        kind="move",
                        arm="right",
                        grasp_position_m=[0.10, -0.20, 0.80],
                        angles_deg=[0.0, 0.0, 0.0],
                    )
                ],
            ),
            {"tool_call_id": "dirty-pf-cached"},
        )
        dirty_id = dirty_pf.data["plan_id"]
        self.assertNotIn("downgraded_from_execute_if_clean", self.h.unified_action_plans[dirty_id]["summary"])

        # 对脏缓存计划调 mode='execute_if_clean', plan_id=... 必须降级返回 plan 副本且默认 0 图、0 次物理执行
        with patch(
            "ni_rua.runtime.actions.inspection._trajectory_image",
            return_value=(dict(status="available"), [dict(type="inputImage", imageUrl="data:image/jpeg;base64,AA==")]),
        ) as overlay_fn:
            downgraded = typed_dispatch(
                self.h,
                dict(mode="execute_if_clean", plan_id=dirty_id),
                {"tool_call_id": "eic-dirty-cached-0img"},
            )
            overlay_fn.assert_not_called()
            wire_0 = wire_response(downgraded)
            self.assertEqual([c for c in wire_0["contentItems"] if c["type"] == "inputImage"], [])

            downgraded_img = typed_dispatch(
                self.h,
                dict(mode="execute_if_clean", plan_id=dirty_id, image_delivery="all"),
                {"tool_call_id": "eic-dirty-cached-img"},
            )
            overlay_fn.assert_called_once()
            wire_img = wire_response(downgraded_img)
            self.assertEqual(len([c for c in wire_img["contentItems"] if c["type"] == "inputImage"]), 1)

        self.assertEqual(downgraded.kind, "plan")
        self.assertTrue(downgraded.data["downgraded_from_execute_if_clean"])
        self.assertIn("geometry_review_risks", downgraded.data["downgrade_reasons"])
        self.assertEqual(downgraded.data["executed_native_steps"], 0)
        self.assertIs(downgraded.data["physics_advanced"], False)
        self.assertEqual(downgraded.data["next_action"]["mode"], "execute")
        self.assertEqual(self.robot.execute.call_count, 0)
        # 活动缓存摘要未被污染，显式 mode='execute' 仍可正常执行一次
        self.assertNotIn("downgraded_from_execute_if_clean", self.h.unified_action_plans[dirty_id]["summary"])
        explicit_exec = typed_dispatch(
            self.h,
            dict(mode="execute", plan_id=dirty_id),
            {"tool_call_id": "explicit-exec-after-downgrade"},
        )
        self.assertEqual(explicit_exec.kind, "action")
        self.assertEqual(self.robot.execute.call_count, 1)

        # 2) awaiting_observation 检查点计划调 preview_downgrade_reasons_for_args 返回 []，允许直接续执行
        ckpt_pf = typed_dispatch(
            self.h,
            dict(
                mode="preflight",
                observation_id=self.h.latest["observation_id"],
                stages=[dict(id="chk", kind="observe", reason="inspect")],
            ),
            {"tool_call_id": "ckpt-pf"},
        )
        ckpt_id = ckpt_pf.data["plan_id"]
        self.h.unified_action_plans[ckpt_id]["summary"]["status"] = "awaiting_observation"
        self.assertEqual(preview_downgrade_reasons_for_args(self.h, dict(plan_id=ckpt_id)), [])
        ckpt_exec = typed_dispatch(
            self.h,
            dict(mode="execute_if_clean", plan_id=ckpt_id),
            {"tool_call_id": "ckpt-resume-eic"},
        )
        self.assertEqual(ckpt_exec.kind, "action")
        self.assertEqual(self.robot.execute.call_count, 2)

        # 3) 实时钟本体（SO-101）上 source_plan_id 续跑展开后检出 closed_jaw_before_grasp:right
        so101_profile = self._load_so101_profile(empty_close_aperture_m=0.0045)
        rt_harness = SimpleNamespace(
            embodiment=so101_profile,
            identity=deepcopy(self.h.identity),
            capabilities=dict(arms=["right"]),
            latest=dict(
                observation_id="obs_rt_1",
                grip_facts=dict(right=dict(measured_aperture_m=0.0045)),
            ),
            robot_state=None,
            unified_action_plans={
                "p_rt_src": dict(
                    identity=deepcopy(self.h.identity),
                    source_stages=[
                        dict(id="approach", kind="move", arm="right", position_m=[0.2, 0.0, 0.12], quaternion_xyzw=[0, 0, 0, 1]),
                        dict(id="close_jaw", kind="gripper", arm="right", goal_normalized=0.0),
                    ],
                    source_args=dict(
                        stages=[
                            dict(id="approach", kind="move", arm="right", position_m=[0.2, 0.0, 0.12], quaternion_xyzw=[0, 0, 0, 1]),
                            dict(id="close_jaw", kind="gripper", arm="right", goal_normalized=0.0),
                        ]
                    ),
                    summary=dict(status="planned", executed_batch_native_steps=0, plan_revision=0),
                )
            },
        )
        reasons_cont = preview_downgrade_reasons_for_args(
            rt_harness,
            dict(
                mode="execute_if_clean",
                observation_id="obs_rt_1",
                source_plan_id="p_rt_src",
                start_from_stage_id="approach",
            ),
            summary=dict(status="planned"),
        )
        self.assertIn("closed_jaw_before_grasp:right", reasons_cont)
        reasons_cached_rt = preview_downgrade_reasons_for_args(rt_harness, dict(mode="execute_if_clean", plan_id="p_rt_src"))
        self.assertIn("closed_jaw_before_grasp:right", reasons_cached_rt)

    def test_inspect_returns_typed_data_and_so101_preflight_help(self):
        """验证 inspect 在 plan_id/action_id/trajectory_ref 三种入口返回类型化 data 且 wire 文本不变，并核验 SO-101 preflight 文案。"""
        from ni_rua.runtime.actions.robot_action_schema import build as build_robot_action_schema

        self.set_preview_stub()
        obs_id = self.h.latest["observation_id"]
        pf = typed_dispatch(
            self.h,
            dict(
                mode="preflight",
                observation_id=obs_id,
                stages=[
                    dict(
                        id="m1",
                        kind="move",
                        arm="left",
                        grasp_position_m=[0.20, 0.15, 0.85],
                        angles_deg=[0.0, 0.0, 0.0],
                    )
                ],
            ),
            {"tool_call_id": "pf-for-inspect"},
        )
        plan_id = pf.data["plan_id"]
        cand_ref = pf.data["trajectory_ref"]

        # 1) inspect(plan_id=...)：kind='plan'，保留顶层计划状态/游标并附带 inspection 采样字典
        plan_insp = self.h.invoke("robot_action", dict(mode="inspect", plan_id=plan_id), {"tool_call_id": "insp-plan"}, role="actor")
        self.assertIsNone(plan_insp.contract_error, plan_insp.contract_error)
        self.assertEqual(plan_insp.kind, "plan")
        self.assertEqual(plan_insp.data["plan_id"], plan_id)
        self.assertEqual(plan_insp.data["status"], "planned")
        self.assertEqual(plan_insp.data["executed_batch_native_steps"], 0)
        self.assertEqual(plan_insp.data["inspection"]["trajectory_ref"], cand_ref)
        self.assertGreaterEqual(len(plan_insp.data["inspection"]["samples"]), 1)
        wire_plan = wire_response(plan_insp)
        self.assertNotIn("inspection", json.loads(wire_plan["contentItems"][0]["text"]))
        self.assertNotIn("available", json.loads(wire_plan["contentItems"][1]["text"]))

        # 2) inspect(action_id=...)：kind='data'，返回采样字典 + available=True，wire 文本不含 available
        act = compile_request(
            self.h,
            dict(
                observation_id=obs_id,
                stages=[dict(id="m1", kind="move", arm="left", grasp_position_m=[0.2, 0.15, 0.85], angles_deg=[0, 0, 0])],
            ),
        )
        act["action_id"] = "archived-insp-1"
        self.store.save("action_request", "archived-insp-1", act)
        common = dict(
            identity=dict(self.h.identity),
            observation_id=obs_id,
            action_id="archived-insp-1",
            frame_id="env_origin",
            tcp_config_id="arx-x5-link6-v1",
        )
        measured_path = dict(
            common,
            source_kind="measured_action_trajectory",
            samples=[dict(native_step=0, arms={"left": dict(q_rad=[0.1] * 6, phase="measured")})],
        )
        fb_doc = dict(common, schema_version="action-trajectory-feedback-v1", paths={"measured": measured_path}, summary={})
        fb_ref = self.store.put_artifact(json.dumps(fb_doc).encode(), "action_trajectory_feedback", "application/json")
        self.store.set_state(
            "unified_action_feedback",
            "archived-insp-1",
            dict(trajectory_ref=fb_ref, identity=dict(self.h.identity)),
        )
        act_insp = self.h.invoke(
            "robot_action",
            dict(mode="inspect", action_id="archived-insp-1"),
            {"tool_call_id": "insp-act"},
            role="program",
        )
        self.assertIsNone(act_insp.contract_error, act_insp.contract_error)
        self.assertEqual(act_insp.kind, "data")
        self.assertIs(act_insp.data["available"], True)
        self.assertEqual(act_insp.data["action_id"], "archived-insp-1")
        self.assertEqual(act_insp.data["source_kind"], "measured_action_trajectory")
        self.assertEqual(act_insp.data["samples"][0]["arms"]["left"]["q_rad"], [0.1] * 6)
        self.assertNotIn("available", json.loads(wire_response(act_insp)["contentItems"][0]["text"]))

        # 3) inspect(trajectory_ref=...)：kind='data'，返回采样字典
        ref_insp = typed_dispatch(self.h, dict(mode="inspect", trajectory_ref=cand_ref), {"tool_call_id": "insp-ref"})
        self.assertEqual(ref_insp.kind, "data")
        self.assertIs(ref_insp.data["available"], True)
        self.assertEqual(ref_insp.data["trajectory_ref"], cand_ref)
        self.assertGreaterEqual(len(ref_insp.data["samples"]), 1)
        self.assertNotIn("available", json.loads(wire_response(ref_insp)["contentItems"][0]["text"]))

        # 4) M4-7：非仿真本体（SO-101）observation_id 说明包含 preflight
        so101_contract = build_robot_action_schema(self._load_so101_profile())
        obs_help = so101_contract.action_field_help["observation_id"]
        self.assertTrue(obs_help.startswith("Required for preview/preflight"), obs_help)

    def test_validate_raw_action_presets_pre_plan_update_and_mode_precedence(self):
        """验证 6 类预设在顶层/阶段/覆盖层前置校验（早于 plan_update）、保留顶层历史别名并严格遵守模式字段优先级。"""
        from ni_rua.core.tool_errors import ToolInputError
        from ni_rua.robot.model.embodiments import ARX_X5_PROFILE
        from ni_rua.runtime.actions.unified_actions import validate_raw_action_presets

        self.set_preview_stub()
        self.robot.preview_batch.reset_mock()
        obs_id = self.h.latest["observation_id"]
        base_stages = [
            dict(
                id="s1",
                kind="move",
                arm="left",
                grasp_position_m=[0.20, 0.15, 0.85],
                angles_deg=[0.0, 0.0, 0.0],
            )
        ]

        # 1) 顶层 6 类预设传非法值 'zzz' 时抛含精确 JSON Pointer 的 ToolInputError，且不污染 task_plan
        for preset_field in (
            "tolerance_preset",
            "stall_guard_preset",
            "grip_guard_preset",
            "joint_travel_preset",
            "clearance_preset",
            "target_exemption_preset",
        ):
            with self.subTest(field=preset_field):
                with self.assertRaises(ToolInputError) as ctx:
                    typed_dispatch(
                        self.h,
                        dict(
                            mode="preflight",
                            observation_id=obs_id,
                            plan_update=dict(plan="Should never be saved when preset is invalid."),
                            stages=base_stages,
                            **{preset_field: "zzz"},
                        ),
                        {"tool_call_id": f"bad-{preset_field}"},
                    )
                self.assertEqual(ctx.exception.field_path, f"/{preset_field}")
                self.assertEqual(ctx.exception.actual, "zzz")
                self.assertIn(preset_field, ctx.exception.suggestion)
        self.assertIsNone(self.h.task_planning.current(self.h.identity))
        self.assertEqual(self.robot.preview_batch.call_count, 0)

        # 2) stages[i] 与 stage_overrides 非法预设精确定位路径
        with self.assertRaises(ToolInputError) as st_ctx:
            validate_raw_action_presets(
                dict(
                    mode="preview",
                    observation_id=obs_id,
                    stages=[dict(base_stages[0], tolerance_preset="zzz")],
                ),
                ARX_X5_PROFILE,
            )
        self.assertEqual(st_ctx.exception.field_path, "/stages/0/tolerance_preset")

        with self.assertRaises(ToolInputError) as ov_ctx:
            validate_raw_action_presets(
                dict(
                    mode="preview",
                    observation_id=obs_id,
                    source_plan_id="p1",
                    start_from_stage_id="s1",
                    stage_overrides={"s1": {"stall_guard_preset": "zzz"}},
                ),
                ARX_X5_PROFILE,
            )
        self.assertEqual(ov_ctx.exception.field_path, "/stage_overrides/s1/stall_guard_preset")

        # 3) ARX-X5 顶层保留历史兼容别名（fine / firm / solid），SO-101 拒绝 X5 独有别名及不支持的能力预设
        validate_raw_action_presets(
            dict(
                mode="preflight",
                observation_id=obs_id,
                tolerance_preset="fine",
                stall_guard_preset="firm",
                grip_guard_preset="solid",
                stages=base_stages,
            ),
            ARX_X5_PROFILE,
        )
        so101_profile = self._load_so101_profile()
        with self.assertRaises(ToolInputError) as so_stall_ctx:
            validate_raw_action_presets(
                dict(
                    mode="preflight",
                    observation_id=obs_id,
                    stall_guard_preset="firm",
                    stages=[dict(id="s1", kind="move", arm="right", position_m=[0.2, 0.0, 0.1], quaternion_xyzw=[0, 0, 0, 1])],
                ),
                so101_profile,
            )
        self.assertEqual(so_stall_ctx.exception.field_path, "/stall_guard_preset")
        for unsupported_field in ("joint_travel_preset", "clearance_preset", "target_exemption_preset"):
            with self.assertRaises(ToolInputError) as so_cap_ctx:
                validate_raw_action_presets(
                    dict(
                        mode="preflight",
                        observation_id=obs_id,
                        stages=[dict(id="s1", kind="move", arm="right", position_m=[0.2, 0.0, 0.1], quaternion_xyzw=[0, 0, 0, 1])],
                        **{unsupported_field: "normal"},
                    ),
                    so101_profile,
                )
            self.assertEqual(so_cap_ctx.exception.field_path, f"/{unsupported_field}")

        # 4) 模式字段边界优先于预设名单诊断：execute/inspect/reach 携带非法预设报模式错误且不触发 plan_update
        with self.assertRaisesRegex(ValueError, "cached_batch_commands_cannot_be_changed"):
            typed_dispatch(
                self.h,
                dict(
                    mode="execute",
                    plan_id="p1",
                    grip_guard_preset="zzz",
                    expected_revision=37,
                    plan_update=dict(plan="Must not commit on bad execute fields."),
                ),
                {"tool_call_id": "bad-exec-preset"},
            )
        with self.assertRaisesRegex(ValueError, "robot_action_inspect_has_command_fields"):
            validate_raw_action_presets(dict(mode="inspect", plan_id="p1", tolerance_preset="zzz"), ARX_X5_PROFILE)
        with self.assertRaisesRegex(ValueError, "robot_action_reach_has_unsupported_fields"):
            validate_raw_action_presets(
                dict(
                    mode="reach",
                    observation_id=obs_id,
                    grip_guard_preset="zzz",
                    stages=[dict(id="s1", kind="move", arm="right", position_m=[0.2, 0.0, 0.1], quaternion_xyzw=[0, 0, 0, 1])],
                ),
                so101_profile,
            )
        self.assertIsNone(self.h.task_planning.current(self.h.identity))



if __name__ == "__main__":
    unittest.main()
