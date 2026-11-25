"""CPU admission regressions: inspect commands, never execute a simulator."""

NI_RUA_TEST_TIER = "fast"

from copy import deepcopy
import json
import math
from pathlib import Path
from support.paths import FIXTURES_ROOT
import threading
import tempfile
from types import SimpleNamespace
import unittest
import numpy as np
from ni_rua.robot.motion.motion_safety import assess_motion
from support.motion_fixtures import Limits, Planner, motion_action, periodic_pose
from support.motion_fixtures import fake_adapter, wait_action
from ni_rua.robot.model.robot_model import urdf_manifest


def safety_adapter(*, after_step=None, **native_options):
    measured = {arm: [0.] * 6 for arm in ("left", "right")}
    jaws = {"left": 1., "right": .3}
    pose_offsets = {arm: np.zeros(3) for arm in ("left", "right")}

    def follow(count):
        for arm in ("left", "right"):
            measured[arm] = list(dispatched[-1][f"{arm}_arm_joint_state"])
            jaws[arm] = dispatched[-1][f"{arm}_ee_joint_state"][0]
        if after_step is not None:
            after_step(count, dispatched[-1], measured)

    adapter, dispatched = fake_adapter(after_step=follow, **({"step_limit": 1000} | native_options))
    adapter.arm_velocity_step_limits = {arm: np.array([.2] * 6) for arm in ("left", "right")}
    adapter.tracking_guard_enabled = False
    adapter.joint_replan_enabled = False

    def state():
        result = {"arms": {}}
        for arm in ("left", "right"):
            pose = periodic_pose(measured[arm])
            pose["position_m"] = (np.asarray(pose["position_m"]) + pose_offsets[arm]).tolist()
            result["arms"][arm] = {
                "q_rad": list(measured[arm]),
                "ee_pose": pose,
                "joint_names": [f"{arm}_joint{i}" for i in range(1, 7)],
                "source_kind": "measurement",
                "gripper": {"position_source_kind": "measurement", "position_m": [jaws[arm]] * 2},
            }
            result["arms"][arm]["ee_pose"].update(
                source_kind="measurement", frame_id="env_origin", tcp_config_id="arx-x5-link6-v1"
            )
        return result

    adapter.state = state
    robots = {
        arm: SimpleNamespace(
            robot_name=arm, arm_name=arm + "_arm", type="target",
            arm_joints_name=[f"{arm}_joint{i}" for i in range(1, 7)], arm_joint_indices=list(range(6)),
            entity_origin_pose=[0., 0., 0., 1., 0., 0., 0.], ee_link_name="tool",
            gripper_joints_name=["jaw1", "jaw2"], gripper_move={"base": "jaw1", "sign": 1}, gripper_scale=[0., 1.],
        )
        for arm in ("left", "right")
    }
    planners = {arm: Planner(wound=False) for arm in ("left", "right")}
    adapter.robot = SimpleNamespace(
        planner=planners, robot_list=list(robots.values()),
        robot_key=[SimpleNamespace(data=SimpleNamespace(soft_joint_pos_limits=Limits())) for _ in ("left", "right")],
        get_robot_by_arm_name=lambda name: robots[name.split("_")[0]],
        get_joint=lambda robot, envs: [np.asarray(measured[robot.robot_name], dtype=float)],
    )
    adapter.pose_offsets = pose_offsets
    adapter.measured_jaws = jaws
    return adapter, planners, dispatched


def batch_move_action():
    base = motion_action()
    return {k: v for k, v in base.items() if k not in ("path", "gripper_events")} | {
        "type": "BatchAction",
        "stages": [
            {"id": "l", "kind": "move", "arm": "left", "action": motion_action("left", (.125,))},
            {"id": "r", "kind": "move", "arm": "right", "action": motion_action("right", (.225,))},
        ],
    }


def enabled(adapter, limit=1000):
    adapter.motion_safety_guard_enabled=True
    adapter.native_action_budget_limit=limit
    adapter.native_actions_dispatched=0
    adapter.robot_model={'joints':[dict(name=f'{a}_joint{i}',type='revolute',limits=dict(lower=-10.,upper=10.))
                                  for a in ('left','right') for i in range(1,7)]}
    return adapter


class MotionSafetyTests(unittest.TestCase):
    def test_real_saved_a27_winding_rejected_without_wrapping(self):
        saved=json.loads(((FIXTURES_ROOT / 'surface_a27_right_home_plan.json')).read_text())
        q=np.array(saved['candidate_q_rad']);copy=q.copy()
        result=assess_motion(saved['current_q_rad'],q,joint_names=saved['joint_names'],joint_types=saved['joint_types'],
                             joint_limits=saved['joint_limits_rad'])
        self.assertEqual(result['status'],'reject')
        self.assertAlmostEqual(math.degrees(result['witnesses'][0]['total_travel_rad']),297.7060073532417)
        self.assertEqual(result['witnesses'][0]['reason'],'periodic_winding')
        np.testing.assert_array_equal(q,copy)

    def test_long_direct_change_small_paths_and_metadata_unknown(self):
        for goal,reject in [(1.,False),(math.pi,False),(3.4,False),(6.,True)]:
            q=np.zeros((1,6));q[0,0]=goal
            result=assess_motion(np.zeros(6),q,joint_names=list('abcdef'),joint_types=['revolute']*6,
                                 joint_limits=[[-10.,10.]]*6)
            self.assertEqual(result['status']=='reject',reject)
        r=assess_motion(np.zeros(6),[[6,0,0,0,0,0]],joint_names=list('abcdef'))
        self.assertEqual(r['periodic_check'],'unknown_joint_types_skipped')

    def test_bounded_joint_does_not_wrap_across_hard_stop_and_unknown_limits_do_not_guess(self):
        start=np.zeros(6);start[0]=math.radians(-170)
        goal=np.zeros((1,6));goal[0,0]=math.radians(170)
        kwargs=dict(joint_names=list('abcdef'),joint_types=['revolute']*6)
        result=assess_motion(start,goal,**kwargs,joint_limits=[[-math.pi,math.pi]]*6)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['periodic_checks'][0]['status'],'shortest_equivalent_outside_limits')
        result=assess_motion(start,goal,**kwargs)
        self.assertEqual(result['status'],'pass')
        self.assertEqual(result['periodic_checks'][0]['status'],'unknown_limits')
        result=assess_motion(start,goal,joint_names=list('abcdef'),joint_types=['continuous']*6)
        self.assertEqual(result['status'],'reject')
        self.assertEqual(result['witnesses'][0]['periodic_eligibility']['status'],'continuous_joint')

    def test_manifest_exports_only_declared_finite_limits(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'robot.urdf'
            path.write_text('<robot name="test"><joint name="hinge" type="revolute">'
                '<parent link="base"/><child link="arm"/><limit lower="-3.14" upper="3.14"/>'
                '</joint><joint name="unknown" type="revolute"><parent link="arm"/>'
                '<child link="tip"/></joint></robot>')
            model=urdf_manifest(path)
        self.assertEqual(model['joints'][0]['limits'],dict(lower=-3.14,upper=3.14))
        self.assertNotIn('limits',model['joints'][1])

    def test_backtracking_and_budget_admission(self):
        q=np.zeros((2,6));q[:,0]=[2.,0.]
        r=assess_motion(np.zeros(6),q,joint_names=list('abcdef'),joint_types=['revolute']*6)
        self.assertEqual(r['witnesses'][0]['reason'],'large_backtracking')
        r=assess_motion(np.zeros(6),np.zeros((9,6)),joint_names=list('abcdef'),episode_remaining=8)
        self.assertEqual(r['exceeded_budgets'],{'episode':8})

    def test_active_execution_modes_reject_initial_winding_before_dispatch(self):
        adapter, planners, rows = safety_adapter(); enabled(adapter)
        planners['left'].wound = True
        result = adapter.execute(motion_action(angles=(.125,)), lambda e: None, threading.Event())
        self.assertEqual(result['reason'], 'motion_safety_rejected')
        self.assertEqual(result['executed_native_steps'], 0)
        self.assertEqual(rows, [])
        self.assertEqual(result['motion_safety']['prior_executed_native_steps'], 0)
        self.assertFalse(any(p.joint_calls for p in planners.values()))

        adapter, planners, rows = safety_adapter(); enabled(adapter)
        planners['left'].wound = True
        plan = adapter.preview_batch(batch_move_action())
        self.assertEqual((plan['status'], plan['reason']), ('failed', 'motion_safety_rejected'))
        self.assertEqual(plan['executed_native_steps'], 0)
        self.assertEqual(rows, [])
        self.assertFalse(any(p.joint_calls for p in planners.values()))

    def test_later_waypoint_rejection_preserves_completed_work(self):
        adapter, planners, rows = safety_adapter(); enabled(adapter)
        plan = planners['left'].plan_path
        calls = 0

        def second_wound(*args):
            nonlocal calls
            calls += 1
            planners['left'].wound = calls == 2
            return plan(*args)

        planners['left'].plan_path = second_wound
        result = adapter.execute(motion_action(angles=(.225, .45)), lambda e: None, threading.Event())
        self.assertEqual(result['reason'], 'motion_safety_rejected')
        self.assertEqual(result['failed_waypoint_index'], 1)
        self.assertEqual(result['executed_native_steps'], 5)
        self.assertEqual(result['motion_safety']['prior_executed_native_steps'], 5)
        self.assertEqual(len(rows), 5)

    def test_episode_budget_rejects_nominal_segment_without_spending_last_rows(self):
        adapter, _, rows = safety_adapter(); enabled(adapter, 2)
        result = adapter.execute(motion_action(angles=(.125,)), lambda e: None, threading.Event())
        self.assertEqual(result['reason'], 'motion_safety_rejected')
        self.assertEqual(result['executed_native_steps'], 0)
        self.assertEqual(rows, [])
        self.assertIn('insufficient_remaining_native_budget', result['motion_safety']['arms']['left']['reasons'])

    def test_standalone_hold_and_gripper_budget_reject_before_dispatch(self):
        for kind in ('hold', 'gripper'):
            for guard in (True, False):
                with self.subTest(kind=kind, guard=guard):
                    adapter, _, rows = safety_adapter(); enabled(adapter, 2)
                    adapter.motion_safety_guard_enabled = guard
                    action = wait_action(3)
                    if kind == 'gripper':
                        del action['hold_native_steps']
                        action['gripper_events'] = [dict(arm='left', phase='after_motion', goal_normalized=0.)]
                    result = adapter.execute(action, lambda e: None, threading.Event())
                    self.assertEqual(result['reason'], 'motion_safety_rejected' if guard else None)
                    self.assertEqual(len(rows), 0 if guard else 3 if kind == 'hold' else 6)

    def test_flag_off_preserves_original_commands(self):
        adapter, planners, rows = safety_adapter(); enabled(adapter)
        adapter.motion_safety_guard_enabled = False; planners['left'].wound = True
        result = adapter.execute(motion_action(angles=(.125,)), lambda e: None, threading.Event())
        self.assertEqual(result['status'], 'completed')
        self.assertGreater(rows[-1]['left_arm_joint_state'][0], math.tau)

    def test_fresh_jaw_growth_rechecks_episode_budget_after_motion(self):
        def drift(count, row, measured):
            if count == 3:
                adapter.measured_jaws['left'] = 1.
        adapter, _, rows = safety_adapter(after_step=drift); enabled(adapter, 9)
        adapter.measured_jaws['left'] = .2
        adapter.last_dispatched_goals['left'] = 0.
        action = motion_action(angles=(.125,), gripper=True); action['gripper_speed_scale'] = .8
        result = adapter.execute(action, lambda e: None, threading.Event())
        self.assertEqual(result['reason'], 'motion_safety_rejected')
        self.assertEqual(result['executed_native_steps'], 3)
        self.assertEqual(result['motion_safety']['arms']['left']['episode_remaining_native_steps'], 6)
        self.assertEqual(len(rows), 3)

    def test_exact_cached_commands_are_checked_without_modification(self):
        adapter, planners, rows = safety_adapter(); enabled(adapter)
        planners['left'].wound=True
        q=np.array([[.04*i,0,0,0,0,0] for i in range(1,151)])
        held=adapter._held_row(adapter.state())
        native=[]
        for joints in q:
            row=list(held);row[:6]=joints.tolist();native.append(row)
        action=motion_action(angles=(6.-math.tau,));action['preview_plan_id']='cached'
        adapter._consume_motion_preview=lambda action:dict(positions=q.copy(),motion_rows=native,gripper_rows=[],gripper_motion=None,state_binding={})
        result=adapter.execute(action,lambda e:None,threading.Event())
        self.assertEqual(result['reason'],'motion_safety_rejected')
        self.assertEqual(rows,[])
        self.assertEqual(result['preview_execution_semantics'],'cached native rows; no replanning, resampling or arrival jaw rebase')
        self.assertEqual(q[-1,0],6.)


if __name__ == "__main__":
    unittest.main()
