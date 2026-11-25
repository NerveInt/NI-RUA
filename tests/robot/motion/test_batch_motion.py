"""Geometric counterexamples for shared-timeline proxy diagnostics."""

NI_RUA_TEST_TIER = "fast"

# === tests/test_batch_geometry.py ===

from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.robot.motion.batch_geometry import review_batch, validate_proxies, annotate_overlay


def fixture(left=(0.,.6),right=(0.,-.6),right_base=1.):
    joints=[]
    for index in range(1,7):
        joints.append(dict(name='joint'+str(index),type='prismatic' if index==1 else 'fixed',
            parent='base' if index==1 else 'link'+str(index-1),child='link'+str(index),
            T_parent_joint=np.eye(4).tolist(),axis=[1.,0.,0.]))
    model=dict(base_link='base',joints=joints)
    before=dict(arms={arm:dict(joint_names=['joint'+str(i) for i in range(1,7)],q_rad=[0.]*6,
        source_kind='measurement',ee_pose=dict(position_m=[base,0.,0.],quaternion_xyzw=[0.,0.,0.,1.],
            source_kind='measurement',frame_id='env_origin',tcp_config_id='arx-x5-link6-v1'),
        gripper=dict(joint_names=[],position_m=[],position_source_kind='measurement'))
        for arm,base in [('left',0.),('right',right_base)]})
    samples=[dict(native_step=i,arms={arm:dict(q_rad=[values[i],0.,0.,0.,0.,0.],jaw_goal_normalized=1.,phase='move')
        for arm,values in [('left',left),('right',right)]}) for i in range(len(left))]
    trajectory=dict(frame_id='env_origin',tcp_config_id='arx-x5-link6-v1',samples=samples)
    hardware=dict(model=model,spheres={'link6':[dict(center=[0.,0.,0.],radius=.02)]},sphere_buffer_m=0.,
        jaws=dict(gripper_scale=[0.,1.],gripper_move=dict(base='jaw',sign=1),gripper_joints_name=[]))
    return SimpleNamespace(root=Path('/no-task-assets')),before,trajectory,hardware


class BatchGeometryTests(unittest.TestCase):
    def review(self,inputs,proxies):
        adapter,before,trajectory,hardware=inputs
        with (patch('ni_rua.robot.motion.trajectory_evidence._hardware',return_value=hardware),
              patch('ni_rua.robot.model.robot_model.collision_mesh_bounds',side_effect=OSError('unavailable'))):
            return review_batch(adapter,before,trajectory,proxies)

    def test_both_moving_arms_cross_while_neither_reaches_peers_start(self):
        result=self.review(fixture(),{})
        self.assertEqual(result['status'],'risks_detected')
        self.assertTrue(any(r['kind']=='inter_arm_proxy_intersection' for r in result['risks']))
        self.assertLess(result['minimum_estimated_clearance_m'],0)
        self.assertIn('Both arms move',result['semantics'])

    def test_obstacle_between_native_endpoints_is_not_missed(self):
        result=self.review(fixture(left=(0.,1.),right=(0.,0.),right_base=2.),dict(obstacles=[
            dict(id='rim',min_m=[.49,-.01,-.01],max_m=[.51,.01,.01],margin_m=0.,evidence='observed rim')]))
        risks=[r for r in result['risks'] if r['obstacle']=='rim']
        self.assertTrue(risks)
        self.assertEqual(risks[0]['native_step'],1)
        self.assertLess(risks[0]['estimated_clearance_m'],0)

    def test_held_extent_moves_rigidly_and_can_hit_when_robot_does_not(self):
        result=self.review(fixture(left=(0.,.4),right=(0.,0.),right_base=2.),dict(
            obstacles=[dict(id='neighbor',min_m=[.68,-.02,-.02],max_m=[.72,.02,.02],margin_m=0.,evidence='visible neighbor')],
            held_objects={'left':dict(min_m=[.29,-.01,-.01],max_m=[.31,.01,.01],margin_m=0.,evidence='estimated held extent')}))
        self.assertTrue(any(r['moving_part']=='left/held_object' and r['obstacle']=='neighbor' for r in result['risks']))
        self.assertTrue(any('release' in text for text in result['unknown']))

    def test_contact_declarations_never_erase_intersection(self):
        proxies=dict(obstacles=[dict(id='target',min_m=[.49,-.01,-.01],max_m=[.51,.01,.01],evidence='visible')],
            contact_entity_id='target',initial_contact=dict(entity_ids=['target'],evidence='initial estimate'))
        result=self.review(fixture(left=(0.,1.),right=(0.,0.),right_base=2.),proxies)
        self.assertTrue(any(r.get('contact_declaration') for r in result['risks']))
        self.assertFalse(result['collision_free_proven'])

    def test_validated_proxies_are_idempotent_and_bad_references_reject(self):
        raw=dict(obstacles=[dict(id='box',min_m=[0,0,0],max_m=[1,1,1],evidence='seen')],
            held_objects={'left':dict(min_m=[0,0,0],max_m=[.1,.1,.1],evidence='seen')})
        checked=validate_proxies(raw)
        self.assertEqual(validate_proxies(checked),checked)
        with self.assertRaisesRegex(ValueError,'contact_entity'):
            validate_proxies(dict(raw,contact_entity_id='missing'))

    def test_overlay_preserves_panels_and_draws_world_and_held_proxies(self):
        camera=dict(T=np.eye(4),K=np.array([[100.,0.,80.],[0.,100.,80.],[0.,0.,1.]]),
            D=np.zeros(5),width=160,height=160)
        trajectory=dict(samples=[dict(native_step=i,arms={'left':dict(link6_pose=dict(position_m=[i*.2,0.,2.],
            rotation_matrix=np.eye(3).tolist()))}) for i in (0,1)],batch_geometry_review=dict(
                obstacles=[dict(id='box',min_m=[-.2,-.2,2.],max_m=[.2,.2,2.2])],
                held_objects={'left':dict(min_m=[-.1,-.1,1.9],max_m=[.1,.1,2.1])}))
        image=np.zeros((160,320,3),dtype=np.uint8)
        metadata=dict(panel_width=160,panel_height=160,panels=[dict(native_step=0),dict(native_step=1)])
        rendered,meta=annotate_overlay(camera,image,metadata,trajectory)
        self.assertEqual(rendered.shape,(160,320,3))
        self.assertTrue(rendered[:140].any())
        self.assertEqual(meta['proxy_overlay']['held_object_arms'],['left'])


# === tests/test_batch_motion.py ===

from copy import deepcopy
import io
import math
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.robot.model import roboprobe_pose
from ni_rua.robot.motion.arx_actions import compile_action
from ni_rua.robot.motion.batch_contract import validate_batch_action
from ni_rua.robot.motion.batch_motion import _open_jaw_tilted_grasp_candidates, consume_batch
from ni_rua.runtime.actions.unified_actions import compile_request
from support.motion_fixtures import preview_adapter
from support.motion_fixtures import motion_action, wait_action


def move(identifier, arm='left', angle=-.125, **values):
    return dict(id=identifier, kind='move', arm=arm, action=motion_action(arm, (angle,)), **values)


def batch(stages, **values):
    own = motion_action()
    return {key: value for key,value in own.items() if key not in ('path', 'gripper_events')} | dict(type='BatchAction', stages=stages, **values)


def jaw(identifier='close', arm='left', goal=0., **values):
    own = motion_action(arm)
    own.update(path=dict(kind='hold', waypoints=[]),
        gripper_events=[dict(arm=arm, phase='after_motion', goal_normalized=goal)])
    return dict(id=identifier, kind='gripper', arm=arm, action=own, **values)


def jaws(adapter):
    robot = adapter.robot.robot_list[0]
    robot.gripper_joints_name = ['jaw']
    robot.gripper_move = dict(base='jaw', sign=1)
    robot.gripper_scale = [0., 1.]
    original = adapter.state
    def state():
        result = original()
        for arm in ('left', 'right'):
            result['arms'][arm]['gripper'] = dict(position_source_kind='measurement',
                position_m=[adapter.last_dispatched_goals[arm]], joint_names=['jaw'])
        return result
    adapter.state = state


class BatchMotionTests(unittest.TestCase):
    def execute(self, adapter, action, plan):
        return adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda update: None, threading.Event())

    def test_full_multistage_preview_uses_planned_endpoints_without_dispatch(self):
        adapter, planner, dispatched = preview_adapter()
        action = batch([move('first'), move('second', angle=-.25)])
        before = deepcopy(adapter.state())
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(adapter.state(), before)
        self.assertEqual(dispatched, [])
        self.assertEqual(adapter.env.take_action_cnt[0], 0)
        self.assertAlmostEqual(planner.pose_calls[1][0][0], -.125)
        self.assertEqual(plan['planned_native_steps'], 6)
        rows = deepcopy(adapter._batch_plans[plan['plan_id']]['rows'])
        with (patch.object(adapter, '_plan_segment', side_effect=AssertionError('execution must not plan')),
                patch.object(adapter, '_resample_action_plan', side_effect=AssertionError('execution must not resample'))):
            result = self.execute(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['executed_native_steps'], 6)
        for row, actual in zip(rows, dispatched):
            np.testing.assert_array_equal(row[:6], actual['left_arm_joint_state'])
            np.testing.assert_array_equal(row[7:13], actual['right_arm_joint_state'])

    def test_unequal_dual_routes_overlap_without_padding(self):
        adapter, _, dispatched = preview_adapter()
        action = batch([move('l1'), move('l2', angle=-.25), move('r1', 'right', -.375)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['planned_native_steps'], 8)
        self.assertEqual(plan['stage_costs']['l2']['start_native_step'], 3)
        self.assertEqual(plan['stage_costs']['r1']['start_native_step'], 0)
        result = self.execute(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(len(dispatched), 8)

    def test_dependency_delays_peer_and_cycles_reject_before_planning(self):
        adapter, planner, _ = preview_adapter()
        action = batch([move('left'), move('right', 'right', depends_on=['left'])])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['planned_native_steps'], 6)
        self.assertEqual(plan['stage_costs']['right']['start_native_step'], 3)
        action['stages'][0]['depends_on'] = ['right']
        with self.assertRaisesRegex(ValueError, 'dependency_cycle'):
            adapter.preview_batch(action)
        self.assertEqual(len(planner.pose_calls), 2)

    def test_late_planning_failure_executes_nothing(self):
        adapter, _, dispatched = preview_adapter(fail_pose_at=2)
        plan = adapter.preview_batch(batch([move('one'), move('two', angle=-.25)]))
        self.assertEqual(plan['reason'], 'path_infeasible')
        self.assertEqual(plan['failed_stage_id'], 'two')
        self.assertEqual(dispatched, [])

    def test_checkpoint_stops_peer_midstage_and_resumes_exact_suffix(self):
        adapter, planner, dispatched = preview_adapter()
        action = batch([move('l1'), dict(id='look', arm='left', kind='observe', reason='check grasp'),
            move('l2', angle=-.25), move('r1', 'right', -.5)])
        plan = adapter.preview_batch(action)
        rows = deepcopy(adapter._batch_plans[plan['plan_id']]['rows'])
        first = self.execute(adapter, action, plan)
        self.assertEqual((first['status'], first['reason']), ('stopped', 'observation_checkpoint'))
        self.assertEqual(first['batch_cursor'], 3)
        self.assertIn('r1', first['pending_stage_ids'])
        self.assertEqual(first['stage_results']['r1']['completed_steps'], 3)
        second = self.execute(adapter, action, plan)
        self.assertEqual(second['status'], 'completed')
        self.assertEqual(len(planner.pose_calls), 3)
        self.assertEqual(len(dispatched), len(rows))
        self.assertEqual(first['executed_native_steps']+second['executed_native_steps'], len(rows))

    def test_global_checkpoint_is_barrier_after_both_prior_stages(self):
        adapter, _, _ = preview_adapter()
        action = batch([move('l1'), move('r1', 'right', -.5), dict(id='look', kind='observe', reason='both arrived'),
            move('l2', angle=-.25)])
        plan = adapter.preview_batch(action)
        result = self.execute(adapter, action, plan)
        self.assertEqual(result['batch_cursor'], 10)
        self.assertEqual(set(result['completed_stage_ids']), {'l1', 'r1', 'look'})
        self.assertEqual(self.execute(adapter, action, plan)['status'], 'completed')

    def test_zero_step_checkpoint_revision_rejects_stale_approval(self):
        adapter, planner, dispatched = preview_adapter()
        action = batch([dict(id='look', kind='observe', reason='check before motion'),
            move('reach', angle=-.04)])
        plan = adapter.preview_batch(action)
        approved = dict(action, batch_plan_id=plan['plan_id'],
            batch_expected_cursor=0, batch_expected_revision=0)
        first = adapter.execute(approved, lambda update: None, threading.Event())
        self.assertEqual((first['status'], first['reason']), ('stopped', 'observation_checkpoint'))
        self.assertEqual((first['batch_cursor'], first['batch_revision']), (0, 1))
        self.assertEqual(first['executed_native_steps'], 0)
        self.assertEqual(dispatched, [])
        stale = adapter.execute(approved, lambda update: None, threading.Event())
        self.assertEqual(stale['status'], 'rejected')
        self.assertEqual(stale['reason'], 'batch_plan_revision_changed_since_review')
        self.assertEqual(dispatched, [])
        fresh = adapter.execute(dict(approved, batch_expected_revision=1),
            lambda update: None, threading.Event())
        self.assertEqual(fresh['status'], 'completed')
        self.assertEqual(fresh['batch_revision'], 2)
        self.assertEqual(fresh['executed_native_steps'], plan['planned_native_steps'])
        self.assertEqual(len(planner.pose_calls), 1)

    def test_changed_first_start_rejects_without_replanning(self):
        adapter, planner, dispatched = preview_adapter()
        action = batch([move('one')]); plan = adapter.preview_batch(action)
        adapter.control_step += 1
        result = self.execute(adapter, action, plan)
        self.assertEqual(result['status'], 'rejected')
        self.assertIn('stale_state', result['reason'])
        self.assertEqual(len(planner.pose_calls), 1)
        self.assertEqual(dispatched, [])

    def test_checkpoint_revalidates_unchanged_robot_after_clock_change(self):
        adapter, _, _ = preview_adapter()
        action = batch([move('one'), dict(id='look', kind='observe', reason='check'), move('two', angle=-.25)])
        plan = adapter.preview_batch(action)
        self.execute(adapter, action, plan)
        adapter.control_step += 1
        result = self.execute(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['resume_validation']['status'], 'valid')

    def test_checkpoint_drift_invalidates_suffix(self):
        adapter, _, dispatched = preview_adapter()
        action = batch([move('one'), dict(id='look', kind='observe', reason='check'), move('two', angle=-.25)])
        plan = adapter.preview_batch(action)
        self.execute(adapter, action, plan)
        original = adapter.state
        def drift():
            value = original(); value['arms']['left']['q_rad'][0] += .2
            return value
        adapter.state = drift
        result = self.execute(adapter, action, plan)
        self.assertEqual(result['status'], 'rejected')
        self.assertEqual(len(dispatched), 3)

    def test_shared_limit_rejects_complete_batch_including_checkpoint_suffix(self):
        adapter, _, dispatched = preview_adapter()
        stages = [move('first', angle=-3), dict(id='look', kind='observe', reason='check')]
        stages += [move('m'+str(i), angle=(3 if i%2 == 0 else -3)) for i in range(3)]
        plan = adapter.preview_batch(batch(stages))
        self.assertEqual(plan['reason'], 'motion_duration_limit')
        self.assertEqual(dispatched, [])

    def test_slow_pure_jaw_and_during_motion_jaw_compile_and_execute(self):
        adapter, _, dispatched = preview_adapter(); jaws(adapter)
        close = motion_action()
        close.update(path=dict(kind='hold', waypoints=[]), gripper_speed_scale=.8,
            gripper_events=[dict(arm='left', phase='after_motion', goal_normalized=.5)])
        movement = move('reach', angle=-.3)
        movement['action'].update(gripper_speed_scale=.8, gripper_events=[dict(arm='left', phase='during_motion',
            goal_normalized=0., start_fraction=.2, end_fraction=1.)])
        action = batch([dict(id='close', arm='left', kind='gripper', action=close), movement])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        self.assertEqual(plan['checkpoints'], [])
        result = self.execute(adapter, action, plan)
        self.assertEqual(result['status'], 'completed')
        values = [1.]+[row['left_ee_joint_state'][0] for row in dispatched]
        self.assertLessEqual(max(abs(b-a) for a,b in zip(values,values[1:])), .18*.8+1e-12)
        self.assertEqual(values[-1], 0.)

    def test_jaw_change_pauses_dependent_motion_then_resumes_exact_rows_once(self):
        for arm in ('left', 'right'):
            for initial, goal in ((1., 0.), (0., 1.)):
                with self.subTest(arm=arm, goal=goal):
                    adapter, planner, dispatched = preview_adapter(); jaws(adapter)
                    adapter.last_dispatched_goals[arm] = initial
                    action = batch([jaw(arm=arm, goal=goal, checkpoint_after_gripper=True), move('carry', arm)])
                    plan = adapter.preview_batch(action)
                    rows = deepcopy(adapter._batch_plans[plan['plan_id']]['rows'])
                    boundary = plan['stage_costs']['close']['end_native_step']
                    self.assertEqual(plan['checkpoints'][0]['cursor'], boundary)
                    self.assertEqual(dispatched, [])
                    with (patch.object(adapter, '_plan_segment', side_effect=AssertionError('must not replan')),
                            patch.object(adapter, '_resample_action_plan', side_effect=AssertionError('must not resample'))):
                        first = self.execute(adapter, action, plan)
                        self.assertEqual((first['status'], first['reason']), ('stopped', 'observation_checkpoint'))
                        self.assertEqual(first['stage_results']['carry']['completed_steps'], 0)
                        self.assertIsNotNone(first['terminal_observation'])
                        self.assertEqual(first['checkpoint_stage_ids'], ['close'])
                        second = self.execute(adapter, action, plan)
                    self.assertEqual(second['status'], 'completed')
                    self.assertEqual(first['executed_native_steps'] + second['executed_native_steps'], len(rows))
                    self.assertEqual(second['batch_revision'], 2)
                    actual = [row['left_arm_joint_state'] + row['left_ee_joint_state'] +
                        row['right_arm_joint_state'] + row['right_ee_joint_state'] for row in dispatched]
                    np.testing.assert_array_equal(actual, rows)
                    self.assertEqual(len(planner.pose_calls), 1)

    def test_synchronized_jaw_change_stops_only_at_end_of_its_move_stage(self):
        adapter, _, dispatched = preview_adapter(); jaws(adapter)
        closing = move('close_while_moving', angle=-.3, checkpoint_after_gripper=True)
        closing['action']['gripper_events'] = [dict(arm='left', phase='during_motion',
            goal_normalized=0., start_fraction=0., end_fraction=.5)]
        action = batch([closing, move('carry', angle=-.4)])
        plan = adapter.preview_batch(action)
        first = self.execute(adapter, action, plan)
        self.assertEqual(first['batch_cursor'], plan['stage_costs']['close_while_moving']['end_native_step'])
        self.assertEqual(first['checkpoint_stage_ids'], ['close_while_moving'])
        self.assertEqual(first['stage_results']['carry']['completed_steps'], 0)
        self.assertEqual(self.execute(adapter, action, plan)['status'], 'completed')
        self.assertEqual(len(dispatched), plan['planned_native_steps'])

    def test_jaw_checkpoint_pauses_independent_peer_midstage_without_serializing_routes(self):
        adapter, _, dispatched = preview_adapter(); jaws(adapter)
        action = batch([jaw(checkpoint_after_gripper=True), move('carry'), move('peer', 'right', -.5)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['stage_costs']['peer']['start_native_step'], 0)
        first = self.execute(adapter, action, plan)
        self.assertEqual(first['stage_results']['peer']['status'], 'paused')
        self.assertGreater(first['stage_results']['peer']['completed_steps'], 0)
        self.assertLess(first['batch_cursor'], plan['stage_costs']['peer']['end_native_step'])
        self.assertEqual(self.execute(adapter, action, plan)['status'], 'completed')
        self.assertEqual(len(dispatched), plan['planned_native_steps'])

    def test_jaw_checkpoint_respects_cross_arm_dependency_and_merges_explicit_marker(self):
        for explicit in (False, True):
            with self.subTest(explicit=explicit):
                adapter, _, _ = preview_adapter(); jaws(adapter)
                stages = [jaw(checkpoint_after_gripper=True)]
                if explicit:
                    stages.append(dict(id='inspect', kind='observe', arm='left', reason='check object'))
                stages.append(move('peer', 'right', depends_on=['close']))
                action = batch(stages)
                plan = adapter.preview_batch(action)
                self.assertEqual(len(plan['checkpoints']), 1)
                first = self.execute(adapter, action, plan)
                self.assertEqual(first['stage_results']['peer']['completed_steps'], 0)
                self.assertEqual(set(first['checkpoint_stage_ids']), {'inspect'} if explicit else {'close'})
                self.assertEqual(self.execute(adapter, action, plan)['status'], 'completed')

    def test_jaw_checkpoint_is_not_duplicated_by_explicit_same_arm_observe(self):
        adapter, _, _ = preview_adapter(); jaws(adapter)
        action = batch([jaw(checkpoint_after_gripper=True), dict(id='inspect', arm='left', kind='observe', reason='check object'),
            move('carry')])
        plan = adapter.preview_batch(action)
        self.assertEqual(len(plan['checkpoints']), 1)
        self.assertEqual(plan['checkpoints'][0]['stage_ids'], ['inspect'])
        self.assertNotIn('close', plan['checkpoints'][0]['reasons'])
        first = self.execute(adapter, action, plan)
        self.assertEqual(first['checkpoint_stage_ids'], ['inspect'])
        self.assertEqual(first['stage_results']['carry']['completed_steps'], 0)
        self.assertEqual(self.execute(adapter, action, plan)['status'], 'completed')

    def test_adjacent_jaw_changes_only_checkpoint_at_physical_boundary(self):
        adapter, _, _ = preview_adapter(); jaws(adapter)
        adapter.last_dispatched_goals['left'] = 1.
        action = batch([jaw(identifier='open', goal=.5), jaw(identifier='close', goal=0.), move('carry')],
            checkpoint_after_gripper=True)
        plan = adapter.preview_batch(action)
        self.assertEqual(len(plan['checkpoints']), 1)
        self.assertEqual(plan['checkpoints'][0]['stage_ids'], ['close'])
        self.assertEqual(self.execute(adapter, action, plan)['checkpoint_stage_ids'], ['close'])
        self.assertEqual(self.execute(adapter, action, plan)['status'], 'completed')

    def test_no_extra_checkpoint_for_default_final_jaw_unchanged_goal_independent_peer_or_stage_override(self):
        for stages, batch_flag in (
                ([jaw(), move('carry')], False),
                ([jaw()], True),
                ([jaw(goal=1.), move('carry')], True),
                ([jaw(), move('peer', 'right', -.5)], True),
                ([jaw(checkpoint_after_gripper=False), move('carry')], True)):
            with self.subTest(stages=stages, batch_flag=batch_flag):
                adapter, _, _ = preview_adapter(); jaws(adapter)
                action = batch(stages, **({'checkpoint_after_gripper': True} if batch_flag else {}))
                plan = adapter.preview_batch(action)
                self.assertEqual(plan['checkpoints'], [])
                self.assertEqual(self.execute(adapter, action, plan)['status'], 'completed')

    def test_multi_jaw_window_merges_into_single_checkpoint_before_shared_move(self):
        adapter, _, _ = preview_adapter(); jaws(adapter)
        adapter.last_dispatched_goals['left'] = 1.
        adapter.last_dispatched_goals['right'] = 1.
        action = batch([
            jaw(identifier='close_left', arm='left', goal=0.),
            jaw(identifier='close_right', arm='right', goal=.5),
            move('carry_left', arm='left', depends_on=['close_left', 'close_right']),
        ], checkpoint_after_gripper=True)
        plan = adapter.preview_batch(action)
        self.assertEqual(len(plan['checkpoints']), 1)
        self.assertEqual(set(plan['checkpoints'][0]['stage_ids']), {'close_left', 'close_right'})
        first = self.execute(adapter, action, plan)
        self.assertEqual((first['status'], first['reason']), ('stopped', 'observation_checkpoint'))
        self.assertEqual(first['stage_results']['carry_left']['completed_steps'], 0)
        self.assertNotIn('likely empty grasp', first['checkpoint_reasons']['close_left'])
        self.assertEqual(first['stage_results']['close_left']['grip_facts']['measured_aperture_m'], 0.0)
        self.assertEqual(self.execute(adapter, action, plan)['status'], 'completed')

    def test_batch_level_jaw_checkpoint_skips_opening_and_pauses_only_on_closing(self):
        adapter, _, _ = preview_adapter(); jaws(adapter)
        adapter.last_dispatched_goals['left'] = 0.2
        action = batch([
            jaw(identifier='open_left', arm='left', goal=1.0),
            move('reach', arm='left'),
            jaw(identifier='close_left', arm='left', goal=0.0),
            move('lift', arm='left', angle=-.25),
        ], checkpoint_after_gripper=True)
        plan = adapter.preview_batch(action)
        self.assertEqual(len(plan['checkpoints']), 1)
        self.assertEqual(plan['checkpoints'][0]['stage_ids'], ['close_left'])

    def test_jaw_checkpoint_keeps_resume_budget_and_drift_gates(self):
        for failure in ('budget', 'drift', 'obs_mismatch'):
            with self.subTest(failure=failure):
                adapter, _, dispatched = preview_adapter(); jaws(adapter)
                action = batch([jaw(checkpoint_after_gripper=True), move('carry')])
                plan = adapter.preview_batch(action)
                first = self.execute(adapter, action, plan)
                count = len(dispatched)
                self.assertEqual(first['reason'], 'observation_checkpoint')
                if failure == 'budget':
                    adapter.native_action_budget_limit = count
                elif failure == 'obs_mismatch':
                    adapter._batch_plans[plan['plan_id']]['expected_observation_id'] = 'obs-checkpoint'
                else:
                    original = adapter.state
                    def drift():
                        state = original(); state['arms']['left']['q_rad'][0] += .2
                        return state
                    adapter.state = drift
                second = self.execute(adapter, action, plan)
                self.assertEqual(second['reason'], 'batch_remaining_native_budget_insufficient' if failure == 'budget'
                    else 'batch_plan_stale_state_requires_new_preview')
                self.assertEqual(len(dispatched), count)

    def test_fault_during_jaw_keeps_unknown_and_never_reaches_dependent_motion(self):
        adapter, _, dispatched = preview_adapter(); jaws(adapter)
        action = batch([jaw(), move('carry')])
        plan = adapter.preview_batch(action)
        native = adapter.env.take_action
        def fault(command):
            native(command)
            raise RuntimeError('after partial jaw dispatch')
        adapter.env.take_action = fault
        with patch('sys.stderr', io.StringIO()):
            result = self.execute(adapter, action, plan)
        self.assertEqual(result['status'], 'unknown')
        self.assertTrue(result['stage_results']['close']['execution_uncertain'])
        self.assertEqual(result['stage_results']['carry']['completed_steps'], 0)
        self.assertEqual(self.execute(adapter, action, plan)['status'], 'rejected')
        self.assertEqual(len(dispatched), 1)

    def test_cancel_retains_executed_prefix_and_consumes_plan(self):
        adapter, _, dispatched = preview_adapter()
        action=batch([move('long',angle=-.5)]); plan=adapter.preview_batch(action)
        cancel=threading.Event()
        def progress(update):
            if update.get('executed_steps') == 1:
                cancel.set()
        result=adapter.execute(dict(action,batch_plan_id=plan['plan_id']),progress,cancel)
        self.assertEqual(result['reason'],'cancelled')
        self.assertEqual(result['executed_native_steps'],1)
        self.assertEqual(result['batch_cursor'],1)
        self.assertEqual(self.execute(adapter,action,plan)['status'],'rejected')

    def test_cached_commands_cannot_change(self):
        adapter, _, dispatched=preview_adapter()
        action=batch([move('one')]); plan=adapter.preview_batch(action)
        action['stages'][0]['action']['speed_scale']=.8
        result=self.execute(adapter,action,plan)
        self.assertEqual(result['reason'],'batch_plan_commands_changed')
        self.assertEqual(dispatched,[])

    def test_resuming_middle_of_peer_stage_readmits_whole_suffix_budget(self):
        adapter, _, dispatched=preview_adapter()
        adapter.native_action_budget_limit=100
        action=batch([move('one'),dict(id='look',kind='observe',arm='left',reason='check'),move('peer','right',-.5)])
        plan=adapter.preview_batch(action)
        self.execute(adapter,action,plan)
        adapter.native_action_budget_limit=4
        result=self.execute(adapter,action,plan)
        self.assertEqual(result['reason'],'batch_remaining_native_budget_insufficient')
        self.assertEqual(result['executed_native_steps'],0)
        self.assertEqual(len(dispatched),3)

    def test_resume_measured_first_step_is_checked_even_with_valid_pose_drift(self):
        adapter, _, dispatched=preview_adapter()
        action=batch([move('one'),dict(id='look',kind='observe',arm='left',reason='check'),move('peer','right',-.5)])
        plan=adapter.preview_batch(action)
        self.execute(adapter,action,plan)
        old=adapter.state
        def shifted():
            value=old(); value['arms']['right']['q_rad'][0]+=.001
            return value
        adapter.state=shifted
        result=self.execute(adapter,action,plan)
        self.assertEqual(result['reason'],'batch_resume_first_joint_step_exceeds_limit')
        self.assertEqual(result['executed_native_steps'],0)
        self.assertEqual(len(dispatched),3)

    def test_faulting_native_call_retains_billed_vs_completed_uncertainty(self):
        adapter, _, dispatched=preview_adapter()
        action=batch([move('one')]); plan=adapter.preview_batch(action)
        native=adapter.env.take_action
        def fault(command):
            native(command)
            raise RuntimeError('after partial native dispatch')
        adapter.env.take_action=fault
        with patch('sys.stderr',io.StringIO()):
            result=self.execute(adapter,action,plan)
        self.assertEqual(result['status'],'unknown')
        self.assertEqual(result['executed_native_steps'],1)
        self.assertEqual(result['completed_native_steps'],0)
        self.assertEqual(result['batch_cursor'],0)
        self.assertTrue(result['stage_results']['one']['execution_uncertain'])
        self.assertEqual(len(dispatched),1)


def _make_grasp_own(arm: str, grasp_pos: list[float], angles_deg: list[float], close: bool = False) -> dict:
    own = compile_action(
        'grasp',
        {
            'arm': arm,
            'observation_id': 'obs_1',
            'grasp_position_m': list(grasp_pos),
            'angles_deg': list(angles_deg),
        },
        {'run_id': 'r1', 'episode_id': 'e1', 'env_instance_id': 'i1', 'env_generation': 0},
        {'frame_id': 'env_origin', 'tcp_config_id': 'arx-x5-link6-v1'},
    )
    if close:
        own['gripper_events'] = [dict(arm=arm, phase='after_motion', goal_normalized=0.0)]
    return own


def _pose_3d(angle_xy: float, z_m: float) -> dict:
    return dict(
        position_m=[math.cos(angle_xy), math.sin(angle_xy), float(z_m)],
        quaternion_xyzw=[0.0, 0.0, math.sin(angle_xy / 2.0), math.cos(angle_xy / 2.0)],
    )


def _move_3d(identifier: str, *, arm: str, angle_xy: float = 0.0, z_m: float = 1.0, **values) -> dict:
    own = wait_action(timeout_s=30)
    del own['hold_native_steps']
    own['path'] = dict(kind='backend_free', waypoints=[{arm: _pose_3d(angle_xy, z_m)}])
    return dict(id=identifier, kind='move', arm=arm, action=own, **values)


def _dual_arm_floor_adapter(
    *,
    z_scale_m_per_rad: float = 0.10,
    left_q2_floor_rad: float | None = None,
    right_q2_floor_rad: float | None = None,
):
    adapter, planner, dispatched = preview_adapter()
    jaws(adapter)
    measured = {'left': [0.0] * 6, 'right': [0.0] * 6}

    def custom_plan_path(current, target, origin):
        planner.pose_calls.append((np.array(current, copy=True), deepcopy(target), origin))
        goal = np.array(current, copy=True)
        goal[0] = math.atan2(target[1], target[0])
        goal[2] = (float(target[2]) - 1.0) / z_scale_m_per_rad
        original = np.stack([current, goal])
        planner.originals.append(original.copy())
        return {'status': 'Success', 'position': original}

    planner.plan_path = custom_plan_path
    robot = adapter.robot.robot_list[0]

    def state():
        arms = {}
        for side in ('left', 'right'):
            q = list(measured[side])
            pose = _pose_3d(q[0], 1.0 + z_scale_m_per_rad * q[2])
            pose.update(source_kind='measurement', frame_id='env_origin', tcp_config_id='arx-x5-link6-v1')
            arms[side] = dict(
                q_rad=q,
                joint_names=list(robot.arm_joints_name),
                source_kind='measurement',
                ee_pose=pose,
                gripper=dict(
                    position_source_kind='measurement',
                    position_m=[adapter.last_dispatched_goals[side]],
                    joint_names=['jaw'],
                ),
            )
        return {'arms': arms}

    adapter.state = state
    raw_take_action = adapter.env.take_action
    floors = {'left': left_q2_floor_rad, 'right': right_q2_floor_rad}

    def take_action(action):
        raw_take_action(action)
        for side in ('left', 'right'):
            q_next = list(dispatched[-1][f'{side}_arm_joint_state'])
            fl = floors[side]
            if fl is not None and q_next[2] < fl:
                q_next[2] = float(fl)
            measured[side] = q_next

    adapter.env.take_action = take_action
    return adapter


class BatchSymmetryAndDualArmReceiptTests(unittest.TestCase):
    """覆盖双臂同行终态残差防覆盖、开爪对称/微倾角接触点不变量、闭爪刚性及步数预算隔离。"""

    def _execute(self, adapter, action: dict, plan: dict) -> dict:
        return adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda update: None, threading.Event())

    def test_dual_arm_same_row_residual_isolation_across_arms(self):
        for name, left_fl, right_fl, exp_stage, exp_arm in (
            ('left_fail_right_pass', -0.20, None, 'left_move', 'left'),
            ('left_pass_right_fail', None, -0.20, 'right_move', 'right'),
            ('both_fail_first_wins', -0.20, -0.60, 'left_move', 'left'),
        ):
            with self.subTest(case=name):
                adapter = _dual_arm_floor_adapter(left_q2_floor_rad=left_fl, right_q2_floor_rad=right_fl)
                action = batch([_move_3d('left_move', arm='left', z_m=0.880), _move_3d('right_move', arm='right', z_m=0.880)])
                plan = adapter.preview_batch(action)
                receipt = self._execute(adapter, action, plan)
                self.assertEqual(receipt['status'], 'failed')
                self.assertEqual(receipt['failed_stage_id'], exp_stage)
                self.assertEqual(receipt['failed_arm'], exp_arm)
                self.assertAlmostEqual(receipt['endpoint_residual_m']['position_error_m'], 0.100, places=3)
                self.assertEqual(receipt['contact_block_summary']['stage_id'], exp_stage)
                if name == 'left_fail_right_pass':
                    self.assertAlmostEqual(receipt['stage_results']['right_move']['endpoint_residual_m']['position_error_m'], 0.0, places=4)
                elif name == 'both_fail_first_wins':
                    self.assertEqual(receipt['stage_results']['right_move']['status'], 'failed')
                    self.assertAlmostEqual(receipt['stage_results']['right_move']['endpoint_residual_m']['position_error_m'], 0.060, places=3)

    def test_open_jaw_default_symmetry_contact_invariant_tilt_and_execute_receipt_angles(self):
        ctx = SimpleNamespace(
            lock=threading.RLock(),
            _unified_action_sequence=0,
            identity=dict(run_id='run-clean', episode_id='ep-clean', env_instance_id='env-clean', env_generation=1),
            capabilities=dict(
                backend_id='robodojo-sim',
                robot_id='arx-x5-bimanual-sim',
                frame_id='env_origin',
                tcp_config_id='arx-x5-link6-v1',
                action_modes=['absolute', 'backend_free'],
                arms=['left', 'right'],
            ),
            config=SimpleNamespace(features=SimpleNamespace(default_motion_speed_scale=1.0)),
        )
        compiled = compile_request(
            ctx,
            {
                'observation_id': 'obs_1',
                'stages': [
                    {'id': 'default_grasp', 'kind': 'move', 'arm': 'left', 'grasp_position_m': [0.22, 0.15, 0.80], 'angles_deg': [0.0, 0.0, 30.0]},
                    {'id': 'opt_out_grasp', 'kind': 'move', 'arm': 'left', 'grasp_position_m': [0.22, 0.10, 0.80], 'angles_deg': [0.0, 0.0, 30.0], 'allow_jaw_symmetry': False},
                    {'id': 'raw_flange', 'kind': 'move', 'arm': 'left', 'position_m': [0.22, 0.10, 0.95], 'quaternion_xyzw': [0.0, 0.0, 0.0, 1.0]},
                ],
            },
        )
        by_id = {st['id']: st for st in compiled['stages']}
        self.assertTrue(by_id['default_grasp']['allow_jaw_symmetry'])
        self.assertFalse(by_id['opt_out_grasp']['allow_jaw_symmetry'])
        self.assertNotIn('allow_jaw_symmetry', by_id['raw_flange'])

        adapter, planner, _ = preview_adapter()
        jaws(adapter)
        orig_angles = list(roboprobe_pose.quat_to_angles(np.asarray([0.0, 1.0, 0.0, 0.0], dtype=float)))
        base_own = motion_action('left', (0.0,))
        base_own['path']['waypoints'][0]['left']['quaternion_xyzw'] = [1.0, 0.0, 0.0, 0.0]
        base_own['input_grasp_pose'] = dict(
            position_m=[1.1501, 0.0, 1.0],
            angles_deg=orig_angles,
            convention='roboprobe_top_down_world_xyz_degrees',
            offset_m=0.1501,
        )
        tilted_cands = _open_jaw_tilted_grasp_candidates(base_own, 'left', base_own['path']['waypoints'][0]['left'])
        self.assertEqual(len(tilted_cands), 4)
        for cand_own, _cand_target, cand_angles in tilted_cands:
            wp = cand_own['path']['waypoints'][0]['left']
            q_xyzw = np.asarray(wp['quaternion_xyzw'], dtype=float)
            rot = roboprobe_pose.quat_to_matrix(np.array([q_xyzw[3], q_xyzw[0], q_xyzw[1], q_xyzw[2]], dtype=float))
            np.testing.assert_allclose(np.asarray(wp['position_m'], dtype=float) + 0.1501 * rot[:, 0], [1.1501, 0.0, 1.0], atol=1e-6)
            self.assertGreaterEqual(float(wp['position_m'][2]), 1.0 - 0.01)
            self.assertEqual(len(cand_angles), 3)

        orig_plan_path = planner.plan_path

        def plan_only_tilted(current, target, origin):
            if np.max(np.abs(np.asarray(target[3:], dtype=float))) > 0.999:
                return {'status': 'Fail'}
            return orig_plan_path(current, target, origin)

        planner.plan_path = plan_only_tilted
        # Default allow_approach_tilt=False: allow_jaw_symmetry=True tries strictly the exact 180-deg twin,
        # never silently tilts or caches a plan, and surfaces feasible_approach_tilt_angles_deg + failure_hint
        cached_before = set(getattr(adapter, '_batch_plans', {}).keys())
        act_no_tilt = batch([dict(id='g_no_tilt', kind='move', arm='left', allow_jaw_symmetry=True, action=deepcopy(base_own))])
        prev_no_tilt = adapter.preview_batch(act_no_tilt)
        self.assertEqual(prev_no_tilt['status'], 'failed')
        self.assertEqual(prev_no_tilt['reason'], 'path_infeasible')
        self.assertIs(prev_no_tilt['physics_advanced'], False)
        self.assertEqual(prev_no_tilt['executed_native_steps'], 0)
        self.assertEqual(set(getattr(adapter, '_batch_plans', {}).keys()), cached_before)
        self.assertEqual(
            prev_no_tilt['failure_feedback']['requested_grasp_pose']['angles_deg'],
            orig_angles,
        )
        self.assertIn('feasible_approach_tilt_angles_deg', prev_no_tilt)
        feasible_tilts = prev_no_tilt['feasible_approach_tilt_angles_deg']
        self.assertGreaterEqual(len(feasible_tilts), 1)
        self.assertIn('allow_approach_tilt=true', prev_no_tilt['failure_hint'])
        self.assertIs(prev_no_tilt['failure_feedback']['allow_approach_tilt'], False)
        self.assertEqual(prev_no_tilt['failure_feedback']['feasible_approach_tilt_angles_deg'], feasible_tilts)
        self.assertIn('allow_approach_tilt=true', prev_no_tilt['failure_feedback']['suggested_adjustments'][0])

        # Explicit opt-in allow_approach_tilt=True evaluates tilted approach candidates and stamps approach_tilt_resolved=True
        act_tilt = batch([dict(id='g_tilt', kind='move', arm='left', allow_jaw_symmetry=True, allow_approach_tilt=True, action=deepcopy(base_own))])
        prev_tilt = adapter.preview_batch(act_tilt)
        self.assertEqual(prev_tilt['status'], 'planned')
        self.assertTrue(prev_tilt['stage_costs']['g_tilt']['jaw_symmetry_resolved'])
        self.assertTrue(prev_tilt['stage_costs']['g_tilt']['approach_tilt_resolved'])
        resolved_ang = prev_tilt['stage_costs']['g_tilt']['resolved_angles_deg']
        self.assertEqual(resolved_ang, feasible_tilts[0])
        planned_target = deepcopy(adapter._batch_plans[prev_tilt['plan_id']]['stages']['g_tilt']['target'])
        prev_state_fn = adapter.state

        def state_at_tilted_target():
            st = prev_state_fn()
            if adapter.env.take_action_cnt[0] > 0:
                st['arms']['left']['ee_pose'].update(
                    position_m=list(planned_target['position_m']),
                    quaternion_xyzw=list(planned_target['quaternion_xyzw']),
                )
            return st

        adapter.state = state_at_tilted_target
        exec_tilt = self._execute(adapter, act_tilt, prev_tilt)
        self.assertEqual(exec_tilt['batch_status'], 'completed', exec_tilt)
        self.assertTrue(exec_tilt['stage_results']['g_tilt'].get('approach_tilt_resolved'))
        self.assertEqual(exec_tilt['stage_results']['g_tilt'].get('resolved_angles_deg'), resolved_ang)
        adapter.state = prev_state_fn

        # Passing the first hinted explicit angles_deg directly with allow_approach_tilt=False also resolves path_infeasible
        explicit_tilt_own = _make_grasp_own('left', [1.1501, 0.0, 1.0], feasible_tilts[0])
        prev_explicit = adapter.preview_batch(
            batch([dict(id='g_explicit', kind='move', arm='left', allow_jaw_symmetry=False, action=explicit_tilt_own)])
        )
        self.assertEqual(prev_explicit['status'], 'planned', prev_explicit)

    def test_open_jaw_path_infeasible_approach_tilt_diagnostic_hint_and_hygiene(self):
        adapter, planner, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals['right'] = 1.0
        orig_plan_path = planner.plan_path
        horiz_own = _make_grasp_own('right', [0.24, -0.14, 0.84], [90.0, 0.0, 0.0])

        def reject_horizontal_approach(current, target, origin):
            p_deg, _, _ = roboprobe_pose.quat_to_angles(np.asarray(target[3:], dtype=float))
            if abs(abs(float(p_deg)) - 90.0) < 5.0:
                return {'status': 'Fail'}
            return orig_plan_path(current, target, origin)

        planner.plan_path = reject_horizontal_approach
        horiz_fail = adapter.preview_batch(
            batch([dict(id='holder_grasp', kind='move', arm='right', allow_jaw_symmetry=True, action=deepcopy(horiz_own))])
        )
        self.assertEqual((horiz_fail['status'], horiz_fail['reason']), ('failed', 'path_infeasible'))
        self.assertIn([75.0, 0.0, 0.0], horiz_fail.get('feasible_approach_tilt_angles_deg', []))
        horiz_ok = adapter.preview_batch(
            batch([dict(id='holder_grasp', kind='move', arm='right', allow_jaw_symmetry=True, allow_approach_tilt=True, action=deepcopy(horiz_own))])
        )
        self.assertEqual(horiz_ok['status'], 'planned')
        self.assertEqual(horiz_ok['stage_costs']['holder_grasp']['resolved_angles_deg'], [75.0, 0.0, 0.0])

        # Negative hygiene: when all +/-15/+/-30 deg tilts also fail, or when jaw is closed, omit feasible_approach_tilt_angles_deg
        planner.plan_path = lambda current, target, origin: {'status': 'Fail'}
        all_infeasible = adapter.preview_batch(
            batch([dict(id='unreachable', kind='move', arm='right', allow_jaw_symmetry=True, action=deepcopy(horiz_own))])
        )
        self.assertEqual((all_infeasible['status'], all_infeasible['reason']), ('failed', 'path_infeasible'))
        self.assertNotIn('feasible_approach_tilt_angles_deg', all_infeasible)
        self.assertNotIn('failure_hint', all_infeasible)

        planner.plan_path = reject_horizontal_approach
        adapter.last_dispatched_goals['right'] = 0.0
        closed_jaw_fail = adapter.preview_batch(
            batch([dict(id='closed_move', kind='move', arm='right', allow_jaw_symmetry=True, action=deepcopy(horiz_own))])
        )
        self.assertEqual((closed_jaw_fail['status'], closed_jaw_fail['reason']), ('failed', 'path_infeasible'))
        self.assertNotIn('feasible_approach_tilt_angles_deg', closed_jaw_fail)

    def test_closed_jaw_rigidity_and_budget_exhaustion_isolation(self):
        adapter, planner, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals['left'] = 1.0
        approach_own = _make_grasp_own('left', [0.18, 0.16, 0.88], [0.0, 0.0, 180.0])
        close_own = deepcopy(approach_own)
        close_own.update(path=dict(kind='hold', waypoints=[]), gripper_events=[dict(arm='left', phase='after_motion', goal_normalized=0.0)])
        carry_own = _make_grasp_own('left', [0.18, 0.16, 0.92], [35.0, 0.0, 90.0])
        orig_plan_path = planner.plan_path

        def reject_only_yaw_180(current, target, origin):
            _, _, yaw_deg = roboprobe_pose.quat_to_angles(np.asarray(target[3:], dtype=float))
            if abs(abs(float(yaw_deg)) - 180.0) < 20.0:
                return {'status': 'Fail'}
            return orig_plan_path(current, target, origin)

        planner.plan_path = reject_only_yaw_180
        plan = adapter.preview_batch(
            batch(
                [
                    dict(id='approach', kind='move', arm='left', allow_jaw_symmetry=True, action=deepcopy(approach_own)),
                    dict(id='close', kind='gripper', arm='left', action=close_own),
                    dict(id='carry', kind='move', arm='left', allow_jaw_symmetry=True, action=deepcopy(carry_own)),
                ]
            )
        )
        self.assertEqual(plan['status'], 'planned', plan)
        self.assertEqual(plan['stage_costs']['approach']['resolved_angles_deg'], [0.0, 0.0, 0.0])
        self.assertNotIn('jaw_symmetry_resolved', plan['stage_costs']['carry'])
        np.testing.assert_allclose(
            adapter._batch_plans[plan['plan_id']]['stages']['carry']['target']['quaternion_xyzw'],
            carry_own['path']['waypoints'][0]['left']['quaternion_xyzw'],
            atol=1e-12,
        )

        # 步数预算耗尽时仅规划主姿态 1 次，绝不触发倾角/对称搜索篡改姿态
        adapter2, planner2, _ = preview_adapter()
        jaws(adapter2)
        adapter2.last_dispatched_goals['right'] = 1.0
        adapter2.motion_safety_guard_enabled = True
        adapter2.native_action_budget_limit = 100
        adapter2.native_actions_dispatched = 98
        grasp_own = _make_grasp_own('right', [0.18, -0.15, 0.85], [0.0, 0.0, 0.0])
        plan_calls = []
        orig2 = planner2.plan_path

        def spy_plan_path(current, target, origin):
            p_deg, r_deg, y_deg = roboprobe_pose.quat_to_angles(np.asarray(target[3:], dtype=float))
            plan_calls.append((round(float(p_deg), 1), round(float(r_deg), 1), round(float(y_deg), 1)))
            return orig2(current, target, origin)

        planner2.plan_path = spy_plan_path
        budget_res = adapter2.preview_batch(batch([dict(id='pick', kind='move', arm='right', allow_jaw_symmetry=True, action=deepcopy(grasp_own))]))
        self.assertEqual(len(plan_calls), 1, plan_calls)
        self.assertEqual((budget_res['status'], budget_res['reason']), ('failed', 'motion_safety_rejected'))

    def test_batch_scheduler_helpers_checkpoint_reason_codes_and_right_arm_resume_layout(self):
        from ni_rua.robot.motion.batch_scheduler import (
            BatchScheduler,
            MotionControllerPort,
            StageScheduleItem,
            order_stages,
            pause_receipt,
            skip_dependents,
            summarize_batch,
        )

        adapter, _, _ = preview_adapter()
        jaws(adapter)
        self.assertIsInstance(adapter, MotionControllerPort)
        scheduler = BatchScheduler(adapter)
        self.assertEqual(scheduler.arm_layout.arms, ('left', 'right'))
        item = StageScheduleItem(id='s1', kind='move', arm='left', start=0, end=3)
        self.assertEqual((item.id, item.end), ('s1', 3))

        ordered = order_stages([
            dict(id='b', kind='move', arm='right', depends_on=['a']),
            dict(id='a', kind='move', arm='left'),
        ])
        self.assertEqual([s['id'] for s in ordered], ['a', 'b'])
        stage_res = {'a': {'status': 'failed'}, 'b': {'status': 'pending'}, 'c': {'status': 'pending'}}
        skipped = skip_dependents(
            'a',
            [dict(id='a', kind='move', arm='left'), dict(id='b', kind='move', arm='right', depends_on=['a']),
             dict(id='c', kind='move', arm='left')],
            stage_results=stage_res,
            status='skipped',
        )
        self.assertEqual(skipped, ['b', 'c'])
        self.assertEqual(stage_res['b']['status'], 'skipped')
        self.assertEqual(stage_res['c']['status'], 'skipped')

        paused_dict = pause_receipt(['obs1'], {'obs1': 'inspect'}, {'confirmed': True})
        self.assertEqual(paused_dict['checkpoint_reason_code'], 'observation_checkpoint')
        self.assertNotIn('checkpoint_reason', paused_dict)

        summary = summarize_batch(
            'plan-1',
            {'status': 'executing', 'completed': {'a'}, 'cursor': 3, 'revision': 1},
            ['a', 'b'],
            {'a': {'status': 'completed'}, 'b': {'status': 'running'}},
            {},
            complete_batch_native_steps=6,
        )
        self.assertEqual(summary['batch_status'], 'failed')
        self.assertEqual(summary['completed_stage_ids'], ['a'])
        self.assertEqual(summary['pending_stage_ids'], ['b'])

        # Verify typed checkpoint_reason_code across observe, jaw_checkpoint, and right-arm ArmLayout resume
        adapter.last_dispatched_goals['right'] = 1.0
        act_chk = batch([
            jaw('r_close', arm='right', goal=0.0, checkpoint_after_gripper=True),
            move('r_lift', arm='right', angle=-0.125),
            dict(id='r_obs', arm='right', kind='observe', reason='verify right lift'),
            move('r_place', arm='right', angle=-0.25),
        ])
        plan_chk = scheduler.preview(act_chk)
        self.assertEqual(plan_chk['status'], 'planned')
        info_before = scheduler.inspect(plan_chk['plan_id'])
        self.assertEqual(info_before['status'], 'planned')

        first = self._execute(adapter, act_chk, plan_chk)
        self.assertEqual((first['status'], first['checkpoint_reason_code']), ('stopped', 'jaw_checkpoint'))
        self.assertNotIn('checkpoint_reason', first)

        second = self._execute(adapter, act_chk, plan_chk)
        self.assertEqual((second['status'], second['checkpoint_reason_code']), ('stopped', 'observation_checkpoint'))
        self.assertNotIn('checkpoint_reason', second)
        # Both are stops the model asked for: the reply shaping treats their receipts as routine (typed code only).
        from ni_rua.core.presentation.feedback_delivery import is_requested_checkpoint_receipt
        from ni_rua.core.presentation.reply_compaction import reports_exception
        for requested in (first, second):
            self.assertTrue(is_requested_checkpoint_receipt(requested), requested)
            self.assertFalse(reports_exception(dict(receipt=requested)))

        # Right-arm drift exceeding step speed limit triggers ArmLayout right-arm step jump limit check
        adapter.resume_drift_tolerance_rad = 1.5
        cache = adapter._batch_plans[plan_chk['plan_id']]
        self.assertEqual(scheduler.arm_layout.offset('right'), 7)
        cache['paused_state']['arms']['right']['q_rad'][0] += 0.9
        third = self._execute(adapter, act_chk, plan_chk)
        self.assertEqual((third['status'], third['reason']), ('failed', 'batch_resume_first_joint_step_exceeds_limit'))
        self.assertEqual(third['failed_stage_id'], 'r_place')

    def test_carry_slip_final_row_enforces_endpoint_on_resume_and_completes_dual_arm_peers(self):
        # 1) Single-arm final-row carry-slip pause with dependent gripper stage:
        #    (a) endpoint within tolerance on resume -> completes predecessor & runs dependent gripper
        #    (b) endpoint out of tolerance on resume -> fails predecessor with tracking_error before gripper starts
        for name, left_floor, expect_ok in (
            ('within_tolerance_completes_all', None, True),
            ('out_of_tolerance_blocks_successor', -0.20, False),
        ):
            with self.subTest(case=name):
                adapter = _dual_arm_floor_adapter(left_q2_floor_rad=left_floor)
                adapter.last_dispatched_goals['left'] = 1.0
                act = batch([
                    jaw('close', arm='left', goal=0.0, grip_guard_preset='rigid_object', carry_slip_guard=True),
                    _move_3d('lift', arm='left', z_m=0.880),
                    jaw('release', arm='left', goal=1.0),
                ])
                plan = adapter.preview_batch(act)
                self.assertEqual(plan['status'], 'planned', plan)
                lift_end = plan['stage_costs']['lift']['end_native_step']
                lift_steps = plan['stage_costs']['lift']['native_steps']
                base_state = adapter.state
                slip_on_final_row = True

                def state_with_final_row_slip():
                    st = base_state()
                    step = adapter.env.take_action_cnt[0]
                    if 1 <= step < lift_end:
                        st['arms']['left']['gripper']['position_m'] = [0.30]
                    elif step == lift_end and slip_on_final_row:
                        st['arms']['left']['gripper']['position_m'] = [0.0]
                    elif step >= lift_end and not slip_on_final_row:
                        st['arms']['left']['gripper']['position_m'] = [adapter.last_dispatched_goals['left']]
                    return st

                adapter.state = state_with_final_row_slip
                first = self._execute(adapter, act, plan)
                self.assertEqual(first['status'], 'stopped', first)
                self.assertEqual(first['checkpoint_stage_id'], 'lift')
                self.assertEqual(first['checkpoint_reason_code'], 'carry_slip_aperture_below_min_m')
                # Same status and reason as a requested checkpoint, but a guard's code: an exception for reply shaping.
                from ni_rua.core.presentation.feedback_delivery import is_requested_checkpoint_receipt
                from ni_rua.core.presentation.reply_compaction import reports_exception
                self.assertEqual((first['reason'], first['paused_at_checkpoint']), ('observation_checkpoint', True))
                self.assertFalse(is_requested_checkpoint_receipt(first))
                self.assertTrue(reports_exception(dict(receipt=first)))
                self.assertEqual(first['slip_detected_at']['step_index'], lift_steps - 1)
                self.assertEqual(first['batch_cursor'], lift_end)
                self.assertEqual(first['completed_stage_ids'], ['close'])
                self.assertEqual(first['pending_stage_ids'], ['lift', 'release'])
                self.assertEqual(first['stage_results']['release']['status'], 'pending')
                self.assertEqual(first['stage_results']['release']['completed_steps'], 0)

                second = self._execute(adapter, act, plan)
                if expect_ok:
                    self.assertEqual(second['status'], 'completed', second)
                    self.assertEqual(second['batch_status'], 'completed')
                    self.assertEqual(second['completed_stage_ids'], ['close', 'lift', 'release'])
                    self.assertEqual(second['pending_stage_ids'], [])
                    self.assertEqual(second['stage_results']['lift']['status'], 'completed')
                    self.assertEqual(second['stage_results']['release']['status'], 'completed')
                else:
                    self.assertEqual(second['status'], 'failed', second)
                    self.assertEqual(second['reason'], 'tracking_error')
                    self.assertEqual(second['failed_stage_id'], 'lift')
                    self.assertEqual(second['failed_arm'], 'left')
                    self.assertEqual(second['executed_native_steps'], 0)
                    self.assertEqual(second['stage_results']['lift']['status'], 'failed')
                    self.assertEqual(second['stage_results']['release']['status'], 'pending')
                    self.assertEqual(second['stage_results']['release']['completed_steps'], 0)
                    self.assertIn('lift', second['pending_stage_ids'])

        # 2) Dual-arm: non-triggering peer stage ending on the exact same row is validated & completed immediately
        for peer_floor, expect_peer_ok in ((None, True), (-0.20, False)):
            with self.subTest(peer_ok=expect_peer_ok):
                adapter = _dual_arm_floor_adapter(right_q2_floor_rad=peer_floor)
                adapter.last_dispatched_goals['left'] = 1.0
                act_dual = batch([
                    jaw('close', arm='left', goal=0.0, grip_guard_preset='rigid_object', carry_slip_guard=True),
                    _move_3d('left_lift', arm='left', z_m=0.880),
                    _move_3d('right_place', arm='right', z_m=0.880, depends_on=['close']),
                    jaw('release', arm='left', goal=1.0, depends_on=['left_lift', 'right_place']),
                ])
                plan_dual = adapter.preview_batch(act_dual)
                self.assertEqual(plan_dual['status'], 'planned', plan_dual)
                l_end = plan_dual['stage_costs']['left_lift']['end_native_step']
                r_end = plan_dual['stage_costs']['right_place']['end_native_step']
                self.assertEqual(l_end, r_end)
                base_dual_state = adapter.state

                def dual_state():
                    st = base_dual_state()
                    step = adapter.env.take_action_cnt[0]
                    if 1 <= step < l_end:
                        st['arms']['left']['gripper']['position_m'] = [0.30]
                    elif step == l_end:
                        st['arms']['left']['gripper']['position_m'] = [0.0]
                    return st

                adapter.state = dual_state
                first_dual = self._execute(adapter, act_dual, plan_dual)
                if not expect_peer_ok:
                    self.assertEqual(first_dual['status'], 'failed', first_dual)
                    self.assertEqual(first_dual['reason'], 'tracking_error')
                    self.assertEqual(first_dual['failed_stage_id'], 'right_place')
                    self.assertEqual(first_dual['failed_arm'], 'right')
                else:
                    self.assertEqual(first_dual['status'], 'stopped', first_dual)
                    self.assertEqual(first_dual['checkpoint_stage_id'], 'left_lift')
                    self.assertEqual(first_dual['completed_stage_ids'], ['close', 'right_place'])
                    self.assertEqual(first_dual['pending_stage_ids'], ['left_lift', 'release'])
                    self.assertEqual(first_dual['stage_results']['right_place']['status'], 'completed')

                    second_dual = self._execute(adapter, act_dual, plan_dual)
                    self.assertEqual(second_dual['status'], 'completed', second_dual)
                    self.assertEqual(
                        second_dual['completed_stage_ids'],
                        ['close', 'left_lift', 'right_place', 'release'],
                    )
                    self.assertEqual(second_dual['pending_stage_ids'], [])

    def test_carry_slip_until_release_persists_across_pause_and_resume_until_release(self):
        adapter, _, _ = preview_adapter()
        jaws(adapter)
        adapter.last_dispatched_goals['left'] = 1.0
        act = batch([
            jaw(
                'close',
                arm='left',
                goal=0.0,
                grip_guard_preset='rigid_object',
                carry_slip_guard=True,
                carry_slip_scope='until_release',
                slip_consecutive_steps=2,
            ),
            move('lift', arm='left', angle=-0.25),
            move('carry', arm='left', angle=-0.50),
            jaw('release', arm='left', goal=0.8),
            move('retreat', arm='left', angle=-0.625),
        ])
        plan = adapter.preview_batch(act)
        self.assertEqual(plan['status'], 'planned', plan)
        close_end = plan['stage_costs']['close']['end_native_step']
        lift_end = plan['stage_costs']['lift']['end_native_step']
        carry_end = plan['stage_costs']['carry']['end_native_step']
        base_state = adapter.state
        phase = {'run': 1}

        def scripted_aperture_state():
            st = base_state()
            step = adapter.env.take_action_cnt[0]
            if phase['run'] == 1:
                # 2-step slip early in lift triggers 1st pause
                ap = 0.0 if step in {close_end + 1, close_end + 2} else 0.30
            elif phase['run'] == 2:
                # Rest of lift recovers (0.30), then 2-step slip during carry triggers 2nd pause!
                ap = 0.0 if step in {lift_end + 1, lift_end + 2} else 0.30
            else:
                # Rest of carry holds 0.30, then after carry_end release disarms so 0.0 in retreat is ignored
                ap = 0.30 if step <= carry_end else 0.0
            if step >= 1:
                st['arms']['left']['gripper']['position_m'] = [ap]
            return st

        adapter.state = scripted_aperture_state
        cache = adapter._batch_plans[plan['plan_id']]

        # Run 1: pauses during lift; monitor stays armed with reset streak
        r1 = self._execute(adapter, act, plan)
        self.assertEqual((r1['status'], r1['checkpoint_stage_id']), ('stopped', 'lift'), r1)
        self.assertEqual(r1['checkpoint_reason_code'], 'carry_slip_aperture_below_min_m')
        self.assertIn('left', cache['active_carry_slip_monitors'])
        self.assertEqual(cache['active_carry_slip_monitors']['left'].scope, 'until_release')

        # Adjust paused jaw reading so resume validation matches recovered 0.30 aperture
        cache['paused_measured_jaws']['left'] = 0.30
        phase['run'] = 2
        r2 = self._execute(adapter, act, plan)
        self.assertEqual((r2['status'], r2['checkpoint_stage_id']), ('stopped', 'carry'), r2)
        self.assertEqual(r2['checkpoint_reason_code'], 'carry_slip_aperture_below_min_m')
        self.assertEqual(r2['slip_detected_at']['consecutive_steps'], 2)
        self.assertIn('lift', r2['completed_stage_ids'])

        cache['paused_measured_jaws']['left'] = 0.30
        phase['run'] = 3
        r3 = self._execute(adapter, act, plan)
        self.assertEqual(r3['status'], 'completed', r3)
        self.assertEqual(r3['completed_stage_ids'], ['close', 'lift', 'carry', 'release', 'retreat'])
        self.assertEqual(r3['pending_stage_ids'], [])


# === tests/test_batch_motion_review.py ===

from copy import deepcopy
import threading
import unittest
from unittest.mock import patch

import numpy as np

from support.motion_fixtures import batch as _review_batch, jaws as _review_jaws, move as _review_move
from support.motion_fixtures import motion_action, periodic_pose
from support.motion_fixtures import preview_adapter


class BatchMotionReviewTests(unittest.TestCase):
    def test_normal_jaw_checkpoint_repeats_cached_goal_when_measurement_is_blocked(self):
        adapter, _, dispatched = preview_adapter()
        _review_jaws(adapter)
        original = adapter.state

        def measured_state():
            value = original()
            value['arms']['right']['gripper']['position_m'] = [.5]
            return value

        adapter.state = measured_state
        close = motion_action('right')
        close.update(path=dict(kind='hold', waypoints=[]),
            gripper_events=[dict(arm='right', phase='after_motion', goal_normalized=0.)])
        action = _review_batch([_review_move('left', angle=-.04),
            dict(id='look', arm='left', kind='observe', reason='inspect grasp'),
            dict(id='jaw', arm='right', kind='gripper', action=close)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        rows = deepcopy(adapter._batch_plans[plan['plan_id']]['rows'])
        self.assertEqual(len(rows), 2)
        request = dict(action, batch_plan_id=plan['plan_id'])
        with (patch.object(adapter, '_plan_segment', side_effect=AssertionError('must not replan')),
                patch.object(adapter, '_resample_action_plan', side_effect=AssertionError('must not resample'))):
            first = adapter.execute(request, lambda _: None, threading.Event())
            self.assertEqual((first['status'], first['reason']), ('stopped', 'observation_checkpoint'))
            self.assertEqual(first['batch_cursor'], 1)
            second = adapter.execute(request, lambda _: None, threading.Event())
        self.assertEqual(second['status'], 'completed')
        self.assertEqual(second['resume_validation']['status'], 'valid')
        self.assertEqual(second['executed_native_steps'], 1)
        actual = [row['left_arm_joint_state'] + row['left_ee_joint_state'] +
                  row['right_arm_joint_state'] + row['right_ee_joint_state'] for row in dispatched]
        np.testing.assert_array_equal(actual, rows)
        self.assertEqual(adapter.state()['arms']['right']['gripper']['position_m'], [.5])
        self.assertEqual(dispatched[-1]['right_ee_joint_state'], [0.])

    def test_zero_cursor_checkpoint_rechecks_first_step_from_measured_joints(self):
        adapter, _, dispatched = preview_adapter()
        action = _review_batch([dict(id='look', kind='observe', reason='inspect before movement'),
                        _review_move('left', angle=-.05)])
        plan = adapter.preview_batch(action)
        request = dict(action, batch_plan_id=plan['plan_id'])
        first = adapter.execute(request, lambda _: None, threading.Event())
        self.assertEqual((first['status'], first['reason']), ('stopped', 'observation_checkpoint'))
        self.assertEqual(first['batch_cursor'], 0)
        original = adapter.state

        def drifted_state():
            value = original()
            left = value['arms']['left']
            left['q_rad'][0] += .0015
            left['ee_pose'].update(periodic_pose(left['q_rad']))
            return value

        adapter.state = drifted_state
        second = adapter.execute(request, lambda _: None, threading.Event())
        self.assertEqual(second['resume_validation']['status'], 'valid')
        self.assertEqual(second['reason'], 'batch_resume_first_joint_step_exceeds_limit')
        self.assertEqual(second['executed_native_steps'], 0)
        self.assertEqual(dispatched, [])

    def test_during_motion_move_retains_closed_command_when_object_blocks_jaws(self):
        adapter, _, dispatched = preview_adapter()
        _review_jaws(adapter)
        adapter.last_dispatched_goals['left'] = 0.
        original = adapter.state

        def blocked_state():
            value = original()
            value['arms']['left']['gripper']['position_m'] = [.48]
            return value

        adapter.state = blocked_state
        carry = _review_move('carry_and_release', angle=-.2)
        carry['action'].update(gripper_events=[dict(
            arm='left', phase='during_motion', goal_normalized=1., start_fraction=.5, end_fraction=1.)])
        action = _review_batch([carry])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        result = adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda _: None, threading.Event())
        self.assertEqual((result['status'], result['reason']), ('completed', None))
        jaw_goals = [row['left_ee_joint_state'][0] for row in dispatched]
        self.assertEqual(jaw_goals[0], 0.)
        self.assertEqual(jaw_goals[-1], 1.)

    def test_opening_jaw_advances_across_preceding_move_before_slow_close(self):
        adapter, _, dispatched = preview_adapter()
        _review_jaws(adapter)
        adapter.last_dispatched_goals['left'] = 1.
        original = adapter.state

        def opening_state():
            value = original()
            value['arms']['left']['gripper']['position_m'] = [.998 if adapter.env.take_action_cnt[0] else .9075]
            return value

        adapter.state = opening_state
        close = motion_action('left')
        close.update(path=dict(kind='hold', waypoints=[]), gripper_speed_scale=.5,
            gripper_events=[dict(arm='left', phase='after_motion', goal_normalized=0.)])
        action = _review_batch([_review_move('approach', angle=-.125),
                        dict(id='close', arm='left', kind='gripper', action=close)])
        plan = adapter.preview_batch(action)
        self.assertEqual(plan['status'], 'planned')
        result = adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda _: None, threading.Event())
        self.assertEqual((result['status'], result['reason']), ('completed', None))
        self.assertAlmostEqual(dispatched[3]['left_ee_joint_state'][0], 1. - 1. / 12)
        self.assertEqual(dispatched[-1]['left_ee_joint_state'], [0.])

    def test_checkpoint_resume_tolerates_stationary_tracking_lag_and_blocked_jaw(self):
        adapter, _, dispatched = preview_adapter()
        _review_jaws(adapter)
        adapter.last_dispatched_goals['left'] = 0.
        original = adapter.state

        def lagged_state():
            value = original()
            left = value['arms']['left']
            if adapter.env.take_action_cnt[0]:
                left['q_rad'][0] += .003
                left['ee_pose'].update(periodic_pose(left['q_rad']))
            left['gripper']['position_m'] = [.1852]
            return value

        adapter.state = lagged_state
        release = _review_move('release', angle=-.25)
        release['action'].update(gripper_speed_scale=.8, gripper_events=[dict(
            arm='left', phase='during_motion', goal_normalized=1., start_fraction=0., end_fraction=1.)])
        action = _review_batch([_review_move('one', angle=-.125),
                        dict(id='look', kind='observe', reason='check before release'),
                        release])
        plan = adapter.preview_batch(action)
        request = dict(action, batch_plan_id=plan['plan_id'])
        first = adapter.execute(request, lambda _: None, threading.Event())
        self.assertEqual((first['status'], first['reason']), ('stopped', 'observation_checkpoint'))
        second = adapter.execute(request, lambda _: None, threading.Event())
        self.assertEqual((second['status'], second['reason']), ('completed', None))
        self.assertEqual(len(dispatched), plan['planned_native_steps'])

    def test_segment_start_admits_joint_error_within_orientation_bound(self):
        adapter, _, _ = preview_adapter()
        lag = [0.]
        original = adapter.state

        def lagged_state():
            value = original()
            if adapter.env.take_action_cnt[0]:
                value['arms']['left']['q_rad'][5] += lag[0]
            return value

        adapter.state = lagged_state
        close = motion_action('left')
        close.update(path=dict(kind='hold', waypoints=[]),
            gripper_events=[dict(arm='left', phase='after_motion', goal_normalized=0.)])
        action = _review_batch([_review_move('approach', angle=-.125),
                        dict(id='close', arm='left', kind='gripper', action=close)])
        plan = adapter.preview_batch(action)
        lag[0] = .075
        completed = adapter.execute(dict(action, batch_plan_id=plan['plan_id']), lambda _: None, threading.Event())
        self.assertEqual((completed['status'], completed['reason']), ('completed', None))


if __name__ == "__main__":
    unittest.main()
