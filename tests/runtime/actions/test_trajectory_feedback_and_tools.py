"""
- test_contact_overlay.py: Contact overlays use calibrated measured geometry, without object inference.
- test_trajectory_feedback.py: Automatic action feedback is actor-only and cannot execute a robot action.
- test_trajectory_tools.py: Candidate commands stay bounded, observation-bound and deployable.
"""
from copy import deepcopy
import base64
import json
from pathlib import Path
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock
import cv2
import numpy as np
from ni_rua.perception.overlays.contact_overlay import render_contact_overlay
from ni_rua.core.presentation.actor_output import ActorOutput
from ni_rua.core.presentation.actor_presentation import ActorPresentation
from ni_rua.core.storage.images import image_settings
from ni_rua.runtime.actions.inspection import contact_overlay
from ni_rua.core.storage.store import Store
from ni_rua.core.tool_output import ToolResponse, wire_response
import io
from unittest.mock import patch
from PIL import Image
from ni_rua.core.contracts import canonical
from ni_rua.robot.motion.trajectory_evidence import build_trajectory_evidence, compact_trajectory_evidence
from ni_rua.runtime.actions.trajectory_feedback import build_action_trajectory_feedback
from ni_rua.runtime.actions.inspection import inspect_trajectory
from ni_rua.perception.overlays.pose_axis_overlay import draw_pose_axis_frame
from ni_rua.perception.overlays.trajectory_overlay import deployable_camera, render_trajectory_overlay
from support.motion_fixtures import camera_fixture
from support.motion_fixtures import DualArmHarnessBase

NI_RUA_TEST_TIER = "component"


# --- Folded from tests/test_contact_overlay.py ---

def _contact_camera_fixture():
    step, domain = 30, 'adapter_control_step'
    observation = dict(world_kind='observed', observation_id='observed', sim_step=step, step_domain=domain)
    frame = dict(camera_id='head', frame_id='head-frame', sim_step=step, step_domain=domain,
                 rgb_ref='original-actor-rgb', pixel_transform='identity')
    state = dict(sim_step=step, step_domain=domain, arms={})
    for side, x in (('left', .1), ('right', -.4)):
        state['arms'][side] = dict(ee_pose=dict(position_m=[x, -.05, 2.], quaternion_xyzw=[0.,0.,0.,1.],
            source_kind='measurement', frame_id='env_origin', tcp_config_id='arx-x5-link6-v1'))
    camera = dict(observation=observation, frame=frame, state=state, width=640, height=480,
                  K=np.array([[500.,0,320.],[0,500.,240.],[0,0,1.]]), D=np.zeros(5), T=np.eye(4),
                  reference_frame_id='env_origin', pose_source='fixed_calibration')
    image = np.zeros((480,640,3), dtype=np.uint8)
    image[:,:,0] = np.arange(640, dtype=np.uint16)[None,:] % 256
    return camera, image


class ContactOverlayTests(unittest.TestCase):
    def test_all_actual_reference_positions_and_pixels_match_x5_calibration(self):
        camera, image = _contact_camera_fixture()
        output, result = render_contact_overlay(camera, image, arms=['left'],
            selections={'left': {'contact_region': 'finger_middle'}})
        refs = result['arms']['left']['actual_references']
        for region, offset in (('pad_center', .1501), ('finger_middle', .1226), ('fingertip', .1576)):
            expected = [.1+offset, -.05, 2.]
            np.testing.assert_allclose(refs[region]['position_m'], expected, atol=1e-12)
            point = result['points'][refs[region]['point_index']]
            np.testing.assert_allclose(point['pixel_xy'], [320+250*(.1+offset), 227.5], atol=1e-10)
            self.assertEqual(point['source_kind'], 'derived_from_measured_link6')
            self.assertTrue(point['in_image'])
            self.assertEqual(point['occlusion'], 'not_evaluated')
        np.testing.assert_allclose(result['arms']['left']['actual_selected_position_m'], [.2226,-.05,2.])
        self.assertEqual(result['original_rgb_ref'], 'original-actor-rgb')
        self.assertEqual(result['observation_id'], 'observed')
        self.assertEqual(output.shape, image.shape)
        self.assertTrue(np.any(output != image))
        self.assertFalse(result['physics_advanced'])
        self.assertEqual((result['executed_native_steps'],result['executed_control_steps']), (0,0))
        json.dumps(result, allow_nan=False)

    def test_rotation_and_camera_transform_use_world_axis_and_optical_convention(self):
        camera, image = _contact_camera_fixture()
        camera['state']['arms']['left']['ee_pose']['quaternion_xyzw'] = [0.,0.,2**-.5,2**-.5]
        angle = .2
        camera['T'][:3,:3] = [[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]]
        camera['T'][:3,3] = [.05,.02,.1]
        _, result = render_contact_overlay(camera, image, arms=['left'],
                                           selections={'left': {'contact_depth_m': .01}})
        own = result['arms']['left']
        np.testing.assert_allclose(own['actual_selected_position_m'], [.1,.0901,2.], atol=1e-12)
        np.testing.assert_allclose(own['actual_approach_axis_world'], [0,1,0], atol=1e-12)
        local = np.linalg.inv(camera['T']) @ np.r_[own['actual_selected_position_m'],1.]
        expected = camera['K'] @ (local[:3]/local[2])
        np.testing.assert_allclose(result['points'][own['actual_selected_point_index']]['pixel_xy'], expected[:2])
        arrow = result['arrows'][0]
        difference = np.array(result['points'][arrow['end']]['position_m'])-result['points'][arrow['start']]['position_m']
        np.testing.assert_allclose(difference, [0,.06,0], atol=1e-12)

    def test_requested_target_and_approach_are_separate_from_actual_measurements(self):
        camera, image = _contact_camera_fixture()
        _, result = render_contact_overlay(camera, image, arms=['right'],
            selections={'right': {'contact_region':'fingertip'}},
            targets={'right': {'position_m':[.3,.1,2.], 'approach_axis_world':[0,-2,0]}})
        own = result['arms']['right']
        np.testing.assert_allclose(own['actual_selected_position_m'], [-.2424,-.05,2.])
        self.assertEqual(own['requested_selected_position_m'], [.3,.1,2.])
        self.assertEqual(own['requested_approach_axis_world'], [0.,-1.,0.])
        point = result['points'][own['requested_point_index']]
        self.assertEqual(point['source_kind'], 'caller_request')
        self.assertIn('REQUESTED', point['label'])
        self.assertEqual(point['marker'], 'cross')
        self.assertIn('not a measured object', own['target_semantics'])
        self.assertIn('does not infer objects', result['limitation'])
        self.assertEqual([arrow['kind'] for arrow in result['arrows']], ['actual_approach','requested_approach'])

    def test_original_image_and_input_records_are_unchanged(self):
        camera, image = _contact_camera_fixture()
        before_image, before_camera = image.copy(), deepcopy(camera)
        selection={'left':{'contact_region':'finger_middle'}}
        target={'left':{'position_m':[.2,.1,2.]}}
        before_selection, before_target = deepcopy(selection), deepcopy(target)
        output, _ = render_contact_overlay(camera,image,selections=selection,targets=target)
        np.testing.assert_array_equal(image,before_image)
        self.assertFalse(np.shares_memory(output,image))
        for key in ('state','observation','frame'):
            self.assertEqual(camera[key],before_camera[key])
        for key in ('K','D','T'):
            np.testing.assert_array_equal(camera[key],before_camera[key])
        self.assertEqual(selection,before_selection)
        self.assertEqual(target,before_target)
        np.testing.assert_array_equal(output[:50,:50],image[:50,:50])

    def test_public_hook_returns_decodable_image_and_actor_scoped_provenance(self):
        camera,image=_contact_camera_fixture()
        identity=dict(run_id='overlay',episode_id='ep',env_instance_id='env',env_generation=1)
        with tempfile.TemporaryDirectory() as directory:
            store=Store(Path(directory)/'store',identity)
            try:
                ok,png=cv2.imencode('.png',image)
                self.assertTrue(ok)
                original=png.tobytes()
                reference=store.put_artifact(original,'rgb','image/png','actor')
                camera['frame']['rgb_ref']=reference
                harness=SimpleNamespace(lock=threading.RLock(),identity=identity,store=store,
                    geometry=SimpleNamespace(camera=Mock(return_value=camera)),
                    observation_profile='rgb_proprio_calibrated_v1',image_settings=image_settings(),
                    latest=dict(observation_id='observed'))
                # No robot/backend exists on this fixture: any motion access fails.
                content=contact_overlay(harness,dict(observation_id='observed',camera_id='head',arms=['left']),{},
                    structured=True)
                response=wire_response(ToolResponse(content))['contentItems']
                metadata=json.loads(response[0]['text'])
                self.assertTrue(metadata['current'])
                self.assertEqual(metadata['original_rgb_ref'],reference)
                self.assertEqual(response[1]['type'],'inputImage')
                uri=response[1]['imageUrl']
                self.assertTrue(uri.startswith('data:image/'))
                decoded=cv2.imdecode(np.frombuffer(base64.b64decode(uri.split(',',1)[1]),np.uint8),cv2.IMREAD_COLOR)
                self.assertEqual(decoded.shape,image.shape)
                self.assertEqual(store.resolve_image(reference,allowed_scopes=('actor',))[1],original)
                store.resolve_image(metadata['overlay_ref'],allowed_scopes=('actor',))
                actor=ActorOutput(store,ActorPresentation(store))
                shown=actor.present(ToolResponse(content))['contentItems']
                shown_metadata=json.loads(shown[0]['text'])
                for field in ('points','arrows','approach_arrow_length_m'):
                    self.assertEqual(shown_metadata[field],metadata[field])
                self.assertEqual(shown[1],response[1])
                detail=json.loads(store.resolve_artifact(actor.restore(shown_metadata['details_ref']))[1])
                self.assertEqual(detail['points'],metadata['points'])
                self.assertEqual(store.records('action_request'),[])
            finally:
                store.close()

    def test_behind_outside_and_zero_depth_are_not_drawn_as_visible(self):
        for position in ([.1,0.,-1.], [100.,0.,1.], [.1,0.,0.]):
            camera,image = _contact_camera_fixture()
            camera['state']['arms']['left']['ee_pose']['position_m']=position
            with self.subTest(position=position):
                _, result=render_contact_overlay(camera,image,arms=['left'])
                self.assertTrue(all(not point['in_image'] for point in result['points']))
                self.assertTrue(all(not arrow['drawn'] for arrow in result['arrows']))
                json.dumps(result,allow_nan=False)

    def test_internal_unsynchronized_predicted_or_wrong_tcp_data_is_rejected(self):
        mutations = [lambda c: c['observation'].update(world_kind='internal'),
            lambda c:c['state'].update(sim_step=29), lambda c:c['frame'].update(sim_step=29),
            lambda c:c['frame'].update(pixel_transform='cropped'),
            lambda c:c['state']['arms']['left']['ee_pose'].update(source_kind='prediction'),
            lambda c:c['state']['arms']['left']['ee_pose'].update(tcp_config_id='other_tcp'),
            lambda c:c['state']['arms']['left']['ee_pose'].update(frame_id='other_frame'),
            lambda c:c['state']['arms']['left']['ee_pose'].update(quaternion_xyzw=[0,0,0,2])]
        for index, mutate in enumerate(mutations):
            camera,image=_contact_camera_fixture()
            mutate(camera)
            with self.subTest(index=index),self.assertRaises(ValueError):
                render_contact_overlay(camera,image,arms=['left'])

    def test_bad_dimensions_arms_selections_and_targets_fail_explicitly(self):
        camera,image=_contact_camera_fixture()
        for args in (dict(arms=[]),dict(arms=['left','left']),dict(arms=['other']),
                     dict(selections={'left':{'contact_region':'middle'}}),
                     dict(selections={'left':{'contact_depth_m':True}}),
                     dict(selections={'left':{'contact_depth_m':.1}}),
                     dict(selections={'left':{'unexpected':1}}),
                     dict(targets={'left':{'position_m':[True,0,1]}}),
                     dict(targets={'left':{'position_m':[float('nan'),0,1]}}),
                     dict(targets={'left':{'position_m':[0,0,1],'approach_axis_world':[0,0,0]}}),
                     dict(targets={'left':{'position_m':[0,0,1],'angles_deg':[0,0,0]}})):
            with self.subTest(args=args),self.assertRaises(ValueError):
                render_contact_overlay(camera,image,**args)
        with self.assertRaisesRegex(ValueError,'size_mismatch'):
            render_contact_overlay(camera,image[:240])



# --- Folded from tests/test_trajectory_feedback.py ---

def hardware():
    joints = []
    for i in range(1, 7):
        t = np.eye(4)
        t[0, 3] = 0.1
        joints.append(dict(
            name=f'joint{i}',
            type='revolute',
            parent='base_link' if i == 1 else f'link{i-1}',
            child=f'link{i}',
            axis=[0, 0, 1],
            T_parent_joint=t.tolist(),
        ))
    for i in (7, 8):
        joints.append(dict(
            name=f'joint{i}',
            type='prismatic',
            parent='link6',
            child=f'link{i}',
            axis=[0, 1 if i == 7 else -1, 0],
            T_parent_joint=np.eye(4).tolist(),
        ))
    return dict(
        model=dict(base_link='base_link', joints=joints, source_sha256='public-fixture'),
        spheres={f'link{i}': [dict(center=[0, 0, 0], radius=0.03)] for i in range(1, 7)},
        sphere_buffer_m=0,
        jaws=dict(
            gripper_scale=[-0.01, 0.044],
            gripper_move=dict(base='joint7', sign=1, mimic=['joint8', 1.0, 0.0]),
            gripper_joints_name=['joint7', 'joint8'],
        ),
        provenance={'kind': 'public-hardware-fixture'},
    )


def public_fixture(separation=3.0):
    h = hardware()
    state = {'arms': {}}
    for arm, base in [('left', 0), ('right', separation)]:
        state['arms'][arm] = dict(
            source_kind='measurement',
            joint_names=[f'joint{i}' for i in range(1, 7)],
            q_rad=[0.0] * 6,
            ee_pose=dict(
                frame_id='env_origin',
                tcp_config_id='arx-x5-link6-v1',
                source_kind='measurement',
                position_m=[base + 0.6, 0, 0],
                quaternion_xyzw=[0, 0, 0, 1],
            ),
            gripper=dict(joint_names=['joint7', 'joint8'], position_source_kind='measurement', position_m=[0.044, 0.044]),
        )
    adapter = SimpleNamespace(control_hz=25, last_dispatched_goals={'left': 1.0, 'right': 1.0})
    row = [0.0] * 6 + [1.0] + [0.0] * 6 + [1.0]
    compiled = {a: (np.zeros((2, 6)), [list(row), list(row)], [], None) for a in ('left', 'right')}
    action = dict(frame_id='env_origin', tcp_config_id='arx-x5-link6-v1')
    return adapter, state, compiled, action, h


class ActionTrajectoryFeedbackTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.identity = dict(run_id='run', episode_id='episode', env_instance_id='environment', env_generation=1)
        self.store = Store(Path(self.temporary.name), self.identity)
        self.addCleanup(self.store.close)
        _, self.before, _, self.action, hardware = public_fixture(.5)
        self.action.update(self.identity, action_id='action')
        self.before.update(sim_step=30, step_domain='adapter_control_step')
        self.harness = SimpleNamespace(store=self.store, identity=self.identity, latest={'observation_id':'terminal'},
            observation_profile='rgb_proprio_calibrated_v1', image_settings=image_settings(),
            config=SimpleNamespace(features=SimpleNamespace(action_trajectory_delivery='diagnostic')))
        self.model_ref = self.put(hardware['model'], 'robot_model')
        camera = np.eye(4); camera[2,3] = -2
        calibration = dict(reference_frame_id='env_origin', cameras={'cam_head':dict(image_size=[640,480],
            K=[[500.,0,320.],[0,500.,240.],[0,0,1.]], distortion=[0.]*5,T_world_camera=camera.tolist())})
        calibration_ref = self.put(calibration, 'calibration')
        buffer = io.BytesIO(); Image.new('RGB',(640,480),(80,120,150)).save(buffer,format='PNG')
        self.rgb_ref = self.store.put_artifact(buffer.getvalue(),'rgb','image/png')
        self.observation = dict(self.identity, observation_id='start', world_kind='observed', sim_step=30,
            step_domain='adapter_control_step', robot_model_ref=self.model_ref,
            robot_state_ref=self.put(self.before,'robot_state'), cameras=[dict(camera_id='cam_head',
                frame_id='head', width=640,height=480,sim_step=30,step_domain='adapter_control_step',
                calibration_ref=calibration_ref, rgb_ref=self.rgb_ref,pixel_transform='identity',pose_sample_ref='FORBIDDEN')])
        self.store.save('observation','start',self.observation)

    def put(self, value, kind, scope='actor'):
        return self.store.put_artifact(canonical(value).encode(),kind,'application/json',scope)

    def receipt(self, plan, counts=2, measured_error=0.):
        actual=[]
        for index in range(counts+1):
            state=deepcopy(self.before); state['sim_step']=30+index*10
            for arm in ('left','right'):
                state['arms'][arm]['q_rad']=[index*.1+(measured_error if index else 0.)]*6
            actual.append(dict(sim_step=state['sim_step'],state=state))
        actual.append(deepcopy(actual[-1]))
        return dict(status='completed', executed_steps=counts, completed_native_steps=counts,
            execution_start_observation_id='start', result_observation_id='terminal',
            trace_sampling=dict(actual_every_native_actions=1, actual_boundary_samples=True),
            planned_trace_ref=self.put(plan,'planned_trace'), actual_trace_ref=self.put(actual,'actual_trace'),
            before_state_ref=self.put(self.before,'before_state'), dispatched_trace_ref=self.put({'samples':[]},'dispatched_trace'))

    @staticmethod
    def rows(count):
        return [[i*.1]*6+[1.]+[i*.1]*6+[1.] for i in range(1,count+1)]

    def test_diagnostic_paired_trace_original_resolution_images_exact_q_and_no_backend(self):
        receipt=self.receipt(dict(actions=self.rows(2),completed_native_steps=2), measured_error=.01)
        original=deepcopy(receipt)
        result,images=build_action_trajectory_feedback(self.harness,self.action,receipt,{},diagnostic=True)
        self.assertEqual(receipt,original)
        self.assertTrue(result['available'],result)
        self.assertEqual(len(images),2)
        self.assertEqual(result['measured']['total_samples'],3)
        self.assertEqual(result['planned']['samples'][-1]['arms']['right']['q_rad'],[.2]*6)
        self.assertAlmostEqual(result['measured_command_error']['max_abs_joint_error_rad'],.01)
        self.assertEqual(result['joint_names'],{arm:self.before['arms'][arm]['joint_names'] for arm in ('left','right')})
        self.assertEqual(result['q_unit'],'rad')
        self.assertNotIn('joint_names',result['measured'])
        self.assertNotIn('geometry',result['planned'])
        self.assertNotIn('measured_jaw',result['measured']['samples'][0]['arms']['left'])
        self.assertEqual(result['final_measured_jaws']['arms']['left']['position_m'],[.044,.044])
        full=json.loads(self.store.resolve_artifact(result['trajectory_feedback_ref'],('actor',))[1])
        for content,metadata in zip(images,full['summary']['overlays'].values()):
            with Image.open(io.BytesIO(base64.b64decode(content['imageUrl'].split(',',1)[1]))) as image:
                self.assertEqual(image.size,(640,480))
            self.assertEqual(metadata['original_rgb_ref'],self.rgb_ref)
            self.assertEqual(metadata['observation_id'],'start')
            self.assertFalse(metadata['current'])
            self.assertFalse(metadata['physics_advanced'])
            self.assertEqual(metadata['path_sample_counts'],{'left':3,'right':3})
        self.assertEqual(full['paths']['measured']['samples'][-1]['sim_step'],50)
        self.assertEqual(result['overlays']['measured']['sample_indices'],[0,1,2])
        self.assertEqual(result['overlay_camera']['panel_width'],640)
        self.assertNotIn('panels',result['overlays']['measured'])
        self.assertNotIn('FORBIDDEN',canonical(full))

    def test_routine_completed_execution_has_no_overlay_render_or_q_delivery_and_preserves_exact_paths(self):
        receipt=self.receipt(dict(actions=self.rows(2),completed_native_steps=2), measured_error=.01)
        receipt.update(position_error_m=.001,orientation_error_rad=.02)
        original=deepcopy(receipt)
        with patch('ni_rua.runtime.actions.inspection._trajectory_image',side_effect=AssertionError('routine rendering')) as render:
            result,images=build_action_trajectory_feedback(self.harness,self.action,receipt,{})
        render.assert_not_called()
        self.assertEqual(images,[])
        self.assertEqual(receipt,original)
        self.assertEqual(result['delivery'],'routine_execution_summary')
        self.assertEqual(result['execution_status'],'completed')
        from ni_rua.core.presentation.actor_payload import lean_action_result
        # Routine feedback accompanies the enclosing receipt view;
        # removing its duplicate echo must leave exact endpoint evidence there.
        direct = lean_action_result(dict(request=self.action, receipt=receipt))['receipt']
        self.assertEqual(direct['position_error_m'], receipt['position_error_m'])
        self.assertEqual(direct['orientation_error_rad'], receipt['orientation_error_rad'])
        self.assertEqual(direct['completed_native_steps'], receipt['completed_native_steps'])
        self.assertAlmostEqual(result['measured_command_error']['max_abs_joint_error_rad'],.01)
        self.assertNotIn('q_rad',canonical(result))
        self.assertNotIn('samples',result)
        self.assertNotIn('measured',result)
        self.assertIn('task success are not assessed',result['semantics'])
        metadata,raw=self.store.resolve_artifact(result['trajectory_feedback_ref'],('actor',))
        self.assertEqual(metadata['kind'],'action_trajectory_feedback')
        full=json.loads(raw)
        self.assertEqual(full['source_refs']['planned_trace_ref'],receipt['planned_trace_ref'])
        self.assertEqual(full['paths']['planned']['samples'][-1]['arms']['right']['q_rad'],[.2]*6)
        self.assertEqual(full['paths']['measured']['samples'][-1]['arms']['left']['q_rad'],[.2+.01]*6)
        self.assertIn('link6_pose',full['paths']['measured']['samples'][-1]['arms']['left'])
        self.assertEqual(full['summary']['overlays'],{})

    def test_uncertain_or_exceptional_completed_and_unknown_execution_retain_diagnostics(self):
        cases=[dict(status='unknown'),dict(status='stopped'),dict(reason='tracking_error'),
            dict(execution_uncertain=True),dict(stop_confirmed=False),dict(stop_confirmation=dict(confirmed=False)),
            dict(gripper_execution=dict(left=dict(execution_uncertain=True))),
            dict(result_observation_id=None),dict(observation_unavailable_reason='camera_unavailable')]
        for updates in cases:
            with self.subTest(updates=updates):
                receipt=self.receipt(dict(actions=self.rows(2),completed_native_steps=2))
                receipt.update(updates)
                result,images=build_action_trajectory_feedback(self.harness,self.action,receipt,{})
                self.assertEqual(len(images),2)
                self.assertIn('q_rad',canonical(result))
                self.assertNotIn('delivery',result)

    def test_independent_failure_keeps_unsynchronized_unexecuted_local_arm_path(self):
        plan=dict(actions=self.rows(1),completed_native_steps=1,row_contexts=[{
            'left':{'phase':'motion'},'right':{'phase':'gripper'}}], arms={
            'left':dict(segments=[dict(compiled_sim_step=30,waypoint_index=0,
                motion_actions=[row[:7] for row in self.rows(4)],gripper_actions=[],completed_motion_steps=1)]),
            'right':dict(segments=[])})
        receipt=self.receipt(plan,counts=1);receipt.update(status='failed',reason='tracking_error')
        result,images=build_action_trajectory_feedback(self.harness,self.action,receipt,{})
        self.assertTrue(result['available'],result)
        self.assertEqual(len(images),3)
        local=result['unexecuted_candidates'][0]
        self.assertEqual(local['remaining_rows'],3)
        self.assertEqual(local['completed_rows'],1)
        self.assertIn('no shared future',local['timing'])
        self.assertEqual(set(local['samples'][-1]['arms']),{'left'})
        self.assertEqual(local['samples'][-1]['arms']['left']['q_rad'],[.4]*6)
        self.assertEqual(result['planned']['total_samples'],2)

    def test_single_arm_compiled_plan_and_measured_other_arm_hold(self):
        plan=dict(arm='left',joint_names=self.before['arms']['left']['joint_names'],
            q_rad=[[.1]*6,[.2]*6,[.3]*6],planned_motion_steps=3,planned_gripper_steps=2,completed_native_steps=1)
        result,_=build_action_trajectory_feedback(self.harness,self.action,self.receipt(plan,counts=1),{})
        self.assertTrue(result['available'])
        self.assertEqual(result['planned']['samples'][-1]['arms']['right']['q_rad'],[0.]*6)
        self.assertEqual(result['planned']['samples'][-1]['execution_state'],'not_recorded_completed')
        self.assertTrue(any('jaw rows' in item for item in result['limitations']))

    def test_zero_step_rejection_reports_missing_plan_and_available_measured_start(self):
        receipt=self.receipt({'segments':[]},counts=0);receipt.update(status='failed',reason='motion_safety_rejected')
        result,images=build_action_trajectory_feedback(self.harness,self.action,receipt,{})
        self.assertTrue(result['available'])
        self.assertFalse(result['planned']['available'])
        self.assertEqual(result['measured']['total_samples'],1)
        self.assertEqual(result['unexecuted_candidates'],[])
        self.assertEqual(len(images),1)

    def test_rejected_solver_endpoint_remains_endpoint_without_fabricated_path(self):
        receipt=self.receipt({'segments':[]},counts=0)
        receipt['failure_feedback']=dict(arm='left',candidates=[dict(stage='ik',
            joint_names=self.before['arms']['left']['joint_names'],endpoint_q_rad=[.42]*6,solver_success=False)])
        result,images=build_action_trajectory_feedback(self.harness,self.action,receipt,{})
        candidate=result['unexecuted_candidates'][0]
        self.assertEqual(candidate['total_samples'],1)
        self.assertEqual(candidate['samples'][0]['execution_state'],'not_executed')
        self.assertIn('endpoint only',candidate['timing'])
        self.assertEqual(len(images),2)

    def test_scope_identity_and_missing_calibration_fail_locally_without_receipt_mutation(self):
        receipt=self.receipt(dict(actions=self.rows(2),completed_native_steps=2))
        receipt['before_state_ref']=self.put(self.before,'before_state','internal')
        original=deepcopy(receipt)
        result,images=build_action_trajectory_feedback(self.harness,self.action,receipt,{})
        self.assertFalse(result['available']);self.assertIn('access_scope_denied',result['detail'])
        self.assertEqual(images,[]);self.assertEqual(receipt,original)
        result,_=build_action_trajectory_feedback(self.harness,{**self.action,'episode_id':'stale'},receipt,{})
        self.assertEqual(result['detail'],'stale_environment_identity')
        self.observation['robot_model_ref']=None
        # Immutable observation gets a different ID for the missing-model case.
        self.observation['observation_id']='missing'
        self.store.save('observation','missing',self.observation)
        receipt=self.receipt(dict(actions=self.rows(2),completed_native_steps=2))
        receipt['execution_start_observation_id']='missing'
        result,images=build_action_trajectory_feedback(self.harness,self.action,receipt,{})
        self.assertTrue(result['available'])
        self.assertEqual(result['geometry']['status'],'unavailable')
        self.assertEqual(result['measured']['samples'][-1]['arms']['left']['q_rad'],[.2]*6)
        full_missing = json.loads(self.store.resolve_artifact(result['trajectory_feedback_ref'], ('actor',))[1])
        for arm_state in full_missing['paths']['measured']['samples'][-1]['arms'].values():
            self.assertNotIn('link6_pose', arm_state)
        self.assertEqual(images,[])
        # Empty sample['arms'] on FK fallback must not raise UnboundLocalError
        with patch('ni_rua.runtime.actions.trajectory_feedback._planned_samples',
                   return_value=([dict(native_step=0, arms={})], [], [])):
            empty_arms_res, _ = build_action_trajectory_feedback(self.harness, self.action, receipt, {})
        self.assertTrue(empty_arms_res['available'])

    def test_unknown_sampling_does_not_invent_plan_measurement_alignment(self):
        receipt=self.receipt(dict(actions=self.rows(2),completed_native_steps=2))
        receipt.pop('trace_sampling')
        result,images=build_action_trajectory_feedback(self.harness,self.action,receipt,{})
        self.assertTrue(result['available'])
        self.assertFalse(result['native_boundary_alignment'])
        self.assertFalse(result['planned']['available'])
        self.assertNotIn('measured_command_error',result)

    def test_direct_measurements_and_derivatives_use_irregular_physics_timestamps(self):
        for state in self.before['arms'].values():
            state['qdot_rad_s'] = [1.] * 6
            state['gripper'].update(goal_normalized=.25, goal_source_kind='command',
                                    command_normalized=.5, command_source_kind='command')
        observation = deepcopy(self.observation)
        observation['observation_id'] = 'timed'
        observation['cameras'][0]['control_dt_s'] = .004
        self.store.save('observation', 'timed', observation)
        receipt = self.receipt(dict(actions=self.rows(2), completed_native_steps=2))
        actual = []
        for index, (step, velocity) in enumerate(((30, 1.), (40, 3.), (44, 4.))):
            state = deepcopy(self.before)
            state['sim_step'] = step
            for arm in ('left', 'right'):
                state['arms'][arm]['qdot_rad_s'] = [velocity] * 6
                state['arms'][arm]['ee_pose']['position_m'][0] += .004 * index
                angle = .02 * index
                state['arms'][arm]['ee_pose']['quaternion_xyzw'] = [0., 0., np.sin(angle / 2), np.cos(angle / 2)]
                state['arms'][arm]['gripper']['position_m'] = [.044 - .001 * index] * 2
            actual.append(dict(sim_step=step, state=state))
        receipt.update(execution_start_observation_id='timed', actual_trace_ref=self.put(actual, 'actual_trace'))
        result, _ = build_action_trajectory_feedback(self.harness, self.action, receipt, {}, diagnostic=True)
        self.assertTrue(result['available'], result)
        full = json.loads(self.store.resolve_artifact(result['trajectory_feedback_ref'], ('actor',))[1])
        measured = full['paths']['measured']
        self.assertEqual(measured['measurement_timing']['physics_dt_s'], .004)
        samples = measured['samples']
        self.assertEqual([item['elapsed_time_s'] for item in samples], [0., .04, .056])
        final = samples[-1]['arms']['left']
        self.assertEqual(final['qdot_rad_s'], [4.] * 6)
        self.assertEqual(final['measured_ee_pose'], actual[-1]['state']['arms']['left']['ee_pose'])
        self.assertNotEqual(final['measured_ee_pose']['position_m'], final['link6_pose']['position_m'])
        self.assertEqual(final['jaw_goal_normalized'], .25)
        self.assertEqual(final['jaw_command_normalized'], .5)
        derived = final['derived_motion']
        self.assertEqual(derived['interval_sim_steps'], [40, 44])
        self.assertEqual(derived['interval_s'], .016)
        np.testing.assert_allclose(derived['joint_acceleration_rad_s2'], [62.5] * 6)
        np.testing.assert_allclose(derived['ee_linear_velocity_m_s'], [.25, 0., 0.])
        np.testing.assert_allclose(derived['ee_angular_velocity_rad_s'], [0., 0., 1.25])
        np.testing.assert_allclose(derived['ee_linear_acceleration_m_s2'], [(.25 - .1) / .028, 0., 0.])
        np.testing.assert_allclose(derived['jaw_velocity_m_s'], [-.0625] * 2)

    def test_missing_clock_keeps_measured_velocity_and_pose_without_invented_derivatives(self):
        for state in self.before['arms'].values():
            state['qdot_rad_s'] = [.125] * 6
        result, _ = build_action_trajectory_feedback(self.harness, self.action,
            self.receipt(dict(actions=self.rows(2), completed_native_steps=2)), {}, diagnostic=True)
        full = json.loads(self.store.resolve_artifact(result['trajectory_feedback_ref'], ('actor',))[1])
        measured = full['paths']['measured']
        self.assertEqual(measured['measurement_timing']['status'], 'unavailable')
        for sample in measured['samples']:
            self.assertNotIn('elapsed_time_s', sample)
            self.assertEqual(sample['arms']['left']['qdot_rad_s'], [.125] * 6)
            self.assertIn('measured_ee_pose', sample['arms']['left'])
            self.assertNotIn('derived_motion', sample['arms']['left'])

    def test_default_without_config_uses_automatic_trajectory_delivery(self):
        """未显式配置 action_trajectory_delivery 时默认采用黄金基线 automatic，正常回执也附带实测轨迹叠图。"""
        harness_no_cfg = SimpleNamespace(
            store=self.store,
            identity=self.identity,
            latest={'observation_id': 'terminal'},
            observation_profile='rgb_proprio_calibrated_v1',
            image_settings=image_settings(),
        )
        receipt = self.receipt(dict(actions=self.rows(2), completed_native_steps=2), measured_error=0.0)
        result, images = build_action_trajectory_feedback(harness_no_cfg, self.action, receipt, {}, diagnostic=False)
        self.assertTrue(result['available'], result)
        self.assertEqual(result['delivery'], 'automatic_sampled_joints_and_image')
        self.assertEqual(len(images), 1)


# ======================================================================
# 公开轨迹几何证据测试 (迁自 test_trajectory_evidence.py)
# ======================================================================


class TrajectoryEvidenceTests(unittest.TestCase):
    def test_public_fk_bases_and_model_near_approach_have_witness_not_collision_claim(self):
        for separation, expected in [(3.0, 'no_near_approach_at_sampled_boundaries'), (0.5, 'model_near_approach')]:
            adapter, state, compiled, action, h = public_fixture(separation)
            with patch('ni_rua.robot.motion.trajectory_evidence._hardware', return_value=h):
                short, full = build_trajectory_evidence(adapter, state, compiled, 'paired', action)
            self.assertEqual(short['geometry']['status'], 'available')
            self.assertEqual(short['inter_arm_proximity']['status'], expected)
            self.assertTrue(short['inter_arm_proximity']['missing_finger_coverage'])
            self.assertAlmostEqual(full['T_world_bases']['right'][0][3], separation)
            self.assertAlmostEqual(full['samples'][0]['arms']['left']['link_positions_m']['link6'][0], 0.6)
            self.assertIn('not actual collision', short['inter_arm_proximity']['semantics'])
            self.assertEqual(len(full['samples']), 3)

    def test_missing_relative_calibration_returns_unknown_without_fallback(self):
        adapter, state, compiled, action, h = public_fixture()
        state['arms']['right']['ee_pose']['frame_id'] = 'right_local_unregistered'
        with patch('ni_rua.robot.motion.trajectory_evidence._hardware', return_value=h):
            short, full = build_trajectory_evidence(adapter, state, compiled, 'paired', action)
        self.assertEqual(short['geometry']['status'], 'unknown')
        self.assertEqual(short['inter_arm_proximity']['status'], 'unknown')
        self.assertNotIn('link_positions_m', full['samples'][0]['arms']['left'])
        self.assertNotIn('link6_pose', full['samples'][0]['arms']['left'])
        self.assertEqual(short['anomalies'][0]['kind'], 'public_geometry_unknown')

    def test_missing_spheres_preserves_fk_but_proximity_unknown(self):
        adapter, state, compiled, action, h = public_fixture()
        h['spheres'] = None
        with patch('ni_rua.robot.motion.trajectory_evidence._hardware', return_value=h):
            short, full = build_trajectory_evidence(adapter, state, compiled, 'paired', action)
        self.assertEqual(short['geometry']['status'], 'available')
        self.assertEqual(short['inter_arm_proximity']['status'], 'unknown')
        self.assertIn('link_positions_m', full['samples'][-1]['arms']['left'])
        self.assertEqual(short['anomalies'][0]['kind'], 'inter_arm_proximity_unknown')

    def test_compact_evidence_bounds_dense_near_path_and_preserves_exact_artifact(self):
        adapter, state, compiled, action, h = public_fixture(0.5)
        for arm in compiled:
            positions, motion, jaws, schedule = compiled[arm]
            compiled[arm] = (positions, motion * 40, jaws, schedule)
        with patch('ni_rua.robot.motion.trajectory_evidence._hardware', return_value=h):
            short, full = build_trajectory_evidence(adapter, state, compiled, 'paired', action)
        original = deepcopy(full)
        self.assertEqual(short, compact_trajectory_evidence(full))
        self.assertEqual(full, original)
        self.assertLessEqual(len(short['key_poses']), 6)
        self.assertEqual(short['q_unit'], 'rad')
        self.assertEqual(short['joint_names'], full['joint_names'])
        self.assertEqual(len(full['samples']), 81)
        self.assertEqual(short['inter_arm_proximity']['near_boundary_count'], 81)
        self.assertEqual(full['inter_arm_proximity']['near_native_steps'], list(range(81)))
        self.assertEqual(short['anomalies'], [dict(kind='model_near_approach', native_step=0)])
        for forbidden in ('link_positions_m', 'center_m', 'radius_m', 'near_native_steps', 'samples'):
            self.assertNotIn('"' + forbidden + '"', json.dumps(short))
        for selected in short['key_poses']:
            for arm in ('left', 'right'):
                self.assertEqual(selected['arms'][arm]['q_rad'], full['samples'][selected['native_step']]['arms'][arm]['q_rad'])
        pose = short['key_poses'][0]['arms']['left']['link6_pose']
        self.assertEqual(pose, dict(position_m=[0.6, 0.0, 0.0], rotation_matrix=np.eye(3).tolist()))
        self.assertIn('center_m', full['inter_arm_proximity']['witness']['left'])
        self.assertEqual(full['samples'][0]['arms']['left']['q_rad'], state['arms']['left']['q_rad'])


# ======================================================================
# 默认自动轨迹回传与历史轨迹审阅测试 (迁自 test_automatic_trajectory_delivery.py / test_action_trajectory_inspection.py)
# ======================================================================


class AutomaticTrajectoryDeliveryTests(unittest.TestCase):
    setUp = ActionTrajectoryFeedbackTests.setUp
    put = ActionTrajectoryFeedbackTests.put
    receipt = ActionTrajectoryFeedbackTests.receipt
    rows = staticmethod(ActionTrajectoryFeedbackTests.rows)

    def test_default_retains_both_q_paths_but_one_measured_image_and_full_artifact(self):
        from ni_rua.core.config import Config

        self.harness.config = Config()
        receipt = self.receipt(dict(actions=self.rows(2), completed_native_steps=2), measured_error=0.01)
        receipt.update(position_error_m=0.001, orientation_error_rad=0.02)
        original = deepcopy(receipt)
        result, images = build_action_trajectory_feedback(self.harness, self.action, receipt, {})
        self.assertTrue(result['available'], result)
        self.assertEqual(result['delivery'], 'automatic_sampled_joints_and_image')
        self.assertEqual(receipt, original)
        self.assertEqual(len(images), 1)
        self.assertEqual(set(result['overlays']), {'measured'})
        self.assertEqual(result['planned']['samples'][-1]['arms']['left']['q_rad'], [0.2] * 6)
        self.assertEqual(result['measured']['samples'][-1]['arms']['left']['q_rad'], [0.2 + 0.01] * 6)
        with Image.open(io.BytesIO(base64.b64decode(images[0]['imageUrl'].split(',', 1)[1]))) as image:
            self.assertEqual(image.size, (640, 480))
        full = json.loads(self.store.resolve_artifact(result['trajectory_feedback_ref'], ('actor',))[1])
        self.assertEqual(full['source_refs']['planned_trace_ref'], receipt['planned_trace_ref'])
        self.assertEqual(len(full['paths']['planned']['samples']), 3)
        self.assertEqual(full['paths']['planned']['samples'][-1]['arms']['left']['q_rad'], [0.2] * 6)
        self.assertIn('link_positions_m', full['paths']['planned']['samples'][-1]['arms']['left'])
        self.assertNotIn('FORBIDDEN', canonical(full))

    def test_diagnostic_mode_preserves_optional_routine_summary_policy(self):
        from ni_rua.core.config import Config

        self.harness.config = Config().with_overrides({'features': {'action_trajectory_delivery': 'diagnostic'}})
        receipt = self.receipt(dict(actions=self.rows(2), completed_native_steps=2))
        with patch('ni_rua.runtime.actions.inspection._trajectory_image', side_effect=AssertionError('routine render')):
            result, images = build_action_trajectory_feedback(self.harness, self.action, receipt, {})
        self.assertEqual(result['delivery'], 'routine_execution_summary')
        self.assertNotIn('q_rad', canonical(result))
        self.assertEqual(images, [])

    def test_failure_retains_planned_and_measured_diagnostics_in_both_modes(self):
        from ni_rua.core.config import Config

        for mode in ('automatic', 'diagnostic'):
            with self.subTest(mode=mode):
                self.harness.config = Config().with_overrides({'features': {'action_trajectory_delivery': mode}})
                receipt = self.receipt(dict(actions=self.rows(2), completed_native_steps=2))
                receipt.update(status='stopped', reason='tracking_error')
                result, images = build_action_trajectory_feedback(self.harness, self.action, receipt, {})
                self.assertEqual(len(images), 2)
                self.assertEqual(set(result['overlays']), {'measured', 'planned'})
                self.assertIn('q_rad', canonical(result))


class ActionTrajectoryInspectionTests(unittest.TestCase):
    setUp = ActionTrajectoryFeedbackTests.setUp
    put = ActionTrajectoryFeedbackTests.put
    receipt = ActionTrajectoryFeedbackTests.receipt
    rows = staticmethod(ActionTrajectoryFeedbackTests.rows)

    def feedback(self, *, plan=None, counts=2, status='completed', complete_sampling=True):
        import threading
        from unittest.mock import Mock
        from ni_rua.core.config import Config

        self.harness.config = Config()
        self.harness.lock = threading.RLock()
        self.harness.robot = Mock()
        self.harness.native_action_count = counts
        receipt = self.receipt(
            plan if plan is not None else dict(actions=self.rows(counts), completed_native_steps=counts),
            counts=counts,
        )
        receipt['status'] = status
        if status != 'completed':
            receipt['reason'] = 'tracking_error'
        if not complete_sampling:
            receipt.pop('trace_sampling')
        summary, _ = build_action_trajectory_feedback(self.harness, self.action, receipt, {})
        self.assertTrue(summary['available'], summary)
        reference = summary['trajectory_feedback_ref']
        self.original_bytes = self.store.resolve_artifact(reference, ('actor',))[1]
        return reference, json.loads(self.original_bytes)

    def inspect(self, reference, **options):
        from ni_rua.runtime.actions.inspection import inspect_trajectory

        before = deepcopy(self.harness.latest)
        count = self.harness.native_action_count
        reply = inspect_trajectory(self.harness, dict(trajectory_ref=reference, **options), {})
        self.assertEqual(self.harness.robot.mock_calls, [])
        self.assertEqual(self.harness.native_action_count, count)
        self.assertEqual(self.harness.latest, before)
        return json.loads(reply[0]['text']), reply

    def test_default_measured_path_preserves_exact_joint_jaw_and_time_values(self):
        reference, full = self.feedback()
        result, _ = self.inspect(reference)
        self.assertEqual(result['path_kind'], 'measured')
        self.assertEqual(result['available_paths'], ['measured', 'planned'])
        self.assertEqual(result['source_kind'], 'measured_action_trajectory')
        self.assertEqual(result['execution_start_observation_id'], 'start')
        self.assertFalse(result['current'])
        self.assertTrue(result['historical'])
        self.assertTrue(result['native_boundary_alignment'])
        self.assertNotIn('native_dt_s', result)
        for value, original in zip(result['samples'], full['paths']['measured']['samples']):
            self.assertEqual(value['sim_step'], original['sim_step'])
            for arm in ('left', 'right'):
                self.assertEqual(value['arms'][arm]['q_rad'], original['arms'][arm]['q_rad'])
                self.assertEqual(value['arms'][arm]['measured_jaw'], original['arms'][arm]['measured_jaw'])
                self.assertNotIn('link6_pose', value['arms'][arm])
        self.assertEqual(result['timing'], full['paths']['measured']['timing'])
        self.assertEqual(self.store.resolve_artifact(reference, ('actor',))[1], self.original_bytes)

    def test_incomplete_sampling_exposes_unknown_alignment_without_inventing_timing(self):
        reference, full = self.feedback(complete_sampling=False)
        result, _ = self.inspect(reference)
        self.assertFalse(result['native_boundary_alignment'])
        self.assertEqual(result['available_paths'], ['measured'])
        self.assertEqual(result['timing'], full['paths']['measured']['timing'])
        self.assertEqual([sample['sim_step'] for sample in result['samples']], [30, 40, 50])
        self.assertNotIn('native_dt_s', result)
        self.assertEqual(result['recorded_limitations'], full['summary']['limitations'])
        self.assertTrue(any('Complete native-boundary sampling' in item for item in result['recorded_limitations']))

    def test_paging_returns_every_exact_sample_and_full_geometry_is_opt_in(self):
        reference, full = self.feedback(counts=5)
        start, observed = 0, []
        while start is not None:
            result, _ = self.inspect(reference, start_native_step=start, max_samples=2, detail='full')
            observed.extend(result['samples'])
            start = result['next_native_step']
        self.assertEqual(observed, full['paths']['measured']['samples'])
        self.assertEqual(result['robot_model'], full['paths']['measured']['robot_model'])
        self.assertEqual(result['T_world_bases'], full['paths']['measured']['T_world_bases'])
        self.assertEqual(self.store.resolve_artifact(reference, ('actor',))[1], self.original_bytes)

    def test_planned_path_is_not_measurement_and_preserves_issued_jaw_goals(self):
        reference, full = self.feedback()
        result, _ = self.inspect(reference, path_kind='planned', native_steps=[1, 2])
        self.assertEqual(result['source_kind'], 'recorded_planned_commands_not_measurement')
        for value, original in zip(result['samples'], full['paths']['planned']['samples'][1:]):
            self.assertEqual(value['execution_state'], original['execution_state'])
            for arm in ('left', 'right'):
                self.assertEqual(value['arms'][arm]['q_rad'], original['arms'][arm]['q_rad'])
                self.assertEqual(value['arms'][arm]['jaw_goal_normalized'], 1.0)
                self.assertNotIn('measured_jaw', value['arms'][arm])

    def test_local_unexecuted_path_retains_its_independent_timing(self):
        plan = dict(
            actions=self.rows(1),
            completed_native_steps=1,
            arms={
                'left': dict(segments=[dict(
                    compiled_sim_step=30,
                    waypoint_index=0,
                    motion_actions=[row[:7] for row in self.rows(4)],
                    gripper_actions=[],
                    completed_motion_steps=1,
                )]),
                'right': dict(segments=[]),
            },
        )
        reference, full = self.feedback(plan=plan, counts=1, status='failed')
        result, _ = self.inspect(reference, path_kind='local_0', start_native_step=1, max_samples=25)
        self.assertIn('local_0', result['available_paths'])
        self.assertEqual(result['source_kind'], 'unexecuted_arm_candidate_not_synchronized')
        self.assertIn('no shared future', result['timing'])
        self.assertEqual(result['remaining_rows'], 3)
        self.assertEqual(result['samples'][-1]['execution_state'], 'not_recorded_completed')
        self.assertEqual(result['samples'][-1]['arms']['left']['q_rad'], [0.4] * 6)
        self.assertEqual(set(result['samples'][-1]['arms']), {'left'})

    def test_saved_action_overlay_uses_archived_start_rgb_and_marks_historical(self):
        reference, full = self.feedback()
        result, reply = self.inspect(reference, camera_id='cam_head')
        overlay = result['overlay']
        self.assertEqual(overlay['status'], 'available', overlay)
        self.assertEqual(len(reply), 2)
        self.assertEqual(overlay['original_rgb_ref'], self.rgb_ref)
        self.assertEqual(overlay['observation_id'], 'start')
        self.assertEqual(overlay['execution_start_observation_id'], 'start')
        self.assertTrue(overlay['historical'])
        self.assertFalse(overlay['current'])
        self.assertFalse(overlay['physics_advanced'])
        self.assertIn('Historical execution-start RGB', overlay['limitations'])
        self.assertEqual(overlay['panel_width'], 640)
        self.assertEqual(self.store.resolve_artifact(reference, ('actor',))[1], self.original_bytes)

    def test_unknown_or_missing_path_and_invalid_path_selector_are_rejected(self):
        reference, _ = self.feedback(plan={'segments': []}, counts=0, status='failed')
        for selector in ('planned', 'local_9', 'unknown'):
            with self.subTest(selector=selector), self.assertRaisesRegex(ValueError, 'path_unavailable:.*available_paths=measured'):
                self.inspect(reference, path_kind=selector)
        for selector in (None, [], 3, ''):
            with self.subTest(selector=selector), self.assertRaisesRegex(ValueError, 'nonempty_string'):
                self.inspect(reference, path_kind=selector)

    def test_actor_scope_identity_binding_source_and_profile_cannot_be_bypassed(self):
        reference, full = self.feedback()
        mutations = [
            ('stale_environment_identity', lambda d: d['identity'].update(episode_id='other')),
            ('stale_environment_identity', lambda d: d['paths']['measured']['identity'].update(episode_id='other')),
            ('binding_mismatch', lambda d: d['paths']['measured'].update(observation_id='other')),
            ('invalid_action_trajectory_path_source', lambda d: d['paths']['measured'].update(source_kind='candidate_command_trajectory_not_measurement')),
            ('privileged', lambda d: d.update(object_poses={'hidden': 1})),
        ]
        for reason, mutation in mutations:
            data = deepcopy(full)
            mutation(data)
            ref = self.put(data, 'action_trajectory_feedback')
            with self.subTest(reason=reason), self.assertRaisesRegex(ValueError, reason):
                self.inspect(ref)
        ref = self.put(full, 'action_trajectory_feedback', scope='internal')
        with self.assertRaisesRegex(ValueError, 'access_scope_denied'):
            self.inspect(ref)
        self.assertEqual(self.harness.robot.mock_calls, [])



# --- Folded from tests/test_trajectory_tools.py ---

def path_fixture():
    return dict(source_kind='candidate_command_trajectory_not_measurement', frame_id='env_origin',
        tcp_config_id='arx-x5-link6-v1', joint_names={a:['joint1'] for a in ('left','right')},native_dt_s=.05,
        robot_model=dict(joints=[dict(parent='base',child='link6')]), samples=[
            dict(native_step=i, arms={a:dict(q_rad=[i*.01]*6,phase='motion',
                link_positions_m={'base':[x,0,2], 'link6':[x+i*.03, .1, 2]})
                for a,x in [('left',-.1),('right',.1)]}) for i in range(4)])


class TrajectoryOverlayTests(unittest.TestCase):
    def test_synced_panels_preserve_original_rgb_and_label_candidate(self):
        camera, image = camera_fixture(); before=image.copy()
        output, metadata = render_trajectory_overlay(camera,image,path_fixture())
        # 默认输出 640x480 单面板多时刻复合轨迹图，保留三采样点与原图不污染契约
        self.assertEqual(output.shape,(480,640,3))
        self.assertEqual([p['native_step'] for p in metadata['panels']],[0,2,3])
        self.assertEqual(set(metadata['panels'][1]['arms']),{'left','right'})
        np.testing.assert_array_equal(image,before)
        self.assertFalse(metadata['physics_advanced'])
        self.assertIn('Unexecuted',metadata['limitations'])

    def test_privileged_camera_frame_mismatch_and_unbounded_samples_reject(self):
        camera,image=camera_fixture(); path=path_fixture()
        for change,reason in [({'pose_source':'simulator_camera_transform'},'deployable'),
                              ({'reference_frame_id':'unknown'},'frame_mismatch')]:
            with self.assertRaisesRegex(ValueError,reason):render_trajectory_overlay({**camera,**change},image,path)
        with self.assertRaisesRegex(ValueError,'one_to_three'):
            render_trajectory_overlay(camera,image,path,native_steps=[0,1,2,3])

    def test_fixed_camera_uses_only_static_calibration_never_dynamic_pose_sample(self):
        camera,image=camera_fixture(); observation=deepcopy(camera['observation'])
        observation['cameras']=[dict(camera['frame'],width=640,height=480,calibration_ref='cal',pose_sample_ref='FORBIDDEN')]
        config=dict(reference_frame_id='env_origin',cameras={'head':dict(K=camera['K'].tolist(),distortion=[0.]*5,
            image_size=[640,480],T_world_camera=np.eye(4).tolist(),T_world_base='FORBIDDEN')})
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        identity=dict(run_id='run',episode_id='ep',env_instance_id='env',env_generation=1)
        store=Store(Path(temporary.name),identity); self.addCleanup(store.close)
        observation.update(identity)
        reference=store.put_artifact(json.dumps(config).encode(),'calibration','application/json')
        observation['cameras'][0]['calibration_ref']=reference
        store.save('observation','observed',observation)
        actual=deployable_camera(store,identity,'observed','head',path_fixture())
        np.testing.assert_array_equal(actual['T'],np.eye(4));self.assertEqual(actual['pose_source'],'fixed_calibration')
        config.pop('reference_frame_id')
        reference=store.put_artifact(json.dumps(config).encode(),'calibration','application/json')
        observation['cameras'][0]['calibration_ref']=reference
        store.save('observation','missing',dict(observation,observation_id='missing'))
        with self.assertRaisesRegex(ValueError,'missing_calibrated_frame'):
            deployable_camera(store,identity,'missing','head',path_fixture())

    def test_wrist_camera_uses_measured_fk_and_never_pose_sample_or_calibration_base(self):
        camera,image=camera_fixture(); observation=deepcopy(camera['observation'])
        identity=dict(run_id='run',episode_id='ep',env_instance_id='env',env_generation=1)
        temporary=tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        store=Store(Path(temporary.name),identity); self.addCleanup(store.close)
        observation.update(identity)
        config=dict(reference_frame_id='env_origin',cameras={'wrist':dict(K=camera['K'].tolist(),distortion=[0.]*5,
            image_size=[640,480],arm='left',link_name='link6',T_link_camera=np.eye(4).tolist(),T_world_base='FORBIDDEN')})
        state=dict(sim_step=observation['sim_step'],step_domain=observation['step_domain'],arms={'left':dict(
            source_kind='measurement',joint_names=['j'+str(i) for i in range(6)],q_rad=[0.]*6,
            gripper=dict(position_source_kind='measurement',joint_names=[],position_m=[]))})
        observation['robot_state_ref']=store.put_artifact(json.dumps(state).encode(),'robot_state','application/json')
        cal=store.put_artifact(json.dumps(config).encode(),'calibration','application/json')
        observation['cameras']=[dict(camera['frame'],camera_id='wrist',width=640,height=480,calibration_ref=cal,pose_sample_ref='FORBIDDEN')]
        store.save('observation','observed',observation)
        path=path_fixture();base=np.eye(4);base[2,3]=1.
        path['T_world_bases']={'left':base.tolist()}
        path['robot_model']=dict(base_link='base',joints=[dict(name='fixed',type='fixed',parent='base',child='link6',T_parent_joint=np.eye(4).tolist(),axis=[1,0,0])])
        actual=deployable_camera(store,identity,'observed','wrist',path)
        np.testing.assert_array_equal(actual['T'],base)
        self.assertEqual(actual['pose_source'],'robot_kinematics')


class TrajectoryBoundaryTests(unittest.TestCase):
    setUp = DualArmHarnessBase.setUp

    def preview(self, data=None):
        data=dict(data or path_fixture(),identity=dict(self.h.identity),
                  observation_id=self.h.latest['observation_id'],label='candidate')
        return dict(trajectory_ref=self.h.store.put_artifact(canonical(data).encode(),'candidate_trajectory','application/json'))

    def test_large_data_stored_and_exact_bounded_samples_retrievable_without_actions(self):
        result=self.preview()
        args=dict(trajectory_ref=result['trajectory_ref'],native_steps=[1,3])
        info=json.loads(inspect_trajectory(self.h,args,{})[0]['text'])
        self.assertEqual(info['samples'][0]['arms']['left']['q_rad'],[.01]*6)
        self.assertNotIn('link_positions_m',info['samples'][0]['arms']['left'])
        self.assertTrue(info['current']);self.assertEqual(self.robot.executions,0)
        with self.assertRaisesRegex(ValueError,'one_to_twenty_five'):
            inspect_trajectory(self.h,{**args,'native_steps':[True]}, {})
        self.h.identity['episode_id']='new'
        with self.assertRaisesRegex(ValueError,'stale_environment_identity'):inspect_trajectory(self.h,args,{})

    def test_pagination_retrieves_every_exact_joint_and_jaw_command(self):
        data=path_fixture()
        data['samples']=[dict(native_step=i,arms={arm:dict(q_rad=[i*.12345678901234567+j*.00000000012345 for j in range(6)],
            jaw_goal_normalized=i/63.,phase='motion',link_positions_m={'link6':[i*.03,0.,2.]})
            for arm in ('left','right')}) for i in range(63)]
        preview=self.preview(data)
        ref=preview['trajectory_ref']
        collected=[]; start=0; sizes=[]
        while start is not None:
            args=dict(trajectory_ref=ref,start_native_step=start,max_samples=25)
            info=json.loads(inspect_trajectory(self.h,args,{})[0]['text'])
            self.assertEqual(info['selection_mode'],'page')
            self.assertEqual(info['total_samples'],63)
            collected.extend(info['samples']);sizes.append(len(info['samples']))
            start=info['next_native_step']
        expected=[dict(native_step=s['native_step'],arms={a:{k:v for k,v in c.items() if k!='link_positions_m'}
                  for a,c in s['arms'].items()}) for s in data['samples']]
        self.assertEqual(collected,expected)
        self.assertEqual(sizes,[25,25,13])
        _,raw=self.h.store.resolve_artifact(ref,allowed_scopes=('actor',))
        self.assertEqual(json.loads(raw)['samples'],data['samples'])
        self.assertEqual(self.robot.executions,0)

    def test_default_key_samples_and_explicit_geometry(self):
        data=path_fixture();data.update(geometry=dict(status='available'),
            inter_arm_proximity=dict(status='unknown',reason='missing_spheres'),hardware_provenance={'kind':'public'})
        preview=self.preview(data);args=dict(trajectory_ref=preview['trajectory_ref'])
        short=json.loads(inspect_trajectory(self.h,args,{})[0]['text'])
        self.assertEqual([s['native_step'] for s in short['samples']],[0,2,3])
        self.assertEqual(short['selection_mode'],'key_samples')
        self.assertNotIn('robot_model',short)
        full=json.loads(inspect_trajectory(self.h,{**args,'include_geometry':True}, {})[0]['text'])
        self.assertEqual(full['samples'],[data['samples'][i] for i in [0,2,3]])
        self.assertEqual(full['robot_model'],data['robot_model'])
        self.assertEqual(full['inter_arm_proximity'],data['inter_arm_proximity'])
        self.assertEqual(full,json.loads(inspect_trajectory(self.h,{**args,'detail':'full'}, {})[0]['text']))

    def test_selector_validation_and_pagination_defaults(self):
        result=self.preview();args=dict(trajectory_ref=result['trajectory_ref'])
        invalid=[dict(native_steps=[[]]),dict(native_steps=[0,0]),dict(native_steps=[]),dict(native_steps=None),
            dict(native_steps=[4]),dict(native_steps=[0],max_samples=1),dict(native_steps=[0],start_native_step=0),
            dict(start_native_step=True),dict(start_native_step=-1),dict(start_native_step=4),dict(start_native_step=0.5),
            dict(max_samples=True),dict(max_samples=0),dict(max_samples=26),dict(max_samples=1.5),
            dict(include_geometry='true'),dict(detail='bad'),dict(detail='full',include_geometry=False),
            dict(detail='summary',include_geometry=True)]
        for extra in invalid:
            with self.subTest(extra=extra),self.assertRaises(ValueError):inspect_trajectory(self.h,{**args,**extra},{})
        first=json.loads(inspect_trajectory(self.h,{**args,'max_samples':2},{})[0]['text'])
        self.assertEqual([s['native_step'] for s in first['samples']],[0,1])
        self.assertEqual(first['next_native_step'],2)
        last=json.loads(inspect_trajectory(self.h,{**args,'start_native_step':2},{})[0]['text'])
        self.assertEqual([s['native_step'] for s in last['samples']],[2,3])
        self.assertIsNone(last['next_native_step'])

    def test_overlay_contract_remains_independent_of_geometry_output(self):
        result=self.preview();args=dict(trajectory_ref=result['trajectory_ref'],native_steps=[1,3],camera_id='head')
        image=dict(type='inputImage',imageUrl='data:image/jpeg;base64,test')
        with patch('ni_rua.runtime.actions.inspection._trajectory_image',return_value=(dict(status='available'),[image])) as render:
            content=inspect_trajectory(self.h,args,{})
        self.assertEqual(content[1],image)
        self.assertEqual(render.call_args.args[-1],[1,3])
        self.assertIn('link_positions_m',render.call_args.args[1]['samples'][0]['arms']['left'])
        with self.assertRaisesRegex(ValueError,'one_to_three'):
            inspect_trajectory(self.h,{**args,'native_steps':[0,1,2,3]}, {})

    def test_nonactor_and_wrong_kind_artifacts_reject(self):
        data=path_fixture();data.update(identity=self.h.identity)
        for kind,scope,reason in [('candidate_trajectory','internal','access_scope_denied'),
                                  ('robot_state','actor','not_a_candidate_trajectory')]:
            ref=self.h.store.put_artifact(canonical(data).encode(),kind,'application/json',access_scope=scope)
            with self.subTest(kind=kind,scope=scope),self.assertRaisesRegex(ValueError,reason):
                inspect_trajectory(self.h,dict(trajectory_ref=ref),{})

    def test_old_preview_is_readable_but_not_overlaid_as_current(self):
        result=self.preview(); self.h.latest={**self.h.latest,'observation_id':'new'}
        info=json.loads(inspect_trajectory(self.h,dict(trajectory_ref=result['trajectory_ref'],camera_id='head'),{})[0]['text'])
        self.assertFalse(info['current']);self.assertEqual(info['overlay']['status'],'unavailable')
        self.assertIn('requires_current_preview',info['overlay']['reason'])

    def test_action_path_selector_cannot_change_candidate_preview_semantics(self):
        result=self.preview()
        with self.assertRaisesRegex(ValueError,'path_kind_requires_action_feedback'):
            inspect_trajectory(self.h,dict(trajectory_ref=result['trajectory_ref'],path_kind='measured'),{})
        self.assertEqual(self.robot.executions,0)

    def test_disabled_trajectory_and_privileged_artifacts_cannot_leak(self):
        result=self.preview()
        data=path_fixture(); data.update(identity=self.h.identity,object_poses={'hidden':1})
        ref=self.h.store.put_artifact(json.dumps(data).encode(),'candidate_trajectory','application/json')
        with self.assertRaisesRegex(ValueError,'privileged'):
            inspect_trajectory(self.h,dict(trajectory_ref=ref),{})
        self.h.config=self.h.config.with_overrides({'features':{'trajectory_preview':False}})
        with self.assertRaisesRegex(ValueError,'disabled'):inspect_trajectory(self.h,dict(trajectory_ref=result['trajectory_ref']),{})


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



if __name__ == "__main__":
    unittest.main()
