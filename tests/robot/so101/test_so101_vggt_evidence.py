"""Existing /points algebra, independent length scale and holdout handling; no network/model/hardware."""

NI_RUA_TEST_TIER = "component"

import base64
import copy
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

import cv2
import numpy as np

from ni_rua.robot.calibration.evidence.vggt_evidence import (
    REVIEW_SCHEMA, analyze_points, compare_support_hypotheses, decode_points_response,
)


class VGGTEvidenceTests(unittest.TestCase):
    def sample(self, *, support=0., remove_last=False):
        image = np.zeros((64, 64, 3), np.uint8)
        encoded = cv2.imencode('.png', image)[1].tobytes()
        K = np.array([[100,0,32], [0,100,32], [0,0,1.]])
        T = np.diag([1., -1., -1., 1.]); T[:3,3] = [.1,.2,.8]
        uv = np.array([[x,y] for y in range(8,57,8) for x in range(8,57,8)], dtype=np.uint16)
        if remove_last:
            uv = uv[:-1]
        camera = np.column_stack([(uv[:,0].astype(float)+.5-32)/100*2,
                                  (uv[:,1].astype(float)+.5-32)/100*2, np.full(len(uv),2.)])
        scale = (.8-support)/2
        xyz = T[:3,3] + scale * camera @ T[:3,:3].T
        arrays = dict(xyz=xyz.astype(np.float32), uv=uv, conf=np.full(len(uv),255,np.uint8),
                      count=np.ones(len(uv),np.uint16), robot=np.zeros(len(uv),bool),
                      support_z=np.float64(support), voxel_m=np.float64(.002), image_size=np.array([64,64],np.int32))
        buf=io.BytesIO();np.savez_compressed(buf,**arrays)
        request=dict(image_b64=base64.b64encode(encoded).decode(), K=K.tolist(), T_world_camera=T.tolist(),
                     support_z_m=support,voxel_m=.002,robot_segments_m=[],params={'z_below':.1,'z_above':.6,'max_range':3.})
        response=dict(status='ok',version='scene3d-points-v1',image_sha256=hashlib.sha256(encoded).hexdigest(),
                      image_size=[64,64],scale=scale,support_z_m=support,voxel_m=.002,n_points=len(uv),
                      npz_b64=base64.b64encode(buf.getvalue()).decode(),support_fit={'tilt_deg':0})
        review=dict(schema=REVIEW_SCHEMA,image_sha256=response['image_sha256'],pixel_coordinates='service_image',
                    plane_roi_px=[[0,0],[64,0],[64,64],[0,64]])
        return request,response,review,camera,uv

    def metric_review(self, review, camera, uv):
        review=copy.deepcopy(review)
        review.update(reference_note='Synthetic independent metric grid, not FK/table-derived',reference_uncertainty_m=None)
        review['metric_landmarks']=[dict(id=str(i),pixel_px=(uv[i].astype(float)+.5).tolist(),
            reference_point_m=(camera[i]*.25).tolist(),role=role)
            for i,role in [(0,'fit'),(6,'fit'),(42,'fit'),(24,'holdout'),(48,'holdout')]]
        return review

    def test_inverse_similarity_shape_is_not_metric_validation(self):
        q,a,r,camera,_=self.sample()
        cloud=decode_points_response(q,a)
        np.testing.assert_allclose(cloud['camera_depth_points'],camera,atol=1e-7)
        result=analyze_points(q,a,r)
        self.assertEqual(result['plane_camera_depth_units']['status'],'conditional_shape_fit')
        self.assertLess(result['plane_camera_depth_units']['normalized_residual_rms'],1e-7)
        self.assertIsNone(result['external_metric_scale'])
        self.assertIsNone(result['metric_plane'])
        self.assertFalse(result['independence']['absolute_height_from_service'])
        self.assertFalse(result['independence']['full_raw_reconstruction'])
        self.assertIsNone(result['physical_acceptance'])
        self.assertEqual(result['contact_evidence'],'unknown')
        json.dumps(result,allow_nan=False)

    def test_support_perturbation_cancels_but_selection_remains_recorded(self):
        q,a,*_=self.sample()
        qb,ab,*_=self.sample(support=.01,remove_last=True)
        result=compare_support_hypotheses(q,a,qb,ab)
        self.assertAlmostEqual(result['actual_service_scale_ratio'],result['support_formula_scale_ratio'])
        self.assertLess(result['common_camera_depth_rms_difference'],1e-7)
        self.assertEqual(result['only_a_count'],1)
        self.assertEqual(result['common_uv_count'],48)
        self.assertFalse(result['table_height_independently_validated'])

    def test_external_known_lengths_supply_scale_without_support_height(self):
        q,a,r,camera,uv=self.sample()
        review=self.metric_review(r,camera,uv)
        result=analyze_points(q,a,review)
        self.assertAlmostEqual(result['external_metric_scale'],.25,places=7)
        self.assertEqual(result['external_metric_scale_status'],'estimated')
        self.assertAlmostEqual(result['metric_plane']['distance_camera_m'],.5,places=7)
        # The service was anchored to table z=0; independent length scale puts the plane at .3
        # under the supplied camera translation. That remains conditional on its extrinsics.
        self.assertAlmostEqual(result['metric_plane']['height_at_base_origin_m'],.3,places=7)
        self.assertIsNone(result['metric_plane']['external_validation'])
        self.assertEqual(len(result['training_edges']),3)
        self.assertEqual(len(result['heldout_edges']),1)
        self.assertLess(result['edge_residual_summary']['heldout_edges']['rms_m'],1e-7)

    def test_heldout_bad_lengths_do_not_influence_scale(self):
        q,a,r,camera,uv=self.sample()
        review=self.metric_review(r,camera,uv)
        review['metric_landmarks'][-1]['reference_point_m'][0]+=.02
        result=analyze_points(q,a,review)
        self.assertAlmostEqual(result['external_metric_scale'],.25,places=7)
        self.assertGreater(result['edge_residual_summary']['heldout_edges']['rms_m'],.01)
        self.assertIsNone(result['physical_acceptance'])

    def test_unknown_landmark_correspondence_never_falls_back_to_service_scale(self):
        q,a,r,camera,uv=self.sample()
        review=self.metric_review(r,camera,uv)
        review['metric_landmarks'][0]['pixel_px']=[1.,1.]
        result=analyze_points(q,a,review,max_landmark_pixel_distance_px=.1)
        self.assertEqual(result['landmarks'][0]['status'],'unknown')
        self.assertIsNone(result['external_metric_scale'])
        self.assertEqual(result['external_metric_scale_status'],'unknown')
        self.assertIsNone(result['metric_plane'])

    def test_collinear_surface_unknown_and_holdout_cannot_rescue_scale(self):
        q,a,r,camera,uv=self.sample()
        r['plane_roi_px']=[[0,8],[64,8],[64,9],[0,9]]
        result=analyze_points(q,a,r)
        self.assertEqual(result['plane_camera_depth_units']['reason'],'selected_points_collinear')
        review=self.metric_review(r,camera,uv)
        review['metric_landmarks'][2]['role']='holdout'
        self.assertIsNone(analyze_points(q,a,review)['external_metric_scale'])

    def test_image_support_mapping_and_npz_contract_tampering_rejected(self):
        q,a,r,_,_=self.sample()
        for field,value in [('image_sha256','wrong'),('support_z_m',.01),('scale',0),('image_size',[65,64]),
                            ('n_points',50),('status','failed'),('voxel_m',.01)]:
            with self.subTest(field=field),self.assertRaises(ValueError):
                decode_points_response(q,dict(a,**{field:value}))
        shifted=copy.deepcopy(q);shifted['T_world_camera'][0][0]=2
        with self.assertRaises(ValueError):decode_points_response(shifted,a)
        with self.assertRaises(ValueError):analyze_points(q,a,dict(r,pixel_coordinates='original_distorted_image'))
        with self.assertRaises(ValueError):analyze_points(q,a,dict(r,image_sha256='wrong'))

    def test_fit_holdout_reference_and_sample_reuse_rejected(self):
        q,a,r,camera,uv=self.sample()
        for kind in ('pixel','reference','id'):
            review=self.metric_review(r,camera,uv)
            key={'pixel':'pixel_px','reference':'reference_point_m','id':'id'}[kind]
            review['metric_landmarks'][-1][key]=review['metric_landmarks'][0][key]
            with self.subTest(kind=kind),self.assertRaises(ValueError):analyze_points(q,a,review)

    def test_comparison_rejects_multiple_interventions(self):
        q,a,*_=self.sample(); qb,ab,*_=self.sample(support=.01)
        qb['params']['max_range']=4
        with self.assertRaises(ValueError):compare_support_hypotheses(q,a,qb,ab)

    def test_cli_actual_archive_shape_preserves_inputs_and_does_not_overwrite(self):
        q,a,r,_,_=self.sample()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            paths={name:root/f'{name}.json' for name in ('request','response','review')}
            for name,value in zip(paths,(q,a,r)):paths[name].write_text(json.dumps(value))
            original={name:path.read_bytes() for name,path in paths.items()}
            command=[sys.executable,'-m','ni_rua.robot.calibration.evidence.vggt_evidence']
            for name,path in paths.items():command += ['--'+name,str(path)]
            command+=['--output',str(root/'result.json')]
            process=subprocess.run(command,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr)
            result=json.loads((root/'result.json').read_text())
            self.assertEqual(result['use_in_joint_solver'],'diagnostic_only')
            self.assertEqual({name:path.read_bytes() for name,path in paths.items()},original)
            again=subprocess.run(command,capture_output=True,text=True)
            self.assertNotEqual(again.returncode,0)
            self.assertEqual({name:path.read_bytes() for name,path in paths.items()},original)


if __name__ == "__main__":
    unittest.main()
