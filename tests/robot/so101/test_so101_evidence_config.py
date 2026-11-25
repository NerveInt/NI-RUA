"""Shared offline file defaults and explicit override precedence; no model or hardware."""

NI_RUA_TEST_TIER = "component"

import contextlib
import inspect
import io
import json
import os
from pathlib import Path
from support.paths import CONFIGS_ROOT, REPO_ROOT
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from ni_rua.robot.calibration.evidence.config import evidence_settings
from ni_rua.robot.calibration.evidence import video_evidence as video, vggt_evidence as vggt
from ni_rua.robot.calibration.evidence import contact_trajectory as contact
from services.vggt_height.sam31 import video_tracking as sam


class EvidenceConfigTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = self.root / 'settings.json'

    def save(self, value):
        self.config.write_text(json.dumps(value))
        return str(self.config)

    def test_defaults_match_example_and_function_contracts(self):
        example = CONFIGS_ROOT / 'so101-evidence.example.json'
        for section, function in [('video_evidence', video.analyze_video), ('vggt_evidence', vggt.analyze_points),
                                  ('sam31_video', sam.run_video_tracking), ('contact_fit', contact.fit_contact_trajectories)]:
            values = evidence_settings(section)
            self.assertEqual(values, evidence_settings(section, example))
            for key, value in values.items():
                default = inspect.signature(function).parameters[key].default
                self.assertEqual(list(default) if isinstance(default, tuple) else default, value)
        values['estimated_joints'].append('wrist_flex')
        self.assertEqual(evidence_settings('contact_fit')['estimated_joints'], [])

    def test_unknown_section_setting_and_bad_types_fail(self):
        for document in [{'typo': {}}, {'video_evidence': {'window_size_pz': 5}},
                         {'sam31_video': {'max_num_objects': True}},
                         {'video_evidence': {'max_frame_gap_s': None}},
                         {'vggt_evidence': {'rank_rtol': float('nan')}},
                         {'sam31_video': {'use_fa3': 1}}]:
            with self.subTest(document=document), self.assertRaises(ValueError):
                evidence_settings('video_evidence', self.save(document))

    def test_video_cli_preserves_file_values_but_explicit_flag_wins(self):
        config = self.save({'video_evidence': {'max_frame_gap_s': .7, 'window_size_px': 31}})
        with patch.object(video, 'analyze_video', return_value={'status': 'completed', 'frames': []}) as call, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(video.main(['--video-manifest','manifest','--camera','fixed','--tracking','tracks',
                                        '--output','new-output','--config',config,'--max-frame-gap-s','.4']), 0)
        self.assertEqual(call.call_args.kwargs['max_frame_gap_s'], .4)
        self.assertEqual(call.call_args.kwargs['window_size_px'], 31)
        self.assertEqual(call.call_args.kwargs['max_joint_gap_s'], .25)

    def test_sam_cli_can_explicitly_disable_file_boolean(self):
        config = self.save({'sam31_video': {'max_num_objects': 8, 'use_fa3': True, 'output_prob_threshold': .7}})
        prompts = self.root / 'prompts.json'; prompts.write_text('[]')
        result = dict(status='completed', reason=None, session_closed=True, frames=[], frames_with_any_mask=0, object_coverage=[])
        with patch.object(sam, 'run_video_tracking', return_value=result) as call, contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(sam.main(['--frames','frames','--prompts',str(prompts),'--output','new',
                                      '--checkpoint','weights','--config',config,'--no-use-fa3']),0)
        self.assertEqual(call.call_args.kwargs, dict(checkpoint_path='weights',max_num_objects=8,use_fa3=False,output_prob_threshold=.7))

    def test_invalid_config_never_invokes_sam(self):
        config = self.save({'sam31_video': {'max_num_object': 8}})
        with patch.object(sam, 'run_video_tracking') as call, self.assertRaises(ValueError):
            sam.main(['--frames','frames','--prompts','missing','--output','new','--checkpoint','weights','--config',config])
        call.assert_not_called()

    def test_vggt_cli_saves_actual_settings(self):
        config = self.save({'vggt_evidence': {'min_confidence_rank': .6, 'rank_rtol': 1e-5}})
        empty = self.root/'input.json';empty.write_text('{}')
        output = self.root/'result.json'
        def analyze(request,response,review,**settings):
            return dict(status='diagnostic',plane_camera_depth_units={'status':'unknown'},external_metric_scale=None,settings=settings)
        with patch.object(vggt,'analyze_points',side_effect=analyze), contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(vggt.main(['--request',str(empty),'--response',str(empty),'--review',str(empty),
                                      '--output',str(output),'--config',config,'--min-confidence-rank','.2']),0)
        settings=json.loads(output.read_text())['settings']
        self.assertEqual(settings,dict(min_confidence_rank=.2,max_landmark_pixel_distance_px=3.,rank_rtol=1e-5))

    def test_contact_review_precedence_is_explicit(self):
        self.save({'contact_fit': {'sample_sigma_m': .002, 'estimated_joints': ['elbow_flex']}})
        effective=evidence_settings('contact_fit',self.config,{'sample_sigma_m':.004})
        self.assertEqual(effective['sample_sigma_m'],.004)
        self.assertEqual(effective['estimated_joints'],['elbow_flex'])

    def test_standalone_sam_declared_companions_need_no_ni_rua_install(self):
        repo = REPO_ROOT
        shutil.copy2(repo/'services/vggt_height/sam31/video_tracking.py', self.root/'video_tracking.py')
        for name in ('config.py', 'defaults.json'):
            shutil.copy2(repo/'ni_rua/robot/calibration/evidence'/name, self.root/name)
        # A blocking import hook also tests environments where NI-RUA is installed:
        # this subprocess must use the explicitly copied standard-library companion.
        code = '''import sys, runpy
class NoNiRua:
    def find_spec(self, fullname, path=None, target=None):
        if fullname == 'ni_rua':
            raise ModuleNotFoundError("standalone environment", name='ni_rua')
sys.meta_path.insert(0, NoNiRua())
sys.argv = ['video_tracking.py', '--help']
runpy.run_path('video_tracking.py', run_name='__main__')
'''
        result = subprocess.run([sys.executable, '-c', code], cwd=self.root,
                                env={k:v for k,v in os.environ.items() if k != 'PYTHONPATH'},
                                capture_output=True, text=True, timeout=20)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn('--config', result.stdout)


if __name__ == "__main__":
    unittest.main()
