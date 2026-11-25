"""CPU synthetic motion and failure evidence; no SAM weights, network or hardware."""

NI_RUA_TEST_TIER = "fast"

# === tests/test_so101_video_evidence.py ===

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np

from ni_rua.robot.calibration.evidence.video_evidence import analyze_video, align_joints, _joint_samples
from services.vggt_height.sam31.video_tracking import run_video_tracking


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class MovingMaskPredictor:
    """Known synthetic object motion; implements only the official predictor protocol."""
    def __init__(self, missing=()):
        self.missing = missing

    def handle_request(self, request):
        if request['type'] == 'start_session':
            return {'session_id': 'synthetic'}
        if request['type'] == 'close_session':
            return {'is_success': True}
        return self.response(request['frame_index'])

    def response(self, i):
        ids = [] if i in self.missing else [1]
        masks = np.zeros((len(ids), 64, 64), bool)
        masks[:, 16+i:48+i, 16+2*i:48+2*i] = True
        return {'frame_index': i, 'outputs': {'out_obj_ids': np.array(ids, np.int64),
                'out_binary_masks': masks, 'out_probs': np.ones(len(ids))}}

    def handle_stream_request(self, request):
        for i in range(3):
            yield self.response(i)


class VideoEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.frames = self.root / 'video/fixed'
        self.frames.mkdir(parents=True)
        texture = np.random.default_rng(123).integers(0, 256, (24, 24, 3), dtype=np.uint8)
        rows = []
        for i in range(3):
            image = np.zeros((64, 64, 3), np.uint8)
            image[20+i:44+i, 20+2*i:44+2*i] = texture
            path = self.frames / f'{i:06d}.png'
            cv2.imwrite(str(path), image)
            rows.append(dict(index=i, path=f'fixed/{path.name}', width=64, height=64,
                source_sha256=digest(path), capture_monotonic_s=10+i*.1, host_read_interval_s=[10+i*.1, 10+i*.1+.01],
                joint_sample=dict(t=10+i*.1, q_rad=[i*.1]*5, gripper_rad=.2)))
        self.manifest = dict(schema='ni_rua.so101.calibration_video.v1', status='completed',
            clock_domain='adapter_monotonic', joint_mapping={'frozen': True},
            cameras={'fixed': {'frames': rows}}, camera_calibration=None)
        self.video = self.root / 'video/manifest.json'
        self.save_manifest()
        self.sam_dir = self.root / 'sam'
        self.track = self.sam_dir / 'tracking.json'
        self.features = [dict(feature_id='tip_texture', frame_index=0, object_id=1, point_px=[32,32])]

    def save_manifest(self):
        self.video.write_text(json.dumps(self.manifest))

    def run_sam(self, missing=()):
        run_video_tracking(self.frames, [dict(frame_index=0, object_id=1, points_px=[[32,32]],
            point_labels=[1])], self.sam_dir, predictor=MovingMaskPredictor(missing))

    def analyze(self, **kw):
        return analyze_video(self.video, 'fixed', self.track, self.features, self.root/'review', **kw)

    def test_end_to_end_known_motion_and_html(self):
        self.run_sam()
        result = self.analyze()
        self.assertEqual(result['status'], 'completed')
        self.assertEqual(result['contact_evidence'], 'unknown')
        for i, row in enumerate(result['frames']):
            f = row['features'][0]
            np.testing.assert_allclose(f['pixel'], [32+2*i, 32+i], atol=.15)
            self.assertIsNone(f['material_identity_verified'])
            self.assertFalse(row['joint_alignment']['exposure_aligned'])
            self.assertTrue((self.root/'review'/row['overlay']).is_file())
        self.assertTrue((self.root/'review/review.html').is_file())
        with self.assertRaises(FileExistsError):
            self.analyze()

    def test_missing_mask_never_reacquires(self):
        self.run_sam(missing=(1,))
        rows = self.analyze()['frames']
        self.assertEqual(rows[1]['features'][0]['reason'], 'sam_object_mask_unavailable')
        self.assertEqual(rows[2]['features'][0]['reason'], 'previously_lost_requires_new_feature_identity')

    def test_gap_loses_feature(self):
        self.run_sam()
        rows = self.analyze(max_frame_gap_s=.05)['frames']
        self.assertEqual(rows[1]['features'][0]['reason'], 'frame_gap_exceeds_limit')
        self.assertIsNone(rows[2]['features'][0]['pixel'])

    def test_textureless_flow_is_unknown(self):
        for path in self.frames.iterdir():
            cv2.imwrite(str(path), np.zeros((64,64,3), np.uint8))
        for row in self.manifest['cameras']['fixed']['frames']:
            row['source_sha256'] = digest(self.frames / Path(row['path']).name)
        self.save_manifest()
        self.run_sam()
        self.assertEqual(self.analyze()['frames'][1]['features'][0]['reason'], 'forward_flow_failed')

    def test_source_tampering_rejected(self):
        self.run_sam()
        (self.frames/'000000.png').write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes'):
            self.analyze()

    def test_mask_tampering_keeps_failed_report(self):
        self.run_sam()
        tracking = json.loads(self.track.read_text())
        (self.sam_dir/tracking['frames'][1]['objects'][0]['mask_file']).write_bytes(b'changed')
        with self.assertRaisesRegex(ValueError, 'bytes'):
            self.analyze()
        result = json.loads((self.root/'review/evidence.json').read_text())
        self.assertEqual(result['status'], 'failed')
        self.assertEqual(len(result['frames']), 1)

    def test_paths_and_duplicate_ids_rejected(self):
        self.run_sam()
        tracking = json.loads(self.track.read_text())
        tracking['frames'][0]['objects'] *= 2
        self.track.write_text(json.dumps(tracking))
        with self.assertRaisesRegex(ValueError, 'unique'):
            self.analyze()
        self.manifest['cameras']['fixed']['frames'][0]['path'] = '../sam/tracking.json'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'escapes'):
            self.analyze()

    def test_clock_and_dimensions_rejected(self):
        self.run_sam()
        self.manifest['clock_domain'] = 'wall_clock'
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'clock'):
            self.analyze()
        self.manifest['clock_domain'] = 'adapter_monotonic'
        self.manifest['cameras']['fixed']['frames'][0]['width'] = 63
        self.save_manifest()
        with self.assertRaisesRegex(ValueError, 'dimensions'):
            self.analyze()

    def test_bound_touch_and_mapping(self):
        self.run_sam()
        touch = self.root/'touch.json'
        touch.write_text(json.dumps(dict(video={'manifest_sha256': 'wrong'})))
        with self.assertRaisesRegex(ValueError, 'exact video'):
            self.analyze(touch_report_path=touch)
        touch.write_text(json.dumps(dict(video={'manifest_sha256': digest(self.video)}, joint_mapping={})))
        with self.assertRaisesRegex(ValueError, 'mappings'):
            self.analyze(touch_report_path=touch)
        touch.write_text(json.dumps(dict(video={'manifest_sha256': digest(self.video)},
            joint_mapping=self.manifest['joint_mapping'], descent_traces=[])))
        self.assertEqual(self.analyze(touch_report_path=touch)['status'], 'completed')

    def test_trace_mapping_and_clock_cannot_hide_behind_top_level(self):
        trace = dict(started_monotonic_s=10., clock_domain='adapter_monotonic',
                     joint_mapping={}, reads=[])
        report = dict(joint_mapping=self.manifest['joint_mapping'], descent_traces=[trace])
        with self.assertRaisesRegex(ValueError, 'trace/video joint mappings'):
            _joint_samples(self.manifest, report)
        trace['joint_mapping'] = self.manifest['joint_mapping']
        trace['clock_domain'] = 'wall_clock'
        with self.assertRaisesRegex(ValueError, 'clock'):
            _joint_samples(self.manifest, report)

    def test_projection_uses_frozen_calibration(self):
        self.run_sam()
        transform = np.eye(4)
        transform[2,3] = -1
        self.manifest['camera_calibration'] = {'cameras': {'fixed': {
            'image_size': [64,64], 'T_base_camera': transform.tolist(), 'K': [[100,0,32],[0,100,32],[0,0,1]], 'distortion': []}}}
        self.save_manifest()
        self.features[0].update(link='base_link', point_in_link_m=[0,0,0])
        projection = self.analyze()['frames'][0]['features'][0]['projection']
        self.assertEqual(projection['error_px'], 0)
        self.assertIsNone(projection['physical_acceptance'])

    def test_invalid_projection_retains_failed_evidence(self):
        self.run_sam()
        transform = np.eye(4)
        transform[2,3] = -1
        self.manifest['camera_calibration'] = {'cameras': {'fixed': {
            'image_size': [64,64], 'T_base_camera': transform.tolist(),
            'K': [[100,0,32],[0,100,32],[0,0,1]], 'distortion': [float('nan'),0,0,0,0]}}}
        self.save_manifest()
        self.features[0].update(link='base_link', point_in_link_m=[0,0,0])
        with self.assertRaisesRegex(ValueError, 'distortion'):
            self.analyze()
        report = json.loads((self.root/'review/evidence.json').read_text())
        self.assertEqual(report['status'], 'failed')
        self.assertIsNone(report['physical_acceptance'])

    def test_seed_validation_and_html_escape(self):
        self.run_sam()
        self.features[0]['point_px'] = [-.4,32]
        with self.assertRaises(ValueError):
            self.analyze()
        self.features[0]['point_px'] = [32,32]
        self.features[0]['feature_id'] = '</script><script>alert(1)</script>'
        self.analyze()
        page = (self.root/'review/review.html').read_text()
        self.assertNotIn('</script><script>', page)

    def test_joint_brackets_conflicts_and_branch(self):
        samples = [dict(t=1., q_rad=[0]*5, gripper_rad=0), dict(t=1.2, q_rad=[.2]*5, gripper_rad=.2)]
        self.assertEqual(align_joints(samples, .9, .3)['status'], 'unknown')
        np.testing.assert_allclose(align_joints(samples, 1.1, .3)['q_estimate_rad'], [.1]*5)
        self.assertEqual(align_joints(samples, 1.1, .1)['reason'], 'joint_bracket_too_wide')
        self.assertEqual(align_joints(samples, 1.1, .3, [1.05])['reason'], 'conflicting_joint_samples')
        samples[1]['q_rad'] = [6.2]*5
        self.assertEqual(align_joints(samples, 1.1, .3)['reason'], 'angular_branch_discontinuity')
        manifest = {'cameras': {'a': {'frames': [{'joint_sample': samples[0]}]},
                                'b': {'frames': [{'joint_sample': dict(samples[0], gripper_rad=1)}]}}}
        clean, conflicts = _joint_samples(manifest, None)
        self.assertEqual(clean, [])
        self.assertEqual(conflicts, [1.])

    def test_video_evidence_does_not_catch_keyboard_interrupt_and_guards_report_write_failure(self):
        self.run_sam()
        with patch("ni_rua.robot.calibration.evidence.video_evidence.cv2.imwrite", side_effect=KeyboardInterrupt()):
            with self.assertRaises(KeyboardInterrupt):
                self.analyze()
        self.assertFalse((self.root / "review/evidence.json").exists())
        self.assertEqual(list((self.root / "review").glob(".evidence.json.*.tmp")), [])

        self.features[0]["point_px"] = [-0.4, 32]
        with self.assertRaises(ValueError):
            self.analyze()

        # Trigger failure inside the analysis loop after output_dir is created, and fail os.replace on evidence.json.
        self.features[0]["point_px"] = [32, 32]
        with patch("ni_rua.robot.calibration.evidence.video_evidence.cv2.imwrite", return_value=False), \
             patch("ni_rua.robot.calibration.evidence.video_evidence.os.replace", side_effect=OSError("readonly fs")):
            with self.assertRaisesRegex(OSError, "failed to save review overlay") as cm:
                analyze_video(
                    self.root / "video/manifest.json",
                    "fixed",
                    self.root / "sam/tracking.json",
                    self.features,
                    self.root / "review_write_err",
                )
        self.assertTrue(any("evidence_report_write_failed" in n for n in getattr(cm.exception, "__notes__", ())))
        self.assertEqual(list((self.root / "review_write_err").glob(".evidence.json.*.tmp")), [])

# === tests/test_so101_video_workflow.py ===

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import threading
import time
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

from ni_rua.robot.calibration.workflow.runner import CalibrationWorkflow
from ni_rua.robot.backends.so101.deployment import ARM_JOINTS, FIELDS, validate_deployment
from robot.so101.test_so101_backend import EXAMPLE


class VideoWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.flow = object.__new__(CalibrationWorkflow)
        self.flow.run = {"directory": str(self.directory)}
        self.flow.settings = {"touch_video_fps": 10.0, "touch_video_max_frames": 600,
                              "touch_video_max_duration_s": 60.0}
        self.flow.adapter = SimpleNamespace(cameras={"wrist": object(), "fixed": object()},
            deployment={"joint_mapping": {n: {"sign": 1, "offset_rad": 0.0} for n in ARM_JOINTS}})
        self.physical = {"status": "unknown", "reason": "stop_confirmation_missing", "physics_advanced": True,
                         "stop_confirmation": {"confirmed": False}, "touches": []}
        self.touch = Mock(side_effect=lambda *a, **kw: deepcopy(self.physical))
        self.recorder = Mock()
        self.recorder.stop.return_value = {"status": "completed", "cameras": {"fixed": {"frames": 3}}}
        def recorder(adapter, directory, **settings):
            directory.mkdir()
            return self.recorder
        self.factory = Mock(side_effect=recorder)
        self.addCleanup(patch.stopall)
        patch("ni_rua.robot.calibration.workflow.touch_off.touch_points", side_effect=self.touch).start()
        patch.dict("sys.modules", {"ni_rua.robot.calibration.workflow.video":
                                  SimpleNamespace(CalibrationVideoRecorder=self.factory)}).start()

    def acquire(self):
        return self.flow._touch([[0.2, 0]], lambda _: None, "S1")

    def assert_physical_unchanged(self, report):
        self.assertEqual({k: report[k] for k in self.physical}, self.physical)
        self.assertEqual(self.touch.call_count, 1)

    def test_records_unique_attempts_and_exposes_relative_paths(self):
        def stopped():
            directory = self.factory.call_args.args[1]
            (directory / "manifest.json").write_text('{"schema":"synthetic-video"}\n')
            return {"status": "completed", "sealed": True, "manifest_path": str(directory / "manifest.json"),
                    "cameras": {"fixed": {"frames": 3}}}
        self.recorder.stop.side_effect = stopped
        report = self.acquire()
        self.assert_physical_unchanged(report)
        video = report["video"]
        self.assertEqual(video["contact_evidence"], "unknown")
        self.assertFalse(Path(video["directory"]).is_absolute())
        self.assertFalse(Path(video["manifest_path"]).is_absolute())
        import hashlib
        self.assertEqual(video["manifest_sha256"], hashlib.sha256(
            (self.directory / video["manifest_path"]).read_bytes()).hexdigest())
        self.assertEqual(self.factory.call_args.kwargs, {"fps": 10.0, "max_frames": 600, "max_duration_s": 60.0})
        self.assertEqual(json.loads((self.directory / "S1_touches.json").read_text())["video"], video)
        self.assertTrue((self.directory / video["directory"] / "video_summary.json").is_file())
        second = self.acquire()
        self.assertNotEqual(second["video"]["directory"], video["directory"])
        self.assertEqual(self.recorder.stop.call_count, 2)

    def test_disabled_and_missing_camera_never_construct_recorder(self):
        self.flow.settings["touch_video_fps"] = 0.0
        self.assertEqual(self.acquire()["video"]["status"], "disabled")
        self.flow.settings["touch_video_fps"] = 10.0
        self.flow.adapter.cameras.pop("wrist")
        video = self.acquire()["video"]
        self.assertEqual(video["status"], "unavailable")
        self.assertEqual(video["termination_reason"], "both_cameras_required")
        self.factory.assert_not_called()

    def test_recording_start_failure_preserves_physical_unknown_and_still_stops(self):
        self.recorder.start.side_effect = OSError("synthetic recorder failure")
        report = self.acquire()
        self.assert_physical_unchanged(report)
        self.assertEqual(report["video"]["status"], "failed")
        self.assertIn("start:", report["video"]["errors"][0])
        self.recorder.stop.assert_called_once()

    def test_recording_stop_failure_preserves_physical_unknown(self):
        self.recorder.stop.side_effect = OSError("synthetic writer failure")
        report = self.acquire()
        self.assert_physical_unchanged(report)
        self.assertEqual(report["video"]["status"], "failed")
        self.assertEqual(report["video"]["termination_reason"], "recorder_stop_failed")

    def test_truncated_recording_does_not_change_motion_completion(self):
        self.physical.update(status="completed", reason=None, stop_confirmation={"confirmed": True})
        self.recorder.stop.return_value = {"status": "truncated", "termination_reason": "max_frames"}
        report = self.acquire()
        self.assert_physical_unchanged(report)
        self.assertEqual(report["video"]["status"], "truncated")

    def test_pending_writer_never_hashes_an_unsealed_manifest(self):
        self.recorder.stop.return_value = {"status": "unknown", "termination_reason": "writer_shutdown_pending",
                                           "sealed": False,
                                           "manifest_path": str(self.directory / "not-yet-written.json")}
        report = self.acquire()
        self.assert_physical_unchanged(report)
        self.assertEqual(report["video"]["status"], "unknown")
        self.assertIsNone(report["video"]["manifest_sha256"])

    def test_touch_exception_stops_recorder_and_keeps_visual_diagnostics(self):
        self.touch.side_effect = RuntimeError("motion read failed")
        with self.assertRaisesRegex(RuntimeError, "motion read failed"):
            self.acquire()
        self.recorder.stop.assert_called_once()
        self.assertEqual(len(list((self.directory / "touch_video").glob("*/video_summary.json"))), 1)

    def test_video_summary_disk_failure_never_replaces_physical_receipt(self):
        from ni_rua.robot.calibration.workflow.runner import write_json

        def write(path, value):
            if path.name == "video_summary.json":
                raise OSError("synthetic video directory unavailable")
            return write_json(path, value)

        with patch("ni_rua.robot.calibration.workflow.runner.write_json", side_effect=write):
            report = self.acquire()
        self.assert_physical_unchanged(report)
        self.assertEqual(report["video"]["status"], "failed")
        self.assertIn("summary:", report["video"]["errors"][0])

    def test_touch_archive_disk_failure_keeps_physical_receipt_and_marks_evidence_missing(self):
        from ni_rua.robot.calibration.workflow.runner import write_json
        stale = self.directory / "S1_touches.json"
        stale.write_text('{"old_attempt":true}')

        def write(path, value):
            if path.name == "S1_touches.json":
                raise OSError("synthetic disk full")
            return write_json(path, value)

        with patch("ni_rua.robot.calibration.workflow.runner.write_json", side_effect=write):
            report = self.acquire()
        self.assert_physical_unchanged(report)
        self.assertFalse(report["evidence_persisted"])
        self.assertIn("synthetic disk full", report["archive_error"])
        self.assertEqual(json.loads(stale.read_text()), {"old_attempt": True})


class VideoDeploymentTests(unittest.TestCase):
    def test_old_deployments_get_defaults_but_explicit_zero_is_preserved(self):
        document = json.loads(EXAMPLE.read_text())
        keys = ("touch_video_fps", "touch_video_max_frames", "touch_video_max_duration_s")
        for key in keys:
            document["calibration"].pop(key, None)
        loaded = validate_deployment(document, base=EXAMPLE.parent)
        self.assertEqual([loaded["calibration"][key] for key in keys], [10.0, 600, 60.0])
        document["calibration"].update(touch_video_fps=0, touch_video_max_frames=123, touch_video_max_duration_s=12.5)
        loaded = validate_deployment(document, base=EXAMPLE.parent)
        self.assertEqual([loaded["calibration"][key] for key in keys], [0, 123, 12.5])
        for key in keys:
            self.assertIn("default", FIELDS["calibration"][key])
            self.assertTrue(FIELDS["calibration"][key]["unit"])

    def test_invalid_recording_limits_are_rejected(self):
        for key, value in (("touch_video_fps", -1), ("touch_video_max_frames", 0),
                           ("touch_video_max_frames", 1.5), ("touch_video_max_duration_s", 0)):
            with self.subTest(key=key, value=value):
                document = json.loads(EXAMPLE.read_text())
                document["calibration"][key] = value
                with self.assertRaisesRegex(ValueError, key):
                    validate_deployment(document, base=EXAMPLE.parent)


class RealRecorderWorkflowTests(unittest.TestCase):
    def test_actual_recorder_creates_attempt_directory_and_seals_bound_manifest(self):
        import hashlib
        import numpy as np
        from ni_rua.robot.calibration.workflow.video import CalibrationVideoRecorder
        sampled = threading.Event()

        class Camera:
            def __init__(self, name):
                self.name = name

            def latest(self):
                return np.zeros((8, 8, 3), dtype=np.uint8), 1.0

        write_manifest = CalibrationVideoRecorder._write_manifest

        def persisted(recorder):
            write_manifest(recorder)
            if all(recorder.manifest["cameras"][name]["frames"] for name in ("wrist", "fixed")):
                sampled.set()

        with tempfile.TemporaryDirectory() as temporary:
            flow = object.__new__(CalibrationWorkflow)
            flow.run = {"directory": temporary}
            flow.settings = {"touch_video_fps": 10.0, "touch_video_max_frames": 600, "touch_video_max_duration_s": 60.0}
            flow.adapter = SimpleNamespace(cameras={n: Camera(n) for n in ("wrist", "fixed")},
                clock=SimpleNamespace(now=time.monotonic), last_sample=None,
                deployment={"joint_mapping": {n: {"sign": 1, "offset_rad": 0.0} for n in ARM_JOINTS}})

            def physical(*args, **kwargs):
                self.assertTrue(sampled.wait(2), "background sampler never reached both cameras")
                return {"status": "stopped", "physics_advanced": False, "stop_confirmation": {"confirmed": True}}

            with patch("ni_rua.robot.calibration.workflow.touch_off.touch_points", side_effect=physical), \
                 patch.object(CalibrationVideoRecorder, "_write_manifest", persisted):
                report = flow._touch([[0.2, 0]], lambda _: None, "S1")
            self.assertEqual(report["status"], "stopped")
            video = report["video"]
            self.assertEqual(video["status"], "completed", video)
            self.assertTrue(video["sealed"])
            manifest_path = Path(temporary) / video["manifest_path"]
            self.assertEqual(video["manifest_sha256"], hashlib.sha256(manifest_path.read_bytes()).hexdigest())
            manifest = json.loads(manifest_path.read_text())
            for name in ("wrist", "fixed"):
                self.assertEqual(len(manifest["cameras"][name]["frames"]), 1)


if __name__ == "__main__":
    unittest.main()
