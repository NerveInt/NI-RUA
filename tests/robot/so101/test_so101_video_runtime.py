"""Synthetic camera/driver + real ffmpeg: motion-only archive, no hardware calls."""

NI_RUA_TEST_TIER = "component"

# === tests/test_so101_motion_video.py ===

import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from ni_rua.robot.backends.so101.driver import FakeCameraSource, FakeClock
from ni_rua.robot.backends.so101.motion_video import MotionVideoRecorder, _CameraVideo
from robot.so101.test_so101_backend import make_adapter, move_action, deployment


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class MotionVideoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.clock = FakeClock()
        configs = {"fixed": dict(width=64, height=48, fps=30), "wrist": dict(width=32, height=24, fps=15)}
        self.adapter = SimpleNamespace(run_dir=Path(temporary.name), clock=self.clock, control_step=0,
                                       deployment={"cameras": configs, "camera": dict(video_crf=23, video_queue_mb=1)},
                                       cameras={name: FakeCameraSource(name, clock=self.clock, width=c["width"], height=c["height"])
                                                for name, c in configs.items()})
        self.recorder = MotionVideoRecorder(self.adapter)
        self.addCleanup(self.recorder.save)

    def emit(self, color, count=1):
        for _ in range(count):
            for source in self.adapter.cameras.values():
                pixels = np.full((source.height, source.width, 3), color, dtype=np.uint8)
                source._deliver_frame(pixels, self.clock.now(), self.clock.now())
            self.clock.sleep(1 / 30)

    def test_all_cameras_keep_native_resolution_fps_and_omit_long_idle(self):
        self.recorder.record()
        self.emit([0, 255, 0], 4)  # idle, never stored
        self.recorder.begin("action-a")
        self.emit([255, 0, 0], 6)
        self.recorder.end()
        self.clock.sleep(3600)
        self.emit([0, 255, 0], 4)
        self.recorder.begin("action-b")
        self.emit([0, 0, 255], 6)
        self.recorder.end()
        self.clock.sleep(12.5)  # post-motion idle before seal
        report = self.recorder.save()
        self.assertIs(report, self.recorder.save())
        self.assertEqual(len(report["videos"]), 2)
        self.assertEqual([s["action_id"] for s in report["segments"]], ["action-a", "action-b"])
        self.assertLessEqual(report["recording_started_monotonic_s"], report["segments"][0]["start_monotonic_s"])
        self.assertGreater(report["segments"][0]["start_monotonic_s"] - report["recording_started_monotonic_s"], 0.1)
        self.assertGreaterEqual(report["recording_ended_monotonic_s"], report["segments"][-1]["end_monotonic_s"] + 12.0)
        self.assertNotIn("composite", report, "worker save() stays fast and leaves composition to parent/CLI")
        for item in report["videos"]:
            self.assertEqual(item["status"], "complete")
            self.assertEqual(item["frames"], 12)
            self.assertEqual(item["segment_frame_counts"], [6, 6])
            config = self.adapter.deployment["cameras"][item["camera_id"]]
            self.assertEqual((item["width"], item["height"], item["fps"]), (config["width"], config["height"], config["fps"]))
            rows = [json.loads(row) for row in Path(item["frame_mapping_path"]).read_text().splitlines()]
            self.assertGreater(rows[6]["capture_monotonic_s"] - rows[5]["capture_monotonic_s"], 3600)
            self.assertAlmostEqual(rows[6]["pts_s"] - rows[5]["pts_s"], 1 / item["fps"])
            raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", item["video_path"], "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
            images = np.frombuffer(raw, dtype=np.uint8).reshape(12, item["height"], item["width"], 3)
            self.assertGreater(images[:6, ..., 0].mean(), 240)
            self.assertGreater(images[6:, ..., 2].mean(), 240)
            self.assertLess(images[..., 1].mean(), 10)

    def test_pause_resume_retains_frames_and_new_export_never_overwrites(self):
        self.recorder.record()
        self.recorder.begin("one")
        self.emit([255, 0, 0])
        self.recorder.end()
        self.recorder.record(False)
        self.recorder.begin("disabled")
        self.emit([0, 255, 0])
        self.recorder.end()
        self.recorder.record(True)
        self.recorder.begin("two")
        self.emit([0, 0, 255])
        self.recorder.end()
        first = self.recorder.save()
        first_path = Path(first["videos"][0]["video_path"])
        original = first_path.read_bytes()
        self.recorder.record(True)
        self.recorder.begin("three")
        self.emit([255, 255, 255])
        second = self.recorder.save()
        self.assertEqual(first_path.read_bytes(), original)
        self.assertNotEqual(first_path, Path(second["videos"][0]["video_path"]))
        self.assertEqual(first["videos"][0]["frames"], 2)

    def test_empty_missing_and_invalid_frames_are_not_success(self):
        self.recorder.record()
        self.recorder.begin("short-action")
        source = self.adapter.cameras["fixed"]
        source._deliver_frame(np.zeros((1, 1, 3), dtype=np.uint8), self.clock.now(), self.clock.now())
        report = self.recorder.save()
        self.assertEqual(report["videos"], [])
        self.assertTrue(all(item["status"] == "empty" for item in report["cameras"]))
        self.assertEqual(next(c for c in report["cameras"] if c["camera_id"] == "fixed")["invalid_frames"], 1)
        self.assertTrue(all(item["segment_frame_counts"] == [0] for item in report["cameras"]))

    def test_missing_camera_does_not_disable_healthy_camera(self):
        self.adapter.cameras.pop("fixed")
        self.recorder.record()
        self.recorder.begin("one")
        self.emit([200, 200, 200], 2)
        report = self.recorder.save()
        self.assertEqual([item["camera_id"] for item in report["videos"]], ["wrist"])
        self.assertEqual(next(c for c in report["cameras"] if c["camera_id"] == "fixed")["status"], "unavailable")

    def test_odd_dimensions_and_negotiated_fractional_fps_are_preserved(self):
        self.adapter.deployment["cameras"]["wrist"].update(width=33, height=25)
        source = self.adapter.cameras["wrist"]
        source.width, source.height = 33, 25
        source.recording_fps = 30000 / 1001
        source.recording_fps_source = "camera_negotiated"
        self.recorder.record()
        self.recorder.begin("one")
        self.emit([150, 150, 150], 4)
        camera = next(c for c in self.recorder.save()["videos"] if c["camera_id"] == "wrist")
        self.assertEqual((camera["width"], camera["height"]), (33, 25))
        self.assertEqual(camera["fps"], source.recording_fps)
        self.assertEqual(camera["status"], "complete")

    def test_encoder_failure_is_reported_and_does_not_escape_capture(self):
        with patch("ni_rua.robot.backends.so101.motion_video.subprocess.Popen", side_effect=FileNotFoundError("ffmpeg absent")):
            self.recorder.record()
            self.recorder.begin("one")
            self.emit([0, 0, 0])
            result = self.recorder.save()
        self.assertEqual(result["videos"], [])
        self.assertTrue(all(item["status"] == "failed" for item in result["cameras"]))
        self.assertTrue(all("FileNotFoundError" in item["error"] for item in result["cameras"]))

    def test_bounded_queue_drops_are_explicit_without_blocking(self):
        release = threading.Event()
        original = _CameraVideo._encode
        def blocked(stream):
            release.wait(5)
            original(stream)
        with patch.object(_CameraVideo, "_encode", blocked):
            self.recorder.record()
            self.recorder.begin("overflow")
            for stream in self.recorder.streams.values():
                stream.limit = stream.width * stream.height * 3
            self.emit([100, 100, 100], 4)
            release.set()
            report = self.recorder.save()
        self.assertTrue(all(item["dropped_frames"] == 3 for item in report["cameras"]))
        self.assertTrue(all(item["status"] == "incomplete" for item in report["cameras"]))


class ExecutionVideoTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.adapter, self.driver, self.clock = make_adapter(temporary.name)
        self.addCleanup(self.adapter.close)

    def test_rejection_and_preview_do_not_open_intervals_and_exceptions_close_gate(self):
        recorder = self.adapter.motion_video
        recorder.record()
        result = self.adapter.execute({"bad": True}, lambda _: None, threading.Event())
        self.assertEqual(result["status"], "rejected")
        self.assertEqual(recorder.segments, [])
        action = move_action(self.adapter)
        original = self.driver.send_goal
        def fault_after_dispatch(*args):
            original(*args)
            raise RuntimeError("motor failure")
        with patch.object(self.driver, "send_goal", fault_after_dispatch):
            result = self.adapter.execute(action, lambda _: None, threading.Event())
        self.assertIsNone(recorder.active)
        self.assertEqual(len(recorder.segments), 1)
        self.assertIsNotNone(recorder.segments[0]["end_monotonic_s"])
        self.assertIn(result["status"], ("unknown", "failed"))
        self.assertEqual(result["stop_state"], "unknown")
        # Even an unexpected post-dispatch exception closes capture without retry.
        with patch.object(self.adapter, "_execute", side_effect=RuntimeError("unexpected")):
            with self.assertRaisesRegex(RuntimeError, "unexpected"):
                self.adapter.execute(action, lambda _: None, threading.Event())
        self.assertIsNone(self.adapter._video_action_id)

    def test_success_ends_before_terminal_observation_and_cancel_preserves_receipt(self):
        recorder = self.adapter.motion_video
        recorder.record()
        original = self.adapter.observe
        def observe():
            self.assertIsNone(recorder.active)
            return original()
        with patch.object(self.adapter, "observe", observe):
            result = self.adapter.execute(move_action(self.adapter), lambda _: None, threading.Event())
        self.assertEqual(result["status"], "completed")
        self.assertEqual(len(recorder.segments), 1)
        cancel = threading.Event()
        def progress(update):
            if update["phase"] == "executing":
                cancel.set()
        result = self.adapter.execute(move_action(self.adapter), progress, cancel)
        self.assertEqual(result["status"], "stopped")
        self.assertIsNone(recorder.active)

    def test_encoding_failure_during_real_execution_path_keeps_physical_receipt(self):
        for name, camera in self.adapter.cameras.items():
            self.adapter.deployment["cameras"][name].update(width=camera.width, height=camera.height)
        self.adapter.motion_video.record()
        tick = self.adapter._tick
        def frames_during_tick(run, period):
            sample = tick(run, period)
            for camera in self.adapter.cameras.values():
                camera._frame(self.clock.now())
            return sample
        before = self.driver.sends
        with patch.object(self.adapter, "_tick", frames_during_tick), \
                patch("ni_rua.robot.backends.so101.motion_video.subprocess.Popen", side_effect=OSError("encoder failed")):
            receipt = self.adapter.execute(move_action(self.adapter), lambda _: None, threading.Event())
            report = self.adapter.motion_video.save()
        self.assertEqual(receipt["status"], "completed")
        self.assertEqual(self.driver.sends - before, receipt["executed_native_steps"])
        self.assertTrue(all(camera["status"] == "failed" for camera in report["cameras"]))
        self.assertIsNotNone(receipt["terminal_observation"])

    def test_new_camera_settings_default_and_validate_for_old_deployments(self):
        from robot.so101.test_so101_backend import EXAMPLE, validate_deployment
        old = json.loads(EXAMPLE.read_text())
        old["camera"].pop("video_crf")
        old["camera"].pop("video_queue_mb")
        settings = validate_deployment(old, base=EXAMPLE.parent)["camera"]
        self.assertEqual((settings["video_crf"], settings["video_queue_mb"]), (23, 64))
        for key, value in (("video_crf", 52), ("video_crf", True), ("video_queue_mb", 0)):
            with self.assertRaises(ValueError):
                deployment(**{f"camera.{key}": value})

# === tests/test_so101_motion_video_composite.py ===

import contextlib
import io
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from types import SimpleNamespace
import unittest

import numpy as np

from ni_rua.robot.backends.so101.driver import FakeCameraSource, FakeClock
from ni_rua.robot.backends.so101.motion_video import MotionVideoRecorder
from ni_rua.robot.backends.so101 import motion_video_composite

RED, GREEN, BLUE, YELLOW = [255, 0, 0], [0, 255, 0], [0, 0, 255], [255, 255, 0]
CAMERAS = {"fixed": dict(width=160, height=120, fps=30), "wrist": dict(width=128, height=96, fps=30)}


def sealed_recording(root, script):
    """Record ``script`` = [(action_id, [(camera, offset_s, rgb), ...]), ...] and seal it.

    Each frame's host capture time is the action start plus ``offset_s`` (s), so
    cameras can be deliberately unsynchronised. Returns the MotionVideoRecorder.save() report.
    """
    clock = FakeClock()
    adapter = SimpleNamespace(run_dir=Path(root), clock=clock, control_step=0,
                              deployment={"cameras": CAMERAS, "camera": dict(video_crf=23, video_queue_mb=4)},
                              cameras={name: FakeCameraSource(name, clock=clock, width=c["width"], height=c["height"])
                                       for name, c in CAMERAS.items()})
    recorder = MotionVideoRecorder(adapter)
    recorder.record()
    for action_id, frames in script:
        recorder.begin(action_id)
        begin = clock.now()
        for camera, offset_s, color in sorted(frames, key=lambda item: item[1]):
            clock.sleep(max(0.0, begin + offset_s - clock.now()))
            source = adapter.cameras[camera]
            source._deliver_frame(np.full((source.height, source.width, 3), color, np.uint8), clock.now(), clock.now())
        recorder.end()
        clock.sleep(600)  # model thinking between actions is never recorded
    return recorder.save()


def decode(path, width, height):
    raw = subprocess.check_output(["ffmpeg", "-v", "error", "-i", str(path), "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1"])
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width, 3)


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "ffmpeg/ffprobe required")
class CompositeTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = temporary.name

    def test_tiles_align_by_capture_time_and_missing_camera_is_black(self):
        def script():
            yield ("action-a", [("fixed", k / 30, RED) for k in range(6)]
                   + [("wrist", (2 * k + 0.5) / 30, color) for k, color in enumerate((GREEN, BLUE, YELLOW))])
            yield ("action-b", [("wrist", k / 30, BLUE) for k in range(3)])
        report = sealed_recording(self.root, list(script()))
        self.assertEqual([camera["frames"] for camera in report["cameras"]], [6, 6])
        result = motion_video_composite.compose(report)
        self.assertEqual(result["status"], "complete", result)
        self.assertEqual(result["camera_order"], ["fixed", "wrist"])
        expected_h = 120 + motion_video_composite.HUD_HEIGHT
        self.assertEqual((result["width"], result["height"], result["tile_height"], result["timeline_hud_height"], result["frames"], result["fps"]),
                         (288, expected_h, 120, motion_video_composite.HUD_HEIGHT, 9, 30.0))
        self.assertAlmostEqual(result["duration_s"], 9 / 30, places=3)
        summary = result["timeline_summary"]
        self.assertEqual(summary["bounds_source"], "manifest")
        self.assertEqual([iv["kind"] for iv in summary["intervals"]], ["motion", "thinking", "motion", "thinking"])
        self.assertEqual([iv["action_id"] for iv in summary["intervals"]], ["action-a", None, "action-b", None])
        self.assertAlmostEqual(summary["total_motion_s"] + summary["total_thinking_skipped_s"], summary["total_task_s"], places=2)
        self.assertGreater(summary["total_thinking_skipped_s"], 1199.0)
        rows = [json.loads(line) for line in Path(result["frame_mapping_path"]).read_text().splitlines()]
        # Segment a: fixed (6 frames) is the reference; wrist holds its latest frame captured at or
        # before each fixed frame, showing its first segment frame until that frame is captured.
        self.assertEqual([row["reference_camera"] for row in rows], ["fixed"] * 6 + ["wrist"] * 3)
        self.assertEqual([row["source_frames"]["wrist"] for row in rows[:6]], [0, 0, 0, 1, 1, 2])
        self.assertEqual([row["source_frames"]["fixed"] for row in rows[:6]], list(range(6)))
        # Segment b: fixed has no frame, its tile stays black instead of reusing segment a.
        self.assertEqual([row["source_frames"] for row in rows[6:]], [dict(fixed=None, wrist=index) for index in (3, 4, 5)])
        self.assertEqual([row["action_id"] for row in rows], ["action-a"] * 6 + ["action-b"] * 3)
        self.assertEqual([row["pts_s"] for row in rows], [index / 30 for index in range(9)])
        self.assertGreater(rows[6]["task_elapsed_s"] - rows[5]["task_elapsed_s"], 599.0)
        images = decode(result["video_path"], 288, expected_h)
        self.assertEqual(len(images), 9)
        fixed_patch, wrist_patch = images[:, 70:90, 110:150], images[:, 50:80, 210:280]
        for index, expected in enumerate([GREEN, GREEN, GREEN, BLUE, BLUE, YELLOW, BLUE, BLUE, BLUE]):
            self.assertTrue(np.allclose(wrist_patch[index].mean(axis=(0, 1)), expected, atol=40), (index, wrist_patch[index].mean(axis=(0, 1))))
        self.assertTrue(np.allclose(fixed_patch[:6].mean(axis=(0, 1, 2)), RED, atol=40))
        self.assertLess(fixed_patch[6:].mean(), 20)
        hud_band = images[:, 120:, :]
        self.assertGreater(hud_band.max(), 180)
        # Even at narrow 288 px width, both Row 1 (elapsed) and Row 2 (active move + skip) render text.
        self.assertGreater(images[0, 126:142, :].max(), 140)
        self.assertGreater(images[0, 148:164, :].max(), 140)
        self.assertGreater(np.abs(hud_band[6].astype(np.int16) - hud_band[0].astype(np.int16)).mean(), 0.2)
        # Repeated calls return the saved result; nothing is re-encoded.
        before = Path(result["video_path"]).stat().st_mtime_ns
        self.assertEqual(motion_video_composite.compose(report), result)
        self.assertEqual(Path(result["video_path"]).stat().st_mtime_ns, before)
        self.assertFalse(Path(result["video_path"]).with_suffix(".partial").exists())

    def test_empty_recording_and_missing_ffmpeg_are_reported(self):
        report = sealed_recording(self.root, [])
        self.assertEqual(report["segments"], [])
        result = motion_video_composite.compose(report)
        self.assertEqual(result["status"], "empty")
        self.assertNotIn("video_path", result)
        self.assertEqual({item["camera_id"] for item in result["missing_cameras"]}, {"fixed", "wrist"})
        timed_empty = dict(report, recording_started_monotonic_s=10.0, recording_ended_monotonic_s=40.0)
        timed_res = motion_video_composite.compose(timed_empty)
        self.assertEqual(timed_res["status"], "empty")
        self.assertAlmostEqual(timed_res["timeline_summary"]["total_task_s"], 30.0, places=2)
        report = sealed_recording(Path(self.root) / "second", [("action-a", [("fixed", 0.0, RED), ("wrist", 0.0, RED)])])
        result = motion_video_composite.compose(report, ffmpeg="/nonexistent/ffmpeg")
        self.assertEqual(result["status"], "failed")
        self.assertIn("FileNotFoundError", result["error"])
        self.assertNotIn("video_path", result)

    def test_cli_composes_a_sealed_manifest(self):
        report = sealed_recording(self.root, [("action-a", [("fixed", 0.0, RED), ("wrist", 0.01, BLUE)])])
        directory = Path(report["cameras"][0]["frame_mapping_path"]).parent
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(motion_video_composite.main([str(directory)]), 0)
        self.assertEqual(json.loads(output.getvalue())["status"], "complete")
        self.assertTrue((directory / "combined.mp4").is_file())
        self.assertEqual(json.loads((directory / "combined.json").read_text())["frames"], 1)

    def test_overwrite_relocation_and_self_healing_cache(self):
        report = sealed_recording(self.root, [("action-a", [("fixed", 0.0, RED), ("wrist", 0.01, BLUE)])])
        orig_dir = Path(report["cameras"][0]["frame_mapping_path"]).parent
        first = motion_video_composite.compose(report)
        self.assertEqual(first["status"], "complete")
        orig_mp4 = orig_dir / "combined.mp4"
        first_mtime = orig_mp4.stat().st_mtime_ns

        # Valid cache hit returns without re-encoding.
        self.assertEqual(motion_video_composite.compose(report)["video_byte_size"], first["video_byte_size"])
        self.assertEqual(orig_mp4.stat().st_mtime_ns, first_mtime)

        # Truncated/corrupted MP4 is detected via byte-size check and self-healed automatically.
        orig_mp4.write_bytes(b"corrupted")
        healed = motion_video_composite.compose(report)
        self.assertEqual(healed["status"], "complete")
        self.assertGreater(orig_mp4.stat().st_size, 100)

        # Pre-HUD legacy combined.json (missing timeline_hud_height / timeline_summary) is upgraded automatically.
        legacy_json = {k: v for k, v in healed.items() if k not in ("timeline_hud_height", "timeline_summary", "video_byte_size")}
        (orig_dir / "combined.json").write_text(json.dumps(legacy_json))
        upgraded = motion_video_composite.compose(report)
        self.assertEqual(upgraded["timeline_hud_height"], motion_video_composite.HUD_HEIGHT)
        self.assertIsInstance(upgraded["timeline_summary"], dict)

        # overwrite=True forces regeneration even when valid.
        before_overwrite = orig_mp4.read_bytes()
        regenerated = motion_video_composite.compose(report, overwrite=True)
        self.assertEqual(regenerated["status"], "complete")
        self.assertEqual(len(orig_mp4.read_bytes()), len(before_overwrite))

        # Relocated directory WITHOUT --overwrite rebases cached paths to relocated dir.
        relocated = Path(self.root) / "copied_recording"
        shutil.copytree(orig_dir, relocated)
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(motion_video_composite.main([str(relocated)]), 0)
        rebased = json.loads(output.getvalue())
        self.assertEqual(Path(rebased["video_path"]), relocated / "combined.mp4")
        self.assertTrue(all(Path(s["video_path"]).parent == relocated for s in rebased["sources"]))

        # Relocated directory WITH --overwrite writes only into relocated, never touching orig_dir.
        orig_mtime = orig_mp4.stat().st_mtime_ns
        with contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(motion_video_composite.main([str(relocated), "--overwrite"]), 0)
        self.assertEqual(orig_mp4.stat().st_mtime_ns, orig_mtime)

        # Incomplete relocated copy (missing mapping file) fails cleanly instead of mixing paths.
        incomplete = Path(self.root) / "incomplete_copy"
        shutil.copytree(orig_dir, incomplete)
        (incomplete / "wrist.frames.jsonl").unlink()
        bad = motion_video_composite.compose(report, directory=incomplete, overwrite=True)
        self.assertEqual(bad["status"], "failed")
        self.assertIn("FileNotFoundError", bad["error"])

    def test_backward_compat_bounds_from_wal_multiturn_events_and_fallback(self):
        import sqlite3
        run_root = Path(self.root) / "legacy_run"
        robot_root = run_root / "private" / "robot"
        report = sealed_recording(
            robot_root,
            [
                ("a4", [("fixed", k / 30, RED) for k in range(3)] + [("wrist", k / 30, GREEN) for k in range(3)]),
                ("a8", [("fixed", k / 30, BLUE) for k in range(3)] + [("wrist", k / 30, YELLOW) for k in range(3)]),
            ],
        )
        legacy_report = dict(report)
        legacy_report.pop("recording_started_monotonic_s", None)
        legacy_report.pop("recording_ended_monotonic_s", None)

        # 1. Without events.sqlite: falls back cleanly to [segments[0].start, segments[-1].end].
        fallback = motion_video_composite.compose(legacy_report, overwrite=True)
        self.assertEqual(fallback["status"], "complete")
        fb_summary = fallback["timeline_summary"]
        self.assertEqual(fb_summary["bounds_source"], "segments_fallback")
        expected_span = legacy_report["segments"][-1]["end_monotonic_s"] - legacy_report["segments"][0]["start_monotonic_s"]
        self.assertAlmostEqual(fb_summary["total_task_s"], expected_span, places=2)
        self.assertEqual([iv["kind"] for iv in fb_summary["intervals"]], ["motion", "thinking", "motion"])
        self.assertEqual([iv.get("segment_index") for iv in fb_summary["intervals"] if iv["kind"] == "motion"], [0, 1])

        # 2. Multi-turn WAL-backed events.sqlite: scopes strictly to Turn 2 (containing a4..a8),
        #    ignoring Turn 1, Turn 3, malformed JSON rows, and late observer export events.
        db_path = run_root / "events.sqlite"
        conn = sqlite3.connect(db_path)
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE events (seq INTEGER PRIMARY KEY AUTOINCREMENT, body TEXT NOT NULL)")
        base_utc = 1794935000.0
        seg0_start = legacy_report["segments"][0]["start_monotonic_s"]
        seg0_end = legacy_report["segments"][0]["end_monotonic_s"]
        seg1_start = legacy_report["segments"][1]["start_monotonic_s"]
        seg1_end = legacy_report["segments"][1]["end_monotonic_s"]
        offset = base_utc - seg0_start

        def iso(mono_s):
            from datetime import datetime, timezone
            return datetime.fromtimestamp(mono_s + offset, tz=timezone.utc).isoformat().replace("+00:00", "Z")

        events = [
            # Turn 1 (earlier turn — must NOT pollute Turn 2's recording)
            {"type": "run.started", "occurred_at": iso(seg0_start - 500.0), "payload": {}},
            {"type": "agent.turn_started", "occurred_at": iso(seg0_start - 490.0), "payload": {}},
            {"type": "action.started", "occurred_at": iso(seg0_start - 450.0), "correlation": {"action_id": "a1"}, "payload": {}},
            {"type": "agent.turn_completed", "occurred_at": iso(seg0_start - 400.0), "payload": {}},
            # Turn 2 (the turn recorded in legacy_report: a4 .. a8)
            {"type": "agent.turn_started", "occurred_at": iso(seg0_start - 45.0), "payload": {}},
            {"type": "action.started", "occurred_at": iso(seg0_start), "correlation": {"action_id": "a4"}, "payload": {}},
            {"type": "action.completed", "occurred_at": iso(seg0_end), "correlation": {"action_id": "a4"}, "payload": {}},
            {"type": "action.started", "occurred_at": iso(seg1_start), "correlation": {"action_id": "a8"}, "payload": {}},
            {"type": "action.completed", "occurred_at": iso(seg1_end), "correlation": {"action_id": "a8"}, "payload": {}},
            {"type": "agent.turn_completed", "occurred_at": iso(seg1_end + 30.0), "payload": {}},
            # Turn 3 + observer export (later — must NOT pollute Turn 2's recording)
            {"type": "agent.turn_started", "occurred_at": iso(seg1_end + 200.0), "payload": {}},
            {"type": "agent.turn_completed", "occurred_at": iso(seg1_end + 320.0), "payload": {}},
            {"type": "observer.video_exported", "occurred_at": iso(seg1_end + 900.0), "payload": {}},
        ]
        for ev in events:
            conn.execute("INSERT INTO events(body) VALUES (?)", (json.dumps(ev),))
        conn.execute("INSERT INTO events(body) VALUES (?)", ("{malformed-json",))
        conn.commit()
        # Keep WAL present while reader snapshots it (open connection does not mutate during read).
        self.addCleanup(conn.close)
        self.assertTrue((run_root / "events.sqlite-wal").exists())

        recovered = motion_video_composite.compose(legacy_report, overwrite=True)
        self.assertEqual(recovered["status"], "complete")
        rec_summary = recovered["timeline_summary"]
        self.assertEqual(rec_summary["bounds_source"], "events_sqlite")
        self.assertAlmostEqual(rec_summary["total_task_s"], expected_span + 45.0 + 30.0, places=1)
        self.assertAlmostEqual(rec_summary["intervals"][0]["duration_s"], 45.0, places=1)
        self.assertAlmostEqual(rec_summary["intervals"][-1]["duration_s"], 30.0, places=1)

        # Unmatched action_ids fall back to segments_fallback rather than aligning unrelated Turn 1 actions.
        unmatched = dict(legacy_report, segments=[dict(s, action_id=f"other-{i}") for i, s in enumerate(legacy_report["segments"])])
        unmatched_res = motion_video_composite.compose(unmatched, overwrite=True)
        self.assertEqual(unmatched_res["timeline_summary"]["bounds_source"], "segments_fallback")

    def test_clock_jitter_playhead_is_monotonic_and_hud_memory_is_constant(self):
        report = sealed_recording(
            self.root,
            [
                ("a1", [("fixed", 0.0, RED), ("fixed", 0.1, RED)]),
                ("a2", [("fixed", 0.0, BLUE), ("fixed", 0.1, BLUE)]),
            ],
        )
        # Inject cross-segment timestamp regression into fixed.frames.jsonl (seg 1 frame earlier than seg 0 end).
        mapping_file = Path(next(c["frame_mapping_path"] for c in report["cameras"] if c["camera_id"] == "fixed"))
        rows = [json.loads(line) for line in mapping_file.read_text().splitlines()]
        rows[0]["capture_monotonic_s"] = 100.0
        rows[1]["capture_monotonic_s"] = 110.0
        rows[2]["capture_monotonic_s"] = 109.0  # jittered 1s earlier than seg 0 end
        rows[3]["capture_monotonic_s"] = 111.0
        mapping_file.write_text("\n".join(json.dumps(r) for r in rows) + "\n")
        jitter_report = dict(
            report,
            recording_started_monotonic_s=100.0,
            recording_ended_monotonic_s=112.0,
            segments=[
                {"segment_index": 0, "action_id": "a1", "start_monotonic_s": 100.0, "end_monotonic_s": 110.0},
                {"segment_index": 1, "action_id": "a2", "start_monotonic_s": 109.5, "end_monotonic_s": 111.0},
            ],
        )
        res = motion_video_composite.compose(jitter_report, overwrite=True)
        self.assertEqual(res["status"], "complete")
        composed_rows = [json.loads(line) for line in Path(res["frame_mapping_path"]).read_text().splitlines()]
        # Raw capture_monotonic_s is preserved honestly, while displayed task_elapsed_s never regresses.
        self.assertEqual([r["capture_monotonic_s"] for r in composed_rows], [100.0, 110.0, 109.0, 111.0])
        elapsed = [r["task_elapsed_s"] for r in composed_rows]
        self.assertEqual(elapsed, sorted(elapsed))

        # O(1) HUD memory check across many segments (only the active segment image is cached).
        many_report = {
            "recording_started_monotonic_s": 0.0,
            "recording_ended_monotonic_s": 200.0,
            "segments": [{"segment_index": i, "action_id": f"a{i}", "start_monotonic_s": 10.0 * i + 1.0, "end_monotonic_s": 10.0 * i + 2.0}
                         for i in range(15)],
        }
        tl = motion_video_composite._build_timeline(many_report, [], None)
        hud = motion_video_composite.TimelineHud(1280, tl, 30, 30.0)
        for idx in range(15):
            hud._segment_layer(idx)
        self.assertEqual(hud._cached_seg_idx, 14)
        self.assertIsNotNone(hud._cached_base_img)
        self.assertEqual(hud._cached_base_img.size, (1280, motion_video_composite.FOOTER_HEIGHT))

    def test_default_header_renders_model_agent_tokens_and_cost_across_backends(self):
        import sqlite3
        from ni_rua.robot.backends.so101 import motion_video_metadata

        # 1. OpenCode run: effective_config.json + agent-raw.jsonl (multi-session usage & cost).
        oc_run = Path(self.root) / "opencode_run"
        oc_run.mkdir(parents=True)
        (oc_run / "effective_config.json").write_text(json.dumps({
            "agent": {"backend": "opencode"},
            "model": {"name": "openrouter/deepseek/deepseek-v4.1-flash", "effort": "high"},
        }))
        (oc_run / "agent-raw.jsonl").write_text("\n".join([
            json.dumps({"direction": "received", "channel": "native", "message": {
                "type": "session.usage.updated",
                "data": {"sessionID": "s1", "cost": 0.10, "tokens": {"input": 1000, "output": 200, "reasoning": 50, "cache": {"read": 500, "write": 0}}},
            }}),
            json.dumps({"direction": "received", "channel": "native", "message": {
                "type": "session.usage.updated",
                "data": {"sessionID": "s1", "cost": 0.25, "tokens": {"input": 4_000_000, "output": 500_000, "reasoning": 100_000, "cache": {"read": 3_000_000, "write": 350_000}}},
            }}),
            json.dumps({"direction": "received", "channel": "native", "message": {
                "type": "session.usage.updated",
                "data": {"sessionID": "s2", "cost": 0.03, "tokens": {"input": 100, "output": 50, "reasoning": 0, "cache": {"read": 0, "write": 0}}},
            }}),
        ]) + "\n")
        report = sealed_recording(
            oc_run / "private" / "robot",
            [("act-1", [("fixed", 0.0, RED), ("wrist", 0.0, BLUE), ("fixed", 1 / 30, RED)])],
        )
        res = motion_video_composite.compose(report)
        self.assertEqual(res["status"], "complete", res)
        self.assertEqual(res["header_hud_height"], motion_video_composite.HEADER_HEIGHT)
        self.assertEqual(res["timeline_hud_height"], motion_video_composite.FOOTER_HEIGHT)
        expected_h = motion_video_composite.HEADER_HEIGHT + 120 + motion_video_composite.FOOTER_HEIGHT
        self.assertEqual(res["height"], expected_h)
        hs = res["header_summary"]
        self.assertEqual(hs["label"], "OpenCode-DeepSeek 4.1 Flash-high")
        self.assertEqual(hs["total_tokens"], 7_950_150)
        self.assertAlmostEqual(hs["cost_usd"], 0.28, places=6)
        self.assertEqual(hs["currency"], "USD")
        self.assertEqual(hs["tokens_source"], "agent_raw.opencode_session_usage")
        self.assertEqual(hs["cost_source"], "agent_raw.opencode_session_usage")
        frames = decode(res["video_path"], res["width"], expected_h)
        # Header band renders bright glyphs on black background, and camera pixels start cleanly below header_h.
        self.assertGreater(frames[0, :motion_video_composite.HEADER_HEIGHT, :].max(), 180)
        self.assertLess(frames[0, :motion_video_composite.HEADER_HEIGHT, :].mean(), 60)
        fixed_patch = frames[0, motion_video_composite.HEADER_HEIGHT + 50:motion_video_composite.HEADER_HEIGHT + 90, 60:140]
        self.assertTrue(np.allclose(fixed_patch.mean(axis=(0, 1)), RED, atol=40))

        # Cache invalidates automatically when usage/cost updates (e.g. after more turns).
        (oc_run / "agent-raw.jsonl").write_text(json.dumps({
            "direction": "received", "channel": "native", "message": {
                "type": "session.usage.updated",
                "data": {"sessionID": "s1", "cost": 0.42, "tokens": {"input": 9_000_000, "output": 100_000, "reasoning": 0, "cache": {"read": 0, "write": 0}}},
            },
        }) + "\n")
        refreshed = motion_video_composite.compose(report)
        self.assertEqual(refreshed["status"], "complete")
        self.assertAlmostEqual(refreshed["header_summary"]["cost_usd"], 0.42, places=6)
        self.assertEqual(refreshed["header_summary"]["total_tokens"], 9_100_000)

        # 2. Codex run with omitted default ``agent`` section + live WAL events.sqlite runtime/token_accounting.
        codex_run = Path(self.root) / "codex_run"
        codex_run.mkdir(parents=True)
        (codex_run / "effective_config.json").write_text(json.dumps({
            "model": {"name": "gpt-6.1-sol", "effort": "high"},
        }))
        conn = sqlite3.connect(codex_run / "events.sqlite")
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("CREATE TABLE state (kind TEXT NOT NULL, id TEXT NOT NULL, body TEXT NOT NULL, PRIMARY KEY (kind, id))")
        conn.execute(
            "INSERT INTO state(kind, id, body) VALUES (?, ?, ?)",
            ("runtime", "token_accounting", json.dumps({
                "tokens": {"total_tokens": 1_250_000},
                "rate_card_estimate": {"status": "complete", "currency": "USD", "total": "0.75"},
            })),
        )
        conn.commit()
        self.addCleanup(conn.close)
        codex_meta = motion_video_metadata.resolve_header_metadata(None, run_dir=codex_run)
        self.assertEqual(codex_meta["label"], "Codex-GPT 6.1 Sol-high")
        self.assertEqual(codex_meta["total_tokens"], 1_250_000)
        self.assertAlmostEqual(codex_meta["cost_usd"], 0.75, places=6)
        self.assertEqual(codex_meta["cost_source"], "events_sqlite.rate_card_estimate")

        # 3. Claude Code run: cumulative ResultMessage per session_id in agent-raw.jsonl.
        cc_run = Path(self.root) / "claude_run"
        cc_run.mkdir(parents=True)
        (cc_run / "effective_config.json").write_text(json.dumps({
            "agent": {"backend": "claude_code"},
            "model": {"name": "claude-opus-5-5", "effort": "high"},
        }))
        (cc_run / "agent-raw.jsonl").write_text("\n".join([
            json.dumps({"direction": "received", "channel": "native", "message": {
                "_type": "ResultMessage", "session_id": "cc-1", "total_cost_usd": 0.12,
                "model_usage": {"claude-opus-5-5": {"inputTokens": 1000, "outputTokens": 500, "cacheReadInputTokens": 0, "cacheCreationInputTokens": 0}},
            }}),
            json.dumps({"direction": "received", "channel": "native", "message": {
                "_type": "ResultMessage", "session_id": "cc-1", "total_cost_usd": 0.31,
                "model_usage": {"claude-opus-5-5": {"inputTokens": 2500, "outputTokens": 1200, "cacheReadInputTokens": 300, "cacheCreationInputTokens": 100}},
            }}),
        ]) + "\n")
        cc_meta = motion_video_metadata.resolve_header_metadata(None, run_dir=cc_run)
        self.assertEqual(cc_meta["label"], "CC-Claude Opus 5 5-high")
        self.assertEqual(cc_meta["total_tokens"], 4100)
        self.assertAlmostEqual(cc_meta["cost_usd"], 0.31, places=6)

        # 4. Offline CLI accepts run_dir directly, supports custom --output and --preview-frame without mutating archive.
        custom_mp4 = Path(self.root) / "exports" / "custom.mp4"
        preview_png = Path(self.root) / "exports" / "preview.png"
        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(
                motion_video_composite.main([str(oc_run), "--output", str(custom_mp4), "--model-label", "DS 4.1"]),
                0,
            )
        exported = json.loads(out.getvalue())
        self.assertEqual(exported["status"], "complete")
        self.assertEqual(Path(exported["video_path"]), custom_mp4)
        self.assertEqual(exported["header_summary"]["label"], "OpenCode-DS 4.1-high")
        self.assertTrue(custom_mp4.is_file())

        with contextlib.redirect_stdout(io.StringIO()) as out:
            self.assertEqual(
                motion_video_composite.main([str(oc_run), "--preview-frame", "0", "--preview-out", str(preview_png)]),
                0,
            )
        previewed = json.loads(out.getvalue())
        self.assertEqual(previewed["status"], "complete")
        self.assertTrue(preview_png.is_file())
        self.assertGreater(preview_png.stat().st_size, 100)

        # 5. Narrow widths, non-USD currency, and missing values render cleanly without crashing.
        from ni_rua.robot.backends.so101 import motion_video_hud
        fonts = motion_video_hud.HudFonts()
        for w in (1280, 640, 288, 160):
            band = motion_video_hud.render_header(
                w,
                {"label": "OpenCode-DeepSeek 4.1 Flash-high", "total_tokens": 7_950_000, "cost_usd": 0.28, "currency": "USD"},
                fonts,
            )
            self.assertEqual(band.shape, (motion_video_hud.HEADER_HEIGHT, w, 3))
            self.assertGreater(band.max(), 150)
        self.assertEqual(motion_video_hud.fmt_cost(None), "n/a")
        self.assertEqual(motion_video_hud.fmt_cost(1.25, "EUR"), "EUR 1.25")
        self.assertEqual(motion_video_hud.fmt_tokens(None), "n/a")
        self.assertEqual(motion_video_hud.fmt_tokens(1500), "1.5K")

        # 6. Custom --output / preview on a recording leaves source recording directory untouched.
        rec_dir = Path(report["cameras"][0]["frame_mapping_path"]).parent
        before_files = sorted(p.name for p in rec_dir.iterdir())
        extra_mp4 = Path(self.root) / "exports" / "readonly_check.mp4"
        res_ro = motion_video_composite.compose(report, directory=rec_dir, run_dir=oc_run, output_path=extra_mp4)
        self.assertEqual(res_ro["status"], "complete")
        self.assertEqual(sorted(p.name for p in rec_dir.iterdir()), before_files)


if __name__ == "__main__":
    unittest.main()
