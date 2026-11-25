"""Operator-only, motion-gated RGB recording; never reads the motor bus.

Camera callbacks enqueue immutable native RGB frames without waiting. Each camera
has a bounded queue and an encoder thread. Omitting idle frames before encoding
concatenates execution intervals without a second encode or temporary PNG archive.
"""
from __future__ import annotations

from collections import deque
import json
from pathlib import Path
import subprocess
import threading
import uuid


class _CameraVideo:
    def __init__(self, directory, name, config, source, settings):
        self.name = name
        self.width, self.height = config["width"], config["height"]
        self.fps = float(getattr(source, "recording_fps", config["fps"]))
        self.fps_source = getattr(source, "recording_fps_source", "configured_fallback")
        self.crf = settings["video_crf"]
        self.limit = settings["video_queue_mb"] * 1024 * 1024
        self.path = directory / f"{name}.mp4"
        self.partial = directory / f"{name}.partial"
        self.mapping = directory / f"{name}.frames.jsonl"
        self.log = directory / f"{name}.ffmpeg.log"
        self.condition = threading.Condition()
        self.pending, self.pending_bytes = deque(), 0
        self.closed, self.error = False, None
        self.frames, self.dropped, self.invalid = 0, 0, 0
        self.process = None
        self.thread = threading.Thread(target=self._encode, name=f"motion-video-{name}", daemon=True)
        self.thread.start()

    def submit(self, image, metadata):
        with self.condition:
            if self.closed or self.error:
                return
            if image.shape != (self.height, self.width, 3) or str(image.dtype) != "uint8":
                self.invalid += 1
                return
            if self.pending_bytes + image.nbytes > self.limit:
                self.dropped += 1
                return
            self.pending.append((image, metadata))
            self.pending_bytes += image.nbytes
            self.condition.notify()

    def _encode(self):
        encoder = None
        try:
            with self.mapping.open("x") as mapping, self.log.open("xb") as log:
                while True:
                    with self.condition:
                        self.condition.wait_for(lambda: self.pending or self.closed)
                        if not self.pending:
                            break
                        image, metadata = self.pending.popleft()
                        self.pending_bytes -= image.nbytes
                    if encoder is None:
                        # yuv444p handles odd dimensions without resizing/padding the source.
                        pixel_format = "yuv420p" if self.width % 2 == self.height % 2 == 0 else "yuv444p"
                        command = ["ffmpeg", "-nostdin", "-hide_banner", "-loglevel", "error", "-n",
                                   "-f", "rawvideo", "-pixel_format", "rgb24", "-video_size", f"{self.width}x{self.height}",
                                   "-framerate", str(self.fps), "-i", "pipe:0", "-an", "-c:v", "libx264",
                                   "-preset", "medium", "-crf", str(self.crf), "-threads", "2", "-pix_fmt", pixel_format,
                                   "-movflags", "+faststart", "-f", "mp4", str(self.partial)]
                        encoder = self.process = subprocess.Popen(command, stdin=subprocess.PIPE,
                                                                  stdout=subprocess.DEVNULL, stderr=log)
                    encoder.stdin.write(image.tobytes())
                    mapping.write(json.dumps(dict(metadata, video_frame=self.frames, pts_s=self.frames / self.fps)) + "\n")
                    mapping.flush()
                    self.frames += 1
                if encoder is not None:
                    encoder.stdin.close()
                    if encoder.wait(timeout=15):
                        raise RuntimeError("ffmpeg_failed")
                    probe = json.loads(subprocess.check_output(
                        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
                         "stream=codec_name,width,height,nb_frames,avg_frame_rate", "-of", "json", str(self.partial)],
                        stderr=log, timeout=15))["streams"][0]
                    from fractions import Fraction
                    # Read the finalized MP4 sample table, not an O(run duration)
                    # full decode on the worker's shutdown path.
                    if (probe["codec_name"] != "h264" or int(probe["nb_frames"]) != self.frames
                            or (probe["width"], probe["height"]) != (self.width, self.height)
                            or abs(float(Fraction(probe["avg_frame_rate"])) - self.fps) > 0.001):
                        raise RuntimeError("encoded_video_verification_failed")
                    self.partial.replace(self.path)
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
        finally:
            if encoder is not None:
                if encoder.poll() is None:
                    encoder.kill()
                encoder.wait()
                if encoder.stdin and not encoder.stdin.closed:
                    try:
                        encoder.stdin.close()
                    except OSError:
                        pass
            with self.condition:
                self.pending.clear()
                self.pending_bytes = 0

    def request_close(self):
        with self.condition:
            self.closed = True
            self.condition.notify()

    def result(self):
        self.thread.join(timeout=35)
        if self.thread.is_alive():
            self.error = "encoder_shutdown_timeout"
            if self.process is not None and self.process.poll() is None:
                self.process.kill()
            self.thread.join(timeout=2)
        status = "failed" if self.error else "empty" if not self.frames else "incomplete" if self.dropped or self.invalid else "complete"
        result = dict(camera_id=self.name, status=status, frames=self.frames, fps=self.fps, fps_source=self.fps_source,
                      width=self.width, height=self.height, codec="h264", crf=self.crf, preset="medium", faststart=True,
                      dropped_frames=self.dropped, invalid_frames=self.invalid, error=self.error,
                      frame_mapping_path=str(self.mapping), encoder_log_path=str(self.log))
        if self.path.exists() and not self.error:
            result["video_path"] = str(self.path)
        return result


class MotionVideoRecorder:
    """Concatenate action execution intervals from every configured camera.

    ``adapter`` supplies existing cameras, deployment and monotonic clock; it is
    never asked to move/read the arm. Default recording is off until ``record``.
    Example: record(True); begin('action-1'); end(); save(). begin/end take no
    physical thresholds: recording covers execution, including settle/stall/stop,
    and does not claim measured nonzero velocity. Files are observer-only.
    """

    def __init__(self, adapter):
        self.adapter = adapter
        self.lock = threading.Lock()
        self.enabled = False
        self.active = None
        self.streams = {}
        self.segments = []
        self.directory = None
        self.started_monotonic_s = None
        self.ended_monotonic_s = None
        self.export = None
        self.error = None

    def record(self, enabled=True):
        """Enable/pause capture (bool, default True); pause keeps previous frames.

        Re-enabling after save starts a fresh, uniquely named archive. No frames
        are captured until begin(), regardless of how long the model thinks.
        """
        if enabled and (self.directory is None or self.export is not None):
            for old_stream in list(self.streams.values()):
                try:
                    old_stream.request_close()
                    old_stream.result()
                except Exception:
                    pass
            self.directory = self.adapter.run_dir / "videos" / str(uuid.uuid4())
            self.streams, self.segments, self.export, self.error = {}, [], None, None
            self.started_monotonic_s = float(self.adapter.clock.now())
            self.ended_monotonic_s = None
            try:
                self.directory.mkdir(parents=True)
            except OSError as error:
                self.error = f"{type(error).__name__}: {error}"
                with self.lock:
                    self.enabled = False
                return dict(enabled=False, sampling="all native RGB frames during execution; model waits omitted", error=self.error)
            for name, config in self.adapter.deployment["cameras"].items():
                try:
                    source = self.adapter.cameras[name]
                    stream = _CameraVideo(self.directory, name, config, source, self.adapter.deployment["camera"])
                    self.streams[name] = stream
                    source.set_frame_sink(lambda image, start, end, name=name: self._frame(name, image, start, end))
                except Exception as error:
                    self.error = "; ".join(filter(None, [self.error, f"{name}: {type(error).__name__}: {error}"]))
        with self.lock:
            self.enabled = bool(enabled)
        return dict(enabled=self.enabled, sampling="all native RGB frames during execution; model waits omitted", error=self.error)

    def begin(self, action_id):
        """Open interval at first command. action_id is its archived identity (no default)."""
        with self.lock:
            if self.enabled and self.active is None:
                self.active = dict(segment_index=len(self.segments), action_id=action_id,
                                   start_monotonic_s=self.adapter.clock.now(), end_monotonic_s=None,
                                   start_control_step=self.adapter.control_step, end_control_step=None)
                self.segments.append(self.active)

    def end(self):
        """Close interval before terminal observation; idempotent, no I/O or waits."""
        with self.lock:
            if self.active is not None:
                self.active.update(end_monotonic_s=self.adapter.clock.now(), end_control_step=self.adapter.control_step)
                self.active = None

    def _frame(self, name, image, start, end):
        with self.lock:
            if not self.enabled or self.active is None or start < self.active["start_monotonic_s"]:
                return
            metadata = dict(segment_index=self.active["segment_index"], action_id=self.active["action_id"],
                            capture_monotonic_s=start, host_read_interval_s=[start, end], exposure_monotonic_s=None,
                            delivered_to_actor=False)
            self.streams[name].submit(image, metadata)

    def save(self):
        """Seal and verify all streams; repeated calls return the same report.

        Report status is recording health only, never a physical action receipt.
        Missing frames/cameras remain empty or failed; no fabricated filler frames.
        """
        if self.export is not None:
            return self.export
        self.end()
        ended_now = float(self.adapter.clock.now())
        with self.lock:
            self.enabled = False
        for source in self.adapter.cameras.values():
            source.set_frame_sink(None)
        for stream in self.streams.values():
            stream.request_close()
        cameras = [stream.result() for stream in self.streams.values()]
        for name in set(self.adapter.deployment["cameras"]) - set(self.streams):
            cameras.append(dict(camera_id=name, status="unavailable", frames=0))
        # Count per-segment gaps explicitly, including very short actions with zero frames.
        for camera in cameras:
            counts = [0] * len(self.segments)
            last_stamps = {}
            max_gap = None
            mapping = camera.get("frame_mapping_path")
            if mapping and Path(mapping).is_file():
                with Path(mapping).open() as source:
                    for line in source:
                        try:
                            row = json.loads(line)
                            index, stamp = row["segment_index"], row["capture_monotonic_s"]
                            counts[index] += 1
                            if index in last_stamps:
                                gap = stamp - last_stamps[index]
                                max_gap = gap if max_gap is None else max(max_gap, gap)
                            last_stamps[index] = stamp
                        except (ValueError, KeyError, IndexError):
                            camera["status"] = "incomplete"
            camera["segment_frame_counts"] = counts
            camera["max_capture_gap_s"] = max_gap
            camera["source_frame_loss"] = None  # UVC provides no hardware frame counter.
            if camera["status"] == "complete" and any(count == 0 for count in counts):
                camera["status"] = "incomplete"
        if self.segments:
            first_start = min(float(seg["start_monotonic_s"]) for seg in self.segments)
            last_end = max(
                float(seg["end_monotonic_s"] if seg.get("end_monotonic_s") is not None else seg["start_monotonic_s"])
                for seg in self.segments
            )
            started_s = first_start if self.started_monotonic_s is None else min(float(self.started_monotonic_s), first_start)
            ended_s = max(ended_now, last_end, started_s)
        elif self.started_monotonic_s is not None:
            started_s = float(self.started_monotonic_s)
            ended_s = max(ended_now, started_s)
        else:
            started_s, ended_s = None, None
        self.started_monotonic_s, self.ended_monotonic_s = started_s, ended_s
        report = dict(audience="observer_only", data_origin="so101_runtime", mode="action_execution_only",
                      status_semantics="complete means all queued frames encoded; source frame loss is unknown",
                      timing="native frames concatenated at source FPS; host read times retained; exposure/synchronization unknown",
                      recording_started_monotonic_s=self.started_monotonic_s,
                      recording_ended_monotonic_s=self.ended_monotonic_s,
                      segments=self.segments, cameras=cameras, videos=[item for item in cameras if "video_path" in item], error=self.error)
        if self.directory is not None and self.directory.is_dir():
            temporary = self.directory / f"manifest.{uuid.uuid4().hex[:12]}.tmp"
            try:
                temporary.write_text(json.dumps(report, indent=2))
                temporary.replace(self.directory / "manifest.json")
            except BaseException:
                temporary.unlink(missing_ok=True)
                raise
        self.export = report
        return report
