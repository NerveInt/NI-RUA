"""Bounded operator-only calibration frame archive from the worker's existing cameras.

This recorder neither opens a camera nor reads the motor bus. PNG frames and
timing are evidence for later analysis, not a physical contact verdict. Sampling
``latest`` can miss camera frames: counters describe sampler attempts, not native
camera frame counts. Exposure timing and cross-camera synchronization are unknown.
"""
from __future__ import annotations

from copy import deepcopy
from datetime import datetime, timezone
import hashlib
import io
import json
import math
from pathlib import Path
import re
import threading
import time

import numpy as np
from PIL import Image

from ni_rua.robot.backends.so101.deployment import FIELDS, _check_number


SCHEMA = "ni_rua.so101.calibration_video.v1"


def _utc():
    return datetime.now(timezone.utc).isoformat()


def _plain(value):
    """Copy numeric calibration/snapshot containers into JSON-native values, preserving units."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: _plain(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_plain(item) for item in value]
    return deepcopy(value)


class CalibrationVideoRecorder:
    """Sample existing RGB cameras asynchronously into a fresh frame directory.

    ``adapter`` owns cameras, the shared monotonic clock and latest measured
    ``last_sample``; no adapter read/move operation is invoked. ``directory`` is
    a new operator archive directory, never an existing recording. ``fps`` is
    sampler frequency in Hz (default 10, range (0,60]); it is not native camera
    FPS. ``max_frames`` is the saved-frame cap per camera (default 600).
    ``max_duration_s`` is the wall-time cap in seconds (default 60). Defaults and
    bounds come from deployment.FIELDS calibration.touch_video_*; None selects
    its declared default. The workflow handles fps=0 as recording disabled and
    never constructs a recorder in that case.
    Frames retain full RGB uint8 resolution as PNG. Every effective value is
    recorded. Example::

        recorder = CalibrationVideoRecorder(adapter, run_dir / "S1-video")
        recorder.start()
        try:
            report = touch_points(...)
        finally:
            video = recorder.stop()

    ``start`` returns immediately without waiting for a frame. ``stop`` requests
    shutdown and joins at most 2 seconds; a blocked writer returns ``unknown``
    rather than claiming a sealed archive. Neither is a robot stop operation.
    ``capture_once`` is a synchronous test/sampler entry; never call it from the
    physical command loop. Frames are keyed by monotonically increasing camera
    stamps. Joint snapshots retain their own timestamp and are not exposure poses.
    """

    def __init__(self, adapter, directory, *, fps=None, max_frames=None, max_duration_s=None):
        values = dict(fps=fps, max_frames=max_frames, max_duration_s=max_duration_s)
        for name, value in values.items():
            spec = FIELDS["calibration"][f"touch_video_{name}"]
            values[name] = _check_number(spec["default"] if value is None else value, spec,
                                         f"calibration_video.{name}")
        fps, max_frames, max_duration_s = (values[key] for key in ("fps", "max_frames", "max_duration_s"))
        if fps == 0:
            raise ValueError("calibration_video_disabled:fps")
        names = sorted(set(adapter.cameras) | set(adapter.deployment.get("cameras", {})))
        if not names or any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in names):
            raise ValueError("calibration_video_invalid:camera_ids")
        self.adapter, self.directory = adapter, Path(directory)
        self.settings = dict(fps=float(fps), max_frames=max_frames,
                             max_duration_s=float(max_duration_s), image_format="png")
        self.event, self.thread = threading.Event(), None
        self._prepared, self._finished = False, False
        self._sealed = False
        self._wall_start = None
        self._last_stamp = {}
        self.manifest = dict(schema=SCHEMA, status="not_started", termination_reason=None,
            visual_evidence_status="pending", coverage=None,
            settings=dict(self.settings), clock_domain="adapter_monotonic",
            started_at=None, finished_at=None, started_monotonic_s=None, finished_monotonic_s=None,
            exposure_timing="unknown", joint_alignment="latest_snapshot_not_exposure_synchronized",
            camera_calibration=_plain(getattr(adapter, "camera_calibration", None)),
            joint_mapping=_plain(adapter.deployment["joint_mapping"]),
            camera_configuration=_plain(adapter.deployment.get("cameras", {})),
            sampling="latest-frame sampling; dropped native frames cannot be counted",
            cameras={name: dict(frames=[], sample_attempts=0, duplicates=0, missing=0,
                                invalid=0, errors=0, nonmonotonic=0, error_details=[], truncated=False)
                     for name in names})

    def _prepare(self):
        if self._prepared:
            return
        self.directory.mkdir(parents=True, exist_ok=False)
        for name in self.manifest["cameras"]:
            (self.directory / name).mkdir()
        self._wall_start = time.monotonic()
        self.manifest.update(status="recording", started_at=_utc(),
                             started_monotonic_s=float(self.adapter.clock.now()))
        self._prepared = True

    def summary(self):
        """Small operator summary; complete per-frame facts live in manifest.json."""
        return {key: self.manifest[key] for key in
                ("schema", "status", "termination_reason", "started_at", "finished_at", "visual_evidence_status")} | dict(
            coverage=deepcopy(self.manifest["coverage"]),
            settings=dict(self.settings), manifest_path=str(self.directory / "manifest.json"),
            sealed=self._sealed,
            cameras={name: {key: value for key, value in camera.items() if key not in ("frames", "error_details")}
                     | {"frames": len(camera["frames"])} for name, camera in self.manifest["cameras"].items()})

    def start(self):
        """Start one background sampler, returning operator summary; no frame/network wait."""
        if self.thread is not None or self._finished:
            raise RuntimeError("calibration_video_already_started_or_finished")
        self._prepare()
        self.thread = threading.Thread(target=self._run, name="so101-calibration-video", daemon=True)
        self.thread.start()
        return self.summary()

    def _write_manifest(self):
        self._update_visual_coverage()
        temporary = self.directory / ".manifest.tmp"
        temporary.write_text(json.dumps(self.manifest, indent=2, allow_nan=False) + "\n")
        temporary.replace(self.directory / "manifest.json")

    def _update_visual_coverage(self):
        """Report archive availability separately from lifecycle and physical/contact validity.

        Two distinct timestamps are the minimum temporal pair; available means
        such pairs exist for each configured camera without recorded faults or
        truncation. It never certifies visibility, exposure alignment or contact.
        """
        cameras = self.manifest["cameras"]
        counts = {name: len(camera["frames"]) for name, camera in cameras.items()}
        absent = [name for name, count in counts.items() if count == 0]
        no_pair = [name for name, count in counts.items() if count < 2]
        missing_source = [name for name in cameras if self.adapter.cameras.get(name) is None]
        faults = [name for name, camera in cameras.items()
                  if camera["truncated"] or any(camera[key] for key in ("missing", "invalid", "errors", "nonmonotonic"))]
        shared = None
        if not absent:
            first = max(camera["frames"][0]["capture_monotonic_s"] for camera in cameras.values())
            last = min(camera["frames"][-1]["capture_monotonic_s"] for camera in cameras.values())
            if first <= last:
                shared = [first, last]
        status = ("unavailable" if not sum(counts.values()) else
                  "incomplete" if no_pair or faults or missing_source or self.manifest["status"] in ("failed", "unknown")
                  else "available")
        self.manifest.update(visual_evidence_status=status, coverage=dict(
            expected_cameras=list(cameras), frames_per_camera=counts, cameras_without_frames=absent,
            cameras_without_temporal_pair=no_pair, cameras_missing_source=missing_source,
            cameras_with_gaps_or_faults=faults, shared_host_time_interval_s=shared,
            scope="archived_rgb_availability_not_contact_or_motion_validation"))

    def _finish(self):
        if self._finished:
            return
        self._finished = True
        if self.manifest["status"] == "recording":
            self.manifest.update(status="completed", termination_reason="stop_requested")
        self.manifest.update(finished_at=_utc(), finished_monotonic_s=float(self.adapter.clock.now()))
        try:
            self._write_manifest()
            self._sealed = True
        except Exception as error:
            self.manifest.update(status="failed", termination_reason=f"manifest_write_failed:{type(error).__name__}")
            self._update_visual_coverage()

    def _run(self):
        try:
            self._write_manifest()
            while not self.event.is_set():
                self.capture_once()
                if self.manifest["status"] != "recording":
                    break
                self.event.wait(1 / self.settings["fps"])
        except Exception as error:
            self.manifest.update(status="failed", termination_reason=f"recorder_failed:{type(error).__name__}")
        finally:
            self._finish()

    def stop(self):
        """Request sampler shutdown; return unknown if disk/camera snapshot I/O has not returned in 2 s."""
        self.event.set()
        if self.thread is not None and self.thread is not threading.current_thread():
            self.thread.join(timeout=2)
            if self.thread.is_alive():
                return self.summary() | dict(status="unknown", termination_reason="writer_shutdown_pending",
                                              visual_evidence_status="incomplete")
        elif self._prepared:
            self._finish()
        return self.summary()

    def capture_once(self):
        """Sample each latest camera frame once; synchronous only in this recorder/test thread."""
        self._prepare()
        if self._finished or self.event.is_set() or self.manifest["status"] != "recording":
            return self.summary()
        if time.monotonic() - self._wall_start >= self.settings["max_duration_s"]:
            self.manifest.update(status="truncated", termination_reason="max_duration_s")
            for camera in self.manifest["cameras"].values():
                camera["truncated"] = True
            return self.summary()
        for name in self.manifest["cameras"]:
            if self.event.is_set():
                break
            camera = self.manifest["cameras"][name]
            if len(camera["frames"]) >= self.settings["max_frames"]:
                camera["truncated"] = True
                continue
            camera["sample_attempts"] += 1
            try:
                source = self.adapter.cameras.get(name)
                if source is None:
                    camera["missing"] += 1
                    continue
                reader = getattr(source, "latest_timed", None)
                if callable(reader):
                    latest = reader()
                else:
                    old = source.latest()
                    latest = None if old is None else dict(image_rgb=old[0], capture_monotonic_s=old[1],
                                                           host_read_interval_s=[old[1], None])
                if latest is None:
                    camera["missing"] += 1
                    continue
                stamp = latest["capture_monotonic_s"]
                interval = latest.get("host_read_interval_s", [stamp, None])
                if (type(stamp) not in (int, float) or not math.isfinite(stamp)
                        or not isinstance(interval, (list, tuple)) or len(interval) != 2
                        or interval[0] != stamp or (interval[1] is not None and
                            (type(interval[1]) not in (int, float) or not math.isfinite(interval[1]) or interval[1] < stamp))):
                    camera["invalid"] += 1
                    continue
                previous = self._last_stamp.get(name)
                if previous is not None and stamp <= previous:
                    camera["duplicates" if stamp == previous else "nonmonotonic"] += 1
                    continue
                self._last_stamp[name] = stamp
                pixels = np.asarray(latest["image_rgb"])
                if pixels.dtype != np.uint8 or pixels.ndim != 3 or pixels.shape[2] != 3 or min(pixels.shape[:2]) < 1:
                    camera["invalid"] += 1
                    continue
                pixels = pixels.copy()
                joint = _plain(getattr(self.adapter, "last_sample", None))
                if joint is not None:
                    joint = {key: joint.get(key) for key in ("t", "q_rad", "gripper_rad")}
                sampled_at = float(self.adapter.clock.now())
                index = len(camera["frames"])
                relative = f"{name}/{index:06d}.png"
                encoded = io.BytesIO()
                Image.fromarray(pixels).save(encoded, format="PNG")
                data = encoded.getvalue()
                if (self.directory / relative).write_bytes(data) != len(data):
                    raise OSError("calibration_video_short_png_write")
                camera["frames"].append(dict(index=index, path=relative, width=pixels.shape[1], height=pixels.shape[0],
                    source_sha256=hashlib.sha256(data).hexdigest(),
                    capture_monotonic_s=float(stamp), host_read_interval_s=list(interval), exposure_monotonic_s=None,
                    sampler_monotonic_s=sampled_at, joint_sample=joint))
            except Exception as error:
                camera["errors"] += 1
                camera["error_details"] = (camera["error_details"] + [type(error).__name__])[-8:]
        if all(len(camera["frames"]) >= self.settings["max_frames"] for camera in self.manifest["cameras"].values()):
            self.manifest.update(status="truncated", termination_reason="max_frames")
            for camera in self.manifest["cameras"].values():
                camera["truncated"] = True
        self._write_manifest()
        return self.summary()
