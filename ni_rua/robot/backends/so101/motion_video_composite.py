"""Operator-only side-by-side composite of one sealed SO-101 motion recording.

Input is the report that ``MotionVideoRecorder.save()`` returns (also written as
``<recording>/manifest.json``): one sealed ``<camera>.mp4`` and
``<camera>.frames.jsonl`` per camera. Output, in the same directory:
``combined.mp4`` (optional ``HEADER_HEIGHT`` model/agent/usage banner above all
camera tiles left to right in report order above a dedicated ``FOOTER_HEIGHT``
bottom timeline HUD band), ``combined.frames.jsonl`` (which source frame each
tile shows), ``combined.json`` (this report, including ``header_summary`` and
``timeline_summary``) and ``combined.ffmpeg.log``.

Alignment uses host capture times only, per action segment. The camera with the
most frames in the segment is the reference (ties: report order). Each output
frame is one reference frame. Every other tile shows that camera's latest frame
captured at or before the reference frame's ``capture_monotonic_s``, or its first
frame in the segment until then. A camera without frames in a segment shows a
black ``no frame`` tile. Exposure time and cross-camera synchronisation are
unknown, so tiles agree to about one frame interval, not to a hardware clock.

Camera pixels are never cropped or covered by the header or timeline footer:
when run metadata is present (every NI-RUA run with ``effective_config.json``),
a ``40 px`` solid-black header sits above ``tile_height`` showing
``Backend-Model-Effort`` on the left and ``Tokens`` / ``Cost`` on the right; a
``132 px`` native-resolution FreeType timeline footer sits below ``tile_height``
(e.g. ``1280x480`` becomes ``1280x652`` ``yuv420p``). ``MOVE`` intervals
represent recorded action-execution gates ``[begin..end]`` (which may include
settle/hold/stall), and ``THINK`` intervals represent omitted non-execution
elapsed gaps within the episode bounds.

Playback rate is the highest source FPS, so with mixed camera rates a segment
whose reference is a lower-FPS camera plays faster than real time (each
reference frame lasts one output frame, exactly as in the per-camera MP4s).
Encoding is libx264, preset ``medium``, ``yuv420p``, ``+faststart``, at the
sources' CRF (default 23), in a single pass directly from the first-generation
per-camera MP4s. The result is published as ``combined.mp4`` only after ffprobe
confirms codec, frame count and size. It never reads the motor bus or a camera
device and is never delivered to the actor.

Offline CLI (accepts either a recording directory containing ``manifest.json``
or a NI-RUA run directory):

    mamba run -n ni-rua python -m ni_rua.robot.backends.so101.motion_video_composite <recording-or-run-dir> [--overwrite]
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager, nullcontext
from datetime import datetime
import fcntl
import json
import math
import os
from pathlib import Path
import shutil
from statistics import median
import subprocess
import sys
import tempfile
import uuid

from .motion_video_hud import (
    FOOTER_HEIGHT,
    HEADER_HEIGHT,
    HUD_HEIGHT,
    HUD_RENDER_VERSION,
    HudFonts,
    TimelineHud,
    fmt_clock as _fmt_clock,
    fmt_cost,
    fmt_tokens,
    make_camera_pill,
    make_no_frame_badge,
    render_header,
    short_action_id as _short_action_id,
    stamp_patch,
)
from .motion_video_metadata import (
    find_run_root,
    read_sqlite_snapshot,
    resolve_header_metadata,
)

NAME = "combined"
DEFAULT_CRF = 23
DEFAULT_THREADS = 2
ALIGNMENT = (
    "per action segment: reference = camera with most frames; other tiles show the latest frame captured "
    "at or before the reference frame's host capture time (first segment frame before that); black tile "
    "when a camera has no frame in the segment"
)
TIMING = "host read times; exposure and cross-camera synchronisation unknown"
TIMELINE_SEMANTICS = (
    "motion intervals are recorded action-execution gates [begin..end] (including holds/stalls); "
    "thinking intervals are omitted non-execution elapsed gaps within episode bounds"
)

_EPISODE_START_EVENTS = frozenset({"run.started", "episode.started", "agent.turn_started"})
_EPISODE_END_EVENTS = frozenset({"agent.turn_completed", "episode.completed", "run.completed", "run.stopped"})
_ACTION_START_PRIMARY = frozenset({"action.started"})
_ACTION_START_SECONDARY = frozenset({"action.accepted", "action.requested"})
_ACTION_END_EVENTS = frozenset({"action.executed", "action.completed", "action.failed", "action.stopped"})

# Backward-compatible symbol alias.
_TimelineHud = TimelineHud


class _Source:
    """Sequential decoder for one sealed camera MP4 and its frame mapping."""

    def __init__(self, camera, ffmpeg, log):
        import numpy

        self.numpy = numpy
        self.camera_id = camera["camera_id"]
        self.width, self.height = int(camera["width"]), int(camera["height"])
        self.fps = float(camera["fps"])
        mapping_file = Path(camera["frame_mapping_path"])
        video_file = Path(camera["video_path"])
        if not mapping_file.is_file():
            raise FileNotFoundError(f"{self.camera_id}: missing frame_mapping_path {mapping_file}")
        if not video_file.is_file():
            raise FileNotFoundError(f"{self.camera_id}: missing video_path {video_file}")
        rows = [json.loads(line) for line in mapping_file.read_text().splitlines() if line.strip()]
        self.segments = [int(row["segment_index"]) for row in rows]
        self.times = [float(row["capture_monotonic_s"]) for row in rows]
        self.actions = [row.get("action_id") for row in rows]
        if any(later < earlier for earlier, later in zip(self.segments, self.segments[1:])):
            raise ValueError(f"{self.camera_id}: frame_mapping_segments_out_of_order")
        self.frame_bytes = self.width * self.height * 3
        self.index, self.image = -1, None
        self.process = subprocess.Popen(
            [
                ffmpeg,
                "-nostdin",
                "-hide_banner",
                "-loglevel",
                "error",
                "-threads",
                "1",
                "-i",
                str(video_file),
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "pipe:1",
            ],
            stdout=subprocess.PIPE,
            stderr=log,
        )

    def count(self, segment):
        return self.segments.count(segment)

    def _advance(self):
        data = self.process.stdout.read(self.frame_bytes)
        if len(data) != self.frame_bytes:
            raise RuntimeError(f"{self.camera_id}: decoded_frames_fewer_than_mapping")
        self.index += 1
        self.image = self.numpy.frombuffer(data, self.numpy.uint8).reshape(self.height, self.width, 3)

    def seek(self, index):
        """Decode forward to mapping row ``index`` (reference camera)."""
        while self.index < index:
            self._advance()
        return self.index, self.image

    def at(self, segment, time_s):
        """Latest frame of ``segment`` captured at or before ``time_s``; first segment frame if none yet."""
        last = len(self.times) - 1
        while self.index < last and (
            self.segments[self.index + 1] < segment
            or (self.segments[self.index + 1] == segment and self.times[self.index + 1] <= time_s)
        ):
            self._advance()
        if self.index < 0 or self.segments[self.index] != segment:
            if self.index < last and self.segments[self.index + 1] == segment:
                self._advance()
            else:
                return None, None
        return self.index, self.image

    def close(self):
        if self.process.poll() is None:
            self.process.kill()
        self.process.wait()
        if self.process.stdout:
            self.process.stdout.close()


def _finite_float(value) -> float | None:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    val = float(value)
    return val if math.isfinite(val) else None


def _parse_utc_timestamp(value) -> float | None:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        val = float(value)
        return val if math.isfinite(val) else None
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    if text.endswith("Z"):
        text = text[:-1] + "+00:00"
    try:
        return datetime.fromisoformat(text).timestamp()
    except ValueError:
        return None


def _recover_episode_bounds_from_events(
    directory: Path | None,
    motion_segments: list[dict],
    *,
    events: list[dict] | None = None,
    run_root: Path | None = None,
) -> tuple[float | None, float | None]:
    """Recover monotonic episode ``(start_s, end_s)`` scoped to the turn(s) matching ``motion_segments``."""
    if not motion_segments:
        return None, None
    if events is None:
        resolved_run = run_root if run_root is not None else find_run_root(directory)
        if resolved_run is None:
            return None, None
        events, _ = read_sqlite_snapshot(resolved_run)
    if not events:
        return None, None

    parsed: list[tuple[str, float, str | None, dict]] = []
    by_action: dict[str, list[tuple[str, float, dict]]] = {}
    for event in events:
        if not isinstance(event, dict):
            continue
        etype = event.get("type")
        if not isinstance(etype, str):
            continue
        ts = _parse_utc_timestamp(event.get("occurred_at") or event.get("timestamp") or event.get("created_at"))
        if ts is None:
            continue
        corr = event.get("correlation") if isinstance(event.get("correlation"), dict) else {}
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        req = payload.get("request") if isinstance(payload.get("request"), dict) else {}
        action_id = corr.get("action_id") or payload.get("action_id") or req.get("action_id")
        action_id = str(action_id) if action_id is not None else None
        parsed.append((etype, ts, action_id, payload))
        if action_id:
            by_action.setdefault(action_id, []).append((etype, ts, payload))

    if not parsed:
        return None, None

    # Require explicit action_id match so a re-armed or unrelated archive never aligns against another recording.
    primary_offsets: list[float] = []
    secondary_offsets: list[float] = []
    for seg in motion_segments:
        aid = str(seg["action_id"]) if seg.get("action_id") is not None else None
        if not aid or aid not in by_action:
            continue
        entries = by_action[aid]
        started = [ts for etype, ts, _ in entries if etype in _ACTION_START_PRIMARY]
        if started:
            primary_offsets.append(started[0] - seg["start_monotonic_s"])
            continue
        requested = [ts for etype, ts, _ in entries if etype in _ACTION_START_SECONDARY]
        if requested:
            secondary_offsets.append(requested[0] - seg["start_monotonic_s"])
            continue
        ended = [(ts, payload) for etype, ts, payload in entries if etype in _ACTION_END_EVENTS]
        if ended:
            end_ts, end_payload = ended[-1]
            wall_dur = _finite_float(end_payload.get("wall_clock_duration_s"))
            if wall_dur is not None and wall_dur >= 0.0:
                secondary_offsets.append((end_ts - wall_dur) - seg["start_monotonic_s"])
            else:
                secondary_offsets.append(end_ts - seg["end_monotonic_s"])

    if primary_offsets:
        utc_minus_mono = float(median(primary_offsets))
    elif secondary_offsets:
        utc_minus_mono = float(median(secondary_offsets))
    else:
        return None, None

    first_seg_utc = motion_segments[0]["start_monotonic_s"] + utc_minus_mono
    last_seg_utc = motion_segments[-1]["end_monotonic_s"] + utc_minus_mono

    # Scope start to the active turn enclosing the first recorded segment (falling back to episode/run start).
    prior_turns = [ts for etype, ts, _, _ in parsed if etype == "agent.turn_started" and ts <= first_seg_utc + 1.0]
    if prior_turns:
        start_utc: float | None = max(prior_turns)
    else:
        prior_runs = [ts for etype, ts, _, _ in parsed if etype in _EPISODE_START_EVENTS and ts <= first_seg_utc + 1.0]
        start_utc = max(prior_runs) if prior_runs else None

    # Scope end to the completion of the turn active at last_seg_utc, never crossing into a later turn.
    later_turns = [ts for etype, ts, _, _ in parsed if etype == "agent.turn_started" and ts > last_seg_utc + 1.0]
    next_turn_utc = min(later_turns) if later_turns else math.inf

    end_completions = [
        ts
        for etype, ts, _, _ in parsed
        if etype in _EPISODE_END_EVENTS and (last_seg_utc - 1.0) <= ts <= next_turn_utc
    ]
    if end_completions:
        end_utc: float | None = min(end_completions)
    else:
        in_turn_events = [
            ts
            for etype, ts, _, _ in parsed
            if not etype.startswith("observer.")
            and etype != "operator.save_video"
            and last_seg_utc <= ts <= next_turn_utc
        ]
        end_utc = max(in_turn_events) if in_turn_events else None

    recovered_start = (start_utc - utc_minus_mono) if start_utc is not None else None
    recovered_end = (end_utc - utc_minus_mono) if end_utc is not None else None
    return recovered_start, recovered_end


def _build_timeline(
    report: dict,
    sources: list[_Source],
    directory: Path | None,
    *,
    events: list[dict] | None = None,
    run_root: Path | None = None,
) -> dict:
    """Build full-episode thinking vs motion intervals and per-segment metadata."""
    frame_bounds: dict[int, list[float]] = {}
    frame_action: dict[int, str | None] = {}
    for source in sources:
        for seg_idx, stamp, aid in zip(source.segments, source.times, source.actions):
            if seg_idx not in frame_bounds:
                frame_bounds[seg_idx] = [stamp, stamp]
            else:
                frame_bounds[seg_idx][0] = min(frame_bounds[seg_idx][0], stamp)
                frame_bounds[seg_idx][1] = max(frame_bounds[seg_idx][1], stamp)
            if frame_action.get(seg_idx) is None and aid is not None:
                frame_action[seg_idx] = str(aid)

    raw_by_idx: dict[int, dict] = {}
    for pos, seg in enumerate(report.get("segments") or []):
        if not isinstance(seg, dict):
            continue
        idx = int(seg.get("segment_index", pos))
        s0 = _finite_float(seg.get("start_monotonic_s"))
        s1 = _finite_float(seg.get("end_monotonic_s"))
        fb = frame_bounds.get(idx)
        if s0 is None and fb is not None:
            s0 = fb[0]
        if s1 is None:
            s1 = fb[1] if fb is not None else s0
        if s0 is None or s1 is None:
            continue
        if fb is not None:
            s0 = min(s0, fb[0])
            s1 = max(s1, fb[1])
        s1 = max(s0, s1)
        aid = seg.get("action_id")
        if aid is None:
            aid = frame_action.get(idx)
        raw_by_idx[idx] = {
            "segment_index": idx,
            "action_id": None if aid is None else str(aid),
            "start_monotonic_s": s0,
            "end_monotonic_s": s1,
        }

    for idx, (f0, f1) in frame_bounds.items():
        if idx not in raw_by_idx:
            raw_by_idx[idx] = {
                "segment_index": idx,
                "action_id": frame_action.get(idx),
                "start_monotonic_s": f0,
                "end_monotonic_s": max(f0, f1),
            }

    manifest_start = _finite_float(report.get("recording_started_monotonic_s"))
    manifest_end = _finite_float(report.get("recording_ended_monotonic_s"))

    motion_segments = [raw_by_idx[idx] for idx in sorted(raw_by_idx)]
    if not motion_segments:
        if manifest_start is not None and manifest_end is not None:
            span = max(0.0, manifest_end - manifest_start)
            intervals = (
                [
                    {
                        "kind": "thinking",
                        "start_s": 0.0,
                        "end_s": round(span, 3),
                        "duration_s": round(span, 3),
                        "action_id": None,
                    }
                ]
                if span >= 1e-3
                else []
            )
            return {
                "task_start_s": manifest_start,
                "task_end_s": manifest_start + span,
                "total_task_s": span,
                "total_motion_s": 0.0,
                "total_thinking_s": span,
                "motion_segments": [],
                "by_segment_index": {},
                "raw_intervals": [],
                "summary": {
                    "bounds_source": "manifest",
                    "semantics": TIMELINE_SEMANTICS,
                    "total_task_s": round(span, 3),
                    "total_motion_s": 0.0,
                    "total_thinking_skipped_s": round(span, 3),
                    "intervals": intervals,
                },
            }
        return {
            "task_start_s": 0.0,
            "task_end_s": 0.0,
            "total_task_s": 0.0,
            "total_motion_s": 0.0,
            "total_thinking_s": 0.0,
            "motion_segments": [],
            "by_segment_index": {},
            "raw_intervals": [],
            "summary": None,
        }

    # Ensure strictly monotonic non-overlapping segment bounds even if host clock jittered.
    running_end = motion_segments[0]["start_monotonic_s"]
    for seg in motion_segments:
        seg["start_monotonic_s"] = max(running_end, seg["start_monotonic_s"])
        seg["end_monotonic_s"] = max(seg["start_monotonic_s"], seg["end_monotonic_s"])
        running_end = seg["end_monotonic_s"]

    first_seg_start = motion_segments[0]["start_monotonic_s"]
    last_seg_end = motion_segments[-1]["end_monotonic_s"]

    if manifest_start is not None and manifest_end is not None:
        task_start_s = min(manifest_start, first_seg_start)
        task_end_s = max(manifest_end, last_seg_end, task_start_s)
        bounds_source = "manifest"
    else:
        ev_start, ev_end = _recover_episode_bounds_from_events(
            directory, motion_segments, events=events, run_root=run_root
        )
        start_src = manifest_start if manifest_start is not None else ev_start
        end_src = manifest_end if manifest_end is not None else ev_end
        if manifest_start is not None or manifest_end is not None:
            bounds_source = "manifest" if (ev_start is None and ev_end is None) else "manifest_and_events_sqlite"
        elif ev_start is not None or ev_end is not None:
            bounds_source = "events_sqlite"
        else:
            bounds_source = "segments_fallback"
        task_start_s = min(start_src, first_seg_start) if start_src is not None else first_seg_start
        task_end_s = max(end_src, last_seg_end, task_start_s) if end_src is not None else max(last_seg_end, task_start_s)

    raw_intervals: list[dict] = []
    public_intervals: list[dict] = []
    by_segment_index: dict[int, dict] = {}
    cursor_s = 0.0
    for move_num, seg in enumerate(motion_segments, 1):
        m_start = max(cursor_s, seg["start_monotonic_s"] - task_start_s)
        m_end = max(m_start, seg["end_monotonic_s"] - task_start_s)
        think_dur = max(0.0, m_start - cursor_s)
        seg["move_number"] = move_num
        seg["start_rel_s"] = m_start
        seg["end_rel_s"] = m_end
        seg["duration_s"] = m_end - m_start
        seg["preceding_thinking"] = {
            "start_s": cursor_s,
            "end_s": m_start,
            "duration_s": think_dur,
        }
        if think_dur >= 1e-3:
            raw_intervals.append({
                "kind": "thinking",
                "start_s": cursor_s,
                "end_s": m_start,
                "duration_s": think_dur,
                "action_id": None,
            })
            public_intervals.append({
                "kind": "thinking",
                "start_s": round(cursor_s, 3),
                "end_s": round(m_start, 3),
                "duration_s": round(think_dur, 3),
                "action_id": None,
            })
        raw_intervals.append({
            "kind": "motion",
            "move_number": move_num,
            "segment_index": seg["segment_index"],
            "start_s": m_start,
            "end_s": m_end,
            "duration_s": m_end - m_start,
            "action_id": seg["action_id"],
        })
        public_intervals.append({
            "kind": "motion",
            "segment_index": seg["segment_index"],
            "start_s": round(m_start, 3),
            "end_s": round(m_end, 3),
            "duration_s": round(m_end - m_start, 3),
            "action_id": seg["action_id"],
        })
        by_segment_index[seg["segment_index"]] = seg
        cursor_s = m_end

    total_task_s = max(cursor_s, task_end_s - task_start_s)
    tail_think_dur = max(0.0, total_task_s - cursor_s)
    if tail_think_dur >= 1e-3:
        raw_intervals.append({
            "kind": "thinking",
            "start_s": cursor_s,
            "end_s": total_task_s,
            "duration_s": tail_think_dur,
            "action_id": None,
        })
        public_intervals.append({
            "kind": "thinking",
            "start_s": round(cursor_s, 3),
            "end_s": round(total_task_s, 3),
            "duration_s": round(tail_think_dur, 3),
            "action_id": None,
        })

    total_motion_s = sum(seg["duration_s"] for seg in motion_segments)
    total_thinking_s = max(0.0, total_task_s - total_motion_s)
    summary = {
        "bounds_source": bounds_source,
        "semantics": TIMELINE_SEMANTICS,
        "total_task_s": round(total_task_s, 3),
        "total_motion_s": round(total_motion_s, 3),
        "total_thinking_skipped_s": round(total_thinking_s, 3),
        "intervals": public_intervals,
    }
    return {
        "task_start_s": task_start_s,
        "task_end_s": task_start_s + total_task_s,
        "total_task_s": total_task_s,
        "total_motion_s": total_motion_s,
        "total_thinking_s": total_thinking_s,
        "motion_segments": motion_segments,
        "by_segment_index": by_segment_index,
        "raw_intervals": raw_intervals,
        "summary": summary,
    }


def _resolve_cameras(report: dict, directory: Path | None) -> tuple[list[dict], list[dict], Path | None]:
    cameras = []
    missing = []
    for raw_camera in report.get("cameras", []):
        video_path = raw_camera.get("video_path")
        if not video_path:
            missing.append({"camera_id": raw_camera.get("camera_id"), "status": raw_camera.get("status")})
            continue
        cam = dict(raw_camera)
        if directory is not None:
            # Bind both video and mapping strictly into the target directory so a partial copy
            # never mixes local video with a stale source mapping from another path.
            cam["video_path"] = str(directory / Path(video_path).name)
            mapping_path = raw_camera.get("frame_mapping_path")
            if mapping_path:
                cam["frame_mapping_path"] = str(directory / Path(mapping_path).name)
        cameras.append(cam)
    resolved_dir = Path(directory) if directory is not None else (Path(cameras[0]["video_path"]).parent if cameras else None)
    return cameras, missing, resolved_dir


def _rebase_cached_report(previous: dict, resolved_dir: Path) -> dict:
    rebased = dict(previous)
    rebased["video_path"] = str(resolved_dir / f"{NAME}.mp4")
    rebased["frame_mapping_path"] = str(resolved_dir / f"{NAME}.frames.jsonl")
    rebased["encoder_log_path"] = str(resolved_dir / f"{NAME}.ffmpeg.log")
    if isinstance(previous.get("sources"), list):
        rebased["sources"] = [
            dict(item, video_path=str(resolved_dir / Path(item["video_path"]).name))
            if isinstance(item, dict) and item.get("video_path")
            else item
            for item in previous["sources"]
        ]
    return rebased


@contextmanager
def _directory_lock(directory: Path):
    lock_path = directory / f".{NAME}.lock"
    handle = lock_path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        yield
    finally:
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
        finally:
            handle.close()


def _cached_composite_if_valid(
    saved: Path,
    resolved_dir: Path,
    expected_header: dict | None,
    *,
    include_header: bool,
) -> dict | None:
    if not saved.is_file():
        return None
    try:
        previous = json.loads(saved.read_text())
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(previous, dict) or previous.get("status") != "complete":
        return None
    if (
        previous.get("hud_render_version") != HUD_RENDER_VERSION
        or previous.get("timeline_hud_height") != FOOTER_HEIGHT
        or not isinstance(previous.get("timeline_summary"), dict)
    ):
        return None
    if not include_header:
        if previous.get("header_hud_height") != 0:
            return None
    elif expected_header is not None:
        if previous.get("header_hud_height") != HEADER_HEIGHT or previous.get("header_summary") != expected_header:
            return None
    expected_vsize = previous.get("video_byte_size")
    expected_msize = previous.get("mapping_byte_size")
    if not isinstance(expected_vsize, int) or not isinstance(expected_msize, int):
        return None
    local_mp4 = resolved_dir / f"{NAME}.mp4"
    local_mapping = resolved_dir / f"{NAME}.frames.jsonl"
    if not local_mp4.is_file() or not local_mapping.is_file():
        return None
    if local_mp4.stat().st_size != expected_vsize or local_mapping.stat().st_size != expected_msize:
        return None
    return _rebase_cached_report(previous, resolved_dir)


def _spawn_encoder(
    ffmpeg_bin: str,
    out_path: Path,
    width: int,
    height: int,
    fps: float,
    crf: int,
    threads: int,
    log,
) -> subprocess.Popen:
    return subprocess.Popen(
        [
            ffmpeg_bin,
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "rawvideo",
            "-pixel_format",
            "rgb24",
            "-video_size",
            f"{width}x{height}",
            "-framerate",
            f"{fps:.6f}",
            "-i",
            "pipe:0",
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "medium",
            "-crf",
            str(crf),
            "-threads",
            str(max(1, int(threads))),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            "-f",
            "mp4",
            str(out_path),
        ],
        stdin=subprocess.PIPE,
        stdout=subprocess.DEVNULL,
        stderr=log,
    )


def _verify_encoded_video(ffprobe_bin: str, path: Path, width: int, height: int, expected_frames: int, log) -> None:
    probe = json.loads(
        subprocess.check_output(
            [
                ffprobe_bin,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=codec_name,width,height,nb_frames",
                "-of",
                "json",
                str(path),
            ],
            stderr=log,
            timeout=60,
        )
    )["streams"][0]
    if (
        probe["codec_name"] != "h264"
        or int(probe["nb_frames"]) != expected_frames
        or (int(probe["width"]), int(probe["height"])) != (width, height)
    ):
        raise RuntimeError("encoded_video_verification_failed")


def find_recording_dir(target: Path) -> tuple[Path, Path | None]:
    """Resolve ``target`` (either a recording dir with ``manifest.json`` or a run dir) to ``(recording_dir, run_dir)``."""
    target = Path(target).expanduser()
    if (target / "manifest.json").is_file():
        manifest = json.loads((target / "manifest.json").read_text(encoding="utf-8"))
        if isinstance(manifest, dict) and "cameras" in manifest and "segments" in manifest:
            return target, find_run_root(target)
    videos_root = target / "private" / "robot" / "videos"
    if videos_root.is_dir():
        candidates = sorted(
            (p.parent for p in videos_root.glob("*/manifest.json")),
            key=lambda p: (p / "manifest.json").stat().st_mtime,
        )
        if candidates:
            return candidates[-1], target
    raise FileNotFoundError(
        f"{target}: expected a recording directory with manifest.json or a run directory with "
        "private/robot/videos/*/manifest.json"
    )


def compose(
    report,
    *,
    directory=None,
    run_dir=None,
    config_dict: dict | None = None,
    ffmpeg=None,
    ffprobe=None,
    overwrite: bool = False,
    output_path=None,
    crf: int | None = None,
    threads: int = DEFAULT_THREADS,
    model_label: str | None = None,
    header_label: str | None = None,
    include_header: bool = True,
    preview_frame: int | None = None,
    preview_out=None,
):
    """Write ``combined.mp4`` (optional header + camera tiles + native FreeType timeline footer).

    ``report``: dict from ``MotionVideoRecorder.save()`` (``cameras`` entries need
    ``camera_id``, ``video_path``, ``frame_mapping_path``, ``width``/``height`` px,
    ``fps``). ``directory``: optional override recording directory (e.g. relocated
    archive copy). ``run_dir``: optional explicit run archive root for header
    metadata/events (auto-discovered from ``directory`` ancestors by default).
    ``output_path``: optional custom MP4 output path (writes atomically without
    replacing ``combined.mp4``). ``preview_frame`` / ``preview_out``: encode and
    decode only that single output frame index to a PNG for visual verification.
    """
    explicit_tools = ffmpeg is not None or ffprobe is not None
    custom_render = (
        output_path is not None
        or preview_frame is not None
        or crf is not None
        or model_label is not None
        or header_label is not None
        or not include_header
    )
    target_dir = Path(directory) if directory is not None else None
    cameras, missing, resolved_dir = _resolve_cameras(report, target_dir)
    resolved_run_root = Path(run_dir).expanduser() if run_dir is not None else find_run_root(resolved_dir)

    sqlite_events, runtime_state = read_sqlite_snapshot(resolved_run_root) if resolved_run_root is not None else ([], {})
    header_summary = resolve_header_metadata(
        resolved_dir,
        run_dir=resolved_run_root,
        model_override=model_label,
        label_override=header_label,
        include_header=include_header,
        runtime_state=runtime_state,
        config_dict=config_dict,
    )
    header_h = HEADER_HEIGHT if header_summary is not None else 0

    empty_timeline = (
        _build_timeline(report, [], resolved_dir, events=sqlite_events, run_root=resolved_run_root)
        if not cameras
        else None
    )
    result = dict(
        audience="observer_only",
        kind="side_by_side_composite",
        status="empty",
        frames=0,
        fps=None,
        width=None,
        height=None,
        tile_height=None,
        header_hud_height=header_h,
        timeline_hud_height=FOOTER_HEIGHT,
        hud_render_version=HUD_RENDER_VERSION,
        duration_s=None,
        codec="h264",
        crf=None,
        preset="medium",
        camera_order=[camera["camera_id"] for camera in cameras],
        missing_cameras=missing,
        alignment=ALIGNMENT,
        timing=TIMING,
        segments=len(report.get("segments", [])),
        header_summary=header_summary,
        timeline_summary=empty_timeline["summary"] if empty_timeline is not None else None,
        error=None,
    )
    if not cameras or resolved_dir is None:
        return result
    if not resolved_dir.is_dir():
        result.update(status="failed", error=f"FileNotFoundError: recording directory not found: {resolved_dir}")
        return result

    saved = resolved_dir / f"{NAME}.json"
    is_preview = preview_frame is not None
    is_custom_out = output_path is not None
    write_default_artifacts = not is_preview and not is_custom_out
    lock_ctx = _directory_lock(resolved_dir) if write_default_artifacts else nullcontext()
    with lock_ctx:
        if not overwrite and not explicit_tools and not custom_render:
            cached = _cached_composite_if_valid(saved, resolved_dir, header_summary, include_header=include_header)
            if cached is not None:
                return cached

        ffmpeg_bin, ffprobe_bin = ffmpeg or shutil.which("ffmpeg"), ffprobe or shutil.which("ffprobe")
        path = Path(output_path).expanduser() if is_custom_out else (resolved_dir / f"{NAME}.mp4")
        mapping_path = resolved_dir / f"{NAME}.frames.jsonl"
        log_path = resolved_dir / f"{NAME}.ffmpeg.log"
        uid = uuid.uuid4().hex[:12]
        partial = (
            path.with_name(f"{path.stem}.{uid}.tmp{path.suffix}")
            if is_custom_out
            else (resolved_dir / f"{NAME}.{uid}.partial")
        )
        mapping_tmp = resolved_dir / f"{NAME}.{uid}.frames.jsonl.tmp"
        saved_tmp = resolved_dir / f"{NAME}.{uid}.json.tmp"

        fps = max(float(camera["fps"]) for camera in cameras)
        resolved_crf = int(crf) if crf is not None else max(int(camera.get("crf") or DEFAULT_CRF) for camera in cameras)
        width = sum(int(camera["width"]) for camera in cameras)
        tile_height = max(int(camera["height"]) for camera in cameras)
        # yuv420p needs even dimensions; pad rather than resizing source pixels.
        width, tile_height = width + width % 2, tile_height + tile_height % 2
        height = header_h + tile_height + FOOTER_HEIGHT
        result.update(
            fps=fps,
            crf=resolved_crf,
            width=width,
            height=height,
            tile_height=tile_height,
            frame_mapping_path=str(mapping_path) if write_default_artifacts else None,
            encoder_log_path=str(log_path) if write_default_artifacts else None,
        )
        if not ffmpeg_bin or not ffprobe_bin:
            result.update(status="failed", error="ffmpeg_or_ffprobe_not_found")
            return result

        import numpy

        sources: list[_Source] = []
        encoder = None
        frames = 0
        try:
            if is_custom_out:
                path.parent.mkdir(parents=True, exist_ok=True)
            log_ctx = log_path.open("ab") if write_default_artifacts else tempfile.TemporaryFile()
            with log_ctx as log:
                mapping_handle = mapping_tmp.open("w") if write_default_artifacts else None
                try:
                    for camera in cameras:  # append one by one so `finally` reaps every started decoder
                        sources.append(_Source(camera, ffmpeg_bin, log))
                    segments = sorted({segment for source in sources for segment in source.segments})
                    total_clip_frames = sum(max(source.count(seg) for source in sources) for seg in segments)
                    timeline = _build_timeline(
                        report, sources, resolved_dir, events=sqlite_events, run_root=resolved_run_root
                    )
                    result["timeline_summary"] = timeline["summary"]
                    if not total_clip_frames:
                        return result
                    if is_preview and not (0 <= int(preview_frame) < total_clip_frames):
                        raise ValueError(
                            f"preview_frame_out_of_range: {preview_frame} not in 0..{total_clip_frames - 1}"
                        )

                    fonts = HudFonts()
                    hud = TimelineHud(width, timeline, total_clip_frames, fps, fonts=fonts)
                    header_img = render_header(width, header_summary, fonts) if header_summary is not None else None
                    camera_pills = {
                        source.camera_id: make_camera_pill(source.camera_id, fonts, max_width=max(24, source.width - 12))
                        for source in sources
                    }
                    no_frame_badge = make_no_frame_badge(fonts)

                    # Reuse a single frame canvas for the entire encode; paint static header once.
                    canvas = numpy.zeros((height, width, 3), numpy.uint8)
                    if header_img is not None:
                        canvas[:header_h, :] = header_img

                    last_display_mono: float | None = None
                    preview_tmp_dir = tempfile.TemporaryDirectory(prefix="ni-rua-video-preview-") if is_preview else None
                    try:
                        enc_target = (
                            Path(preview_tmp_dir.name) / "preview.mp4" if preview_tmp_dir is not None else partial
                        )
                        for segment in segments:
                            reference = max(sources, key=lambda source: source.count(segment))  # max() keeps first on ties
                            rows = [index for index, value in enumerate(reference.segments) if value == segment]
                            seg_meta = timeline["by_segment_index"].get(segment)
                            for row in rows:
                                time_s = reference.times[row]
                                display_s = time_s
                                if seg_meta is not None:
                                    display_s = min(seg_meta["end_monotonic_s"], max(seg_meta["start_monotonic_s"], display_s))
                                if last_display_mono is not None and display_s < last_display_mono:
                                    display_s = last_display_mono
                                last_display_mono = display_s

                                should_render = (not is_preview) or (frames == int(preview_frame))
                                if should_render:
                                    canvas[header_h:header_h + tile_height, :] = 0
                                shown, left = {}, 0
                                for source in sources:
                                    index, image = reference.seek(row) if source is reference else source.at(segment, time_s)
                                    if should_render:
                                        if image is not None:
                                            canvas[header_h:header_h + source.height, left:left + source.width] = image
                                        else:
                                            stamp_patch(
                                                canvas,
                                                no_frame_badge,
                                                left + max(6, (source.width - no_frame_badge.shape[1]) // 2),
                                                header_h + max(6, (source.height - no_frame_badge.shape[0]) // 2),
                                            )
                                        stamp_patch(canvas, camera_pills[source.camera_id], left + 6, header_h + 6)
                                    shown[source.camera_id] = index
                                    left += source.width
                                if should_render:
                                    hud.render_into(canvas, header_h + tile_height, segment, display_s, frames / fps)
                                    if encoder is None:
                                        encoder = _spawn_encoder(
                                            ffmpeg_bin, enc_target, width, height, fps, resolved_crf, threads, log
                                        )
                                    encoder.stdin.write(canvas.tobytes())
                                if mapping_handle is not None:
                                    mapping_handle.write(
                                        json.dumps(
                                            dict(
                                                video_frame=frames,
                                                pts_s=frames / fps,
                                                segment_index=segment,
                                                action_id=reference.actions[row],
                                                reference_camera=reference.camera_id,
                                                capture_monotonic_s=time_s,
                                                task_elapsed_s=round(max(0.0, display_s - timeline["task_start_s"]), 4),
                                                source_frames=shown,
                                            )
                                        )
                                        + "\n"
                                    )
                                frames += 1
                                if is_preview and frames > int(preview_frame):
                                    break
                            if is_preview and frames > int(preview_frame):
                                break

                        if encoder is None:
                            return result
                        encoder.stdin.close()
                        if encoder.wait(timeout=600):
                            raise RuntimeError("ffmpeg_failed")

                        if is_preview:
                            out_png = Path(preview_out or "hud_preview.png").expanduser()
                            out_png.parent.mkdir(parents=True, exist_ok=True)
                            subprocess.run(
                                [ffmpeg_bin, "-v", "error", "-y", "-i", str(enc_target), "-frames:v", "1", str(out_png)],
                                check=True,
                                stderr=log,
                                timeout=60,
                            )
                            result.update(
                                status="complete",
                                frames=total_clip_frames,
                                duration_s=round(total_clip_frames / fps, 3),
                                preview_frame=int(preview_frame),
                                preview_path=str(out_png),
                                sources=[
                                    dict(camera_id=source.camera_id, video_path=str(camera["video_path"]), frames=len(source.times))
                                    for source, camera in zip(sources, cameras)
                                ],
                            )
                            return result

                        _verify_encoded_video(ffprobe_bin, partial, width, height, frames, log)
                    finally:
                        if preview_tmp_dir is not None:
                            preview_tmp_dir.cleanup()
                finally:
                    if mapping_handle is not None:
                        mapping_handle.close()

            video_bytes = partial.stat().st_size
            result.update(
                status="complete",
                frames=frames,
                duration_s=round(frames / fps, 3),
                video_path=str(path),
                video_byte_size=video_bytes,
                sources=[
                    dict(camera_id=source.camera_id, video_path=str(camera["video_path"]), frames=len(source.times))
                    for source, camera in zip(sources, cameras)
                ],
            )
            if write_default_artifacts:
                mapping_bytes = mapping_tmp.stat().st_size
                result["mapping_byte_size"] = mapping_bytes
                saved_tmp.write_text(json.dumps(result, indent=2))
                # Invalidate prior saved marker before replacing artifacts so an interrupted rename never leaves a stale "complete" receipt.
                saved.unlink(missing_ok=True)
                mapping_tmp.replace(mapping_path)
                partial.replace(path)
                saved_tmp.replace(saved)
            else:
                os.replace(partial, path)
        except Exception as error:
            result.update(status="failed", frames=frames, error=f"{type(error).__name__}: {error}")
        finally:
            for source in sources:
                source.close()
            if encoder is not None:
                if encoder.poll() is None:
                    encoder.kill()
                encoder.wait()
            partial.unlink(missing_ok=True)
            mapping_tmp.unlink(missing_ok=True)
            saved_tmp.unlink(missing_ok=True)
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Compose a sealed SO-101 recording into one side-by-side MP4 with native-resolution HUD."
    )
    parser.add_argument(
        "target",
        help="Recording directory containing manifest.json, OR a NI-RUA run directory (private/robot/videos/*)",
    )
    parser.add_argument("--recording", help="Explicit recording dir when target is a run directory.")
    parser.add_argument("--output", "-o", help="Export to a custom MP4 path instead of replacing combined.mp4.")
    parser.add_argument("--overwrite", action="store_true", help="Regenerate combined.mp4 even if already complete.")
    parser.add_argument("--model-label", help="Override the prettified model display name in the header.")
    parser.add_argument("--header-label", help="Override the entire left header text.")
    parser.add_argument("--no-header", action="store_true", help="Omit the top model/usage header band.")
    parser.add_argument("--crf", type=int, help="Override libx264 CRF (default: source camera CRF, typically 23).")
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="libx264 encoder threads (default: 2).")
    parser.add_argument("--preview-frame", type=int, help="Render only this output frame index to --preview-out.")
    parser.add_argument("--preview-out", default="hud_preview.png", help="PNG path for --preview-frame.")
    arguments = parser.parse_args(argv)

    target = Path(arguments.target).expanduser()
    if arguments.recording:
        recording_dir = Path(arguments.recording).expanduser()
        run_dir = target
    else:
        try:
            recording_dir, run_dir = find_recording_dir(target)
        except FileNotFoundError as error:
            parser.error(f"{error} (stop the gateway or save from the console to seal the recording)")
    manifest = recording_dir / "manifest.json"
    if not manifest.is_file():
        parser.error(f"{manifest} not found: the recording is not sealed yet (stop the gateway or save from the console)")

    result = compose(
        json.loads(manifest.read_text(encoding="utf-8")),
        directory=recording_dir,
        run_dir=run_dir,
        overwrite=arguments.overwrite,
        output_path=arguments.output,
        crf=arguments.crf,
        threads=arguments.threads,
        model_label=arguments.model_label,
        header_label=arguments.header_label,
        include_header=not arguments.no_header,
        preview_frame=arguments.preview_frame,
        preview_out=arguments.preview_out,
    )
    print(json.dumps(result, indent=2))
    return 0 if result["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())
