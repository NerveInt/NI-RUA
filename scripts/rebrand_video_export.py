#!/usr/bin/env python3
"""Compatibility CLI wrapper over ``ni_rua.robot.backends.so101.motion_video_composite``.

The crisp native-resolution FreeType HUD (header with Backend-Model-Effort,
Tokens and Cost + 1:1 camera tiles with pill labels + timeline footer) is now
the default automatic export pipeline inside NI-RUA. This script remains as a
convenience wrapper for exporting a run directory to a custom output MP4 or
single-frame PNG preview::

    mamba run -n ni-rua python scripts/rebrand_video_export.py RUN_DIR --preview-frame 100 --preview-out f.png
    mamba run -n ni-rua python scripts/rebrand_video_export.py RUN_DIR OUT.mp4
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

_REPO_ROOT = Path(__file__).resolve().parent.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from ni_rua.robot.backends.so101.motion_video_composite import (
    DEFAULT_CRF,
    DEFAULT_THREADS,
    compose,
    find_recording_dir,
    fmt_cost,
    fmt_tokens,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", help="NI-RUA run archive directory or sealed recording directory")
    parser.add_argument("dst", nargs="?", help="Output MP4 path (required unless --preview-frame is given)")
    parser.add_argument("--recording", help="Recording dir; default: newest private/robot/videos/*/manifest.json")
    parser.add_argument("--model-label", help="Override the prettified model display name")
    parser.add_argument("--header-label", help="Override the full left header text")
    parser.add_argument("--crf", type=int, default=DEFAULT_CRF)
    parser.add_argument("--threads", type=int, default=DEFAULT_THREADS, help="libx264 threads (default 2)")
    parser.add_argument("--preview-frame", type=int, help="Render only this output frame index to --preview-out")
    parser.add_argument("--preview-out", default="hud_preview.png")
    args = parser.parse_args(argv)

    if args.preview_frame is None and not args.dst:
        raise SystemExit("DST is required unless --preview-frame is given")

    target = Path(args.run_dir).expanduser()
    if args.recording:
        recording_dir = Path(args.recording).expanduser()
        run_dir = target
    else:
        recording_dir, run_dir = find_recording_dir(target)

    manifest_path = recording_dir / "manifest.json"
    if not manifest_path.is_file():
        raise SystemExit(f"{manifest_path}: recording manifest not found")

    result = compose(
        json.loads(manifest_path.read_text(encoding="utf-8")),
        directory=recording_dir,
        run_dir=run_dir,
        overwrite=True,
        output_path=args.dst,
        crf=args.crf,
        threads=args.threads,
        model_label=args.model_label,
        header_label=args.header_label,
        include_header=True,
        preview_frame=args.preview_frame,
        preview_out=args.preview_out,
    )
    if result.get("status") != "complete":
        raise SystemExit(f"export failed: {result.get('error') or result.get('status')}")

    header = result.get("header_summary") or {}
    info = (
        f"{result['width']}x{result['height']} fps={result['fps']:.5f} frames={result['frames']} "
        f"tokens={fmt_tokens(header.get('total_tokens'))} "
        f"cost={fmt_cost(header.get('cost_usd'), header.get('currency'))}"
    )
    if args.preview_frame is not None:
        print(f"preview {result['preview_path']}: {info}")
    else:
        print(f"wrote {result['video_path']}: {info}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
