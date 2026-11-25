"""Derive a camera calibration for another capture resolution of the same SO-101 camera (offline, no hardware).

The SO-101 worker refuses a calibration whose ``image_size`` differs from the deployment's capture ``width`` x
``height`` (intrinsics must belong to the pixels actually captured). Changing the capture mode (for example the fixed
camera from 640x480 to 1280x720 or 1920x1080) therefore needs intrinsics for the new mode. A 16:9 mode of a UVC camera
is usually a crop and/or a different scaling of the sensor than its 4:3 mode, so K cannot be scaled by the ratio of
the image sizes; this script measures the pixel map instead:

1. Two images of one static ChArUco board (the calibration file's board), taken by the same camera without moving
   it or the board, one in the calibrated mode (``--from-image``) and one in the new mode (``--to-image``). Take them
   with ``scripts/so101_perception_bench.py snapshot`` (needs camera permission) or any capture tool.
2. The board corners detected in both images are matched by id and ``so101_calibration.fit_pixel_mapping`` fits
   ``new_px = scale * old_px + offset`` per axis (RMS residual reported).
3. ``so101_calibration.remap_camera_intrinsics`` writes K for the new mode (distortion and extrinsic unchanged,
   residuals carried over, ``intrinsics_source`` naming the derivation) into a NEW calibration file (``--out``; an
   existing file is never replaced). The fit is refused when its RMS exceeds ``--max-rms-px`` (default: the
   deployment's ``calibration.max_intrinsics_rms_px`` when ``--deployment`` is given, else 1.0 px): the two modes are
   then not one affine map of the same pixels, and the camera must be recalibrated in the new mode
   (/calibration S3 hand-held intrinsics, then S4).

Then point the deployment's ``camera_calibration_file`` at the new file (or install it through the calibration
workflow) and set ``cameras.<camera>.width`` / ``height`` to the new mode. Example (NI-RUA or so101 env)::

    PYTHONPATH=. mamba run -n ni-rua python scripts/so101_remap_calibration.py \\
        --calibration ~/.cache/ni-rua/so101/camera_calibration.json --camera fixed \\
        --from-image /tmp/board-640x480.png --to-image /tmp/board-1920x1080.png \\
        --out ~/.cache/ni-rua/so101/camera_calibration-fixed-1920x1080.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from ni_rua.robot.calibration.workflow import charuco as cal  # noqa: E402


def matched_corners(board, source_image, target_image, min_corners):
    """(source (N, 2), target (N, 2)) px of the board corners detected in both images, matched by id."""
    found = []
    for path in (source_image, target_image):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            raise SystemExit(f"cannot read image {path}")
        detection = cal.detect_charuco(image, board, min_corners=min_corners)
        if detection is None:
            raise SystemExit(f"no ChArUco board ({min_corners}+ corners) in {path}")
        found.append((dict(zip(detection["ids"].tolist(), detection["corners"])), image.shape[1::-1]))
    (source, size_source), (target, size_target) = found
    ids = sorted(set(source) & set(target))
    if len(ids) < min_corners:
        raise SystemExit(f"only {len(ids)} corners seen in both images (need {min_corners})")
    return (np.array([source[i] for i in ids]), np.array([target[i] for i in ids]),
            [int(v) for v in size_source], [int(v) for v in size_target])


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--calibration", required=True, help="Existing camera_calibration.json (read only).")
    parser.add_argument("--camera", required=True, choices=("fixed", "wrist"), help="Camera entry to remap.")
    parser.add_argument("--from-image", required=True, help="Board image in the calibrated capture mode.")
    parser.add_argument("--to-image", required=True, help="Board image in the new capture mode (same camera and board pose).")
    parser.add_argument("--out", required=True, help="New calibration file to write (must not exist).")
    parser.add_argument("--deployment", help="Deployment file whose calibration.max_intrinsics_rms_px is the fit gate.")
    parser.add_argument("--max-rms-px", type=float, help="Largest accepted fit RMS (new-mode px); overrides --deployment.")
    parser.add_argument("--min-corners", type=int, default=12, help="Corners required in each image and in common (default 12).")
    args = parser.parse_args(argv)
    if Path(args.out).expanduser().exists():
        raise SystemExit(f"{args.out} exists; choose a new file (an existing calibration is never replaced)")
    document = json.loads(Path(args.calibration).expanduser().read_text(encoding="utf-8"))
    validated = cal.validate_calibration_document(document)
    if args.camera not in validated["cameras"]:
        raise SystemExit(f"{args.calibration} has no {args.camera} camera")
    gate = args.max_rms_px
    if gate is None and args.deployment:
        from ni_rua.robot.backends.so101.deployment import load_deployment
        gate = load_deployment(args.deployment)["calibration"]["max_intrinsics_rms_px"]
    gate = 1.0 if gate is None else float(gate)
    board = cal.make_board(validated["board"])
    source, target, size_source, size_target = matched_corners(board, args.from_image, args.to_image, args.min_corners)
    old_size = validated["cameras"][args.camera]["image_size"]
    if size_source != list(old_size):
        raise SystemExit(f"--from-image is {size_source[0]}x{size_source[1]} px; the calibration is {old_size[0]}x{old_size[1]}")
    mapping = cal.fit_pixel_mapping(source, target)
    report = dict(camera=args.camera, from_size=old_size, to_size=size_target, **mapping, gate_rms_px=gate)
    print(json.dumps(report, indent=1))
    if mapping["rms_px"] > gate:
        raise SystemExit(f"fit rms {mapping['rms_px']:.3f} px > {gate:.3f} px: the two modes are not one affine map of "
                         "the same pixels; recalibrate this camera in the new mode instead")
    cameras = {name: dict(camera) for name, camera in document["cameras"].items()}
    cameras[args.camera] = cal.remap_camera_intrinsics(cameras[args.camera], mapping, size_target, from_size=old_size)
    written = cal.write_calibration_file(Path(args.out).expanduser(), board=document["board"], cameras=cameras,
                                         samples=document.get("samples", 0), overwrite=False,
                                         joint_calibration=document.get("joint_calibration"))
    print(json.dumps(dict(written=str(Path(args.out).expanduser()), K=written["cameras"][args.camera]["K"],
                          image_size=written["cameras"][args.camera]["image_size"]), indent=1))


if __name__ == "__main__":
    main()
