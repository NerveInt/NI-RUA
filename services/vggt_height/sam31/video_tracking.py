"""Explicit offline SAM 3.1 multiplex mask tracking; no live-sidecar changes or motion.

Official API verified at the revision pinned by services/vggt_height/setup.sh. Point prompts select
an instance mask; they do not produce a persistent physical/material keypoint. All contact and
physical-visibility judgments remain unknown. Importing this module never imports torch or SAM3.
"""
from __future__ import annotations

import argparse
import functools
import hashlib
import inspect
import json
from pathlib import Path
import subprocess

import numpy as np

try:
    from ni_rua.robot.calibration.evidence.config import evidence_settings
except ModuleNotFoundError as error:
    if error.name != "ni_rua":
        raise
    # Standalone GPU deployment copies this standard-library companion and its
    # JSON from ni_rua/robot/calibration/evidence alongside this script; no duplicated defaults.
    from config import evidence_settings

DEFAULTS = evidence_settings("sam31_video")
from PIL import Image

SAM3_REVISION = "7f473cfd983d5eb43040ff608f66bff0d7bdcb53"
SCHEMA = "ni_rua.sam31.video_tracking.v1"
IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"}


def _frames(directory):
    directory = Path(directory).resolve()
    if not directory.is_dir():
        raise ValueError("frames_dir must be an existing image directory")
    paths = [p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTENSIONS]
    try:
        paths.sort(key=lambda p: int(p.stem))
    except ValueError as error:
        raise ValueError("frame filenames must be consecutive integers starting at 0") from error
    if not paths or [int(p.stem) for p in paths] != list(range(len(paths))):
        raise ValueError("frame filenames must be unique consecutive integers starting at 0")
    size, inventory = None, []
    for i, path in enumerate(paths):
        raw = path.read_bytes()
        with Image.open(path) as image:
            image.load()
            if size is None:
                size = image.size
            if image.size != size:
                raise ValueError("all original frames must have identical pixel dimensions")
        inventory.append({"frame_index": i, "source_file": path.name,
                          "source_sha256": hashlib.sha256(raw).hexdigest()})
    return directory, inventory, size


def _prompts(prompts, count, size):
    if not isinstance(prompts, list) or not prompts:
        raise ValueError("prompts must be a nonempty list")
    width, height = size
    requests = []
    for prompt in prompts:
        if not isinstance(prompt, dict):
            raise ValueError("each prompt must be a mapping")
        index = prompt.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < count:
            raise ValueError("prompt frame_index must be an existing zero-based integer")
        request = {"type": "add_prompt", "frame_index": index}
        if "text" in prompt:
            if set(prompt) != {"frame_index", "text"} or not isinstance(prompt["text"], str) or not prompt["text"].strip():
                raise ValueError("text prompts require only frame_index and nonempty text; no assigned object_id")
            request["text"] = prompt["text"]
        else:
            if set(prompt) != {"frame_index", "object_id", "points_px", "point_labels"}:
                raise ValueError("instance prompts require frame_index/object_id/points_px/point_labels")
            oid = prompt["object_id"]
            if isinstance(oid, bool) or not isinstance(oid, int) or oid < 0:
                raise ValueError("object_id must be a nonnegative integer mask identity")
            points, labels = np.asarray(prompt["points_px"], float), np.asarray(prompt["point_labels"])
            if points.ndim != 2 or points.shape[1:] != (2,) or not len(points) or labels.shape != (len(points),):
                raise ValueError("points_px must be nonempty N x 2 with N point_labels")
            if not np.isfinite(points).all() or np.any(points < 0) or np.any(points >= [width, height]):
                raise ValueError("points_px must be finite original-image coordinates inside the image")
            if not np.isin(labels, [0, 1]).all():
                raise ValueError("point_labels must be 0 (background) or 1 (foreground)")
            request.update(points=(points / [width, height]).tolist(), point_labels=labels.astype(int).tolist(),
                           obj_id=oid, rel_coordinates=True, clear_old_points=True)
        requests.append(request)
    return requests


def build_predictor(checkpoint_path, *, max_num_objects=DEFAULTS["max_num_objects"], use_fa3=DEFAULTS["use_fa3"],
                       output_prob_threshold=DEFAULTS["output_prob_threshold"]):
    """Explicitly load the pinned official SAM3.1 multiplex model; this function requires its GPU env.

    checkpoint_path: required local sam3.1_multiplex.pt file; never implicitly downloads weights.
    max_num_objects: positive integer object-mask capacity, default 16 (official builder default).
    use_fa3: enable FlashAttention 3 only when installed/supported, default False for compatibility.
    output_prob_threshold: dimensionless output threshold in [0,1], default 0.5, forwarded unchanged;
        the official instance point-prompt branch uses its tracker logit-zero mask rule instead.
    Returns official predictor. Requires the editable checkout revision pinned in setup.sh; rejects
    other or unverifiable revisions. No CPU fallback claims and no online sidecar/session are touched.
    Example: build_predictor("/data/vggt/weights/sam3.1/sam3.1_multiplex.pt").
    """
    checkpoint = Path(checkpoint_path).resolve()
    if not checkpoint.is_file():
        raise ValueError("an existing local SAM3.1 checkpoint is required")
    import sam3
    root = Path(sam3.__file__).resolve().parent.parent
    identity = subprocess.run(["git", "-C", str(root), "rev-parse", "HEAD"], capture_output=True, text=True, check=True)
    if identity.stdout.strip() != SAM3_REVISION:
        raise ValueError("installed SAM3 checkout does not match setup.sh pinned revision")
    changed = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--untracked-files=no"],
                             capture_output=True, text=True, check=True)
    if changed.stdout.strip():
        raise ValueError("installed SAM3 tracked source differs from pinned revision")
    from sam3.model_builder import build_sam3_multiplex_video_predictor
    return build_sam3_multiplex_video_predictor(
        checkpoint_path=str(checkpoint), max_num_objects=max_num_objects, multiplex_count=16,
        use_fa3=use_fa3, compile=False, warm_up=False,
        default_output_prob_thresh=output_prob_threshold, async_loading_frames=False)


def _array(value):
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value)


def _adapt_session_init(predictor):
    """Bridge the pinned base-predictor/multiplex init signature mismatch on this instance only.

    ``predictor`` is the privately owned offline predictor (no global class or installed source edit).
    The pinned base always supplies ``offload_state_to_cpu=False``, although its multiplex model has
    no such parameter. When that parameter is absent, accept only the unsupported *false* default and
    omit it from the original bound method call. True is rejected, not silently downgraded. All other
    arguments and exceptions pass unchanged, and initialization is never retried. Return a JSON-safe
    diagnostic; already compatible/injected predictors need no change. Session registration and close
    remain wholly owned by the official base predictor.
    """
    model = getattr(predictor, "model", None)
    initialize = getattr(model, "init_state", None)
    if initialize is None:
        return {"status": "not_required", "reason": "no_model_init_state"}
    if getattr(initialize, "_ni_rua_false_state_offload_compat", False):
        return {"status": "applied", "adapter": "omit_unsupported_false_state_offload"}
    parameters = inspect.signature(initialize).parameters
    if "offload_state_to_cpu" in parameters or any(p.kind == inspect.Parameter.VAR_KEYWORD for p in parameters.values()):
        return {"status": "not_required", "reason": "init_accepts_state_offload"}

    @functools.wraps(initialize)
    def compatible_initialize(*args, offload_state_to_cpu=False, **kwargs):
        if offload_state_to_cpu is not False:
            raise ValueError("this SAM3.1 multiplex init does not support offload_state_to_cpu=True")
        return initialize(*args, **kwargs)

    compatible_initialize._ni_rua_false_state_offload_compat = True
    model.init_state = compatible_initialize
    return {"status": "applied", "adapter": "omit_unsupported_false_state_offload"}


def _adapt_uncached_refinements(predictor):
    """Preserve real point-tracker masks when the pinned multiplex merger has no VG cache entry.

    ``predictor`` is this offline instance. The pinned ``_build_sam2_output`` returns {} immediately
    for an uncached frame, ignoring newly computed ``refined_obj_id_to_mask``. Pure point sessions
    therefore discard the prompt mask and each first-visited video frame. Only that missing-cache,
    supplied-refinements branch is corrected: copy actual object/mask pairs with the same non-None
    assertion as the existing-cache branch. Never construct masks or insert a cache entry. Official
    callers still own cache writes, suppression and empty-mask filtering. Existing cache/no-refinement
    calls and their exceptions go through the original method unchanged. Return recorded compatibility.
    """
    model = getattr(predictor, "model", None)
    merge = getattr(model, "_build_sam2_output", None)
    if merge is None:
        return {"status": "not_required", "reason": "no_multiplex_mask_merger"}
    if getattr(merge, "_ni_rua_uncached_refinement_compat", False):
        return {"status": "applied", "adapter": "preserve_refined_masks_without_existing_vg_cache"}

    @functools.wraps(merge)
    def compatible_merge(inference_state, frame_idx, refined_obj_id_to_mask=None):
        if frame_idx in inference_state["cached_frame_outputs"] or refined_obj_id_to_mask is None:
            return merge(inference_state, frame_idx, refined_obj_id_to_mask)
        obj_id_to_mask = {}
        for obj_id, refined_mask in refined_obj_id_to_mask.items():
            assert refined_mask is not None, f"Refined mask data must be provided for obj_id {obj_id}"
            obj_id_to_mask[obj_id] = refined_mask
        return obj_id_to_mask

    compatible_merge._ni_rua_uncached_refinement_compat = True
    model._build_sam2_output = compatible_merge
    return {"status": "applied", "adapter": "preserve_refined_masks_without_existing_vg_cache"}


def run_video_tracking(frames_dir, prompts, output_dir, *, predictor=None, checkpoint_path=None,
                       max_num_objects=DEFAULTS["max_num_objects"], use_fa3=DEFAULTS["use_fa3"],
                       output_prob_threshold=DEFAULTS["output_prob_threshold"]):
    """Track masks in an immutable offline clip, closing the official session on all terminal paths.

    frames_dir: existing original image directory; numeric filenames 0..N-1, equal width/height px.
        No resampling or timestamps are invented. Join capture timestamps by frame_index/source_file.
    prompts: list of {frame_index, object_id, points_px:[[u,v],...], point_labels:[0|1,...]} for instance
        masks, OR {frame_index,text} for a concept (possibly many model-assigned object IDs). Pixels
        are original-image coordinates with u right/v down; index zero-based. No implicit prompt.
    output_dir: new directory, must not already exist; writes tracking.json and binary PNG masks.
        Input images are read only; source-byte bindings are operator audit metadata, never actor IDs.
    predictor: optional injected official-compatible handle_request/handle_stream_request instance;
        default None explicitly calls build_predictor(checkpoint_path). Injection is for CPU tests.
    checkpoint_path: local multiplex weights path required when predictor is None; no default download.
    max_num_objects: positive integer mask capacity, default 16; not a keypoint count.
    use_fa3: bool, default False; passed to the explicit model builder, no effect on injected predictor.
    output_prob_threshold: dimensionless threshold [0,1], default 0.5, sent on prompt/propagation;
        official instance point prompts use the tracker's logit-zero rule, not this scalar.

    Returns JSON-safe report; partial model or cleanup failure persists partial outputs/status and
    missing frames remain unknown. model_score is upstream out_probs (not calibrated visibility or
    contact confidence); mask_visibility describes mask output only. material_keypoints stays None.
    object_coverage counts each final-frame mask visibility per ID, with first/last nonempty-mask
    frame indices (or None). Completed means processing ended, never that an object was tracked well.
    Example: run_video_tracking("frames", [{"frame_index":0,"object_id":1,"points_px":[[20,30]],
    "point_labels":[1]}], "new-tracking", checkpoint_path="sam3.1_multiplex.pt").
    """
    directory, inventory, size = _frames(frames_dir)
    requests = _prompts(prompts, len(inventory), size)
    if isinstance(max_num_objects, bool) or not isinstance(max_num_objects, int) or max_num_objects <= 0:
        raise ValueError("max_num_objects must be a positive integer")
    if not isinstance(use_fa3, bool) or not np.isfinite(output_prob_threshold) or not 0 <= output_prob_threshold <= 1:
        raise ValueError("use_fa3 must be bool; output_prob_threshold must be finite in [0,1]")
    if len({r["obj_id"] for r in requests if "obj_id" in r}) > max_num_objects:
        raise ValueError("explicit object identities exceed max_num_objects")
    out = Path(output_dir).resolve()
    if out.exists():
        raise FileExistsError("output_dir already exists; refusing overwrite")
    injected = predictor is not None
    if predictor is None:
        if checkpoint_path is None:
            raise ValueError("checkpoint_path required for actual GPU inference")
        predictor = build_predictor(checkpoint_path, max_num_objects=max_num_objects, use_fa3=use_fa3,
                                    output_prob_threshold=output_prob_threshold)
    out.mkdir(parents=True, exist_ok=False)
    (out / "masks").mkdir()
    report = {"schema": SCHEMA, "status": "running", "reason": None,
              "model": "SAM3.1 Object Multiplex", "source_revision": SAM3_REVISION,
              "source_verified": not injected, "predictor_injected": injected,
              "image_size_px": list(size), "prompts": prompts, "effective_settings": {
                  "max_num_objects": max_num_objects, "use_fa3": use_fa3,
                  "output_prob_threshold": output_prob_threshold, "compile": False, "warm_up": False,
                  "propagation_direction": "both", "offload_video_to_cpu": True,
                  "offload_state_to_cpu": False},
              "material_keypoints": None, "contact_evidence": "unknown", "session_closed": None,
              "frames": [{**frame, "status": "unknown", "objects": []} for frame in inventory],
              "frame_updates": []}
    session = None
    known_ids = {r["obj_id"] for r in requests if "obj_id" in r}

    def accept(response, phase):
        index = response.get("frame_index")
        if isinstance(index, bool) or not isinstance(index, (int, np.integer)) or not 0 <= index < len(inventory):
            raise ValueError("predictor returned invalid frame_index")
        outputs = response.get("outputs")
        if outputs is None:
            report["frame_updates"].append({"frame_index": int(index), "phase": phase, "status": "unknown"})
            return
        ids, masks = _array(outputs["out_obj_ids"]), _array(outputs["out_binary_masks"])
        if ids.ndim != 1 or masks.shape != (len(ids), size[1], size[0]):
            raise ValueError("predictor mask/ID shape does not match original frames")
        if not np.issubdtype(ids.dtype, np.integer) or np.any(ids < 0) or len(set(ids.tolist())) != len(ids):
            raise ValueError("predictor IDs must be unique nonnegative integers")
        if masks.dtype != np.bool_:
            raise ValueError("official out_binary_masks must contain boolean masks")
        scores = _array(outputs["out_probs"]) if "out_probs" in outputs else np.full(len(ids), np.nan)
        if scores.shape != (len(ids),):
            raise ValueError("predictor scores do not align with IDs")
        objects = []
        update = len(report["frame_updates"])
        for oid, mask, score in zip(ids.tolist(), masks, scores):
            known_ids.add(oid)
            relative = f"masks/{index:06d}-obj{oid}-update{update}.png"
            Image.fromarray(mask.astype(np.uint8) * 255).save(out / relative)
            area = int(mask.sum())
            objects.append({"object_id": oid, "mask_file": relative, "mask_area_px": area,
                            "mask_sha256": hashlib.sha256((out / relative).read_bytes()).hexdigest(),
                            "mask_visibility": "mask_present" if area else "empty_mask",
                            "model_score": float(score) if np.isfinite(score) else None,
                            "physical_visibility": "unknown"})
        row = {"frame_index": int(index), "phase": phase, "status": "mask_output", "objects": objects}
        report["frame_updates"].append(row)
        report["frames"][index].update(status="mask_output", objects=list(objects))

    try:
        report["session_init_compatibility"] = _adapt_session_init(predictor)
        report["refinement_merge_compatibility"] = _adapt_uncached_refinements(predictor)
        started = predictor.handle_request({"type": "start_session", "resource_path": str(directory),
                                             "offload_video_to_cpu": True, "offload_state_to_cpu": False})
        session = started["session_id"]
        if not isinstance(session, str) or not session:
            raise ValueError("predictor returned invalid session identity")
        for request in requests:
            accept(predictor.handle_request({**request, "session_id": session,
                                              "output_prob_thresh": output_prob_threshold}), "prompt")
        stream = {"type": "propagate_in_video", "session_id": session, "propagation_direction": "both",
                  "start_frame_index": min(r["frame_index"] for r in requests),
                  "max_frame_num_to_track": len(inventory), "output_prob_thresh": output_prob_threshold}
        for response in predictor.handle_stream_request(stream):
            accept(response, "propagation")
        report["status"] = "completed" if all(f["status"] == "mask_output" for f in report["frames"]) else "incomplete"
        if report["status"] == "incomplete":
            report["reason"] = "missing_frame_outputs"
        if any(hashlib.sha256((directory / f["source_file"]).read_bytes()).hexdigest() != f["source_sha256"]
               for f in inventory):
            report.update(status="failed", reason="source_frames_changed_during_inference")
    except KeyboardInterrupt:
        report.update(status="cancelled", reason="operator_interrupt")
    except Exception as error:
        report.update(status="failed", reason=f"{type(error).__name__}: {error}")
    finally:
        if session is not None:
            try:
                closed = predictor.handle_request({"type": "close_session", "session_id": session})
                report["session_closed"] = closed.get("is_success") is True
                if not report["session_closed"]:
                    report.update(status="failed", cleanup_error="session_close_not_confirmed")
            except Exception as error:
                report.update(status="failed", session_closed=False, cleanup_error=f"{type(error).__name__}: {error}")
        for frame in report["frames"]:
            present = {obj["object_id"] for obj in frame["objects"]}
            for oid in sorted(known_ids - present):
                frame["objects"].append({"object_id": oid, "mask_file": None, "mask_area_px": None,
                                         "mask_sha256": None,
                                         "mask_visibility": "not_returned" if frame["status"] == "mask_output" else "unknown",
                                         "model_score": None, "physical_visibility": "unknown"})
        report["object_coverage"] = []
        for oid in sorted(known_ids):
            counts = {visibility: 0 for visibility in ("mask_present", "empty_mask", "not_returned", "unknown")}
            present_frames = []
            for frame in report["frames"]:
                obj = next(obj for obj in frame["objects"] if obj["object_id"] == oid)
                counts[obj["mask_visibility"]] += 1
                if obj["mask_visibility"] == "mask_present":
                    present_frames.append(frame["frame_index"])
            report["object_coverage"].append({"object_id": oid, "total_frames": len(inventory), **counts,
                "first_mask_frame_index": present_frames[0] if present_frames else None,
                "last_mask_frame_index": present_frames[-1] if present_frames else None})
        report["frames_with_any_mask"] = sum(any(obj["mask_visibility"] == "mask_present" for obj in frame["objects"])
                                              for frame in report["frames"])
        (out / "tracking.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main(argv=None):
    """Offline CLI; --prompts is a JSON prompt list, --output a new directory, --checkpoint local weights."""
    parser = argparse.ArgumentParser(description="SAM3.1 offline mask tracking; no material keypoints or contact verdict")
    parser.add_argument("--frames", required=True, help="original equal-size images numbered 0..N-1; PNG supported")
    parser.add_argument("--prompts", required=True, help="JSON list: original-pixel instance point prompts or text prompts")
    parser.add_argument("--output", required=True, help="new output directory for tracking.json and binary PNG masks")
    parser.add_argument("--checkpoint", required=True, help="existing local SAM3.1 multiplex weights; no implicit download")
    parser.add_argument("--max-num-objects", type=int, default=argparse.SUPPRESS, help="positive tracked-mask capacity (default 16)")
    parser.add_argument("--use-fa3", action=argparse.BooleanOptionalAction, default=argparse.SUPPRESS, help="enable FlashAttention 3 if available (default disabled); --no-use-fa3 overrides a true file value")
    parser.add_argument("--output-prob-threshold", type=float, default=argparse.SUPPRESS,
                        help="dimensionless official output threshold in [0,1] (default .5); point branch uses tracker rule")
    parser.add_argument("--config", help="shared offline evidence JSON; explicit CLI settings override the file; omitted uses packaged defaults")
    args = parser.parse_args(argv)
    settings = evidence_settings("sam31_video", args.config,
        {key: getattr(args, key) for key in DEFAULTS if hasattr(args, key)})
    report = run_video_tracking(args.frames, json.loads(Path(args.prompts).read_text()), args.output,
                                checkpoint_path=args.checkpoint, **settings)
    print(json.dumps({"status": report["status"], "reason": report["reason"], "session_closed": report["session_closed"],
                      "total_frames": len(report["frames"]), "frames_with_any_mask": report["frames_with_any_mask"],
                      "object_coverage": report["object_coverage"]}))
    return 0 if report["status"] == "completed" else 2


if __name__ == "__main__":
    raise SystemExit(main())
