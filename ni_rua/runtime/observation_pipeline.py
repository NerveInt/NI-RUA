"""Formal observation publication, enrichment orchestration, and adaptive view delivery."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
from typing import Any

from ..core.action_validation import capability_arms
from ..core.contracts import canonical, digest, new_id, validate_observation
from ..core.enrichment import Enrichment, EnrichmentRequest, ObservationEnricher
from ..core.observation_policy import (
    actor_robot_state,
    allows_depth,
    has_calibration,
    require_actor_observation,
)
from ..core.presentation.actor_payload import lean_observation
from ..core.storage.images import camera_path, encode_image, image_url, share_capture_file
from ..core.tool_output import StructuredText, resolve_detail, validate_display, wire_content
from ..robot.model import roboprobe_pose


def _omitted_image_feedback(policy, observation_id, included_ids, omitted_ids, reason):
    """生成透明的动作后图像筛选回执，供模型按需调用 history_view_observation 取回保存原图或局部 ROI。"""
    return dict(
        policy=policy,
        observation_id=observation_id,
        included_camera_ids=list(included_ids),
        omitted_camera_ids=list(omitted_ids),
        reason=reason,
        retrieval=dict(
            tool="history_view_observation",
            observation_id=observation_id,
            camera_id="choose an omitted camera_id or pass focus_3d; saved image; no native steps",
        ),
        omission_meaning="Raw camera frames remain archived lossless for this observation; omitted from inline reply because structured 3D state is attached.",
    )


def _action_focus_roi(metadata, camera, action, *, patch_px=256, profile=None):
    """当显式指定 image_delivery='focus_roi' 时，将活跃机械臂实测夹持中心反投影为目标相机局部像素裁剪框。"""
    import numpy as np
    from ..core.action_validation import embodiment_or_default
    from ..robot.model.geometry import project as geometry_project

    if not isinstance(metadata, dict) or not isinstance(camera, dict):
        return None
    profile = embodiment_or_default(profile)
    cid = camera.get("camera_id")
    cal_doc = metadata.get("camera_calibration")
    if isinstance(cal_doc, dict):
        cam_cal = (
            (cal_doc.get("cameras") or {}).get(cid)
            if isinstance(cal_doc.get("cameras"), dict) and "K" not in cal_doc
            else cal_doc
        )
    else:
        cam_cal = None
    if not isinstance(cam_cal, dict) or "K" not in cam_cal:
        return None
    T_wc_raw = None
    pose_doc = metadata.get("camera_pose_sample")
    if isinstance(pose_doc, dict):
        cam_pose = (
            (pose_doc.get("cameras") or {}).get(cid)
            if isinstance(pose_doc.get("cameras"), dict) and "T_world_camera" not in pose_doc
            else pose_doc
        )
        if isinstance(cam_pose, dict) and "T_world_camera" in cam_pose:
            T_wc_raw = cam_pose["T_world_camera"]
    if T_wc_raw is None and "T_world_camera" in cam_cal:
        T_wc_raw = cam_cal["T_world_camera"]
    if T_wc_raw is None:
        return None
    arms_state = (metadata.get("robot_state") or {}).get("arms") or {}
    arms_order = list(profile.arms)
    active_arms = []
    if isinstance(action, dict) and isinstance(action.get("stages"), list):
        active_arms = [st.get("arm") for st in action["stages"] if isinstance(st, dict) and st.get("arm") in arms_order]
    active_arms = list(dict.fromkeys(active_arms)) or arms_order
    pad_offset_m = (
        profile.gripper.grasp_reference_offset_m
        if profile.gripper.grasp_reference == "tcp_offset"
        else None
    )
    approach_axis = np.asarray(profile.gripper.approach_axis_local, dtype=np.float64)
    points = []
    for arm in active_arms:
        arm_entry = arms_state.get(arm) or {}
        gp = arm_entry.get("grasp_pose")
        if isinstance(gp, dict) and isinstance(gp.get("position_m"), (list, tuple)) and len(gp["position_m"]) == 3:
            points.append(np.asarray(gp["position_m"], dtype=np.float64))
            continue
        ee = arm_entry.get("ee_pose") or {}
        pos = ee.get("position_m")
        quat_xyzw = ee.get("quaternion_xyzw")
        if isinstance(pos, (list, tuple)) and len(pos) == 3:
            p_arr = np.asarray(pos, dtype=np.float64)
            if pad_offset_m and isinstance(quat_xyzw, (list, tuple)) and len(quat_xyzw) == 4:
                q = np.asarray(quat_xyzw, dtype=np.float64)
                nrm = float(np.linalg.norm(q))
                if np.isfinite(nrm) and nrm > 1e-9:
                    qx, qy, qz, qw = q / nrm
                    rot = np.array(
                        [
                            [1.0 - 2.0 * (qy * qy + qz * qz), 2.0 * (qx * qy - qw * qz), 2.0 * (qx * qz + qw * qy)],
                            [2.0 * (qx * qy + qw * qz), 1.0 - 2.0 * (qx * qx + qz * qz), 2.0 * (qy * qz - qw * qx)],
                            [2.0 * (qx * qz - qw * qy), 2.0 * (qy * qz + qw * qx), 1.0 - 2.0 * (qx * qx + qy * qy)],
                        ],
                        dtype=np.float64,
                    )
                    points.append(p_arr + float(pad_offset_m) * (rot @ approach_axis))
                    continue
            points.append(p_arr)
    if not points:
        return None
    try:
        width, height = int(camera["width"]), int(camera["height"])
        K = np.asarray(cam_cal["K"], dtype=np.float64).reshape(3, 3)
        T_wc = np.asarray(T_wc_raw, dtype=np.float64).reshape(4, 4)
        raw_dist = cam_cal.get("distortion")
        dist_coeffs = (
            np.asarray(raw_dist, dtype=np.float64).reshape(-1)
            if isinstance(raw_dist, (list, tuple)) and len(raw_dist) in (4, 5, 8, 12, 14)
            else np.zeros(5, dtype=np.float64)
        )
        pixels, depth, visible = geometry_project(
            {"T": T_wc, "K": K, "D": dist_coeffs, "width": width, "height": height},
            np.asarray(points, dtype=np.float64),
        )
    except (ValueError, TypeError, KeyError, np.linalg.LinAlgError):
        return None
    uvs = [
        (float(px[0]), float(px[1]))
        for px, z_cam, vis in zip(pixels, depth, visible)
        if bool(vis) and z_cam > 0.05 and np.isfinite(px[0]) and np.isfinite(px[1])
    ]
    if not uvs:
        return None
    side = min(max(128, int(patch_px)), width, height)
    cu = sum(p[0] for p in uvs) / len(uvs)
    cv = sum(p[1] for p in uvs) / len(uvs)
    x = max(0, min(width - side, int(round(cu - side / 2))))
    y = max(0, min(height - side, int(round(cv - side / 2))))
    return [x, y, side, side]


class ObservationPipeline:
    """Owns formal observation archival, enricher execution, and adaptive image projection."""

    def __init__(self, harness: Any, *, enrichers: tuple[ObservationEnricher, ...] | None = None) -> None:
        self._harness = harness
        if enrichers is None:
            from ..perception.external.obs3d import Obs3dEnricher

            enrichers = (Obs3dEnricher(),)
        self.enrichers: tuple[ObservationEnricher, ...] = tuple(enrichers)

    def record_object_table_delta(self, before_observation_id: str, after_observation_id: str, delta: dict) -> None:
        from .actions.action_state_ledger import record_object_table_delta

        record_object_table_delta(self._harness, before_observation_id, after_observation_id, delta)

    def enrich(
        self,
        bundle: dict[str, Any],
        correlation: dict[str, Any] | None = None,
        *,
        reply: str = "observe",
        before_observation_id: str | None = None,
        suppress_overlay: bool = False,
    ) -> Enrichment:
        h = self._harness
        corr = dict(correlation or {})
        audience = "program" if corr.get("program_id") else "actor"
        basis_valid = h.episode_state.is_basis(before_observation_id) if before_observation_id else True
        req = EnrichmentRequest(
            observation=bundle,
            correlation=corr,
            audience=audience,
            reply=reply,
            before_observation_id=before_observation_id,
            suppress_overlay=suppress_overlay,
            basis_valid=basis_valid,
        )
        parts: list[Any] = []
        facts: dict[str, Any] = {}
        for enricher in self.enrichers:
            out = enricher.enrich(h, req)
            parts.extend(out.parts)
            facts.update(out.facts)
            delta_rec = out.facts.get("delta_record")
            if (
                isinstance(delta_rec, dict)
                and isinstance(delta_rec.get("delta"), dict)
                and not callable(getattr(h, "on_object_table_delta", None))
            ):
                self.record_object_table_delta(
                    delta_rec["before_observation_id"],
                    delta_rec["after_observation_id"],
                    delta_rec["delta"],
                )
        return Enrichment(parts=tuple(parts), facts=facts)

    def publish_observation(
        self,
        raw,
        correlation=None,
        last_action_id=None,
        make_current=True,
        *,
        identity=None,
        can_make_current=None,
    ):
        h = self._harness
        observation_id = new_id("obs")
        cameras = []
        require_actor_observation(raw, h.observation_profile)
        state = actor_robot_state(raw["robot_state"], h.observation_profile, arms=h.embodiment.arms)
        calibrated = has_calibration(h.observation_profile)
        calibration_ref = pose_ref = model_ref = None
        if calibrated:
            if not all(raw.get(key) for key in ("camera_calibration", "camera_pose_sample", "robot_model")):
                raise ValueError("calibrated_observation_missing_geometry")
            calibration_ref = h.store.put_artifact(
                canonical(raw["camera_calibration"]).encode(), "camera_calibration", "application/json"
            )
            pose_ref = h.store.put_artifact(
                canonical(raw["camera_pose_sample"]).encode(), "camera_pose_sample", "application/json"
            )
            model_ref = h.store.put_artifact(
                canonical(raw["robot_model"]).encode(), "robot_model", "application/json"
            )
        for camera in raw["cameras"]:
            data = camera_path(camera).read_bytes()
            encoding = camera.get("image_encoding")
            if not (
                encoding
                and encoding.get("format") == "jpeg"
                and encoding.get("quality") == h.image_settings["storage_jpeg_quality"]
                and encoding.get("sha256") == hashlib.sha256(data).hexdigest()
            ):
                data, encoding = encode_image(data, h.image_settings, storage=True)
            ref = h.store.put_artifact(data, "rgb", "image/jpeg")
            if camera.get("image_encoding") == encoding:
                share_capture_file(h.store, camera_path(camera), ref)
            cameras.append(
                dict(
                    camera_id=camera["camera_id"],
                    frame_id=camera.get("frame_id") or new_id("frame"),
                    rgb_ref=ref,
                    width=camera["width"],
                    height=camera["height"],
                    pixel_transform="identity",
                    capture_time=raw.get("captured_at"),
                    clock_domain=raw.get("clock_domain"),
                    sim_step=camera.get("sim_step", raw.get("sim_step")),
                    step_domain=camera.get("step_domain", raw.get("step_domain")),
                    synchronization=camera.get("capture_alignment"),
                    control_dt_s=raw.get("control_dt_s"),
                    image_encoding=encoding,
                )
            )
            if calibrated:
                cameras[-1].update(calibration_ref=calibration_ref, pose_sample_ref=pose_ref)
            if allows_depth(h.observation_profile):
                from ..perception.depth.depth_capture import validate_depth_archive

                depth_data = Path(camera["depth_path"]).read_bytes()
                validate_depth_archive(depth_data, camera)
                depth_ref = h.store.put_artifact(depth_data, "depth", "application/x-npz")
                cameras[-1].update(
                    depth_ref=depth_ref,
                    **{key: camera[key] for key in ("depth_kind", "depth_unit", "depth_clip_m", "depth_shape")},
                )
        state_ref = h.store.put_artifact(canonical(state).encode(), "robot_state", "application/json")
        bundle_identity = dict(identity) if isinstance(identity, dict) else dict(h.identity)
        bundle = dict(
            schema_version="0.1",
            **bundle_identity,
            world_kind="observed",
            observation_id=observation_id,
            sim_step=raw.get("sim_step"),
            step_domain=raw.get("step_domain"),
            sim_time_s=raw.get("sim_time_s"),
            clock_domain=f'{bundle_identity["env_instance_id"]}/generation-{bundle_identity["env_generation"]}',
            observation_profile=h.observation_profile,
            robot_model_id=(h.capabilities or {}).get("robot_model_id", h.embodiment.robot_model_id),
            calibration_id=digest(raw["camera_calibration"]) if calibrated else None,
            cameras=cameras,
            robot_state_ref=state_ref,
            last_action_id=last_action_id,
            data_origin=raw.get("data_origin", "robodojo_runtime"),
        )
        bundle["action_budget"] = h.action_budget()
        if calibrated:
            bundle["robot_model_ref"] = model_ref
        if raw.get("instruction") is not None:
            bundle["task_instruction"] = raw["instruction"]
        if raw.get("calibration_correction") is not None:
            from ..robot.model.calibration_correction import CalibrationCorrection

            bundle["calibration_correction"] = CalibrationCorrection.from_mapping(raw["calibration_correction"]).to_dict()
        if raw.get("max_sync_skew_ns") is not None:
            bundle["max_sync_skew_ns"] = raw["max_sync_skew_ns"]
        validate_observation(bundle)
        h.store.save("observation", observation_id, bundle)
        if h.scene_artifacts is not h.store:
            for reference in [state_ref, calibration_ref, pose_ref, model_ref] + [
                camera[key] for camera in cameras for key in ("rgb_ref", "depth_ref") if key in camera
            ]:
                if reference:
                    info, content = h.store.resolve_artifact(reference)
                    copied = h.scene_artifacts.put_artifact(
                        content, info["kind"], info["mime_type"], info["access_scope"]
                    )
                    if copied != reference:
                        raise ValueError("scene_artifact_copy_identity_mismatch")
        h.scene.import_observation(bundle)
        with h.lock:
            same_episode = h.identity == bundle_identity
            effective_make_current = bool(
                make_current and same_episode and (can_make_current is None or can_make_current())
            )
            if same_episode:
                h.episode_state.observation_published(observation_id, make_current=effective_make_current)
            if effective_make_current:
                h.latest = bundle
                if h.spatial_stream is not None:
                    h.spatial_stream.set_decision_anchor(observation_id)
                if h.home_check is not None and not h.active_action:
                    h.home_check.update_current(bundle)
        h.store.event(
            "observation.available",
            {"observation_id": observation_id, "historical": not effective_make_current},
            {**(correlation or {}), "observation_id": observation_id},
            sim_step=bundle["sim_step"],
            step_domain=bundle["step_domain"],
            sim_time_s=bundle["sim_time_s"],
        )
        return bundle

    def observation_content(
        self,
        bundle,
        correlation,
        camera_id=None,
        roi=None,
        scale=1,
        numeric_only=False,
        *,
        feedback_action=None,
        feedback_receipt=None,
        detail=None,
        feedback_context=None,
        full_view_reason=None,
        metadata_delivery=True,
        include_images=True,
        structured=False,
    ):
        h = self._harness
        compact_delivery = (
            resolve_detail(
                detail,
                program=bool(correlation.get("program_id")),
                default="full" if h.actor_payload_profile == "full_v1" else "summary",
            )
            == "summary"
        )
        if (roi is not None or scale != 1) and camera_id is None:
            raise ValueError("image_view_requires_camera_id")
        if type(scale) is not int or not 1 <= scale <= 4:
            raise ValueError("invalid_image_scale")
        if camera_id is not None and camera_id not in [camera["camera_id"] for camera in bundle["cameras"]]:
            raise ValueError("unknown_camera")
        require_actor_observation(bundle, h.observation_profile)
        _, state_bytes = h.store.resolve_artifact(bundle["robot_state_ref"], allowed_scopes=("actor",))
        metadata = dict(
            bundle=bundle,
            robot_state=actor_robot_state(json.loads(state_bytes), h.observation_profile, arms=h.embodiment.arms),
        )
        if has_calibration(h.observation_profile) and not numeric_only:
            metadata["camera_calibration"] = json.loads(
                h.store.resolve_artifact(bundle["cameras"][0]["calibration_ref"])[1]
            )
            metadata["camera_pose_sample"] = json.loads(
                h.store.resolve_artifact(bundle["cameras"][0]["pose_sample_ref"])[1]
            )
        grasp_angles = "grasp_angles" in h.embodiment.pose_inputs
        if "jaw_grasp_centre" in h.embodiment.pose_inputs:
            from ..robot.model.geometry import measured_jaw_grasp

            for arm, state in metadata["robot_state"].get("arms", {}).items():
                measured = measured_jaw_grasp(h.embodiment, arm, state)
                if measured is not None:
                    state["jaw_grasp"] = measured
        for state in metadata["robot_state"].get("arms", {}).values():
            pose = state.get("ee_pose", {})
            if (
                not grasp_angles
                or pose.get("tcp_config_id") != h.embodiment.tcp_config_id
                or pose.get("source_kind") != "measurement"
            ):
                continue
            x, y, z, w = pose["quaternion_xyzw"]
            if h.embodiment.robot_model_id == "arx-x5":
                values = roboprobe_pose.pose_to_values([*pose["position_m"], w, x, y, z])
                source_kind = "derived_from_measured_link6"
                contact_meaning = "center_between_fingers_at_distal_15mm_pad_center"
            else:
                from ..robot.embodiments.libero_franka_panda import pose_convention as panda_pose

                values = panda_pose.pose_to_values([*pose["position_m"], w, x, y, z])
                source_kind = f"derived_from_measured_{h.embodiment.tcp_link}"
                contact_meaning = "center_between_fingers_at_pad_center"
            state["grasp_pose"] = dict(
                position_m=[values[key] for key in ("x", "y", "z")],
                angles_deg=[values[key] for key in ("pitch_deg", "roll_deg", "yaw_deg")],
                frame_id=pose["frame_id"],
                source_kind=source_kind,
                convention="robot_move_grasp",
                contact_region="pad_center",
                contact_depth_m=0.0,
                contact_point_meaning=contact_meaning,
            )
        content = [StructuredText(metadata, projection="observation")]
        delivered = []
        resolved_images = {}
        selection = None
        epoch = None
        view_snapshot: dict[str, Any] = {}
        is_model = not bool(correlation.get("program_id"))
        if is_model:
            epoch = h.decision_context.actor_delivery_epoch(correlation)
            if epoch is not None:
                _, view_snapshot = h.decision_context.sync_view_epoch(epoch)
        auto_roi = None
        action_policy = None
        if (
            is_model
            and feedback_action is None
            and correlation.get("actor_log_delivery")
            and bundle.get("last_action_id")
            and camera_id is None
            and roi is None
            and scale == 1
        ):
            try:
                feedback_action = h.store.read("action_request", bundle["last_action_id"])
                feedback_receipt = h.store.read("action_receipt", bundle["last_action_id"])
            except KeyError:
                feedback_action = feedback_receipt = None
        if (
            is_model
            and feedback_action is not None
            and not numeric_only
            and include_images
            and camera_id is None
            and roi is None
            and scale == 1
        ):
            from ..core.presentation.feedback_delivery import is_clean_completed_receipt, resolve_action_image_policy

            action_policy = resolve_action_image_policy(h, feedback_action)
            all_camera_ids = [camera["camera_id"] for camera in bundle["cameras"]]
            supports_action_obs3d = isinstance(feedback_action, dict) and (
                feedback_action.get("type") == "BatchAction" or "hold_native_steps" in feedback_action
            )
            if action_policy == "none":
                if supports_action_obs3d:
                    from ..perception.external.obs3d import prefetch_action_attachments

                    prefetch_action_attachments(
                        h,
                        bundle,
                        correlation,
                        before_observation_id=feedback_action.get("based_on_observation_id"),
                        suppress_overlay=True,
                    )
                selection = dict(
                    policy="none", selected_camera_ids=[], skipped_camera_ids=all_camera_ids, reason="explicit_none"
                )
                metadata["image_feedback"] = _omitted_image_feedback(
                    "none", bundle["observation_id"], [], all_camera_ids, "explicit_none"
                )
            elif action_policy == "head_only":
                overview = h.embodiment.overview_camera or "cam_head"
                selected_ids = [overview] if overview in all_camera_ids else all_camera_ids[:1]
                skipped_ids = [cid for cid in all_camera_ids if cid not in selected_ids]
                selection = dict(
                    policy="head_only",
                    selected_camera_ids=selected_ids,
                    skipped_camera_ids=skipped_ids,
                    reason="explicit_head_only",
                )
                if skipped_ids:
                    metadata["image_feedback"] = _omitted_image_feedback(
                        "head_only", bundle["observation_id"], selected_ids, skipped_ids, "explicit_head_only"
                    )
            elif action_policy == "focus_roi":
                overview = h.embodiment.overview_camera or "cam_head"
                selected_ids = [overview] if overview in all_camera_ids else all_camera_ids[:1]
                skipped_ids = [cid for cid in all_camera_ids if cid not in selected_ids]
                head_cam = next((c for c in bundle["cameras"] if c["camera_id"] == selected_ids[0]), None)
                auto_roi = (
                    _action_focus_roi(metadata, head_cam, feedback_action, profile=h.embodiment) if head_cam else None
                )
                selection = dict(
                    policy="focus_roi",
                    selected_camera_ids=selected_ids,
                    skipped_camera_ids=skipped_ids,
                    reason="explicit_focus_roi",
                )
                if skipped_ids:
                    metadata["image_feedback"] = _omitted_image_feedback(
                        "focus_roi", bundle["observation_id"], selected_ids, skipped_ids, "explicit_focus_roi"
                    )
            elif action_policy == "on_anomaly":
                from .actions.rollout_tracker import has_process_slip

                slip_anomaly = has_process_slip(feedback_receipt)
                clean_receipt = is_clean_completed_receipt(feedback_receipt) and not full_view_reason
                obs3d_healthy = False
                if clean_receipt and supports_action_obs3d:
                    from ..perception.external.obs3d import prefetch_action_attachments

                    obs3d_healthy = prefetch_action_attachments(
                        h,
                        bundle,
                        correlation,
                        before_observation_id=feedback_action.get("based_on_observation_id"),
                        suppress_overlay=True,
                    )
                if clean_receipt and slip_anomaly:
                    overview = h.embodiment.overview_camera or "cam_head"
                    selected_ids = [overview] if overview in all_camera_ids else all_camera_ids[:1]
                    skipped_ids = [cid for cid in all_camera_ids if cid not in selected_ids]
                    selection = dict(
                        policy="on_anomaly",
                        selected_camera_ids=selected_ids,
                        skipped_camera_ids=skipped_ids,
                        reason="slip_detected",
                    )
                    if skipped_ids:
                        metadata["image_feedback"] = _omitted_image_feedback(
                            "on_anomaly", bundle["observation_id"], selected_ids, skipped_ids, "slip_detected"
                        )
                elif clean_receipt and obs3d_healthy:
                    selection = dict(
                        policy="on_anomaly",
                        selected_camera_ids=[],
                        skipped_camera_ids=all_camera_ids,
                        reason="completed_with_valid_obs3d",
                    )
                    metadata["image_feedback"] = _omitted_image_feedback(
                        "on_anomaly", bundle["observation_id"], [], all_camera_ids, "completed_with_valid_obs3d"
                    )
                elif h.image_settings.get("feedback_policy", "all") == "adaptive":
                    action_policy = "adaptive"
        elif not is_model and camera_id is None and roi is None and scale == 1 and not numeric_only:
            from ..perception.external.obs3d import (
                flags as obs3d_flags,
                prefetch_action_attachments,
                prefetch_observe_attachments,
            )

            if any(obs3d_flags(h)):
                if feedback_action is None:
                    is_live_observe = bool(
                        h.latest
                        and bundle.get("observation_id") == h.latest.get("observation_id")
                        and not bundle.get("last_action_id")
                    )
                    shown_3d = prefetch_observe_attachments(
                        h,
                        bundle,
                        correlation,
                        cache_pending=is_live_observe,
                    )
                    if isinstance(shown_3d, dict):
                        for key in ("object_table", "scene3d", "object_state_delta"):
                            if isinstance(shown_3d.get(key), dict):
                                metadata[key] = deepcopy(shown_3d[key])
                elif isinstance(feedback_action, dict) and (
                    feedback_action.get("type") == "BatchAction" or "hold_native_steps" in feedback_action
                ):
                    prefetch_action_attachments(
                        h,
                        bundle,
                        correlation,
                        before_observation_id=feedback_action.get("based_on_observation_id"),
                        suppress_overlay=True,
                    )
        use_adaptive = action_policy == "adaptive" or (
            feedback_action is None and h.image_settings.get("feedback_policy", "all") == "adaptive"
        )
        if (
            use_adaptive
            and is_model
            and not numeric_only
            and include_images
            and camera_id is None
            and roi is None
            and scale == 1
        ):
            if epoch is None:
                all_camera_ids = [camera["camera_id"] for camera in bundle["cameras"]]
                reason = "context_epoch_volatile" if h.decision_context.context_epoch_volatile else "isolated_context"
                selection = dict(
                    selected_camera_ids=all_camera_ids,
                    skipped_camera_ids=[],
                    force_full_reason=reason,
                )
                metadata["image_feedback"] = dict(
                    policy="adaptive",
                    observation_id=bundle["observation_id"],
                    included_camera_ids=all_camera_ids,
                    omitted_camera_ids=[],
                    retrieval=dict(
                        tool="history_view_observation",
                        observation_id=bundle["observation_id"],
                        camera_id="choose an omitted camera ID; saved image; no native steps",
                    ),
                    full_view_reason=reason,
                    omission_meaning="Original view is below all change thresholds versus its last actual model delivery; not proof of unchanged task state. Omitted views remain retrievable.",
                    omitted_views=[],
                )
            else:
                from ..perception.view_selection import select_observation_views

                for camera in bundle["cameras"]:
                    resolved_images[camera["camera_id"]] = h.store.resolve_image(
                        camera["rgb_ref"], allowed_scopes=("actor",)
                    )
                active = None
                action = feedback_action or {}
                if action.get("type") == "ArmQueueAction":
                    active = [
                        arm
                        for arm, stages in action.get("arm_queues", {}).items()
                        if any(stage.get("kind") in ("move", "gripper") for stage in stages)
                    ]
                elif action.get("path") and not action.get("hold_native_steps"):
                    active = sorted(
                        {
                            arm
                            for point in action["path"].get("waypoints", [])
                            for arm in capability_arms(h.capabilities, h.embodiment)
                            if arm in point
                        }
                        | {
                            event["arm"]
                            for event in action.get("gripper_events", [])
                            if event.get("arm") in capability_arms(h.capabilities, h.embodiment)
                        }
                    )
                receipt = feedback_receipt or {}
                from ..core.presentation.feedback_delivery import execution_exception

                if not full_view_reason and execution_exception(receipt):
                    full_view_reason = "execution_exception"
                calibration_valid = {}
                for camera in bundle["cameras"]:
                    try:
                        h.geometry.camera(h.identity, bundle["observation_id"], camera["camera_id"])
                        calibration_valid[camera["camera_id"]] = True
                    except Exception:
                        calibration_valid[camera["camera_id"]] = False
                try:
                    selection = select_observation_views(
                        bundle,
                        {key: value[1] for key, value in resolved_images.items()},
                        view_snapshot,
                        active_arms=active,
                        context=feedback_context or ("post_action" if feedback_action is not None else "explicit_observe"),
                        receipt_status=receipt.get("status"),
                        receipt_reason=receipt.get("reason"),
                        force_full=bool(full_view_reason or receipt.get("execution_uncertain")),
                        full_reason=full_view_reason or ("execution_uncertain" if receipt.get("execution_uncertain") else None),
                        calibration_valid=calibration_valid,
                        profile=h.embodiment,
                    )
                except Exception as error:
                    selection = dict(
                        selected_camera_ids=[camera["camera_id"] for camera in bundle["cameras"]],
                        skipped_camera_ids=[],
                        fallback_reason="selection_failed:" + type(error).__name__,
                    )
                metadata["image_feedback"] = dict(
                    policy="adaptive",
                    observation_id=bundle["observation_id"],
                    included_camera_ids=selection["selected_camera_ids"],
                    omitted_camera_ids=selection["skipped_camera_ids"],
                    retrieval=dict(
                        tool="history_view_observation",
                        observation_id=bundle["observation_id"],
                        camera_id="choose an omitted camera ID; saved image; no native steps",
                    ),
                    full_view_reason=selection.get("force_full_reason", selection.get("fallback_reason")),
                    omission_meaning="Original view is below all change thresholds versus its last actual model delivery; not proof of unchanged task state. Omitted views remain retrievable.",
                )
                metadata["image_feedback"]["omitted_views"] = [
                    {
                        key: decision[key]
                        for key in ("camera_id", "reason", "baseline_observation_id", "retrieval")
                        if key in decision
                    }
                    for decision in selection.get("decisions", [])
                    if not decision["include"]
                ]
        baselines, failed_baseline_cameras, baseline_errors = {}, [], []
        for camera in bundle["cameras"]:
            if numeric_only or not include_images:
                continue
            if camera_id is not None and camera["camera_id"] != camera_id:
                continue
            if selection is not None and camera["camera_id"] not in selection["selected_camera_ids"]:
                continue
            artifact, data = (
                resolved_images[camera["camera_id"]]
                if camera["camera_id"] in resolved_images
                else h.store.resolve_image(camera["rgb_ref"], allowed_scopes=("actor",))
            )
            effective_roi = (
                roi
                if roi is not None
                else (
                    auto_roi
                    if selection is not None and camera["camera_id"] in selection["selected_camera_ids"]
                    else None
                )
            )
            if (
                is_model
                and epoch is not None
                and effective_roi is None
                and scale == 1
                and h.image_settings.get("feedback_policy", "all") == "adaptive"
            ):
                from ..perception.view_selection import baseline_record

                try:
                    baselines[camera["camera_id"]] = baseline_record(bundle, camera["camera_id"], data)
                except Exception as error:
                    failed_baseline_cameras.append(camera["camera_id"])
                    baseline_errors.append(dict(camera_id=camera["camera_id"], reason=type(error).__name__))
            reference = camera["rgb_ref"]
            delivery = dict(
                camera_id=camera["camera_id"], rgb_ref=reference, sha256=artifact["sha256"], preprocessing="identity"
            )
            if artifact.get("reconstruction"):
                delivery["reconstruction"] = artifact["reconstruction"]
                metadata["bundle"] = h.store.observation_view(bundle)
            if effective_roi is not None or scale != 1:
                from PIL import Image

                bounds = effective_roi if effective_roi is not None else [0, 0, camera["width"], camera["height"]]
                if not isinstance(bounds, list) or len(bounds) != 4 or any(type(value) is not int for value in bounds):
                    raise ValueError("invalid_image_roi")
                x, y, width, height = bounds
                if (
                    x < 0
                    or y < 0
                    or width <= 0
                    or height <= 0
                    or x + width > camera["width"]
                    or y + height > camera["height"]
                ):
                    raise ValueError("image_roi_out_of_bounds")
                with Image.open(io.BytesIO(data)) as original:
                    cropped_image = original.crop((x, y, x + width, y + height)).resize(
                        (width * scale, height * scale), Image.Resampling.LANCZOS
                    )
                    data, _ = encode_image(cropped_image, h.image_settings, storage=True)
                reference = h.store.put_artifact(data, "rgb_crop", "image/jpeg")
                cropped, _ = h.store.artifact(reference)
                delivery.update(
                    rgb_ref=reference,
                    original_rgb_ref=camera["rgb_ref"],
                    sha256=cropped["sha256"],
                    preprocessing="crop_resize_lanczos",
                    source_roi=bounds,
                    display_scale=scale,
                    width=width * scale,
                    height=height * scale,
                    pixel_transform=[[scale, 0, scale * (0.5 - x) - 0.5], [0, scale, scale * (0.5 - y) - 0.5], [0, 0, 1]],
                )
                metadata["image_view"] = delivery
            _img_url = getattr(sys.modules.get("ni_rua.runtime.harness"), "image_url", image_url)
            url, encoding = _img_url(data, h.image_settings)
            delivery.update(source_sha256=delivery["sha256"], sha256=encoding["sha256"], encoding=encoding)
            content.append(dict(type="inputImage", imageUrl=url))
            delivered.append(delivery)
        if numeric_only:
            metadata = dict(
                observation_id=bundle["observation_id"],
                robot_state=metadata["robot_state"],
                action_budget=bundle.get("action_budget"),
                images="saved; use history_view_observation when needed",
            )
        elif compact_delivery and h.actor_payload_profile == "compact_v1":
            from ..core.presentation.actor_payload import compact_observation

            metadata = compact_observation(metadata)
        elif compact_delivery:
            metadata = lean_observation(metadata)
        from ..core.presentation.observation_output import observation_metadata

        compact_geometry = compact_delivery and h.actor_payload_profile not in ("full_v1", "lean_v1")
        if compact_geometry and "camera_calibration" in metadata:
            metadata["camera_calibration"]["full_ref"] = bundle["cameras"][0].get("calibration_ref")
            metadata["camera_pose_sample"]["full_ref"] = bundle["cameras"][0].get("pose_sample_ref")
        metadata = observation_metadata(
            metadata,
            current_observation_id=h.latest["observation_id"] if h.latest else None,
            current_action_budget=h.action_budget(),
            compact_geometry=compact_geometry,
        )
        inline_metadata = metadata
        content[0] = StructuredText(inline_metadata, projection="observation")
        projection = None
        if (
            compact_delivery
            and is_model
            and metadata_delivery
            and epoch is not None
        ):
            try:
                live_epoch = h.decision_context.live_context_epoch()
                projection = h.decision_context.metadata_delivery.prepare(
                    metadata,
                    epoch=epoch,
                    live_epoch=live_epoch,
                )
                validate_display(projection.metadata)
                content[0] = StructuredText(projection.metadata, projection="observation")
            except Exception:
                h.decision_context.reset_metadata_delivery()
                projection = None
                content[0] = StructuredText(inline_metadata, projection="observation")
        omitted_bytes_error = None
        skipped_camera_ids = (selection or {}).get("skipped_camera_ids", [])
        if skipped_camera_ids and resolved_images:
            try:
                _img_url = getattr(sys.modules.get("ni_rua.runtime.harness"), "image_url", image_url)
                omitted_transport_bytes = sum(
                    _img_url(resolved_images[key][1], h.image_settings)[1]["byte_size"] for key in skipped_camera_ids
                )
            except Exception as error:
                omitted_transport_bytes, omitted_bytes_error = None, type(error).__name__
        else:
            omitted_transport_bytes = 0 if not skipped_camera_ids else None
        metrics = dict(
            image_count=len(delivered),
            transport_bytes=sum(item["encoding"]["byte_size"] for item in delivered),
            omitted_image_count=len(skipped_camera_ids),
            omitted_transport_bytes=omitted_transport_bytes,
        )
        if baseline_errors:
            metrics["baseline_errors"] = baseline_errors
        if omitted_bytes_error:
            metrics["omitted_transport_bytes_unavailable"] = omitted_bytes_error
        delivery_channel = (
            "program_fd"
            if correlation.get("program_id")
            else "startup_turn"
            if correlation.get("source") == "startup_grounding"
            else h.decision_context.delivery_channel
        )
        h.store.save(
            "delivery",
            new_id("delivery"),
            dict(
                observation_id=bundle["observation_id"],
                correlation=correlation,
                image_selection=selection,
                image_metrics=metrics,
                images=delivered,
                delivered_to_actor=not bool(correlation.get("program_id")),
                audience="actor_program" if correlation.get("program_id") else "astra_model",
                delivery_channel=delivery_channel,
            ),
        )
        h.store.event(
            "observation.delivered",
            {
                "images": delivered,
                "image_selection": selection,
                "image_metrics": metrics,
                "audience": "actor_program" if correlation.get("program_id") else "astra_model",
                "delivered_to_model": is_model,
            },
            {**correlation, "observation_id": bundle["observation_id"]},
        )
        if epoch is not None and (baselines or failed_baseline_cameras):
            h.decision_context.commit_delivered_views(epoch, baselines, failed_baseline_cameras)
        if projection is not None:
            try:
                now_epoch = h.decision_context.live_context_epoch()
                committed = (
                    not h.decision_context.context_epoch_volatile
                    and now_epoch == epoch
                    and h.decision_context.metadata_delivery.commit(projection)
                )
            except Exception:
                h.decision_context.reset_metadata_delivery()
                committed = False
            if not committed:
                content[0] = StructuredText(inline_metadata, projection="observation")
        return content if structured else wire_content(content)
