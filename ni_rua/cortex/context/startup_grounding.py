"""首轮内联场景接地模块（``features.startup_inline_grounding``）。

在网关完成 ``harness.start()`` 采集初始观测 ``obs_1`` 且 ``session.start()`` 建立真实模型
上下文 epoch 后，将脱敏后的结构化观测状态（含实测位姿/夹爪、标定、动作预算）与可用时的
``obs3d`` 结构化感知（``object_table`` + ``grasp_hints`` + ``scene3d`` 句柄）直接内联至首轮
Prompt，并在当前会话与模型支持多模态回合输入时附带概览图或全相机图，省去首轮盲调
``robot_observe()`` 的空跑回合。
"""
from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from ...core.tool_output import StructuredText, ToolResponse
from ...perception.external.obs3d import attachments as obs3d_attachments


VERSION = "startup_inline_grounding_v1"
SOURCE = "startup_grounding"


@dataclass(frozen=True)
class StartupGroundingResult:
    """首轮内联接地的组装产物。"""

    section_text: str
    image_items: tuple[dict[str, Any], ...]
    record: dict[str, Any]


def select_overview_camera(
    bundle: dict[str, Any],
    profile: Any = None,
    *,
    embodiment: Any = None,
) -> str | None:
    """按构型优先选出首轮概览相机（profile/embodiment.overview_camera -> cam_head/head/fixed/agentview -> 首台相机）。"""
    cameras = bundle.get("cameras") or ()
    camera_ids = [cam["camera_id"] for cam in cameras if isinstance(cam, dict) and cam.get("camera_id")]
    if not camera_ids:
        return None
    effective = embodiment if embodiment is not None else profile
    overview = getattr(effective, "overview_camera", None)
    if isinstance(overview, str) and overview and overview in camera_ids:
        return overview
    for preferred in ("cam_head", "head", "fixed", "agentview"):
        if preferred in camera_ids:
            return preferred
    return camera_ids[0]


def _wrist_camera_ids(bundle: dict[str, Any], embodiment: Any, overview_camera_id: str | None) -> tuple[str, ...]:
    """按构型与实际观测相机列表提取腕部相机 ID（覆盖 ARX-X5 与 SO-101）。"""
    cameras = bundle.get("cameras") or ()
    available = [cam["camera_id"] for cam in cameras if isinstance(cam, dict) and cam.get("camera_id")]
    declared = getattr(embodiment, "wrist_cameras", None)
    if isinstance(declared, dict) or hasattr(declared, "values"):
        candidates = list(dict.fromkeys(cid for cid in declared.values() if isinstance(cid, str) and cid))
        if available:
            matched = [cid for cid in candidates if cid in available]
            if matched:
                return tuple(matched)
        elif candidates:
            return tuple(candidates)
    return tuple(cid for cid in available if cid != overview_camera_id and "wrist" in cid)


def build_startup_grounding(
    harness: Any,
    bundle: dict[str, Any] | None = None,
    *,
    correlation: dict[str, Any] | None = None,
    include_images: bool = True,
) -> StartupGroundingResult:
    """组装首轮内联结构化观测状态、3D 感知表与可用概览/多相机图像，并同步内部状态基线。"""
    raw_bundle = bundle if bundle is not None else harness.latest
    if raw_bundle is None:
        raise ValueError("startup_grounding_observation_missing")

    obs_bundle = {**raw_bundle, "action_budget": harness.action_budget()}
    call_correlation = dict(correlation or {"tool_call_id": "startup-grounding", "source": SOURCE})
    if getattr(harness, "thread_id", None):
        call_correlation.setdefault("thread_id", harness.thread_id)

    embodiment = getattr(harness, "embodiment", None)
    camera_id = select_overview_camera(obs_bundle, embodiment=embodiment)
    wrist_cameras = _wrist_camera_ids(obs_bundle, embodiment, camera_id)
    features_cfg = getattr(getattr(harness, "config", None), "features", None)
    all_cameras_mode = getattr(features_cfg, "startup_grounding_views", "overview") == "all_cameras"

    # 先触发 obs3d 感知附件计算，同步预热 _obs3d_memo / _obs3d_history 与 _virtual_spatial_field
    obs3d_parts = obs3d_attachments(harness, obs_bundle, call_correlation, reply="observe")
    obs3d_text_parts = [part for part in obs3d_parts if isinstance(part, StructuredText)]
    obs3d_images = [
        part for part in obs3d_parts
        if isinstance(part, dict) and part.get("type") == "inputImage" and part.get("imageUrl")
    ] if include_images else []

    # 生成观测内容并在当前真实会话 epoch 下提交 MetadataDelivery 基线
    if not include_images:
        obs_parts = harness.observation_content(
            obs_bundle,
            call_correlation,
            camera_id=None if all_cameras_mode else camera_id,
            numeric_only=True,
            include_images=False,
            structured=True,
        )
    elif all_cameras_mode:
        obs_parts = harness.observation_content(
            obs_bundle,
            call_correlation,
            camera_id=None,
            numeric_only=False,
            include_images=True,
            structured=True,
        )
    else:
        obs_parts = harness.observation_content(
            obs_bundle,
            call_correlation,
            camera_id=camera_id,
            numeric_only=(camera_id is None),
            include_images=not bool(obs3d_images),
            structured=True,
        )
    obs_text_part = obs_parts[0]
    obs_images = [
        part for part in obs_parts
        if isinstance(part, dict) and part.get("type") == "inputImage" and part.get("imageUrl")
    ] if include_images else []

    # 统一走 ActorOutput / ActorPresentation 脱敏与短引用映射（obs_1、artifact:ref_N）
    presented = harness.actor_output.present(
        ToolResponse([obs_text_part, *obs3d_text_parts], kind="observation"),
        tool="robot_observe",
    )
    if not presented.get("success", False):
        raise RuntimeError("startup_grounding_presentation_failed")

    content_items = presented["contentItems"]
    obs_json_text = content_items[0]["text"]
    obs3d_json_texts = [item["text"] for item in content_items[1:] if item.get("type") == "inputText"]
    obs_payload = json.loads(obs_json_text)
    obs_alias = obs_payload.get("current_observation_id") or "obs_1"
    overview_name = camera_id or "overview"

    if all_cameras_mode:
        combined_imgs = [*obs_images, *obs3d_images]
        image_items = tuple({"type": "image", "url": img["imageUrl"]} for img in combined_imgs)
        if obs_images and obs3d_images:
            image_kind = "all_cameras_with_obs3d_overlay"
            image_label = f"{len(obs_images)} raw camera views + {overview_name} with 3D object IDs"
        elif obs_images:
            image_kind = "all_cameras_raw_rgb"
            image_label = f"{len(obs_images)} raw camera views"
        elif obs3d_images:
            image_kind = "obs3d_overlay"
            image_label = f"{overview_name} with 3D object IDs"
        else:
            image_kind = "none"
            image_label = "no camera image"
        header = (
            f"Initial scene grounding (inline {obs_alias}; "
            + (f"{len(image_items)} images attached: {image_label}" if image_items else "no camera image attached")
            + "):"
        )
    else:
        # 默认仅附带 1 张概览图：优先使用 obs3d 产出的概览 ID 标注图，缺失时回退为单张原图
        if obs3d_images:
            image_items = ({"type": "image", "url": obs3d_images[0]["imageUrl"]},)
            image_kind = "obs3d_overlay"
            image_label = f"{overview_name} with 3D object IDs"
        elif obs_images:
            image_items = ({"type": "image", "url": obs_images[0]["imageUrl"]},)
            image_kind = "raw_rgb"
            image_label = f"{overview_name} raw RGB"
        else:
            image_items = ()
            image_kind = "none"
            image_label = "no camera image"
        header = (
            f"Initial scene grounding (inline {obs_alias}; "
            + (f"1 overview image attached: {image_label}" if image_items else "no camera image attached")
            + "):"
        )

    if len(wrist_cameras) > 1:
        wrist_feedback = "close-range wrist RGB views"
    elif len(wrist_cameras) == 1:
        wrist_feedback = f"{wrist_cameras[0]} close-range RGB view"
    else:
        wrist_feedback = "configured camera views"
    action_feedback_note = (
        f"Omit image_delivery on physical robot_action calls so post-action {overview_name}, {wrist_feedback}, "
        "and trajectory feedback arrive together in the same turn."
    )

    lines = [header, "Observation state:", obs_json_text]
    if obs3d_json_texts:
        lines.append("3D scene perception (object_table + grasp_hints + scene3d):")
        lines.extend(obs3d_json_texts)
    if not image_items:
        lines.append(
            f"Structured initial observation state for {obs_alias} is inlined above without attached camera images; "
            "call robot_observe if visual inspection is needed, otherwise submit your task_plan and proceed directly. "
            + action_feedback_note
        )
    elif all_cameras_mode:
        lines.append(
            f"All calibrated raw RGB camera views ({overview_name}"
            + (f" + {', '.join(wrist_cameras)}" if wrist_cameras else "")
            + f") and any 3D ID overlay for {obs_alias} "
            "are already attached above; submit your task_plan and proceed directly without calling a redundant full "
            f"robot_observe() or re-reading {obs_alias} via history_view_observation (use focus_3d crops on demand when "
            "sub-centimetre hole/slot/edge detail is needed). "
            + action_feedback_note
        )
    else:
        wrist_clause = (
            "Wrist camera views are saved on this initial observation; "
            "if a target, elevated fixture, or contact region is occluded or needs two-view stereo triangulation, call "
            "robot_observe with "
            + " or ".join(f'camera_id="{cid}"' for cid in wrist_cameras)
            + " (reads the saved view at zero simulation step cost), otherwise "
            if wrist_cameras
            else ""
        )
        lines.append(
            wrist_clause
            + "submit your task_plan and proceed directly without calling a redundant full "
            f"robot_observe() or re-reading {obs_alias} via full history_view_observation "
            f"({overview_name} is already attached above). "
            + action_feedback_note
        )
    section_text = "\n" + "\n".join(lines)

    record = dict(
        version=VERSION,
        enabled=True,
        status="applied",
        source=SOURCE,
        observation_id=obs_bundle["observation_id"],
        actor_observation_alias=obs_alias,
        overview_camera_id=camera_id,
        image_kind=image_kind,
        image_count=len(image_items),
        obs3d_included=bool(obs3d_json_texts),
    )
    harness.store.event(
        "startup_grounding.attached",
        {key: value for key, value in record.items() if key != "version"},
        call_correlation,
        world_kind="internal",
    )
    return StartupGroundingResult(section_text=section_text, image_items=image_items, record=record)
