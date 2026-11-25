"""Tool handlers. Registration binds these functions directly; all receive ToolContext."""
from copy import deepcopy

from ...core.contracts import new_id
from ...core.tool_output import ToolOutput, ToolResponse, resolve_detail
from ...core.storage.images import image_url
from .context import runtime_port


def _enrich_observe(runtime, bundle, correlation):
    pipeline = getattr(runtime, "observation_pipeline", None)
    if pipeline is not None:
        return list(pipeline.enrich(bundle, correlation, reply="observe").parts)
    from ...perception.external.obs3d import attachments as obs3d_attachments
    return obs3d_attachments(runtime, bundle, correlation)


def robot_information(context, args, correlation):
    runtime = runtime_port(context)
    if set(args) - {"section", "robot_id", "resource_id"}:
        raise ValueError("invalid_robot_information_arguments")
    section = args.get("section", "default")
    if not isinstance(section, str) or not section:
        raise ValueError("invalid_robot_information_section")
    robot_id = args.get("robot_id")
    if robot_id is not None and (not isinstance(robot_id, str) or not robot_id):
        raise ValueError("invalid_robot_information_robot_id")
    if "resource_id" in args and section != "resource":
        raise ValueError("robot_resource_id_requires_resource_section")
    if section in {"resources", "resource"}:
        from ...robot.model.robot_information import robot_resources, robot_resource_path
        selected = robot_id if robot_id is not None else (runtime.capabilities or {}).get("robot_model_id")
        result = robot_resources(selected or "unknown")
        if section == "resource":
            resource_id = args.get("resource_id")
            if not isinstance(resource_id, str) or not resource_id:
                raise ValueError("robot_resource_id_required")
            path = robot_resource_path(selected or "unknown", resource_id)
            if path is None:
                result = dict(robot_id=selected, available=False, error="robot_resource_unavailable")
            else:
                resource = next(item for item in result["resources"] if item["resource_id"] == resource_id)
                result = dict(robot_id=selected, available=True, resource=resource,
                    resource_ref=runtime.store.put_artifact(path.read_bytes(),
                        "robot_geometry_model" if resource["format"] == "ZIP" else "robot_cad",
                        "application/zip" if resource["format"] == "ZIP" else "model/step"))
        return ToolOutput(result, projection="owner_record").response()
    if section == "geometry_image":
        from ...robot.model.robot_information import robot_geometry_path
        selected = robot_id if robot_id is not None else (runtime.capabilities or {}).get("robot_model_id")
        path = robot_geometry_path(selected or "unknown")
        if path is None:
            return ToolOutput(dict(robot_id=selected, available=False, section=section,
                                   error="robot_geometry_image_unavailable")).response()
        url, _ = image_url(path.read_bytes(), runtime.image_settings)
        return ToolOutput(dict(robot_id=selected, available=True, section=section,
            description="Illustrative robot geometry and contact references; not a measured scene or executed trajectory."),
            extra_content=(dict(type="inputImage", imageUrl=url),)).response()
    return ToolOutput(runtime.robot_information(section, robot_id), projection="robot_information").response()


def robot_observe(context, args, correlation):
    runtime = runtime_port(context)
    from ...perception.focus_3d import deliver_focus_3d_observation, validate_observation_view_args
    from ..actions.inspection import embodiment_of
    overview_cam = embodiment_of(runtime).overview_camera or 'cam_head'
    mode, focus_spec = validate_observation_view_args(args, overview_camera=overview_cam)
    camera_id, roi, scale = args.get('camera_id'), args.get('roi'), args.get('scale', 1)
    detail = args.get('detail')
    focus_result = None
    captured_live = False
    bundle = None
    try:
        if focus_spec is not None:
            # 3D 球域反投影特写：当前帧未失效时直接复用最新观测做 1:1 原始像素裁剪，不推进仿真且不替换当前 observation_id
            if runtime.latest is not None and not runtime.active_action:
                bundle = {**runtime.latest, "action_budget": runtime.action_budget()}
            else:
                bundle = runtime.observe(correlation)
                captured_live = True
            content, focus_result = deliver_focus_3d_observation(
                runtime, bundle, args['focus_3d'], correlation, camera_id=camera_id, detail=detail,
            )
        elif mode == 'none':
            # 纯数值无图观测：快速刷新关节/末端状态与预算（0 张图像），并写入真实空图 delivery 通过审计
            bundle = runtime.observe(correlation)
            captured_live = True
            content = runtime.observation_content(
                bundle, correlation, numeric_only=(detail != 'full'), include_images=False,
                detail=detail, structured=True,
            )
        elif mode == 'head_only':
            if (roi is not None or scale != 1) and runtime.latest is not None and not runtime.active_action:
                bundle = {**runtime.latest, "action_budget": runtime.action_budget()}
            else:
                bundle = runtime.observe(correlation)
                captured_live = True
            content = runtime.observation_content(
                bundle, correlation, overview_cam, roi, scale, detail=detail, structured=True,
            )
            if roi is None and scale == 1:
                if content and hasattr(content[0], 'value') and isinstance(content[0].value, dict):
                    all_ids = [c['camera_id'] for c in bundle.get('cameras', []) if isinstance(c, dict) and c.get('camera_id')]
                    content[0].value['image_feedback'] = dict(
                        policy='head_only',
                        observation_id=bundle['observation_id'],
                        included_camera_ids=[overview_cam] if overview_cam in all_ids else all_ids[:1],
                        omitted_camera_ids=[cid for cid in all_ids if cid != overview_cam],
                    )
                content = [*content, *_enrich_observe(runtime, bundle, correlation)]
        elif mode == 'all':
            bundle = runtime.observe(correlation)
            captured_live = True
            content = runtime.observation_content(
                bundle, correlation, full_view_reason='explicit_image_delivery_all',
                detail=detail, structured=True,
            )
            content = [*content, *_enrich_observe(runtime, bundle, correlation)]
        else:
            if camera_id is not None and runtime.latest is not None and not runtime.active_action:
                bundle = {**runtime.latest, "action_budget": runtime.action_budget()}
            else:
                bundle = runtime.observe(correlation)
                captured_live = True
            content = runtime.observation_content(bundle, correlation, camera_id, roi, scale,
                                                  detail=detail, structured=True)
            if camera_id is None:
                content = [*content, *_enrich_observe(runtime, bundle, correlation)]

        if mode is not None and content and hasattr(content[0], 'value') and isinstance(content[0].value, dict):
            content[0].value['image_delivery'] = mode
        if correlation.get('program_id'):
            obs_val = deepcopy(content[0].value) if content and hasattr(content[0], 'value') else {}
            obs_data = {**(obs_val if isinstance(obs_val, dict) else {}), **bundle}
        else:
            obs_data = dict(bundle) if (focus_result is not None or mode is not None) else bundle
        if focus_result is not None:
            obs_data = {**obs_data, 'focus_3d': deepcopy(focus_result)}
        if mode is not None:
            obs_data = {**obs_data, 'image_delivery': mode}
        response = ToolResponse(content, data=obs_data, kind='observation', data_projection='observation')
        # 内部观测采样来源事实（供 W5 runner 区分实时新采帧与缓存复用，不进入模型可见文本）
        response.captured_live = bool(captured_live)
        return response
    except Exception:
        if not captured_live or not isinstance(bundle, dict):
            raise
        from ...core.tool_output import StructuredText, output_contract_error
        try:
            fallback_content = runtime.observation_content(
                bundle, correlation, numeric_only=(detail != 'full'), include_images=False,
                detail=detail, structured=True,
            )
        except Exception:
            fallback_content = [
                StructuredText(
                    dict(
                        bundle=dict(
                            observation_id=bundle['observation_id'],
                            sim_step=bundle.get('sim_step'),
                            sim_time_s=bundle.get('sim_time_s'),
                            world_kind=bundle.get('world_kind', 'observed'),
                            action_budget=bundle.get('action_budget'),
                        ),
                        robot_state=dict(arms={}, sim_step=bundle.get('sim_step')),
                        current_observation_id=bundle['observation_id'],
                        observation_is_current=True,
                    ),
                    projection='observation',
                )
            ]
        fallback_data = dict(bundle)
        if correlation.get('program_id') and fallback_content and hasattr(fallback_content[0], 'value'):
            val0 = deepcopy(fallback_content[0].value)
            if isinstance(val0, dict):
                fallback_data = {**val0, **fallback_data}
        if mode is not None:
            fallback_data['image_delivery'] = mode
        err_resp = output_contract_error(
            ToolResponse(fallback_content, data=fallback_data, kind='observation', data_projection='observation'),
            'robot_observe',
            'observation_render_unavailable',
            '/content',
        )
        err_resp.captured_live = bool(captured_live)
        return err_resp


def history_view_observation(context, args, correlation):
    runtime = runtime_port(context)
    from ...perception.focus_3d import deliver_focus_3d_observation, validate_observation_view_args
    from ..actions.inspection import embodiment_of
    overview_cam = embodiment_of(runtime).overview_camera or "cam_head"
    mode, focus_spec = validate_observation_view_args(args, overview_camera=overview_cam)
    bundle = runtime.store.read("observation", args["observation_id"])
    camera_id, roi, scale = args.get("camera_id"), args.get("roi"), args.get("scale", 1)
    detail = args.get("detail")

    if focus_spec is not None:
        content, _ = deliver_focus_3d_observation(
            runtime, bundle, args["focus_3d"], correlation, camera_id=camera_id, detail=detail,
        )
    elif mode == "none":
        # 历史观测无图读取：保留完整已存 bundle 与位姿/标定元数据（满足 OBSERVATION_METADATA），仅省略图像传输
        content = runtime.observation_content(
            bundle, correlation, numeric_only=False, include_images=False, detail=detail, structured=True,
        )
        if content and hasattr(content[0], "value") and isinstance(content[0].value, dict):
            all_ids = [c["camera_id"] for c in bundle.get("cameras", []) if isinstance(c, dict) and c.get("camera_id")]
            content[0].value["image_feedback"] = dict(
                policy="none",
                observation_id=bundle["observation_id"],
                included_camera_ids=[],
                omitted_camera_ids=all_ids,
            )
    elif mode == "head_only":
        content = runtime.observation_content(
            bundle, correlation, overview_cam, roi, scale, detail=detail, structured=True,
        )
        if roi is None and scale == 1 and content and hasattr(content[0], "value") and isinstance(content[0].value, dict):
            all_ids = [c["camera_id"] for c in bundle.get("cameras", []) if isinstance(c, dict) and c.get("camera_id")]
            content[0].value["image_feedback"] = dict(
                policy="head_only",
                observation_id=bundle["observation_id"],
                included_camera_ids=[overview_cam] if overview_cam in all_ids else all_ids[:1],
                omitted_camera_ids=[cid for cid in all_ids if cid != overview_cam],
            )
    elif mode == "all":
        content = runtime.observation_content(
            bundle, correlation, full_view_reason="explicit_image_delivery_all", detail=detail, structured=True,
        )
    else:
        content = runtime.observation_content(
            bundle, correlation, camera_id, roi, scale, detail=detail, structured=True,
        )

    if mode is not None and content and hasattr(content[0], "value") and isinstance(content[0].value, dict):
        content[0].value["image_delivery"] = mode
    obs_val = deepcopy(content[0].value)
    obs_data = {**(obs_val.get("bundle") if isinstance(obs_val, dict) and isinstance(obs_val.get("bundle"), dict) else {}),
                **(obs_val if isinstance(obs_val, dict) else {})}
    return ToolResponse(content, data=obs_data)


def task_plan(context, args, correlation):
    runtime = runtime_port(context)
    with runtime.lock:
        result = runtime.task_planning.call(args, runtime.identity, runtime.latest, correlation)
    return ToolOutput(result, projection="owner_record").response()


def robot_capabilities(context, args, correlation):
    runtime = runtime_port(context)
    detail = resolve_detail(args.get("detail"), program=bool(correlation.get("program_id")),
        default="full" if runtime.actor_payload_profile == "full_v1" else "summary")
    if detail == "summary" and not correlation.get("program_id"):
        dc = getattr(runtime, "decision_context", None)
        if dc is not None:
            epoch = dc.actor_delivery_epoch(correlation)
            live_epoch = dc.live_context_epoch()
        else:
            epoch = runtime._model_context_epoch()
            live_epoch = epoch
        result = runtime._capability_delivery.project(
            runtime.capabilities,
            epoch=epoch,
            current_action_budget=runtime.action_budget(),
            live_epoch=live_epoch,
        )
    else:
        result = runtime.capabilities
    return ToolOutput(result, projection="capabilities").response()


def vision_compare(context, args, correlation):
    runtime = runtime_port(context)
    from ...perception.visual_feedback import compare_observations
    result = compare_observations(runtime.store, runtime.identity, **args)
    return ToolOutput(result, projection="owner_record").response()


def robot_action_status(context, args, correlation):
    runtime = runtime_port(context)
    from ...core.presentation.history_output import action_status
    result = action_status(runtime, args, correlation)
    return ToolOutput(result, projection="action_result").response()


def history_list_actions(context, args, correlation):
    runtime = runtime_port(context)
    from ...core.presentation.history_output import list_actions
    with runtime.lock:
        result = list_actions(runtime, args, correlation)
    return ToolOutput(result, projection="action_history").response()


def history_read_trace(context, args, correlation):
    runtime = runtime_port(context)
    from ...core.presentation.history_output import read_trace
    result = read_trace(runtime, args, correlation)
    return ToolOutput(result).response()


def history_annotate_action(context, args, correlation):
    runtime = runtime_port(context)
    runtime.action(args["action_id"])
    if not isinstance(args["note"], str) or len(args["note"]) > 16000: raise ValueError("invalid_action_note")
    result = runtime.store.save("action_note", new_id("note"), dict(action_id=args["action_id"], note=args["note"], source_kind="actor_judgment", correlation=correlation))
    return ToolOutput(result, projection="owner_record").response()


def program_status(context, args, correlation):
    runtime = runtime_port(context)
    from ..programs.program_result import program_result
    from .program_tools import program_content
    detail = resolve_detail(args.get("detail"), program=bool(correlation.get("program_id")))
    job = runtime.programs.status(args["job_id"])
    result = program_result(runtime.store, job, detail=detail)
    return ToolResponse(program_content(runtime, job, result, detail, correlation), data=result, data_projection='owner_record')


def vision_review(context, args, correlation):
    runtime = runtime_port(context)
    args = dict(args)
    operation = args.pop('evidence_mode', 'images') == 'operation'
    reviewer = runtime.visual_reviewer.review_operation if operation else runtime.visual_reviewer.review
    return ToolOutput(reviewer(runtime.store, runtime.identity, **args,
        **runtime._review_controls(correlation)),
        projection="operation_review" if operation else "owner_record").response()


def robot_contact_overlay(context, args, correlation):
    runtime = runtime_port(context)
    from ..actions.inspection import contact_overlay
    content = contact_overlay(runtime, args, correlation, structured=True)
    return ToolResponse(content, data=deepcopy(content[0].value), data_projection="contact_overlay")


def robot_action(context, args, correlation):
    runtime = runtime_port(context)
    from ..actions.unified_actions import dispatch
    return dispatch(runtime, args, correlation)


def robot_wait(context, args, correlation):
    from .actions import execute_action
    return execute_action(context, 'robot_wait', args, correlation)


def robot_policy_step(context, args, correlation):
    from .actions import execute_action
    return execute_action(context, 'robot_policy_step', args, correlation)


def program_run(context, args, correlation):
    runtime = runtime_port(context)
    from .program_tools import run_program
    return run_program(runtime, 'program_run', args, correlation)


def operation_memory(context, args, correlation):
    runtime = runtime_port(context)
    return ToolOutput(runtime.operation_memory.call(args['request'], correlation)).response()


def depth_query(context, args, correlation):
    runtime = runtime_port(context)
    result = runtime.depth.query(args['request'], correlation)
    if not correlation.get('program_id'):
        from ...perception.depth.depth_payload import numeric_summary
        result = numeric_summary(result)
    return ToolOutput(result, projection="owner_record").response()


def object_table_query(context, args, correlation):
    """features.object_table: geometry_query kind object_table (ni_rua/object_table.py)."""
    runtime = runtime_port(context)
    from ...perception.external.object_table import object_table_query as run_object_table
    result, overlays = run_object_table(runtime, args['request'])
    val = result.get("value")
    obs_id = args["request"].get("observation_id")
    if isinstance(val, dict) and obs_id:
        from ...perception.external.obs3d import _memo, _remember, attached_table_view
        if "object_table" not in _memo(runtime, obs_id):
            _remember(runtime, obs_id, "object_table", (attached_table_view(val), None, result))
    model_copy = {key: value for key, value in result.items() if key != "diagnostics"}
    from ...core.storage.images import image_settings
    settings = runtime.image_settings if runtime.image_settings is not None else image_settings(None)
    content = ToolOutput(model_copy, projection="geometry_result").content_parts()
    for camera_id, data, original in overlays:
        reference = runtime.store.put_artifact(data, "geometry_overlay", "image/jpeg")
        url, encoding = image_url(data, settings)
        content.append(dict(type="inputImage", imageUrl=url))
        runtime.store.event("geometry.overlay_delivered", {"image_ref": reference, "source_sha256": encoding["sha256"],
            "sha256": encoding["sha256"], "encoding": encoding, "original_rgb_ref": original, "camera_id": camera_id,
            "source_kind": "object_table_overlay", "delivered_to_model": not bool(correlation.get("program_id"))},
            correlation, world_kind="internal")
        result.setdefault("overlay_refs", {})[camera_id] = reference
    return ToolResponse(content, data=result, data_projection='geometry_result')


def reanchor_tracks_query(context, args, correlation):
    """慢→快语义重锚定：`geometry_query(request={"kind": "reanchor_tracks", ...})`。"""
    runtime = runtime_port(context)
    from ...perception.external.reanchor_tracks import reanchor_tracks_query as run_reanchor_tracks
    result, overlays = run_reanchor_tracks(runtime, args['request'])
    val = result.get("value")
    obs_id = args["request"].get("observation_id")
    if isinstance(val, dict) and obs_id:
        from ...perception.external.obs3d import _memo, _remember, _sync_virtual_field, attached_table_view
        shown = attached_table_view(val)
        _remember(runtime, obs_id, "object_table", (shown, None, result))
        raw_objs = (result.get("diagnostics") or {}).get("objects") if isinstance(result, dict) else None
        s3_bytes = None
        cached_s3 = _memo(runtime, obs_id).get("scene3d")
        if isinstance(cached_s3, tuple) and len(cached_s3) == 2 and isinstance(cached_s3[0], dict) and cached_s3[0].get("artifact"):
            resolve_art = getattr(getattr(runtime, "store", None), "resolve_artifact", None)
            if callable(resolve_art):
                try:
                    _, s3_bytes = resolve_art(cached_s3[0]["artifact"])
                except Exception:
                    s3_bytes = None
        _sync_virtual_field(runtime, {"observation_id": obs_id}, shown, cloud_npz=s3_bytes, raw_objects=raw_objs)
    model_copy = {key: value for key, value in result.items() if key != "diagnostics"}
    from ...core.storage.images import image_settings
    settings = runtime.image_settings if runtime.image_settings is not None else image_settings(None)
    content = ToolOutput(model_copy, projection="geometry_result").content_parts()
    for camera_id, data, original in overlays:
        reference = runtime.store.put_artifact(data, "geometry_overlay", "image/jpeg")
        url, encoding = image_url(data, settings)
        content.append(dict(type="inputImage", imageUrl=url))
        runtime.store.event("geometry.overlay_delivered", {"image_ref": reference, "source_sha256": encoding["sha256"],
            "sha256": encoding["sha256"], "encoding": encoding, "original_rgb_ref": original, "camera_id": camera_id,
            "source_kind": "reanchor_tracks_overlay", "delivered_to_model": not bool(correlation.get("program_id"))},
            correlation, world_kind="internal")
        result.setdefault("overlay_refs", {})[camera_id] = reference
    return ToolResponse(content, data=result, data_projection='geometry_result')


def scene3d_query(context, args, correlation):
    """features.scene3d: geometry_query kind scene3d (ni_rua/scene3d.py); the cloud is stored, not returned."""
    runtime = runtime_port(context)
    from ...perception.external.scene3d import scene3d_query as run_scene3d, ARTIFACT_KIND, ARTIFACT_MIME
    result, data = run_scene3d(runtime, args['request'])
    obs_id = args['request'].get("observation_id")
    if data is not None:
        reference = runtime.store.put_artifact(data, ARTIFACT_KIND, ARTIFACT_MIME)
        result["value"]["artifact"] = reference
        runtime.store.event("geometry.scene3d_stored", {"artifact": reference, "bytes": len(data),
            "points": result["value"]["points"], "observation_id": obs_id,
            "camera_id": result["value"]["camera_id"]}, correlation, world_kind="internal")
        if obs_id:
            from ...perception.external.obs3d import _memo, _sync_virtual_field
            cached_ot = _memo(runtime, obs_id).get("object_table")
            cached_table = cached_ot[0] if isinstance(cached_ot, tuple) and cached_ot and isinstance(cached_ot[0], dict) else None
            cached_res = cached_ot[2] if isinstance(cached_ot, tuple) and len(cached_ot) > 2 and isinstance(cached_ot[2], dict) else None
            raw_objs = (cached_res.get("diagnostics") or {}).get("objects") if isinstance(cached_res, dict) else None
            _sync_virtual_field(runtime, {"observation_id": obs_id}, cached_table, cloud_npz=data, raw_objects=raw_objs)
    model_copy = {key: value for key, value in result.items() if key != "diagnostics"}
    return ToolResponse(ToolOutput(model_copy, projection="geometry_result").content_parts(), data=result,
                        data_projection='geometry_result')


_GEOMETRY_BATCH_INHERITED_KEYS = (
    'observation_id', 'camera_id', 'artifact', 'plane_origin_m',
    'plane_normal', 'plane_source', 'plane_origin_alt_m', 'target_camera_id',
)
_GEOMETRY_BATCH_KIND_ALIASES = {'ray': 'pixel_ray', 'project': 'project_points'}
_GEOMETRY_BATCH_MAX_DEPTH = 8


def is_geometry_batch_request(req):
    """判断几何查询请求是否为批次形态（保留省略 kind 的兼容写法）。"""
    return isinstance(req, dict) and isinstance(req.get('queries'), list) and req.get('kind') in (None, 'batch')


def normalize_geometry_batch_item(parent_req, item):
    """按父级批次上下文规范化单个子查询：补齐可继承字段、归一化别名与内联 scene3d op。"""
    sub_req = dict(item)
    for inherited in _GEOMETRY_BATCH_INHERITED_KEYS:
        if inherited in parent_req and inherited not in sub_req:
            sub_req[inherited] = parent_req[inherited]
    if sub_req.get('kind') in _GEOMETRY_BATCH_KIND_ALIASES:
        sub_req['kind'] = _GEOMETRY_BATCH_KIND_ALIASES[sub_req['kind']]
    if 'op' in sub_req and 'kind' not in sub_req:
        sub_req = {
            'kind': 'scene3d',
            'query': dict(item),
            **{k: sub_req[k] for k in ('observation_id', 'camera_id', 'artifact') if k in sub_req},
        }
    return sub_req


def expand_geometry_requests(req, *, max_depth=_GEOMETRY_BATCH_MAX_DEPTH, strict=False):
    """递归展开 geometry_query 批次为叶子子请求列表（处理器与 builtins 分类器共用同一展开逻辑）。

    strict=True：遇非法结构、空/超长 queries 或超过嵌套深度时抛 ToolInputError。
    strict=False：用于分类器/可用性预判，非法结构或超深返回 None（调用方按最保守串行/拒绝处理）。
    """
    from ...core.tool_errors import ToolInputError
    if not isinstance(req, dict):
        if strict:
            raise ToolInputError('invalid_geometry_request', field_path='/request', actual=req)
        return []
    if not is_geometry_batch_request(req):
        return [req]

    leaves = []

    def _walk(node, depth, path):
        if depth > max_depth:
            if strict:
                raise ToolInputError(
                    'invalid_geometry_batch_depth',
                    field_path=path,
                    actual=depth,
                    suggestion=f'Nest geometry_query batch requests at most {max_depth} levels deep.',
                )
            return False
        queries = node.get('queries')
        if not isinstance(queries, list) or len(queries) > 32 or (depth > 1 and not queries):
            if strict:
                raise ToolInputError(
                    'invalid_geometry_batch_queries',
                    field_path=path + '/queries',
                    actual=queries,
                    suggestion='Provide 1..32 query objects in queries.',
                )
            return False
        for idx, item in enumerate(queries):
            item_path = f'{path}/queries/{idx}'
            if not isinstance(item, dict):
                if strict:
                    raise ToolInputError(
                        'invalid_geometry_batch_item',
                        field_path=item_path,
                        actual=item,
                        suggestion='Each geometry_query batch item must be a query object.',
                    )
                return False
            sub_req = normalize_geometry_batch_item(node, item)
            if is_geometry_batch_request(sub_req):
                if not _walk(sub_req, depth + 1, item_path):
                    return False
            else:
                leaves.append((item_path, sub_req))
        return True

    if not _walk(req, 1, '/request'):
        return None
    return [sub for _, sub in leaves] if not strict else leaves


def _admit_geometry_batch(runtime, req):
    """在执行任何写入子项前完成整批递归结构、RGB-D 权限与禁用工具准入检查。"""
    from ...core.observation_policy import allows_depth
    from ...core.tool_errors import ToolInputError
    from ...perception.depth.depth_runtime import DEPTH_KINDS

    leaves = expand_geometry_requests(req, strict=True)
    profile = getattr(runtime, 'observation_profile', None)
    disabled = getattr(runtime, 'disabled_tools', ()) or ()
    for item_path, sub_req in leaves:
        kind = sub_req.get('kind')
        if isinstance(kind, str) and kind in DEPTH_KINDS:
            if not allows_depth(profile):
                raise ToolInputError(
                    'depth_not_enabled_for_run',
                    field_path=item_path,
                    actual=kind,
                    suggestion='RGB-D depth kinds are not enabled for this run; remove depth sub-queries from the batch.',
                )
            if 'depth_query' in disabled:
                raise ToolInputError(
                    'tool_disabled_by_config:depth_query',
                    field_path=item_path,
                    actual=kind,
                    suggestion='depth_query is disabled by run configuration; remove depth sub-queries from the batch.',
                )


def _execute_geometry_batch(context, runtime, req, correlation):
    from ...core.tool_errors import ToolInputError, extract_domain_code
    from ...perception.external import embodiment_of

    batch_results, extra_images = [], []
    for item in req['queries']:
        sub_req = normalize_geometry_batch_item(req, item)
        try:
            if is_geometry_batch_request(sub_req):
                sub_resp = _execute_geometry_batch(context, runtime, sub_req, correlation)
            else:
                sub_resp = _execute_geometry_leaf(context, runtime, sub_req, correlation)
        except (ToolInputError, ValueError) as error:
            reason = error.code if isinstance(error, ToolInputError) else str(error)
            if not isinstance(error, ToolInputError) and extract_domain_code(reason) is None:
                raise
            # 仅捕获已准入子项的输入/领域错误并写回原位；非领域内部异常照常上抛
            batch_results.append({
                'query_type': sub_req.get('kind') if isinstance(sub_req.get('kind'), str) else 'unknown',
                'status': 'invalid_input',
                'reason': reason,
            })
            continue
        sub_data = sub_resp.data if isinstance(sub_resp.data, dict) else {}
        entry = {k: sub_data[k] for k in ('query_type', 'status', 'reason', 'value', 'assumptions') if k in sub_data}
        if 'query_type' not in entry and isinstance(sub_data.get('kind'), str):
            entry['query_type'] = sub_data['kind']
        for depth_key in ('measurement_id', 'target_id', 'target_status', 'targets'):
            if depth_key in sub_data and depth_key not in entry:
                entry[depth_key] = sub_data[depth_key]
        batch_results.append(entry)
        for part in sub_resp.content:
            if isinstance(part, dict) and part.get('type') in ('inputImage', 'input_image', 'image'):
                extra_images.append(part)
    obs_ids = list(dict.fromkeys(
        ([req['observation_id']] if req.get('observation_id') else [])
        + [q.get('observation_id') for q in req['queries'] if isinstance(q, dict) and q.get('observation_id')]
    ))
    common_assumptions = []
    if len(batch_results) > 1 and all('assumptions' in r for r in batch_results):
        first_a = batch_results[0]['assumptions']
        if all(r['assumptions'] == first_a for r in batch_results):
            common_assumptions = first_a
            for r in batch_results:
                r.pop('assumptions', None)
    outer_status = 'supported'
    outer_reason = None
    if batch_results and all(r.get('status') not in ('supported', 'conditional') for r in batch_results):
        if all(r.get('status') == 'invalid_input' for r in batch_results):
            outer_status = 'invalid_input'
            outer_reason = 'all_batch_queries_invalid_input'
        else:
            outer_status = 'unknown'
            outer_reason = 'all_batch_queries_failed'
    result = {
        'schema_version': '0.1',
        **runtime.identity,
        'world_kind': 'internal',
        'query_type': 'batch',
        'observation_ids': obs_ids,
        'status': outer_status,
        **({'reason': outer_reason} if outer_reason else {}),
        'frame_id': embodiment_of(runtime).frame_id,   # the run's pose frame (ARX-X5 env_origin, SO-101 so101_base)
        'unit': 'm',
        **({'assumptions': common_assumptions} if common_assumptions else {}),
        'value': {'count': len(batch_results), 'batch_results': batch_results},
    }
    content = [*ToolOutput(result, projection='geometry_result').content_parts(), *extra_images]
    return ToolResponse(content, data=result, data_projection='geometry_result')


def _enrich_geometry_help(runtime, result):
    """按当前运行已启用的外部感知扩展补充 geometry_query(kind='help') 索引。"""
    requests = result.get('requests')
    if not isinstance(requests, dict):
        return
    from ...perception.external.object_table import enabled as object_table_enabled
    from ...perception.external.reanchor_tracks import enabled as reanchor_tracks_enabled
    from ...perception.external.scene3d import enabled as scene3d_enabled
    from ...perception.external.vggt_height import mode as vggt_height_mode

    if vggt_height_mode(runtime):
        requests['pixel_plane'] = requests.get('pixel_plane', '') + ' Optional plane_source="vggt"|"caller" estimates object centre height via VGGT.'
    if object_table_enabled(runtime):
        requests['object_table'] = 'observation_id, optional camera_id; detects tabletop objects with 3D centre/bounds/yaw estimates'
    if reanchor_tracks_enabled(runtime):
        requests['reanchor_tracks'] = 'observation_id, optional camera_id; re-anchors persistent object tracks on the current observation'
    if scene3d_enabled(runtime):
        requests['scene3d'] = 'observation_id (or artifact), optional camera_id/query/queries; stores or queries a monocular 3D point cloud artifact'


def _execute_geometry_leaf(context, runtime, req, correlation):
    args = {'request': req}
    from ...perception.depth.depth_runtime import DEPTH_KINDS
    if req.get('kind') in DEPTH_KINDS:
        return depth_query(context, args, correlation)
    from ...robot.model.geometry import GeometryTools
    from ...perception.external import run_embodiment
    from ...perception.external.object_table import applies as object_table_applies
    if object_table_applies(runtime, req):
        return object_table_query(context, args, correlation)
    from ...perception.external.reanchor_tracks import applies as reanchor_tracks_applies
    if reanchor_tracks_applies(runtime, req):
        return reanchor_tracks_query(context, args, correlation)
    from ...perception.external.scene3d import applies as scene3d_applies
    if scene3d_applies(runtime, req) or (req.get('kind') == 'scene3d' and ('artifact' in req or 'query' in req or 'queries' in req)):
        return scene3d_query(context, args, correlation)
    from ...perception.external.vggt_height import applies as vggt_height_applies
    if vggt_height_applies(runtime, req):
        from ...perception.external.vggt_height import pixel_plane_with_estimate
        result = pixel_plane_with_estimate(runtime, req)
    else:
        result = GeometryTools(runtime.scene, settings=runtime.image_settings, profile=run_embodiment(runtime)).execute(
            {**req, **runtime.identity, 'schema_version': '0.1'})
        if req.get('kind') == 'help':
            _enrich_geometry_help(runtime, result)
    content = ToolOutput(result, projection="geometry_result").content_parts()
    if isinstance(result.get('value'), dict) and result['value'].get('overlay_ref'):
        reference = result["value"]["overlay_ref"]
        info, data = runtime.scene.resolve_artifact(reference)
        copied = runtime.store.put_artifact(data, info["kind"], info["mime_type"], info["access_scope"])
        if copied != reference: raise ValueError("geometry_overlay_copy_identity_mismatch")
        url, encoding = image_url(data, runtime.image_settings)
        content.append(dict(type="inputImage", imageUrl=url))
        runtime.store.event("geometry.overlay_delivered", {"image_ref": reference, "source_sha256": info["sha256"],
            "sha256": encoding["sha256"], "encoding": encoding,
            "original_rgb_ref": result["value"].get("original_rgb_ref"), "source_kind": "geometry_overlay",
            "delivered_to_model": not bool(correlation.get("program_id"))}, correlation, world_kind="internal")
    return ToolResponse(content, data=result, data_projection='geometry_result')


def geometry_query(context, args, correlation):
    runtime = runtime_port(context)
    req = args['request']
    if is_geometry_batch_request(req):
        _admit_geometry_batch(runtime, req)
        return _execute_geometry_batch(context, runtime, req, correlation)
    return _execute_geometry_leaf(context, runtime, req, correlation)


def pose_plan(context, args, correlation):
    """features.pose_skill: read-only pose planning from the object table (ni_rua/pose_skill.py)."""
    runtime = runtime_port(context)
    from ...perception.external.pose_skill import handle
    return handle(runtime, args, correlation)
