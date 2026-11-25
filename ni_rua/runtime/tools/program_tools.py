"""Tool handlers. Registration binds these functions; dispatch has no name switch."""
import json

from ...core.tool_output import ToolOutput, ToolResponse, resolve_detail


def program_display(runtime, result, detail, correlation):
    if correlation.get('program_id') or not correlation.get('actor_log_delivery'):
        return result
    dc = getattr(runtime, 'decision_context', None)
    if dc is not None:
        epoch = dc.actor_delivery_epoch(correlation)
        if epoch is None:
            return result
        live_epoch = dc.live_context_epoch()
    else:
        epoch = runtime._model_context_epoch()
        live_epoch = epoch
    projected = runtime._program_output_delivery.project(
        result,
        epoch=epoch,
        call_id=correlation.get('tool_call_id'),
        detail=detail,
        live_epoch=live_epoch,
    )
    now_epoch = dc.live_context_epoch() if dc is not None else runtime._model_context_epoch()
    return projected if now_epoch == epoch else result


def program_content(runtime, job, result, detail, correlation):
    """Forward broker-owned images with their original labels, never stdout media."""
    visual_parts = {}
    errors = []
    for feedback in result.get('visual_feedback', []):
        try:
            info, raw = runtime.store.resolve_artifact(feedback['reply_ref'], allowed_scopes=('actor',))
            if info.get('kind') != 'program_visual_reply':
                raise ValueError('program_visual_artifact_kind_mismatch')
            parts = json.loads(raw)['contentItems']
        except Exception as error:
            feedback['delivery'] = dict(status='unavailable', reason=str(error))
            errors.append(dict(source='supplementary', call_id=feedback['call_id'], reason=str(error)))
            continue
        visual_parts[feedback['call_id']] = ToolOutput(dict(program_visual_feedback=feedback,
            current_observation_id_at_delivery=(runtime.latest or {}).get('observation_id'),
            semantics='Saved program tool reply at this call; preview/historical images are not a new current observation. '
                'Original currentness flags and budgets are as of the source call, not delivery time.'
            )).content_parts()
        visual_parts[feedback['call_id']] += parts
    extra = []
    if job.get('status') == 'awaiting_review':
        try:
            extra += [item for item in runtime.programs.review_content(job['job_id'])
                        if item.get('type') == 'inputImage']
        except Exception as error:
            errors.append(dict(source='pending_review', reason=str(error)))
    if job.get('final_observation_id'):
        try:
            bundle = runtime.store.read('observation', job['final_observation_id'])
            extra += runtime.observation_content(bundle,
                correlation, structured=True)
        except Exception as error:
            errors.append(dict(source='final_observation', reason=str(error)))
        else:
            try:
                recent = (job.get('action_summary') or {}).get('recent_actions') or []
                pipeline = getattr(runtime, 'observation_pipeline', None)
                if recent:
                    before_obs = next(
                        (item.get('based_on_observation_id') for item in recent
                         if isinstance(item, dict) and item.get('based_on_observation_id')),
                        None,
                    )
                    if pipeline is not None:
                        extra += list(pipeline.enrich(bundle, correlation, reply='action', before_observation_id=before_obs).parts)
                    else:
                        from ...perception.external.obs3d import action_attachments
                        extra += action_attachments(runtime, bundle, correlation, before_observation_id=before_obs)
                else:
                    if pipeline is not None:
                        extra += list(pipeline.enrich(bundle, correlation, reply='observe').parts)
                    else:
                        from ...perception.external.obs3d import attachments
                        extra += attachments(runtime, bundle, correlation, reply='observe')
            except Exception:
                pass
    if errors:
        # A display failure must not erase already recorded action/process facts.
        result['visual_feedback_errors'] = errors
        if (correlation or {}).get('context_scope', 'main') == 'main':
            runtime._program_output_delivery.invalidate()
    # Preparing final-frame content can reset the image context. Project afterwards.
    result = program_display(runtime, result, detail, correlation)
    content = ToolOutput(result, projection='owner_record').content_parts()
    for feedback in result.get('visual_feedback', []):
        if not feedback.get('delivery'):
            content += visual_parts[feedback['call_id']]
    return content + extra


def run_program(runtime, name, args, correlation):
    if not runtime.programs: raise ValueError("program_runner_not_started")
    detail = resolve_detail(args.get("detail"), program=bool(correlation.get("program_id")))
    args = {key: value for key, value in args.items() if key != "detail"}
    clean_correlation = (
        {key: value for key, value in correlation.items() if key != "_raw_inline_source"}
        if isinstance(correlation, dict) and "_raw_inline_source" in correlation
        else correlation
    )
    from ..programs.program_result import program_result
    job = runtime.programs.run(args, correlation)
    result = program_result(runtime.store, job, detail=detail)
    content = program_content(runtime, job, result, detail, clean_correlation)
    return ToolResponse(content, data=job, kind="program", data_projection="owner_record")
