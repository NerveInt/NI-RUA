"""Broker-owned program review and execution evidence, never stdout parsing."""
from copy import deepcopy

from ...core.presentation.actor_payload import lean_action_result
from ...core.presentation.actor_presentation import ActorPresentation
from ...core.contracts import canonical
from .program_result import program_result
from ...core.tool_output import ToolResponse


LEGACY_ACTION_TOOLS = frozenset({
    "robot_execute_ee", "robot_move_ee", "robot_move_grasp", "robot_move_dual_grasp",
    "robot_gripper", "robot_execute_plan", "robot_execute_queue",
    "depth_move",
})


def reviewed_plan(response):
    value = response.data if isinstance(response, ToolResponse) and response.kind == 'plan' else None
    if not isinstance(value, dict) or not isinstance(value.get('plan_id'), str):
        raise ValueError('program_action_preview_missing_plan')
    if response.get('success') is False or value.get('status') not in {'planned', 'awaiting_observation'}:
        raise ValueError('program_action_preview_not_executable')
    return deepcopy(value)


def review_execution_state(plan):
    """The cached execution segment whose approval the main actor reviewed."""
    return {name: deepcopy(plan.get(name, 0 if name in {"executed_batch_native_steps", "plan_revision"} else None))
            for name in ("plan_id", "status", "trajectory_ref", "last_action_id",
                         "executed_batch_native_steps", "plan_revision")}


def action_evidence(store, tool, arguments, response, call_id, *, physical=None):
    """Capture actual dispatched action replies with their original references."""
    if physical is False or (physical is None and tool != 'robot_wait' and
            (tool != 'robot_action' or arguments.get('mode', 'preview') not in {'execute', 'direct'})):
        return None
    value = response.data if isinstance(response, ToolResponse) and response.kind == "action" else {}
    value = value if isinstance(value, dict) else {}
    receipt = value.get("receipt", {})
    receipt = receipt if isinstance(receipt, dict) else {}
    request = value.get("request", {})
    request = request if isinstance(request, dict) else {}
    status = receipt.get('status')
    result = lean_action_result(dict(request=request, receipt=receipt))["receipt"]
    result.update(call_id=call_id, tool=tool, status=status if isinstance(status, str) else 'unknown',
                  dispatch_success=response.get("success"),
                  action_id=receipt.get("action_id") or value.get("action_id") or request.get("action_id"),
                  plan_id=value.get("plan_id", arguments.get("plan_id")),
                  result_observation_id=value.get("current_observation_id") or receipt.get("result_observation_id"))
    based_on = request.get("based_on_observation_id") or arguments.get("observation_id")
    if isinstance(based_on, str) and based_on:
        result["based_on_observation_id"] = based_on
    if isinstance(value.get("next_action"), dict):
        result["next_action"] = deepcopy(value["next_action"])
    if isinstance(value.get("repair_action"), dict):
        result["repair_action"] = deepcopy(value["repair_action"])
    if status in ("stopped", "failed") and isinstance(value.get("episode_ledger"), dict):
        result["episode_ledger"] = deepcopy(value["episode_ledger"])
    if "repeat_failure_warning" in value:
        result["repeat_failure_warning"] = deepcopy(value["repeat_failure_warning"])
    plan = response.data if isinstance(response, ToolResponse) and response.kind == 'plan' else None
    is_auto_clean = arguments.get('mode') == 'execute_if_clean' or bool(arguments.get('auto_execute_if_clean'))
    if (tool == 'robot_action'
            and (arguments.get('mode') == 'direct' or is_auto_clean)
            and isinstance(plan, dict) and response.contract_error is None
            and (plan.get('status') == 'failed' or (is_auto_clean and plan.get('status') == 'planned' and plan.get('downgraded_from_execute_if_clean') is True))
            and plan.get('physics_advanced') is False
            and type(plan.get('executed_native_steps')) is int and plan['executed_native_steps'] == 0):
        # A direct or execute_if_clean call can stop or downgrade during planning without producing an action receipt.
        result.update(status='not_executed', phase='planning', physics_advanced=False,
                      executed_native_steps=0)
        for name, item in plan.items():
            if name not in result:
                result[name] = deepcopy(item)
    if value.get('current_observation_id') and value['current_observation_id'] != receipt.get('result_observation_id'):
        result['motion_observation_id'] = receipt.get('result_observation_id')
    if isinstance(value.get('settling'), dict):
        result['settling'] = deepcopy(value['settling'])
    try:
        result["reply_ref"] = store.put_artifact(canonical(ActorPresentation(store).present_response(response)).encode(),
            "program_action_reply", "application/json")
    except Exception as error:
        # A reply-copy failure must not erase an already established action outcome.
        result["reply_archive_error"] = dict(reason="program_action_reply_archival_failed",
                                            error_type=type(error).__name__)
    return result


def live_observation_id(tool, response):
    if tool != "robot_observe":
        return None
    if not isinstance(response, ToolResponse) or response.kind != 'observation':
        return None
    if response.get('success') is False or response.contract_error is not None:
        return None
    bundle = response.data
    if isinstance(bundle, dict) and isinstance(bundle.get('observation_id'), str) and bundle['observation_id']:
        return bundle['observation_id']
    return None


def observation_grounds_current_world(
    tool,
    response,
    *,
    clock: str,
    owner_observation_id: str | None = None,
    current_identity: dict | None = None,
    latched_meta: dict | None = None,
) -> tuple[bool, str | None, bool]:
    """按本体时钟与采样来源事实统一判定观测是否锚定当前真实世界状态。

    返回 `(grounded, observation_id, captured_live)`：
    - 实时钟（`clock != 'simulated_steps'`）：必须本次真实采样（`captured_live=True`）且与当前有效观测及环境身份一致；
    - 仿真步进时钟（`clock == 'simulated_steps'`）：允许 `captured_live=True`，或锁存动作具备可信终帧且满足
      `latched_result_observation_id == 返回帧 ID == 所有者当前有效 latest ID`（同代且停止已确认）。
    """
    obs_id = live_observation_id(tool, response)
    if not obs_id:
        return False, None, False
    bundle = response.data if isinstance(response.data, dict) else {}
    captured_attr = getattr(response, 'captured_live', None)
    captured_live = bool(bundle.get('captured_live', False) if captured_attr is None else captured_attr)
    if isinstance(current_identity, dict):
        for key in ('run_id', 'episode_id', 'env_instance_id', 'env_generation'):
            if current_identity.get(key) is not None and bundle.get(key) not in (None, current_identity[key]):
                return False, None, captured_live
    if owner_observation_id is not None and obs_id != owner_observation_id:
        return False, None, captured_live
    if captured_live:
        return True, obs_id, True
    if clock == 'simulated_steps' and isinstance(latched_meta, dict):
        latched_obs = latched_meta.get('result_observation_id')
        if (
            latched_meta.get('stop_confirmed') is True
            and isinstance(latched_obs, str)
            and bool(latched_obs)
            and owner_observation_id is not None
            and latched_obs == obs_id == owner_observation_id
        ):
            return True, obs_id, False
    return False, obs_id, False


def trusted_summary(store, evidence):
    statuses = {}
    for item in evidence:
        status = item.get("status", "unknown")
        statuses[status] = statuses.get(status, 0) + 1
    known_steps = [item["executed_native_steps"] for item in evidence
                   if type(item.get("executed_native_steps")) is int]
    settling = [item["settling"] for item in evidence if isinstance(item.get("settling"), dict)]
    settling_steps = [item["executed_native_steps"] for item in settling
                      if type(item.get("executed_native_steps")) is int]
    result = dict(source_kind="program_broker_action_replies", action_calls=len(evidence),
                  status_counts=statuses, executed_native_steps_known=sum(known_steps)+sum(settling_steps),
                  action_native_steps_known=sum(known_steps), settling_native_steps_known=sum(settling_steps),
                  actions_without_step_count=len(evidence)-len(known_steps),
                  settling_actions_without_step_count=len(settling)-len(settling_steps),
                  recent_actions=deepcopy(evidence), task_success="not_assessed",
                  semantics="Recorded action replies only. Process exit and stdout do not establish motion, contact, retention or task success.")
    if len(evidence) > 8:
        result["all_actions_ref"] = store.put_artifact(canonical(evidence).encode(), "program_action_evidence", "application/json")
    return result
