"""Benchmark continuation and observer-only final evidence, independent of HTTP.

This module consumes existing runtime owners and the shared scoring policy; it
never supplies evaluator facts to actor continuation or executes a robot action.
"""
import asyncio
import json
import time
from pathlib import Path

from ...core.contracts import IDENTITY, canonical
from ...cortex.supervision.actor_stall import (ACTOR_STALL_REASON, ActorStallTracker, ACTOR_PROGRESS_REASON,
                          ActorProgressTracker, completed_turn_activity)
from ...cortex.supervision.completion_guidance import COMPLETION_GUIDANCE_TRIGGER, hook_completion_guidance
from ...runtime.supervision import internal_runtime_tripped, model_budget_exhausted, wait_turn
from ..library.scoring import classify_episode_result
from ...cortex.session.usage import account_usage
from ...cortex.agents.base import session_backend_kind
from ...robot.model.embodiments import EmbodimentProfile

BENCHMARK_CONTINUATION = "任务还没完成，请继续。"


async def wait_benchmark(workbench, shutdown):
    """Continue early finals with bounded recovery; never supply task success."""
    session, harness = workbench.session, workbench.harness
    features = getattr(getattr(workbench, "config", None), "features", None)
    bounded = getattr(features, "bounded_continuation", False)
    evidence_aware = bounded and getattr(features, "recovery_evidence", False)
    stall = ActorProgressTracker(evidence_aware=evidence_aware) if bounded else ActorStallTracker()
    def program_state():
        inspect = getattr(workbench, 'program_continuation_state', None)
        return inspect() if inspect is not None else dict(blocked=workbench.has_active_programs(), pending_reviews=[])
    while True:
        finished = await wait_turn(session, shutdown, harness=harness)
        if finished and session.state == "disconnected":
            raise RuntimeError("benchmark_actor_disconnected")
        if not finished or session.state != "completed":
            return finished
        if harness.episode_done.is_set() or model_budget_exhausted(session):
            continue  # A terminal callback may race the completed-turn event.
        if (shutdown.is_set() or not harness.accepting or workbench.current_unknown_actions()
                or harness.active_action or program_state()['blocked']):
            # Reconciliation, explicit stop and executing or uncertain programs
            # stop continuation. A confirmed review pause needs an actor turn.
            harness.wait_stop_reason = (
                "internal_runtime_error"
                if internal_runtime_tripped(harness)
                else (harness.wait_stop_reason or "harness_stop_before_continuation")
            )
            return False
        async with workbench.control_lock:
            if not session.turn_done.is_set() or session.state != "completed":
                continue
            # Recheck native terminal, budgets and elapsed time after waiting for
            # control ownership; a reminder never overrides a supervisor stop.
            if not await wait_turn(session, shutdown, harness=harness):
                return False
            if (shutdown.is_set() or not harness.accepting or workbench.current_unknown_actions()
                    or harness.active_action or program_state()['blocked']):
                harness.wait_stop_reason = (
                    "internal_runtime_error"
                    if internal_runtime_tripped(harness)
                    else (harness.wait_stop_reason or "harness_stop_before_continuation")
                )
                return False
            previous_turn = session.turn_id
            context = dict(thread_id=session.thread_id, turn_id=previous_turn,
                           episode_id=workbench.identity.get("episode_id"))
            if bounded:
                activity = completed_turn_activity(workbench.store, **context)
                evidence_args = {}
                if evidence_aware:
                    from ...cortex.supervision.recovery_evidence import completed_turn_evidence
                    try:
                        budget = harness.action_budget()
                    except (AttributeError, TypeError, ValueError):
                        budget = {}
                    remaining = budget.get('native_steps_remaining') if isinstance(budget, dict) else None
                    if type(remaining) is not int or remaining < 0:
                        remaining = None
                    evidence_args['evidence'] = completed_turn_evidence(workbench.store, **context,
                        native_action_count=harness.native_action_count, native_steps_remaining=remaining)
                observation = stall.observe(**context, native_action_count=harness.native_action_count,
                    activity=activity, **evidence_args)
            else:
                final_answer = getattr(session, "completed_final_answer", lambda: None)()
                observation = stall.observe(**context,
                    native_action_count=harness.native_action_count, text=final_answer)
            if observation:
                harness.actor_stall = observation
                if bounded or observation["matched_reason"] or observation["triggered"]:
                    workbench.store.event("benchmark.actor_progress" if bounded else "benchmark.actor_inability",
                        observation, {"thread_id": session.thread_id, "turn_id": previous_turn})
            # Public evidence collection and logging can race native/controller callbacks
            # even on the baseline turn, before a stall could be triggered.
            if harness.episode_done.is_set() or model_budget_exhausted(session):
                continue
            if (shutdown.is_set() or not harness.accepting or workbench.current_unknown_actions()
                    or harness.active_action or program_state()['blocked']):
                harness.wait_stop_reason = (
                    "internal_runtime_error"
                    if internal_runtime_tripped(harness)
                    else (harness.wait_stop_reason or "harness_stop_before_continuation")
                )
                return False
            if observation and observation["triggered"]:
                harness.wait_stop_reason = ACTOR_PROGRESS_REASON if bounded else ACTOR_STALL_REASON
                return False
            count = getattr(harness, "actor_continuation_count", 0) + 1
            mode = getattr(workbench, "completion_guidance", "none")
            initial_observation_id = getattr(workbench, "initial_observation_id", None)
            home_check_result = None
            if mode in {"hook", "both"}:
                home_check = getattr(harness, "home_check", None)
                home_check_result = (home_check.check(workbench.identity) if home_check is not None
                                     else {"status": "unknown", "reason": "home_check_unavailable"})
                # Reset/recovery starts a new episode with a new reference.
                initial_observation_id = home_check_result.get("initial_observation_id")
            needs_home = home_check_result is not None and home_check_result["status"] == "away"
            continuation = hook_completion_guidance(initial_observation_id) if needs_home else BENCHMARK_CONTINUATION
            recovery_context = None
            if bounded:
                from ...cortex.supervision.recovery_guidance import recovery_guidance
                recovery_context, recovery_prompt = recovery_guidance(harness, home_check_result,
                    **({'evidence': stall.recovery_context()} if evidence_aware else {}))
                continuation += "\n" + recovery_prompt
            if harness.episode_done.is_set() or model_budget_exhausted(session):
                continue
            programs = program_state()
            if (shutdown.is_set() or not harness.accepting or workbench.current_unknown_actions()
                    or harness.active_action or programs['blocked']):
                harness.wait_stop_reason = (
                    "internal_runtime_error"
                    if internal_runtime_tripped(harness)
                    else (harness.wait_stop_reason or "harness_stop_before_continuation")
                )
                return False
            if programs['pending_reviews']:
                continuation += ("\nA program is paused for your action review. Use the pending_review evidence "
                    "to approve or reject with program_run(operation='resume', job_id=..., review_id=..., "
                    "decision='approve' or 'reject'). The broker executes an approved cached plan; do not "
                    "execute it separately. Pending reviews: " + canonical(programs['pending_reviews']))
            harness.actor_continuation_count = count
            workbench.store.event("benchmark.actor_turn_continuation", {
                "count": count, "previous_turn_id": previous_turn,
                "thread_id": session.thread_id, "reason": "actor_turn_completed_before_native_terminal",
                "completion_guidance": mode, "trigger": COMPLETION_GUIDANCE_TRIGGER, "prompt": continuation,
                "initial_observation_id": initial_observation_id,
                "continuation_kind": "home_reminder" if needs_home else "generic",
                "home_check": home_check_result,
                **({"recovery_context": recovery_context} if bounded else {})})
            # run_turn reuses the existing thread and ModelRequestProxy; neither
            # native progress clocks nor cumulative model budget are re-created.
            turn_id = await asyncio.to_thread(session.run_turn, continuation)
            record_settings = getattr(workbench, "record_turn_settings", None)
            if callable(record_settings):
                record_settings(turn_id, "benchmark_continuation")   # run.turn_settings (observer data)


def latest_observer_evaluation(store, identity, final_status):
    """返回与当前运行身份匹配且未被后续动作/重置失效的最新 ``observer.evaluated`` 评测文档，不可用时返回 None。"""
    if final_status == "unknown":
        return None
    evaluations = store.events_by_type(("observer.evaluated",), limit=1, newest_first=True)
    changes = store.events_by_type(
        ("action.requested", "environment.reset_requested", "environment.recovery_reset"),
        limit=1,
        newest_first=True,
    )
    if not evaluations or (changes and evaluations[0]["server_seq"] <= changes[0]["server_seq"]):
        return None
    ref = evaluations[0]["payload"].get("result_ref")
    if not ref:
        return None
    _, content = store.resolve_artifact(ref, allowed_scopes=("observer",))
    evaluation = json.loads(content)
    if any(evaluation.get(key) != identity[key] for key in IDENTITY):
        return None
    return {**evaluation, "audience": "observer", "result_ref": ref}


def observer_summary(store, identity, final_status):
    evaluation = latest_observer_evaluation(store, identity, final_status)
    if evaluation is None:
        return {}
    native = evaluation.get("result", {})
    return {key: native[key] for key in ("task_success", "normalized_score", "valid_episode",
            "native_reward", "native_process_score", "success_authority", "authority") if key in native}


# Settings tuple of run.turn_settings / runtime/agent_settings (apps/gateway/agent_settings.SETTING_FIELDS).
AGENT_SETTING_FIELDS = ("backend", "profile", "provider", "wire_api", "model", "effort", "service_tier",
                        "thinking_display")
# Live-change events of the operator console: event type -> (setting, payload key of the new value).
AGENT_SETTING_CHANGES = {"operator.model_changed": ("model", "model"), "operator.effort_changed": ("effort", "effort"),
                         "operator.service_tier_changed": ("service_tier", "tier")}


def agent_settings_summary(store):
    """``summary.agent_settings`` of a run, derived from its observer records (design 16 section 7.2):

    * ``initial``: the settings tuple (``AGENT_SETTING_FIELDS``) of the first host turn (``run.turn_settings``); None
      when no turn was admitted. ``final``: the current tuple of ``runtime/agent_settings`` at summary time (else the
      last turn's).
    * ``changes``: every applied live change (``operator.model_changed`` / ``effort_changed`` /
      ``service_tier_changed`` whose value differs from the previous one), in order, as ``{field, previous, value,
      applies, request_id, at (UTC), after_first_turn}``.
    * ``turns``: per host turn ``{turn_id, origin, model, effort, service_tier, served_models}``; ``served_models``
      counts the provider-reported model of every delivered completion (actor, helper and compaction calls) between
      this turn's admission and the next turn's admission or the summary (``runtime/model_requests.served_models``
      delta); None when the relay recorded no counter.
    * ``mixed``: True when the run is not a single-condition sample: the turns ran with different (model, effort,
      service tier) tuples, or a change that reaches the running session (not ``applies: next_session_start``) was
      applied after the first turn. A comparison must not treat a mixed run as one condition.
    """
    turns, changes, cursor = [], [], 0
    types = ("run.turn_settings", *AGENT_SETTING_CHANGES)
    while page := store.events_by_type(types, after=cursor, limit=1000):
        for event in page:
            payload = event.get("payload") or {}
            if event["type"] == "run.turn_settings":
                turns.append(payload)
                continue
            field, key = AGENT_SETTING_CHANGES[event["type"]]
            if payload.get("previous") == payload.get(key):
                continue
            changes.append(dict(field=field, previous=payload.get("previous"), value=payload.get(key),
                                applies=payload.get("applies"), request_id=payload.get("request_id"),
                                at=event.get("occurred_at"), after_first_turn=bool(turns)))
        cursor = page[-1]["server_seq"]
    final_served = (store.read_state("runtime", "model_requests") or {}).get("served_models")
    state = store.read_state("runtime", "agent_settings")

    def settings(value):
        return {key: value.get(key) for key in AGENT_SETTING_FIELDS} if isinstance(value, dict) else None

    def delta(before, after):
        if not isinstance(after, dict):
            return None
        before = before if isinstance(before, dict) else {}
        return {model: count - int(before.get(model) or 0) for model, count in after.items()
                if type(count) is int and count - int(before.get(model) or 0) > 0}

    rows = []
    for index, turn in enumerate(turns):
        after = turns[index + 1].get("served_models_at_admission") if index + 1 < len(turns) else final_served
        rows.append(dict(turn_id=turn.get("turn_id"), origin=turn.get("origin"), model=turn.get("model"),
                         effort=turn.get("effort"), service_tier=turn.get("service_tier"),
                         served_models=delta(turn.get("served_models_at_admission"), after)))
    distinct = {(row["model"], row["effort"], row["service_tier"]) for row in rows}
    mixed = len(distinct) > 1 or any(change["after_first_turn"] and change["applies"] != "next_session_start"
                                     for change in changes)
    return dict(initial=settings(turns[0]) if turns else None,
                final=settings(state) or (settings(turns[-1]) if turns else None),
                changes=changes, turns=rows, mixed=mixed)


SETTLE_IN_SUMMARY = object()   # episode_summary's default: settle the relay itself


def settle_model_records(session):
    """Let the session's model relay finish writing what it already delivered (``ModelRequestProxy.settle``, bounded
    by its ``SETTLE_TIMEOUT_S``): its callbacks write ``runtime/model_requests`` (``completed_calls``,
    ``served_models``), ``runtime/token_usage`` and ``runtime/model_usage_relay`` after the answer reached the agent,
    so they can trail a finished turn. Returns True when caught up at this moment, False when not within the bound
    (those records may then miss the last completions), None when the session has no relay (the external backend,
    or a session whose relay never started). Blocking: the gateway calls it off its event loop. Never raises."""
    settle = getattr(getattr(session, "model_proxy", None), "settle", None)
    if not callable(settle):
        return None
    try:
        return settle() is True
    except Exception:  # noqa: BLE001 - the summary is still written; False says the records may trail
        return False


def episode_summary(store, identity, session, harness, *, benchmark, final_status,
                    started_at, infrastructure_error=None, model_records_settled=SETTLE_IN_SUMMARY):
    """``runtime_protocol`` is the native action protocol of the run's embodiment (``harness.embodiment
    .limits_profiles[0]``: RoboDojo ``robodojo-native-v1``, SO-101 ``so101-deployment-v1``); a harness double
    without ``embodiment`` is the default ARX-X5 robot. ``model_records_settled``: the caller's
    ``settle_model_records(session)`` result when it already settled the relay (the gateway does, off its event
    loop); by default the relay is settled here, before its records are read."""
    embodiment = getattr(harness, "embodiment", None)
    if not isinstance(embodiment, EmbodimentProfile):
        from ...robot.model.embodiments import ARX_X5_PROFILE as embodiment
    if model_records_settled is SETTLE_IN_SUMMARY:   # before the relay-derived records below are read
        model_records_settled = settle_model_records(session)
    tripped = getattr(harness, "internal_runtime_tripped", None)
    if infrastructure_error is None and isinstance(tripped, dict):
        infrastructure_error = (
            f"internal_runtime_error: {tripped.get('tool', 'unknown')}: "
            f"{tripped.get('error_type', 'Exception')}: {tripped.get('message', '')}"
        )[:4096]
    elif infrastructure_error is None and getattr(harness, "wait_stop_reason", None) == "internal_runtime_error":
        infrastructure_error = "internal_runtime_error"
    receipts = store.records("action_receipt")
    reviews = []
    cursor = 0
    while page := store.events_by_type(("visual.model_reviewed",), after=cursor):
        reviews.extend(event["payload"] for event in page)
        cursor = page[-1]["server_seq"]
    result = {"action_count": harness.action_count,
              "completed_actions": sum(value.get("status") == "completed" for value in receipts),
              "failed_actions": sum(value.get("status") == "failed" for value in receipts),
              "task_success": None, "normalized_score": None, "thread_id": session.thread_id,
              "codex_turn_status": (session.turn_result or {}).get("status"),
              "agent_turn_status": (session.turn_result or {}).get("status"),
              "agent_backend": session_backend_kind(session),
              "token_usage": store.read_state("runtime", "token_usage"),
              "token_accounting": account_usage(store.read_state("runtime", "token_usage"),
                  model=session.model, provider=session.provider, price_card=session.price_card),
              "tool_calls": len(store.records("tool_call")),
              "elapsed_wall_s": time.monotonic() - started_at,
              "infrastructure_error": infrastructure_error,
              "runtime_protocol": embodiment.limits_profiles[0],
              "benchmark": benchmark,
              "actor_continuation_count": getattr(harness, "actor_continuation_count", 0),
              "actor_stall": getattr(harness, "actor_stall", None),
              "native_episode": store.read_state("runtime", "native_episode"),
              "model_requests": store.read_state("runtime", "model_requests"),
              # Provider-reported served model per delivered completion, and the relay-measured usage every backend
              # shares (None = not recorded); configured vs served and native vs relay usage are compared, not equated.
              "served_models": (store.read_state("runtime", "model_requests") or {}).get("served_models"),
              "model_usage_relay": store.read_state("runtime", "model_usage_relay"),
              "model_records_settled": model_records_settled,
              "model_budget_exhausted": model_budget_exhausted(session),
              "visual_review_usage": [dict(model=value["model"], usage=value.get("usage")) for value in reviews],
              "token_accounting_scope": "main_actor_only; auxiliary visual model usage recorded separately",
              "termination_reason": getattr(harness, "wait_stop_reason", None),
              # Backend, profile, model, effort and tier per turn and every live change (observer data only).
              "agent_settings": agent_settings_summary(store)}
    for key in getattr(session, "summary_null_fields", ()):
        # Not measurable on this backend (external: no relay, so no served model or token usage): null, never 0.
        result[key] = None
    result.update(observer_summary(store, identity, final_status))
    return classify_episode_result(result, final_status)

def record_evaluation(store, identity, result):
    evidence = {**identity, "result": result}
    reference = store.put_artifact(canonical(evidence).encode(), "native_evaluation", "application/json", "observer")
    store.event("observer.evaluated", {"result_ref": reference}, source="private_evaluator")
    return {**evidence, "audience": "observer", "result_ref": reference}


def record_video(store, identity, videos):
    result = {**identity, **videos}
    for item in result.get("videos", []):
        item["video_ref"] = store.put_artifact(Path(item["video_path"]).read_bytes(), "native_video", "video/mp4", "observer")
    composite = result.get("composite")
    if isinstance(composite, dict) and composite.get("video_path") and Path(composite["video_path"]).is_file():
        composite["video_ref"] = store.put_artifact(Path(composite["video_path"]).read_bytes(), "native_video", "video/mp4", "observer")
    store.event("observer.video_exported", result, source="private_evaluator")
    return result
