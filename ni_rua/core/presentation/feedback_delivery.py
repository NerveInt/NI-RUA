"""Model-facing feedback composition; execution and evidence remain untouched."""

ACTION_IMAGE_POLICIES = ("on_anomaly", "none", "focus_roi", "head_only", "adaptive", "all")

_trajectory_feedback_formatter = None


def register_trajectory_feedback(fn):
    """Register `(harness, action, receipt, correlation, *, diagnostic=False) -> (summary, images)`."""
    global _trajectory_feedback_formatter
    _trajectory_feedback_formatter = fn


def resolve_action_image_policy(harness, action=None):
    """解析动作后自动观测的有效图像交付策略（显式请求参数优先，兼容旧 adaptive 配置）。"""
    explicit = action.get("image_delivery") if isinstance(action, dict) else None
    if explicit in ACTION_IMAGE_POLICIES:
        return explicit
    settings = getattr(harness, "image_settings", None) or {}
    if settings.get("feedback_policy", "all") == "adaptive":
        return "adaptive"
    configured = settings.get("action_feedback_policy")
    if configured in ACTION_IMAGE_POLICIES:
        return configured
    images_cfg = getattr(getattr(harness, "config", None), "images", None)
    cfg_policy = getattr(images_cfg, "action_feedback_policy", None)
    if cfg_policy in ACTION_IMAGE_POLICIES:
        return cfg_policy
    return "all"


def is_clean_completed_receipt(receipt):
    """仅依据契约级客观回执状态判定动作是否正常完成（严禁引入夹爪开度或几何魔法阈值）。"""
    return (
        isinstance(receipt, dict)
        and receipt.get("status") == "completed"
        and not execution_exception(receipt)
    )


# Why a batch stopped at a checkpoint is typed by ``receipt.checkpoint_reason_code`` (robot/motion/batch_scheduler
# ``pause_receipt``; every pause has ``status="stopped"``, ``reason="observation_checkpoint"`` and
# ``paused_at_checkpoint=true``, so those three do not tell a requested stop from a guard):
#
# | checkpoint_reason_code            | Written when                                                    | Class           |
# |-----------------------------------|-----------------------------------------------------------------|-----------------|
# | observation_checkpoint            | the batch reached an ``observe`` stage the model put there      | model-requested |
# | jaw_checkpoint                    | ``checkpoint_after_gripper`` paused after a jaw stage           | model-requested |
# | aperture_below_min_m              | grip guard: measured aperture under the stage's minimum         | guard           |
# | aperture_above_max_m              | grip guard: measured aperture over the stage's maximum          | guard           |
# | carry_slip_aperture_below_min_m   | carry slip guard during a later move                            | guard           |
# | any other value, or no code       | a guard code this table does not know, or an older receipt      | guard           |
#
# A stop that carries both (an ``observe`` stage reached while a guard triggered) has the guard's code.
MODEL_REQUESTED_CHECKPOINT_CODES = frozenset({"observation_checkpoint", "jaw_checkpoint"})


def is_requested_checkpoint_receipt(receipt):
    """True for a batch that stopped where the model asked it to and reports nothing else: ``status="stopped"`` at a
    checkpoint (``paused_at_checkpoint``) whose ``checkpoint_reason_code`` is model-requested (table above), batch
    ``awaiting_observation``, stop confirmed, and none of ``error``, ``slip_detected_at``, ``effective_grip_guard``,
    an invalid ``resume_validation`` or an arm/gripper result with an exception. Typed receipt facts only."""
    if not isinstance(receipt, dict):
        return False
    validation = receipt.get("resume_validation")
    return (
        receipt.get("status") == "stopped"
        and receipt.get("reason") == "observation_checkpoint"
        and receipt.get("paused_at_checkpoint") is True
        and receipt.get("checkpoint_reason_code") in MODEL_REQUESTED_CHECKPOINT_CODES
        and receipt.get("batch_status", "awaiting_observation") == "awaiting_observation"
        and not receipt.get("error")
        and not receipt.get("slip_detected_at")
        and not receipt.get("effective_grip_guard")
        and not (isinstance(validation, dict) and validation.get("status") != "valid")
        and not execution_unconfirmed(receipt)
        and not any(execution_exception(arm) for field in ("arm_results", "gripper_execution")
                    if isinstance(receipt.get(field), dict) for arm in receipt[field].values())
    )


def is_unexecuted_stage(entry):
    """A stage entry that never ran (``status="pending"``) and reports nothing: the stages after a stop."""
    return isinstance(entry, dict) and entry.get("status") == "pending" and not execution_exception(entry)


def action_trajectory_feedback(harness, action, receipt, correlation, *, diagnostic=False):
    if correlation.get("program_id") or not harness.config.features.trajectory_preview:
        return None, []
    try:
        if _trajectory_feedback_formatter is None:
            raise RuntimeError("trajectory_feedback_formatter_not_registered")
        if diagnostic:
            return _trajectory_feedback_formatter(harness, action, receipt, correlation, diagnostic=True)
        return _trajectory_feedback_formatter(harness, action, receipt, correlation)
    except Exception as error:
        # Optional rendering after execution must not turn a completed action
        # into a tool exception that invites accidental re-execution.
        return dict(available=False, reason="trajectory_feedback_failed:" + type(error).__name__,
                    action_id=action.get("action_id"), execution_status=receipt.get("status")), []


def execution_unconfirmed(value):
    """Keep unknown execution and negative stop acknowledgements authoritative."""
    if not isinstance(value, dict):
        return False
    if (value.get("status") in ("unknown", "status_unknown") or value.get("execution_uncertain") is True
            or value.get("stop_confirmed") is False or value.get("stop_confirmation") is False):
        return True
    confirmation = value.get("stop_confirmation")
    if isinstance(confirmation, dict) and confirmation.get("confirmed") is False:
        return True
    return any(execution_unconfirmed(arm) for field in ("arm_results", "gripper_execution", "stage_results")
               if isinstance(value.get(field), dict) for arm in value[field].values())


def execution_exception(value):
    """Public uncertainty flags override automatic image omission."""
    if not isinstance(value, dict):
        return False
    if value.get("reason") or value.get("error") or execution_unconfirmed(value):
        return True
    return any(execution_exception(arm) for field in ("arm_results", "gripper_execution")
               if isinstance(value.get(field), dict) for arm in value[field].values())
