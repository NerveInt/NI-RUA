"""Model-context epoch, delivery baselines, and post-compaction decision-reminder bridge."""

from __future__ import annotations

import threading
from typing import Any

from ..core.contracts import new_id
from ..core.presentation.capability_output import CapabilityDelivery
from ..core.presentation.observation_output import MetadataDelivery
from ..core.presentation.program_output_delivery import ProgramOutputDelivery
from ..cortex.agents.identity import read_agent_identity
from ..cortex.context.task_planning import decision_reminder_text

# Decision-context reminder channels (design 04 §4.5); recorded on model_context.* events.
REMINDER_IN_COMPACT_REQUEST = "compact_request_assistant_message"
REMINDER_IN_TOOL_RESULT = "post_compaction_tool_result"


class DecisionContextBridge:
    """Owns model-context epoch tracking, delivery deduplication baselines, and decision reminders."""

    def __init__(self, harness: Any) -> None:
        self._harness = harness
        self.delivery_channel: str = "codex_tool"
        self.context_epoch_volatile: bool = False
        self.decision_reminder_channel: str = REMINDER_IN_COMPACT_REQUEST
        self.decision_reminder_due: bool = False
        self._decision_reminder_lock = threading.Lock()
        self._view_lock = threading.Lock()
        self.last_delivered_views: dict[str, Any] = {}
        self.view_context_epoch: Any = None
        self.capability_delivery = CapabilityDelivery()
        self.program_output_delivery = ProgramOutputDelivery()
        self.metadata_delivery = MetadataDelivery(harness.embodiment)

    def reset_metadata_delivery(self) -> None:
        self.metadata_delivery = MetadataDelivery(self._harness.embodiment)

    def model_context_epoch(self) -> tuple[Any, ...]:
        h = self._harness
        proxy = getattr(h, "model_proxy", None)
        epoch: tuple[Any, ...] = (
            dict(h.identity),
            h.thread_id,
            h.store.read_state("runtime", "image_context_epoch"),
            (read_agent_identity(h.store) or {}).get("instance_id"),
            getattr(proxy, "transport_id", None),
            getattr(proxy, "image_context_epoch", None),
        )
        if self.context_epoch_volatile:
            epoch += (new_id("volatile-context"),)
        return epoch

    def live_context_epoch(self) -> tuple[Any, ...]:
        """读取当前活跃模型上下文纪元（兼容测试桩覆写 ``_model_context_epoch``）。"""
        epoch_fn = getattr(self._harness, "_model_context_epoch", None)
        return epoch_fn() if callable(epoch_fn) else self.model_context_epoch()

    def actor_delivery_epoch(self, correlation: dict[str, Any] | None = None) -> tuple[Any, ...] | None:
        """返回主链调用可用于跨调用去重的上下文纪元；易变模式或非主链作用域返回 ``None``。"""
        if self.context_epoch_volatile:
            return None
        corr = correlation or {}
        if corr.get("context_scope", "main") != "main":
            return None
        bound = corr.get("context_epoch")
        return bound if bound is not None else self.live_context_epoch()

    def sync_view_epoch(self, epoch: Any = None) -> tuple[Any, dict[str, Any]]:
        live_epoch = self.live_context_epoch()
        target = live_epoch if epoch is None else epoch
        if target != live_epoch:
            return target, {}
        with self._view_lock:
            if target != self.view_context_epoch:
                self.last_delivered_views.clear()
                self.view_context_epoch = target
            return target, dict(self.last_delivered_views)

    def commit_delivered_views(
        self,
        epoch: Any,
        baselines: dict[str, Any],
        failed_cameras: list[str] | tuple[str, ...] = (),
    ) -> bool:
        """仅在绑定纪元仍为当前活跃纪元且未进入易变模式时写回视图基线。"""
        if epoch is None or (not baselines and not failed_cameras):
            return False
        with self._view_lock:
            if self.context_epoch_volatile or epoch != self.live_context_epoch() or epoch != self.view_context_epoch:
                return False
            for camera_id in failed_cameras:
                self.last_delivered_views.pop(camera_id, None)
            self.last_delivered_views.update(baselines)
            return True

    def model_context_reset(self, reason: Any) -> None:
        norm_reason = reason if isinstance(reason, str) and reason else "unspecified"
        sideband = self.decision_reminder_channel == REMINDER_IN_TOOL_RESULT
        if sideband:
            with self._decision_reminder_lock:
                self.decision_reminder_due = True
        try:
            self._harness.store.event(
                "model_context.reset",
                dict(
                    reason=norm_reason,
                    decision_reminder_channel=self.decision_reminder_channel,
                    decision_reminder_due=sideband,
                ),
            )
        except Exception:
            pass

    def deliver_decision_reminder(self, response: Any, correlation: dict[str, Any] | None) -> Any:
        if (correlation or {}).get("context_scope", "main") != "main":
            return response
        if not self.decision_reminder_due:
            return response
        if not isinstance(response, dict) or not isinstance(response.get("contentItems"), list):
            return response
        h = self._harness
        failure = None
        context = None
        text = None
        with self._decision_reminder_lock:
            if not self.decision_reminder_due:
                return response
            try:
                context = h.task_planning.decision_context(h.store.identity)
                text = (
                    None
                    if context is None and not h.task_planning.required
                    else h.actor_presentation.present_text(decision_reminder_text(context))
                )
            except Exception as error:
                failure = type(error).__name__
            else:
                self.decision_reminder_due = False
        if failure is not None:
            try:
                h.store.event("model_context.decision_reminder_failed", dict(error_type=failure), correlation)
            except Exception:
                pass
            return response
        if text is not None:
            response = dict(response, contentItems=[*response["contentItems"], dict(type="inputText", text=text)])
        try:
            h.store.event(
                "model_context.decision_reminder_delivered",
                dict(included=text is not None, plan_revision=(context or {}).get("revision")),
                correlation,
            )
        except Exception:
            pass
        return response

    def invalidate_actor_delivery(self, call_id: str | None) -> None:
        self.program_output_delivery.invalidate()
        self._harness.store.set_state(
            "runtime",
            "image_context_epoch",
            dict(
                source="tool_output_failed",
                tool_call_id=call_id,
                marker=new_id("tool-output-failure"),
            ),
        )
