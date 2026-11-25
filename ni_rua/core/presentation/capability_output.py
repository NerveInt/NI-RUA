"""Reuse immutable robot capabilities within one model context.

Live execution budgets remain explicit. Programs and explicit full reads do not
consume the model's delivery baseline, and any changed capability is resent.
"""

from copy import deepcopy
import threading

from ..contracts import digest


class CapabilityDelivery:
    def __init__(self):
        self._lock = threading.Lock()
        self._baseline: tuple | None = None

    def project(self, capabilities, *, epoch, current_action_budget, compact_inline=False, live_epoch=None):
        version = digest(capabilities)
        if epoch is None or (live_epoch is not None and epoch != live_epoch):
            result = deepcopy(capabilities)
            if compact_inline:
                result["capabilities_delivery"] = dict(status="inline")
            else:
                result["capabilities_delivery"] = dict(
                    version=version,
                    status="inline",
                    scope="current_model_context",
                    read_full=dict(tool="robot_capabilities", detail="full"),
                )
            result["current_action_budget"] = deepcopy(current_action_budget)
            return result
        with self._lock:
            repeated = self._baseline == (epoch, version)
            if not repeated:
                self._baseline = (deepcopy(epoch), version)
        result = {} if repeated else deepcopy(capabilities)
        # 显式紧凑内联或工具调用回复投影时仅保留最小状态标记，避免摘要反超 detail="full" 体积
        if not repeated and compact_inline:
            result["capabilities_delivery"] = dict(status="inline")
        else:
            result["capabilities_delivery"] = dict(
                version=version,
                status="unchanged" if repeated else "inline",
                scope="current_model_context",
                read_full=dict(tool="robot_capabilities", detail="full"),
            )
        result["current_action_budget"] = deepcopy(current_action_budget)
        return result
