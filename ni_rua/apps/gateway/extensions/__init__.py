"""Pluggable capability extensions mounted onto the Gateway aiohttp application."""
from __future__ import annotations

from typing import Any, Protocol, runtime_checkable


@runtime_checkable
class CapabilityRouter(Protocol):
    """Contract for optional gateway capability extensions mounted onto ``Workbench.app()``."""

    name: str

    def supports(self, workbench: Any) -> bool:
        """Return True when this capability extension should be mounted for ``workbench``."""
        ...

    def mount(self, app: Any) -> None:
        """Mount this extension's HTTP routes onto ``app``."""
        ...


__all__ = [
    "AgentSettingsExtension",
    "CalibrationExtension",
    "CapabilityRouter",
    "HandoffExtension",
    "OperatorConsoleExtension",
]


def __getattr__(name: str):
    if name == "OperatorConsoleExtension":
        from .operator_console import OperatorConsoleExtension
        return OperatorConsoleExtension
    if name == "CalibrationExtension":
        from .calibration import CalibrationExtension
        return CalibrationExtension
    if name == "AgentSettingsExtension":
        from .agent_settings import AgentSettingsExtension
        return AgentSettingsExtension
    if name == "HandoffExtension":
        from .handoff import HandoffExtension
        return HandoffExtension
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
