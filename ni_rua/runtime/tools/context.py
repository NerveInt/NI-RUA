"""Typed tool execution ports; physical execution is gated by declared ToolSpec effect."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable

from ...core.contracts import digest


@dataclass(frozen=True)
class ToolAvailability:
    features: object
    disabled_tools: frozenset
    observation_profile: str
    capabilities: dict | None
    enable_wait: bool
    vision_compare: bool
    visual_review: bool
    policy: bool
    programs: bool
    embodiment: object = None

    @classmethod
    def from_runtime(cls, runtime):
        return cls(
            runtime.config.features,
            frozenset(runtime.disabled_tools),
            runtime.observation_profile,
            deepcopy(runtime.capabilities),
            runtime.enable_wait,
            runtime.vision_compare,
            runtime.visual_reviewer is not None,
            runtime.action_policy is not None,
            runtime.programs is not None,
            getattr(runtime, "embodiment", None),
        )


@dataclass(frozen=True)
class ObservationPort:
    """Live observation read/capture/enrichment port for tool handlers."""

    latest: Callable[[], dict | None]
    observe: Callable[..., dict]
    content: Callable[..., Any]
    enrich: Callable[..., Any] | None = None


@dataclass(frozen=True)
class RecordPort:
    """Scoped archive read/write port without exposing raw Store internals directly on ToolContext."""

    read: Callable[[str, str], Any]
    save: Callable[[str, str, Any], Any]
    put_artifact: Callable[..., str]
    resolve_artifact: Callable[..., tuple[dict, bytes]]
    event: Callable[..., Any]


class _RuntimePortProxy:
    """Effect-enforcing runtime collaborator port for built-in handlers.

    Non-physical tools (``effect != 'physical'``) cannot invoke ``execute`` or ``policy_step`` even accidentally.
    """

    def __init__(self, runtime: Any, *, name: str, effect: str) -> None:
        object.__setattr__(self, "_runtime", runtime)
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_effect", effect)

    def __getattr__(self, attr: str) -> Any:
        if attr in ("execute", "policy_step") and self._effect != "physical":
            raise PermissionError(f"tool_effect_forbids_physical_execution:{self._name}:{self._effect}")
        return getattr(self._runtime, attr)

    def __setattr__(self, attr: str, value: Any) -> None:
        setattr(self._runtime, attr, value)

    def __delattr__(self, attr: str) -> None:
        delattr(self._runtime, attr)


@dataclass(frozen=True)
class ToolContext:
    identity: dict
    observation: dict | None
    capabilities: dict | None
    action_budget: dict
    read_action: Callable
    execute: Callable | None = None
    effect: str = "read"
    embodiment: object = None
    observations: ObservationPort | None = None
    records: RecordPort | None = None
    services: _RuntimePortProxy | None = None

    @classmethod
    def from_runtime(cls, runtime, name, effect, correlation):
        correlation = deepcopy(correlation)
        sequence = 0

        def execute(action):
            nonlocal sequence
            # A repeated tool invocation owns the same ordered action IDs.
            # Harness then enforces immutable content, admission and no replay
            # of an uncertain acknowledgement, exactly as for built-in actions.
            if not correlation.get("tool_call_id"):
                raise ValueError("plugin_physical_action_requires_tool_call_id")
            identifier = "action-plugin-" + digest([name, correlation, sequence])[:24]
            sequence += 1
            action = {**action, "action_id": identifier}
            return runtime.execute(action, correlation)

        pipeline = getattr(runtime, "observation_pipeline", None)
        obs_port = (
            ObservationPort(
                latest=lambda: deepcopy(getattr(runtime, "latest", None)),
                observe=runtime.observe,
                content=runtime.observation_content,
                enrich=pipeline.enrich if pipeline is not None else None,
            )
            if hasattr(runtime, "observe") and hasattr(runtime, "observation_content")
            else None
        )
        store = getattr(runtime, "store", None)
        rec_port = (
            RecordPort(
                read=store.read,
                save=store.save,
                put_artifact=store.put_artifact,
                resolve_artifact=store.resolve_artifact,
                event=store.event,
            )
            if store is not None
            else None
        )
        services = _RuntimePortProxy(runtime, name=name, effect=effect)
        return cls(
            deepcopy(runtime.identity),
            deepcopy(runtime.latest),
            deepcopy(runtime.capabilities),
            deepcopy(runtime.action_budget()),
            lambda identifier: deepcopy(runtime.action(identifier)),
            execute if effect == "physical" else None,
            effect=effect,
            embodiment=getattr(runtime, "embodiment", None),
            observations=obs_port,
            records=rec_port,
            services=services,
        )


def runtime_port(context_or_runtime: Any) -> Any:
    """Extract the effect-checked runtime collaborator port from ``ToolContext`` (or pass test doubles through)."""
    if isinstance(context_or_runtime, ToolContext):
        if context_or_runtime.services is None:
            raise ValueError("tool_context_services_unavailable")
        return context_or_runtime.services
    return context_or_runtime
