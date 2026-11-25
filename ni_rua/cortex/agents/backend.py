"""Unified agent backend contracts: descriptors, traits, hooks/deps injection, profile sections, and transports."""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any, Callable


class _Hook:
    """Slot descriptor on ``AgentSession`` with a default value and zero ``__new__`` instance state.

    Keeps ``vars(AgentSession.__new__(AgentSession)) == {}`` for legacy test fakes while letting callers
    bind hooks via ``SessionHooks`` or normal attribute assignment.
    """

    def __init__(self, default=None):
        self.default = default
        self.name = None

    def __set_name__(self, owner, name):
        self.name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self.default
        return instance.__dict__.get(self.name, self.default)

    def __set__(self, instance, value):
        instance.__dict__[self.name] = value

    def __delete__(self, instance):
        instance.__dict__.pop(self.name, None)


@dataclass(frozen=True)
class SessionHooks:
    """Explicit callbacks wired onto an ``AgentSession`` after construction."""

    on_disconnect: Callable[[], None] | None = None
    transport_recovery_gate: Callable[[], str | None] | None = None
    tool_parallel_safe: Callable[[str, dict], bool] | None = None
    tool_parallel_hints: dict | None = None
    tool_effect: Callable[[str, dict], str | None] | None = None
    on_context_reset: Callable[[str], None] | None = None


@dataclass(frozen=True)
class SessionDeps:
    """Core runtime dependencies supplied to an ``AgentSession`` implementation."""

    run_dir: Any
    store: Any
    tool_handler: Callable
    config: Any = None
    price_card: Any = None
    task_planning: Any = None
    embodiment: Any = None


def bind_session_hooks(session, hooks: SessionHooks) -> None:
    """Attach non-``None`` callbacks from ``hooks`` without touching session-owned fields like ``turn_done``."""
    for field_name in (
        "on_disconnect",
        "transport_recovery_gate",
        "tool_parallel_safe",
        "tool_parallel_hints",
        "tool_effect",
        "on_context_reset",
    ):
        value = getattr(hooks, field_name)
        if value is not None:
            setattr(session, field_name, value)


@dataclass(frozen=True)
class ProfileSection:
    """Backend-owned ``backends.<kind>`` schema & config-mapping adapter inside provider profiles."""

    key: str
    section_fields: tuple[str, ...]
    parse: Callable[[dict, list, str, dict], dict]
    serialize: Callable[[Any], dict | None]
    public_view: Callable[[Any], dict]
    forwarded_env: Callable[[Any], dict]
    expand_config: Callable[[Any, Any, Any, Any], tuple[str, Any]]
    apply_config: Callable[[Any, str, Any, Any], dict]
    collect_hints: Callable[[dict, dict, str], None]


@dataclass(frozen=True)
class BackendTraits:
    """Static + lazy-runtime traits of one registered agent backend."""

    kind: str
    hosted: bool
    wires: tuple[str, ...]
    native_effort_choices: tuple[str, ...] | None = None
    live_effort_default: bool = False
    service_tier: bool = False
    relay_compat_fields: bool = True
    evidence_native: bool = False
    upstream_config_attr: str | None = None
    capabilities_loader: Callable[[], Any] | None = None
    effort_choices_loader: Callable[[str], tuple[str, ...]] | None = None
    thinking_display_loader: Callable[[str], tuple[str, ...]] | None = None


@dataclass(frozen=True)
class AgentBackend:
    """Registered backend entry uniting traits, profile section, launch gating, and session creation."""

    traits: BackendTraits
    profile_section: ProfileSection | None = None
    requirements_checker: Callable[[Any], None] | None = None
    session_creator: Callable[..., Any] | None = None
    upstream_accessor: Callable[[Any], Any] | None = None

    @property
    def kind(self) -> str:
        return self.traits.kind

    def capabilities(self):
        if self.traits.capabilities_loader is None:
            from .base import AgentCapabilities

            return AgentCapabilities()
        return self.traits.capabilities_loader()

    def effort_choices(self, wire_api: str) -> tuple[str, ...]:
        if self.traits.effort_choices_loader is not None:
            return self.traits.effort_choices_loader(wire_api)
        if self.traits.native_effort_choices is not None:
            return self.traits.native_effort_choices
        from .base import model_effort_choices

        return model_effort_choices()

    def thinking_display_choices(self, wire_api: str) -> tuple[str, ...]:
        if self.traits.thinking_display_loader is not None:
            return self.traits.thinking_display_loader(wire_api)
        return ()


class AgentTransport:
    """Base transport contract shared by App-Server RPC and host MCP loopback transports."""

    transport_kind = "unknown"

    def close(self) -> None:
        raise NotImplementedError


class AppServerTransport(AgentTransport):
    """Run-owned Unix socket IPC directory + app-server subprocess lifecycle for Codex."""

    transport_kind = "app_server"

    def __init__(self, root, *, longest: str = "codex.sock"):
        from .isolation import resolve_ipc_layout

        self.root = root
        self.longest = longest
        self.layout = resolve_ipc_layout(root, prefix="codex", longest=longest)
        self.socket_path = self.layout.ipc / "codex.sock"
        self.prepared = False

    def prepare(self) -> None:
        from .isolation import ensure_ipc_dir

        ensure_ipc_dir(self.layout, prefix="codex", require_runtime_dir_private=True)
        self.prepared = True

    def close(self) -> None:
        if self.prepared:
            self.socket_path.unlink(missing_ok=True)
            try:
                self.layout.ipc.rmdir()
            except OSError:
                pass
            self.prepared = False


class McpLoopbackTransport(AgentTransport):
    """Host-side HTTP/SSE MCP loopback endpoint owned by MCP-backed agent sessions."""

    transport_kind = "mcp_loopback"

    def __init__(self, bridge):
        self.bridge = bridge

    @property
    def url(self):
        return self.bridge.url

    @property
    def headers(self):
        return self.bridge.headers

    def close(self) -> None:
        if self.bridge is not None:
            self.bridge.stop_accepting()
            self.bridge.join()
