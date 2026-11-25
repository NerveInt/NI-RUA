"""Agent backend selection: session construction and launch gating (delegating to ``agents.registry``)."""

from ...core.config import ConfigError
from .base import AgentCapabilityError
from .external.protocol import BACKEND_KIND
from .registry import (
    BACKENDS_REGISTRY,
    actor_upstream,
    agent_backend,
    claude_code_runtime_problem,
    external_port_problem,
    get_backend,
    opencode_runtime_problem,
)

# Backends NI-RUA launches itself (hosted agents).
BACKENDS = BACKENDS_REGISTRY.hosted_kinds()
EXTERNAL_BACKEND = BACKEND_KIND
BACKEND_WIRES = BACKENDS_REGISTRY.backend_wires()
NATIVE_EFFORT_CHOICES = BACKENDS_REGISTRY.native_effort_choices()

EVIDENCE_CLIENT_CODEX_AGENT = "codex_agent"
EVIDENCE_CLIENT_RELAY = "relay"
OPENCODE_WIRES = BACKEND_WIRES["opencode"]


def evidence_client(config, backend=None):
    """Return effective evidence helper client (``codex_agent`` vs ``relay``) for ``config``."""
    kind = agent_backend(config) if backend is None else backend
    setting = getattr(getattr(getattr(config, "agent", None), "evidence", None), "client", "native")
    try:
        native = get_backend(kind).traits.evidence_native
    except AgentCapabilityError:
        native = False
    return EVIDENCE_CLIENT_CODEX_AGENT if native and setting == "native" else EVIDENCE_CLIENT_RELAY


def backend_requirements(config):
    """Refuse a launch ``config`` the selected agent backend cannot run; return None when valid."""
    kind = agent_backend(config)
    try:
        backend = get_backend(kind, error_cls=ConfigError)
    except ConfigError:
        raise ConfigError(
            f"unknown_agent_backend:{kind}: set agent.backend to one of {', '.join(BACKENDS)}"
        ) from None
    return backend.requirements_checker(config)


def create_session(config, run_dir, store, tool_handler, *, price_card=None, task_planning=None):
    """Construct (without starting) the configured backend's ``AgentSession`` for one run."""
    kind = agent_backend(config)
    backend = get_backend(kind)
    return backend.session_creator(
        run_dir,
        store,
        tool_handler,
        price_card=price_card,
        config=config,
        task_planning=task_planning,
    )
