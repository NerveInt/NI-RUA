"""Single writer and reader for backend-neutral run agent identity (``runtime/agent`` + ``runtime/codex`` fallback)."""

from __future__ import annotations

_UNSET = object()


def _codex_identity(codex):
    """Neutral identity view of a ``runtime/codex`` record (pre-change archives and Codex mirror)."""
    identity = dict(backend="codex", wire_api="openai_responses", source="runtime/codex")
    for target, key in (
        ("thread_id", "thread_id"),
        ("instance_id", "app_server_instance_id"),
        ("pid", "app_server_pid"),
        ("model", "model"),
        ("provider", "provider"),
        ("effort", "effort"),
        ("isolation", "isolation"),
        ("tools_digest", "tools_digest"),
        ("sdk_version", "sdk_version"),
        ("cli_version", "cli_version"),
        ("cli_executable", "cli_executable"),
    ):
        if key in codex:
            identity[target] = codex[key]
    if "socket_uri" in codex:
        identity["native"] = dict(socket_uri=codex["socket_uri"])
    return identity


def read_agent_identity(store):
    """Return the run's recorded agent identity, synthesizing from ``runtime/codex`` when needed."""
    agent = store.read_state("runtime", "agent")
    agent = agent if isinstance(agent, dict) else None
    if agent is not None and agent.get("backend") != "codex":
        return agent
    codex = store.read_state("runtime", "codex")
    if not isinstance(codex, dict) or not codex:
        return agent
    if (
        agent is not None
        and agent.get("instance_id") == codex.get("app_server_instance_id")
        and agent.get("thread_id") == codex.get("thread_id")
    ):
        return agent
    return _codex_identity(codex)


def build_agent_identity(
    *,
    backend,
    thread_id,
    instance_id,
    pid=None,
    model=None,
    provider=None,
    effort=None,
    isolation=None,
    tools_digest=None,
    instructions_digest=None,
    counting=None,
    wire_api=None,
    sdk_version=_UNSET,
    cli_version=_UNSET,
    cli_executable=_UNSET,
    cli_sha256=_UNSET,
    native=_UNSET,
    mcp=_UNSET,
    resumed=_UNSET,
    clean_shutdown=_UNSET,
):
    """Build the canonical ``runtime/agent`` identity dict shared across all agent backends."""
    record = dict(
        backend=backend,
        thread_id=thread_id,
        instance_id=instance_id,
        pid=pid,
        model=model,
        provider=provider,
        effort=effort,
        isolation=isolation,
        tools_digest=tools_digest,
        instructions_digest=instructions_digest,
        counting=counting,
        wire_api=wire_api,
    )
    for key, value in (
        ("sdk_version", sdk_version),
        ("cli_version", cli_version),
        ("cli_executable", cli_executable),
        ("cli_sha256", cli_sha256),
        ("native", native),
        ("mcp", mcp),
        ("resumed", resumed),
        ("clean_shutdown", clean_shutdown),
    ):
        if value is not _UNSET:
            record[key] = value
    return record


def write_identity(store, **fields):
    """Persist ``runtime/agent`` identity (preserving ``codex_item`` transcript schema) and return it."""
    record = build_agent_identity(**fields)
    store.set_state("runtime", "agent", record)
    return record


def write_codex_identity(store, codex_record):
    """Persist Codex's authoritative ``runtime/codex`` mirror for legacy resume and archive compatibility."""
    store.set_state("runtime", "codex", dict(codex_record))


def update_identity_shutdown(store, identity, *, clean_shutdown):
    """Record ``clean_shutdown`` on an existing ``runtime/agent`` identity record."""
    updated = dict(identity, clean_shutdown=clean_shutdown)
    store.set_state("runtime", "agent", updated)
    return updated
