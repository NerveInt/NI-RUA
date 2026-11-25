"""Explicit central registry of NI-RUA agent backends, traits, profile sections, and session factories."""

from __future__ import annotations

from dataclasses import replace
import os
from pathlib import Path
import tempfile

from ...core.config import ConfigError, ModelConfig, OpenCodeConfig, UpstreamConfig
from .backend import AgentBackend, BackendTraits, ProfileSection
from .base import AgentCapabilityError
from .external.protocol import BACKEND_KIND as EXTERNAL_BACKEND_KIND

_CLAUDE_EFFORTS = ("low", "medium", "high", "xhigh", "max")


def _lazy_codex_capabilities():
    from .codex.session import CodexSession

    return CodexSession.capabilities


def _lazy_claude_capabilities():
    from .claude_code.session import ClaudeCodeSession

    return ClaudeCodeSession.capabilities


def _lazy_opencode_capabilities():
    from .opencode.session import OpenCodeSession

    return OpenCodeSession.capabilities


def _lazy_opencode_efforts(wire_api: str) -> tuple[str, ...]:
    from .opencode import config as opencode_config

    return tuple(opencode_config.EFFORT_CHOICES.get(wire_api, ()))


def _lazy_codex_efforts(_wire_api: str) -> tuple[str, ...]:
    from .codex.session import CodexSession

    return tuple(CodexSession.effort_choices)


def _lazy_claude_efforts(_wire_api: str) -> tuple[str, ...]:
    from .claude_code.session import ClaudeCodeSession

    return tuple(ClaudeCodeSession.effort_choices)


def _lazy_opencode_thinking_display(wire_api: str) -> tuple[str, ...]:
    from .opencode import config as opencode_config

    try:
        opencode_config.effort_variants(wire_api, thinking_display=opencode_config.THINKING_DISPLAY_CHOICES[0])
        return tuple(opencode_config.THINKING_DISPLAY_CHOICES)
    except opencode_config.OpenCodeConfigError:
        return ()


def _codex_actor_upstream(config):
    return getattr(getattr(config, "agent", None), "codex_upstream", None)


def _sub_actor_upstream(kind: str):
    def _get(config):
        return getattr(getattr(getattr(config, "agent", None), kind, None), "upstream", None)

    return _get


def _check_codex_requirements(config):
    upstream = _codex_actor_upstream(config)
    if upstream is not None:
        _check_profile_requirements(config, "codex")
    return None


def _check_profile_requirements(config, kind: str) -> None:
    from .profiles import resolve_profile_upstream

    backend = get_backend(kind)
    upstream = backend.upstream_accessor(config)
    try:
        resolve_profile_upstream(upstream, kind)
    except ValueError as error:
        raise ConfigError(str(error)) from None


def _check_hosted_shared_requirements(config, kind: str) -> str | None:
    features, model = config.features, config.model
    backend = get_backend(kind)
    upstream = backend.upstream_accessor(config)
    actor_wire = None
    if upstream is not None:
        from .opencode.upstream_file import configured_wire

        actor_wire = configured_wire(upstream)
    if (
        features.visual_review_model
        and config.agent.evidence.upstream is None
        and actor_wire not in (None, "openai_responses")
    ):
        raise ConfigError(
            f"evidence_upstream_required_for_visual_review_model: the {actor_wire} actor upstream cannot "
            f"serve {features.visual_review_model}; set agent.evidence.upstream (for example "
            '{"source": "codex_home"}) or features.visual_review_model=null'
        )
    choices = backend.traits.native_effort_choices
    if choices is not None and model.effort not in choices:
        raise ConfigError(
            f"agent_effort_unsupported:{kind}:{model.effort}: choose model.effort from "
            f"{', '.join(choices)}"
        )
    return actor_wire


def claude_code_runtime_problem(cli_path=None):
    """Return None when the pinned Claude Code runtime is usable, else a single-line reason."""
    try:
        from .claude_code.options import resolve_runtime

        with tempfile.TemporaryDirectory(prefix="ni-rua-claude-probe-") as home:
            resolve_runtime(
                cli_path,
                dict(PATH=os.defpath, LANG="C.UTF-8", HOME=home, TMPDIR=home, CLAUDE_CONFIG_DIR=home),
            )
    except ImportError as error:
        return f"claude_code_sdk_missing:{type(error).__name__}"
    except (RuntimeError, ValueError, OSError) as error:
        return str(error).split(":", 1)[0] if str(error) else type(error).__name__
    return None


def _check_claude_code_requirements(config):
    _check_hosted_shared_requirements(config, "claude_code")
    upstream = config.agent.claude_code.upstream
    if upstream.source == "profile":
        _check_profile_requirements(config, "claude_code")
    elif upstream.source == "claude_settings":
        from .claude_code.options import read_claude_settings_env, read_claude_settings_upstream

        try:
            read_claude_settings_upstream(upstream.settings)
            read_claude_settings_env(upstream.settings)
        except ValueError as error:
            raise ConfigError(str(error)) from None
    elif (
        upstream.source != "explicit"
        or upstream.wire_api not in (None, "anthropic_messages")
        or not upstream.base_url
        or not upstream.api_key_env
    ):
        raise ConfigError(
            "claude_code_upstream_requires_anthropic_messages: set agent.claude_code.upstream to an "
            "explicit anthropic_messages base_url and api_key_env"
        )
    from . import factory as _factory

    problem = _factory.claude_code_runtime_problem(config.agent.claude_code.cli_path)
    if problem is not None:
        raise ConfigError(f"agent_backend_unavailable:claude_code: {problem}; install ni-rua-harness[claude]")
    return None


def opencode_runtime_problem(executable):
    """校验固定版本 OpenCode 二进制及可执行 ripgrep（rg）；均可用时返回 None，否则返回错误码。"""
    from .isolation import require_exact_version
    from .opencode.config import OPENCODE_VERSION, probe_cli_version, resolve_ripgrep

    if not executable:
        return "opencode_executable_not_set"
    if not os.path.isabs(executable) or not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        return "opencode_executable_invalid"
    try:
        require_exact_version("opencode", probe_cli_version(executable), OPENCODE_VERSION)
        resolve_ripgrep(executable=executable)
    except (RuntimeError, ValueError, OSError) as error:
        return str(error).split(":", 1)[0] if str(error) else type(error).__name__
    return None


def _check_opencode_requirements(config):
    actor_wire = _check_hosted_shared_requirements(config, "opencode")
    upstream = config.agent.opencode.upstream
    opencode_wires = get_backend("opencode").traits.wires
    if upstream.wire_api is not None and upstream.wire_api not in opencode_wires:
        raise ConfigError(
            f"opencode_upstream_wire_unsupported:{upstream.wire_api}: set "
            "agent.opencode.upstream.wire_api to anthropic_messages or openai_responses"
        )
    if upstream.source == "profile":
        _check_profile_requirements(config, "opencode")
    elif upstream.source == "codex_home":
        if upstream.wire_api not in (None, "openai_responses"):
            raise ConfigError(
                "opencode_upstream_codex_home_requires_openai_responses: the Codex credential source "
                "serves the openai_responses wire only"
            )
    elif upstream.source == "opencode_config":
        from .opencode.upstream_file import read_opencode_config_upstream

        try:
            read_opencode_config_upstream(upstream.settings, provider=upstream.provider, wire_api=upstream.wire_api)
        except ValueError as error:
            raise ConfigError(str(error)) from None
    elif upstream.wire_api is None or not upstream.base_url or not upstream.api_key_env:
        raise ConfigError(
            "opencode_upstream_incomplete: set agent.opencode.upstream.wire_api (anthropic_messages or "
            "openai_responses), base_url and api_key_env, or source=codex_home"
        )
    from .opencode.config import EFFORT_CHOICES, OPENCODE_INSTALL_HINT

    choices = EFFORT_CHOICES[actor_wire]
    if config.model.effort not in choices:
        raise ConfigError(
            f"agent_effort_unsupported:opencode:{config.model.effort}: on the "
            f"{actor_wire} wire choose model.effort from {', '.join(choices)}"
        )
    from . import factory as _factory

    problem = _factory.opencode_runtime_problem(config.agent.opencode.executable)
    if problem is not None:
        raise ConfigError(
            f"agent_backend_unavailable:opencode: {problem}; {OPENCODE_INSTALL_HINT} and set "
            "agent.opencode.executable to that absolute path"
        )
    return None


def external_port_problem(port):
    """Return None when ``port`` can be bound now, else ``port_unavailable(port, error)``."""
    if port == 0:
        return None
    import socket
    from .external.session import port_unavailable

    probe = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        probe.bind(("127.0.0.1", port))
    except OSError as error:
        return port_unavailable(port, error)
    finally:
        probe.close()
    return None


def _check_external_requirements(config):
    from .external.protocol import RUN_BEGIN, RUN_FINISH

    run, features = config.run, config.features
    if run.benchmark or run.exit_after_turn:
        raise ConfigError(
            "external_backend_is_interactive_only: remove --benchmark / --exit-after-turn; an external "
            "agent attaches when its user starts it, so benchmark and one-turn runs need agent.backend "
            "codex, claude_code or opencode"
        )
    if features.sandbox_enabled:
        raise ConfigError(
            "external_backend_cannot_sandbox: NI-RUA does not start the agent; set "
            "features.sandbox_enabled=false or use agent.backend=codex, claude_code or opencode for "
            "sandboxed runs"
        )
    if features.visual_review_model:
        raise ConfigError(
            "external_backend_has_no_model_relay: set features.visual_review_model=null; an external "
            "agent's model requests do not pass through NI-RUA"
        )
    if config.model.price_card:
        raise ConfigError(
            "external_backend_has_no_model_relay: set model.price_card=null; an external agent's token "
            "usage is not recorded"
        )
    disabled = sorted({RUN_BEGIN, RUN_FINISH} & set(features.disabled_tools))
    if disabled:
        raise ConfigError(
            f"external_bracket_tool_cannot_be_disabled: remove {', '.join(disabled)} from "
            "features.disabled_tools; the external agent opens and closes its turns with them"
        )
    from . import factory as _factory

    problem = _factory.external_port_problem(config.agent.external.mcp_port)
    if problem is not None:
        raise ConfigError(problem)
    return None


def _create_codex_session(run_dir, store, tool_handler, *, price_card=None, config=None, task_planning=None):
    from .codex.session import CodexSession

    return CodexSession(
        run_dir,
        store,
        tool_handler,
        price_card=price_card,
        config=config,
        task_planning=task_planning,
    )


def _create_claude_session(run_dir, store, tool_handler, *, price_card=None, config=None, task_planning=None):
    from .claude_code.session import ClaudeCodeSession

    return ClaudeCodeSession(
        run_dir,
        store,
        tool_handler,
        price_card=price_card,
        config=config,
        task_planning=task_planning,
    )


def _create_opencode_session(run_dir, store, tool_handler, *, price_card=None, config=None, task_planning=None):
    from .opencode.session import OpenCodeSession

    return OpenCodeSession(
        run_dir,
        store,
        tool_handler,
        price_card=price_card,
        config=config,
        task_planning=task_planning,
    )


def _create_external_session(run_dir, store, tool_handler, *, price_card=None, config=None, task_planning=None):
    from .external.session import ExternalSession

    return ExternalSession(
        run_dir,
        store,
        tool_handler,
        price_card=price_card,
        config=config,
        task_planning=task_planning,
    )


def _thinking_display_choices():
    return next(item.metadata["choices"] for item in OpenCodeConfig.__dataclass_fields__.values() if item.name == "thinking_display")


# ---- profile section adapters ----------------------------------------------------------------------------------
def _codex_parse(raw, wires, location, ctx):
    return {}


def _codex_serialize(profile):
    return None


def _codex_public(profile):
    return {}


def _codex_forwarded_env(profile):
    return {}


def _codex_expand(agent, expanded, profile, wire_api):
    return "codex_upstream", expanded


def _codex_apply(upstream_dict, wire_api, profile, thinking_display):
    if thinking_display is not _PROFILE_UNSET:
        raise ValueError("thinking_display_requires_opencode: codex has no thinking_display setting")
    return dict(codex_upstream=upstream_dict)


def _collect_noop(section, hint, path_display):
    return


def _claude_parse(raw, wires, location, helpers):
    return dict(env=helpers["check_env"](raw.get("env", {}), location))


def _claude_serialize(profile):
    return {"env": dict(profile.claude_code_env)} if profile.claude_code_env else None


def _claude_public(profile):
    return dict(env=dict(profile.claude_code_env))


def _claude_forwarded_env(profile):
    return dict(profile.claude_code_env)


def _claude_expand(agent, expanded, profile, wire_api):
    return "claude_code", replace(agent.claude_code, upstream=expanded)


def _claude_apply(upstream_dict, wire_api, profile, thinking_display):
    if thinking_display is not _PROFILE_UNSET:
        raise ValueError("thinking_display_requires_opencode: claude_code has no thinking_display setting")
    return dict(claude_code={"upstream": upstream_dict})


def _claude_collect_hints(section, hint, path_display):
    return


def _opencode_parse(raw, wires, location, helpers):
    display = raw.get("thinking_display")
    choices = _thinking_display_choices()
    if display is not None and display not in choices:
        helpers["fail"](
            "invalid_thinking_display",
            location,
            "backends.opencode.thinking_display must be null or one of " + ", ".join(choices),
        )
    if display is not None and "anthropic_messages" not in wires:
        helpers["fail"](
            "thinking_display_requires_anthropic_messages",
            location,
            "backends.opencode.thinking_display applies to the anthropic_messages wire, which this profile does not declare",
        )
    return dict(thinking_display=display)


def _opencode_serialize(profile):
    return {"thinking_display": profile.opencode_thinking_display} if profile.opencode_thinking_display is not None else None


def _opencode_public(profile):
    return dict(thinking_display=profile.opencode_thinking_display)


def _opencode_forwarded_env(profile):
    return {}


def _opencode_expand(agent, expanded, profile, wire_api):
    display = agent.opencode.thinking_display
    if display is None and expanded.wire_api == "anthropic_messages":
        display = profile.opencode_thinking_display
    return "opencode", replace(agent.opencode, upstream=expanded, thinking_display=display)


def _opencode_apply(upstream_dict, wire_api, profile, thinking_display):
    if thinking_display is _PROFILE_UNSET:
        thinking_display = profile.opencode_thinking_display if wire_api == "anthropic_messages" else None
    return dict(opencode={"upstream": upstream_dict, "thinking_display": thinking_display})


def _opencode_collect_hints(section, hint, path_display):
    if "thinking_display" in section:
        hint["thinking_display"] = section["thinking_display"]


def helpers_unset():
    return _PROFILE_UNSET


_PROFILE_UNSET = object()


class BackendRegistry:
    """Ordered explicit registry of NI-RUA's known agent backends."""

    def __init__(self, backends: tuple[AgentBackend, ...]):
        self._backends = {backend.kind: backend for backend in backends}
        self._order = tuple(backend.kind for backend in backends)

    def get(self, kind: str, *, error_cls=AgentCapabilityError) -> AgentBackend:
        backend = self._backends.get(kind)
        if backend is None:
            raise error_cls(f"unknown_agent_backend:{kind}")
        return backend

    def hosted_kinds(self) -> tuple[str, ...]:
        return tuple(kind for kind in self._order if self._backends[kind].traits.hosted)

    def hosted_backends(self) -> tuple[AgentBackend, ...]:
        return tuple(self._backends[kind] for kind in self._order if self._backends[kind].traits.hosted)

    def backend_wires(self) -> dict[str, tuple[str, ...]]:
        return {kind: self._backends[kind].traits.wires for kind in self.hosted_kinds()}

    def native_effort_choices(self) -> dict[str, tuple[str, ...]]:
        out = {}
        for kind in self.hosted_kinds():
            choices = self._backends[kind].traits.native_effort_choices
            if choices is not None:
                out[kind] = choices
        return out

    def profile_sections(self) -> dict[str, ProfileSection]:
        return {
            backend.profile_section.key: backend.profile_section
            for backend in self.hosted_backends()
            if backend.profile_section is not None and backend.profile_section.section_fields
        }


BACKENDS_REGISTRY = BackendRegistry(
    (
        AgentBackend(
            traits=BackendTraits(
                kind="codex",
                hosted=True,
                wires=("openai_responses",),
                native_effort_choices=None,
                live_effort_default=True,
                service_tier=True,
                relay_compat_fields=False,
                evidence_native=True,
                upstream_config_attr="codex_upstream",
                capabilities_loader=_lazy_codex_capabilities,
                effort_choices_loader=_lazy_codex_efforts,
            ),
            profile_section=ProfileSection(
                key="codex",
                section_fields=(),
                parse=_codex_parse,
                serialize=_codex_serialize,
                public_view=_codex_public,
                forwarded_env=_codex_forwarded_env,
                expand_config=_codex_expand,
                apply_config=_codex_apply,
                collect_hints=_collect_noop,
            ),
            requirements_checker=_check_codex_requirements,
            session_creator=_create_codex_session,
            upstream_accessor=_codex_actor_upstream,
        ),
        AgentBackend(
            traits=BackendTraits(
                kind="claude_code",
                hosted=True,
                wires=("anthropic_messages",),
                native_effort_choices=_CLAUDE_EFFORTS,
                live_effort_default=False,
                service_tier=False,
                relay_compat_fields=True,
                evidence_native=False,
                capabilities_loader=_lazy_claude_capabilities,
                effort_choices_loader=_lazy_claude_efforts,
            ),
            profile_section=ProfileSection(
                key="claude_code",
                section_fields=("env",),
                parse=_claude_parse,
                serialize=_claude_serialize,
                public_view=_claude_public,
                forwarded_env=_claude_forwarded_env,
                expand_config=_claude_expand,
                apply_config=_claude_apply,
                collect_hints=_claude_collect_hints,
            ),
            requirements_checker=_check_claude_code_requirements,
            session_creator=_create_claude_session,
            upstream_accessor=_sub_actor_upstream("claude_code"),
        ),
        AgentBackend(
            traits=BackendTraits(
                kind="opencode",
                hosted=True,
                wires=("anthropic_messages", "openai_responses"),
                native_effort_choices=None,
                live_effort_default=True,
                service_tier=True,
                relay_compat_fields=True,
                evidence_native=False,
                capabilities_loader=_lazy_opencode_capabilities,
                effort_choices_loader=_lazy_opencode_efforts,
                thinking_display_loader=_lazy_opencode_thinking_display,
            ),
            profile_section=ProfileSection(
                key="opencode",
                section_fields=("thinking_display",),
                parse=_opencode_parse,
                serialize=_opencode_serialize,
                public_view=_opencode_public,
                forwarded_env=_opencode_forwarded_env,
                expand_config=_opencode_expand,
                apply_config=_opencode_apply,
                collect_hints=_opencode_collect_hints,
            ),
            requirements_checker=_check_opencode_requirements,
            session_creator=_create_opencode_session,
            upstream_accessor=_sub_actor_upstream("opencode"),
        ),
        AgentBackend(
            traits=BackendTraits(
                kind=EXTERNAL_BACKEND_KIND,
                hosted=False,
                wires=(),
                relay_compat_fields=False,
            ),
            profile_section=None,
            requirements_checker=_check_external_requirements,
            session_creator=_create_external_session,
            upstream_accessor=lambda _config: None,
        ),
    )
)


def wire_supports_service_tier(wire_api: str | None) -> bool:
    """按协议线判定是否支持 Responses service_tier（仅 openai_responses 支持）。"""
    if not wire_api:
        return False
    from ..session.wire import AnthropicMessagesWire, OpenAIResponsesWire

    adapters = {
        "openai_responses": OpenAIResponsesWire,
        "anthropic_messages": AnthropicMessagesWire,
    }
    adapter = adapters.get(wire_api)
    return bool(adapter.supports_service_tier) if adapter is not None else False


def get_backend(kind: str, *, error_cls=AgentCapabilityError) -> AgentBackend:
    """Return the registered ``AgentBackend`` for ``kind`` or raise ``error_cls('unknown_agent_backend:<kind>')``."""
    return BACKENDS_REGISTRY.get(kind, error_cls=error_cls)


def agent_backend(config) -> str:
    """Return ``config.agent.backend`` (default ``'codex'`` when no agent section exists)."""
    return getattr(getattr(config, "agent", None), "backend", "codex")


def actor_upstream(config, kind: str | None = None):
    """Return the actor ``UpstreamConfig`` of ``kind`` (default ``agent_backend(config)``), or None."""
    kind = agent_backend(config) if kind is None else kind
    try:
        backend = BACKENDS_REGISTRY.get(kind)
    except AgentCapabilityError:
        return getattr(getattr(getattr(config, "agent", None), kind, None), "upstream", None)
    return backend.upstream_accessor(config)
