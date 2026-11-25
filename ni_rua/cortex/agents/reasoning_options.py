"""Reasoning controls of the operator console, generated from backend code and the model's catalogue entry.

Design 16 §7.1. ``reasoning_options(backend, wire_api, model, ...)`` returns the controls the console renders for one
(backend, wire, model) selection: reasoning ``effort``, Codex ``service_tier`` (speed) and OpenCode
``thinking_display``. No per-model table exists anywhere: every choice comes from the backend's own code
(``backend_traits``) and from what the provider's listing (``model_catalogue.CatalogueModel``) or, for Codex, the
pinned native catalogue (``cortex/session/codex_models.json``) reports for the model.

* Effort: ``backend`` = the backend's accepted values (``CodexSession.effort_choices`` = every ``ModelConfig.effort``
  choice; ``ClaudeCodeSession.effort_choices`` = ``factory.NATIVE_EFFORT_CHOICES["claude_code"]``; OpenCode
  ``opencode.config.EFFORT_CHOICES[wire]``, which ``OpenCodeSession.effort_choices`` returns); ``model`` = the
  catalogue's ``reasoning_efforts`` (else, for Codex only, ``supported_reasoning_levels`` of the native catalogue; a
  model reporting ``reasoning: false`` declares none); None when the model reports nothing. ``supported`` =
  backend ∩ model (backend alone when the model reports nothing). A backend value the model does not declare stays
  selectable and is marked (catalogues can be wrong; the provider is the judge); a value the backend refuses is never
  selectable (``set_effort`` would answer ``invalid_effort``).
* Applies: ``capabilities.live_effort`` (Codex, OpenCode) -> ``next_turn`` through ``set_effort``; otherwise (Claude
  Code) ``next_session`` through a handoff to a successor run.
* Service tier: only with ``capabilities.service_tier`` (Codex); choices = ``SERVICE_TIERS``
  (re-exported by ``apps.gateway.extensions.operator_console``), each with the relay's probe result for the current model and, for a model
  the native Codex catalogue knows, whether it lists that tier.
* Thinking display: only where the backend can send it (OpenCode on ``anthropic_messages``: ``opencode.config
  .effort_variants`` accepts a display there and refuses it on other wires); Claude Code's thinking switches are CLI
  environment variables kept in the provider profile (``backends.claude_code.env``), so no console knob is made.

Everything here is operator data: it is never put into the actor's prompt, instructions or tool results. Pure: no
I/O except reading the pinned native Codex catalogue once (``model_catalogue.native_codex_model``).
"""
from __future__ import annotations

from dataclasses import dataclass, fields

from .model_catalogue import CatalogueModel, merge_model, native_codex_entry, native_codex_model

__all__ = ["APPLIES_NEXT_TURN", "APPLIES_NEXT_SESSION", "APPLIES_NEXT_MODEL_REQUEST", "SCOPE_LIVE", "SCOPE_SESSION",
           "SERVICE_TIERS", "BackendTraits", "backend_traits", "reasoning_options"]

# Responses service_tier values offered in the operator console and relay UI controls.
SERVICE_TIERS = ("default", "priority", "ultrafast")

# When a control's new value takes effect (``applies``) and how it is applied (``scope``).
APPLIES_NEXT_TURN = "next_turn"                    # the running session uses it from its next turn
APPLIES_NEXT_SESSION = "next_session"              # needs a new session: a handoff to a successor run
APPLIES_NEXT_MODEL_REQUEST = "next_model_request"  # the relay uses it from its next upstream request, also mid-turn
SCOPE_LIVE = "live"            # applied in the running run (set_effort / set_service_tier / apply_agent_settings)
SCOPE_SESSION = "session"      # applied by handoff_run (finalize this run, start a successor run)

_APPLIES_LABELS = {APPLIES_NEXT_TURN: "下一轮生效", APPLIES_NEXT_SESSION: "需要新会话",
                   APPLIES_NEXT_MODEL_REQUEST: "下一次模型请求生效（轮内也生效）"}
_CONTROL_LABELS = {"effort": "强度", "service_tier": "速度", "thinking_display": "思考显示"}
# Labels of the console's service tiers when the native catalogue names none (same words as console.html).
_TIER_LABELS = {"default": "正常", "priority": "Fast", "ultrafast": "Ultrafast"}
_THINKING_LABELS = {None: "不发送（默认）", "summarized": "summarized（摘要）", "omitted": "omitted（不返回思考文本）"}
_NOTE_MODEL_UNDECLARED = "模型未声明支持"
_NOTE_BACKEND_REFUSES = "当前后端不支持"
_NOTE_TIER_REJECTED = "供应商拒绝了该档位"
_NOTE_NO_EFFORT_UPDATES = "模型声明不支持在会话中途更改强度（由供应商裁决）"


@dataclass(frozen=True)
class BackendTraits:
    """Reasoning-related facts of one agent backend on one wire, read from the backend's code (``backend_traits``).

    ``effort_choices``: ``model.effort`` values the backend accepts on that wire, in schema order (``()`` = none: the
    wire has no effort variants). ``live_effort`` / ``service_tier``: ``AgentCapabilities`` flags of the session
    class. ``thinking_display_choices``: values of ``agent.opencode.thinking_display`` the backend can send on that
    wire (``()`` = no such knob)."""
    backend: str
    wire_api: str | None
    effort_choices: tuple
    live_effort: bool
    service_tier: bool
    thinking_display_choices: tuple


def backend_traits(backend, wire_api):
    """``BackendTraits`` of ``backend`` (``"codex"`` | ``"claude_code"`` | ``"opencode"``) on ``wire_api``.

    Read dynamically from the registered backend descriptor (``registry.get_backend``); constructing nothing,
    starting nothing, importing no agent SDK. Raises ``ValueError("unknown_agent_backend:<backend>")``.
    """
    from .registry import get_backend, wire_supports_service_tier

    entry = get_backend(backend, error_cls=ValueError)
    if not entry.traits.hosted:
        raise ValueError(f"unknown_agent_backend:{backend}")
    capabilities = entry.capabilities()
    # service_tier 按后端与协议线联合判定（OpenCode 仅在 openai_responses 线支持）。
    supports_tier = bool(
        (capabilities.service_tier or entry.traits.service_tier)
        and (wire_api is None or wire_supports_service_tier(wire_api))
    )
    return BackendTraits(
        backend=backend,
        wire_api=wire_api,
        effort_choices=entry.effort_choices(wire_api),
        live_effort=bool(capabilities.live_effort),
        service_tier=supports_tier,
        thinking_display_choices=entry.thinking_display_choices(wire_api),
    )


def _schema_efforts():
    """``(choices, default)`` of ``ModelConfig.effort`` (the config schema)."""
    from ...core.config import ModelConfig
    item = next(entry for entry in fields(ModelConfig) if entry.name == "effort")
    return tuple(item.metadata["choices"]), item.default


def _catalogue_model(model):
    """A ``CatalogueModel`` from ``model`` (``CatalogueModel``, its ``to_dict()`` form, or None)."""
    if model is None or isinstance(model, CatalogueModel):
        return model
    if isinstance(model, dict):
        return CatalogueModel.from_dict(model)
    raise TypeError("model must be a CatalogueModel, its to_dict() form or None")


def _model_efforts(entry, native):
    """``(efforts tuple | None, source)``: the catalogue's list, ``()`` for a catalogue ``reasoning: false``, else
    the native Codex catalogue's list, else ``(None, None)``."""
    if entry is not None and entry.reasoning_efforts is not None:
        return entry.reasoning_efforts, "catalogue"
    if entry is not None and entry.reasoning is False:
        return (), "catalogue"
    if native is not None and native.reasoning_efforts is not None:
        return native.reasoning_efforts, "codex_models"
    return None, None


def _unavailable(control, reason, message):
    return dict(available=False, label=_CONTROL_LABELS[control], reason=reason, message=message)


def _effort_control(traits, efforts, source, value, descriptions, effort_updates):
    if not traits.effort_choices:
        return _unavailable("effort", "effort_unsupported_on_wire",
                            f"{traits.backend} 在 {traits.wire_api} 上没有推理强度选项")
    schema, schema_default = _schema_efforts()
    backend_set, model_set = set(traits.effort_choices), None if efforts is None else set(efforts)
    choices = []
    for name in schema:
        backend = name in backend_set
        model = None if model_set is None else name in model_set
        if not backend and not model:
            continue
        choices.append(dict(value=name, label=name, description=descriptions.get(name), backend=backend, model=model,
                            selectable=backend, recommended=backend and model is not False,
                            note=(_NOTE_BACKEND_REFUSES if not backend
                                  else _NOTE_MODEL_UNDECLARED if model is False else None)))
    supported = [choice["value"] for choice in choices if choice["recommended"]]
    default = (schema_default if schema_default in backend_set
               else (supported or [name for name in schema if name in backend_set])[0])
    live = traits.live_effort
    applies = APPLIES_NEXT_TURN if live else APPLIES_NEXT_SESSION
    return dict(available=True, label=_CONTROL_LABELS["effort"], value=value,
                value_selectable=None if value is None else value in backend_set, default=default,
                command="set_effort" if live else "handoff_run", scope=SCOPE_LIVE if live else SCOPE_SESSION,
                applies=applies, applies_label=_APPLIES_LABELS[applies], model_source=source, supported=supported,
                note=_NOTE_NO_EFFORT_UPDATES if live and effort_updates is False else None, choices=choices)


def _service_tier_control(traits, model_id, value, tier_probes, service_tiers):
    if not traits.service_tier:
        message = (
            f"当前协议线（{traits.wire_api}）不支持服务档位设置"
            if traits.backend == "opencode" and traits.wire_api
            else f"当前 agent 后端（{traits.backend}）不支持服务档位设置"
        )
        return _unavailable("service_tier", "service_tier_unsupported_by_backend", message)
    if service_tiers is None:
        service_tiers = SERVICE_TIERS
    native = native_codex_entry(model_id) if traits.backend == "codex" else None
    listed = {}
    if native is not None:
        for tier in native.get("service_tiers") or ():
            if isinstance(tier, dict) and isinstance(tier.get("id"), str):
                listed[tier["id"]] = tier
    probes = tier_probes if isinstance(tier_probes, dict) else {}
    choices = []
    for name in service_tiers:
        probe = probes.get(name) if name != "default" else None
        probe = dict(probe) if isinstance(probe, dict) else None
        tier = listed.get(name) or {}
        rejected = probe is not None and probe.get("supported") is False
        model = None if native is None else (True if name == "default" else name in listed)
        choices.append(dict(value=name, label=tier.get("name") if isinstance(tier.get("name"), str) and tier.get("name")
                            else _TIER_LABELS.get(name, name),
                            description=tier.get("description") if isinstance(tier.get("description"), str) else None,
                            model=model, probe=probe, selectable=not rejected,
                            note=_NOTE_TIER_REJECTED if rejected else None))
    return dict(available=True, label=_CONTROL_LABELS["service_tier"], value=value or "default", default="default",
                command="set_service_tier", scope=SCOPE_LIVE, applies=APPLIES_NEXT_MODEL_REQUEST,
                applies_label=_APPLIES_LABELS[APPLIES_NEXT_MODEL_REQUEST],
                model_source="codex_models" if native is not None else None, choices=choices)


def _thinking_display_control(traits, value):
    if traits.backend == "codex":
        return _unavailable("thinking_display", "backend_has_no_thinking_display", "Codex 没有思考显示选项")
    if traits.backend == "claude_code":
        return _unavailable("thinking_display", "claude_code_thinking_in_profile_env",
                            "Claude Code 的思考显示与预算由 CLI 环境变量决定，在 profile 的 backends.claude_code.env 中设置"
                            "（例如 CLAUDE_CODE_THINKING_DISPLAY_UPDATES）")
    if not traits.thinking_display_choices:
        return _unavailable("thinking_display", "opencode_thinking_display_requires_anthropic_messages",
                            f"思考显示只在 anthropic_messages 上可用（当前 {traits.wire_api}）")
    choices = [dict(value=name, label=_THINKING_LABELS.get(name, name), selectable=True)
               for name in (None, *traits.thinking_display_choices)]
    return dict(available=True, label=_CONTROL_LABELS["thinking_display"], value=value,
                value_selectable=value is None or value in traits.thinking_display_choices, default=None,
                command="handoff_run", scope=SCOPE_SESSION, applies=APPLIES_NEXT_SESSION,
                applies_label=_APPLIES_LABELS[APPLIES_NEXT_SESSION], choices=choices)


def reasoning_options(backend, wire_api, model=None, *, model_id=None, effort=None, service_tier=None,
                      thinking_display=None, tier_probes=None, service_tiers=None):
    """The reasoning controls for one (backend, wire, model) selection, as a JSON-ready dict (design 16 §7.1).

    Arguments:
        backend: ``"codex"`` | ``"claude_code"`` | ``"opencode"`` (``ValueError("unknown_agent_backend:<x>")``).
        wire_api: the upstream wire of the selection (``openai_responses`` for Codex, ``anthropic_messages`` for Claude
            Code, either for OpenCode).
        model: the model's provider-listing entry (``CatalogueModel`` or its ``to_dict()`` form), or None when the
            model is not listed (manual entry, listing unavailable).
        model_id: the model id when ``model`` is None (used for the native Codex catalogue lookup); ignored otherwise.
        effort: the current / candidate ``model.effort`` (str) or None (unknown).
        service_tier: the current relay tier (``ModelRequestProxy.service_tier``; None = ``"default"``).
        thinking_display: the current / candidate ``agent.opencode.thinking_display`` (None = not sent).
        tier_probes: ``{tier: probe}`` of the relay for the current model (``ModelRequestProxy.select_model(model)``;
            each probe ``{"tier", "supported", "served_tier", "error", ...}``), or None (nothing probed).
        service_tiers: the tier values the console offers; None (default) = ``SERVICE_TIERS``.

    Returns ``{"backend", "wire_api", "model", "effort", "service_tier", "thinking_display"}``:

    * ``model``: ``{"id", "listed" (in the provider listing), "native_codex" (known to the pinned native Codex
      catalogue; Codex only), "reasoning", "efforts", "toggle", "default_effort", "budget", "effort_updates"}`` —
      the listing's claims, with native Codex values filling what the listing does not report (Codex only);
      None = not reported. ``efforts`` is the list the effort control's ``model`` flags use (``[]`` for a model
      reporting ``reasoning: false``).
    * every control: ``available`` (bool) and ``label`` (Chinese UI label); unavailable ones add ``reason`` (stable
      code) and ``message`` (Chinese sentence). Available ones add ``value``, ``default``, ``command`` (the control
      command that applies it; ``handoff_run`` = a successor run), ``scope`` (``live`` | ``session``), ``applies``
      (``next_turn`` | ``next_session`` | ``next_model_request``), ``applies_label`` and ``choices``.
    * ``effort`` adds ``value_selectable``, ``model_source`` (``catalogue`` | ``codex_models`` | None),
      ``supported`` (backend ∩ model, in schema order), ``note`` (a live control for a model reporting
      ``effort_updates: false`` says so; else None) and per choice ``{value, label, description, backend, model,
      selectable, recommended, note}``; ``backend`` False = never selectable, ``model`` False = selectable with the
      note "模型未声明支持", ``model`` None = the model reports nothing.
    * ``service_tier`` adds ``model_source`` and per choice ``{value, label, description, model, probe, selectable,
      note}``; a probe with ``supported: false`` makes the tier unselectable.
    * ``thinking_display`` adds ``value_selectable`` and per choice ``{value, label, selectable}`` (value None = the
      field is not sent).
    """
    traits = backend_traits(backend, wire_api)
    entry = _catalogue_model(model)
    identifier = entry.id if entry is not None else model_id
    native = native_codex_model(identifier) if backend == "codex" else None
    merged = merge_model(entry, native)
    efforts, source = _model_efforts(entry, native)
    descriptions = {}
    if source == "codex_models":
        for level in (native_codex_entry(identifier) or {}).get("supported_reasoning_levels") or ():
            if isinstance(level, dict) and isinstance(level.get("effort"), str) and isinstance(level.get("description"),
                                                                                                str):
                descriptions[level["effort"]] = level["description"]
    model_block = dict(
        id=identifier if isinstance(identifier, str) and identifier else None, listed=entry is not None,
        native_codex=native is not None,
        reasoning=getattr(merged, "reasoning", None),
        efforts=list(efforts) if efforts is not None else None,
        toggle=getattr(merged, "reasoning_toggle", None),
        default_effort=getattr(merged, "reasoning_default_effort", None),
        budget=getattr(merged, "reasoning_budget", None), effort_updates=getattr(merged, "effort_updates", None))
    return dict(backend=backend, wire_api=wire_api, model=model_block,
                effort=_effort_control(traits, efforts, source, effort, descriptions, model_block["effort_updates"]),
                service_tier=_service_tier_control(traits, identifier, service_tier, tier_probes, service_tiers),
                thinking_display=_thinking_display_control(traits, thinking_display))
