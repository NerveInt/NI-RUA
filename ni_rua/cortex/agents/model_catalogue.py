"""Provider model catalogue: the models an upstream lists at ``GET <base>/v1/models`` (operator suggestions).

The host (never the agent) reads the upstream's model listing with the run's host credentials, so the operator console
can offer the provider's models for ``set_model`` (``AgentSession.model_options``). The listing is operator data: it
is never put into the actor's prompt, instructions or tool results, and a failed listing never fails a launch (the
result is a ``ModelCatalogue`` with ``status="unavailable"`` and a short ``reason``, never an exception).

Shapes read (every field except ``id`` optional):

* OpenAI-style multi-model gateways: ``{"object": "list", "data": [{"id", "name", "protocols": ["COMPLETIONS",
  "MESSAGES", "RESPONSES"] | ["NATIVE"], "capabilities": {"tools", "image_input", "context_window",
  "max_output_tokens", ...}}], "next_cursor": <str | null>}``; a non-empty ``next_cursor`` is sent back as the
  ``cursor`` query parameter (not observed live: the tested gateway lists all 32 models on one page);
* Anthropic: ``{"data": [{"id", "display_name", ...}], "has_more": bool, "last_id": str}``; the next page is requested
  with ``after_id=<last_id>``;
* models.dev-style listings (OpenCode Console, verified 2026-11-14 at ``https://opencode.ai/inference/v1/models``):
  ``{"object": "list", "data": [{"id", "name", "modalities": {"input": [media types]}, "capabilities": {"tools":
  {"supported": bool}, "reasoning": {"supported": bool, "toggle": bool, "effort": [str], "budget": {}, "default":
  {"type": "effort", "value": str}}, "effort_updates": bool}, "limit": {"context", "output"}}]}``.

Reasoning capabilities are provider claims, read tolerantly (``CatalogueModel.reasoning*``); a claim never blocks a
selection the backend accepts. ``CatalogueCache`` keeps one catalogue per (profile id, wire) in memory and on disk
(no credential, no URL) with a TTL, an explicit refresh and one in-flight read per key. ``native_codex_model`` reads
the pinned native Codex catalogue (``cortex/session/codex_models.json``) for ids a Responses listing gives without
metadata.

Nothing here imports a backend; ``fetch_upstream_catalogue`` is the one call a session makes (stdlib HTTP only).
"""
from __future__ import annotations

import functools
import json
import os
import re
import socket
import tempfile
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, fields, replace
from datetime import datetime, timezone
from pathlib import Path

__all__ = ["WIRE_PROTOCOLS", "DEFAULT_TIMEOUT_S", "DEFAULT_MAX_PAGES", "ANTHROPIC_VERSION", "CatalogueModel",
           "ModelCatalogue", "models_url", "parse_models_page", "usable_models", "fetch_model_catalogue",
           "fetch_upstream_catalogue", "FAILURE_KINDS", "failure_kind", "native_codex_entry", "native_codex_model",
           "merge_model", "CREDENTIAL_REASON", "CACHE_TTL_S", "CACHE_SCHEMA", "CACHE_DIR_NAME", "CatalogueView",
           "CatalogueCache"]

# Upstream wire -> the ``protocols`` entry a listed model must carry to be served on that wire (when it lists any).
WIRE_PROTOCOLS = {"anthropic_messages": "MESSAGES", "openai_responses": "RESPONSES", "openai_chat": "COMPLETIONS"}
DEFAULT_TIMEOUT_S = 5.0       # per page request, seconds
DEFAULT_MAX_PAGES = 10        # page cap; more pages -> status "truncated"
ANTHROPIC_VERSION = "2023-06-01"   # header the Anthropic API requires on GET /v1/models
_MAX_PAGE_BYTES = 8 << 20


@dataclass(frozen=True)
class CatalogueModel:
    """One listed model. ``id``: model id as sent in requests. ``name``: display name (``name`` or Anthropic
    ``display_name``), None when absent. ``protocols``: upper-case wire protocols the provider serves it on
    (``MESSAGES``, ``RESPONSES``, ``COMPLETIONS``, ``NATIVE``), None when not listed. ``tools`` / ``image_input``:
    ``capabilities.tools`` / ``capabilities.image_input`` (True / False / None = not reported).
    ``context_window`` / ``max_output_tokens``: ``capabilities`` token limits (positive int tokens, None = not
    reported).

    Reasoning claims (provider data; None = not reported, never "unsupported"):
    ``reasoning``: the model reasons (``capabilities.reasoning`` as a bool or ``{"supported": bool}``).
    ``reasoning_efforts``: effort names the provider lists for the model, in the listing's order, lower-case, no
    duplicates (``capabilities.reasoning.effort``, or an item-level ``supported_reasoning_levels`` of strings or
    ``{"effort": str}``); ``()`` = an empty list was reported. Names outside ``ModelConfig.effort`` are kept as listed.
    ``reasoning_toggle``: reasoning can be switched off (``capabilities.reasoning.toggle``).
    ``reasoning_default_effort``: the effort the provider applies when none is sent
    (``capabilities.reasoning.default = {"type": "effort", "value": str}``).
    ``reasoning_budget``: True when the listing reports a token-budget form of thinking
    (``capabilities.reasoning.budget`` present; the listing gives no bounds).
    ``effort_updates``: the provider allows changing the effort inside a conversation
    (``capabilities.effort_updates``)."""
    id: str
    name: str | None = None
    protocols: tuple | None = None
    tools: bool | None = None
    image_input: bool | None = None
    context_window: int | None = None
    max_output_tokens: int | None = None
    reasoning: bool | None = None
    reasoning_efforts: tuple | None = None
    reasoning_toggle: bool | None = None
    reasoning_default_effort: str | None = None
    reasoning_budget: bool | None = None
    effort_updates: bool | None = None

    def to_dict(self):
        value = asdict(self)
        value["protocols"] = list(self.protocols) if self.protocols is not None else None
        value["reasoning_efforts"] = list(self.reasoning_efforts) if self.reasoning_efforts is not None else None
        return value

    @classmethod
    def from_dict(cls, value):
        """Inverse of ``to_dict`` (cache files); unknown keys are ignored. Raises ``ValueError("invalid_model")``
        without a non-empty string ``id``."""
        if not isinstance(value, dict) or not isinstance(value.get("id"), str) or not value["id"]:
            raise ValueError("invalid_model")
        known = {item.name for item in fields(cls)}
        data = {key: item for key, item in value.items() if key in known}
        for key in ("protocols", "reasoning_efforts"):
            if isinstance(data.get(key), list):
                data[key] = tuple(data[key])
        return cls(**data)


@dataclass(frozen=True)
class ModelCatalogue:
    """Result of one catalogue read. ``status``: ``"ok"`` (every page read), ``"truncated"`` (stopped at the page cap
    or a repeated cursor; ``models`` holds what was read), ``"unavailable"`` (nothing usable; ``models`` is
    empty) or ``"not_read"`` (``CatalogueCache`` only: the source has no listing). ``reason``: short code for a
    non-ok status (``http_401``, ``timeout``, ``URLError``, ``invalid_json``, ``invalid_shape``, ``page_cap``,
    ``cursor_repeated``), never response text. ``pages``: pages read. ``auth``:
    ``"configured"`` (the host credential headers as given) or ``"bearer_fallback"`` (an ``x-api-key`` credential was
    refused with 401/403 and accepted as ``Authorization: Bearer``), None when no page was read."""
    status: str
    reason: str | None = None
    models: tuple = ()
    pages: int = 0
    auth: str | None = None

    def usable(self, wire_api):
        """``usable_models(self.models, wire_api)``."""
        return usable_models(self.models, wire_api)

    @property
    def failure(self):
        """Typed failure of this read (``failure_kind(status, reason)``): None for ``ok``, else one of
        ``FAILURE_KINDS``."""
        return failure_kind(self.status, self.reason)

    def record(self, wire_api):
        """Observer record (event payload): status, reason, pages, auth, counts and the usable models with their
        limits and image flag. Holds no credential and no response text."""
        usable = self.usable(wire_api)
        return dict(status=self.status, reason=self.reason, pages=self.pages, auth=self.auth, wire_api=wire_api,
                    listed=len(self.models), usable=len(usable), models=[model.to_dict() for model in usable])


# User-Agent of the listing request (gateways may refuse anonymous library agents).
USER_AGENT = "ni-rua-model-catalogue/0.2"


def models_url(base_url, wire_api):
    """URL of the listing for an upstream ``base_url`` in the relay's convention: Anthropic Messages bases have no
    ``/v1`` (the relay appends ``/v1/messages``), so the listing is ``<base>/v1/models``; Responses / Chat bases
    include it (``/responses``), so it is ``<base>/models``."""
    base = str(base_url).rstrip("/")
    return base + ("/v1/models" if wire_api == "anthropic_messages" else "/models")


def _flag(value):
    """True / False / None from a capability value: a bool, or an object ``{"supported": bool}`` (the models.dev-style
    listing OpenCode Console serves); anything else is None (not reported)."""
    if isinstance(value, dict):
        value = value.get("supported")
    return value if isinstance(value, bool) else None


def _image_input(capabilities, item):
    """``capabilities.image_input`` when reported, else whether ``modalities.input`` lists an image modality
    (``"image"`` or an ``image/<subtype>`` media type); None when neither is reported."""
    flag = _flag(capabilities.get("image_input"))
    if flag is not None:
        return flag
    modalities = item.get("modalities") if isinstance(item.get("modalities"), dict) else {}
    inputs = modalities.get("input")
    if not isinstance(inputs, list):
        return None
    return any(isinstance(entry, str) and (entry == "image" or entry.startswith("image/")) for entry in inputs)


def _tokens(value):
    return value if type(value) is int and value > 0 else None


def _effort_name(value):
    """A listed effort name (``"high"`` or ``{"effort": "high"}``) as a stripped lower-case string, else None."""
    if isinstance(value, dict):
        value = value.get("effort")
    return value.strip().lower() if isinstance(value, str) and value.strip() else None


def _efforts(values):
    """Tuple of effort names (order kept, duplicates and non-names dropped) of a listed list; None when not a list."""
    if not isinstance(values, list):
        return None
    names = []
    for value in values:
        name = _effort_name(value)
        if name is not None and name not in names:
            names.append(name)
    return tuple(names)


def _reasoning(capabilities, item):
    """The reasoning fields of ``CatalogueModel`` from ``capabilities.reasoning`` (bool or ``{"supported", "toggle",
    "effort", "budget", "default"}``), an item-level ``supported_reasoning_levels`` and ``capabilities.effort_updates``.
    A reported effort list without ``supported`` implies ``reasoning=True``."""
    value = capabilities.get("reasoning")
    detail = value if isinstance(value, dict) else {}
    efforts = _efforts(detail.get("effort"))
    if efforts is None:
        efforts = _efforts(item.get("supported_reasoning_levels"))
    supported = _flag(value)
    if supported is None and efforts:
        supported = True
    default = detail.get("default")
    default = (_effort_name(default.get("value")) if isinstance(default, dict) and default.get("type") == "effort"
               else None)
    toggle = detail.get("toggle")
    updates = capabilities.get("effort_updates")
    return dict(reasoning=supported, reasoning_efforts=efforts,
                reasoning_toggle=toggle if isinstance(toggle, bool) else None, reasoning_default_effort=default,
                reasoning_budget=True if isinstance(detail.get("budget"), dict) else None,
                effort_updates=updates if isinstance(updates, bool) else None)


def _model(item):
    if not isinstance(item, dict) or not isinstance(item.get("id"), str) or not item["id"].strip():
        return None
    name = item.get("name") if isinstance(item.get("name"), str) else item.get("display_name")
    protocols = item.get("protocols")
    protocols = (tuple(str(entry).upper() for entry in protocols if isinstance(entry, str))
                 if isinstance(protocols, list) else None)
    capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
    limit = item.get("limit") if isinstance(item.get("limit"), dict) else {}
    return CatalogueModel(id=item["id"].strip(), name=name.strip() if isinstance(name, str) and name.strip() else None,
                          protocols=protocols, tools=_flag(capabilities.get("tools")),
                          image_input=_image_input(capabilities, item),
                          context_window=_tokens(capabilities.get("context_window")) or _tokens(limit.get("context")),
                          max_output_tokens=(_tokens(capabilities.get("max_output_tokens"))
                                             or _tokens(limit.get("output"))),
                          **_reasoning(capabilities, item))


def parse_models_page(document):
    """``(models, next query dict or None)`` of one listing page; ``ValueError("invalid_shape")`` without a
    ``data`` list. Items without a string ``id`` are skipped."""
    if not isinstance(document, dict) or not isinstance(document.get("data"), list):
        raise ValueError("invalid_shape")
    models = [model for model in (_model(item) for item in document["data"]) if model is not None]
    cursor = document.get("next_cursor")
    if isinstance(cursor, str) and cursor:
        return models, {"cursor": cursor}
    if document.get("has_more") is True and isinstance(document.get("last_id"), str) and document["last_id"]:
        return models, {"after_id": document["last_id"]}
    return models, None


def usable_models(models, wire_api):
    """The models an agent can drive on ``wire_api``: ``protocols`` (when listed) include the wire's protocol
    (``WIRE_PROTOCOLS``) and ``tools`` is not False. Order kept, duplicates dropped. Image-less models stay (the
    operator sees ``image_input``)."""
    protocol = WIRE_PROTOCOLS.get(wire_api)
    seen, usable = set(), []
    for model in models:
        if model.id in seen or model.tools is False:
            continue
        if model.protocols is not None and protocol not in model.protocols:
            continue
        seen.add(model.id)
        usable.append(model)
    return tuple(usable)


def _with_query(url, query):
    parts = urllib.parse.urlsplit(url)
    merged = dict(urllib.parse.parse_qsl(parts.query))
    merged.update(query or {})
    return urllib.parse.urlunsplit(parts._replace(query=urllib.parse.urlencode(merged)))


def _bearer_fallback(headers):
    """``Authorization: Bearer <key>`` for headers that carry only ``x-api-key``, else None."""
    names = {name.lower(): name for name in headers}
    if "authorization" in names or "x-api-key" not in names:
        return None
    rest = {name: value for name, value in headers.items() if name.lower() != "x-api-key"}
    return {**rest, "Authorization": "Bearer " + headers[names["x-api-key"]]}


def _get_json(url, headers, timeout_s, opener):
    # An explicit client name: some gateways (Cloudflare-fronted, e.g. OpenCode Console) refuse urllib's default
    # "Python-urllib/x" agent with HTTP 403 (error code 1010).
    agent = {} if any(name.lower() == "user-agent" for name in headers) else {"User-Agent": USER_AGENT}
    request = urllib.request.Request(url, headers={"Accept": "application/json", **agent, **headers}, method="GET")
    with (opener or urllib.request.urlopen)(request, timeout=timeout_s) as response:
        raw = response.read(_MAX_PAGE_BYTES + 1)
    if len(raw) > _MAX_PAGE_BYTES:
        raise ValueError("response_too_large")
    try:
        return json.loads(raw)
    except ValueError:
        raise ValueError("invalid_json") from None


def _failure_reason(error):
    if isinstance(error, urllib.error.HTTPError):
        return f"http_{error.code}"
    if isinstance(error, (socket.timeout, TimeoutError)) or isinstance(getattr(error, "reason", None),
                                                                        (socket.timeout, TimeoutError)):
        return "timeout"
    if isinstance(error, ValueError) and str(error) in ("invalid_json", "invalid_shape", "response_too_large"):
        return str(error)
    return type(error).__name__


def fetch_model_catalogue(url, headers, *, timeout_s=DEFAULT_TIMEOUT_S, max_pages=DEFAULT_MAX_PAGES, opener=None):
    """Read the listing at ``url`` (page after page, at most ``max_pages``) and return a ``ModelCatalogue``.

    ``headers``: host credential headers (plus e.g. ``anthropic-version``); sent only to ``url``'s host. When the
    first page answers 401/403 and the credential is an ``x-api-key`` only, the read is repeated once with the same
    key as ``Authorization: Bearer`` (OpenAI-style listings require it even where ``/v1/messages`` accepts
    ``x-api-key``); ``auth`` records which form worked. ``timeout_s``: socket timeout per page (s). ``opener``:
    ``urllib.request.urlopen``-compatible callable (tests). Never raises for an HTTP, network or format problem: a
    first-page failure is ``status="unavailable"``, a later one ``status="truncated"`` with the pages read so far.
    """
    headers = dict(headers or {})
    models, pages, auth, query, cursors = [], 0, "configured", None, set()
    while True:
        if pages >= max_pages:
            return ModelCatalogue("truncated", "page_cap", tuple(models), pages, auth)
        target = _with_query(url, query) if query else url
        try:
            try:
                document = _get_json(target, headers, timeout_s, opener)
            except urllib.error.HTTPError as error:
                fallback = _bearer_fallback(headers) if pages == 0 and error.code in (401, 403) else None
                if fallback is None:
                    raise
                headers, auth = fallback, "bearer_fallback"
                document = _get_json(target, headers, timeout_s, opener)
            page, query = parse_models_page(document)
        except Exception as error:  # noqa: BLE001 - a catalogue failure is a status, never a launch failure
            reason = _failure_reason(error)
            if pages == 0:
                return ModelCatalogue("unavailable", reason, (), 0, None)
            return ModelCatalogue("truncated", reason, tuple(models), pages, auth)
        pages += 1
        models.extend(page)
        if query is None:
            return ModelCatalogue("ok", None, tuple(models), pages, auth)
        key = tuple(sorted(query.items()))
        if key in cursors:
            return ModelCatalogue("truncated", "cursor_repeated", tuple(models), pages, auth)
        cursors.add(key)


def fetch_upstream_catalogue(base_url, headers, *, wire_api, timeout_s=DEFAULT_TIMEOUT_S,
                             max_pages=DEFAULT_MAX_PAGES, opener=None, listing_url=None):
    """``fetch_model_catalogue`` of an upstream in the relay's convention (``models_url``): ``base_url`` and
    ``headers`` are what the relay forwards model calls with; on ``anthropic_messages`` the ``anthropic-version``
    header is added when absent. ``listing_url``: explicit listing URL (a provider profile's
    ``wires.<wire>.models_url``) used instead of ``models_url(base_url, wire_api)``; None (default) = derived."""
    headers = dict(headers or {})
    if wire_api == "anthropic_messages" and not any(name.lower() == "anthropic-version" for name in headers):
        headers["anthropic-version"] = ANTHROPIC_VERSION
    return fetch_model_catalogue(listing_url or models_url(base_url, wire_api), headers, timeout_s=timeout_s,
                                 max_pages=max_pages, opener=opener)


# ---- typed failures ---------------------------------------------------------------------------------------------
# Typed failure of one catalogue read, derived from (status, reason) only, so callers branch on a kind and never parse
# a reason string. key_refused: HTTP 401/403, the provider refused the credential (a handoff refuses such a profile);
# not_found: 404, no listing at that URL; rate_limited: 429; request_refused: another 4xx; provider_error: 5xx;
# timeout: no answer within the per-page timeout; network: DNS, connection or TLS failure (reason = exception type);
# invalid_listing: not JSON, no ``data`` list, or a page above 8 MiB; incomplete: status ``truncated`` (page cap,
# repeated cursor or a later page failed; the models read so far are kept); credential_unavailable: the profile gave
# no usable credential (reason ``credential_unavailable:<code>``, set by ``CatalogueCache``); not_read: the source
# has no listing.
FAILURE_KINDS = ("key_refused", "not_found", "rate_limited", "request_refused", "provider_error", "timeout",
                 "network", "invalid_listing", "incomplete", "credential_unavailable", "not_read")
CREDENTIAL_REASON = "credential_unavailable"


def failure_kind(status, reason):
    """The ``FAILURE_KINDS`` entry of a read with ``status`` (``ok`` | ``truncated`` | ``unavailable`` |
    ``not_read``) and ``reason`` (``ModelCatalogue.reason``); None for ``ok``."""
    if status == "ok":
        return None
    if status == "not_read":
        return "not_read"
    if status == "truncated":
        return "incomplete"
    reason = reason if isinstance(reason, str) else ""
    match = re.fullmatch(r"http_(\d{3})", reason)
    if match:
        code = int(match.group(1))
        if code in (401, 403):
            return "key_refused"
        return {404: "not_found", 429: "rate_limited"}.get(code, "provider_error" if code >= 500 else "request_refused")
    if reason == "timeout":
        return "timeout"
    if reason in ("invalid_json", "invalid_shape", "response_too_large"):
        return "invalid_listing"
    if reason == CREDENTIAL_REASON or reason.startswith(CREDENTIAL_REASON + ":"):
        return "credential_unavailable"
    return "network"


# ---- pinned native Codex catalogue ------------------------------------------------------------------------------
_CODEX_MODELS = Path(__file__).resolve().parents[1] / "session" / "codex_models.json"


@functools.lru_cache(maxsize=1)
def _codex_models():
    """``{slug: entry}`` of the pinned native Codex catalogue (``cortex/session/codex_models.json``); {} when the file
    cannot be read."""
    try:
        document = json.loads(_CODEX_MODELS.read_text())
    except (OSError, ValueError):
        return {}
    models = document.get("models") if isinstance(document, dict) else None
    return {entry["slug"]: entry for entry in models or () if isinstance(entry, dict)
            and isinstance(entry.get("slug"), str)}


def native_codex_entry(model_id):
    """A copy of the pinned native Codex catalogue entry of ``model_id`` (dict with ``supported_reasoning_levels``,
    ``service_tiers``, ``context_window``, ...), or None for an id the catalogue does not know."""
    entry = _codex_models().get(model_id) if isinstance(model_id, str) else None
    return json.loads(json.dumps(entry)) if entry is not None else None


def native_codex_model(model_id):
    """``CatalogueModel`` of ``model_id`` from the pinned native Codex catalogue, or None for an unknown id.

    ``name`` = ``display_name``; ``image_input`` = ``"image" in input_modalities``; ``context_window`` (tokens);
    ``reasoning_efforts`` = the ``effort`` names of ``supported_reasoning_levels`` (``reasoning`` True when any);
    ``reasoning_default_effort`` = ``default_reasoning_level``; ``effort_updates`` =
    ``supports_reasoning_effort_updates``. Used for ids a Responses listing gives without metadata (design 16 §6)."""
    entry = _codex_models().get(model_id) if isinstance(model_id, str) else None
    if entry is None:
        return None
    inputs = entry.get("input_modalities")
    efforts = _efforts(entry.get("supported_reasoning_levels"))
    name, updates = entry.get("display_name"), entry.get("supports_reasoning_effort_updates")
    return CatalogueModel(id=model_id, name=name if isinstance(name, str) and name else None,
                          image_input=("image" in inputs) if isinstance(inputs, list) else None,
                          context_window=_tokens(entry.get("context_window")),
                          reasoning=True if efforts else None, reasoning_efforts=efforts,
                          reasoning_default_effort=_effort_name(entry.get("default_reasoning_level")),
                          effort_updates=updates if isinstance(updates, bool) else None)


def merge_model(listed, fallback):
    """``listed`` (a ``CatalogueModel`` or None) with every field it does not report (None) taken from ``fallback``
    (for example ``native_codex_model(listed.id)``); ``id`` stays ``listed.id``. Returns the other one when either is
    None."""
    if listed is None or fallback is None:
        return listed if fallback is None else fallback
    return replace(listed, **{item.name: getattr(fallback, item.name) for item in fields(listed)
                              if item.name != "id" and getattr(listed, item.name) is None})


# ---- catalogue cache --------------------------------------------------------------------------------------------
CACHE_TTL_S = 600.0                 # s a read (also a failed one) is served without a new request (design 16 §6)
CACHE_SCHEMA = "ni_rua.model_catalogue_cache.v1"
CACHE_DIR_NAME = ".cache"           # cache directory under the profiles directory: <profiles_dir>/.cache
_CACHE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}\Z")   # profile id = profile file stem (design 16 §3.1)


def _utc(epoch):
    if epoch is None:
        return None
    return datetime.fromtimestamp(epoch, timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _epoch(text):
    if not isinstance(text, str):
        return None
    try:
        parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
    except ValueError:
        return None
    return (parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)).timestamp()


@dataclass(frozen=True)
class CatalogueView:
    """What the operator console shows for one (profile, wire) listing (``GET /v1/console/profiles/{id}/models``).

    ``status``: ``ok`` | ``truncated`` | ``unavailable`` | ``not_read`` (the source has no listing) | ``loading``
    (``peek`` only: the first read is in flight and nothing is cached). ``reason``: code of a non-ok status
    (``http_401``, ``timeout``, ``page_cap``, ``credential_unavailable:<code>``, ...), never response text.
    ``failure``: ``failure_kind(status, reason)`` (one of ``FAILURE_KINDS``), None for ``ok`` and ``loading``.
    ``models``: ``CatalogueModel`` tuple shown: the last read's list or, when that read failed, the list of the last
    successful read of the same profile version (``stale`` True). ``fetched_at`` / ``checked_at``: UTC ISO-8601
    (seconds, ``Z``) of the read that produced ``models`` / of the last read attempt; None when there was none.
    ``age_s``: seconds since ``fetched_at`` (0.1 s), None without a successful read. ``stale``: ``models`` come from an
    earlier read than the last attempt. ``pages`` / ``auth``: of the read that produced ``models``. ``origin``:
    ``network`` (read by this call), ``shared`` (joined another caller's read), ``memory`` or ``disk`` (cached).
    ``refreshing``: a read for this key is in flight (``peek`` only)."""
    status: str
    reason: str | None = None
    failure: str | None = None
    models: tuple = ()
    fetched_at: str | None = None
    checked_at: str | None = None
    age_s: float | None = None
    stale: bool = False
    pages: int = 0
    auth: str | None = None
    origin: str | None = None
    refreshing: bool = False

    def usable(self, wire_api):
        """``usable_models(self.models, wire_api)``."""
        return usable_models(self.models, wire_api)

    def to_dict(self):
        """JSON-ready dict; ``models`` as ``CatalogueModel.to_dict()`` items."""
        value = asdict(self)
        value["models"] = [model.to_dict() for model in self.models]
        return value


@dataclass(frozen=True)
class _Entry:
    profile_mtime: object
    status: str
    reason: str | None
    models: tuple
    pages: int
    auth: str | None
    fetched_at: float | None     # epoch s of the read that produced models
    checked_at: float            # epoch s of the last attempt
    stale: bool


class _Flight:
    def __init__(self):
        self.done = threading.Event()
        self.view = None


def _credential_reason(error):
    """``credential_unavailable:<code>`` for an exception of the credential callable: the leading stable code of its
    message (``profile_missing_key:<file>: ...`` -> ``profile_missing_key``) or the exception type; never the rest of
    the message (it could carry a path or a secret)."""
    code = str(error).split(":", 1)[0].strip()
    if not re.fullmatch(r"[a-z][a-z0-9_]{0,79}", code):
        code = type(error).__name__
    return f"{CREDENTIAL_REASON}:{code}"


class CatalogueCache:
    """Model listings of provider profiles for the operator console: one per (profile id, wire) (design 16 §6).

    ``directory``: cache directory (``<profiles_dir>/.cache``, ``CACHE_DIR_NAME``) or None = memory only. Created on
    the first write (mode 0700) with a ``.gitignore`` of ``*``, so no checkout ever tracks it; one file
    ``<id>.<wire>.json`` per key holding status, times, the profile version and the models. Never a credential, a
    header or a URL. Disk problems are ignored (the cache is a convenience; the list is read again).
    ``ttl_s``: seconds a read, also a failed one, is answered from the cache without a request (default
    ``CACHE_TTL_S`` = 600; the console's refresh / retry passes ``refresh=True``). ``timeout_s`` / ``max_pages`` /
    ``opener``: passed to ``fetch_upstream_catalogue`` (per-page socket timeout in s, page cap, urlopen-compatible
    callable for tests). ``clock``: ``() -> epoch seconds`` (wall clock, because the disk copy outlives the process).
    Thread-safe; the gateway calls ``read`` from worker threads (``asyncio.to_thread``)."""

    def __init__(self, directory=None, *, ttl_s=CACHE_TTL_S, timeout_s=DEFAULT_TIMEOUT_S, max_pages=DEFAULT_MAX_PAGES,
                 opener=None, clock=time.time):
        self.directory = Path(directory) if directory is not None else None
        self.ttl_s, self.timeout_s, self.max_pages, self.opener, self.clock = ttl_s, timeout_s, max_pages, opener, clock
        self._lock = threading.Lock()
        self._entries = {}
        self._flights = {}

    # -- public -----------------------------------------------------------------------------------------------------
    def read(self, profile_id, wire_api, *, credential, profile_mtime=None, listing_url=None, refresh=False):
        """``CatalogueView`` of profile ``profile_id`` (file stem, ``[a-z0-9][a-z0-9_-]{0,39}``) on ``wire_api``
        (a ``WIRE_PROTOCOLS`` key); ``ValueError("catalogue_cache_key_invalid")`` otherwise.

        ``credential``: ``() -> (base_url, headers)`` of the profile on that wire (base URL in the relay's convention),
        called only when a request is made; its result is used for that one request and never stored. Returning None
        means the source has no listing (``not_read``); raising gives ``unavailable`` with reason
        ``credential_unavailable:<code>``. ``profile_mtime``: version token of the profile file (``st_mtime_ns``
        recommended; any JSON scalar); an entry cached for another value is discarded (the profile changed).
        ``listing_url``: the profile's explicit ``models_url`` or None (derived). ``refresh``: read now even when the
        cached answer is fresh.

        Without ``refresh``, an entry of the same ``profile_mtime`` whose last attempt is younger than ``ttl_s`` is
        returned without a request (``origin`` ``memory`` / ``disk``). Concurrent calls for one key share one request
        (``origin`` ``shared``). A failed read returns the last successful list of the same profile version with
        ``stale=True`` and the failure's ``status``/``reason``. Never raises for an HTTP, network or credential
        problem."""
        key = self._key(profile_id, wire_api)
        with self._lock:
            entry, origin = self._cached(key, profile_mtime)
            if entry is not None and not refresh and self.clock() - entry.checked_at < self.ttl_s:
                return self._view(entry, origin)
            flight = self._flights.get(key)
            leader = flight is None
            if leader:
                flight = self._flights[key] = _Flight()
        if not leader:
            flight.done.wait()
            if flight.view is None:
                raise RuntimeError("catalogue_read_failed")
            return replace(flight.view, origin="shared")
        view = None
        try:
            view = self._read_upstream(key, entry, credential, profile_mtime, listing_url)
            return view
        finally:
            with self._lock:
                self._flights.pop(key, None)
                flight.view = view
            flight.done.set()

    def peek(self, profile_id, wire_api, *, profile_mtime=None):
        """The cached ``CatalogueView`` of the same ``profile_mtime`` (any age; judge freshness by ``age_s``) with
        ``refreshing`` True while a read is in flight; a ``loading`` view when the first read is in flight and nothing
        is cached; None when nothing is known. Never makes a request."""
        key = self._key(profile_id, wire_api)
        with self._lock:
            entry, origin = self._cached(key, profile_mtime)
            refreshing = key in self._flights
        if entry is None:
            return CatalogueView("loading", refreshing=True) if refreshing else None
        return replace(self._view(entry, origin), refreshing=refreshing)

    def invalidate(self, profile_id, wire_api=None):
        """Forget the memory and disk entries of ``profile_id`` on ``wire_api`` (None = every wire), e.g. after the
        profile was saved or removed."""
        wires = (wire_api,) if wire_api is not None else tuple(WIRE_PROTOCOLS)
        for wire in wires:
            key = self._key(profile_id, wire)
            with self._lock:
                self._entries.pop(key, None)
            path = self._path(key)
            if path is not None:
                try:
                    path.unlink()
                except OSError:
                    pass

    # -- internals ----------------------------------------------------------------------------------------------------
    @staticmethod
    def _key(profile_id, wire_api):
        if not isinstance(profile_id, str) or not _CACHE_ID.match(profile_id) or wire_api not in WIRE_PROTOCOLS:
            raise ValueError("catalogue_cache_key_invalid")
        return profile_id, wire_api

    def _path(self, key):
        return None if self.directory is None else self.directory / f"{key[0]}.{key[1]}.json"

    def _cached(self, key, profile_mtime):
        """``(entry, origin)`` of ``key`` for ``profile_mtime`` (memory first, then disk), ``(None, None)`` when none
        matches. Caller holds the lock."""
        entry, origin = self._entries.get(key), "memory"
        if entry is None:
            entry, origin = self._load(key), "disk"
            if entry is not None:
                self._entries[key] = entry
        if entry is None or entry.profile_mtime != profile_mtime:
            return None, None
        return entry, origin

    def _view(self, entry, origin):
        now = self.clock()
        age = None if entry.fetched_at is None else round(max(0.0, now - entry.fetched_at), 1)
        return CatalogueView(status=entry.status, reason=entry.reason, failure=failure_kind(entry.status, entry.reason),
                             models=entry.models, fetched_at=_utc(entry.fetched_at), checked_at=_utc(entry.checked_at),
                             age_s=age, stale=entry.stale, pages=entry.pages, auth=entry.auth, origin=origin)

    def _read_upstream(self, key, previous, credential, profile_mtime, listing_url):
        try:
            source = credential()
        except Exception as error:  # noqa: BLE001 - a bad profile is a listing status, never a console failure
            catalogue = ModelCatalogue("unavailable", _credential_reason(error))
        else:
            if source is None:
                catalogue = ModelCatalogue("not_read")
            else:
                base_url, headers = source
                catalogue = fetch_upstream_catalogue(base_url, headers, wire_api=key[1], timeout_s=self.timeout_s,
                                                     max_pages=self.max_pages, opener=self.opener,
                                                     listing_url=listing_url)
            del source
        now = self.clock()
        if catalogue.status in ("ok", "truncated"):
            entry = _Entry(profile_mtime, catalogue.status, catalogue.reason, tuple(catalogue.models), catalogue.pages,
                           catalogue.auth, now, now, False)
        else:
            good = previous if (catalogue.status == "unavailable" and previous is not None
                                and previous.fetched_at is not None) else None
            entry = _Entry(profile_mtime, catalogue.status, catalogue.reason, good.models if good else (),
                           good.pages if good else 0, good.auth if good else None,
                           good.fetched_at if good else None, now, good is not None)
        with self._lock:
            self._entries[key] = entry
        self._store(key, entry)
        return self._view(entry, "network")

    def _load(self, key):
        path = self._path(key)
        if path is None:
            return None
        try:
            document = json.loads(path.read_text())
            if (not isinstance(document, dict) or document.get("schema") != CACHE_SCHEMA
                    or document.get("id") != key[0] or document.get("wire_api") != key[1]):
                return None
            checked = _epoch(document.get("checked_at"))
            if checked is None or document.get("status") not in ("ok", "truncated", "unavailable", "not_read"):
                return None
            models = tuple(CatalogueModel.from_dict(item) for item in document.get("models") or ())
            pages = document.get("pages")
            return _Entry(document.get("profile_mtime"), document["status"], document.get("reason"), models,
                          pages if type(pages) is int else 0, document.get("auth"), _epoch(document.get("fetched_at")),
                          checked, bool(document.get("stale")))
        except (OSError, ValueError, TypeError):
            return None

    def _store(self, key, entry):
        path = self._path(key)
        if path is None:
            return
        document = dict(schema=CACHE_SCHEMA, id=key[0], wire_api=key[1], profile_mtime=entry.profile_mtime,
                        status=entry.status, reason=entry.reason, stale=entry.stale, pages=entry.pages,
                        auth=entry.auth, fetched_at=_utc(entry.fetched_at), checked_at=_utc(entry.checked_at),
                        models=[model.to_dict() for model in entry.models])
        try:
            self.directory.mkdir(mode=0o700, parents=True, exist_ok=True)
            ignore = self.directory / ".gitignore"
            if not ignore.exists():
                ignore.write_text("*\n")
            handle, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=self.directory)
            try:
                with os.fdopen(handle, "w") as stream:
                    json.dump(document, stream, ensure_ascii=False, separators=(",", ":"))
                os.replace(temporary, path)
            except BaseException:
                try:
                    os.unlink(temporary)
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError):
            pass
