"""``agent.opencode.upstream.source = "opencode_config"``: upstream endpoint and key from an OpenCode config file in the
tool's official format (the project-local, git-ignored ``configs/opencode.local.json``), read by the gateway only.

Both OpenCode config shapes are read (one file uses one of them):

* V1 (opencode 1.x): ``{"provider": {"<id>": {"npm": "@ai-sdk/anthropic" | "@ai-sdk/openai", "name": ...,
  "options": {"baseURL": ..., "apiKey": ...}, "models": {...}}}}``;
* V2 (opencode 2.x): ``{"providers": {"<id>": {"package": "aisdk:@ai-sdk/anthropic" | "aisdk:@ai-sdk/openai",
  "settings": {"baseURL": ..., "apiKey": ...}, ...}}}`` (the ``aisdk:`` prefix is optional, as in OpenCode).

The bundled AI SDK requests ``<baseURL>/messages`` (``@ai-sdk/anthropic``, default ``https://api.anthropic.com/v1``)
or ``<baseURL>/responses`` (``@ai-sdk/openai``, default ``https://api.openai.com/v1``). The relay's upstream base
convention differs per wire (Anthropic Messages: no ``/v1``, the relay appends ``/v1/messages``; Responses: with it,
the relay appends ``/responses``), so ``baseURL`` is mapped to the same final URL the SDK would call. The key is sent
as the SDK sends it: ``x-api-key`` (Anthropic) or ``Authorization: Bearer`` (OpenAI). Only a literal ``apiKey`` is
read: OpenCode's ``{env:...}`` / ``{file:...}`` substitutions are refused (no environment variables, no extra key
files). Errors are ``ValueError`` naming the file (and the provider id), never the key. The key is held in memory
only (``OpenCodeUpstream.headers``, excluded from ``repr``); it never enters ``Config.to_dict()``, events, the agent's
environment or the generated OpenCode config (which keeps the relay placeholder key).
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

__all__ = ["PACKAGE_WIRES", "SDK_DEFAULT_BASE_URLS", "OpenCodeUpstream", "read_opencode_config_upstream",
           "configured_wire"]

# Provider SDK package (``npm`` / ``package`` without the ``aisdk:`` prefix) -> relay wire.
PACKAGE_WIRES = {"@ai-sdk/anthropic": "anthropic_messages", "@ai-sdk/openai": "openai_responses"}
# baseURL the bundled SDK uses when the file sets none.
SDK_DEFAULT_BASE_URLS = {"anthropic_messages": "https://api.anthropic.com/v1",
                         "openai_responses": "https://api.openai.com/v1"}
_PACKAGE_PREFIX = "aisdk:"
_SUBSTITUTION = re.compile(r"\{(?:env|file):[^}]*\}")


@dataclass(frozen=True)
class OpenCodeUpstream:
    """The selected provider of an OpenCode config file, in the relay's terms.

    ``provider_id``: the ``<id>`` key. ``wire_api``: ``anthropic_messages`` | ``openai_responses``. ``base_url``:
    relay upstream base (see module doc). ``headers``: host credential header (never in ``repr``). ``shape``:
    ``"v1"`` (``provider``) or ``"v2"`` (``providers``). ``package``: the SDK package without prefix, or None."""
    provider_id: str
    wire_api: str
    base_url: str
    headers: dict = field(repr=False)
    shape: str
    package: str | None = None


def _load(path):
    location = Path(path).expanduser()
    failure = None
    data = None
    try:
        data = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        failure = type(error).__name__
    if failure is not None:
        raise ValueError(f"opencode_config_unreadable:{location}: {failure} (plain JSON, no comments)")
    if not isinstance(data, dict):
        raise ValueError(f"opencode_config_unreadable:{location}: expected a JSON object")
    return location, data


def _providers(location, data):
    """``(shape, {id: entry})`` of the file; ``ValueError`` for a mixed, empty or malformed provider section."""
    if "provider" in data and "providers" in data:
        raise ValueError(f'opencode_config_mixed_shapes:{location}: use either "provider" (V1) or "providers" (V2)')
    shape, key = ("v2", "providers") if "providers" in data else ("v1", "provider")
    entries = data.get(key)
    if not isinstance(entries, dict) or not entries:
        raise ValueError(f"opencode_config_no_provider:{location}: declare provider.<id> (V1) or providers.<id> (V2)")
    return shape, entries


def _select(location, entries, provider):
    if provider is not None:
        if provider not in entries:
            raise ValueError(f"opencode_config_provider_missing:{location}: {provider} is not declared "
                             f"(declared: {', '.join(sorted(map(str, entries)))})")
        return provider
    if len(entries) != 1:
        raise ValueError(f"opencode_config_provider_ambiguous:{location}: set agent.opencode.upstream.provider to "
                         f"one of {', '.join(sorted(map(str, entries)))}")
    return next(iter(entries))


def _package(entry, shape):
    value = entry.get("package" if shape == "v2" else "npm")
    if not isinstance(value, str) or not value.strip():
        return None
    value = value.strip()
    return value[len(_PACKAGE_PREFIX):] if value.startswith(_PACKAGE_PREFIX) else value


def _relay_base_url(location, provider_id, wire_api, base_url):
    """The relay base for an SDK ``baseURL`` (see module doc); ``ValueError`` when it cannot be represented."""
    if base_url is None:
        base_url = SDK_DEFAULT_BASE_URLS[wire_api]
    if not isinstance(base_url, str) or _SUBSTITUTION.search(base_url):
        raise ValueError(f"opencode_config_invalid_base_url:{location}:{provider_id}: write baseURL literally")
    base = base_url.strip().rstrip("/")
    bad_url = False
    parts = None
    try:
        parts = urlsplit(base)
        parts.port  # noqa: B018 - raises ValueError for a malformed port
    except ValueError:
        bad_url = True
    if bad_url:
        raise ValueError(f"opencode_config_invalid_base_url:{location}:{provider_id}")
    if (parts.scheme not in ("http", "https") or not parts.hostname or parts.username is not None
            or parts.password is not None or parts.query or parts.fragment):
        raise ValueError(f"opencode_config_invalid_base_url:{location}:{provider_id}: http(s) URL without "
                         "credentials, query or fragment")
    if wire_api == "anthropic_messages":
        if not base.endswith("/v1"):
            raise ValueError(f"opencode_config_base_url_unsupported:{location}:{provider_id}: an anthropic_messages "
                             "baseURL must end in /v1 (the SDK calls <baseURL>/messages, the relay forwards to "
                             "<base>/v1/messages)")
        return base[:-len("/v1")]
    return base


def read_opencode_config_upstream(path, *, provider=None, wire_api=None):
    """``OpenCodeUpstream`` of the provider ``provider`` (or the only one) of the OpenCode config file ``path``.

    ``wire_api``: ``agent.opencode.upstream.wire_api`` when set (wins over the package), else from the package
    (``PACKAGE_WIRES``). Raises ``ValueError`` (``opencode_config_*:<file>[:<id>]: ...``) for an unreadable file,
    mixed shapes, no / ambiguous / unknown provider, an unknown package without ``wire_api``, an unsupported wire, a
    missing, substituted (``{env:...}``/``{file:...}``) or non-printable ``apiKey``, or a ``baseURL`` the relay
    cannot represent; never with the key value.
    """
    location, data = _load(path)
    shape, entries = _providers(location, data)
    provider_id = _select(location, entries, provider)
    entry = entries[provider_id]
    section = "settings" if shape == "v2" else "options"
    options = entry.get(section) if isinstance(entry, dict) else None
    if not isinstance(entry, dict) or (options is not None and not isinstance(options, dict)):
        raise ValueError(f"opencode_config_provider_invalid:{location}:{provider_id}: expected an object")
    options = options or {}
    package = _package(entry, shape)
    wire = wire_api or PACKAGE_WIRES.get(package)
    if wire is None:
        raise ValueError(f"opencode_config_wire_unknown:{location}:{provider_id}: package {package or 'unset'}; use "
                         "@ai-sdk/anthropic or @ai-sdk/openai, or set agent.opencode.upstream.wire_api")
    if wire not in SDK_DEFAULT_BASE_URLS:
        raise ValueError(f"opencode_config_wire_unsupported:{location}:{provider_id}: {wire}; use anthropic_messages "
                         "or openai_responses")
    key_path = f"{'providers' if shape == 'v2' else 'provider'}.{provider_id}.{section}.apiKey"
    key = options.get("apiKey")
    if not isinstance(key, str) or not key.strip():
        raise ValueError(f"opencode_config_missing_key:{location}:{provider_id}: set {key_path}")
    if _SUBSTITUTION.search(key):
        raise ValueError(f"opencode_config_key_substitution_unsupported:{location}:{provider_id}: write {key_path} "
                         "literally ({env:...} and {file:...} are not read)")
    key = key.strip()
    if any(ord(char) < 33 or ord(char) > 126 for char in key):
        raise ValueError(f"opencode_config_invalid_key:{location}:{provider_id}: {key_path} must be printable ASCII")
    base_url = _relay_base_url(location, provider_id, wire, options.get("baseURL"))
    headers = {"x-api-key": key} if wire == "anthropic_messages" else {"Authorization": "Bearer " + key}
    return OpenCodeUpstream(provider_id=provider_id, wire_api=wire, base_url=base_url, headers=headers, shape=shape,
                            package=package)


def configured_wire(upstream):
    """Relay wire of an ``UpstreamConfig`` without raising: ``wire_api`` when set, ``openai_responses`` for
    ``codex_home``, the provider package's wire for ``opencode_config`` (None when the file cannot be read; the launch
    gate reports why), else None."""
    if upstream.wire_api is not None:
        return upstream.wire_api
    if upstream.source == "codex_home":
        return "openai_responses"
    if upstream.source == "opencode_config" and upstream.settings:
        try:
            return read_opencode_config_upstream(upstream.settings, provider=upstream.provider).wire_api
        except ValueError:
            return None
    return None
