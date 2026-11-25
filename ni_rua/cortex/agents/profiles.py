"""Provider profiles: one git-ignored JSON file per provider + key, ``<profiles_dir>/<id>.local.json``.

A profile is NI-RUA's one canonical credential format (design 16 section 13, user decision Q1). It holds the provider
key inline and declares the wires it serves, each with its base URL, so one key drives every agent backend whose wire
the profile has (``factory.BACKEND_WIRES``: Codex ``openai_responses``, Claude Code ``anthropic_messages``, OpenCode
either). ``agent.profiles_dir`` names the directory (null = ``<checkout>/configs/profiles``); ``<id>`` is the file stem,
``[a-z0-9][a-z0-9_-]{0,39}``. Files are written atomically with mode 0600 (directory 0700 when created here).

A configuration selects a profile with ``UpstreamConfig(source="profile", settings=<profile path>)`` in
``agent.claude_code.upstream``, ``agent.opencode.upstream`` or ``agent.codex_upstream``. The profile owns the
upstream's non-secret fields (wire_api, base_url, auth_scheme, provider = profile id, tool_result_images,
tool_schema_max_depth; api_key_env null): ``Config.load`` and ``apply_profile`` write them from the profile, so the
archived ``effective_config.json`` shows the concrete endpoint, and the launch gate (``factory.backend_requirements``
through ``resolve_profile_upstream``) refuses a configuration that no longer matches its profile. Only
``credential`` / ``resolve_profile_upstream`` touch the key; it never enters ``Config``, ``Config.to_dict()``,
``ProfileInfo``, ``repr``, exception messages, events or logs. Errors are ``ValueError("profile_<code>:<file>: ...")``
naming the file and the field, never a value.

Example (all values are placeholders)::

    {
      "schema": "ni_rua.provider_profile.v1",
      "name": "Example Gateway",
      "credential": {"api_key": "sk-REPLACE-WITH-YOUR-KEY"},
      "wires": {
        "anthropic_messages": {"base_url": "https://gateway.example/gateway", "auth_scheme": "bearer"},
        "openai_responses": {"base_url": "https://gateway.example/gateway/v1", "auth_scheme": "bearer"}
      },
      "compat": {"tool_result_images": "user_blocks", "tool_schema_max_depth": 13},
      "backends": {
        "claude_code": {"env": {"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"}},
        "opencode": {"thinking_display": null}
      },
      "defaults": {"model": "example-model", "effort": "high"}
    }

One-shot import of the official-format key files that predate profiles (Claude Code settings files
``configs/claude*.local.json``, OpenCode config files ``configs/opencode*.local.json`` in the V1 ``provider`` and V2
``providers`` shapes, Codex provider homes ``~/.config/ni-rua/providers/<name>/{config.toml,auth.json}``)::

    mamba run -n ni-rua python -m ni_rua.cortex.agents.profiles import --dry-run   # print the plan, write nothing
    mamba run -n ni-rua python -m ni_rua.cortex.agents.profiles import             # write the profiles
    mamba run -n ni-rua python -m ni_rua.cortex.agents.profiles list               # the profiles, never a key

Files that hold the same key are merged into one profile with several wires where nothing conflicts (see
``legacy_profile_import.plan_import``); compat opt-ins and OpenCode ``thinking_display`` come from the run configurations that reference
the files (``configs/ni-rua.json``, ``configs/ni-rua.local.json``). The old files are never changed or deleted and an
existing profile is never overwritten. The legacy upstream sources (``claude_settings``, ``opencode_config``,
``codex_home``, ``explicit``) stay readable for existing configurations, archives and resume.

Field contract (the web form's help text is ``FIELD_HELP``, the same table):
"""
from __future__ import annotations

import argparse
import contextlib
import copy
import json
import os
import re
import sys
import tempfile
from dataclasses import asdict, dataclass, field, fields, replace
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

from ...core.config import (DEFAULT_PROFILES_DIR, Config, ConfigError, ModelConfig, OpenCodeConfig, UpstreamConfig)
from .registry import BACKENDS_REGISTRY, _PROFILE_UNSET, actor_upstream, get_backend

BACKENDS = BACKENDS_REGISTRY.hosted_kinds()
BACKEND_WIRES = BACKENDS_REGISTRY.backend_wires()

__all__ = [
    "SCHEMA", "PROFILE_SUFFIX", "PROFILE_ID", "PROFILE_WIRES", "FIELD_HELP", "Profile", "ProfileWire", "ProfileInfo",
    "ProfileUpstream", "SelectionCheck", "profiles_directory", "profile_path", "load_profile", "list_profiles",
    "public_view", "resolve_wire", "credential", "forwarded_env", "upstream_fields", "expand_upstream",
    "expand_profile_references", "apply_profile", "resolve_profile_upstream", "compatibility", "matrix_cell",
    "check_selection", "save_profile", "save_profile_key", "remove_profile", "main",
]

SCHEMA = "ni_rua.provider_profile.v1"
PROFILE_SUFFIX = ".local.json"
PROFILE_ID = re.compile(r"[a-z0-9][a-z0-9_-]{0,39}")
# Every wire some backend runs on (anthropic_messages, openai_responses).
PROFILE_WIRES = tuple(sorted({wire for backend in BACKENDS for wire in BACKEND_WIRES[backend]}))


def _choices(section, name):
    return next(item.metadata["choices"] for item in fields(section) if item.name == name)


AUTH_SCHEMES = _choices(UpstreamConfig, "auth_scheme")                    # bearer, x-api-key
TOOL_RESULT_IMAGES = _choices(UpstreamConfig, "tool_result_images")       # inline, user_blocks
THINKING_DISPLAY_CHOICES = _choices(OpenCodeConfig, "thinking_display")   # summarized, omitted
EFFORT_CHOICES = _choices(ModelConfig, "effort")
NAME_MAX_CHARS = 60
MODEL_MAX_CHARS = 200
SCHEMA_DEPTH_MINIMUM = 3                                                  # as UpstreamConfig.tool_schema_max_depth

FIELD_HELP = (
    ("schema", f'Must be "{SCHEMA}".'),
    ("name", f"Display name in the operator console, 1-{NAME_MAX_CHARS} characters. Default: the id."),
    ("credential.api_key", "The provider key, printable ASCII without spaces. Read by the gateway only and sent "
     "upstream in the header that wires.<wire>.auth_scheme names; never shown again, never archived, never given to "
     "the agent. Absent = the profile has no key yet (the launch gate refuses it: profile_missing_key)."),
    ("wires.<wire>", "One entry per wire the provider serves: anthropic_messages (Claude Code, OpenCode) and/or "
     "openai_responses (Codex, OpenCode). At least one."),
    ("wires.<wire>.base_url", "Upstream base URL in the relay's convention, http(s) without credentials, query or "
     "fragment: Anthropic Messages without /v1 (the relay appends /v1/messages), OpenAI Responses with /v1 (the relay "
     "appends /responses). A trailing / is dropped."),
    ("wires.<wire>.auth_scheme", '"bearer" (default; Authorization: Bearer <key>) or "x-api-key" (x-api-key: <key>).'),
    ("wires.<wire>.models_url", "Optional explicit model listing URL. Default: model_catalogue.models_url(base_url, "
     "wire)."),
    ("compat.tool_result_images", '"inline" (default; images stay inside tool results) or "user_blocks" (for '
     "providers that refuse images in tool results: each image moves unchanged to a labelled user block). Same "
     "meaning as UpstreamConfig.tool_result_images; Claude Code and OpenCode only."),
    ("compat.tool_schema_max_depth", f"Maximum JSON nesting depth of a tool input schema the provider accepts, integer "
     f">= {SCHEMA_DEPTH_MINIMUM}; null (default) = no limit. Same meaning as UpstreamConfig.tool_schema_max_depth; "
     "Claude Code and OpenCode only."),
    ("backends.claude_code.env", "Variables forwarded to the Claude Code CLI environment, e.g. "
     '{"CLAUDE_CODE_THINKING_DISPLAY_UPDATES": "0"} for a gateway that refuses thinking.display, or '
     "CLAUDE_CODE_MAX_OUTPUT_TOKENS for a model with a small output limit. Names must be shell variable names that "
     "NI-RUA does not set itself (as a Claude settings file's env); values are strings. Default {}."),
    ("backends.opencode.thinking_display", 'null (default; the field is not sent), "summarized" or "omitted": '
     "Anthropic thinking.display of every OpenCode effort variant, used as agent.opencode.thinking_display when the "
     "run does not set one. Needs an anthropic_messages wire and applies on that wire only."),
    ("defaults.model", "Model pre-selected in the console when the profile is chosen. Optional."),
    ("defaults.effort", "Effort pre-selected in the console (one of " + ", ".join(EFFORT_CHOICES) + "). Optional."),
)

__doc__ += "".join(f"\n* ``{name}``: {meaning}" for name, meaning in FIELD_HELP) + "\n"

_TOP_KEYS = ("schema", "name", "credential", "wires", "compat", "backends", "defaults")
_WIRE_KEYS = ("base_url", "auth_scheme", "models_url")
_COMPAT_KEYS = ("tool_result_images", "tool_schema_max_depth")
_REFERENCE_KEYS = ("format", "file", "provider")
# Names echoed in an error: letters and underscores only (a pasted key has digits or dashes and is never echoed).
_SAFE_NAME = re.compile(r"[A-Za-z_]{1,40}")
_ENV_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# Upstream fields a profile owns; the launch gate compares them with the profile (``resolve_profile_upstream``).
_OWNED_FIELDS = ("wire_api", "base_url", "auth_scheme", "api_key_env", "tool_result_images", "tool_schema_max_depth")
_UNSET = _PROFILE_UNSET


class _Secret:
    """The key, held so that ``repr``/``str`` never show it and JSON serialisation fails instead of leaking it."""
    __slots__ = ("_value",)

    def __init__(self, value):
        self._value = value

    def reveal(self):
        return self._value

    def __bool__(self):
        return bool(self._value)

    def __repr__(self):
        return "<redacted>" if self._value else "<absent>"

    __str__ = __repr__

    def __reduce__(self):
        raise TypeError("provider keys are not picklable")


_NO_KEY = _Secret(None)


@dataclass(frozen=True)
class ProfileWire:
    """One wire of a profile. ``base_url``: relay convention (see ``FIELD_HELP``), no trailing ``/``.
    ``auth_scheme``: ``bearer`` | ``x-api-key``. ``models_url``: explicit listing URL or None."""
    base_url: str
    auth_scheme: str = "bearer"
    models_url: str | None = None


@dataclass(frozen=True)
class Profile:
    """A validated profile. ``id``: file stem. ``path``: absolute path of the file. ``wires``: ``{wire: ProfileWire}``.
    The key is held privately (``key_present``); only ``credential`` reads it."""
    id: str
    path: Path
    name: str
    wires: dict
    tool_result_images: str = "inline"
    tool_schema_max_depth: int | None = None
    claude_code_env: dict = field(default_factory=dict)
    opencode_thinking_display: str | None = None
    default_model: str | None = None
    default_effort: str | None = None
    _secret: _Secret = field(default=_NO_KEY, repr=False, compare=False)

    @property
    def key_present(self):
        return bool(self._secret)


@dataclass(frozen=True)
class ProfileInfo:
    """The public view of one profile file for the console and the CLI: never the key.

    ``valid``: the file loads; otherwise ``error`` is ``{"code": "profile_<code>", "message": ...}`` and only ``id``,
    ``path``, ``key_present`` and ``mtime_ns`` are filled. ``wires``: ``{wire: {base_url, host, auth_scheme,
    models_url}}``. ``compat``: ``{tool_result_images, tool_schema_max_depth}``. ``backends``: ``{"claude_code":
    {"env": {...}}, "opencode": {"thinking_display": ...}}``. ``defaults``: ``{model, effort}``. ``mtime_ns``: file
    modification time (ns), e.g. to invalidate cached model listings."""
    id: str
    path: str
    valid: bool
    name: str | None = None
    error: dict | None = None
    key_present: bool = False
    wires: dict = field(default_factory=dict)
    compat: dict = field(default_factory=dict)
    backends: dict = field(default_factory=dict)
    defaults: dict = field(default_factory=dict)
    mtime_ns: int | None = None

    def to_dict(self):
        return json.loads(json.dumps(asdict(self)))


@dataclass(frozen=True)
class ProfileUpstream:
    """What a session needs to forward through a profile: ``wire_api``, ``base_url`` (no trailing ``/``) and the host
    credential ``headers`` (never in ``repr``)."""
    profile: Profile
    wire_api: str
    base_url: str
    headers: dict = field(repr=False)


@dataclass(frozen=True)
class SelectionCheck:
    """Result of ``check_selection``: ``ok``; ``reason`` (code, None when ok) and ``message`` (sentence); the resolved
    ``wire_api``; ``config``: the candidate configuration (None unless ok). ``to_dict`` leaves the config out."""
    ok: bool
    reason: str | None = None
    message: str | None = None
    wire_api: str | None = None
    config: Config | None = field(default=None, repr=False, compare=False)

    def to_dict(self):
        return dict(ok=self.ok, reason=self.reason, message=self.message, wire_api=self.wire_api)


# ---------------------------------------------------------------------------------------------- loading


def _fail(code, location, message):
    raise ValueError(f"profile_{code}:{location}: {message}")


def _code(error):
    return str(error).split(":", 1)[0]


def _safe_names(names):
    """Key names for an error message; anything that does not look like an identifier is not echoed (it could be a
    pasted key)."""
    shown = sorted(str(name) for name in names if isinstance(name, str) and _SAFE_NAME.fullmatch(name))
    hidden = len(names) - len(shown)
    return ", ".join(shown + ([f"<{hidden} name(s) not shown>"] if hidden else []))


def _object(value, location, where, allowed):
    """``value`` as a dict (None -> {}), refusing anything else and unknown keys."""
    if value is None:
        return {}
    if not isinstance(value, dict):
        _fail("not_object" if where == "document" else "invalid_" + where.split(".")[0], location,
              f"{where} must be a JSON object")
    unknown = [key for key in value if key not in allowed]
    if unknown:
        _fail("unknown_keys", location, f"unknown keys in {where}: {_safe_names(unknown)} (allowed: "
              f"{', '.join(allowed)})")
    return value


def _check_url(value, location, where, code):
    if not isinstance(value, str) or not value.strip():
        _fail(code, location, f"{where} must be an http(s) URL")
    url = value.strip().rstrip("/")
    try:
        UpstreamConfig(base_url=url)   # the one URL rule of every upstream
    except ConfigError:
        _fail(code, location, f"{where} must be an http(s) URL without credentials, query or fragment")
    return url


def _check_key(value, location):
    if value is None or value == "":
        return _NO_KEY
    if not isinstance(value, str) or not value.strip() or any(ord(char) < 33 or ord(char) > 126
                                                               for char in value.strip()):
        _fail("invalid_key", location, "credential.api_key must be printable ASCII without spaces")
    return _Secret(value.strip())


def _check_env(env, location):
    """``backends.claude_code.env`` validated as ``claude_options.read_claude_settings_env`` validates a settings
    file's ``env``; credential variables are refused too (the key belongs in ``credential.api_key``)."""
    if env is None or env == {}:
        return {}
    from .claude_code.options import NI_RUA_OWNED_ENV, SETTINGS_CREDENTIAL_ENV
    if not isinstance(env, dict):
        _fail("invalid_backends", location, "backends.claude_code.env must be a JSON object")
    forwarded = {}
    for name, item in env.items():
        if not isinstance(name, str) or not _ENV_NAME.fullmatch(name):
            _fail("env_invalid_name", location, "backends.claude_code.env names must be shell variable names")
        if name in NI_RUA_OWNED_ENV or name in SETTINGS_CREDENTIAL_ENV:
            _fail("env_reserved", location, f"backends.claude_code.env.{name} is set by NI-RUA (the key belongs in "
                  "credential.api_key, the endpoint in wires); remove it")
        if not isinstance(item, str) or "\0" in item:
            _fail("env_not_string", location, f"backends.claude_code.env.{_safe_names([name])} must be a string")
        forwarded[name] = item
    return forwarded


def _parse(document, location, profile_id):
    """``Profile`` of a profile ``document`` (dict) read from ``location``; ``ValueError("profile_<code>:...")``."""
    document = _object(document, location, "document", _TOP_KEYS)
    if document.get("schema") != SCHEMA:
        _fail("schema", location, f'set "schema": "{SCHEMA}"')
    name = document.get("name", profile_id)
    if (not isinstance(name, str) or not name.strip() or len(name.strip()) > NAME_MAX_CHARS
            or any(ord(char) < 32 or ord(char) == 127 for char in name)):
        _fail("invalid_name", location, f"name must be 1-{NAME_MAX_CHARS} printable characters")
    credential_section = document.get("credential")
    if isinstance(credential_section, dict) and any(key in credential_section for key in _REFERENCE_KEYS):
        _fail("reference_unsupported", location, "a profile holds its key inline in credential.api_key; convert an "
              "official-format file with `python -m ni_rua.cortex.agents.profiles import`")
    credential_section = _object(credential_section, location, "credential", ("api_key",))
    secret = _check_key(credential_section.get("api_key"), location)
    wires_section = document.get("wires")
    if not isinstance(wires_section, dict) or not wires_section:
        _fail("no_wires", location, "declare at least one wire in wires (" + ", ".join(PROFILE_WIRES) + ")")
    wires = {}
    for wire, entry in wires_section.items():
        if wire not in PROFILE_WIRES:
            _fail("unknown_wire", location, f"wires.{_safe_names([wire])} is not a relay wire; "
                  f"use {', '.join(PROFILE_WIRES)}")
        entry = _object(entry, location, f"wires.{wire}", _WIRE_KEYS)
        base_url = _check_url(entry.get("base_url"), location, f"wires.{wire}.base_url", "invalid_base_url")
        auth_scheme = entry.get("auth_scheme", "bearer")
        if auth_scheme not in AUTH_SCHEMES:
            _fail("invalid_auth_scheme", location, f"wires.{wire}.auth_scheme must be one of {', '.join(AUTH_SCHEMES)}")
        models_url = entry.get("models_url")
        if models_url is not None:
            models_url = _check_url(models_url, location, f"wires.{wire}.models_url", "invalid_models_url")
        wires[wire] = ProfileWire(base_url=base_url, auth_scheme=auth_scheme, models_url=models_url)
    compat = _object(document.get("compat"), location, "compat", _COMPAT_KEYS)
    images = compat.get("tool_result_images", "inline")
    if images not in TOOL_RESULT_IMAGES:
        _fail("invalid_compat", location, f"compat.tool_result_images must be one of {', '.join(TOOL_RESULT_IMAGES)}")
    depth = compat.get("tool_schema_max_depth")
    if depth is not None and (type(depth) is not int or depth < SCHEMA_DEPTH_MINIMUM):
        _fail("invalid_compat", location, f"compat.tool_schema_max_depth must be null or an integer >= "
              f"{SCHEMA_DEPTH_MINIMUM}")
    sections = BACKENDS_REGISTRY.profile_sections()
    backends = _object(document.get("backends"), location, "backends", tuple(sections))
    parsed_sections = {}
    parse_helpers = {"check_env": _check_env, "fail": _fail}
    for key, section in sections.items():
        sub = _object(backends.get(key), location, f"backends.{key}", section.section_fields)
        parsed_sections[key] = section.parse(sub, wires, location, parse_helpers)
    env = parsed_sections.get("claude_code", {}).get("env", {})
    display = parsed_sections.get("opencode", {}).get("thinking_display")
    defaults = _object(document.get("defaults"), location, "defaults", ("model", "effort"))
    model = defaults.get("model")
    if model is not None and (not isinstance(model, str) or not model or len(model) > MODEL_MAX_CHARS
                              or any(char.isspace() or ord(char) < 32 for char in model)):
        _fail("invalid_defaults", location, f"defaults.model must be a model id of 1-{MODEL_MAX_CHARS} characters "
              "without spaces")
    effort = defaults.get("effort")
    if effort is not None and effort not in EFFORT_CHOICES:
        _fail("invalid_defaults", location, "defaults.effort must be one of " + ", ".join(EFFORT_CHOICES))
    return Profile(id=profile_id, path=Path(location), name=name.strip(), wires=wires, tool_result_images=images,
                   tool_schema_max_depth=depth, claude_code_env=env, opencode_thinking_display=display,
                   default_model=model, default_effort=effort, _secret=secret)


def _document(profile, *, with_key):
    """The canonical JSON document of ``profile`` (``with_key``: include ``credential.api_key`` when present)."""
    document = {"schema": SCHEMA, "name": profile.name}
    if with_key and profile.key_present:
        document["credential"] = {"api_key": profile._secret.reveal()}
    document["wires"] = {wire: {key: value for key, value in asdict(entry).items()
                                if value is not None} for wire, entry in profile.wires.items()}
    if profile.tool_result_images != "inline" or profile.tool_schema_max_depth is not None:
        document["compat"] = {"tool_result_images": profile.tool_result_images,
                              "tool_schema_max_depth": profile.tool_schema_max_depth}
    backends = {}
    for key, section in BACKENDS_REGISTRY.profile_sections().items():
        serialized = section.serialize(profile)
        if serialized is not None:
            backends[key] = serialized
    if backends:
        document["backends"] = backends
    defaults = {key: value for key, value in (("model", profile.default_model), ("effort", profile.default_effort))
                if value is not None}
    if defaults:
        document["defaults"] = defaults
    return document


def _location(path):
    return Path(os.path.abspath(Path(path).expanduser()))


def _profile_id(location):
    name = location.name
    stem = name[:-len(PROFILE_SUFFIX)] if name.endswith(PROFILE_SUFFIX) else None
    if stem is None or not PROFILE_ID.fullmatch(stem):
        _fail("invalid_file_name", location, f"name the file <id>{PROFILE_SUFFIX}, <id> matching "
              f"{PROFILE_ID.pattern}")
    return stem


def _read_document(location):
    """The JSON object in ``location``; the parser's exception (whose ``doc`` holds the text) is not chained."""
    failure = None
    try:
        document = json.loads(location.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        failure = type(error).__name__
    if failure is not None:
        _fail("unreadable", location, f"{failure} (plain JSON)")
    if not isinstance(document, dict):
        _fail("not_object", location, "the file must hold a JSON object")
    return document


def load_profile(path):
    """``Profile`` of the profile file ``path`` (``<id>.local.json``; ``~`` expanded, relative to the cwd).

    Raises ``ValueError("profile_<code>:<file>: ...")`` for an unreadable or invalid file, never with a value of the
    file: ``unreadable``, ``not_object``, ``invalid_file_name``, ``schema``, ``unknown_keys``, ``invalid_name``,
    ``reference_unsupported``, ``invalid_credential``, ``invalid_key``, ``no_wires``, ``unknown_wire``,
    ``invalid_wires``, ``invalid_base_url``, ``invalid_auth_scheme``, ``invalid_models_url``, ``invalid_compat``,
    ``invalid_backends``, ``env_invalid_name``, ``env_reserved``, ``env_not_string``, ``invalid_thinking_display``,
    ``thinking_display_requires_anthropic_messages``, ``invalid_defaults``. A profile without a key loads
    (``key_present`` False).
    """
    location = _location(path)
    profile_id = _profile_id(location)
    return _parse(_read_document(location), location, profile_id)


def profiles_directory(config=None):
    """The profiles directory of ``config`` (``agent.profiles_dir``), ``DEFAULT_PROFILES_DIR`` when unset."""
    configured = getattr(getattr(config, "agent", None), "profiles_dir", None)
    return Path(configured).expanduser() if configured else DEFAULT_PROFILES_DIR


def profile_path(directory, profile_id):
    """``<directory>/<profile_id>.local.json``; ``ValueError("profile_invalid_id:...")`` for a malformed id."""
    if not isinstance(profile_id, str) or not PROFILE_ID.fullmatch(profile_id):
        _fail("invalid_id", _location(directory), f"a profile id matches {PROFILE_ID.pattern}")
    return _location(directory) / (profile_id + PROFILE_SUFFIX)


def _mtime_ns(location):
    try:
        return location.stat().st_mtime_ns
    except OSError:
        return None


def public_view(profile):
    """``ProfileInfo`` of a loaded ``Profile`` (never the key)."""
    wires = {wire: dict(base_url=entry.base_url, host=urlsplit(entry.base_url).hostname, auth_scheme=entry.auth_scheme,
                        models_url=entry.models_url) for wire, entry in profile.wires.items()}
    return ProfileInfo(
        id=profile.id, path=str(profile.path), valid=True, name=profile.name, key_present=profile.key_present,
        wires=wires, compat=dict(tool_result_images=profile.tool_result_images,
                                 tool_schema_max_depth=profile.tool_schema_max_depth),
        backends={key: section.public_view(profile) for key, section in BACKENDS_REGISTRY.profile_sections().items()},
        defaults=dict(model=profile.default_model, effort=profile.default_effort), mtime_ns=_mtime_ns(profile.path))


def _raw_key_present(location):
    with contextlib.suppress(OSError, ValueError, AttributeError):
        value = json.loads(location.read_text(encoding="utf-8")).get("credential", {}).get("api_key")
        return isinstance(value, str) and bool(value.strip())
    return False


def _invalid_info(location, error):
    stem = location.name[:-len(PROFILE_SUFFIX)] if location.name.endswith(PROFILE_SUFFIX) else location.name
    return ProfileInfo(id=stem, path=str(location), valid=False, error=dict(code=_code(error), message=str(error)),
                       key_present=_raw_key_present(location), mtime_ns=_mtime_ns(location))


def list_profiles(directory):
    """``ProfileInfo`` of every ``*.local.json`` file directly in ``directory`` (sorted by file name; symlinks and
    other files skipped). An invalid file is listed with ``valid=False`` and its error, never hidden. A missing
    directory lists nothing."""
    root = _location(directory)
    try:
        entries = sorted(root.iterdir(), key=lambda item: item.name)
    except OSError:
        return []
    result = []
    for entry in entries:
        if not entry.name.endswith(PROFILE_SUFFIX) or entry.is_symlink() or not entry.is_file():
            continue
        try:
            result.append(public_view(load_profile(entry)))
        except ValueError as error:
            result.append(_invalid_info(entry, error))
    return result


# ---------------------------------------------------------------------------------------------- using a profile


def _backend(backend):
    if backend not in BACKEND_WIRES:
        raise ValueError(f"unknown_agent_backend:{backend}: use one of {', '.join(BACKEND_WIRES)}")
    return BACKEND_WIRES[backend]


def resolve_wire(profile, backend, wire_api=None):
    """The wire ``backend`` uses with ``profile``: ``wire_api`` when given, else the only wire of
    ``BACKEND_WIRES[backend]`` the profile serves. Raises ``ValueError``: ``profile_wire_unsupported`` (the backend
    does not run on ``wire_api``), ``profile_wire_not_served`` (the profile does not declare it),
    ``profile_no_common_wire`` (no wire in common), ``profile_wire_ambiguous`` (two candidates; name one)."""
    accepted = _backend(backend)
    served = ", ".join(profile.wires)
    if wire_api is not None:
        if wire_api not in accepted:
            _fail("wire_unsupported", profile.path, f"{backend} runs on {', '.join(accepted)}, not {wire_api}")
        if wire_api not in profile.wires:
            _fail("wire_not_served", profile.path, f"the profile serves {served}, not {wire_api}")
        return wire_api
    common = [wire for wire in accepted if wire in profile.wires]
    if not common:
        _fail("no_common_wire", profile.path, f"the profile serves {served}; {backend} needs {' or '.join(accepted)}")
    if len(common) > 1:
        _fail("wire_ambiguous", profile.path, f"the profile serves {', '.join(common)}; set upstream.wire_api")
    return common[0]


def credential(profile, wire):
    """``(base_url, headers)`` for ``wire`` of ``profile``: the only function that reads the key. ``base_url`` is in
    the relay's convention (no trailing ``/``); ``headers`` is ``{"Authorization": "Bearer <key>"}`` or
    ``{"x-api-key": "<key>"}`` per the wire's ``auth_scheme``; never log or archive it. Raises ``ValueError`` with a
    leading stable code and no key: ``profile_wire_not_served``, ``profile_missing_key`` (the catalogue cache reports
    these as ``credential_unavailable:<code>``). Never returns None."""
    if wire not in profile.wires:
        _fail("wire_not_served", profile.path, f"the profile serves {', '.join(profile.wires)}, not {wire}")
    if not profile.key_present:
        _fail("missing_key", profile.path, "the profile has no credential.api_key; save a key for it")
    entry = profile.wires[wire]
    key = profile._secret.reveal()
    headers = {"Authorization": "Bearer " + key} if entry.auth_scheme == "bearer" else {"x-api-key": key}
    return entry.base_url, headers


def forwarded_env(profile, backend):
    """Variables forwarded to the agent CLI of ``backend`` via its registered ``ProfileSection``."""
    _backend(backend)
    entry = get_backend(backend, error_cls=ValueError)
    return entry.profile_section.forwarded_env(profile) if entry.profile_section is not None else {}


def upstream_fields(profile, backend, wire):
    """Every ``UpstreamConfig`` field of a ``source="profile"`` upstream of ``backend`` on ``wire`` (a dict; no key)."""
    entry = profile.wires[wire]
    mcp = get_backend(backend, error_cls=ValueError).traits.relay_compat_fields
    return dict(source="profile", settings=str(profile.path), wire_api=wire, base_url=entry.base_url,
                api_key_env=None, auth_scheme=entry.auth_scheme, provider=profile.id,
                tool_result_images=profile.tool_result_images if mcp else "inline",
                tool_schema_max_depth=profile.tool_schema_max_depth if mcp else None)


def expand_upstream(upstream, backend):
    """``(UpstreamConfig, Profile)``: the ``source="profile"`` ``upstream`` of ``backend`` with every field the profile
    owns written from its file (``upstream.settings``; the path itself is kept). ``upstream.wire_api`` selects the wire
    when set. Raises ``ValueError`` (``profile_*``, see ``load_profile`` / ``resolve_wire``)."""
    profile = load_profile(upstream.settings)
    values = upstream_fields(profile, backend, resolve_wire(profile, backend, upstream.wire_api))
    values.pop("settings")
    return replace(upstream, **values), profile


def expand_profile_references(config, *, strict=True):
    """Expand every ``source="profile"`` actor upstream via registered backend ``ProfileSection`` adapters."""
    agent = config.agent
    sections = {}
    for entry in BACKENDS_REGISTRY.hosted_backends():
        kind = entry.kind
        upstream = actor_upstream(config, kind)
        if upstream is None or upstream.source != "profile":
            continue
        try:
            expanded, profile = expand_upstream(upstream, kind)
            field_name, field_value = entry.profile_section.expand_config(agent, expanded, profile, expanded.wire_api)
            sections[field_name] = field_value
        except ValueError as error:
            if strict:
                raise ConfigError(str(error)) from None
    return replace(config, agent=replace(agent, **sections)) if sections else config


def apply_profile(config, backend, profile, wire_api=None, *, model=None, effort=None, thinking_display=_UNSET):
    """Return candidate ``config`` switched to ``backend`` running through ``profile`` on ``wire_api``."""
    if not isinstance(profile, Profile):
        profile = load_profile(profile)
    wire = resolve_wire(profile, backend, wire_api)
    agent = {"backend": backend}
    upstream = upstream_fields(profile, backend, wire)
    entry = get_backend(backend, error_cls=ValueError)
    agent.update(entry.profile_section.apply_config(upstream, wire, profile, thinking_display))
    overrides = {"agent": agent}
    selected = {key: value for key, value in (("name", model), ("effort", effort)) if value is not None}
    if selected:
        overrides["model"] = selected
    return config.with_overrides(overrides)


def _auth_scheme_problem(wire, auth_scheme):
    """None when the relay's adapter for ``wire`` forwards the credential header ``auth_scheme`` names
    (``WireAdapter.auth_header_names``), else a one-line reason. Example: the OpenAI Responses adapter forwards only
    ``Authorization``, so an ``x-api-key`` profile wire would pass the gate and then fail at the first request."""
    from ..session.wire import AnthropicMessagesWire, OpenAIResponsesWire
    adapters = {"anthropic_messages": AnthropicMessagesWire, "openai_responses": OpenAIResponsesWire}
    adapter = adapters.get(wire)
    header = {"bearer": "authorization", "x-api-key": "x-api-key"}.get(auth_scheme)
    if adapter is None or header is None or header in adapter.auth_header_names:
        return None
    accepted = ", ".join(sorted({"authorization": "bearer", "x-api-key": "x-api-key"}[name]
                                for name in adapter.auth_header_names if name in ("authorization", "x-api-key")))
    return f"the {wire} relay forwards only {accepted} credentials, not {auth_scheme}"


def resolve_profile_upstream(upstream, backend):
    """``ProfileUpstream`` of the ``source="profile"`` actor ``upstream`` of ``backend``: the profile loads, serves
    a wire the backend runs on, matches the configuration and holds a key. The launch gate calls it (and drops the
    result); a session calls it at start to get ``base_url`` and ``headers``.

    Raises ``ValueError``: the ``load_profile`` / ``resolve_wire`` codes; ``profile_upstream_stale`` when wire_api,
    base_url, auth_scheme, api_key_env (must be null) or the compat fields of ``upstream`` differ from the profile
    (the file changed after the configuration was built, or the configuration was built without ``Config.load`` /
    ``apply_profile``); ``profile_missing_key``.
    """
    if upstream is None or upstream.source != "profile":
        raise ValueError('profile_upstream_required: upstream.source must be "profile"')
    profile = load_profile(upstream.settings)
    wire = resolve_wire(profile, backend, upstream.wire_api)
    problem = _auth_scheme_problem(wire, profile.wires[wire].auth_scheme)
    if problem:
        _fail("auth_scheme_unsupported", profile.path, problem + f"; set wires.{wire}.auth_scheme")
    expected = upstream_fields(profile, backend, wire)
    stale = [f"{name} is {getattr(upstream, name)!r} here, {expected[name]!r} in the profile"
             for name in _OWNED_FIELDS if getattr(upstream, name) != expected[name]]
    if stale:
        _fail("upstream_stale", profile.path, "; ".join(stale) + ". Load the configuration with Config.load or "
              "rebuild it with profiles.apply_profile")
    base_url, headers = credential(profile, wire)
    return ProfileUpstream(profile=profile, wire_api=wire, base_url=base_url, headers=headers)


def compatibility(backend, profile):
    """One row per wire ``backend`` runs on: ``{"wire", "compatible", "reason", "message"}``.

    ``profile``: ``Profile`` or ``ProfileInfo`` (an invalid file). Reasons: ``profile_invalid:<code>`` (the file does
    not load), ``wire_not_in_profile``, ``profile_missing_key``; None when compatible. Runtime availability and the
    full launch gate are ``check_selection``'s job."""
    rows = []
    for wire in _backend(backend):
        if isinstance(profile, ProfileInfo) and not profile.valid:
            reason, message = "profile_invalid:" + profile.error["code"], profile.error["message"]
        elif wire not in profile.wires:
            reason = "wire_not_in_profile"
            message = f"the profile serves {', '.join(profile.wires)}; {backend} on {wire} needs that wire"
        elif _auth_scheme_problem(wire, profile.wires[wire]["auth_scheme"] if isinstance(profile.wires[wire], dict)
                                  else profile.wires[wire].auth_scheme):
            reason = "profile_auth_scheme_unsupported"
            message = _auth_scheme_problem(wire, profile.wires[wire]["auth_scheme"] if isinstance(profile.wires[wire], dict)
                                           else profile.wires[wire].auth_scheme)
        elif not profile.key_present:
            reason, message = "profile_missing_key", "the profile has no key yet"
        else:
            reason = message = None
        rows.append(dict(wire=wire, compatible=reason is None, reason=reason, message=message))
    return rows


def matrix_cell(backend, profile):
    """``{"compatible", "wires", "reason", "message"}`` of the console matrix cell (``backend``, ``profile``):
    ``wires`` lists the compatible wires; when there is none, ``reason`` is ``no_common_wire`` if no wire of the
    backend is in the profile, else the first row's reason (``compatibility``)."""
    rows = compatibility(backend, profile)
    wires = [row["wire"] for row in rows if row["compatible"]]
    if wires:
        return dict(compatible=True, wires=wires, reason=None, message=None)
    if all(row["reason"] == "wire_not_in_profile" for row in rows):
        served = ", ".join(profile.wires)
        return dict(compatible=False, wires=[], reason="no_common_wire",
                    message=f"the profile serves {served}; {backend} needs {' or '.join(BACKEND_WIRES[backend])}")
    return dict(compatible=False, wires=[], reason=rows[0]["reason"], message=rows[0]["message"])


_WIRE_REASONS = {"profile_no_common_wire": "no_common_wire", "profile_wire_ambiguous": "wire_required",
                 "profile_wire_not_served": "wire_not_in_profile", "profile_wire_unsupported": "wire_unsupported"}


def check_selection(config, backend, profile, wire_api=None, *, model=None, effort=None, thinking_display=_UNSET,
                    requirements=None):
    """Dry run of an exact selection: ``apply_profile`` then the launch gate, nothing started or written.

    ``config``: the configuration to start from (e.g. the running one). ``profile``: ``Profile`` or path.
    ``requirements``: the gate (default ``factory.backend_requirements``; it runs ``--version`` of the agent CLI).
    Returns ``SelectionCheck``; the first failure is its reason: ``profile_invalid:<code>``, ``no_common_wire`` /
    ``wire_required`` / ``wire_not_in_profile`` / ``wire_unsupported``, ``profile_missing_key``,
    ``config_invalid:<code>`` (the candidate is not a valid configuration), ``backend_unavailable`` (the runtime is not
    installed) or ``gate:<code>`` (any other launch-gate refusal); ``message`` is the sentence verbatim.
    """
    if requirements is None:
        from .factory import backend_requirements as requirements
    try:
        if not isinstance(profile, Profile):
            profile = load_profile(profile)
    except ValueError as error:
        return SelectionCheck(False, "profile_invalid:" + _code(error), str(error))
    try:
        wire = resolve_wire(profile, backend, wire_api)
    except ValueError as error:
        return SelectionCheck(False, _WIRE_REASONS.get(_code(error), _code(error)), str(error))
    if not profile.key_present:
        return SelectionCheck(False, "profile_missing_key", f"{profile.path}: the profile has no key yet", wire)
    try:
        candidate = apply_profile(config, backend, profile, wire, model=model, effort=effort,
                                  thinking_display=thinking_display)
    except ValueError as error:
        return SelectionCheck(False, "config_invalid:" + _code(error), str(error), wire)
    try:
        requirements(candidate)
    except ConfigError as error:
        code = _code(error)
        reason = "backend_unavailable" if code == "agent_backend_unavailable" else "gate:" + code
        return SelectionCheck(False, reason, str(error), wire)
    return SelectionCheck(True, wire_api=wire, config=candidate)


# ---------------------------------------------------------------------------------------------- writing


def _fsync_directory(directory):
    with contextlib.suppress(OSError):
        handle = os.open(directory, os.O_RDONLY)
        try:
            os.fsync(handle)
        finally:
            os.close(handle)


def _write_private_json(location, document):
    """Write ``document`` to ``location`` atomically (temp file in the same directory, fsync, rename) with mode 0600;
    the directory is created with mode 0700 when missing. The original file is untouched if anything fails."""
    directory = location.parent
    if not directory.exists():
        directory.mkdir(mode=0o700, parents=True)
        os.chmod(directory, 0o700)
    handle, temporary = tempfile.mkstemp(prefix="." + location.name + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(handle, "w", encoding="utf-8") as stream:
            stream.write(json.dumps(document, indent=2, ensure_ascii=False) + "\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.chmod(temporary, 0o600)
        os.replace(temporary, location)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(temporary)
        raise
    _fsync_directory(directory)


def _stored_key(location):
    """The key of the existing file ``location`` (``_Secret``), ``_NO_KEY`` when there is no file or no key."""
    if not location.exists():
        return _NO_KEY
    try:
        document = _read_document(location)
    except ValueError:
        _fail("key_not_kept", location, "the stored profile cannot be read, so its key cannot be kept; send the key")
    section = document.get("credential")
    value = section.get("api_key") if isinstance(section, dict) else None
    return _Secret(value.strip()) if isinstance(value, str) and value.strip() else _NO_KEY


def save_profile(directory, profile_id, document, *, keep_key=True):
    """Create or replace the profile ``<directory>/<profile_id>.local.json`` with ``document``; returns its
    ``ProfileInfo`` (never the key).

    ``document``: a profile document (see ``FIELD_HELP``). A non-empty ``credential.api_key`` in it is stored. Without
    one, ``keep_key=True`` (default) keeps the key of the existing file (so an edit form with an empty key field keeps
    the stored key) and ``keep_key=False`` stores no key. The document is validated like ``load_profile`` and written
    in canonical form atomically with mode 0600 (``_write_private_json``). Raises ``ValueError`` (``profile_*``).
    """
    location = profile_path(directory, profile_id)
    if not isinstance(document, dict):
        _fail("not_object", location, "the profile document must be a JSON object")
    document = copy.deepcopy(document)
    section = document.get("credential")
    supplied = isinstance(section, dict) and section.get("api_key") not in (None, "")
    if not supplied:
        if isinstance(section, dict):
            section.pop("api_key", None)
        if keep_key:
            stored = _stored_key(location)
            if stored:
                document["credential"] = {**(section if isinstance(section, dict) else {}),
                                          "api_key": stored.reveal()}
    profile = _parse(document, location, profile_id)
    _write_private_json(location, _document(profile, with_key=True))
    return public_view(profile)


def save_profile_key(directory, profile_id, api_key):
    """Store ``api_key`` as the key of the existing profile ``profile_id`` (everything else unchanged); returns the
    ``ProfileInfo`` (``key_present: true``), never the key. Raises ``ValueError``: ``profile_invalid_key`` (not
    printable ASCII), ``profile_not_found``, or the existing file's own ``profile_*`` error."""
    location = profile_path(directory, profile_id)
    secret = _check_key(api_key, location)
    if not secret:
        _fail("invalid_key", location, "the key is empty")
    if not location.is_file():
        _fail("not_found", location, "create the profile before saving its key")
    profile = replace(load_profile(location), _secret=secret)
    _write_private_json(location, _document(profile, with_key=True))
    return public_view(profile)


def remove_profile(directory, profile_id, *, in_use=()):
    """Retire the profile ``profile_id`` by renaming it to ``<id>.local.json.removed-<UTC YYYYmmddTHHMMSSZ>`` (never
    unlinked; the name no longer matches discovery). ``in_use``: profile ids or paths the running run uses; one of
    them is refused (``profile_in_use``). Returns the new path. Raises ``ValueError``: ``profile_not_found``,
    ``profile_in_use``."""
    location = profile_path(directory, profile_id)
    if location.is_symlink() or not location.is_file():
        _fail("not_found", location, "no such profile")
    used = {str(item) if PROFILE_ID.fullmatch(str(item)) else str(_location(item)) for item in in_use}
    if profile_id in used or str(location) in used:
        _fail("in_use", location, "the running run uses this profile")
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    target = location.with_name(f"{location.name}.removed-{stamp}")
    counter = 1
    while target.exists():
        counter += 1
        target = location.with_name(f"{location.name}.removed-{stamp}-{counter}")
    os.rename(location, target)
    _fsync_directory(location.parent)
    return target


# ---------------------------------------------------------------------------------------------- CLI


def _configured_profiles_dir():
    try:
        return profiles_directory(Config.load())
    except (ConfigError, ValueError, OSError):
        return DEFAULT_PROFILES_DIR


def main(argv=None):
    """``python -m ni_rua.cortex.agents.profiles {import,list}``; see the module doc. Never prints a key."""
    parser = argparse.ArgumentParser(prog="python -m ni_rua.cortex.agents.profiles",
                                     description="NI-RUA provider profiles (keys are never printed).")
    commands = parser.add_subparsers(dest="command", required=True)
    importer = commands.add_parser("import", help="convert the official-format key files into profiles")
    importer.add_argument("--dry-run", action="store_true", help="print the plan and write nothing")
    importer.add_argument("--source-dir", default=str(DEFAULT_PROFILES_DIR.parent),
                          help="directory of claude*.local.json / opencode*.local.json (default: <checkout>/configs)")
    importer.add_argument("--codex-homes", default="~/.config/ni-rua/providers",
                          help="directory of Codex provider homes (default: ~/.config/ni-rua/providers); '' = none")
    importer.add_argument("--profiles-dir", default=None,
                          help="target directory (default: agent.profiles_dir of the configuration, else "
                               "<checkout>/configs/profiles)")
    importer.add_argument("--config", action="append", default=None, dest="configs",
                          help="run configuration read for compat opt-ins (repeatable; default: ni-rua.json and "
                               "ni-rua.local.json in --source-dir)")
    lister = commands.add_parser("list", help="list the profiles (public view, never a key)")
    lister.add_argument("--profiles-dir", default=None)
    args = parser.parse_args(argv)
    directory = _location(args.profiles_dir) if args.profiles_dir else _configured_profiles_dir()
    if args.command == "list":
        for info in list_profiles(directory):
            print(json.dumps(info.to_dict(), ensure_ascii=False))
        return 0
    from .legacy_profile_import import run_import_command
    return run_import_command(args, directory)


if __name__ == "__main__":
    from ni_rua.cortex.agents.profiles import main as _main   # the canonical module, not a __main__ copy
    sys.exit(_main())
