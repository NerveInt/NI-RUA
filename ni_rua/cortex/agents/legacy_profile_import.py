"""Legacy one-shot importer converting pre-profile official-format key files into provider profiles."""
from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass, field
from pathlib import Path

from .profiles import (
    NAME_MAX_CHARS,
    PROFILE_SUFFIX,
    PROFILE_WIRES,
    SCHEMA,
    _COMPAT_KEYS,
    _Secret,
    _fail,
    _location,
    _parse,
    profile_path,
    save_profile,
)
from .registry import BACKENDS_REGISTRY

__all__ = [
    "IMPORT_LISTING_URLS",
    "ImportEntry",
    "plan_import",
    "run_import_command",
    "write_import",
]

_FORMAT_SUFFIX = {"opencode_config": "opencode", "codex_home": "codex", "claude_settings": "claude"}
_FORMAT_RANK = {"opencode_config": 0, "codex_home": 1, "claude_settings": 2}  # whose provider id names a profile
# wires.<wire>.models_url the importer writes for provider base URLs (relay convention, exact match) whose listing
# derived from the base URL is poorer than the provider's full one: OpenCode Console's /anthropic/v1/models answers the
# bare Anthropic shape, /inference/v1/models the models.dev shape with reasoning efforts, image input and limits.
# Written into the profile file at import time (visible, editable); nothing reads this table at run time.
IMPORT_LISTING_URLS = {
    "https://opencode.ai/inference/anthropic": "https://opencode.ai/inference/v1/models",
    "https://opencode.ai/inference/openai/v1": "https://opencode.ai/inference/v1/models",
}


@dataclass
class _Source:
    """One credential found in an official-format file. ``auth_fixed``: the scheme is dictated by that tool's SDK
    (OpenCode's @ai-sdk/anthropic always sends x-api-key, @ai-sdk/openai and Codex always Bearer), not chosen in the
    file; a Claude settings file chooses (ANTHROPIC_AUTH_TOKEN = bearer, ANTHROPIC_API_KEY = x-api-key)."""

    origin: Path
    format: str
    provider: str | None
    id_hint: str
    name: str | None
    wire: str
    base_url: str
    auth_scheme: str
    auth_fixed: bool
    secret: _Secret = field(repr=False)
    env: dict | None = None
    hints: dict = field(default_factory=dict)

    @property
    def label(self):
        kind = f"{self.format}:{self.provider}" if self.format == "opencode_config" else self.format
        return f"{_display(self.origin)} ({kind})"


@dataclass
class ImportEntry:
    """One line of an import plan. ``action``: ``create`` (``document`` holds the key; never print it) or ``skip``
    (``reason``). ``sources``: labels of the files it comes from. ``notes``: merge decisions."""

    action: str
    profile_id: str | None
    sources: list
    reason: str | None = None
    notes: list = field(default_factory=list)
    document: dict | None = field(default=None, repr=False)

    def summary_lines(self):
        """Human-readable lines without the key: id, name, wires with base URL and auth scheme, compat, forwarded
        variable names, sources, notes."""
        if self.action != "create":
            return [f"skip   {', '.join(self.sources) or '-'}: {self.reason}"]
        document = self.document
        lines = [f"create {self.profile_id}{PROFILE_SUFFIX}  name={document['name']!r}"]
        for wire, entry in document["wires"].items():
            listing = f"  models_url={entry['models_url']}" if entry.get("models_url") else ""
            lines.append(
                f"         wire {wire}: base_url={entry['base_url']}  auth_scheme={entry['auth_scheme']}" + listing
            )
        compat = document.get("compat")
        if compat:
            lines.append(
                f"         compat: tool_result_images={compat['tool_result_images']}  "
                f"tool_schema_max_depth={compat['tool_schema_max_depth']}"
            )
        backends = document.get("backends", {})
        if backends.get("claude_code"):
            lines.append("         backends.claude_code.env: " + ", ".join(sorted(backends["claude_code"]["env"])))
        if backends.get("opencode"):
            lines.append(f"         backends.opencode.thinking_display: {backends['opencode']['thinking_display']}")
        lines.append("         key: present  from: " + "; ".join(self.sources))
        lines.extend("         note: " + note for note in self.notes)
        return lines


def _display(path):
    home = str(Path.home())
    text = str(path)
    return "~" + text[len(home):] if text.startswith(home + os.sep) else text


def _id_hint(text):
    hint = re.sub(r"[^a-z0-9_-]+", "-", str(text).lower()).strip("-_")
    return hint[:40].rstrip("-_") or "profile"


def _key_from(headers):
    if "Authorization" in headers:
        return headers["Authorization"][len("Bearer "):], "bearer"
    return headers["x-api-key"], "x-api-key"


def _read_claude_file(path):
    from .claude_code.options import read_claude_settings_env, read_claude_settings_upstream

    base_url, headers = read_claude_settings_upstream(path)
    env = read_claude_settings_env(path)
    key, scheme = _key_from(headers)
    stem = path.name[:-len(PROFILE_SUFFIX)] if path.name.endswith(PROFILE_SUFFIX) else path.stem
    hint = stem.split(".", 1)[1] if "." in stem else stem
    return [
        _Source(
            origin=path,
            format="claude_settings",
            provider=None,
            id_hint=_id_hint(hint),
            name=None,
            wire="anthropic_messages",
            base_url=base_url.rstrip("/"),
            auth_scheme=scheme,
            auth_fixed=False,
            secret=_Secret(key),
            env=env,
        )
    ]


def _read_opencode_file(path, hints):
    from .opencode.upstream_file import read_opencode_config_upstream

    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        data = None
    entries = data.get("providers", data.get("provider")) if isinstance(data, dict) else None
    if not isinstance(entries, dict) or not entries or ("provider" in data and "providers" in data):
        read_opencode_config_upstream(path)  # the official reader's own refusal (unreadable, mixed, empty)
        return []
    sources = []
    for provider_id, entry in entries.items():
        hint = hints.get((str(path), provider_id)) or (hints.get((str(path), None)) if len(entries) == 1 else None) or {}
        resolved = read_opencode_config_upstream(path, provider=provider_id, wire_api=hint.get("wire_api"))
        key, scheme = _key_from(resolved.headers)
        name = entry.get("name") if isinstance(entry, dict) and isinstance(entry.get("name"), str) else None
        sources.append(
            _Source(
                origin=path,
                format="opencode_config",
                provider=provider_id,
                id_hint=_id_hint(provider_id),
                name=name,
                wire=resolved.wire_api,
                base_url=resolved.base_url,
                auth_scheme=scheme,
                auth_fixed=True,
                secret=_Secret(key),
                hints=hint,
            )
        )
    return sources


def _read_codex_home(home):
    import tomllib
    from .codex.session import _chatgpt_login, _model_provider, _upstream_auth

    failure = None
    try:
        document = tomllib.loads((home / "config.toml").read_text())
        auth_path = home / "auth.json"
        auth = json.loads(auth_path.read_text()) if auth_path.exists() else {}
    except (OSError, ValueError) as error:
        failure = type(error).__name__
    if failure is not None:
        raise ValueError(f"codex_home_unreadable:{home}: {failure}")
    name, provider, url = _model_provider(document, home)
    if provider.get("wire_api", "responses") != "responses":
        raise ValueError(f"codex_home_wire_unsupported:{home}: the provider is not on the responses wire")
    if provider.get("requires_openai_auth") and _chatgpt_login(home):
        raise ValueError(
            f"codex_home_chatgpt_login:{home}: a ChatGPT login serves Codex only; keep it as model.codex_home"
        )
    if provider.get("env_key") or provider.get("env_http_headers"):
        raise ValueError(
            f"codex_home_key_in_environment:{home}: the key comes from an environment variable; not imported"
        )
    if provider.get("requires_openai_auth") and not (isinstance(auth, dict) and auth.get("OPENAI_API_KEY")):
        raise ValueError(f"codex_home_missing_key:{home}: auth.json has no OPENAI_API_KEY")
    headers = _upstream_auth(provider, home)
    if set(headers) != {"Authorization"} or not headers["Authorization"].startswith("Bearer "):
        raise ValueError(f"codex_home_unsupported_auth:{home}: only an Authorization: Bearer key is imported")
    key, scheme = _key_from(headers)
    display = provider.get("name") if isinstance(provider.get("name"), str) else None
    return [
        _Source(
            origin=home,
            format="codex_home",
            provider=name,
            id_hint=_id_hint(name),
            name=display,
            wire="openai_responses",
            base_url=url.rstrip("/"),
            auth_scheme=scheme,
            auth_fixed=True,
            secret=_Secret(key),
        )
    ]


def _run_config_hints(paths):
    """``{(file, provider or None): {tool_result_images?, tool_schema_max_depth?, thinking_display?, wire_api?,
    configs}}`` from the ``agent.claude_code`` / ``agent.opencode`` upstream sections of run configuration files that
    reference a credential file (``source`` ``claude_settings`` / ``opencode_config``). Later files win."""
    hints = {}
    for path in paths:
        try:
            data = json.loads(Path(path).read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        agent = data.get("agent") if isinstance(data, dict) else None
        for key, prof_section in BACKENDS_REGISTRY.profile_sections().items():
            section = agent.get(key) if isinstance(agent, dict) else None
            upstream = section.get("upstream") if isinstance(section, dict) else None
            if not isinstance(upstream, dict) or upstream.get("source") not in ("claude_settings", "opencode_config"):
                continue
            settings = upstream.get("settings")
            if not isinstance(settings, str) or not settings.strip():
                continue
            target = (Path(path).resolve().parent / Path(settings).expanduser()).resolve()
            provider = upstream.get("provider") if upstream["source"] == "opencode_config" else None
            hint = hints.setdefault((str(target), provider), {"configs": []})
            for name in _COMPAT_KEYS:
                if name in upstream:
                    hint[name] = upstream[name]
            if upstream["source"] == "opencode_config" and upstream.get("wire_api"):
                hint["wire_api"] = upstream["wire_api"]
            prof_section.collect_hints(section, hint, _display(Path(path).resolve()))
            hint["configs"].append(_display(Path(path).resolve()))
    return hints


def _merge(sources):
    """``(document without name, notes)`` of one profile for ``sources`` that share a key, or ``(None, reason)``
    when they conflict: a wire declared with two base URLs, two chosen auth schemes for one wire (a scheme fixed by an
    SDK yields to a chosen one: the endpoint accepts both), different Claude Code variables, compat values or
    thinking_display values."""
    notes = []
    wires = {}
    for wire in PROFILE_WIRES:
        items = [source for source in sources if source.wire == wire]
        if not items:
            continue
        bases = sorted({source.base_url for source in items})
        if len(bases) > 1:
            return None, f"{wire} is declared with different base URLs ({', '.join(bases)})"
        chosen = {source.auth_scheme for source in items if not source.auth_fixed}
        fixed = {source.auth_scheme for source in items if source.auth_fixed}
        if len(chosen) > 1 or (not chosen and len(fixed) > 1):
            return None, f"{wire} is used with different auth schemes ({', '.join(sorted(chosen | fixed))})"
        scheme = next(iter(chosen or fixed))
        if chosen and fixed - chosen:
            yielded = [source.label for source in items if source.auth_fixed and source.auth_scheme != scheme]
            chooser = next(source.label for source in items if not source.auth_fixed)
            notes.append(
                f"{wire} auth_scheme {scheme} as chosen in {chooser}; {', '.join(yielded)} sent "
                f"{', '.join(sorted(fixed - chosen))} because its SDK fixes the header (same endpoint)"
            )
        wires[wire] = {"base_url": bases[0], "auth_scheme": scheme}
        if bases[0] in IMPORT_LISTING_URLS:
            wires[wire]["models_url"] = IMPORT_LISTING_URLS[bases[0]]
            notes.append(
                f"{wire} models_url {IMPORT_LISTING_URLS[bases[0]]} (the provider's full listing; IMPORT_LISTING_URLS)"
            )
    envs = [source.env for source in sources if source.format == "claude_settings"]
    if any(env != envs[0] for env in envs):
        return None, "the Claude settings files forward different variables"
    document = {"schema": SCHEMA, "credential": {"api_key": sources[0].secret.reveal()}, "wires": wires}
    compat = {}
    for name in _COMPAT_KEYS:
        values = {json.dumps(source.hints[name]) for source in sources if name in source.hints}
        if len(values) > 1:
            return None, f"the run configurations set different {name} for these files"
        if values:
            compat[name] = json.loads(values.pop())
    if compat:
        document["compat"] = {
            "tool_result_images": compat.get("tool_result_images", "inline"),
            "tool_schema_max_depth": compat.get("tool_schema_max_depth"),
        }
    backends = {}
    if envs and envs[0]:
        backends["claude_code"] = {"env": dict(envs[0])}
    displays = {json.dumps(source.hints["thinking_display"]) for source in sources if "thinking_display" in source.hints}
    if len(displays) > 1:
        return None, "the run configurations set different agent.opencode.thinking_display for these files"
    display = json.loads(displays.pop()) if displays else None
    if display is not None:
        backends["opencode"] = {"thinking_display": display}
    if backends:
        document["backends"] = backends
    configs = sorted({config for source in sources for config in source.hints.get("configs", ())})
    if compat or display is not None:
        notes.append("compat / thinking_display from " + ", ".join(configs))
    return document, notes


def _preferred(sources, attribute):
    for kind in sorted(_FORMAT_RANK, key=_FORMAT_RANK.get):
        for source in sources:
            value = getattr(source, attribute)
            if source.format == kind and value:
                return value
    return None


def plan_import(*, source_dir, codex_homes=None, profiles_dir, run_configs=None):
    """The import plan (list of ``ImportEntry``) for the official-format key files; reads files, writes nothing.

    ``source_dir``: directory of ``claude*.local.json`` (Claude Code settings) and ``opencode*.local.json`` (OpenCode
    config, V1 or V2; every provider entry is a credential). ``codex_homes``: directory whose subdirectories are Codex
    homes (``config.toml`` + ``auth.json``; a ChatGPT login or an environment-variable key is skipped) or None.
    ``profiles_dir``: where the profiles would go (an existing ``<id>.local.json`` is never overwritten: ``skip``).
    ``run_configs``: run configuration files read for compat opt-ins and OpenCode ``thinking_display`` of the
    upstreams that reference these files (default: ``ni-rua.json`` and ``ni-rua.local.json`` in ``source_dir``).

    Credentials with the same key become one profile with every wire they serve when ``_merge`` finds no conflict,
    otherwise one profile each (the entry notes why). The id is the OpenCode provider id, else the Codex
    ``model_provider``, else the Claude file's middle name (``claude.<name>.local.json``; ``claude.local.json`` ->
    ``claude``); a clash gets ``-<tool>`` then ``-2``... The name is the OpenCode or Codex display name, else the id.
    A wire whose base URL is in ``IMPORT_LISTING_URLS`` gets that ``models_url``.
    """
    source_dir = _location(source_dir)
    if run_configs is None:
        run_configs = [path for path in (source_dir / "ni-rua.json", source_dir / "ni-rua.local.json") if path.is_file()]
    hints = _run_config_hints(run_configs)
    sources, entries = [], []
    candidates = [("claude_settings", path) for path in sorted(source_dir.glob("claude*" + PROFILE_SUFFIX))]
    candidates += [("opencode_config", path) for path in sorted(source_dir.glob("opencode*" + PROFILE_SUFFIX))]
    if codex_homes is not None and _location(codex_homes).is_dir():
        candidates += [
            ("codex_home", path)
            for path in sorted(_location(codex_homes).iterdir())
            if path.is_dir() and (path / "config.toml").is_file()
        ]
    for kind, path in candidates:
        failure = None
        try:
            if kind == "claude_settings":
                found = _read_claude_file(path)
                for source in found:
                    source.hints = hints.get((str(path.resolve()), None), {})
            elif kind == "opencode_config":
                found = _read_opencode_file(path.resolve(), hints)
            else:
                found = _read_codex_home(path)
            sources.extend(found)
        except (OSError, ValueError) as error:  # the readers never put a key in their messages
            failure = str(error) if isinstance(error, ValueError) else type(error).__name__
        if failure is not None:
            entries.append(ImportEntry("skip", None, [f"{_display(path)} ({kind})"], reason=failure))
    groups = {}
    for source in sources:
        groups.setdefault(source.secret.reveal(), []).append(source)
    target = _location(profiles_dir)
    taken = {item.name[:-len(PROFILE_SUFFIX)] for item in target.glob("*" + PROFILE_SUFFIX)} if target.is_dir() else set()
    planned = []
    for group in groups.values():
        document, detail = _merge(group)
        if document is not None:
            planned.append((group, document, detail))
            continue
        others = ", ".join(source.label for source in group)
        for source in sorted(group, key=lambda item: _FORMAT_RANK[item.format]):
            single, notes = _merge([source])
            planned.append(([source], single, notes + [f"same key as {others}; not merged: {detail}"]))
    used = set()
    planned.sort(key=lambda item: min(_FORMAT_RANK[source.format] for source in item[0]))
    for group, document, notes in planned:
        preferred = _preferred(group, "id_hint")
        if preferred in used:
            preferred = _id_hint(f"{preferred}-{_FORMAT_SUFFIX[group[0].format]}")
        profile_id, counter = preferred, 1
        while profile_id in used:
            counter += 1
            profile_id = f"{preferred[:36]}-{counter}"
        used.add(profile_id)
        labels = [source.label for source in group]
        location = profile_path(profiles_dir, profile_id)
        if profile_id in taken:
            entries.append(
                ImportEntry(
                    "skip",
                    profile_id,
                    labels,
                    reason=f"profile_exists:{location}: an existing profile is never overwritten",
                )
            )
            continue
        document = {
            "schema": SCHEMA,
            "name": (_preferred(group, "name") or profile_id)[:NAME_MAX_CHARS].strip(),
            **{key: value for key, value in document.items() if key != "schema"},
        }
        failure = None
        try:
            _parse(document, location, profile_id)
        except ValueError as error:
            failure = str(error)
        if failure is not None:
            entries.append(ImportEntry("skip", profile_id, labels, reason=failure))
            continue
        entries.append(ImportEntry("create", profile_id, labels, notes=list(notes), document=document))
    return entries


def write_import(entries, profiles_dir):
    """Write every ``create`` entry of ``plan_import`` as a profile (``save_profile``, mode 0600, never over an
    existing file); returns the ``ProfileInfo`` list. The source files are not touched."""
    written = []
    for entry in entries:
        if entry.action != "create":
            continue
        location = profile_path(profiles_dir, entry.profile_id)
        if location.exists():
            _fail("exists", location, "an existing profile is never overwritten")
        written.append(save_profile(profiles_dir, entry.profile_id, entry.document, keep_key=False))
    return written


def run_import_command(args, directory):
    """Execute the ``import`` CLI subcommand against ``directory``."""
    entries = plan_import(
        source_dir=args.source_dir,
        codex_homes=args.codex_homes or None,
        profiles_dir=directory,
        run_configs=args.configs,
    )
    mode = "dry run, nothing written" if args.dry_run else "writing"
    print(f"profiles directory: {directory}  ({mode}; the source files are never changed)")
    for entry in entries:
        print("\n".join(entry.summary_lines()))
    creates = sum(entry.action == "create" for entry in entries)
    if not args.dry_run:
        write_import(entries, directory)
    print(f"{creates} profile(s) {'to create' if args.dry_run else 'created'}, {len(entries) - creates} skipped")
    return 0
