"""OpenCode 2.x launch builder (design 11 sections 1.1-1.4, 1.10, 2.1-2.4, 3): config file, private dirs,
instructions file, ``rg`` exposure, environment, version and route-list checks, model declarations, provenance.

Nothing here starts a process except the one ``<executable> --version`` probe, and nothing contacts a model.
``build_launch`` writes the generated config to ``<run>/private/opencode/xdg/config/opencode/opencode.json`` (the one
OpenCode config dir) and returns an ``OpenCodeLaunch`` (argv, environment, working directory, config) for the session
to spawn. A model selected during the run is declared by ``add_model`` + ``write_config``: OpenCode hot-reloads the
file, so ``GET /api/model`` lists the new model without a restart.
"""
from __future__ import annotations

import copy
import hashlib
import json
import os
import re
import shutil
import stat
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

from ni_rua.cortex.agents.isolation import (
    clean_environment, file_sha256_cached, private_dirs, require_exact_version,
)

__all__ = [
    "OPENCODE_VERSION", "OPENCODE_PACKAGE", "OPENCODE_INSTALL_HINT", "RELAY_PROVIDER_ID", "RELAY_PLACEHOLDER_KEY",
    "WIRE_PACKAGES", "MANAGED_CONFIG_PATHS", "CONFIG_DIR", "CONFIG_FILE_NAME", "INSTRUCTIONS_FILE_NAME", "SUBAGENT_TOOL",
    "MEDIA_IMAGE", "TOOL_OUTPUT", "FALLBACK_CONTEXT_LIMIT_TOKENS", "OUTPUT_LIMIT_CEILING_TOKENS",
    "THINKING_DISPLAY_CHOICES",
    "OpenCodeConfigError", "OpenCodeLaunch", "ModelSpec", "resolve_limits", "model_spec", "model_entry",
    "build_config", "add_model", "declared_model_ids", "write_config", "build_launch", "check_managed_config",
    "resolve_ripgrep", "parse_cli_version", "probe_cli_version", "check_health_version", "check_served_models",
    "load_expected_routes", "route_list_digest", "check_route_list", "native_tool_ids", "provenance_block",
    "SANDBOX_EXECUTABLE", "SANDBOX_RIPGREP", "SANDBOX_HARNESS_SOURCE", "sandbox_spec",
    "EFFORT_CHOICES", "EFFORT_WIRE_FIELDS", "effort_variants", "tui_attach_command",
]

OPENCODE_VERSION = "2.2.7"                      # exact pin of the OpenCode platform binary
OPENCODE_PACKAGE = "@opencode/cli"              # npm scope + name; one tarball per target: @opencode/cli-<target>
# Install sentence of the launch refusal (design 11 section 5 "Version pin and install").
OPENCODE_INSTALL_HINT = (f"download the npm tarball {OPENCODE_PACKAGE}-<target>@{OPENCODE_VERSION} (<target>: "
                         "darwin-arm64, darwin-x64, linux-x64, linux-x64-baseline, linux-arm64), extract "
                         f"package/bin/opencode to ~/.local/share/ni-rua/opencode-{OPENCODE_VERSION}/opencode")
RELAY_PROVIDER_ID = "relay"                     # neutral provider alias; the model id is the configured model name
RELAY_PLACEHOLDER_KEY = "ni-rua-local-relay"    # never a real credential; the relay holds the upstream key
CONFIG_FILE_NAME = "opencode.json"              # global config file of the private config dir
INSTRUCTIONS_FILE_NAME = "NI-RUA.md"
SANDBOX_EXECUTABLE = "/opt/ni-rua-bin/opencode"  # the pinned binary inside the bubblewrap namespace (read-only mount)
SANDBOX_RIPGREP = "/opt/ni-rua-bin/rg"           # ``rg`` mounted by AgentSandbox (first on the in-sandbox PATH)
SANDBOX_HARNESS_SOURCE = "/harness"              # read-only snapshot of the published Harness source
# Provider package bundled in the binary, per relay wire (request paths under baseURL: /messages, /responses).
# ``openai_chat`` is absent: the relay has no Chat Completions adapter for OpenCode (user decision D1).
WIRE_PACKAGES = {
    "anthropic_messages": "@opencode/ai/providers/anthropic-compatible",
    "openai_responses": "@opencode/ai/providers/openai-compatible/responses",
}
MANAGED_CONFIG_PATHS = (                        # a managed config has the highest precedence (overrides ours)
    "/etc/opencode",
    "/Library/Application Support/opencode",
    "/Library/Managed Preferences/ai.opencode.managed.plist",
)
# Isolation switches the 2.2.7 binary still reads: no project config / .opencode dirs / upward AGENTS.md, no ~/.claude
# skills, no self-update, no models.dev fetch.
_FLAGS = {
    "OPENCODE_DISABLE_PROJECT_CONFIG": "1", "OPENCODE_DISABLE_CLAUDE_CODE": "1",
    "OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_DISABLE_MODELS_FETCH": "1",
}
# Native tools removed from the agent: one ``{action, "*", deny}`` permission rule each (a ``*`` deny takes the tool
# out of the model's tool list). webfetch / websearch reach the internet (also Code Mode ``fetch``, governed by the
# webfetch permission); question waits for a human. Everything else (shell, edit, subagent, skill, execute, todowrite,
# ...) stays: agents keep their native abilities.
_DENIED_TOOLS = {"webfetch": "online_access", "websearch": "online_access", "question": "waits_for_human"}
SUBAGENT_TOOL = "subagent"                      # denied as well when agent.opencode.subagents=false
# ``launch.dirs`` key (run-relative) of the one OpenCode config dir: OPENCODE_CONFIG_DIR == $XDG_CONFIG_HOME/opencode.
CONFIG_DIR = "private/opencode/xdg/config/opencode"
_ROUTES_FILE = Path(__file__).with_name("opencode_routes.json")
# OpenCode re-encodes PNG tool-result images above 393,216 base64 chars to JPEG by default; these limits switch its
# own resizing off so the bytes the host sent reach the provider unchanged (no NI-RUA image logic involved).
MEDIA_IMAGE = {"auto_resize": False, "max_width": 1000000, "max_height": 1000000, "max_base64_bytes": 1073741824}
TOOL_OUTPUT = {"max_lines": 100000, "max_bytes": 16000000}    # lines / bytes before OpenCode truncates a tool output
_MCP_TIMEOUTS_MS = {"startup": 30000, "catalog": 30000}       # ms; ``execution`` is ``build_config(mcp_timeout_ms)``
FALLBACK_CONTEXT_LIMIT_TOKENS = 200000          # limit.context without an explicit value or a catalogue entry
# limit.output ceiling for a catalogue value and fallback without one: OpenCode sends limit.output as max_tokens, and
# catalogues list values gateways refuse as a request value (some gateways list 131072-1048576, refuse > 32768).
OUTPUT_LIMIT_CEILING_TOKENS = 32000
THINKING_DISPLAY_CHOICES = ("summarized", "omitted")
_MODEL_ID = re.compile(r"[^#\s]+\Z")              # "#" separates the variant in an OpenCode model ref

# Reasoning effort (``model.effort``) per upstream wire: every name is declared as a model ``variant`` (an explicit
# request ``body`` fragment) on every declared model and selected through the session's model ref
# (``{id, providerID, variant}``). Identity mapping, no per-model table: a tier the model rejects fails at the provider.
# ``EFFORT_WIRE_FIELDS`` names the request-body field the variant sets (provenance only).
EFFORT_CHOICES = {
    "anthropic_messages": ("low", "medium", "high", "xhigh", "max"),
    "openai_responses": ("none", "minimal", "low", "medium", "high", "xhigh"),
}
EFFORT_WIRE_FIELDS = {"anthropic_messages": "output_config.effort", "openai_responses": "reasoning.effort"}


class OpenCodeConfigError(RuntimeError):
    """Start-time refusal; ``str(error)`` is a stable code (``opencode_managed_config_present``, ...)."""


@dataclass(frozen=True)
class ModelSpec:
    """One model declared on the relay provider (``providers.relay.models.<id>``); built by ``model_spec``.

    id: model id exactly as sent upstream (never aliased).
    limits: ``resolve_limits`` result ``{"context": tokens, "output": tokens, "source": {"context": ..., "output":
        ...}}``; ``context`` is ``limit.context`` (drives native auto-compaction), ``output`` is ``limit.output``
        (sent as ``max_tokens`` / ``max_output_tokens``).
    image_input: the provider catalogue's ``image_input`` (True / False / None = not reported). Only False declares
        ``capabilities.input = ["text"]``; OpenCode then replaces a tool-result image with its own model-visible text
        "ERROR: Cannot read image (this model does not support image input)..." instead of sending it.
    """
    id: str
    limits: dict
    image_input: bool | None = None

    @property
    def input_modalities(self) -> list:
        """``capabilities.input`` of the declaration: ``["text"]`` when ``image_input`` is False, else ``["text",
        "image"]``."""
        return ["text"] if self.image_input is False else ["text", "image"]

    def record(self) -> dict:
        """Provenance item of ``agent_backend.declared_models``: ``{id, limits, input}``."""
        return {"id": self.id, "limits": copy.deepcopy(self.limits), "input": self.input_modalities}


@dataclass(frozen=True)
class OpenCodeLaunch:
    """Everything needed to spawn ``opencode serve`` (all paths absolute).

    ``config`` is the document written to ``config_path`` at launch and ``models`` the declared ``ModelSpec`` list
    (first = the configured model); a later ``add_model`` + ``write_config`` does not change either (the session
    keeps the current document and the declared set)."""
    argv: tuple
    env: dict
    cwd: Path
    config: dict
    config_path: Path
    instructions_path: Path
    executable: str
    sha256: str
    rg_path: str
    models: tuple = ()
    wire_api: str | None = None
    thinking_display: str | None = None
    verbosity: str | None = None
    dirs: dict = field(default_factory=dict)


# ---------------------------------------------------------------------------------------- model declarations
def _positive_tokens(value):
    return value if type(value) is int and value > 0 else None


def resolve_limits(*, context_limit_tokens: int | None = None, output_limit_tokens: int | None = None,
                   catalogue_model: object = None) -> dict:
    """Effective ``limit.context`` / ``limit.output`` (tokens) of one declared model (design 11 section 2.2).

    context_limit_tokens: ``agent.opencode.context_limit_tokens`` (positive int, used exactly) or None.
    output_limit_tokens: ``agent.opencode.output_limit_tokens`` (positive int, used exactly) or None.
    catalogue_model: the provider catalogue entry of the model (``model_catalogue.CatalogueModel``: positive int
        ``context_window`` / ``max_output_tokens`` or None) or None when the catalogue has no entry or was not read.

    context: explicit -> catalogue ``context_window`` -> ``FALLBACK_CONTEXT_LIMIT_TOKENS`` (200000).
    output: explicit -> ``min(catalogue max_output_tokens, OUTPUT_LIMIT_CEILING_TOKENS)`` -> 32000.
    Returns ``{"context": int, "output": int, "source": {"context": "explicit" | "catalogue" | "fallback",
    "output": "explicit" | "catalogue" | "catalogue_capped" | "fallback"}}``. A non-positive or non-int explicit value
    raises ``OpenCodeConfigError("opencode_limit_invalid:<context|output>")``.
    """
    for name, value in (("context", context_limit_tokens), ("output", output_limit_tokens)):
        if value is not None and _positive_tokens(value) is None:
            raise OpenCodeConfigError(f"opencode_limit_invalid:{name}")
    window = _positive_tokens(getattr(catalogue_model, "context_window", None))
    maximum = _positive_tokens(getattr(catalogue_model, "max_output_tokens", None))
    if context_limit_tokens is not None:
        context, context_source = context_limit_tokens, "explicit"
    elif window is not None:
        context, context_source = window, "catalogue"
    else:
        context, context_source = FALLBACK_CONTEXT_LIMIT_TOKENS, "fallback"
    if output_limit_tokens is not None:
        output, output_source = output_limit_tokens, "explicit"
    elif maximum is not None:
        output = min(maximum, OUTPUT_LIMIT_CEILING_TOKENS)
        output_source = "catalogue" if maximum <= OUTPUT_LIMIT_CEILING_TOKENS else "catalogue_capped"
    else:
        output, output_source = OUTPUT_LIMIT_CEILING_TOKENS, "fallback"
    return {"context": context, "output": output, "source": {"context": context_source, "output": output_source}}


def model_spec(model_id: str, *, context_limit_tokens: int | None = None, output_limit_tokens: int | None = None,
               catalogue_model: object = None) -> ModelSpec:
    """``ModelSpec`` of ``model_id`` (free text, as sent upstream; non-empty, no whitespace and no ``#``, which
    separates the variant in OpenCode's model ref -> ``OpenCodeConfigError("opencode_model_id_invalid")``).

    Limits from ``resolve_limits`` (explicit values apply to every model of the run); ``image_input`` from
    ``catalogue_model.image_input`` (None when there is no catalogue entry)."""
    if not isinstance(model_id, str) or not _MODEL_ID.match(model_id):
        raise OpenCodeConfigError("opencode_model_id_invalid")
    image_input = getattr(catalogue_model, "image_input", None)
    return ModelSpec(id=model_id, image_input=image_input if isinstance(image_input, bool) else None,
                     limits=resolve_limits(context_limit_tokens=context_limit_tokens,
                                           output_limit_tokens=output_limit_tokens, catalogue_model=catalogue_model))


def effort_variants(wire_api: str, *, thinking_display: str | None = None) -> list:
    """Model ``variants`` of the generated config: ``[{"id": <effort name>, "body": <request-body fragment>}]`` for
    every ``EFFORT_CHOICES[wire_api]`` name, in that order. OpenCode merges the selected variant's ``body`` into the
    request; explicit variants replace the ones the provider package would generate.

    anthropic_messages: ``{"thinking": {"type": "adaptive"}, "output_config": {"effort": <name>}}``; with
    ``thinking_display`` (``"summarized"`` | ``"omitted"``, default None = not sent) the ``thinking`` object also
    carries ``"display": <value>`` (whether the endpoint returns summarized thinking text; non-Anthropic gateways
    refuse the field).
    openai_responses: ``{"reasoning": {"effort": <name>}}`` (the package itself sends ``store: false`` and
    ``include: ["reasoning.encrypted_content"]``); ``thinking_display`` set raises
    ``OpenCodeConfigError("opencode_thinking_display_requires_anthropic_messages")``.
    Another wire raises ``OpenCodeConfigError("opencode_effort_unsupported_wire:<wire>")``; an unknown display value
    ``opencode_thinking_display_invalid``. Every call returns fresh objects."""
    if thinking_display is not None and thinking_display not in THINKING_DISPLAY_CHOICES:
        raise OpenCodeConfigError("opencode_thinking_display_invalid")
    if wire_api == "anthropic_messages":
        thinking = {"type": "adaptive"}
        if thinking_display is not None:
            thinking["display"] = thinking_display
        return [{"id": name, "body": {"thinking": dict(thinking), "output_config": {"effort": name}}}
                for name in EFFORT_CHOICES[wire_api]]
    if wire_api == "openai_responses":
        if thinking_display is not None:
            raise OpenCodeConfigError("opencode_thinking_display_requires_anthropic_messages")
        return [{"id": name, "body": {"reasoning": {"effort": name}}} for name in EFFORT_CHOICES[wire_api]]
    raise OpenCodeConfigError(f"opencode_effort_unsupported_wire:{wire_api}")


def model_entry(spec: ModelSpec, *, wire_api: str, thinking_display: str | None = None,
                verbosity: str | None = None) -> dict:
    """``providers.relay.models.<spec.id>`` value: ``{name, capabilities: {tools: true, input, output: ["text"]},
    limit: {context, output}, variants: effort_variants(wire_api, thinking_display=...)}``；走
    ``openai_responses`` 且配置了 ``verbosity`` 时写入 ``settings.textVerbosity``。"""
    entry = {
        "name": spec.id,
        "capabilities": {"tools": True, "input": spec.input_modalities, "output": ["text"]},
        "limit": {"context": spec.limits["context"], "output": spec.limits["output"]},
        "variants": effort_variants(wire_api, thinking_display=thinking_display),
    }
    if wire_api == "openai_responses" and verbosity:
        entry["settings"] = {"textVerbosity": verbosity}
    return entry


# --------------------------------------------------------------------------------------------- config body
def build_config(*, models: Sequence[ModelSpec], wire_api: str, relay_base_url: str,
                 instructions_path: str | os.PathLike, mcp_url: str | None = None,
                 mcp_headers: Mapping[str, str] | None = None, subagents: bool = True,
                 thinking_display: str | None = None, verbosity: str | None = None,
                 mcp_timeout_ms: int = 600000, allow_shell: bool = True,
                 allow_external_read: bool = True, actor_dir: str | os.PathLike | None = None) -> dict:
    """The generated OpenCode 2.x config document (native V2 shape, design 11 section 2.1).

    Parameters
    ----------
    models:
        Declared ``ModelSpec`` list, non-empty, unique ids; the first is the default ``model`` (``relay/<id>``).
        Only declared models can be selected (an undeclared id fails with ``provider.no-route``), so the subagent
        ``model`` parameter cannot leave the run's recorded models.
    wire_api:
        Upstream wire (``anthropic_messages`` | ``openai_responses``); selects the bundled provider package
        (``WIRE_PACKAGES``) and the effort variants. Another wire raises ``opencode_unsupported_wire:<wire>``.
    relay_base_url:
        Root URL of the local accounting relay (``http://127.0.0.1:<port>``); ``/v1`` is appended.
    instructions_path:
        Absolute path of the read-only instructions file (config ``instructions``; survives compaction and steers).
    mcp_url, mcp_headers:
        Streamable-HTTP MCP endpoint and headers (bearer) of the host tool server, declared as the ``ni_rua`` server
        with Code Mode off (``codemode: false``: registry tools stay direct ``ni_rua_<tool>`` calls) and timeouts
        startup 30000 ms, catalog 30000 ms, execution ``mcp_timeout_ms`` (ms, default 600000: tool calls can be
        long). Omit ``mcp_url`` for no MCP block.
    subagents:
        True (default) leaves the native ``subagent`` tool allowed; False adds a ``subagent`` deny rule.
    thinking_display:
        ``agent.opencode.thinking_display`` (None | ``"summarized"`` | ``"omitted"``); see ``effort_variants``.
    verbosity:
        ``model.verbosity``（仅在 ``openai_responses`` 线写入 ``settings.textVerbosity``）。

    Fixed settings: ``update: disable``, ``share: disabled``, ``snapshots: false``, ``tool_output`` (``TOOL_OUTPUT``),
    ``media.image`` (``MEDIA_IMAGE``: OpenCode's image re-encoding off), ``experimental.todo: true`` (the todo tool),
    permissions ``read * allow`` (closes the shipped ``read *.env -> ask``, which would wait for a human) followed by
    one ``deny`` per ``_DENIED_TOOLS`` name (no global allow: rules are appended after the shipped agent rules), and
    policies that make ``relay`` the only usable provider. ``lsp`` / ``formatter`` / ``compaction`` are left at
    OpenCode's defaults.
    """
    if wire_api not in WIRE_PACKAGES:
        raise OpenCodeConfigError(f"opencode_unsupported_wire:{wire_api}")
    specs = list(models)
    if not specs:
        raise OpenCodeConfigError("opencode_model_required")
    ids = [spec.id for spec in specs]
    if len(set(ids)) != len(ids):
        raise OpenCodeConfigError("opencode_model_declared_twice")
    if allow_external_read:
        permissions = [{"action": "read", "resource": "*", "effect": "allow"}]
    else:
        actor_root = os.fspath(Path(actor_dir).resolve()) if actor_dir is not None else None
        actor_glob = f"{actor_root}/*" if actor_root is not None else "*"
        actor_deep = f"{actor_root}/**" if actor_root is not None else "*"
        instr_file = os.fspath(instructions_path)
        permissions = [
            {"action": "external_directory", "resource": "*", "effect": "deny"},
            {"action": "read", "resource": "*", "effect": "deny"},
            {"action": "read", "resource": instr_file, "effect": "allow"},
            {"action": "read", "resource": actor_glob, "effect": "allow"},
            {"action": "read", "resource": actor_deep, "effect": "allow"},
            {"action": "glob", "resource": "*", "effect": "deny"},
            {"action": "glob", "resource": actor_deep, "effect": "allow"},
            {"action": "grep", "resource": "*", "effect": "deny"},
            {"action": "grep", "resource": actor_deep, "effect": "allow"},
            {"action": "list", "resource": "*", "effect": "deny"},
            {"action": "list", "resource": actor_glob, "effect": "allow"},
            {"action": "list", "resource": actor_deep, "effect": "allow"},
            {"action": "edit", "resource": "*", "effect": "deny"},
            {"action": "edit", "resource": actor_glob, "effect": "allow"},
            {"action": "edit", "resource": actor_deep, "effect": "allow"},
            {"action": "write", "resource": "*", "effect": "deny"},
            {"action": "write", "resource": actor_glob, "effect": "allow"},
            {"action": "write", "resource": actor_deep, "effect": "allow"},
            {"action": "patch", "resource": "*", "effect": "deny"},
            {"action": "patch", "resource": actor_glob, "effect": "allow"},
            {"action": "patch", "resource": actor_deep, "effect": "allow"},
        ]
    permissions += [{"action": name, "resource": "*", "effect": "deny"} for name in _DENIED_TOOLS]
    if not allow_shell:
        for shell_tool in ("shell", "bash", "execute"):
            permissions.append({"action": shell_tool, "resource": "*", "effect": "deny"})
    if not subagents:
        permissions.append({"action": SUBAGENT_TOOL, "resource": "*", "effect": "deny"})
    config = {
        "update": "disable", "share": "disabled", "snapshots": False,
        "model": f"{RELAY_PROVIDER_ID}/{specs[0].id}",
        "instructions": [os.fspath(instructions_path)],
        "tool_output": dict(TOOL_OUTPUT),
        "media": {"image": dict(MEDIA_IMAGE)},
        "experimental": {"todo": True},
        "permissions": permissions,
        "policies": [{"action": "provider.use", "resource": "*", "effect": "deny"},
                     {"action": "provider.use", "resource": RELAY_PROVIDER_ID, "effect": "allow"}],
        "providers": {RELAY_PROVIDER_ID: {
            "name": RELAY_PROVIDER_ID, "package": WIRE_PACKAGES[wire_api],
            "settings": {"baseURL": relay_base_url.rstrip("/") + "/v1", "apiKey": RELAY_PLACEHOLDER_KEY,
                         "timeout": False, "headerTimeout": False, "chunkTimeout": False},
            "models": {spec.id: model_entry(spec, wire_api=wire_api, thinking_display=thinking_display,
                                            verbosity=verbosity)
                       for spec in specs}}},
    }
    if mcp_url:
        config["mcp"] = {"servers": {"ni_rua": {
            "type": "remote", "url": mcp_url, "headers": dict(mcp_headers or {}), "oauth": False, "codemode": False,
            "timeout": {**_MCP_TIMEOUTS_MS, "execution": int(mcp_timeout_ms)}}}}
    return config


def _relay_models(config: Mapping) -> dict:
    try:
        models = config["providers"][RELAY_PROVIDER_ID]["models"]
    except (KeyError, TypeError):
        models = None
    if not isinstance(models, dict):
        raise OpenCodeConfigError("opencode_config_provider_missing")
    return models


def declared_model_ids(config: Mapping) -> list:
    """Ids of the models declared on the relay provider of a generated ``config``, in declaration order."""
    return list(_relay_models(config))


def add_model(config: Mapping, spec: ModelSpec, *, wire_api: str, thinking_display: str | None = None,
              verbosity: str | None = None) -> dict:
    """A copy of ``config`` with ``spec`` declared on the relay provider (an entry of the same id is replaced; the
    default ``model`` is unchanged). ``wire_api`` / ``thinking_display`` / ``verbosity`` must be the run's.
    Pure: write the result with ``write_config`` (OpenCode hot-reloads the file; poll ``GET /api/model``)."""
    updated = copy.deepcopy(dict(config))
    _relay_models(updated)[spec.id] = model_entry(spec, wire_api=wire_api, thinking_display=thinking_display,
                                                  verbosity=verbosity)
    return updated


def write_config(path: str | os.PathLike, config: Mapping) -> None:
    """Atomically replace the config file ``path`` with ``config`` (JSON): a temporary file in the same directory
    (``.ni-rua-config-*.tmp``, never a config file name OpenCode reads) is written, fsynced, made read-only (0444)
    and renamed over ``path``, so OpenCode's watcher never reads a partial document. A read-only directory (0555,
    as ``build_launch`` leaves it) is made owner-writable for the rename and restored afterwards."""
    target = Path(path)
    directory = target.parent
    mode = stat.S_IMODE(directory.stat().st_mode)
    text = json.dumps(config, ensure_ascii=False, indent=1) + "\n"
    if not mode & stat.S_IWUSR:
        os.chmod(directory, mode | stat.S_IWUSR | stat.S_IXUSR)
    try:
        handle, temporary = tempfile.mkstemp(prefix=".ni-rua-config-", suffix=".tmp", dir=directory)
        try:
            with os.fdopen(handle, "w", encoding="utf-8") as stream:
                stream.write(text)
                stream.flush()
                os.fsync(stream.fileno())
            os.chmod(temporary, 0o444)
            os.replace(temporary, target)
        except BaseException:
            try:
                os.unlink(temporary)
            except OSError:
                pass
            raise
    finally:
        if not mode & stat.S_IWUSR:
            os.chmod(directory, mode)


# ------------------------------------------------------------------------------------------------ checks
def check_managed_config(paths: Sequence[str] | None = None, *, user: str | None = None) -> None:
    """Single existence check: a system-managed OpenCode config overrides the generated tool permissions.

    Raises ``OpenCodeConfigError("opencode_managed_config_present")`` when any of ``MANAGED_CONFIG_PATHS`` (or the
    per-user macOS managed-preferences plist for ``user``) exists.  ``paths`` replaces the default list (tests).
    """
    candidates = list(MANAGED_CONFIG_PATHS if paths is None else paths)
    if paths is None:
        name = user or os.environ.get("USER") or os.environ.get("LOGNAME")
        if name:
            candidates.append(f"/Library/Managed Preferences/{name}/ai.opencode.managed.plist")
    for candidate in candidates:
        if os.path.lexists(candidate):
            raise OpenCodeConfigError("opencode_managed_config_present")


def _is_executable_file(path: str | os.PathLike | None) -> bool:
    if not path:
        return False
    text = os.fspath(path)
    return os.path.isabs(text) and os.path.isfile(text) and os.access(text, os.X_OK)


def resolve_ripgrep(
    explicit: str | None = None,
    search_path: str | None = None,
    *,
    env: Mapping[str, str] | None = None,
    executable: str | os.PathLike | None = None,
) -> str:
    """解析宿主可执行 ``rg`` 绝对路径（通过私有 bin 目录暴露给 OpenCode，避免 grep 失败或触发下载）。

    优先级：``explicit`` > ``PATH``（``search_path`` / ``env["PATH"]`` / 宿主 ``PATH``）>
    ``executable`` 同目录（含 realpath 父目录）的 ``rg`` > ``~/.local/share/opencode/bin/rg``。
    均不可执行时抛出 ``OpenCodeConfigError("opencode_ripgrep_missing")``。
    """
    if explicit:
        if _is_executable_file(explicit):
            return os.fspath(explicit)
        raise OpenCodeConfigError("opencode_ripgrep_missing")
    path_arg = search_path if search_path is not None else (env.get("PATH", "") if env is not None else None)
    found = shutil.which("rg", path=path_arg)
    if _is_executable_file(found):
        return found
    candidates: list[Path] = []
    if executable:
        exe_path = Path(executable)
        candidates.append(exe_path.parent / "rg")
        try:
            resolved_parent = exe_path.resolve().parent
        except OSError:
            resolved_parent = exe_path.parent
        if resolved_parent != exe_path.parent:
            candidates.append(resolved_parent / "rg")
    home_dir = (Path(env["HOME"]) if env.get("HOME") else None) if env is not None else Path.home()
    if home_dir is not None:
        candidates.append(home_dir / ".local" / "share" / "opencode" / "bin" / "rg")
    for candidate in candidates:
        if _is_executable_file(candidate):
            return os.fspath(candidate)
    raise OpenCodeConfigError("opencode_ripgrep_missing")


_VERSION_LINE = re.compile(r"^(?:opencode\s+)?v?(\d+\.\d+\.\d+(?:[-+][0-9A-Za-z.+-]+)?)\Z")


def parse_cli_version(output: str) -> str:
    """Bare version token of ``opencode --version`` output: 2.x prints ``opencode v2.2.7`` -> ``"2.2.7"`` (a bare
    ``2.2.7`` or ``v2.2.7`` line parses too). The first non-empty line is read; any other text is returned stripped
    as it is, so the exact-version check rejects it; no output -> ``""``."""
    line = next((item.strip() for item in (output or "").splitlines() if item.strip()), "")
    match = _VERSION_LINE.match(line)
    return match.group(1) if match else line


def probe_cli_version(executable: str, *, timeout_s: float = 15.0) -> str:
    """Run ``<executable> --version`` in a minimal environment; return ``parse_cli_version`` of its output.

    ``--version`` starts no background service. Without ``HOME``/``XDG_*`` set, OpenCode resolves the invoking
    user's real passwd home and creates ``~/.config/opencode``, ``~/.cache/opencode/bin`` ... there.  The probe instead
    runs with a fresh temporary directory as ``HOME`` and every ``XDG_*_HOME`` (and ``TMPDIR``: Bun extracts native
    libraries there), so nothing outside it is read or written; the directory is discarded after the call.
    """
    try:
        with tempfile.TemporaryDirectory(prefix="ni-rua-opencode-probe-") as home:
            env = {"PATH": os.defpath, "LC_ALL": "C", "HOME": home, "TMPDIR": home,
                   "XDG_CONFIG_HOME": os.path.join(home, "config"), "XDG_DATA_HOME": os.path.join(home, "data"),
                   "XDG_CACHE_HOME": os.path.join(home, "cache"), "XDG_STATE_HOME": os.path.join(home, "state")}
            completed = subprocess.run([executable, "--version"], capture_output=True, text=True, timeout=timeout_s,
                                       env=env, stdin=subprocess.DEVNULL)
    except (OSError, subprocess.SubprocessError) as error:
        raise OpenCodeConfigError(f"opencode_version_probe_failed:{type(error).__name__}") from error
    return parse_cli_version(completed.stdout)


_VERSION_CACHE: dict[tuple[str, int, int, int], str] = {}


def probe_cli_version_cached(executable: str, *, timeout_s: float = 15.0) -> str:
    """Cached ``probe_cli_version`` keyed by ``(realpath, mtime_ns, size, ino)`` for high-concurrency launches."""
    try:
        resolved = os.path.realpath(executable)
        st = os.stat(resolved)
        key = (resolved, int(st.st_mtime_ns), int(st.st_size), int(st.st_ino))
    except OSError:
        return probe_cli_version(executable, timeout_s=timeout_s)
    cached = _VERSION_CACHE.get(key)
    if cached is not None:
        return cached
    ver = probe_cli_version(executable, timeout_s=timeout_s)
    _VERSION_CACHE[key] = ver
    return ver


def check_health_version(info: object) -> str:
    """The server is ready and pinned: ``info`` is the HTTP 200 body of ``GET /api/info`` (``{version, pid, urls,
    paths}``; the server answers 503 with the same body while it is still starting, which is not ready). Returns the
    version; anything but ``OPENCODE_VERSION`` raises ``isolation.VersionMismatch("opencode_version_mismatch")``."""
    version = info.get("version") if isinstance(info, dict) else None
    return require_exact_version("opencode", version, OPENCODE_VERSION)


def check_served_models(served: object, config: Mapping) -> None:
    """``GET /api/model`` ``data`` (``served``: list of model infos) must list exactly the relay models of the
    generated ``config`` (provider ``relay``; the ``provider.use`` policy hides every other provider), each with its
    declared ``limit.context`` / ``limit.output``, ``capabilities.input`` and variants (ids and bodies, in order).
    Otherwise ``OpenCodeConfigError("opencode_served_models_mismatch")`` with ``.problems`` (sorted short codes such
    as ``missing:<id>``, ``unexpected:<provider>/<id>``, ``limit:<id>``, ``input:<id>``, ``variants:<id>``)."""
    declared = _relay_models(config)
    problems, seen = [], set()
    for item in served if isinstance(served, list) else ():
        if not isinstance(item, dict):
            problems.append("malformed")
            continue
        provider, model_id = item.get("providerID"), item.get("id")
        if provider != RELAY_PROVIDER_ID or model_id not in declared:
            problems.append(f"unexpected:{provider}/{model_id}")
            continue
        seen.add(model_id)
        want = declared[model_id]
        limit = item.get("limit") if isinstance(item.get("limit"), dict) else {}
        if (limit.get("context"), limit.get("output")) != (want["limit"]["context"], want["limit"]["output"]):
            problems.append(f"limit:{model_id}")
        capabilities = item.get("capabilities") if isinstance(item.get("capabilities"), dict) else {}
        if capabilities.get("input") != want["capabilities"]["input"]:
            problems.append(f"input:{model_id}")
        variants = item.get("variants") if isinstance(item.get("variants"), list) else None
        got = ([{"id": entry.get("id"), "body": entry.get("body")} if isinstance(entry, dict) else None
                for entry in variants] if variants is not None else None)
        if got != want["variants"]:
            problems.append(f"variants:{model_id}")
    if not isinstance(served, list):
        problems.append("malformed")
    problems += [f"missing:{model_id}" for model_id in declared if model_id not in seen]
    if problems:
        error = OpenCodeConfigError("opencode_served_models_mismatch")
        error.problems = sorted(set(problems))
        raise error


def load_expected_routes() -> list[str]:
    """The checked-in sorted ``"METHOD /path"`` operation list of the pinned binary's ``GET /openapi.json``
    (``opencode_routes.json``, 2.2.7: 170 operations)."""
    return list(json.loads(_ROUTES_FILE.read_text(encoding="utf-8"))["routes"])


def route_list_digest(routes: Sequence[str]) -> str:
    """Canonical sha256 (sorted, unique, compact JSON array) recorded as ``agent_backend.route_list_sha256``."""
    canonical = json.dumps(sorted(set(routes)), separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(canonical.encode("ascii")).hexdigest()


def check_route_list(actual: Sequence[str], expected: Sequence[str] | None = None) -> str:
    """Compare the server's ``GET /openapi.json`` operations with the checked-in list; return the digest or raise
    ``OpenCodeConfigError("opencode_route_list_mismatch")`` (``.added`` / ``.removed`` attached)."""
    wanted = set(load_expected_routes() if expected is None else expected)
    got = set(actual)
    if got != wanted:
        error = OpenCodeConfigError("opencode_route_list_mismatch")
        error.added, error.removed = sorted(got - wanted), sorted(wanted - got)
        raise error
    return route_list_digest(got)


def native_tool_ids(tool_names: Sequence[str], *, tool_name_prefix: str = "ni_rua_") -> list[str]:
    """Sorted native tool names of one model request: ``tool_names`` (the tool names the relay forwarded in the main
    thread's first request that declares tools, i.e. what the model really got; 2.x has no tool-listing route)
    without the host's MCP tools (``tool_name_prefix``). Feed it to ``provenance_block(enabled_tools=...)``."""
    return sorted({name for name in tool_names if isinstance(name, str) and not name.startswith(tool_name_prefix)})


# ----------------------------------------------------------------------------------------------- launch
def build_launch(run_dir: str | os.PathLike, *, executable: str, models: Sequence[ModelSpec], wire_api: str,
                 relay_base_url: str, instructions_text: str, port: int, password: str,
                 mcp_url: str | None = None, mcp_headers: Mapping[str, str] | None = None,
                 subagents: bool = True, thinking_display: str | None = None, verbosity: str | None = None,
                 ripgrep: str | None = None, base_env: Mapping[str, str] | None = None,
                 check_version: bool = True, managed_config_paths: Sequence[str] | None = None,
                 sandbox: bool = False, instance_id: str | None = None,
                 allow_shell: bool = True, allow_external_read: bool = True) -> OpenCodeLaunch:
    """Prepare the private tree below ``run_dir`` and return the launch description.

    Order: managed-config check -> executable + exact ``--version`` + sha256 -> ``rg`` -> private dirs
    (``<run>/private/opencode/{xdg/{config/opencode,data,cache,state},instructions,bin}``, ``private/tmp``) ->
    read-only instructions file (0444) -> ``build_config(models=..., ...)`` written atomically to
    ``xdg/config/opencode/opencode.json`` (0444) -> config dir chmod 0555 (the agent cannot drop agents, commands,
    skills or plugins into it; ``write_config`` lifts it for a rewrite) -> clean environment.

    The config dir is both ``OPENCODE_CONFIG_DIR`` (which replaces OpenCode's global config dir) and
    ``$XDG_CONFIG_HOME/opencode``, so every reader resolves the same private directory. ``port`` is the free loopback
    port chosen by the host and ``password`` the per-run ``OPENCODE_PASSWORD`` (basic auth, user ``opencode``).
    ``models`` is the declared ``ModelSpec`` list (first = the configured model), ``thinking_display`` the run's
    ``agent.opencode.thinking_display``, ``verbosity`` the run's ``model.verbosity``. ``check_version=False`` skips the ``--version`` probe (tests only).

    ``sandbox=True`` prepares the launch for ``AgentSandbox`` (design R4.4): ``argv`` starts with
    ``SANDBOX_EXECUTABLE`` (the namespace path of the pinned binary, mounted from ``executable``), no private ``rg``
    link is made (the sandbox mounts the host ``rg`` at ``SANDBOX_RIPGREP`` and puts it on ``PATH``; ``rg_path``
    records that path), and ``env`` holds only the OpenCode variables: the sandbox sets ``PATH``, ``HOME`` (the
    actor directory) and ``TMPDIR`` (``/tmp`` tmpfs, must allow exec) itself.  ``sandbox_spec`` turns the launch into
    the mounts.  ``instance_id`` (the session's ``runtime/agent.instance_id``, default None) sets
    ``isolation.AGENT_MARKER_ENV`` so ``isolation.terminate_agent_processes`` finds the server's detached children.
    """
    check_managed_config(managed_config_paths)
    if not os.path.isabs(executable) or not os.path.isfile(executable) or not os.access(executable, os.X_OK):
        raise OpenCodeConfigError("opencode_executable_invalid")
    if check_version:
        require_exact_version("opencode", probe_cli_version_cached(executable), OPENCODE_VERSION)
    digest = file_sha256_cached(executable)
    rg = resolve_ripgrep(ripgrep, executable=executable)
    run = Path(run_dir).resolve(strict=True)
    actor = run / "actor"
    actor.mkdir(exist_ok=True)
    dirs = private_dirs(run, ["private/tmp", CONFIG_DIR, "private/opencode/xdg/data", "private/opencode/xdg/cache",
                              "private/opencode/xdg/state", "private/opencode/instructions", "private/opencode/bin"])
    bin_dir = dirs["private/opencode/bin"]
    if not sandbox:
        link = bin_dir / "rg"
        if link.is_symlink() or link.exists():
            link.unlink()
        os.symlink(rg, link)
    instructions = dirs["private/opencode/instructions"] / INSTRUCTIONS_FILE_NAME
    if instructions.exists():
        os.chmod(instructions, 0o600)
    instructions.write_text(instructions_text, encoding="utf-8")
    os.chmod(instructions, 0o444)
    specs = tuple(models)
    config = build_config(models=specs, wire_api=wire_api, relay_base_url=relay_base_url,
                          instructions_path=instructions, mcp_url=mcp_url, mcp_headers=mcp_headers,
                          subagents=subagents, thinking_display=thinking_display, verbosity=verbosity,
                          allow_shell=allow_shell, allow_external_read=allow_external_read,
                          actor_dir=actor)
    config_dir = dirs[CONFIG_DIR]
    config_path = config_dir / CONFIG_FILE_NAME
    write_config(config_path, config)
    os.chmod(config_dir, 0o555)
    extra = {
        "XDG_CONFIG_HOME": str(config_dir.parent), "XDG_DATA_HOME": str(dirs["private/opencode/xdg/data"]),
        "XDG_CACHE_HOME": str(dirs["private/opencode/xdg/cache"]),
        "XDG_STATE_HOME": str(dirs["private/opencode/xdg/state"]),
        "OPENCODE_CONFIG_DIR": str(config_dir), "OPENCODE_PASSWORD": password, **_FLAGS,
    }
    env = clean_environment(base_env or {}, path_dirs=[] if sandbox else [bin_dir], home=actor,
                            tmpdir=dirs["private/tmp"], extra=extra, instance_id=instance_id)
    if sandbox:
        for name in ("PATH", "HOME", "TMPDIR"):     # owned by the sandbox (reserved or its own defaults)
            env.pop(name, None)
    argv = (SANDBOX_EXECUTABLE if sandbox else executable, "serve", "--hostname", "127.0.0.1", "--port", str(int(port)))
    return OpenCodeLaunch(argv=argv, env=env, cwd=actor, config=config, config_path=config_path,
                          instructions_path=instructions, executable=os.path.realpath(executable), sha256=digest,
                          rg_path=SANDBOX_RIPGREP if sandbox else rg, models=specs, wire_api=wire_api,
                          thinking_display=thinking_display, verbosity=verbosity, dirs=dirs)


def tui_attach_command(launch: OpenCodeLaunch, *, server_url: str, session_id: str,
                       directory: str | os.PathLike) -> tuple[list, dict]:
    """``(argv, env)`` of the read-only native TUI attached to the running server (``features.terminal``).

    OpenCode 2.2.7 form (``attach`` no longer exists): ``<pinned binary> --server <server_url> --session <session_id>
    <directory>``. Verified with the pinned binary: it reads the session, makes no model call, creates no session,
    spawns no background service and writes only ``xdg/state/opencode/latest/tui/tabs.json``.

    server_url: the server's own loopback URL (``http://127.0.0.1:<port>``) that the session's client uses.
    session_id: the existing main session (``ses_...``) to show. Pass only an id the server has: ``--session``
        creates a missing one. An empty id raises ``ValueError("opencode_tui_session_required")``.
    directory: the server's working directory (the actor dir), so the TUI talks to the server's existing location.
    env: a copy of ``launch.env`` (private XDG dirs, read-only config dir, the disable flags, the process marker);
        the TUI reads the basic-auth password from ``OPENCODE_PASSWORD`` in it, so it is never on the argv.
    """
    if not isinstance(session_id, str) or not session_id.strip():
        raise ValueError("opencode_tui_session_required")
    argv = [launch.executable, "--server", server_url, "--session", session_id, os.fspath(directory)]
    return argv, dict(launch.env)


def sandbox_spec(launch: OpenCodeLaunch, *, harness_source: str | os.PathLike) -> dict:
    """Keyword arguments of ``AgentSandbox`` for a ``build_launch(..., sandbox=True)`` launch (design R4.4).

    * ``state_dirs``: the private XDG ``data``, ``cache`` and ``state`` directories, read-write at their host paths
      (OpenCode's database, logs, shell output and repos live there).
    * ``executables``: the pinned binary, read-only, at ``SANDBOX_EXECUTABLE``.
    * ``read_paths``: the XDG config home holding the one config dir (read-only; the host rewrites
      ``opencode.json`` on its side of the bind and OpenCode hot-reloads it), the instructions file (read-only) and
      ``harness_source`` (the ``actor-source`` snapshot) at ``/harness``.
    * ``env``: ``launch.env`` (XDG roots, ``OPENCODE_CONFIG_DIR``, server password, flags).

    The caller adds ``ports`` (relay and MCP, outbound) and ``inbound_ports=[server port]`` (the host reaches the
    server through ``AgentSandbox.inbound_socket_path``).  The checkout, the user's home and the RoboDojo /
    simulator tree are never mounted.
    """
    state = [launch.dirs[f"private/opencode/xdg/{name}"] for name in ("data", "cache", "state")]
    read_only = [launch.dirs[CONFIG_DIR].parent, launch.instructions_path]
    return {
        "state_dirs": [(path, path) for path in state],
        "executables": [(launch.executable, SANDBOX_EXECUTABLE)],
        "read_paths": [(path, path) for path in read_only] + [(os.fspath(harness_source), SANDBOX_HARNESS_SOURCE)],
        "env": dict(launch.env),
    }


def provenance_block(launch: OpenCodeLaunch, *, route_list_sha256: str | None, wire_api: str, model: str,
                     effort: str | None = None, instructions_sha256: str | None = None,
                     tool_name_prefix: str = "ni_rua_", subagents: bool = True,
                     enabled_tools: Sequence[str] | None = None, models: Sequence[ModelSpec] | None = None,
                     thinking_display: str | None = None) -> dict:
    """``manifest.agent_backend`` fragment for OpenCode (observer-only; never enters model context).  Secrets
    (password, MCP bearer) are not recorded; the model id is the configured name (``native_model_id``).

    ``models``: the declared ``ModelSpec`` list (default ``launch.models``; pass the session's current list after
    ``add_model``) -> ``declared_models`` (``ModelSpec.record``) and ``effective_limits`` (the limits of ``model``,
    ``{context, output, source}``). ``thinking_display``: default ``launch.thinking_display``.
    ``native_effort``: ``{variant, body, thinking_display, declared_variants, wire_field}`` (``body`` = the request
    fragment of the ``effort`` variant), None when ``effort`` is None.
    ``native_tools.disabled`` lists ``{name, reason}`` (``online_access`` | ``waits_for_human`` |
    ``subagents_disabled``); ``enabled_tools`` (``native_tool_ids`` of the first with-tools request the relay
    forwarded) is recorded sorted, ``None`` until known."""
    specs = list(launch.models if models is None else models)
    display = launch.thinking_display if thinking_display is None else thinking_display
    disabled = [{"name": name, "reason": reason} for name, reason in _DENIED_TOOLS.items()]
    if not subagents:
        disabled.append({"name": SUBAGENT_TOOL, "reason": "subagents_disabled"})
    current = next((spec for spec in specs if spec.id == model), None)
    native_effort = None
    if effort is not None:
        variants = effort_variants(wire_api, thinking_display=display) if wire_api in EFFORT_CHOICES else []
        native_effort = {
            "variant": effort, "body": next((item["body"] for item in variants if item["id"] == effort), None),
            "thinking_display": display, "declared_variants": [item["id"] for item in variants],
            "wire_field": EFFORT_WIRE_FIELDS.get(wire_api)}
    return {
        "kind": "opencode", "sdk": None,
        "cli": {"version": OPENCODE_VERSION, "executable": launch.executable, "sha256": launch.sha256},
        "wire_api": wire_api, "native_model_id": model, "provider_alias": RELAY_PROVIDER_ID,
        "provider_package": WIRE_PACKAGES.get(wire_api),
        "route_list_sha256": route_list_sha256, "native_effort": native_effort,
        "effective_limits": copy.deepcopy(current.limits) if current is not None else None,
        "declared_models": [spec.record() for spec in specs],
        "tool_transport": {"kind": "mcp_streamable_http", "tool_name_prefix": tool_name_prefix},
        "prompt_channel": {"instructions_channel": "config_instructions_file", "native_base_prompt": "preserved",
                           "instructions_sha256": instructions_sha256},
        "native_tools": {"enabled": sorted(enabled_tools) if enabled_tools is not None else None,
                         "disabled": disabled},
        "native_settings": {"flags": dict(_FLAGS), "config_channel": "config_file", "config_dir_read_only": True,
                            "rg": launch.rg_path, "media_image": dict(MEDIA_IMAGE), "tool_output": dict(TOOL_OUTPUT),
                            "experimental": {"todo": True}, "mcp_codemode": False,
                            "provider_timeouts": {"timeout": False, "chunkTimeout": False, "headerTimeout": False}},
    }
