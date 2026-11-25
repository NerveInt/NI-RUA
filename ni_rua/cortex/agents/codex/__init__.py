"""Codex agent backend package: session, bubblewrap sandbox, and WS-stdio transport bridge."""

from importlib import import_module

_SUBMODULES = frozenset({"bridge", "sandbox", "session"})
_EXPORTS = {
    "CODEX_CLI_VERSION": ".session",
    "CODEX_PROTOCOL_VERSION": ".session",
    "CODEX_SDK_VERSION": ".session",
    "DEFAULT_MODEL_VERBOSITY": ".session",
    "DYNAMIC_TOOL_NAMESPACE": ".session",
    "MODEL_COMPLETION_COUNTING": ".session",
    "MODEL_PROXY_PROVIDER": ".session",
    "NATIVE_RESPONSE_SCHEMAS": ".session",
    "UNIX_SOCKET_PATH_MAX": ".session",
    "VALID_MODEL_VERBOSITIES": ".session",
    "CodexSandbox": ".sandbox",
    "CodexSession": ".session",
    "RecordedClient": ".session",
    "build_bridge_command": ".bridge",
    "normalize_token_usage": ".session",
    "proxy": ".bridge",
    "require_codex_version": ".session",
    "write_direct_model_catalog": ".session",
}

__all__ = list(_EXPORTS)


def __getattr__(name):
    if name in _SUBMODULES:
        return import_module(f".{name}", __name__)
    module_name = _EXPORTS.get(name)
    if module_name is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(module_name, __name__), name)
    globals()[name] = value
    return value


def __dir__():
    return sorted(set(globals()) | set(__all__))
