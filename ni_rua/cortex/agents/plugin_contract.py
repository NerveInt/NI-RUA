"""Single-source contract for the ``agent-plugin`` NPM distribution (zero runtime/apps imports)."""

from __future__ import annotations

import json
from pathlib import Path

from ...core.config import ExternalConfig
from .claude_code.options import _MCP_FLAGS
from .opencode.config import MEDIA_IMAGE, OPENCODE_VERSION, TOOL_OUTPUT

_CODEX_MODELS_PATH = Path(__file__).resolve().parents[1] / "session" / "codex_models.json"
CODEX_DIRECT_CATALOG_FILE = "codex-models.direct.json"

SCHEMA = "ni_rua.agent_plugin.manifest.v1"
SERVER_NAME = "ni_rua"
PLUGIN_ID = "ni_rua@ni_rua"
MARKETPLACE = "ni_rua"
CLAUDE_ALLOW = "mcp__plugin_ni_rua_ni_rua"

# Codex cuts long MCP text outputs unless per-tool output_token_limit >= 100,000 (~4 bytes/token vs 120k max).
OUTPUT_TOKEN_LIMIT = 100_000
LARGEST_KNOWN_RESULT_CHARS = 120_000
DEFAULT_WEB_PORT = 8781
CODEX_TIMEOUTS = {"tool_timeout_sec": 3600, "startup_timeout_sec": 30}
OPENCODE_TIMEOUT_MS = {"startup": 30000, "catalog": 30000, "execution": 3600000}
SKILLS = ("ni-rua-setup", "ni-rua-robot", "ni-rua-experiments")

_VERIFIED_HOSTS = {
    "claude": ("2.1.321", "2.1.325", "2.1.328"),
    "codex": ("0.160.0", "0.167.0", "0.169.0"),
    "opencode": (OPENCODE_VERSION,),
}


def default_port() -> int:
    """Return the default external-backend loopback MCP port from ``ExternalConfig``."""
    return int(ExternalConfig().mcp_port)


def default_web_port() -> int:
    """返回默认本地 Web 控制台端口（与 SO-101 独立预设一致）。"""
    return DEFAULT_WEB_PORT


def default_mcp_url(port: int | None = None) -> str:
    """Return the external MCP loopback URL for ``port`` (default ``default_port()``)."""
    resolved = default_port() if port is None else int(port)
    return f"http://127.0.0.1:{resolved}/mcp"


def claude_env() -> dict[str, str]:
    """Return Claude Code host MCP environment overrides without idle-timeout termination."""
    return {
        key: value
        for key, value in _MCP_FLAGS.items()
        if key != "CLAUDE_CODE_MCP_TOOL_IDLE_TIMEOUT"
    }


def opencode_tool_output() -> dict[str, int]:
    """Return OpenCode's host-mode ``tool_output`` line/byte bounds."""
    return dict(TOOL_OUTPUT)


def opencode_media_image() -> dict[str, object]:
    """Return OpenCode's host-mode ``media.image`` settings (disables lossy PNG re-encoding)."""
    return dict(MEDIA_IMAGE)


def codex_code_mode_only_models() -> list[str]:
    """返回内置标记为 ``tool_mode="code_mode_only"`` 的 Codex 模型列表（在 Linux 上默认触发 code-mode host 崩溃）。"""
    catalog = json.loads(_CODEX_MODELS_PATH.read_text(encoding="utf-8"))
    return sorted(
        str(model["slug"])
        for model in catalog.get("models", ())
        if isinstance(model, dict) and model.get("tool_mode") == "code_mode_only" and model.get("slug")
    )


def render_codex_direct_model_catalog() -> str:
    """生成将全部 Codex 内置模型改为 ``tool_mode="direct"`` 且关闭 ``supports_search_tool`` 的直连工具目录 JSON。"""
    catalog = json.loads(_CODEX_MODELS_PATH.read_text(encoding="utf-8"))
    for model in catalog.get("models", ()):
        if isinstance(model, dict):
            model["tool_mode"] = "direct"
            model["supports_search_tool"] = False
    return json.dumps(catalog, ensure_ascii=False) + "\n"


def verified_hosts() -> dict[str, list[str]]:
    """Return verified CLI version strings keyed by host (``claude``, ``codex``, ``opencode``)."""
    return {host: list(versions) for host, versions in _VERIFIED_HOSTS.items()}


def _normalize_tool_definitions(tool_names, tool_definitions=None) -> list[dict]:
    """规范化静态 MCP 工具目录定义（供 stdio 桥接器在网关未启动时返回合法 tools/list）。"""
    sorted_names = sorted(set(tool_names))
    if tool_definitions is None:
        return [
            {
                "name": name,
                "description": f"NI-RUA robot tool `{name}` (local gateway starts automatically on first call).",
                "inputSchema": {"type": "object", "properties": {}},
            }
            for name in sorted_names
        ]
    by_name = {str(item["name"]): dict(item) for item in tool_definitions}
    return [by_name[name] for name in sorted_names if name in by_name]


def build_manifest(*, version: str, tool_names, tool_definitions=None) -> dict:
    """Assemble the complete machine-readable ``agent-plugin`` manifest dictionary."""
    sorted_names = sorted(set(tool_names))
    return {
        "schema": SCHEMA,
        "version": str(version),
        "server_name": SERVER_NAME,
        "plugin_id": PLUGIN_ID,
        "marketplace": MARKETPLACE,
        "claude_allow": CLAUDE_ALLOW,
        "default_port": default_port(),
        "web_port": default_web_port(),
        "mcp_url": default_mcp_url(),
        "claude_env": claude_env(),
        "opencode_tool_output": opencode_tool_output(),
        "opencode_media_image": opencode_media_image(),
        "opencode_timeout_ms": dict(OPENCODE_TIMEOUT_MS),
        "codex_timeouts": dict(CODEX_TIMEOUTS),
        "codex_direct_catalog": CODEX_DIRECT_CATALOG_FILE,
        "codex_code_mode_only_models": codex_code_mode_only_models(),
        "output_token_limit": OUTPUT_TOKEN_LIMIT,
        "largest_known_result_chars": LARGEST_KNOWN_RESULT_CHARS,
        "skills": list(SKILLS),
        "verified_hosts": verified_hosts(),
        "tool_names": sorted_names,
        "tool_definitions": _normalize_tool_definitions(sorted_names, tool_definitions),
    }


def render_files(*, version: str, tool_names, tool_definitions=None) -> dict[str, str]:
    """Render generated ``agent-plugin/`` artifacts keyed by package-relative path."""
    manifest = build_manifest(version=version, tool_names=tool_names, tool_definitions=tool_definitions)
    codex_entry = {
        "command": "ni-rua-plugin",
        "args": ["mcp"],
        "default_tools_approval_mode": "approve",
        **CODEX_TIMEOUTS,
        "tools": {name: {"output_token_limit": OUTPUT_TOKEN_LIMIT} for name in manifest["tool_names"]},
    }
    codex_doc = {"mcpServers": {SERVER_NAME: codex_entry}}
    claude_doc = {
        "mcpServers": {
            SERVER_NAME: {
                "command": "node",
                "args": ["${CLAUDE_PLUGIN_ROOT}/bin/cli.mjs", "mcp"],
                "alwaysLoad": True,
            }
        }
    }
    return {
        ".codex-mcp.json": json.dumps(codex_doc, indent=2) + "\n",
        ".mcp.json": json.dumps(claude_doc, indent=2) + "\n",
        "manifest.generated.json": json.dumps(manifest, indent=2) + "\n",
        CODEX_DIRECT_CATALOG_FILE: render_codex_direct_model_catalog(),
    }
