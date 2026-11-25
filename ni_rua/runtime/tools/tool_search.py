"""Provider-agnostic ``tool_search`` function tool for deferred actor tools.

Hosts without OpenAI's proprietary ``{"type": "tool_search"}`` namespace tool (such as
OpenCode, Claude Code, or standalone ``Harness`` loops) publish only non-deferred core
tools plus ``tool_search`` in the model's ``tools`` array while keeping every registered
MCP route callable once unlocked. Calling ``tool_search`` searches ``deferLoading=True``
actor tools by exact name or keywords, unlocks matched tools in the episode ledger, and
returns their complete descriptions (including ``Input field guide``) and JSON schemas
without mutating the system prompt or ``tools`` prefix across turns.
"""
from __future__ import annotations

from copy import deepcopy
import re
from typing import Any

from ...core.episode_state import StateSlot, slot_get, slot_set
from ...core.schema_guide import strip_guided_schema_descriptions
from ...core.tool_errors import ToolInputError
from ...core.tool_output import ToolOutput
from .registry import ToolSpec

TOOL_SEARCH_NAME = "tool_search"
UNLOCKED_DEFERRED_TOOLS_SLOT: StateSlot[set] = StateSlot("unlocked_deferred_tools", set, scope="run")

_KNOWN_TOOL_PREFIXES = ("mcp__ni_rua__", "ni_rua_")
_TOP_LEVEL_COMBINATORS = ("anyOf", "oneOf", "allOf", "if", "then", "else")
_TOKEN_RE = re.compile(r"[a-z0-9_]+")

TOOL_SEARCH_DESCRIPTION = (
    "Search and unlock on-demand (deferred) tools so they can be called in this episode. "
    "Pass exact tool names in names=[...] or capability keywords in query='...' (for example "
    "'robot_information', 'contact overlay', 'trace', or 'memory'). Matched on-demand tools are "
    "unlocked immediately and their complete description, Input field guide, and inputSchema are "
    "returned. No robot motion or world advancement."
)

TOOL_SEARCH_INPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "query": {
            "type": "string",
            "maxLength": 512,
            "description": (
                "Case-insensitive tool name or capability keywords to match against on-demand tools."
            ),
        },
        "names": {
            "type": "array",
            "minItems": 1,
            "maxItems": 16,
            "uniqueItems": True,
            "items": {"type": "string", "minLength": 1, "maxLength": 128},
            "description": "Optional exact tool names to load directly (with or without MCP prefix).",
        },
        "max_results": {
            "type": "integer",
            "minimum": 1,
            "maximum": 16,
            "default": 8,
            "description": "Maximum number of keyword-matched tools to return (1..16, default 8).",
        },
    },
    "additionalProperties": False,
}

TOOL_SEARCH_OUTPUT_SCHEMA = {
    "type": "object",
    "required": ["matched_count", "unlocked_tools", "available_on_demand_tools", "tools"],
    "additionalProperties": True,
    "properties": {
        "matched_count": {"type": "integer", "minimum": 0},
        "unlocked_tools": {"type": "array", "items": {"type": "string"}},
        "available_on_demand_tools": {"type": "array", "items": {"type": "string"}},
        "not_found_names": {"type": "array", "items": {"type": "string"}},
        "tools": {
            "type": "array",
            "items": {
                "type": "object",
                "required": ["name", "description", "inputSchema"],
                "additionalProperties": True,
                "properties": {
                    "name": {"type": "string"},
                    "description": {"type": "string"},
                    "inputSchema": {"type": "object"},
                    "deferred": {"type": "boolean"},
                },
            },
        },
    },
}


def strip_mcp_tool_prefix(name: str) -> str:
    """Normalize an MCP-prefixed tool name (e.g. ``ni_rua_robot_information``) to its registry name."""
    cleaned = name.strip()
    for prefix in _KNOWN_TOOL_PREFIXES:
        if cleaned.startswith(prefix) and len(cleaned) > len(prefix):
            return cleaned[len(prefix) :]
    return cleaned


def _portable_schema_if_mcp(runtime: Any, schema: dict[str, Any], description: str) -> dict[str, Any]:
    stripped = strip_guided_schema_descriptions(schema, description)
    if getattr(runtime, "delivery_channel", "codex_tool") == "agent_tool":
        if "oneOf" in stripped and not stripped.get("properties"):
            return {"type": "object", "properties": {}, "additionalProperties": True}
        return {k: deepcopy(v) for k, v in stripped.items() if k not in _TOP_LEVEL_COMBINATORS}
    return deepcopy(stripped)


def _score_tool(tool: dict[str, Any], raw_query: str, tokens: list[str]) -> int:
    name = tool["name"].lower()
    desc = (tool.get("description") or "").lower()
    first_sentence = desc.split("\n", 1)[0]
    q_norm = strip_mcp_tool_prefix(raw_query.lower())
    score = 0
    if q_norm == name:
        score += 1000
    elif q_norm and q_norm in name:
        score += 400
    elif name in q_norm:
        score += 300
    for tok in tokens:
        if tok == name:
            score += 500
        elif tok in name.split("_"):
            score += 150
        elif tok in name:
            score += 80
        elif tok in first_sentence:
            score += 35
        elif tok in desc:
            score += 12
    return score


def handle_tool_search(context, arguments: dict[str, Any], correlation) -> Any:
    services = getattr(context, "services", None)
    runtime = getattr(services, "_runtime", services) if services is not None else getattr(context, "runtime", context)
    raw_query = (arguments.get("query") or "").strip() if isinstance(arguments.get("query"), str) else ""
    raw_names = arguments.get("names") if isinstance(arguments.get("names"), list) else []
    max_results = int(arguments.get("max_results", 8))

    if not raw_query and not raw_names:
        raise ToolInputError(
            "tool_search_query_required",
            field_path="/query",
            actual=arguments,
            suggestion=(
                "Supply non-empty query keywords (e.g. {'query': 'robot_information'}) or exact "
                "names (e.g. {'names': ['robot_information']})."
            ),
            minimal_example={"query": "robot_information"},
        )

    catalog = [
        item
        for item in runtime.tool_registry.catalog(runtime, role="actor")
        if item.get("name") != TOOL_SEARCH_NAME
    ]
    by_name = {item["name"]: item for item in catalog}
    deferred_names = [item["name"] for item in catalog if item.get("deferLoading")]
    deferred_set = set(deferred_names)

    matched_names: list[str] = []
    not_found: list[str] = []
    for entry in raw_names:
        norm = strip_mcp_tool_prefix(str(entry))
        if norm in by_name:
            if norm not in matched_names:
                matched_names.append(norm)
        else:
            not_found.append(str(entry))

    if raw_query:
        tokens = [
            strip_mcp_tool_prefix(tok)
            for tok in _TOKEN_RE.findall(raw_query.lower())
            if tok not in {"ni_rua", "ni", "rua", "mcp", "tool", "tools"}
        ]
        if not tokens:
            tokens = _TOKEN_RE.findall(raw_query.lower())
        scored_deferred = []
        scored_eager = []
        for item in catalog:
            if item["name"] in matched_names:
                continue
            sc = _score_tool(item, raw_query, tokens)
            if sc > 0:
                if item["name"] in deferred_set:
                    scored_deferred.append((sc, item["name"]))
                else:
                    scored_eager.append((sc, item["name"]))
        ranked = scored_deferred if scored_deferred else scored_eager
        ranked.sort(key=lambda pair: (-pair[0], pair[1]))
        for _, tname in ranked[:max_results]:
            if tname not in matched_names:
                matched_names.append(tname)

    newly_unlocked = {name for name in matched_names if name in deferred_set}
    lock = getattr(runtime, "lock", None)
    if lock is not None:
        with lock:
            unlocked = set(slot_get(runtime, UNLOCKED_DEFERRED_TOOLS_SLOT))
            unlocked.update(newly_unlocked)
            slot_set(runtime, UNLOCKED_DEFERRED_TOOLS_SLOT, unlocked)
            unlocked_sorted = sorted(unlocked & deferred_set)
    else:
        unlocked = set(slot_get(runtime, UNLOCKED_DEFERRED_TOOLS_SLOT))
        unlocked.update(newly_unlocked)
        slot_set(runtime, UNLOCKED_DEFERRED_TOOLS_SLOT, unlocked)
        unlocked_sorted = sorted(unlocked & deferred_set)

    returned_tools = []
    for name in matched_names:
        item = by_name[name]
        returned_tools.append(
            {
                "name": name,
                "deferred": name in deferred_set,
                "description": item["description"],
                "inputSchema": _portable_schema_if_mcp(runtime, item["inputSchema"], item["description"]),
            }
        )

    payload: dict[str, Any] = {
        "matched_count": len(returned_tools),
        "unlocked_tools": unlocked_sorted,
        "available_on_demand_tools": sorted(deferred_names),
        "tools": returned_tools,
    }
    if not_found:
        payload["not_found_names"] = not_found
    return ToolOutput(payload, kind="tool_search").response()


def make_tool_search_tool() -> ToolSpec:
    """Build the ``tool_search`` ``ToolSpec`` for provider-agnostic deferred tool loading."""
    definition = {
        "type": "function",
        "name": TOOL_SEARCH_NAME,
        "description": TOOL_SEARCH_DESCRIPTION,
        "inputSchema": deepcopy(TOOL_SEARCH_INPUT_SCHEMA),
    }
    return ToolSpec(
        definition,
        handle_tool_search,
        roles=frozenset({"actor"}),
        effect="read",
        api_version="1.0",
        output_schemas={"tool_search": deepcopy(TOOL_SEARCH_OUTPUT_SCHEMA)},
        parallel=True,
    )
