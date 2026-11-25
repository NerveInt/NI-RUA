"""Host-owned streamable-HTTP MCP endpoint for registry tools (Claude Code, OpenCode).

One loopback endpoint serves the central tool registry to every MCP-speaking agent backend (design 04 §3.1-§3.4).
It implements the minimal JSON-RPC subset both pinned clients use, in-repo and without the ``mcp`` package:

==============================  ===================================================================================
POST ``initialize``             protocol negotiation (``SUPPORTED_PROTOCOL_VERSIONS``), ``tools`` capability, no
                                ``instructions`` (NI-RUA instructions have one channel) unless the owner passes the
                                short fixed ``instructions`` text (external agent backend only)
POST ``notifications/*``        202, no body; ``notifications/cancelled`` abandons a call that is still in flight
                                and is ignored once that call's response has been completely written
POST ``ping``                   ``{}``
POST ``tools/list``             the whole catalog, scrubbed once by ``ActorPresentation`` (see ``mcp_tool_catalog``);
                                each ``inputSchema`` is projected to the portable form: top-level ``anyOf``/``oneOf``/
                                ``allOf``/``if``/``then``/``else`` removed (``task_plan``)
POST ``tools/call``             ``call_tool`` -> ``encode_mcp_result``; SSE with ``notifications/progress`` keepalive
                                when the client accepts ``text/event-stream``, else one JSON body
POST anything else              JSON-RPC error -32601 (includes Claude Code's ``server/discover`` probe)
GET / DELETE                    405 (no server-initiated stream, no MCP session to delete)
==============================  ===================================================================================

Facts versus presentation: the server never parses tool text. ``call_tool`` returns the wire envelope
``{"success": bool, "contentItems": [...]}`` that ``Harness.handle_tool`` produces (after the session's second
presentation scrub); ``encode_mcp_result`` maps it item by item, passing image and audio bytes through unchanged.

Delivery certainty: a ``tools/call`` whose answer may not have reached the agent calls
``on_delivery_failure(call_id, reason)`` exactly once. The admitted call itself is never aborted; its late result
is discarded. Reasons (``DELIVERY_FAILURE_REASONS``):

* ``"tool_reply_delivery_uncertain"`` - a write error before the response was completely written, a
  ``notifications/cancelled`` or a peer close while the call was in flight, or ``close()`` with the call in flight.
  A call abandoned before ``call_tool`` was entered is never executed and not reported.
* ``"tool_reply_encoding_failed"`` - the result could not be encoded (e.g. a non-``data:`` image URL); the agent
  received a JSON-RPC error instead of the result.

Raw log: ``raw_log(direction, message)`` (normally ``Store.raw_agent(..., channel="mcp")``) receives every received
JSON-RPC request/notification (``"received"``) and every completely written response (``"sent"``), full content.
The logged copy of a ``tools/call`` request carries one host annotation, ``params._meta.item_id = <call id>``; it is
never sent to anyone. Progress keepalives are not logged, and a response whose write failed is not logged as sent,
so the log proves only what was actually delivered.
"""
from copy import deepcopy
from dataclasses import dataclass, field
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from importlib.metadata import PackageNotFoundError, version as package_version
import inspect
import json
import re
import secrets
import select
import socket
import threading
import time
import traceback
from urllib.parse import urlsplit

from ...core.contracts import canonical, new_id
from ...core.tool_errors import (
    INFRASTRUCTURE_DOMAIN_CODES,
    INFRASTRUCTURE_TYPE_NAMES,
    ToolInputError,
    _IK_SERVICE_UNREACHABLE_SUGGESTION,
    _camel_to_snake,
    _extract_wire_error_type,
    _is_infrastructure_message,
    _is_known_domain_value_message,
    _jsonschema,
    error_response,
    extract_domain_code,
    scrub_host_paths as _scrub_host_path_str,
)
from ...core.tool_output import InvalidContentItem, ToolOutput, validate_content_part, wire_response

__all__ = ["NiRuaMcpServer", "DEFAULT_RESULT_SIZE_HINT_CHARS", "DELIVERY_FAILURE_REASONS", "MCP_PATH",
           "PostExecutionDeliveryError", "SUPPORTED_PROTOCOL_VERSIONS", "TOP_LEVEL_COMBINATORS",
           "encode_mcp_result", "mcp_tool_catalog", "SCHEMA_DEPTH_MEASURE", "FOLDED_SCHEMA_NOTE", "json_depth",
           "fold_schema_depth", "fold_catalog_schemas", "portable_input_schema", "safe_present_tool_result",
           "scrub_host_paths", "tool_bridge_error_response"]

MCP_PATH = "/mcp"
# Newest first. Both pinned clients (Claude Code 2.1.321, OpenCode 1.19.5) offer 2025-11-25.
SUPPORTED_PROTOCOL_VERSIONS = ("2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05")
# Claude Code replaces any text result longer than 50,000 characters by a pointer to a private file unless the tool
# publishes _meta["anthropic/maxResultSizeChars"] (spike S2); 10,000,000 characters covers every registry result.
DEFAULT_RESULT_SIZE_HINT_CHARS = 10_000_000
RESULT_SIZE_META_KEY = "anthropic/maxResultSizeChars"
NATIVE_CALL_ID_META_KEY = "claudecode/toolUseId"
# The client's own session key of the call (OpenCode 2.2.7 sends it with every ``tools/call``; it equals the
# ``x-opencode-session-id`` header of that session's model requests). Claude Code and Codex send none under this key.
NATIVE_THREAD_ID_META_KEY = "ai.opencode/sessionID"
DELIVERY_UNCERTAIN = "tool_reply_delivery_uncertain"
ENCODING_FAILED = "tool_reply_encoding_failed"
DELIVERY_FAILURE_REASONS = (DELIVERY_UNCERTAIN, ENCODING_FAILED)
LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost"})
# Largest unread request body drained before a rejection is answered (bytes).
DISCARD_LIMIT_BYTES = 1024 * 1024
_MEDIA = {"inputImage": ("imageUrl", "image", "invalid_tool_image_url"),
          "inputAudio": ("audioUrl", "audio", "invalid_tool_audio_url")}
# data:<type>/<subtype>[;param=value...];base64,<payload>
_DATA_URL = re.compile(r"data:([A-Za-z0-9!#$&^_.+-]+/[A-Za-z0-9!#$&^_.+-]+)((?:;[^;,]*)*);base64,(.*)", re.DOTALL)
_DOMAIN_VALUE_ERROR_RE = re.compile(r"\A([a-z][a-z0-9_]{1,95})(?::|\Z)")
_HOST_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_:])(?:/(?:Users|home|tmp|var|private|opt|etc|mnt|root|proc|sys|dev)(?:/[^\s\"',;)\]]*)?)"
)


class PostExecutionDeliveryError(ValueError):
    """Raised when a tool call executed, so post-execution presentation failure must never claim no motion."""


def scrub_host_paths(value):
    """Strip host filesystem paths from model-visible error fields without touching non-path tokens."""
    if isinstance(value, str):
        return _HOST_PATH_RE.sub("<host-path>", _scrub_host_path_str(value))
    if isinstance(value, dict):
        return {key: scrub_host_paths(item) for key, item in value.items()}
    if isinstance(value, list):
        return [scrub_host_paths(item) for item in value]
    return value


def _contains_string_value(value, target: str) -> bool:
    if isinstance(value, str):
        return value == target
    if isinstance(value, dict):
        return any(_contains_string_value(item, target) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_string_value(item, target) for item in value)
    return False


def _resolve_bridge_error_type(error) -> str:
    return (
        getattr(error, "remote_type", None)
        or getattr(error, "error_type", None)
        or _extract_wire_error_type(error)
        or type(error).__name__
    )


def _is_caller_contract_error(error, arguments=None):
    _, validation_error_cls = _jsonschema()
    if validation_error_cls and isinstance(error, validation_error_cls):
        return True
    if isinstance(error, ToolInputError):
        return True
    category = getattr(error, "category", None)
    if category is not None:
        return category == "input_error" and getattr(error, "code", None) not in INFRASTRUCTURE_DOMAIN_CODES
    if isinstance(error, ValueError):
        stripped = str(error).strip()
        is_infra, _ = _is_infrastructure_message(stripped)
        if is_infra or extract_domain_code(stripped) == "execution_status_unknown":
            return False
        return _is_known_domain_value_message(stripped)
    if isinstance(error, KeyError) and len(error.args) == 1 and isinstance(error.args[0], str):
        return _contains_string_value(arguments, error.args[0])
    if type(error) is RuntimeError:
        head, sep, tail = str(error).partition(":")
        if sep and head.strip() == "ValueError":
            is_infra, _ = _is_infrastructure_message(tail)
            return not is_infra and _is_known_domain_value_message(tail)
    return False


def _record_internal_tool_error(store, *, tool, call_id, error, errors_sink=None):
    err_type = _resolve_bridge_error_type(error)
    trace_text = "".join(traceback.format_exception(type(error), error, error.__traceback__))
    if errors_sink is not None:
        errors_sink.append(("tool_internal_error", f"{tool}:{call_id}:{err_type}"))
    if store is None:
        return
    try:
        ref = store.put_artifact(trace_text.encode("utf-8", "replace"), "tool_internal_traceback", "text/plain",
                                 "observer")
        store.event(
            "agent.tool_internal_error",
            dict(
                tool=tool,
                call_id=call_id,
                error_type=err_type,
                remote_type=getattr(error, "remote_type", None),
                category=getattr(error, "category", None) or "internal_runtime_error",
                traceback_ref=ref,
            ),
            dict(tool_call_id=call_id) if call_id else None,
            source="agent_backend",
        )
    except Exception as log_error:  # noqa: BLE001 - diagnostic logging must never mask the tool error
        if errors_sink is not None:
            errors_sink.append(("tool_internal_error_log", repr(log_error)))


def tool_bridge_error_response(error, *, tool=None, arguments=None, store=None, call_id=None, errors_sink=None):
    """Convert a tool-bridge exception into a sanitized wire envelope without masking internal code bugs."""
    if _is_caller_contract_error(error, arguments=arguments):
        return scrub_host_paths(error_response(error, tool=tool, arguments=arguments))
    _record_internal_tool_error(store, tool=tool, call_id=call_id, error=error, errors_sink=errors_sink)
    category = getattr(error, "category", None)
    err_type = _resolve_bridge_error_type(error)
    raw_text = str(error)
    lower_msg = raw_text.lower()
    is_infra_msg, infra_code = _is_infrastructure_message(raw_text)
    if (
        isinstance(error, TimeoutError)
        or category == "unknown_execution"
        or "status unknown" in lower_msg
        or extract_domain_code(raw_text) == "execution_status_unknown"
    ):
        unknown_code = (getattr(error, "code", None) if category == "unknown_execution" else None) or "execution_status_unknown"
        payload = dict(
            error=scrub_host_paths(raw_text) or f"{unknown_code}:{err_type}",
            code=unknown_code,
            error_category="unknown_execution",
            error_type=err_type,
            retryable=False,
            field_path="",
            actual=None,
            suggestion=("Physical execution status is unknown. Do NOT replay or modify arguments to retry; inspect "
                        "robot_action_status / robot_observe first."),
            minimal_example=None,
            outcome_semantics="Physical execution outcome is UNKNOWN; motion may have occurred or still be stopping.",
        )
    elif (
        category == "infrastructure_error"
        or is_infra_msg
        or err_type in INFRASTRUCTURE_TYPE_NAMES
        or err_type == "StoreClosed"
        or isinstance(error, OSError)
        or (type(error) is RuntimeError and _DOMAIN_VALUE_ERROR_RE.match(raw_text.strip()) is not None)
    ):
        domain_match = _DOMAIN_VALUE_ERROR_RE.match(raw_text.strip())
        code = (
            (getattr(error, "code", None) if category == "infrastructure_error" else None)
            or infra_code
            or (domain_match.group(1) if domain_match else None)
            or _camel_to_snake(err_type)
        )
        err_text = scrub_host_paths(raw_text) or f"{code}:{err_type}"
        suggestion = (
            _IK_SERVICE_UNREACHABLE_SUGGESTION
            if code == "ik_service_unreachable"
            else "Backend/infrastructure fault before motion; changing tool arguments will not fix this fault."
        )
        payload = dict(
            error=err_text,
            code=code,
            error_category="infrastructure_error",
            error_type=err_type,
            retryable=False,
            field_path="",
            actual=None,
            suggestion=suggestion,
            minimal_example=None,
            outcome_semantics="Backend/infrastructure fault before motion; tool arguments are NOT at fault.",
        )
    else:
        err_text = (
            f"internal_runtime_error: KeyError: {scrub_host_paths(raw_text)}"
            if isinstance(error, KeyError)
            else (scrub_host_paths(raw_text) or f"internal_runtime_error:{err_type}")
        )
        payload = dict(
            error=err_text,
            code="internal_runtime_error",
            error_category="internal_runtime_error",
            error_type=err_type,
            retryable=False,
            field_path="",
            actual=None,
            suggestion=("Do NOT rewrite arguments to satisfy schema—your arguments passed schema validation and hit "
                        "an internal runtime bug. Report or bypass the broken path."),
            minimal_example=None,
            outcome_semantics=(
                "Internal software bug in tool implementation; no motion or task outcome is inferred unless "
                "execution_uncertain is set."
            ),
        )
    if tool:
        payload["tool"] = tool
    response = ToolOutput(payload).response()
    response.success = False
    return wire_response(response)


def safe_present_tool_result(presentation, result):
    """Present an already-sanitized error result, falling back to path-scrubbed copy if Store is unavailable."""
    try:
        return scrub_host_paths(presentation.present(result))
    except Exception:  # noqa: BLE001
        return scrub_host_paths(result)


def _server_version():
    """Version of the installed distribution, for ``serverInfo.version``."""
    try:
        return package_version("ni-rua-harness")
    except PackageNotFoundError:
        return "unknown"


def _media_part(item, kind):
    field_name, mcp_type, error_code = _MEDIA[kind]
    match = _DATA_URL.fullmatch(item[field_name])
    if match is None:
        raise ValueError(error_code)
    return {"type": mcp_type, "data": match.group(3), "mimeType": match.group(1)}


def _has_executed_contract_error(content):
    """检查首项文本块是否携带已执行物理动作的 ``output_contract_error`` 标记，防止 MCP 客户端将其误判为可重试错误。"""
    if not content or content[0].get("type") != "text":
        return False
    text = content[0].get("text")
    if not isinstance(text, str) or not text.startswith("{"):
        return False
    try:
        payload = json.loads(text)
    except ValueError:
        return False
    error = payload.get("output_contract_error") if isinstance(payload, dict) else None
    return isinstance(error, dict) and error.get("physical_action_executed") is True


def encode_mcp_result(response):
    """Encode one tool wire envelope as an MCP ``CallToolResult`` (pure; never parses text for facts).

    response: ``{"success": bool, "contentItems": [...]}`` as returned by ``Harness.handle_tool`` /
    ``core.tool_errors.error_response`` after presentation. ``success`` is the response-contract flag, not physical
    success. Items map in order (the relay's trim placeholders rely on "text then its images"):

    * ``{"type": "inputText", "text": t}`` -> ``{"type": "text", "text": t}``
    * ``{"type": "inputImage", "imageUrl": "data:<mime>;base64,<b64>"}`` -> ``{"type": "image", "data": <b64>,
      "mimeType": <mime>}``; the base64 text is passed through byte-identically (no decoding, resizing or
      re-encoding).
    * ``{"type": "inputAudio", "audioUrl": "data:<mime>;base64,<b64>"}`` -> ``{"type": "audio", ...}`` likewise.

    ``isError`` is ``not success`` only for results without image/audio items and without a leading
    ``output_contract_error`` whose ``physical_action_executed`` is True: both clients collapse an ``isError``
    result to its joined text and OpenCode drops its images, and an already-executed action must not look like a
    safe-to-replay failure (its error stays readable in the leading typed text).

    Raises ``ValueError``: ``invalid_tool_result`` (not an envelope), ``invalid_native_content_item`` (an item outside
    the text/image/audio union), ``invalid_tool_image_url`` / ``invalid_tool_audio_url`` (anything but a base64
    ``data:`` URL; such a URL is never forwarded).

    Example: ``encode_mcp_result({"success": False, "contentItems": [{"type": "inputText", "text": "e"}]})`` ->
    ``{"content": [{"type": "text", "text": "e"}], "isError": True}``.
    """
    if (not isinstance(response, dict) or type(response.get("success")) is not bool
            or not isinstance(response.get("contentItems"), list)):
        raise ValueError("invalid_tool_result")
    content, media = [], False
    for item in response["contentItems"]:
        try:
            validate_content_part(item)
        except (InvalidContentItem, TypeError, ValueError, OverflowError):
            raise ValueError("invalid_native_content_item") from None
        if not isinstance(item, dict):  # e.g. an unserialized StructuredText
            raise ValueError("invalid_native_content_item")
        if item["type"] == "inputText":
            content.append({"type": "text", "text": item["text"]})
        else:
            content.append(_media_part(item, item["type"]))
            media = True
    executed_contract_error = _has_executed_contract_error(content)
    return {"content": content, "isError": (not response["success"]) and not media and not executed_contract_error}


# Keywords an Anthropic tool input schema does not accept at its top level (Claude Code 2.1.321 drops the tool, OpenCode on
# anthropic_messages forwards it to a provider that rejects it).
TOP_LEVEL_COMBINATORS = ("anyOf", "oneOf", "allOf", "if", "then", "else")

_UNION_COMMON_CONSTRAINT_KEYS = (
    "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
    "minLength", "maxLength", "pattern",
    "items", "prefixItems", "minItems", "maxItems", "uniqueItems",
    "properties", "required", "additionalProperties", "default",
)


def _merge_union_property(nodes):
    """合并顶层 ``oneOf``/``anyOf`` 多分支下的同名属性：完全一致直接复用；同类型合并 ``enum``/``const``
    并集且仅保留全分支一致的约束（避免把单分支长度/范围限制强加给其他分支）；异类型保留属性内 ``anyOf``。
    """
    if len(nodes) == 1 or all(node == nodes[0] for node in nodes[1:]):
        return deepcopy(nodes[0])
    types = [node.get("type") for node in nodes]
    descriptions = []
    for node in nodes:
        desc = node.get("description")
        if isinstance(desc, str) and desc and desc not in descriptions:
            descriptions.append(desc)
    if types[0] is not None and all(kind == types[0] for kind in types[1:]):
        merged = {"type": deepcopy(types[0])}
        if descriptions:
            merged["description"] = descriptions[0] if len(descriptions) == 1 else " ".join(descriptions)
        if all("enum" in node or "const" in node for node in nodes):
            values = []
            for node in nodes:
                branch_vals = [node["const"]] if "const" in node else list(node["enum"])
                for val in branch_vals:
                    if val not in values:
                        values.append(deepcopy(val))
            merged["enum"] = values
        for key in _UNION_COMMON_CONSTRAINT_KEYS:
            if all(key in node and node[key] == nodes[0][key] for node in nodes):
                merged[key] = deepcopy(nodes[0][key])
        return merged
    merged = {"anyOf": deepcopy(nodes)}
    if len(descriptions) == 1:
        merged["description"] = descriptions[0]
    return merged


def portable_input_schema(schema):
    """Copy of ``schema`` without top-level ``anyOf``/``oneOf``/``allOf``/``if``/``then``/``else``.

    When ``schema`` has no top-level ``properties`` and declares an object union via top-level ``oneOf``/``anyOf``
    (e.g. ``vision_review``), promotes the union of branch ``properties`` (merging ``enum``/``const`` values and
    keeping only shared constraints) and the intersection of branch ``required`` so MCP hosts never see an empty
    object schema. Nested combinators stay untouched; full branch rules remain enforced by the registry.
    """
    result = {key: deepcopy(value) for key, value in schema.items() if key not in TOP_LEVEL_COMBINATORS}
    if result.get("properties"):
        return result
    branches = schema.get("oneOf") if isinstance(schema.get("oneOf"), list) else schema.get("anyOf")
    if not (isinstance(branches, list) and branches
            and all(isinstance(branch, dict) and branch.get("type") in (None, "object")
                    and isinstance(branch.get("properties"), dict) for branch in branches)
            and any(branch["properties"] for branch in branches)):
        return result
    prop_names = []
    for branch in branches:
        for name in branch["properties"]:
            if name not in prop_names:
                prop_names.append(name)
    result.setdefault("type", "object")
    result["properties"] = {
        name: _merge_union_property([branch["properties"][name] for branch in branches if name in branch["properties"]])
        for name in prop_names
    }
    first_req = branches[0].get("required") or ()
    other_reqs = [set(branch.get("required") or ()) for branch in branches[1:]]
    common_req = [name for name in first_req if all(name in reqs for reqs in other_reqs)]
    if common_req:
        result["required"] = common_req
    return result


# ``UpstreamConfig.tool_schema_max_depth``: the depth measure (recorded in provenance) and the folding note.
SCHEMA_DEPTH_MEASURE = "json_nonempty_container_depth_v1"
FOLDED_SCHEMA_NOTE = "JSON schema of this value (not expanded: the provider limits tool-schema depth): "
# Keywords whose value holds sub-schemas: maps of schemas (child at level + 2), lists of schemas (level + 2) and single
# schemas (level + 1). Every other keyword value is data (enum, const, default, examples, required, ...).
_SCHEMA_MAPS = frozenset({"properties", "patternProperties", "$defs", "definitions", "dependentSchemas"})
_SCHEMA_LISTS = frozenset({"anyOf", "oneOf", "allOf", "prefixItems", "items"})
_SCHEMA_SINGLE = frozenset({"items", "additionalProperties", "additionalItems", "not", "if", "then", "else",
                            "contains", "propertyNames", "unevaluatedProperties", "unevaluatedItems"})


def json_depth(value):
    """Nesting depth of a JSON value counting every non-empty object and array (a scalar, ``{}`` and ``[]`` are 0):
    ``{"type": "number"}`` is 1, ``{"type": "object", "properties": {"x": {"type": "number"}}}`` is 3. This is the
    measure tool-schema depth limits are stated in (``SCHEMA_DEPTH_MEASURE``)."""
    if isinstance(value, dict):
        return 1 + max(map(json_depth, value.values()), default=0) if value else 0
    if isinstance(value, list):
        return 1 + max(map(json_depth, value), default=0) if value else 0
    return 0


def _pointer(path, token):
    return path + "/" + str(token).replace("~", "~0").replace("/", "~1")


def _folded(node):
    """Permissive replacement of a schema node: its ``type`` when it is a string, and a description = the original
    description + ``FOLDED_SCHEMA_NOTE`` + the compact JSON of the node without its description."""
    rest = {key: value for key, value in node.items() if key != "description"}
    note = FOLDED_SCHEMA_NOTE + json.dumps(rest, ensure_ascii=False, separators=(",", ":"))
    original = node.get("description")
    folded = {"type": node["type"]} if isinstance(node.get("type"), str) else {}
    folded["description"] = (original.rstrip() + " " + note) if isinstance(original, str) and original.strip() else note
    return folded


def _fold(node, level, limit, path):
    """``(node, folded pointers)`` with every sub-schema of ``node`` (at container level ``level``) folded as deep
    as possible so that nothing exceeds ``limit``; the node itself is folded only when its own data still would."""
    if not isinstance(node, dict) or level - 1 + json_depth(node) <= limit:
        return node, []
    result, pointers = {}, []
    for key, value in node.items():
        if key in _SCHEMA_MAPS and isinstance(value, dict):
            children = {}
            for name, child in value.items():
                children[name], folded = _fold(child, level + 2, limit, _pointer(_pointer(path, key), name))
                pointers += folded
            result[key] = children
        elif key in _SCHEMA_LISTS and isinstance(value, list):
            children = []
            for index, child in enumerate(value):
                folded_child, folded = _fold(child, level + 2, limit, _pointer(_pointer(path, key), index))
                children.append(folded_child)
                pointers += folded
            result[key] = children
        elif key in _SCHEMA_SINGLE and isinstance(value, dict):
            result[key], folded = _fold(value, level + 1, limit, _pointer(path, key))
            pointers += folded
        else:
            result[key] = value
    if level - 1 + json_depth(result) <= limit:
        return result, pointers
    return _folded(node), [path]


def fold_schema_depth(schema, max_depth):
    """``(schema, pointers)``: ``schema`` (a tool input schema, the root at depth 1) with every sub-schema that would
    make ``json_depth`` exceed ``max_depth`` folded into a permissive node (``_folded``), deepest first, so as little
    as possible is folded; ``pointers`` lists the JSON pointers (RFC 6901) of the folded nodes in schema order (``""``
    = the whole schema, folded to ``{"type": "object", "description": ...}`` only when even that is needed).
    ``max_depth``: int >= 3 (the root object, its ``properties`` map and a folded property). The input is not
    mutated; an unchanged schema is returned as is."""
    if type(max_depth) is not int or max_depth < 3:
        raise ValueError("invalid_tool_schema_max_depth")
    return _fold(schema, 1, max_depth, "")


def fold_catalog_schemas(catalog, max_depth):
    """``(catalog, folded)`` for ``UpstreamConfig.tool_schema_max_depth``: every ``inputSchema`` of an
    ``mcp_tool_catalog`` result through ``fold_schema_depth``; ``folded`` = ``{tool name: [pointer, ...]}`` of the
    tools that changed (operator provenance). ``max_depth=None`` returns the catalog unchanged and ``{}``."""
    if max_depth is None:
        return catalog, {}
    result, folded = [], {}
    for tool in catalog:
        schema, pointers = fold_schema_depth(tool["inputSchema"], max_depth)
        if pointers:
            folded[tool["name"]] = pointers
            tool = dict(tool, inputSchema=schema)
        result.append(tool)
    return result, folded


def mcp_tool_catalog(tools, presentation, *, parallel_hints=None, result_size_hint_chars=None):
    """Build the ``tools/list`` catalog from the registry tool definitions Codex receives.

    tools: ``ToolRegistry.catalog(runtime, role="actor")`` (what ``Workbench.actor_tools()`` passes to
    ``session.start``); each definition needs ``name``, ``description``, ``inputSchema``. ``deferLoading`` and
    ``type`` are dropped: MCP backends publish every tool eagerly (design §3.8).
    presentation: the session's ``ActorPresentation``. ``description`` goes through ``present_text`` and
    ``inputSchema`` through ``present_data``, exactly as Codex ``dynamicTools`` are projected
    (``wire.openai_responses.present_responses_request``), then loses its top-level ``anyOf``/``oneOf``/``allOf``/
    ``if``/``then``/``else`` (``portable_input_schema``; harness difference, the rules stay in the description and
    in the registry's runtime validation).
    parallel_hints: ``{tool name: ToolSpec.parallel as published by ToolRegistry.manifest()}``, i.e. ``True``,
    ``False`` or ``"request_dependent"``; build it with
    ``{t["name"]: t["parallel"] for t in registry.manifest(runtime)["tools"]}``. ``annotations.readOnlyHint`` is
    ``True`` only when the hint is exactly ``True`` and ``False`` otherwise (missing, ``False``, request-dependent).
    Claude Code runs consecutive read-only calls concurrently and everything else serially; the host
    ``OrderedBarrier`` stays the ordering authority. Default ``None``: every tool ``readOnlyHint: false``.
    result_size_hint_chars: ``_meta["anthropic/maxResultSizeChars"]`` published on every tool, in characters of
    text result; positive int, default ``DEFAULT_RESULT_SIZE_HINT_CHARS`` (10,000,000). Mandatory for Claude Code,
    which otherwise rewrites text results over 50,000 characters to a private file pointer.

    Returns a new list of ``{name, description, inputSchema, annotations, _meta}``; raises ``ValueError``
    ``invalid_mcp_tool_definition`` / ``duplicate_mcp_tool:<name>`` / ``invalid_result_size_hint_chars``.
    """
    size = DEFAULT_RESULT_SIZE_HINT_CHARS if result_size_hint_chars is None else result_size_hint_chars
    if type(size) is not int or size <= 0:
        raise ValueError("invalid_result_size_hint_chars")
    hints = dict(parallel_hints or {})
    catalog, names = [], set()
    for tool in tools:
        if (not isinstance(tool, dict) or not isinstance(tool.get("name"), str) or not tool["name"]
                or not isinstance(tool.get("description"), str) or not isinstance(tool.get("inputSchema"), dict)):
            raise ValueError("invalid_mcp_tool_definition")
        if tool["name"] in names:
            raise ValueError("duplicate_mcp_tool:" + tool["name"])
        names.add(tool["name"])
        from ...core.schema_guide import strip_guided_schema_descriptions

        stripped_schema = strip_guided_schema_descriptions(tool["inputSchema"], tool["description"])
        catalog.append({"name": tool["name"],
                        "description": presentation.present_text(tool["description"]),
                        "inputSchema": portable_input_schema(presentation.present_data(stripped_schema)),
                        "annotations": {"readOnlyHint": hints.get(tool["name"]) is True},
                        "_meta": {RESULT_SIZE_META_KEY: size}})
    return catalog


def _json_bytes(message):
    return json.dumps(message, ensure_ascii=False, allow_nan=False, separators=(",", ":")).encode("utf-8")


def _rpc_error(request_id, code, message):
    return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}


def _host_name(value):
    """``(host, port)`` of a ``Host`` header value (``"127.0.0.1:8080"`` -> ``("127.0.0.1", 8080)``);
    ``(None, None)`` when unparseable."""
    try:
        parts = urlsplit(value if "//" in value else "//" + value)
        return parts.hostname, parts.port
    except ValueError:
        return None, None


@dataclass(eq=False)
class _Call:
    """One in-flight ``tools/call``. ``state``: running -> answered | abandoned | failed, and ``dispatched`` (set
    once ``call_tool`` is entered), both changed only under ``lock``; ``lock`` also serializes writes."""
    key: object
    rpc_id: object
    rpc_id_key: str
    session_id: str | None
    call_id: str
    progress_token: object
    sse: bool
    connection: object
    wfile: object
    next_keepalive: float = 0.0
    progress: int = 0
    state: str = "running"
    dispatched: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock)


def _client_session_id(headers):
    """提取合法的 ``Mcp-Session-Id`` 请求头用于按客户端隔离 in-flight 映射，缺失或非法返回 None。"""
    raw = headers.get("Mcp-Session-Id") if headers is not None else None
    if not isinstance(raw, str):
        return None
    value = raw.strip()
    if not value or len(value) > 256 or any(ord(ch) < 0x21 or ord(ch) > 0x7E for ch in value):
        return None
    return value


def _accepts_keyword(fn, keyword):
    """检查可调用对象是否接受指定关键字参数或 ``**kwargs``，以兼容不接收该参数的旧测试桩。"""
    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):
        return False
    if keyword in params:
        return params[keyword].kind in (
            inspect.Parameter.POSITIONAL_OR_KEYWORD,
            inspect.Parameter.KEYWORD_ONLY,
        )
    return any(param.kind is inspect.Parameter.VAR_KEYWORD for param in params.values())


class NiRuaMcpServer:
    """Loopback streamable-HTTP MCP endpoint (``POST http://127.0.0.1:<ephemeral>/mcp``) for registry tools.

    Keyword arguments (all required unless a default is shown):

    tools: registry tool definitions (``ToolRegistry.catalog(runtime, role="actor")``), see ``mcp_tool_catalog``.
    presentation: the session's ``ActorPresentation``; scrubs the catalog once at construction and renders the
        ``error_response`` of a ``call_tool`` exception.
    call_tool: ``call_tool(name, arguments, *, call_id, native_call_id, rpc_id, native_thread_id) -> {"success",
        "contentItems"}``, called on the request's handler thread; it blocks until the tool finished (normally
        ``ToolDispatcher.submit(params).result()`` -> ``Harness.handle_tool``, then the presentation scrub).
        ``call_id`` is the host call id: ``native_call_id`` when the client sent one
        (``params._meta["claudecode/toolUseId"]``, Claude Code) else a fresh ``new_id("call")`` (OpenCode); it is
        also the logged ``params._meta.item_id``. ``rpc_id`` is the JSON-RPC request id. ``native_thread_id`` is
        the client's own session key (``params._meta["ai.opencode/sessionID"]``, a non-empty string of at most 256
        characters) or None when the client sent none; it is self-reported, used only to correlate the call with
        the model step it belongs to (``agents.dispatch.EmittedOrder``), never for admission. Handler threads of
        concurrently sent calls enter ``call_tool`` in no defined order. An exception is answered as
        ``core.tool_errors.error_response`` (as the Codex transport does).
    raw_log: ``raw_log(direction, message)`` with direction ``"received"``/``"sent"`` (``Store.raw_agent`` with
        ``channel="mcp"``). A failure to log a received ``tools/call`` refuses the call before dispatch.
    on_delivery_failure: ``on_delivery_failure(call_id, reason)``, called at most once per call with a reason from
        ``DELIVERY_FAILURE_REASONS``; the session invalidates ``runtime/image_context_epoch`` and emits
        ``agent.tool_reply_unknown``. Must not raise. Failures of ``raw_log``/``on_delivery_failure`` never stop
        the server; they are kept in ``errors`` as ``(call id or "raw_log", repr(error))``.
    parallel_hints: per-tool ``ToolSpec.parallel`` from ``ToolRegistry.manifest()``; drives
        ``annotations.readOnlyHint`` (see ``mcp_tool_catalog``). Default None: no tool is read-only.
    server_name: ``serverInfo.name`` and the name agents prefix tools with (``mcp__ni_rua__…``, ``ni_rua_…``);
        default ``"ni_rua"``.
    keepalive_s: seconds between SSE keepalives while a call runs, default 5.0: a ``notifications/progress`` for the
        call's ``progressToken`` (progress notifications, not SSE comments, reset Claude Code's 300 s MCP idle timer;
        OpenCode 2.2.7 does NOT extend its per-call ``timeout.execution`` on progress, so only that setting bounds an
        OpenCode call: spike 21, a 35 s call was cancelled at 20 s despite 5 s progress frames) or an SSE comment
        when the client sent no token. Must be > 0.
    peer_check_s: seconds between peer-close checks of in-flight calls (a closed socket abandons the call),
        default 0.25; must be > 0. The keepalive thread wakes at ``min(keepalive_s, peer_check_s)``.
    max_request_bytes: largest accepted request body in bytes (``Content-Length``), default 16 MiB; larger -> 413.
    io_timeout_s: socket timeout in seconds for reading a request and writing a response, default 30.0.
    result_size_hint_chars: ``_meta["anthropic/maxResultSizeChars"]`` on every tool, default 10,000,000.
    schema_max_depth: ``UpstreamConfig.tool_schema_max_depth`` (int >= 3) or None (default: schemas unchanged). The
        published ``inputSchema`` of every tool is folded to that ``json_depth`` (``fold_catalog_schemas``); the
        folded tools and pointers are ``folded_schemas``. Only the published catalog changes: ``call_tool`` receives
        the arguments unchanged and the registry validates them against the full schema.
    port: loopback TCP port to bind, int 0..65535; default 0 = an ephemeral port chosen by the OS (every host-mode
        backend). A fixed port is for an agent configured with a static URL (``agent.external.mcp_port``); a port
        another process holds makes ``start()`` raise the bind ``OSError`` (errno ``EADDRINUSE``). The socket is bound
        with ``SO_REUSEADDR`` (``allow_reuse_address``, as ``HTTPServer`` always did), so a restarted gateway rebinds
        a fixed port at once while its previous connections are still in TIME_WAIT.
    require_bearer: True (default; every host-mode backend) = the per-run bearer below is required on every request.
        False = no ``Authorization`` check and ``headers == {}``, for an agent whose static configuration cannot know
        a per-run token (external backend); the loopback bind and the ``Host`` check stay, and a request carrying an
        ``Origin`` other than ``http://127.0.0.1:<port>`` / ``http://localhost:<port>`` is refused with 403 (a web
        page's cross-site request; the console API's rule). Any local process can still call (documented residual).
    instructions: str or None (default). A non-empty text is returned as ``initialize.instructions``, which MCP
        clients show the model; None = no ``instructions`` key (every host-mode backend: their instructions have one
        channel). The owner passes model-visible text only after its presentation scrub.
    on_initialize: ``on_initialize(params)`` with the ``initialize`` request's ``params`` object (``clientInfo``,
        ``protocolVersion``; self-reported by the client), called after the response was written; None (default) =
        not called. Must not raise; a failure is kept in ``errors``.

    After ``start()``: ``url`` (``http://127.0.0.1:<port>/mcp``) and ``headers`` (``{"Authorization": "Bearer
    <per-run token>"}``) configure the agent. Every request must carry that bearer (else 401; unless
    ``require_bearer=False``) and a loopback ``Host`` (``127.0.0.1`` or ``localhost`` with this port, else 403). These
    are the only access controls (loopback bind, per-run bearer, Host check, request size cap): the agent's own shell
    can read the bearer from its environment, a documented residual (design §12.3, user decision: document only, no
    further hardening).
    """

    def __init__(self, *, tools, presentation, call_tool, raw_log, on_delivery_failure, parallel_hints=None,
                 server_name="ni_rua", keepalive_s=5.0, peer_check_s=0.25, max_request_bytes=16 * 1024 * 1024,
                 io_timeout_s=30.0, result_size_hint_chars=None, schema_max_depth=None, port=0, require_bearer=True,
                 instructions=None, on_initialize=None, on_call_settled=None):
        for name, value in (("keepalive_s", keepalive_s), ("peer_check_s", peer_check_s),
                            ("io_timeout_s", io_timeout_s)):
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not value > 0:
                raise ValueError("invalid_" + name)
        if type(max_request_bytes) is not int or max_request_bytes <= 0:
            raise ValueError("invalid_max_request_bytes")
        if not isinstance(server_name, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,64}", server_name):
            raise ValueError("invalid_mcp_server_name")
        if type(port) is not int or not 0 <= port <= 65535:
            raise ValueError("invalid_mcp_port")
        if type(require_bearer) is not bool:
            raise ValueError("invalid_mcp_require_bearer")
        if instructions is not None and (not isinstance(instructions, str) or not instructions.strip()):
            raise ValueError("invalid_mcp_instructions")
        if on_initialize is not None and not callable(on_initialize):
            raise ValueError("invalid_mcp_on_initialize")
        if on_call_settled is not None and not callable(on_call_settled):
            raise ValueError("invalid_mcp_on_call_settled")
        self.port, self.require_bearer = port, require_bearer
        self.instructions, self.on_initialize, self.on_call_settled = instructions, on_initialize, on_call_settled
        self.server_name = server_name
        self.catalog, self.folded_schemas = fold_catalog_schemas(
            mcp_tool_catalog(tools, presentation, parallel_hints=parallel_hints,
                             result_size_hint_chars=result_size_hint_chars), schema_max_depth)
        self.presentation, self.call_tool, self.raw_log = presentation, call_tool, raw_log
        self.on_delivery_failure = on_delivery_failure
        self.keepalive_s, self.peer_check_s = float(keepalive_s), float(peer_check_s)
        self.max_request_bytes, self.io_timeout_s = max_request_bytes, float(io_timeout_s)
        if require_bearer:
            self.token = secrets.token_urlsafe(32)
            self.headers = {"Authorization": "Bearer " + self.token}
        else:
            self.token, self.headers = None, {}
        self.url = None
        self.server = self.thread = self.keepalive_thread = None
        self.closed = False
        self.lock = threading.Lock()
        self.inflight = {}
        self.errors = []
        self._stop = threading.Event()

    @property
    def read_only_tools(self):
        """Names published with ``readOnlyHint: true`` (provenance ``tool_transport.client_parallel_hint``)."""
        return sorted(tool["name"] for tool in self.catalog if tool["annotations"]["readOnlyHint"])

    # lifecycle -------------------------------------------------------------------------------------------------
    def start(self):
        if self.server is not None or self.closed:
            raise ValueError("mcp_server_already_started_or_closed")
        owner = self

        class Handler(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"
            timeout = owner.io_timeout_s
            body_consumed = False

            def log_message(self, *_):
                pass

            def do_GET(self):
                self.close_connection = True
                if self.admitted():
                    self.send_empty(405, {"Allow": "POST"})

            do_DELETE = do_GET

            def do_POST(self):
                self.close_connection = True  # one request per connection, as ModelRequestProxy
                if self.admitted():
                    owner._post(self)

            def admitted(self):
                """Path, loopback Host and bearer checks; sends the rejection and returns False."""
                if urlsplit(self.path).path != MCP_PATH:
                    self.send_empty(404)
                    return False
                host, port = _host_name(self.headers.get("Host") or "")
                if (host or "").lower() not in LOOPBACK_HOSTS or port != owner.server.server_port:
                    self.send_empty(403)
                    return False
                if not owner.require_bearer:
                    # No bearer (external backend): refuse a browser page's cross-site request, as the console API
                    # does (routes.py), so a web page cannot reach the robot tools; a client that sends no Origin
                    # (every MCP host) is not affected.
                    origin = self.headers.get("Origin")
                    if origin is not None and origin not in (f"http://127.0.0.1:{port}", f"http://localhost:{port}"):
                        self.send_empty(403)
                        return False
                    return True
                supplied = self.headers.get("Authorization") or ""
                if not hmac.compare_digest(supplied.encode("utf-8", "replace"),
                                           owner.headers["Authorization"].encode()):
                    self.send_empty(401, {"WWW-Authenticate": "Bearer"})
                    return False
                return True

            def discard_body(self):
                """Read and drop an unread request body of at most ``DISCARD_LIMIT_BYTES`` so that closing the
                socket after a rejection does not reset the connection before the client read the status line;
                a larger body is not read (the client may then see a reset instead of the status)."""
                if self.command != "POST" or self.body_consumed or self.headers.get("Transfer-Encoding"):
                    return
                self.body_consumed = True
                try:
                    remaining = int(self.headers.get("Content-Length", "0"))
                    if remaining > DISCARD_LIMIT_BYTES:
                        return
                    while remaining > 0:
                        chunk = self.rfile.read(min(remaining, 65536))
                        if not chunk:
                            break
                        remaining -= len(chunk)
                except (OSError, ValueError):
                    pass

            def send_empty(self, status, headers=None):
                self.discard_body()
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Length", "0")
                self.send_header("Connection", "close")
                self.end_headers()
                self.close_connection = True

            def send_json(self, status, message, headers=None):
                body = _json_bytes(message)
                self.send_response(status)
                for key, value in (headers or {}).items():
                    self.send_header(key, value)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.send_header("Connection", "close")
                self.end_headers()
                self.wfile.write(body)
                self.wfile.flush()
                self.close_connection = True

        class Server(ThreadingHTTPServer):
            allow_reuse_address = True   # HTTPServer's own value, stated: a fixed port rebinds at once after a restart
            daemon_threads = True
            block_on_close = False
            request_queue_size = 64

        self.server = Server(("127.0.0.1", self.port), Handler)
        self.url = f"http://127.0.0.1:{self.server.server_port}{MCP_PATH}"
        self.thread = threading.Thread(target=self.server.serve_forever, kwargs={"poll_interval": .05},
                                       name="ni-rua-mcp-http", daemon=True)
        self.thread.start()
        self.keepalive_thread = threading.Thread(target=self._keepalive_loop, name="ni-rua-mcp-keepalive",
                                                 daemon=True)
        self.keepalive_thread.start()
        return self

    def close(self):
        """Stop accepting requests; idempotent.

        A call still in flight can no longer be answered: it is abandoned (``on_delivery_failure`` with
        ``tool_reply_delivery_uncertain``) and its socket closed. Its handler thread keeps waiting for
        ``call_tool``; the session joins in-flight tool work through its ``ToolDispatcher`` after the robot closed.
        """
        with self.lock:
            if self.closed:
                return
            self.closed = True
            calls = list(self.inflight.values())
        self._stop.set()
        for call in calls:
            self._abandon(call)
        if self.server is not None:
            self.server.shutdown()
            self.server.server_close()
        for thread in (self.thread, self.keepalive_thread):
            if thread is not None and thread is not threading.current_thread():
                thread.join(2)

    def __enter__(self):
        return self.start()

    def __exit__(self, *_):
        self.close()

    # request handling ------------------------------------------------------------------------------------------
    def _log(self, direction, message):
        """Append one record to the raw log; returns False (recorded in ``errors``) when the sink failed."""
        try:
            self.raw_log(direction, message)
        except Exception as error:  # noqa: BLE001 - a logging failure must not stop the endpoint
            self.errors.append(("raw_log", repr(error)))
            return False
        return True

    def _post(self, handler):
        if self.closed:
            handler.send_empty(503)
            return
        if handler.headers.get("Transfer-Encoding"):
            handler.send_empty(411)
            return
        try:
            length = int(handler.headers.get("Content-Length", ""))
        except ValueError:
            handler.send_empty(411)
            return
        if length < 0:
            handler.send_empty(400)
            return
        if length > self.max_request_bytes:
            handler.send_empty(413)
            return
        try:
            raw = handler.rfile.read(length)
        except OSError:
            handler.close_connection = True
            return
        if len(raw) != length:
            handler.close_connection = True
            return
        handler.body_consumed = True
        try:
            message = json.loads(raw)
        except (UnicodeDecodeError, ValueError):
            self._reply(handler, 400, _rpc_error(None, -32700, "Parse error"))
            return
        if not isinstance(message, dict):
            # A batch or a bare value: one JSON-RPC message per request (protocol 2025-06-18 and later).
            self._reply(handler, 400, _rpc_error(None, -32600, "Invalid Request: one JSON-RPC message per request"))
            return
        method = message.get("method")
        if "id" not in message or (method is None and ("result" in message or "error" in message)):
            # Notification, or a client response to a server request (this server sends none).
            self._log("received", message)
            if method == "notifications/cancelled":
                self._cancelled(message.get("params"), session_id=_client_session_id(handler.headers))
            handler.send_empty(202)
            return
        request_id = message["id"]
        if not isinstance(method, str) or not isinstance(request_id, (str, int)) or isinstance(request_id, bool):
            self._log("received", message)
            self._reply(handler, 400, _rpc_error(None, -32600, "Invalid Request"))
            return
        if method == "tools/call":
            self._tools_call(handler, message)
            return
        self._log("received", message)
        params = message.get("params") if isinstance(message.get("params"), dict) else {}
        reply_headers = None
        if method == "initialize":
            offered = params.get("protocolVersion")
            result = {"protocolVersion": offered if offered in SUPPORTED_PROTOCOL_VERSIONS
                      else SUPPORTED_PROTOCOL_VERSIONS[0],
                      "capabilities": {"tools": {"listChanged": False}},
                      "serverInfo": {"name": self.server_name, "version": _server_version()}}
            if self.instructions is not None:
                result["instructions"] = self.instructions
            # 为每个客户端分配独立 Mcp-Session-Id 以便按客户端隔离 in-flight 请求与取消，同时保持跨重启无状态（不拒绝未知/缺失头）
            reply_headers = {"Mcp-Session-Id": new_id("mcp")}
        elif method == "ping":
            result = {}
        elif method == "tools/list":
            result = {"tools": deepcopy(self.catalog)}
        else:
            self._reply(handler, 200, _rpc_error(request_id, -32601, "Method not found"))
            return
        self._reply(handler, 200, {"jsonrpc": "2.0", "id": request_id, "result": result}, headers=reply_headers)
        if method == "initialize" and self.on_initialize is not None:
            try:
                self.on_initialize(params)
            except Exception as error:  # noqa: BLE001 - an observer callback must not stop the endpoint
                self.errors.append(("on_initialize", repr(error)))

    def _reply(self, handler, status, message, headers=None):
        """Send one JSON response and log it as sent once completely written."""
        try:
            handler.send_json(status, message, headers=headers)
        except OSError:
            return False
        self._log("sent", message)
        return True

    def _tools_call(self, handler, message):
        request_id = message["id"]
        params = message.get("params")
        name = params.get("name") if isinstance(params, dict) else None
        arguments = params.get("arguments", {}) if isinstance(params, dict) else None
        if not isinstance(name, str) or not name or not isinstance(arguments, dict):
            self._log("received", message)
            self._reply(handler, 200, _rpc_error(
                request_id, -32602, "Invalid params: tools/call needs a string name and an object arguments"))
            return
        meta = params.get("_meta") if isinstance(params.get("_meta"), dict) else {}
        native = meta.get(NATIVE_CALL_ID_META_KEY)
        native_call_id = native if isinstance(native, str) and 0 < len(native) <= 256 else None
        thread = meta.get(NATIVE_THREAD_ID_META_KEY)
        native_thread_id = thread if isinstance(thread, str) and 0 < len(thread) <= 256 else None
        call_id = native_call_id or new_id("call")
        logged = {**message, "params": {**params, "_meta": {**meta, "item_id": call_id}}}
        sse = "text/event-stream" in (handler.headers.get("Accept") or "").lower()
        session_id = _client_session_id(handler.headers)
        rpc_id_key = canonical(request_id)
        # 带 Mcp-Session-Id 时按 (session_id, rpc_id) 建键；无 bearer 多客户端模式且未带 session 头时按独立请求建键，避免跨客户端 id 或 _meta 冲突
        if session_id is not None:
            inflight_key = ("session", session_id, rpc_id_key)
        elif self.require_bearer:
            inflight_key = ("single", rpc_id_key)
        else:
            inflight_key = ("unscoped", new_id("inflight"))
        call = _Call(key=inflight_key, rpc_id=request_id, rpc_id_key=rpc_id_key, session_id=session_id,
                     call_id=call_id, progress_token=meta.get("progressToken"), sse=sse,
                     connection=handler.connection, wfile=handler.wfile)
        with self.lock:
            duplicate = call.key in self.inflight
            if not duplicate and not self.closed:
                self.inflight[call.key] = call
        if duplicate or self.closed:
            self._log("received", message)
            self._reply(handler, 200 if duplicate else 503, _rpc_error(
                request_id, -32600, "Invalid Request: duplicate in-flight request id" if duplicate
                else "Server closing"))
            return
        try:
            if not self._log("received", logged):  # no audit record, no dispatch
                self._reply(handler, 200, _rpc_error(request_id, -32603, "Internal error: tool call not recorded; "
                                                                         "it was not executed"))
                return
            if sse:
                with call.lock:
                    try:
                        handler.send_response(200)
                        handler.send_header("Content-Type", "text/event-stream")
                        handler.send_header("Cache-Control", "no-cache")
                        handler.send_header("Connection", "close")
                        handler.end_headers()
                        handler.wfile.flush()
                    except OSError:
                        if call.state == "running":
                            call.state = "failed"
                        return  # Peer already gone: nothing was dispatched, nothing to report.
                    call.next_keepalive = time.monotonic() + self.keepalive_s
            with call.lock:
                if call.state != "running":
                    return  # Cancelled or peer gone before dispatch: never executed, nothing to report.
                call.dispatched = True
            self._run_call(handler, call, name, arguments, native_call_id, native_thread_id)
        finally:
            with self.lock:
                if self.inflight.get(call.key) is call:
                    del self.inflight[call.key]

    def _run_call(self, handler, call, name, arguments, native_call_id, native_thread_id=None):
        encoding_failed = False
        delivered = False
        try:
            call_kwargs = dict(call_id=call.call_id, native_call_id=native_call_id,
                               rpc_id=call.rpc_id, native_thread_id=native_thread_id)
            if _accepts_keyword(self.call_tool, "is_abandoned"):
                call_kwargs["is_abandoned"] = lambda: call.state != "running"
            try:
                result = self.call_tool(name, arguments, **call_kwargs)
            except PostExecutionDeliveryError as error:
                encoding_failed = True
                code = str(error) if str(error).startswith(("invalid_tool_", "invalid_native_")) else "invalid_tool_result"
                response = _rpc_error(call.rpc_id, -32603, code + ": the tool call ran but its result could not be "
                                      "delivered; it may have taken effect, so check its status before any retry")
                body = _json_bytes(response)
            except Exception as error:  # noqa: BLE001 - answered like the Codex transport (RecordedClient._approval)
                raw_err = tool_bridge_error_response(error, tool=name, arguments=arguments,
                                                     store=getattr(self.presentation, "store", None), call_id=call.call_id,
                                                     errors_sink=self.errors)
                result = safe_present_tool_result(self.presentation, raw_err)
            if not encoding_failed:
                try:
                    response = {"jsonrpc": "2.0", "id": call.rpc_id, "result": encode_mcp_result(result)}
                    body = _json_bytes(response)
                except (ValueError, TypeError) as error:
                    encoding_failed = True
                    code = str(error) if str(error).startswith(("invalid_tool_", "invalid_native_")) else "invalid_tool_result"
                    response = _rpc_error(call.rpc_id, -32603, code + ": the tool call ran but its result could not be "
                                          "delivered; it may have taken effect, so check its status before any retry")
                    body = _json_bytes(response)
            with call.lock:
                if call.state != "running":
                    return  # Abandoned: the late result is discarded (delivery already reported uncertain).
                gone = self._peer_gone(call)
                if not gone:
                    try:
                        if call.sse:
                            call.wfile.write(b"event: message\ndata: " + body + b"\n\n")
                            call.wfile.flush()
                        else:
                            handler.send_response(200)
                            handler.send_header("Content-Type", "application/json")
                            handler.send_header("Content-Length", str(len(body)))
                            handler.send_header("Connection", "close")
                            handler.end_headers()
                            handler.wfile.write(body)
                            handler.wfile.flush()
                    except OSError:
                        gone = True
                if gone:
                    call.state = "failed"
                else:
                    call.state = "answered"
                    self._log("sent", response)
                    delivered = not encoding_failed
            if gone:
                self._report(call.call_id, DELIVERY_UNCERTAIN)
            elif encoding_failed:
                self._report(call.call_id, ENCODING_FAILED)
        finally:
            self._settle(call.call_id, delivered)

    # delivery tracking -----------------------------------------------------------------------------------------
    def _settle(self, call_id, delivered):
        """在单次 tools/call 完成写回或失败/放弃退出后结算状态（绝不抛异常）。"""
        callback = self.on_call_settled
        if callback is None:
            return
        try:
            callback(call_id, delivered)
        except Exception as error:  # noqa: BLE001 - 结算回调异常不得中断服务端
            self.errors.append((f"on_call_settled:{call_id}", repr(error)))

    def _report(self, call_id, reason):
        try:
            self.on_delivery_failure(call_id, reason)
        except Exception as error:  # noqa: BLE001 - must not stop the keepalive thread or the handler
            self.errors.append((call_id, repr(error)))

    def _abandon(self, call, *, blocking=True):
        """Mark a running call abandoned (late result discarded) and close its socket; report uncertain delivery
        only when ``call_tool`` was already entered (an undispatched call never runs). ``blocking=False`` (the
        shared keepalive thread) skips a call whose lock is held by a write in progress; returns whether the call
        was abandoned now."""
        if not call.lock.acquire(blocking=blocking):
            return False
        try:
            if call.state != "running":
                return False
            call.state = "abandoned"
            dispatched = call.dispatched
        finally:
            call.lock.release()
        with self.lock:
            if self.inflight.get(call.key) is call:
                del self.inflight[call.key]
        if dispatched:
            self._report(call.call_id, DELIVERY_UNCERTAIN)
        try:
            call.connection.shutdown(socket.SHUT_RDWR)
        except OSError:
            pass
        return True

    def _cancelled(self, params, *, session_id=None):
        """``notifications/cancelled``: abandon the call only while its response is not completely written.

        OpenCode's MCP client sends a cancel 6-30 ms after every answered call; that cancel finds no running call
        (the state turned ``answered`` under the call lock right after the final write) and changes nothing.
        """
        if not isinstance(params, dict) or "requestId" not in params:
            return
        try:
            rpc_id_key = canonical(params["requestId"])
        except (TypeError, ValueError):
            return
        with self.lock:
            if session_id is not None:
                call = self.inflight.get(("session", session_id, rpc_id_key))
            elif self.require_bearer:
                call = self.inflight.get(("single", rpc_id_key))
                if call is None:
                    matches = [c for c in self.inflight.values() if c.rpc_id_key == rpc_id_key]
                    call = matches[0] if len(matches) == 1 else None
            else:
                # 外部多客户端模式：未携带 Mcp-Session-Id 的取消通知绝不跨客户端掐断已隔离会话，仅在无歧义匹配唯一未隔离调用时生效
                unscoped = [c for c in self.inflight.values()
                            if c.session_id is None and c.rpc_id_key == rpc_id_key]
                call = unscoped[0] if len(unscoped) == 1 else None
        if call is not None:
            self._abandon(call)

    @staticmethod
    def _peer_gone(call):
        """True when the client closed (EOF) or reset the connection; never blocks.

        ``poll`` (not ``select``, which fails for descriptors >= 1024 in a busy gateway) with a zero timeout; only a
        readable socket is peeked, so pending request bytes are never consumed.
        """
        try:
            if hasattr(select, "poll"):
                poller = select.poll()
                poller.register(call.connection, select.POLLIN | select.POLLPRI)
                readable = bool(poller.poll(0))
            else:  # pragma: no cover - platforms without poll()
                readable = bool(select.select([call.connection], [], [], 0)[0])
            return readable and not call.connection.recv(1, socket.MSG_PEEK)
        except (OSError, ValueError):
            return True

    def _keepalive_loop(self):
        tick = min(self.keepalive_s, self.peer_check_s)
        while not self._stop.wait(tick):
            with self.lock:
                calls = list(self.inflight.values())
            for call in calls:
                if call.state != "running":
                    continue
                if self._peer_gone(call):
                    self._abandon(call, blocking=False)
                    continue
                if call.sse and call.next_keepalive and time.monotonic() >= call.next_keepalive:
                    self._keepalive(call)

    def _keepalive(self, call):
        # Never wait behind a call's final write (possibly megabytes): this thread serves every call.
        if not call.lock.acquire(blocking=False):
            return
        try:
            if call.state != "running":
                return
            if call.progress_token is not None:
                call.progress += 1
                note = {"jsonrpc": "2.0", "method": "notifications/progress",
                        "params": {"progressToken": call.progress_token, "progress": call.progress}}
                frame = b"event: message\ndata: " + _json_bytes(note) + b"\n\n"
            else:
                frame = b": keepalive\n\n"
            try:
                call.wfile.write(frame)
                call.wfile.flush()
            except OSError:
                failed = True
            else:
                failed = False
                call.next_keepalive = time.monotonic() + self.keepalive_s
        finally:
            call.lock.release()
        if failed:
            self._abandon(call)
