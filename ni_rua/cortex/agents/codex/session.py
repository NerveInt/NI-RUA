"""Codex agent backend: pinned Python SDK over a WS-to-stdio bridge to the run's private app-server."""

import asyncio
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
from importlib.metadata import version
import json
import os
from pathlib import Path
import select
import signal
import stat
import subprocess
import sys
import tempfile
import threading
import time
import tomllib
from urllib.parse import urlsplit

from codex_cli_bin import bundled_codex_path
from jsonschema import validate as validate_json
from openai_codex.client import CodexClient, CodexConfig
from openai_codex.errors import JsonRpcError

from ....core.contracts import canonical, digest, new_id, utc_now
from ....core.presentation.image_reuse import IMAGE_REUSE_POLICY
from ....core.storage.images import store_image_settings
from ...context.task_planning import decision_reminder_text
from ...relay.guard import DEFAULT_MAX_RETRIES, RETRY_POLICY
from ...relay.server import ModelRequestProxy
from ...session.account_control import ACCOUNT_BASE_URL, AccountControlRelay
from ...session.actor_source import snapshot_actor_source
from ...session.agent_sandbox import ACTOR_ISOLATION  # shared by every sandboxed backend (design R4.4)
from ...session.egress_cache import EgressProjectionCache
from ...session.pidfd import open_process, process_argv
from ...session.usage import RelayUsageTotal, account_usage, add_served_model, responses_counters
from ..backend import AppServerTransport
from ..base import AgentCapabilities, AgentSession, model_effort_choices
from ..dispatch import OrderedBarrier, is_parallel_call
from ..identity import write_codex_identity, write_identity
from ..isolation import UNIX_SOCKET_PATH_MAX
from ..model_listing import CodexModelListingMixin
from .bridge import build_bridge_command, proxy

_SESSION_DATA_DIR = Path(__file__).resolve().parents[2] / "session"
MODEL_COMPLETION_COUNTING = "delivered_usable_actor_helper_compact_responses_v2"
MODEL_PROXY_PROVIDER = "ni_rua_model_proxy"
DYNAMIC_TOOL_NAMESPACE = "ni_rua"
CODEX_SDK_VERSION = version("openai-codex")
CODEX_CLI_VERSION = version("openai-codex-cli-bin")
_CODEX_PROTOCOL = json.loads((_SESSION_DATA_DIR / "codex_responses.json").read_text())
CODEX_PROTOCOL_VERSION = _CODEX_PROTOCOL["build"]
NATIVE_RESPONSE_SCHEMAS = _CODEX_PROTOCOL["responses"]
DEFAULT_MODEL_VERBOSITY = "high"
VALID_MODEL_VERBOSITIES = ("low", "medium", "high")
# Information boundary (AGENTS.md "Context carries only necessary information", "Restricted Information Sources"):
# ``project_root_markers = []`` makes the actor directory the project root so runs below a git checkout never read host
# AGENTS.md files into actor context while still reading ``<actor>/AGENTS.md``.
BOUNDARY_CONFIG_LINES = ("project_root_markers = []",)
# Same boundary towards the provider (review 30, D1): git stops searching for a repository at GIT_CEILING_DIRECTORIES,
# so unsandboxed Codex finds no host repository above ``actor_dir`` to send in ``X-Codex-Turn-Metadata`` /
# ``client_metadata.workspaces``.
BOUNDARY_ENV_NAME = "GIT_CEILING_DIRECTORIES"


def native_tool_path(executable):
    """``PATH`` of the Codex process: ``os.defpath`` behind the pinned distribution's own tool directory
    (``<package>/codex-path``, holding ``rg``; the directory the native launcher prepends), when ``executable``
    (the resolved Codex binary) has one. Inside the sandbox ``AgentSandbox`` replaces this value."""
    from .sandbox import distribution_tool_directory
    tools = distribution_tool_directory(executable)
    return (str(tools) + os.pathsep if tools is not None else "") + os.defpath


def boundary_env(actor_dir):
    """``{"GIT_CEILING_DIRECTORIES": <directory above actor_dir>}`` for an unsandboxed Codex process whose working
    directory is ``actor_dir`` (absolute path)."""
    return {BOUNDARY_ENV_NAME: os.fspath(Path(actor_dir).parent)}


def require_codex_version():
    if CODEX_SDK_VERSION != CODEX_PROTOCOL_VERSION or CODEX_CLI_VERSION != CODEX_PROTOCOL_VERSION:
        raise RuntimeError(
            f"codex_version_mismatch: expected SDK and CLI {CODEX_PROTOCOL_VERSION}; "
            f"installed SDK={CODEX_SDK_VERSION}, CLI={CODEX_CLI_VERSION}. "
            "Install the repository requirements.lock.txt in the active Harness environment before starting or resuming."
        )


def write_direct_model_catalog(path, *, supports_search_tool=False):
    """保留固定模型元数据，统一使用直连工具模式并按通道支持情况设置 ``supports_search_tool``。"""
    catalog = json.loads((_SESSION_DATA_DIR / "codex_models.json").read_text())
    for model in catalog["models"]:
        model["tool_mode"] = "direct"
        model["supports_search_tool"] = bool(supports_search_tool)
    path = Path(path)
    path.write_text(json.dumps(catalog, ensure_ascii=False) + "\n")
    return path


# Codex keeps 1.2 x the model's tool-output token limit of every tool output it records, counting one token per
# started 4 UTF-8 bytes of a text item (measured on CLI 0.160.0; rule and evidence in
# core/presentation/reply_budget.py and docs/development/tool-output-contract.md).
TOOL_OUTPUT_SERIALIZATION_FACTOR = 1.2
TOOL_OUTPUT_BYTES_PER_TOKEN = 4


def codex_tool_output_token_limit(model):
    """``truncation_policy.limit`` (tokens) of ``model`` (catalogue slug) in the pinned model catalogue that
    ``write_direct_model_catalog`` hands to Codex, or None when the model is not listed or its policy is not in
    tokens. This catalogue entry is what Codex applies, so it is the one source of the Harness's budget."""
    catalog = json.loads((_SESSION_DATA_DIR / "codex_models.json").read_text())
    for entry in catalog["models"]:
        if entry.get("slug") == model:
            policy = entry.get("truncation_policy")
            if isinstance(policy, dict) and policy.get("mode") == "tokens" and type(policy.get("limit")) is int:
                return policy["limit"]
            return None
    return None


def codex_tool_output_budget_bytes(model):
    """UTF-8 bytes of text Codex keeps of one tool output for ``model`` (48,000 for a 10,000-token limit), or None
    when ``codex_tool_output_token_limit`` is None."""
    limit = codex_tool_output_token_limit(model)
    if limit is None:
        return None
    return int(limit * TOOL_OUTPUT_SERIALIZATION_FACTOR) * TOOL_OUTPUT_BYTES_PER_TOKEN


def _drop_guided_descriptions(schema, guide):
    """沿 JSON Schema 结构节点剥离已被 ``Input field guide`` 同路径或共享行覆盖的描述。"""
    from ....core.schema_guide import strip_guided_schema_descriptions

    return strip_guided_schema_descriptions(schema, guide)


def _native_dynamic_tools(tools, *, catalog_profile="full_v1", supports_search_tool=False):
    """构造 Codex ``dynamicTools`` 传输载荷：在传输层统一剥离与字段指南重复的 Schema 描述；
    当通道不支持 ``tool_search``（默认 ``supports_search_tool=False``）时扁平直出全部可用工具，
    仅在 ``supports_search_tool=True`` 时将 ``deferLoading`` 工具归入 ``DYNAMIC_TOOL_NAMESPACE``。
    """
    _ = catalog_profile
    eager, deferred = [], []
    for tool in tools:
        tool = deepcopy(tool)
        if isinstance(tool.get("inputSchema"), dict):
            tool["inputSchema"] = _drop_guided_descriptions(tool["inputSchema"], tool.get("description") or "")
        if supports_search_tool and tool.get("deferLoading", False):
            deferred.append(tool)
        else:
            if not supports_search_tool and tool.get("deferLoading", False):
                tool.pop("deferLoading", None)
            eager.append(tool)
    if deferred:
        eager.append(
            dict(
                type="namespace",
                name=DYNAMIC_TOOL_NAMESPACE,
                description="NI-RUA tools for robot observations, geometry, execution evidence and task support.",
                tools=deferred,
            )
        )
    return eager


def _toml_value(value):
    """Encode native extension tables without changing their keys or values."""
    if isinstance(value, dict):
        return "{" + ", ".join(json.dumps(key) + " = " + _toml_value(item) for key, item in value.items()) + "}"
    if isinstance(value, list):
        return "[" + ", ".join(_toml_value(item) for item in value) + "]"
    return json.dumps(value, ensure_ascii=False, allow_nan=False)


def _read_codex_auth_json(source):
    """Read ``<source>/auth.json`` without retaining raw credential text on parse errors."""
    auth_path = Path(source) / "auth.json"
    if not auth_path.exists():
        return {}
    failure = None
    auth = None
    try:
        auth = json.loads(auth_path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        failure = type(error).__name__
    if failure is not None or not isinstance(auth, dict):
        raise ValueError(f"codex_auth_unreadable:{auth_path}: {failure or 'not_object'}")
    return auth


def _chatgpt_login(source, auth=None):
    auth = _read_codex_auth_json(source) if auth is None else auth
    return auth.get("auth_mode") in ("chatgpt", "chatgptAuthTokens") or (
        not auth.get("auth_mode") and bool(auth.get("tokens")) and not auth.get("OPENAI_API_KEY")
    )


def _model_provider(config, source):
    """Resolve the upstream before replacing it with the private Responses relay."""
    name = config.get("model_provider", "openai")
    provider = dict(config.get("model_providers", {}).get(name) or {})
    if name == "openai":
        provider.setdefault("requires_openai_auth", True)
    explicit = provider.get("base_url")
    if name == "openai":
        explicit = explicit or config.get("openai_base_url") or os.environ.get("OPENAI_BASE_URL")
    if explicit:
        return name, provider, explicit
    if provider.get("requires_openai_auth") and _chatgpt_login(source):
        base = config.get("chatgpt_base_url", "https://chatgpt.com/backend-api").rstrip("/")
        upstream = base if base.endswith("/codex") else base + "/codex"
    else:
        upstream = "https://api.openai.com/v1"
    return name, provider, upstream


def _upstream_auth(provider, source):
    """Credentials stay in the host relay, never in the actor's home or env."""
    auth = _read_codex_auth_json(source)
    headers = dict(provider.get("http_headers") or {})
    for header, key in (provider.get("env_http_headers") or {}).items():
        if key in os.environ:
            headers[header] = os.environ[key]
    if any(key.lower() not in {"authorization", "chatgpt-account-id"} for key in headers):
        raise ValueError("sandbox_provider_requires_supported_auth_headers")
    key = provider.get("env_key")
    token = os.environ.get(key) if key else provider.get("experimental_bearer_token")
    if provider.get("requires_openai_auth"):
        if _chatgpt_login(source, auth=auth):
            tokens = auth.get("tokens")
            tokens = tokens if isinstance(tokens, dict) else {}
            token = tokens.get("access_token")
            if tokens.get("account_id"):
                headers["ChatGPT-Account-Id"] = tokens["account_id"]
        else:
            token = auth.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY")
    if token:
        headers = {key: value for key, value in headers.items() if key.lower() != "authorization"}
        headers["Authorization"] = "Bearer " + token
    return headers


def normalize_token_usage(value):
    """Preserve native last/total snapshots; absent counters are unknown, not zero."""
    value = value if isinstance(value, dict) else {}
    fields = {
        "input_tokens": "inputTokens",
        "cached_input_tokens": "cachedInputTokens",
        "cache_write_input_tokens": "cacheWriteInputTokens",
        "output_tokens": "outputTokens",
        "reasoning_output_tokens": "reasoningOutputTokens",
        "total_tokens": "totalTokens",
    }

    def counter(number):
        return number if type(number) is int and number >= 0 else None

    return {
        **{
            kind: (
                {target: counter(value[kind].get(native)) for target, native in fields.items()}
                if isinstance(value.get(kind), dict)
                else None
            )
            for kind in ("total", "last")
        },
        "model_context_window": counter(value.get("modelContextWindow")),
    }


class RecordedClient(CodexClient):
    """SDK client recording raw JSON-RPC frames and enforcing registry-declared tool concurrency."""

    def __init__(
        self,
        config,
        store,
        tool_handler,
        notification_handler,
        request_handler=None,
        disconnect_handler=None,
        *,
        route_notifications=True,
        parallel_safe=None,
        context_reset_handler=None,
    ):
        super().__init__(config, approval_handler=self._approval)
        self.store, self.tool_handler, self.notification_handler = store, tool_handler, notification_handler
        self.record_send_lock = threading.Lock()
        self.request_handler, self.disconnect_handler = request_handler, disconnect_handler
        self.context_reset_handler = context_reset_handler
        self.route_notifications = route_notifications
        self.server_pool = ThreadPoolExecutor(max_workers=8, thread_name_prefix="codex-tools")
        self.parallel_safe = parallel_safe
        self._tool_barrier = OrderedBarrier()

    def _submit_request(self, message):
        """Run non-tool RPCs immediately; serialize or parallelize tool calls per registry contract."""
        if message.get("method") != "item/tool/call":
            return self.server_pool.submit(self._reply_request, message)
        parallel = is_parallel_call(self.parallel_safe, message.get("params"))
        return self._tool_barrier.schedule(self.server_pool, parallel, self._reply_request, message)

    def _reader_loop(self):
        try:
            while True:
                message = self._read_message()
                if "method" in message and "id" in message:
                    self._submit_request(message)
                elif "method" in message:
                    if self.route_notifications and isinstance(message["method"], str):
                        self._router.route_notification(
                            self._coerce_notification(message["method"], message.get("params"))
                        )
                else:
                    self._router.route_response(message)
        except BaseException as error:
            self._router.fail_all(error)
            if self.disconnect_handler:
                self.disconnect_handler(error)

    def _reply_request(self, message):
        try:
            response = (
                self.request_handler(message)
                if self.request_handler and message["method"] != "item/tool/call"
                else self._handle_server_request(message)
            )
            if self.request_handler and message["method"] != "item/tool/call" and response is not None:
                response = self.request_handler(message, before_send=True)
            if response is None:
                return
            self._write_message(dict(id=message["id"], result=response))
            if self.request_handler and message["method"] != "item/tool/call":
                self.request_handler(message, sent=True)
        except Exception as error:
            try:
                if message.get("method") == "item/tool/call":
                    try:
                        self.store.set_state(
                            "runtime",
                            "image_context_epoch",
                            dict(
                                source="tool_reply_delivery_uncertain",
                                rpc_id=message.get("id"),
                                marker=new_id("image-delivery-failure"),
                            ),
                        )
                    finally:
                        if self.context_reset_handler is not None:
                            try:
                                self.context_reset_handler("tool_reply_delivery_uncertain")
                            except Exception:
                                pass
                self.store.event("codex.request_response_unknown", {"rpc_id": message.get("id"), "reason": str(error)})
            except Exception:
                pass
            finally:
                if self.request_handler and message.get("method") != "item/tool/call":
                    self.request_handler(message, failed=str(error))

    def _read_message(self):
        message = super()._read_message()
        self.store.raw_codex("received", message)
        if "method" in message:
            self.notification_handler(message)
        return message

    def _write_message(self, message):
        with self.record_send_lock:
            self.store.raw_codex("sent", message)
            super()._write_message(message)

    def _approval(self, method, params):
        if method == "item/tool/call":
            params = params if isinstance(params, dict) else {}
            from ....core.presentation.actor_presentation import ActorPresentation
            from ..mcp_server import safe_present_tool_result, tool_bridge_error_response

            presentation = ActorPresentation(self.store)
            executed = False
            try:
                result = self.tool_handler(params)
                executed = True
            except Exception as error:
                result = tool_bridge_error_response(
                    error,
                    tool=params.get("tool"),
                    arguments=params.get("arguments"),
                    store=self.store,
                    call_id=params.get("callId"),
                )
            if not executed:
                return safe_present_tool_result(presentation, result)
            try:
                return presentation.present(result)
            except Exception as present_error:
                if self.context_reset_handler is not None:
                    try:
                        self.context_reset_handler("tool_reply_encoding_failed")
                    except Exception:
                        pass
                from ....core.tool_output import ToolOutput, wire_response

                delivery_err = dict(
                    error=f"post_execution_presentation_failed:{type(present_error).__name__}",
                    code="post_execution_presentation_failed",
                    error_category="unknown_execution",
                    retryable=False,
                    field_path="",
                    actual=None,
                    suggestion=(
                        "The tool call ran, but its result could not be presented. Do NOT replay physical "
                        "actions blindly; inspect robot_action_status / robot_observe first."
                    ),
                    minimal_example=None,
                    outcome_semantics="Tool executed; physical motion may have occurred before presentation failed.",
                )
                if params.get("tool"):
                    delivery_err["tool"] = params["tool"]
                response = ToolOutput(delivery_err).response()
                response.success = False
                return wire_response(response)
        if method in NATIVE_RESPONSE_SCHEMAS:
            response = CodexSession._approve(method, params or {})
            validate_json(response, NATIVE_RESPONSE_SCHEMAS[method])
            return response
        if method.endswith("requestApproval"):
            return CodexSession._decline(method)
        return {}


class CodexSession(CodexModelListingMixin, AgentSession):
    """Codex backend: pinned Python SDK over a stdio bridge to the run's private app-server."""

    backend_kind = "codex"
    counting = MODEL_COMPLETION_COUNTING
    supports_search_tool = False
    capabilities = AgentCapabilities(
        steer=True,
        native_requests=True,
        terminal_attach=True,
        native_terminal_replay=True,
        fork=True,
        deferred_tools=True,
        account_relay=True,
        native_evidence_agent=True,
        sandbox=True,
        service_tier=True,
        native_memories=True,
        live_effort=True,
        compaction_signal="native+relay",
        turn_images=True,
    )
    effort_choices = model_effort_choices()

    def __init__(
        self,
        run_dir,
        store,
        tool_handler,
        model=None,
        effort=None,
        price_card=None,
        *,
        config=None,
        task_planning=None,
        verbosity=None,
    ):
        from ....core.config import Config
        from ....core.presentation.actor_presentation import ActorPresentation

        self.config = config if config is not None else Config.load()
        self.sandbox_enabled = self.config.features.sandbox_enabled
        self.isolation = ACTOR_ISOLATION if self.sandbox_enabled else "disabled"
        self.root = Path(run_dir).resolve()
        self.store, self.tool_handler = store, tool_handler
        self.task_planning = task_planning
        self.presentation = ActorPresentation(store)
        self.model = self.config.model.name if model is None else model
        self.active_turn_model = self.model
        self.effort = self.config.model.effort if effort is None else effort
        self._explicit_verbosity = verbosity
        self.verbosity = self._resolve_verbosity()
        self.provider = None
        self.upstream_profile = None
        self.price_card = price_card
        self.image_settings = store_image_settings(store, self.config.images.to_dict())
        self.actor_dir = self.root / "actor"
        self.private = self.root / "private"
        self.codex_home = self.private / "codex-home"
        longest = "sandbox-owner.sock" if self.sandbox_enabled else "codex.sock"
        self.transport = AppServerTransport(self.root, longest=longest)
        self.ipc_runtime_dir = self.transport.layout.runtime_dir
        self.ipc_base = self.transport.layout.base
        self.ipc = self.transport.layout.ipc
        self.socket_path = self.transport.socket_path
        self.process = self.client = None
        self.thread_id = self.turn_id = None
        self.turn_done = threading.Event()
        self.turn_result = None
        self.state = "not_started"
        self.last_activity_at = time.monotonic()
        self.pending = {}
        self.pending_lock = threading.RLock()
        self.native_connection_id = new_id("native-connection")
        self.resolved_requests = set()
        self.closing = False
        self.close_report = None
        self.on_disconnect = None
        self.model_proxy = None
        self.transport_recovery_gate = None
        self.account_control = None
        self.sandbox = None
        self.command_prefix = []
        self.model_limit_reached = threading.Event()
        self.projection_lock = threading.RLock()
        self.live_revision = self.turn_revision = 0
        self.item_revisions = {}
        self.response_schemas = deepcopy(NATIVE_RESPONSE_SCHEMAS)
        self.deferred_tools = frozenset()
        self._egress_cache = EgressProjectionCache()

    @property
    def ipc_prepared(self):
        transport = getattr(self, "transport", None)
        if transport is not None:
            return transport.prepared
        return self.__dict__.get("_ipc_prepared", False)

    @ipc_prepared.setter
    def ipc_prepared(self, value):
        transport = getattr(self, "transport", None)
        if transport is not None:
            transport.prepared = bool(value)
        else:
            self.__dict__["_ipc_prepared"] = bool(value)

    def _resolve_verbosity(self):
        explicit = getattr(self, "_explicit_verbosity", None)
        configured = getattr(getattr(getattr(self, "config", None), "model", None), "verbosity", None)
        value = explicit if explicit is not None else (configured if configured is not None else DEFAULT_MODEL_VERBOSITY)
        if value not in VALID_MODEL_VERBOSITIES:
            raise ValueError(f"invalid_model_verbosity:{value}")
        return value

    def _task_decision_context(self, request, identity):
        """Carry advisory actor planning state into context compaction."""
        if (
            self.task_planning is None
            or not self.thread_id
            or identity.get("native_thread_id") != self.thread_id
            or identity.get("source") != "actor"
            or identity.get("endpoint") != "/responses/compact"
        ):
            return request
        context = self.task_planning.decision_context(self.store.identity)
        if context is None and not self.task_planning.required:
            return request
        reminder = decision_reminder_text(context)
        result = deepcopy(request)
        items = result.get("input", [])
        if isinstance(items, str):
            items = [dict(role="user", content=items)]
        if not isinstance(items, list):
            raise ValueError("invalid_model_request_input")

        def is_decision_reminder(item):
            if not isinstance(item, dict) or item.get("role") != "assistant":
                return False
            if item.get("type") != "message" or not isinstance(item.get("content"), list):
                return False
            parts = item["content"]
            if len(parts) != 1 or not isinstance(parts[0], dict):
                return False
            text = parts[0].get("text")
            if parts[0].get("type") != "output_text" or not isinstance(text, str):
                return False
            return any(
                text.startswith(prefix)
                for prefix in (
                    "Current task decision boundary\nThe saved plan and attention",
                    "Current task decision context\nThe saved plan and attention",
                )
            )

        items = [item for item in items if not is_decision_reminder(item)]
        items.append(dict(type="message", role="assistant", content=[dict(type="output_text", text=reminder)]))
        result["input"] = items
        return result

    def tool_output_text_budget_bytes(self):
        """Codex's text budget for one tool output of the session model (``codex_tool_output_budget_bytes``)."""
        return codex_tool_output_budget_bytes(self.model)

    def _tool_output_limit_lines(self):
        """With ``features.budgeted_action_reply`` (default) the catalogue limit is also written as
        ``tool_output_token_limit``, so the number the Harness fits replies to is stated in the native configuration
        itself. Off: no line."""
        features = getattr(getattr(self, "config", None), "features", None)
        limit = codex_tool_output_token_limit(self.model) if getattr(features, "budgeted_action_reply", False) else None
        return () if limit is None else (f"tool_output_token_limit = {limit}",)

    def _present_model_request(self, request):
        """Project readable Responses history, including resumed/compact input."""
        from ...session.wire.openai_responses import present_responses_request

        return present_responses_request(
            request,
            self.presentation,
            deferred_tools=self.deferred_tools,
            namespace=DYNAMIC_TOOL_NAMESPACE,
            cache=getattr(self, "_egress_cache", None),
        )

    def _request_id(self, message):
        return "native-" + digest([self.native_connection_id, message["id"]])[:24]

    def _server_request(self, message, sent=False, failed=None, before_send=False):
        identifier = self._request_id(message)
        if before_send:
            with self.pending_lock:
                entry = self.pending.get(identifier)
                if (
                    identifier in self.resolved_requests
                    or self.closing
                    or self.state == "disconnected"
                    or (entry and entry.get("expired"))
                ):
                    return None
                if entry:
                    return entry["response"]
                return self._decline(message["method"]) if message["method"] in self.response_schemas else {}
        if sent or failed:
            with self.pending_lock:
                entry = self.pending.get(identifier)
                if entry:
                    info = self.store.read_state("native_request", identifier)
                    info.update(
                        status="expired" if entry.get("expired") else "unknown" if failed else "resolved",
                        response_digest=digest(entry["response"]),
                        response_sent_at=utc_now(),
                        response_error=failed,
                    )
                    self.store.set_state("native_request", identifier, info)
                    self.store.event(
                        "codex.request_resolved",
                        info,
                        {k: info[k] for k in ("thread_id", "turn_id", "item_id") if info.get(k)},
                    )
                    entry["sent"].set()
            return
        method, params = message["method"], message.get("params") or {}
        if method not in self.response_schemas:
            self.store.event("codex.unsupported_request", {"rpc_id": message["id"], "method": method})
            return {}
        if params.get("threadId") != self.thread_id or (self.turn_id and params.get("turnId") != self.turn_id):
            self.store.event("codex.request_rejected", {"rpc_id": message["id"], "reason": "request_identity_mismatch"})
            return self._decline(method)
        response = self._approve(method, params)
        validate_json(response, self.response_schemas[method])
        info = dict(
            request_id=identifier,
            rpc_id=message["id"],
            method=method,
            params=params,
            status="responding",
            thread_id=params.get("threadId"),
            turn_id=params.get("turnId"),
            item_id=params.get("itemId"),
            created_at=utc_now(),
            response_policy="unattended",
            response_digest=digest(response),
            reason="unattended_no_human_input"
            if method == "item/tool/requestUserInput"
            else "unattended_permission_granted",
        )
        entry = dict(
            event=threading.Event(),
            sent=threading.Event(),
            method=method,
            params=params,
            response=response,
            automatic=True,
        )
        with self.pending_lock:
            self.pending[identifier] = entry
            if self.closing or self.state == "disconnected" or identifier in self.resolved_requests:
                entry["expired"] = True
                info.update(
                    status="expired",
                    reason="server_request_resolved"
                    if identifier in self.resolved_requests
                    else "session_closed_or_disconnected",
                )
            self.store.set_state("native_request", identifier, info)
            self.store.event("codex.request_expired" if entry.get("expired") else "codex.request_automatic", info)
            entry["event"].set()
            return None if entry.get("expired") else response

    def _resolve_request(self, params):
        rpc_id = params.get("requestId")
        if params.get("threadId") != self.thread_id or type(rpc_id) not in (str, int):
            return
        identifier = self._request_id({"id": rpc_id})
        with self.pending_lock:
            self.resolved_requests.add(identifier)
            entry = self.pending.get(identifier)
            if entry:
                info = self.store.read_state("native_request", identifier)
                if info["status"] in ("pending", "responding"):
                    entry["expired"] = True
                    info.update(status="expired", reason="server_request_resolved")
                    self.store.set_state("native_request", identifier, info)
                    self.store.event("codex.request_expired", info)
                entry["event"].set()
                entry["sent"].set()

    @staticmethod
    def _decline(method):
        if method == "item/tool/requestUserInput":
            return {"answers": {}}
        if method == "item/permissions/requestApproval":
            return {"permissions": {}, "scope": "turn"}
        return {"decision": "decline"}

    @staticmethod
    def _approve(method, params):
        if method == "item/permissions/requestApproval":
            return {
                "permissions": deepcopy(params.get("permissions", {})),
                "scope": "session",
                "strictAutoReview": False,
            }
        if method in {"item/commandExecution/requestApproval", "item/fileChange/requestApproval"}:
            return {"decision": "acceptForSession"}
        return CodexSession._decline(method)

    def reply(self, identifier, response):
        with self.pending_lock:
            entry = self.pending.get(identifier)
            info = self.store.read_state("native_request", identifier)
            if not entry or not info or entry.get("automatic") or info["status"] in ("expired", "unknown", "resolved"):
                raise ValueError("native_request_expired_or_not_owned")
            if entry["response"] is not None:
                if digest(entry["response"]) != digest(response):
                    raise ValueError("native_response_content_conflict")
                return info
            if not isinstance(response, dict):
                raise ValueError("native_response_requires_object")
            try:
                validate_json(response, self.response_schemas[entry["method"]])
            except Exception as error:
                raise ValueError("invalid_native_response:" + str(error).splitlines()[0]) from error
            if entry["method"] == "item/tool/requestUserInput":
                question_ids = {question["id"] for question in entry["params"].get("questions", [])}
                if set(response["answers"]) - question_ids:
                    raise ValueError("unknown_native_question_id")
            entry["response"] = response
            info.update(status="responding", response_digest=digest(response))
            self.store.set_state("native_request", identifier, info)
            entry["event"].set()
        entry["sent"].wait(5)
        return self.store.read_state("native_request", identifier)

    def expire_requests(self, reason):
        with self.pending_lock:
            for info in self.store.states("native_request"):
                if info["request_id"] not in self.pending and info["status"] in ("pending", "responding"):
                    info.update(status="expired", reason=reason)
                    self.store.set_state("native_request", info["request_id"], info)
                    self.store.event("codex.request_expired", info)
            for identifier, entry in self.pending.items():
                info = self.store.read_state("native_request", identifier)
                if info["status"] in ("pending", "responding"):
                    entry.update(expired=True, response=self._decline(entry["method"]))
                    info.update(status="expired", reason=reason)
                    self.store.set_state("native_request", identifier, info)
                    self.store.event("codex.request_expired", info)
                    entry["event"].set()
                    entry["sent"].set()

    def _disconnected(self, error):
        lock = getattr(self, "projection_lock", None)
        if lock is None:
            self.projection_lock = lock = threading.RLock()
        with lock:
            if getattr(self, "closing", False) or getattr(self, "state", None) in ("disconnected", "closed"):
                return
            self.state = "disconnected"
            if not self.turn_done.is_set():
                self.turn_result = {
                    "id": getattr(self, "turn_id", None),
                    "status": "failed",
                    "error": {"message": "upstream_connection_lost: " + type(error).__name__},
                }
        try:
            try:
                self.store.event(
                    "codex.connection_lost",
                    {"reason": type(error).__name__, "detail": str(error)[:200], "history_complete": False},
                )
            except Exception:
                pass
            try:
                self.expire_requests("upstream_connection_lost")
            except Exception:
                pass
        finally:
            try:
                if self.on_disconnect:
                    self.on_disconnect()
            finally:
                self.turn_done.set()

    def _prepare_ipc(self):
        self.transport.prepare()

    def _transport_recovery_block(self):
        gate = self.transport_recovery_gate
        return "recovery_gate_unavailable" if gate is None else gate()

    def _profile_upstream(self):
        from ..factory import actor_upstream
        from ..profiles import resolve_profile_upstream

        upstream = actor_upstream(self.config, "codex")
        if upstream is None:
            return None
        resolved = resolve_profile_upstream(upstream, "codex")
        if set(resolved.headers) != {"Authorization"}:
            raise ValueError(
                f"codex_profile_auth_scheme_unsupported:{resolved.profile.path}: the Responses relay "
                'sends the key as Authorization: Bearer; set wires.openai_responses.auth_scheme to "bearer"'
            )
        return upstream.provider or resolved.profile.id, resolved

    def _prepare(self, expected_identity=None):
        self._prepare_ipc()
        previous = self.store.read_state("runtime", "codex")
        if previous and previous.get("isolation") != self.isolation:
            raise ValueError("resume_isolation_changed: use a fresh run when changing sandbox mode")
        if self.codex_home.is_symlink() or any(
            (self.codex_home / item).is_symlink()
            for item in ("config.toml", "models.json", "auth.json", "account-control-ca.pem")
        ):
            raise ValueError("sandbox_home_symlink: use a fresh run")
        if previous is None and any(
            (self.codex_home / item).exists() for item in (".credentials.json", "skills", "plugins")
        ):
            raise ValueError("sandbox_home_contains_host_extensions: use a fresh run")
        for path in [self.actor_dir, self.codex_home, self.private / "tmp"]:
            path.mkdir(parents=True, exist_ok=True)
        self.private.chmod(0o700)
        profile = self._profile_upstream()
        if profile is None:
            source = Path(self.config.model.codex_home).expanduser()
            config = tomllib.loads((source / "config.toml").read_text())
            provider_name, provider, upstream_url = _model_provider(config, source)
            self.upstream_profile = None
        else:
            provider_name, resolved = profile
            source, config, provider, upstream_url = None, {}, {}, resolved.base_url
            self.upstream_profile = dict(
                id=resolved.profile.id,
                name=resolved.profile.name,
                wire_api=resolved.wire_api,
            )
            self._profile = resolved.profile
        if expected_identity is not None and expected_identity != (self.model, provider_name):
            raise ValueError("resume_usage_identity_conflict: cumulative usage requires the same model/provider")
        if self.sandbox:
            self.sandbox.close()
            self.sandbox = None
        for relay in (self.model_proxy, self.account_control):
            if relay:
                relay.close()
        self.model_proxy = self.account_control = None
        needs_account_control = provider.get("requires_openai_auth") and _chatgpt_login(source)
        upstream_auth = _upstream_auth(provider, source) if profile is None else dict(resolved.headers)
        if needs_account_control and config.get("chatgpt_base_url", ACCOUNT_BASE_URL).rstrip("/") not in (
            ACCOUNT_BASE_URL,
            "https://chatgpt.com",
        ):
            raise ValueError("account_control_requires_default_chatgpt_backend")
        verbosity = self._resolve_verbosity()
        self.verbosity = verbosity
        model_catalog = write_direct_model_catalog(
            self.codex_home / "models.json",
            supports_search_tool=self.supports_search_tool,
        )
        lines = [
            f"model = {json.dumps(self.model)}",
            f"model_provider = {json.dumps(MODEL_PROXY_PROVIDER)}",
            f"model_catalog_json = {json.dumps(str(model_catalog))}",
            f"model_reasoning_effort = {json.dumps(self.effort)}",
            'approval_policy = "never"',
            'sandbox_mode = "danger-full-access"',
            'web_search = "disabled"',
            f"model_verbosity = {json.dumps(verbosity)}",
            'service_tier = "default"',
            "suppress_unstable_features_warning = true",
            *BOUNDARY_CONFIG_LINES,
            *self._tool_output_limit_lines(),
            "[features]",
            "code_mode_host = false",
            "shell_tool = true",
            "unified_exec = true",
            "browser_use = false",
            "browser_use_external = false",
            "computer_use = false",
            "apps = false",
            "plugins = false",
            "artifact = false",
            "image_generation = false",
            "goals = true",
            "hooks = true",
            "memories = " + str(self.config.model.native_memories).lower(),
            "network_proxy = false",
            "mentions_v2 = true",
            "multi_agent = true",
            "prevent_idle_sleep = true",
            "guardian_approval = true",
            "multi_agent_v2 = true",
            "fast_mode = false",
            "context_management.experimental_mode = true",
            "[features.code_mode]",
            "enabled = false",
            "[memories]",
            f"extract_model = {json.dumps(self.model)}",
            f"consolidation_model = {json.dumps(self.model)}",
        ]
        if provider.get("wire_api", "responses") != "responses":
            raise ValueError("official_model_limits_require_responses_provider")
        previous = self.store.read_state("runtime", "model_requests", {})
        if previous and previous.get("counting") != MODEL_COMPLETION_COUNTING:
            raise ValueError("resume_model_budget_semantics_changed: choose a fresh run")
        used = previous.get("completed_calls", 0)
        previous_attempts = previous.get("http_attempt_count", 0)
        if used >= 180:
            raise ValueError("episode_model_request_limit_exhausted")
        self.model_limit_reached.clear()
        recovery = (
            (previous.get("transport_recovery") or {}).get("policy")
            if previous
            else self.config.model.transport_recovery.to_dict()
        )
        self.store.set_state(
            "runtime",
            "model_image_policy",
            dict(
                enabled=self.image_settings["reuse_exact_images"],
                policy=IMAGE_REUSE_POLICY,
                compaction="unchanged",
                oversized_trim=self.image_settings["trim_oversized_requests"],
                max_inline_images=self.image_settings["trim_max_inline_images"],
            ),
        )
        relay_seed = self.store.read_state("runtime", "model_usage_relay", None)
        relay_usage, served_models = RelayUsageTotal(relay_seed), dict(previous.get("served_models") or {})
        if used and not isinstance(relay_seed, dict):
            relay_usage.total, relay_usage.calls = dict.fromkeys(relay_usage.total), used
            relay_usage.unknown_calls = used
        attempts, relay_lock = {}, threading.Lock()

        def request_event(event):
            kind, call = event.get("kind"), event.get("logical_call_id")
            with relay_lock:
                if kind == "model_request_finished" and call is not None:
                    attempts[call] = (
                        responses_counters(event.get("usage")) if event.get("usage_known") else None,
                        event.get("served_model"),
                    )
                elif kind in ("model_call_completed", "model_call_finished") and call is not None:
                    counters, served = attempts.pop(call, (None, None))
                    if kind == "model_call_completed":
                        served_models.update(add_served_model(served_models, served))
                        self.store.set_state(
                            "runtime",
                            "model_usage_relay",
                            relay_usage.add(counters, thread_id=self.thread_id),
                        )
            if event.get("endpoint") == "/responses/compact" and event.get("kind") == "model_call_started":
                self.store.set_state("runtime", "image_context_epoch", event.get("logical_call_id"))
            value = dict(
                completed_calls=used + self.model_proxy.request_count,
                http_attempt_count=previous_attempts + self.model_proxy.attempt_count,
                counting=MODEL_COMPLETION_COUNTING,
                limit=180,
                timeout_s=None,
                max_retries=DEFAULT_MAX_RETRIES,
                retry_policy=RETRY_POLICY,
                transport_blocked=self.model_proxy.transport_blocked,
                transport_error=self.model_proxy.blocked_error,
                transport_recovery=self.model_proxy.transport_recovery_summary,
                served_models=dict(served_models),
                last_event=event,
            )
            self.store.set_state("runtime", "model_requests", value)
            self.store.event("model." + event["kind"], event, source="private_model_transport")

        self._start_model_catalogue(upstream_url, upstream_auth)
        try:
            self.model_proxy = ModelRequestProxy(
                upstream_url,
                upstream_auth_headers=upstream_auth,
                max_requests=180 - used,
                request_timeout_s=None,
                max_retries=DEFAULT_MAX_RETRIES,
                on_event=request_event,
                image_reuse=self.image_settings["reuse_exact_images"],
                image_trim=self.image_settings["trim_oversized_requests"],
                max_inline_images=self.image_settings["trim_max_inline_images"],
                transport_recovery=recovery,
                recovery_gate=self._transport_recovery_block,
                request_context=self._task_decision_context,
                request_projection=self._present_model_request,
                on_limit=lambda _event: self.model_limit_reached.set(),
                metrics_path=self.private / (new_id("model-requests") + ".jsonl"),
                classify_call_purpose=True,
            ).start()
            if needs_account_control:
                account_control = AccountControlRelay(
                    self.private / "account-control",
                    self.codex_home / "account-control-ca.pem",
                    upstream_auth_headers=upstream_auth,
                )
                try:
                    self.account_control = account_control.start()
                except BaseException:
                    account_control.close()
                    raise
                lines.insert(0, f"chatgpt_base_url = {json.dumps(self.account_control.base_url)}")
            provider = dict(name=provider_name, requires_openai_auth=True)
            provider.update(
                base_url=self.model_proxy.base_url,
                wire_api="responses",
                supports_websockets=False,
                request_max_retries=0,
                stream_max_retries=0,
                stream_idle_timeout_ms=2**63 - 1,
            )
            provider.setdefault("name", provider_name)
            lines.append(f"[model_providers.{MODEL_PROXY_PROVIDER}]")
            for key, value in provider.items():
                lines.append(f"{key} = {json.dumps(value)}")
            config_text = "\n".join(lines) + "\n"
            self.native_config = tomllib.loads(config_text)
            (self.codex_home / "config.toml").write_text(config_text)
            (self.codex_home / "config.toml").chmod(0o600)
            (self.codex_home / "auth.json").write_text(json.dumps({"OPENAI_API_KEY": "ni-rua-local-relay"}))
            (self.codex_home / "auth.json").chmod(0o600)
            executable = bundled_codex_path().resolve()
            self.cli_executable = executable
            from ....core.config import DEFAULT_SUBPROCESS_THREAD_ENV

            self.env = {"PATH": native_tool_path(executable), "LANG": "C.UTF-8", **DEFAULT_SUBPROCESS_THREAD_ENV}
            self.env.update(HOME=str(self.actor_dir), CODEX_HOME=str(self.codex_home), TMPDIR=str(self.private / "tmp"))
            if not self.sandbox_enabled:
                self.env.update(boundary_env(self.actor_dir))
            if self.account_control:
                self.env["CODEX_CA_CERTIFICATE"] = str(self.account_control.ca_path)
            self.account_ca_bytes = (
                self.account_control.ca_path.read_bytes()
                if (self.account_control and self.account_control.ca_path.is_file())
                else None
            )
            source_snapshot = snapshot_actor_source(self.private / "actor-source")
            self.command_prefix = []
            if self.sandbox_enabled:
                from .sandbox import CodexSandbox

                ports = [urlsplit(self.model_proxy.base_url).port]
                if self.account_control:
                    ports.append(urlsplit(self.account_control.base_url).port)
                self.sandbox = CodexSandbox(
                    self.private / "sandbox",
                    self.ipc,
                    self.actor_dir,
                    self.codex_home,
                    executable,
                    ports=ports,
                    packages=True,
                    read_paths=[(source_snapshot, Path("/harness"))],
                )
                self.command_prefix, overrides = self.sandbox.prepare()
                self.env.update(overrides)
            self.command = self.command_prefix + [
                str(executable),
                "--no-daemon",
                "app-server",
                "--listen",
                f"unix://{self.socket_path}",
            ]
            return provider_name
        except BaseException:
            self._close_model_catalogue()
            if self.sandbox:
                try:
                    self.sandbox.close()
                except Exception:
                    pass
                self.sandbox = None
            if self.account_control:
                try:
                    self.account_control.close()
                except Exception:
                    pass
                self.account_control = None
            if self.model_proxy:
                try:
                    self.model_proxy.close()
                except Exception:
                    pass
                self.model_proxy = None
            raise

    def _stop_previous_owned_processes(self, previous):
        """A crashed gateway may leave its native app-server alive; never attach to it."""
        self._prepare_ipc()
        uri = f"unix://{self.socket_path}"
        runtime = self.store.read_state("runtime", "app_server") or {}
        candidates = [
            ((previous or {}).get("app_server_pid"), (previous or {}).get("socket_uri") or uri),
            (runtime.get("pid"), runtime.get("socket_uri") or uri),
        ]
        terminal_file = self.root / "terminal.json"
        if terminal_file.exists():
            terminal = json.loads(terminal_file.read_text())
            candidates.append((terminal.get("pid"), terminal.get("socket_uri") or uri))
        for pid, owned_uri in set(candidates):
            if pid is None:
                continue
            try:
                handle = open_process(pid)
            except ProcessLookupError:
                continue
            try:
                arguments = process_argv(pid)
                if owned_uri not in arguments:
                    continue
                handle.send_signal(signal.SIGTERM)
                if not handle.wait(5):
                    handle.send_signal(signal.SIGKILL)
                    if not handle.wait(5):
                        raise RuntimeError("previous_owned_codex_stop_unconfirmed")
                self.store.event("codex.previous_process_stopped", {"pid": pid, "stop_confirmed": True})
            except (FileNotFoundError, ProcessLookupError, PermissionError):
                pass
            finally:
                handle.close()
        self.socket_path.unlink(missing_ok=True)

    def start(self, tools, instructions, resume_thread_id=None):
        if self.process and self.process.poll() is None:
            raise ValueError("session_already_started")
        require_codex_version()
        previous = self.store.read_state("runtime", "codex")
        if previous and previous.get("isolation") != self.isolation:
            raise ValueError("resume_isolation_changed: use a fresh run when changing sandbox mode")
        if resume_thread_id and previous and previous.get("tools_digest") != digest(tools):
            raise ValueError("resume_tool_registry_conflict")
        if self.client is not None:
            self.close()
        try:
            self.closing = False
            self._reopen_model_catalogue()
            return self._start(tools, instructions, resume_thread_id, previous)
        except BaseException:
            try:
                self.close()
            except Exception:
                pass
            raise

    def _start(self, tools, instructions, resume_thread_id, previous):
        self.deferred_tools = (
            frozenset(tool["name"] for tool in tools if tool.get("deferLoading", False))
            if self.supports_search_tool
            else frozenset()
        )
        with self.pending_lock:
            self.native_connection_id = new_id("native-connection")
            self.pending.clear()
            self.resolved_requests.clear()
        for info in self.store.states("native_request"):
            if info["status"] in ("pending", "responding"):
                info.update(status="expired", reason="app_server_restarted")
                self.store.set_state("native_request", info["request_id"], info)
        self._stop_previous_owned_processes(previous)
        expected_identity = None
        if resume_thread_id:
            saved_config = tomllib.loads((self.codex_home / "config.toml").read_text())
            saved_provider = saved_config.get("model_provider")
            if saved_provider == MODEL_PROXY_PROVIDER:
                saved_provider = (previous or {}).get("provider")
            expected_identity = (saved_config.get("model"), saved_provider)
        provider_name = self._prepare(expected_identity)
        self.provider = provider_name
        self.store.set_state("runtime", "price_card", self.price_card)
        self.log = (self.private / "app-server.log").open("a")
        self.process = subprocess.Popen(
            self.command,
            env=self.env,
            stdin=subprocess.DEVNULL,
            stdout=self.log,
            stderr=self.log,
            cwd=self.actor_dir,
        )
        self.store.set_state(
            "runtime",
            "app_server",
            {"pid": self.process.pid, "socket_uri": f"unix://{self.socket_path}"},
        )
        deadline = time.monotonic() + 20
        while not (self.sandbox.repair_socket_alias(self.socket_path) if self.sandbox else self.socket_path.is_socket()):
            if self.process.poll() is not None or time.monotonic() > deadline:
                raise RuntimeError(f"app-server startup failed; inspect {self.private / 'app-server.log'}")
            time.sleep(0.05)
        connection_id = self.native_connection_id

        def notification(message):
            if self.native_connection_id == connection_id:
                self._notification(message)

        def request(message, **kwargs):
            if self.native_connection_id == connection_id:
                return self._server_request(message, **kwargs)

        def disconnected(error):
            if self.native_connection_id == connection_id:
                self._disconnected(error)

        def context_reset(reason):
            if self.native_connection_id == connection_id and self.on_context_reset:
                self.on_context_reset(reason)

        def tool(params):
            if (
                self.native_connection_id != connection_id
                or self.closing
                or self.state in {"disconnected", "closed"}
            ):
                raise ValueError("native_tool_call_stale_connection")
            namespace = DYNAMIC_TOOL_NAMESPACE if params.get("tool") in self.deferred_tools else None
            if params.get("namespace") != namespace:
                raise ValueError("native_tool_call_namespace_mismatch")
            return self.tool_handler(params)

        socket_arg = self.sandbox.host_socket_path if self.sandbox else self.socket_path
        self.client = RecordedClient(
            CodexConfig(
                launch_args_override=build_bridge_command(socket_arg),
                client_name="ni-rua",
                client_title="NI-RUA",
                cwd=str(Path(__file__).resolve().parents[4]),
            ),
            self.store,
            tool,
            notification,
            request,
            disconnected,
            route_notifications=False,
            parallel_safe=self.tool_parallel_safe,
            context_reset_handler=context_reset,
        )
        self.client.start()
        initialized = self.client.initialize()
        verbosity = self._resolve_verbosity()
        self.verbosity = verbosity
        parameters = dict(
            model=self.model,
            modelProvider=MODEL_PROXY_PROVIDER,
            cwd=str(self.actor_dir),
            approvalPolicy="never",
            sandbox="danger-full-access",
            developerInstructions=self.presentation.present_egress_text(instructions),
            config={"model_reasoning_effort": self.effort, "model_verbosity": verbosity},
            allowProviderModelFallback=False,
        )
        if resume_thread_id:
            parameters.pop("allowProviderModelFallback")
            started = self.client.thread_resume(resume_thread_id, parameters)
        else:
            parameters["dynamicTools"] = self._present_model_request(dict(tools=_native_dynamic_tools(
                tools,
                catalog_profile=self.config.features.tool_catalog_profile,
                supports_search_tool=self.supports_search_tool,
            )))["tools"]
            started = self.client.thread_start(parameters)
        self.thread_id = started.thread.id
        if self.model_proxy is not None:
            self.model_proxy.register_thread(self.thread_id, "actor", self.thread_id)
        self.state = "idle"
        instance_id = new_id("appserver")
        tools_digest = digest(tools)
        write_codex_identity(
            self.store,
            dict(
                thread_id=self.thread_id,
                app_server_instance_id=instance_id,
                app_server_pid=self.process.pid,
                socket_uri=f"unix://{self.socket_path}",
                model=self.model,
                provider=self.provider,
                effort=self.effort,
                verbosity=verbosity,
                sdk_version=CODEX_SDK_VERSION,
                cli_version=CODEX_CLI_VERSION,
                cli_executable=str(self.cli_executable),
                initialization=initialized.model_dump(by_alias=True, mode="json"),
                isolation=self.isolation,
                tools_digest=tools_digest,
            ),
        )
        write_identity(
            self.store,
            backend=self.backend_kind,
            thread_id=self.thread_id,
            instance_id=instance_id,
            pid=self.process.pid,
            model=self.model,
            provider=self.provider,
            effort=self.effort,
            isolation=self.isolation,
            tools_digest=tools_digest,
            instructions_digest=digest(parameters["developerInstructions"]),
            counting=MODEL_COMPLETION_COUNTING,
            wire_api="openai_responses",
            sdk_version=CODEX_SDK_VERSION,
            cli_version=CODEX_CLI_VERSION,
            cli_executable=str(self.cli_executable),
            cli_sha256=None,
            native=dict(codex_home="private/codex-home", socket_uri=f"unix://{self.socket_path}"),
        )
        if resume_thread_id:
            self.recover_history()
        return self.thread_id

    def recover_history(self):
        with self.projection_lock:
            watermark = self.live_revision
        thread = self.client.thread_read(self.thread_id, include_turns=True).thread.model_dump(
            by_alias=True,
            mode="json",
        )
        recovery_ref = self.store.put_artifact(
            canonical(thread).encode(),
            "codex_history_snapshot",
            "application/json",
            "observer",
        )
        for turn in thread.get("turns", []):
            for item in turn.get("items", []):
                self._project_item(
                    item,
                    turn["id"],
                    item.get("status", "completed" if turn["status"] != "inProgress" else "unknown"),
                    raw_ref=recovery_ref,
                    recovered=True,
                    watermark=watermark,
                )
        with self.projection_lock:
            turns = thread.get("turns", [])
            if turns and self.turn_revision <= watermark:
                turn = turns[-1]
                self.turn_id = turn["id"]
                self.state = "running" if turn["status"] == "inProgress" else turn["status"]
                if self.state != "running":
                    self.turn_result = turn
                    self.turn_done.set()
        self.store.event(
            "codex.history_recovered",
            {"turns": len(turns), "delta_history_complete": False, "raw_ref": recovery_ref},
            {"thread_id": self.thread_id},
        )
        return thread

    def _project_item(self, item, turn_id, status, raw_ref=None, recovered=False, watermark=None):
        with self.projection_lock:
            return self._project_item_locked(item, turn_id, status, raw_ref, recovered, watermark)

    def _project_item_locked(self, item, turn_id, status, raw_ref=None, recovered=False, watermark=None):
        identifier = item.get("id")
        if not identifier:
            return
        previous = self.store.read_state("codex_item", identifier, {})
        if recovered and self.item_revisions.get(identifier, 0) > (watermark or 0):
            return
        if not recovered:
            self.live_revision += 1
            self.item_revisions[identifier] = self.live_revision
        refs = previous.get("raw_refs", []) + ([raw_ref] if raw_ref else [])
        previous.update(item)
        previous.update(
            id=identifier,
            thread_id=self.thread_id,
            turn_id=turn_id,
            status=item.get("status", status),
            raw_refs=refs,
        )
        self.store.set_state("codex_item", identifier, previous)

    def _notification(self, message):
        with self.projection_lock:
            return self._notification_locked(message)

    def _notification_locked(self, message):
        method = message["method"]
        params = message.get("params") or {}
        if not isinstance(params, dict):
            params = {}
        correlation = {
            target: params[key]
            for key, target in [("threadId", "thread_id"), ("turnId", "turn_id"), ("callId", "tool_call_id")]
            if params.get(key)
        }
        if isinstance(params.get("item"), dict) and params["item"].get("id"):
            correlation["item_id"] = params["item"]["id"]
        owned = self.thread_id is not None and params.get("threadId") == self.thread_id
        if owned and (
            method in {"item/started", "item/completed", "item/tool/call"}
            or (
                method.startswith("item/")
                and method.endswith(("Delta", "/delta"))
                and isinstance(params.get("delta"), str)
                and params["delta"]
            )
        ):
            self.last_activity_at = time.monotonic()
        if method == "serverRequest/resolved":
            self._resolve_request(params)
        if (
            owned
            and method in ("item/started", "item/completed")
            and isinstance(params.get("item"), dict)
            and params["item"].get("type") == "contextCompaction"
        ):
            self.store.set_state(
                "runtime",
                "image_context_epoch",
                dict(
                    source="native_context_compaction",
                    thread_id=self.thread_id,
                    item_id=params["item"].get("id"),
                    phase=method,
                ),
            )
        if owned and method in ("turn/started", "turn/completed"):
            self.live_revision += 1
            self.turn_revision = self.live_revision
        if method == "turn/started" and owned:
            self.turn_id = params["turn"]["id"]
            correlation["turn_id"] = self.turn_id
            self.state = "running"
        if method == "turn/completed" and owned and (params.get("turn") or {}).get("id") == self.turn_id:
            self.turn_result = params["turn"]
            self.state = params["turn"]["status"]
            self.turn_done.set()
        if method == "item/tool/call":
            correlation["item_id"] = params.get("callId")
        ref = self.store.put_artifact(canonical(message).encode(), "codex_event", "application/json", "observer")
        if owned and isinstance(params.get("item"), dict):
            self._project_item(
                params["item"],
                params.get("turnId"),
                "completed" if method == "item/completed" else "running",
                ref,
            )
        elif owned and params.get("itemId"):
            item = self.store.read_state("codex_item", params["itemId"], dict(id=params["itemId"]))
            if isinstance(params.get("delta"), str):
                key = "aggregatedOutput" if "outputDelta" in method else "text"
                item[key] = (item.get(key) or "") + params["delta"]
            self._project_item(item, params.get("turnId"), "running", ref)
        payload = {"raw_ref": ref, "method": method}
        if owned and method == "thread/tokenUsage/updated":
            usage = dict(
                normalize_token_usage(params.get("tokenUsage")),
                thread_id=params.get("threadId"),
                turn_id=params.get("turnId"),
                raw_ref=ref,
            )
            self.store.set_state("runtime", "token_usage", usage)
            accounting = account_usage(
                usage,
                model=getattr(self, "active_turn_model", None) or self.model,
                provider=self.provider,
                price_card=self.price_card,
            )
            self.store.set_state("runtime", "token_accounting", accounting)
            if usage["turn_id"]:
                self.store.set_state("token_usage", usage["turn_id"], usage)
                self.store.set_state("token_accounting", usage["turn_id"], accounting)
            payload["token_usage"] = usage
            payload["token_accounting"] = accounting
        self.store.event("codex." + method, payload, correlation, source="codex_app_server")

    def run_turn(self, prompt, *, images=()):
        with self.projection_lock:
            if self.state == "running":
                raise ValueError("turn_already_active")
            if self.state in ("disconnected", "unknown", "closed"):
                raise ValueError("session_not_connected")
            prompt = self.presentation.present_text(prompt)
            presented_images = self.presentation.present(list(images)) if images else []
            previous = self.state, self.turn_id, self.turn_result, getattr(self, "active_turn_model", self.model)
            revision = self.turn_revision
            turn_model, turn_effort = self.model, self.effort
            self.active_turn_model = turn_model
            self.turn_done.clear()
            self.turn_id = None
            self.turn_result = None
            self.state = "running"
        try:
            result = self.client.turn_start(
                self.thread_id,
                [dict(type="text", text=prompt), *presented_images],
                dict(model=turn_model, effort=turn_effort),
            )
        except JsonRpcError as error:
            with self.projection_lock:
                restored = self.state == "running" and self.turn_id is None and self.turn_revision == revision
                if restored:
                    self.state, self.turn_id, self.turn_result, self.active_turn_model = previous
                    self.turn_done.set()
                self.store.event(
                    "codex.turn_start_rejected",
                    dict(code=error.code, message=error.message, state_restored=restored),
                    {"thread_id": self.thread_id},
                )
            raise
        except Exception as error:
            if self.state != "disconnected":
                self._disconnected(error)
            raise
        with self.projection_lock:
            if self.turn_id is None:
                self.turn_id = result.turn.id
        return result.turn.id

    def steer(self, text):
        """Add operator text to the ACTIVE turn (native ``turn/steer``)."""
        with self.projection_lock:
            if self.state != "running" or not self.turn_id:
                raise ValueError("no_active_turn_to_steer")
            turn_id = self.turn_id
        text = self.presentation.present_text(text)
        result = self.client.turn_steer(self.thread_id, turn_id, [dict(type="text", text=text)])
        self.store.event(
            "codex.turn_steered",
            {"turn_id": turn_id, "chars": len(text)},
            {"thread_id": self.thread_id, "turn_id": turn_id},
        )
        return result.turn_id

    def owned_processes(self):
        client = getattr(self, "client", None)
        processes = (("codex_bridge", getattr(client, "_proc", None)), ("codex", getattr(self, "process", None)))
        return [(role, process) for role, process in processes if process is not None]

    def terminal_attach_spec(self):
        executable = getattr(self, "cli_executable", None)
        if executable is None:
            return None
        return dict(
            cwd=self.actor_dir,
            command_prefix=self.command_prefix
            + [
                str(executable),
                "-c",
                "model=" + json.dumps(self.model),
                "-c",
                "model_reasoning_effort=" + json.dumps(self.effort),
            ],
            env=self.env,
            attach_uri=f"unix://{self.socket_path}",
            thread_id=self.thread_id,
        )

    def evidence_agent(self, **options):
        from ..factory import EVIDENCE_CLIENT_RELAY, evidence_client

        if evidence_client(getattr(self, "config", None), "codex") == EVIDENCE_CLIENT_RELAY:
            from ..evidence import RelayEvidenceClient

            return RelayEvidenceClient(self, **options)
        from ...session.codex_evidence_agent import CodexEvidenceAgent

        return CodexEvidenceAgent(self, **options)

    def provenance(self):
        from ..factory import evidence_client

        executable = getattr(self, "cli_executable", None)
        capabilities = getattr(self, "capabilities", None) or CodexSession.capabilities
        block = dict(
            kind="codex",
            sdk=CODEX_SDK_VERSION,
            cli=dict(
                version=CODEX_CLI_VERSION,
                executable=str(executable) if executable is not None else None,
                sha256=None,
            ),
            wire_api="openai_responses",
            upstream_provider=getattr(self, "provider", None),
            model_configured=getattr(self, "model", None),
            effort=getattr(self, "effort", None),
            native_effort=dict(model_reasoning_effort=getattr(self, "effort", None)),
            isolation=getattr(self, "isolation", None),
            counting=MODEL_COMPLETION_COUNTING,
            context_delivery=getattr(self, "context_delivery", CodexSession.context_delivery),
            tool_transport=dict(
                kind="codex_dynamic_tools",
                tool_name_prefix="",
                deferred_tools=("native_namespace" if getattr(self, "deferred_tools", None) else "none"),
                deferred_namespace=DYNAMIC_TOOL_NAMESPACE,
            ),
            prompt_channel=dict(instructions_channel="developer_instructions", native_base_prompt="preserved"),
            native_settings=dict(
                config=list(BOUNDARY_CONFIG_LINES),
                env=({} if getattr(self, "sandbox_enabled", False) else {BOUNDARY_ENV_NAME: "<run>"}),
            ),
            evidence_client=evidence_client(getattr(self, "config", None), "codex"),
            capabilities=capabilities.to_dict(),
        )
        profile = getattr(self, "upstream_profile", None)
        if profile is not None:
            block["upstream_profile"] = dict(profile)
        return block

    def interrupt(self):
        self.expire_requests("turn_interrupted")
        if self.model_proxy:
            self.model_proxy.cancel_pending("turn_interrupted")
        if self.state == "running" and self.turn_id:
            try:
                self.client.turn_interrupt(self.thread_id, self.turn_id)
            except JsonRpcError as error:
                if error.code != -32600 or "no active turn to interrupt" not in str(error.message).lower():
                    raise

    def close(self):
        self.closing = True
        report = self.close_report = dict(threads_alive=[], model_listing_pending=self._close_model_catalogue())
        try:
            self.expire_requests("session_closed")
        except Exception:
            pass
        if self.account_control:
            self.account_control.close()
        if self.model_proxy:
            self.model_proxy.close()
        if self.client:
            client = self.client
            sdk_threads = (
                ("codex-reader", getattr(client, "_reader_thread", None)),
                ("codex-stderr", getattr(client, "_stderr_thread", None)),
            )
            try:
                proc = getattr(client, "_proc", None)
                if proc and proc.poll() is None:
                    try:
                        proc.terminate()
                    except ProcessLookupError:
                        pass
                try:
                    client.close()
                except (BrokenPipeError, OSError):
                    pass
                report["threads_alive"] = [
                    name
                    for name, thread in sdk_threads
                    if thread is not None and thread is not threading.current_thread() and thread.is_alive()
                ]
            finally:
                if hasattr(client, "server_pool"):
                    client.server_pool.shutdown(wait=True, cancel_futures=True)
                self.client = None
        if report["threads_alive"] or report["model_listing_pending"]:
            from ....core.storage.store import StoreClosed

            try:
                self.store.event("codex.session_close_incomplete", dict(report), source="agent_backend")
            except (StoreClosed, Exception):
                pass
        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                try:
                    self.process.kill()
                except ProcessLookupError:
                    pass
                self.process.wait()
        if self.sandbox:
            self.sandbox.close()
        if hasattr(self, "log") and self.log is not None:
            try:
                self.log.close()
            except OSError:
                pass
        if hasattr(self, "daemon_socket"):
            self.daemon_socket.unlink(missing_ok=True)
        if self.ipc_prepared:
            self.transport.close()
        self.state = "closed"
        self.turn_done.set()
