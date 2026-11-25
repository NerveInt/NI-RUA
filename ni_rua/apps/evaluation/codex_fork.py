"""Codex-only ``thread/fork`` and rollout JSONL surgery for checkpoint slices."""

from copy import deepcopy
import json
from pathlib import Path
import subprocess
import time

from ...core.contracts import canonical, digest, new_id
from ...core.storage.store import open_read_store
from ...cortex.agents.base import read_agent_identity

try:
    from openai_codex import CodexConfig
    from ...cortex.agents.codex.bridge import build_bridge_command
    from ...cortex.agents.codex.session import (
        CODEX_CLI_VERSION,
        CODEX_SDK_VERSION,
        DYNAMIC_TOOL_NAMESPACE,
        MODEL_PROXY_PROVIDER,
        CodexSession,
        RecordedClient,
        _native_dynamic_tools,
        require_codex_version,
    )
except ModuleNotFoundError:
    CodexConfig = None
    build_bridge_command = None
    CodexSession = object
    MODEL_PROXY_PROVIDER = CODEX_SDK_VERSION = CODEX_CLI_VERSION = None
    DYNAMIC_TOOL_NAMESPACE = None
    RecordedClient = None
    _native_dynamic_tools = None
    require_codex_version = None


CODEX_0160_SCHEMA_BYTE_LIMIT = 5000


def _normalize_codex_0160_schema(node):
    """按固定版 Codex CLI ``0.160.0`` 规则规范化动态工具 JSON Schema。"""
    if not isinstance(node, dict):
        return {}
    out = {}
    kind = node.get("type")
    has_obj = any(key in node for key in ("properties", "required", "additionalProperties"))
    if kind is None and has_obj:
        kind = "object"
    enum_values = None
    if "enum" in node and isinstance(node["enum"], list):
        enum_values = deepcopy(node["enum"])
    elif "const" in node:
        enum_values = [deepcopy(node["const"])]
    if kind is None and enum_values is not None and all(isinstance(val, str) for val in enum_values):
        kind = "string"
    if kind is not None:
        out["type"] = deepcopy(kind)
    if isinstance(node.get("description"), str):
        out["description"] = node["description"]
    if enum_values is not None:
        out["enum"] = enum_values
    if "minItems" in node and type(node["minItems"]) is int:
        out["minItems"] = node["minItems"]
    is_array = kind == "array" or (isinstance(kind, list) and "array" in kind)
    if "items" in node and isinstance(node["items"], dict):
        out["items"] = _normalize_codex_0160_schema(node["items"])
    elif is_array:
        # 0.160.0 忽略 prefixItems，缺少 items 的数组默认补齐为 {"type": "string"}
        out["items"] = {"type": "string"}
    is_obj = kind == "object" or (isinstance(kind, list) and "object" in kind) or has_obj
    if is_obj:
        props = node.get("properties")
        out["properties"] = (
            {key: _normalize_codex_0160_schema(val) for key, val in props.items()}
            if isinstance(props, dict)
            else {}
        )
    if "required" in node and isinstance(node["required"], list):
        out["required"] = list(node["required"])
    if "additionalProperties" in node:
        additional = node["additionalProperties"]
        if isinstance(additional, bool):
            out["additionalProperties"] = additional
        elif isinstance(additional, dict):
            out["additionalProperties"] = _normalize_codex_0160_schema(additional)
    for combinator in ("allOf", "anyOf", "oneOf"):
        if combinator in node and isinstance(node[combinator], list):
            out[combinator] = [_normalize_codex_0160_schema(branch) for branch in node[combinator]]
    return out


def _strip_schema_descriptions(node):
    if isinstance(node, dict):
        return {key: _strip_schema_descriptions(val) for key, val in node.items() if key != "description"}
    if isinstance(node, list):
        return [_strip_schema_descriptions(item) for item in node]
    return node


def _collapse_deep_containers(node, max_depth, depth=0):
    if not isinstance(node, dict):
        return node
    kind = node.get("type")
    is_container = (
        kind in ("object", "array")
        or (isinstance(kind, list) and any(item in ("object", "array") for item in kind))
        or any(key in node for key in ("properties", "items", "additionalProperties"))
    )
    if depth >= max_depth and is_container:
        return {}
    out = {}
    for key, val in node.items():
        if key == "properties" and isinstance(val, dict):
            out[key] = {name: _collapse_deep_containers(child, max_depth, depth + 1) for name, child in val.items()}
        elif key in ("items", "additionalProperties") and isinstance(val, dict):
            out[key] = _collapse_deep_containers(val, max_depth, depth + 1)
        elif key in ("allOf", "anyOf", "oneOf") and isinstance(val, list):
            out[key] = [_collapse_deep_containers(branch, max_depth, depth + 1) for branch in val]
        else:
            out[key] = val
    return out


def _collapse_combinators(node):
    if not isinstance(node, dict):
        return node
    if any(key in node for key in ("allOf", "anyOf", "oneOf")):
        return {}
    out = {}
    for key, val in node.items():
        if key == "properties" and isinstance(val, dict):
            out[key] = {name: _collapse_combinators(child) for name, child in val.items()}
        elif key in ("items", "additionalProperties") and isinstance(val, dict):
            out[key] = _collapse_combinators(val)
        else:
            out[key] = val
    return out


def measure_codex_0160_dynamic_tool_schema(schema: dict) -> dict:
    """精确复现固定版 Codex CLI ``0.160.0`` 的动态工具 Schema 规范化与 5000 B 分级压缩。

    返回 ``{"schema", "serialized_bytes", "passes_applied", "byte_limit", "within_limit"}``，
    供离线双本体预算门禁与向量结构保真校验调用。
    """
    current = _normalize_codex_0160_schema(schema)
    serialized_bytes = len(canonical(current).encode("utf-8"))
    passes_applied = 0
    if serialized_bytes > CODEX_0160_SCHEMA_BYTE_LIMIT:
        current = _strip_schema_descriptions(current)
        serialized_bytes = len(canonical(current).encode("utf-8"))
        passes_applied = 1
    if serialized_bytes > CODEX_0160_SCHEMA_BYTE_LIMIT:
        current = _collapse_deep_containers(current, 3)
        serialized_bytes = len(canonical(current).encode("utf-8"))
        passes_applied = 2
    if serialized_bytes > CODEX_0160_SCHEMA_BYTE_LIMIT:
        current = _collapse_combinators(current)
        serialized_bytes = len(canonical(current).encode("utf-8"))
        passes_applied = 3
    return {
        "schema": current,
        "serialized_bytes": serialized_bytes,
        "passes_applied": passes_applied,
        "byte_limit": CODEX_0160_SCHEMA_BYTE_LIMIT,
        "within_limit": serialized_bytes <= CODEX_0160_SCHEMA_BYTE_LIMIT,
    }


def require_codex_archive(source_run: str | Path) -> None:
    """Refuse an archive not recorded by the Codex backend (design 04 §2.6).

    Slicing, forked probes and live slices rebuild a Codex rollout and use the native ``thread/fork``; an archive of
    another agent backend (``runtime/agent.backend``) has neither, so it is refused with
    ``ValueError("checkpoint_requires_codex_archive:<backend>")`` before anything is copied. Archives without an
    agent identity (older Codex runs) are accepted.
    """
    source_run = Path(source_run).resolve()
    if not (source_run / "events.sqlite").exists():
        return
    with open_read_store(source_run) as store:
        backend = (read_agent_identity(store) or {}).get("backend", "codex")
    if backend != "codex":
        raise ValueError(f"checkpoint_requires_codex_archive:{backend}")


def find_rollout_path(run_dir: Path) -> Path:
    """定位运行目录内 Codex 原生保存的 rollout JSONL 文件。"""
    sessions_dir = Path(run_dir) / "private" / "codex-home" / "sessions"
    rollouts = sorted(sessions_dir.rglob("rollout-*.jsonl")) if sessions_dir.exists() else []
    if not rollouts:
        raise FileNotFoundError(f"未找到 rollout 文件: {sessions_dir}")
    return rollouts[-1]


def _to_rollout_output(value: object) -> object:
    """将 RPC camelCase 的 contentItems 转换为 Codex rollout JSONL 原生 snake_case 格式。"""
    if isinstance(value, dict) and "contentItems" in value:
        value = value["contentItems"]
    if not isinstance(value, list):
        return value
    items = []
    for item in value:
        if not isinstance(item, dict):
            items.append(item)
            continue
        kind = item.get("type")
        if kind == "inputText":
            items.append({"type": "input_text", "text": item.get("text", "")})
        elif kind == "inputImage":
            items.append({"type": "input_image", "image_url": item.get("imageUrl", "")})
        else:
            items.append(item)
    return items


def slice_rollout_jsonl(
    source_rollout_path: str | Path,
    target_rollout_path: str | Path,
    *,
    until_call_id: str | None = None,
    replacement_output: dict | list | str | None = None,
    dynamic_tools: list | None = None,
    developer_instructions: str | None = None,
    close_open_turn: bool = True,
) -> dict:
    """裁剪 Codex rollout JSONL 到指定 call_id 的输出处，可选替换该切点输出、更新工具契约并闭合半回合。"""
    source_rollout_path = Path(source_rollout_path)
    target_rollout_path = Path(target_rollout_path)
    raw_lines = [json.loads(line) for line in source_rollout_path.read_text().splitlines() if line.strip()]

    cut_idx = len(raw_lines) - 1
    found_call = False
    if until_call_id is not None:
        for idx, entry in enumerate(raw_lines):
            if entry.get("type") != "response_item":
                continue
            payload = entry.get("payload") or {}
            if payload.get("type") in {"function_call_output", "custom_tool_call_output"} and payload.get("call_id") == until_call_id:
                cut_idx = idx
                found_call = True
                if replacement_output is not None:
                    entry = deepcopy(entry)
                    entry["payload"]["output"] = _to_rollout_output(replacement_output)
                    raw_lines[idx] = entry
                break
        if not found_call:
            raise KeyError(f"在 rollout 中未找到目标 call_id 的输出项: {until_call_id}")

    kept = deepcopy(raw_lines[: cut_idx + 1])

    # 若同一批次存在尚未输出结果的并行 function_call，剔除未配对的调用项以保证 /responses 协议完整性
    answered_calls = {
        (e.get("payload") or {}).get("call_id")
        for e in kept
        if e.get("type") == "response_item"
        and (e.get("payload") or {}).get("type") in {"function_call_output", "custom_tool_call_output"}
    }
    filtered = []
    source_thread_id = None
    open_turn_id = None
    open_turn_started_at = 0
    for entry in kept:
        if entry.get("type") == "session_meta":
            meta = entry.get("payload") or {}
            source_thread_id = meta.get("id")
            if dynamic_tools is not None:
                meta["dynamic_tools"] = dynamic_tools
            if developer_instructions is not None:
                meta["base_instructions"] = {"text": developer_instructions}
        elif entry.get("type") == "turn_context" and developer_instructions is not None:
            payload = entry.get("payload") or {}
            if "developer_instructions" in payload:
                payload["developer_instructions"] = developer_instructions
        elif entry.get("type") == "event_msg":
            payload = entry.get("payload") or {}
            ev_kind = payload.get("type")
            if ev_kind == "task_started":
                open_turn_id = payload.get("turn_id")
                open_turn_started_at = int(payload.get("started_at") or 0)
            elif ev_kind in {"task_complete", "turn_aborted"} and payload.get("turn_id") == open_turn_id:
                open_turn_id = None
        elif entry.get("type") == "response_item":
            payload = entry.get("payload") or {}
            if payload.get("type") in {"function_call", "custom_tool_call"} and payload.get("call_id") not in answered_calls:
                continue
        filtered.append(entry)

    # 闭合切点处尚未结束的 turn，防止 Codex app-server 在 thread/fork 时自动注入 <turn_aborted>
    closed_turn = False
    if close_open_turn and open_turn_id is not None and filtered:
        last_ts = filtered[-1].get("timestamp", "2026-10-26T00:00:00Z")
        closing_text = "Observation and tool output recorded."
        msg_id = f"msg_{open_turn_id}"
        filtered.append({
            "timestamp": last_ts,
            "type": "response_item",
            "payload": {
                "type": "message",
                "id": msg_id,
                "role": "assistant",
                "content": [{"type": "output_text", "text": closing_text}],
                "phase": "final_answer",
            },
        })
        filtered.append({
            "timestamp": last_ts,
            "type": "event_msg",
            "payload": {
                "type": "task_complete",
                "turn_id": open_turn_id,
                "last_agent_message": closing_text,
                "started_at": open_turn_started_at,
                "completed_at": open_turn_started_at + 1,
                "duration_ms": 1000,
                "time_to_first_token_ms": 500,
            },
        })
        closed_turn = True

    if any("ordinal" in item for item in filtered):
        for idx, item in enumerate(filtered):
            item["ordinal"] = idx

    target_rollout_path.parent.mkdir(parents=True, exist_ok=True)
    target_rollout_path.write_text("\n".join(json.dumps(item, ensure_ascii=False) for item in filtered) + "\n")
    return {
        "source_thread_id": source_thread_id,
        "source_lines": len(raw_lines),
        "kept_lines": len(filtered),
        "until_call_id": until_call_id,
        "replaced_output": replacement_output is not None,
        "closed_open_turn": closed_turn,
        "target_rollout_path": str(target_rollout_path),
    }


def prepare_codex_rollout_slice(
    source_run: str | Path,
    target_run: str | Path,
    *,
    until_call_id: str | None = None,
    replacement_output: dict | list | str | None = None,
    dynamic_tools: list | None = None,
    developer_instructions: str | None = None,
    close_open_turn: bool = True,
) -> dict:
    """定位源运行目录的 Codex rollout JSONL 并将其裁剪写入目标切片目录的 ``codex-home``。"""
    source_run = Path(source_run).resolve()
    target_run = Path(target_run).resolve()
    src_rollout = find_rollout_path(source_run)
    rel_rollout = src_rollout.relative_to(source_run / "private" / "codex-home")
    dst_rollout = target_run / "private" / "codex-home" / rel_rollout
    return slice_rollout_jsonl(
        src_rollout,
        dst_rollout,
        until_call_id=until_call_id,
        replacement_output=replacement_output,
        dynamic_tools=dynamic_tools,
        developer_instructions=developer_instructions,
        close_open_turn=close_open_turn,
    )


class ForkedCodexSession(CodexSession):
    """继承自 CodexSession，支持通过 Codex 原生 `thread/fork` 从切片 rollout 启动独立分支线程。"""

    def start_forked(self, tools, instructions, source_thread_id: str):
        """使用原生 `client.thread_fork` 从 `CODEX_HOME/sessions/` 下的切片 rollout 创建新线程。"""
        if self.process and self.process.poll() is None:
            raise ValueError("session_already_started")
        require_codex_version()
        if self.client is not None:
            self.close()
        self.deferred_tools = (
            frozenset(tool["name"] for tool in tools if tool.get("deferLoading", False))
            if self.supports_search_tool
            else frozenset()
        )
        self.closing = False
        self._reopen_model_catalogue()   # close() above detached the listing read; this object runs again
        with self.pending_lock:
            self.native_connection_id = new_id("native-connection")
            self.pending.clear()
            self.resolved_requests.clear()
        self._stop_previous_owned_processes(None)
        provider_name = self._prepare(None)
        self.provider = provider_name
        self.store.set_state("runtime", "price_card", self.price_card)
        self.log = (self.private / "app-server.log").open("a")
        self.process = subprocess.Popen(
            self.command, env=self.env, stdin=subprocess.DEVNULL,
            stdout=self.log, stderr=self.log, cwd=self.actor_dir,
        )
        self.store.set_state("runtime", "app_server", {"pid": self.process.pid, "socket_uri": f"unix://{self.socket_path}"})
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

        def tool(params):
            if self.native_connection_id != connection_id or self.closing or self.state in {"disconnected", "closed"}:
                raise ValueError("native_tool_call_stale_connection")
            namespaces = {DYNAMIC_TOOL_NAMESPACE} if params.get("tool") in self.deferred_tools else {None}
            if params.get("namespace") not in namespaces:
                raise ValueError("native_tool_call_namespace_mismatch")
            return self.tool_handler(params)

        socket_arg = self.sandbox.host_socket_path if self.sandbox else self.socket_path
        self.client = RecordedClient(
            CodexConfig(
                launch_args_override=build_bridge_command(socket_arg),
                client_name="ni-rua", client_title="NI-RUA", cwd=str(Path(__file__).resolve().parents[3]),
            ),
            self.store, tool, notification, request, disconnected,
            route_notifications=False,
        )
        self.client.start()
        initialized = self.client.initialize()
        # 在 fork 前将最新 dynamic_tools 同步写入 rollout 的 session_meta，使分叉线程立即具备最新工具契约
        rollout_path = find_rollout_path(self.root)
        dyn_tools = self._present_model_request(dict(tools=_native_dynamic_tools(
            tools,
            catalog_profile=self.config.features.tool_catalog_profile,
            supports_search_tool=self.supports_search_tool,
        )))["tools"]
        dev_text = self.presentation.present_text(instructions)
        slice_rollout_jsonl(rollout_path, rollout_path, dynamic_tools=dyn_tools, developer_instructions=dev_text)

        parameters = dict(
            model=self.model,
            modelProvider=MODEL_PROXY_PROVIDER,
            cwd=str(self.actor_dir),
            approvalPolicy="never",
            sandbox="danger-full-access",
            developerInstructions=dev_text,
            config={"model_reasoning_effort": self.effort},
        )
        started = self.client.thread_fork(source_thread_id, parameters)
        self.thread_id = started.thread.id
        if self.model_proxy is not None:
            self.model_proxy.register_thread(self.thread_id, "actor", self.thread_id)
        self.state = "idle"
        self.store.set_state(
            "runtime", "codex",
            dict(
                thread_id=self.thread_id,
                forked_from_thread_id=source_thread_id,
                app_server_instance_id=new_id("appserver"),
                app_server_pid=self.process.pid,
                socket_uri=f"unix://{self.socket_path}",
                model=self.model,
                provider=self.provider,
                effort=self.effort,
                sdk_version=CODEX_SDK_VERSION,
                cli_version=CODEX_CLI_VERSION,
                cli_executable=str(self.cli_executable),
                initialization=initialized.model_dump(by_alias=True, mode="json"),
                isolation=self.isolation,
                tools_digest=digest(tools),
            ),
        )
        self.recover_history()
        return self.thread_id
