"""Backend-neutral JSONL worker protocol with cancellation and optional operator preemption.

All motion-producing adapter operations run sequentially on the caller thread;
only ``cancel`` and explicitly declared ``preemptive_operator_commands`` run off
the main execution queue so safety holds/status reads never wait behind an
in-flight physical trajectory. Owner stdin EOF is a process-exit boundary, never
an action retry.
"""
from __future__ import annotations

from collections.abc import Callable, Collection, Mapping
import json
import os
import queue
import re
import sys
import threading
import traceback
from typing import Any

from ...core.tool_errors import (
    INFRASTRUCTURE_DOMAIN_CODES,
    INFRASTRUCTURE_TYPE_NAMES,
    INTERNAL_BUG_TYPES,
    ToolInputError,
    extract_domain_code,
)

WIRE_PREFIX = "NI_RUA_JSON "
_CAMEL_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_WORKER_INTERNAL_BUG_TYPES: tuple[type[BaseException], ...] = (*INTERNAL_BUG_TYPES, KeyError)


def _camel_to_snake(name: str) -> str:
    cleaned = _CAMEL_RE.sub("_", name or "error").lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned).strip("_")
    return cleaned or "infrastructure_error"


def classify_worker_exception(
    error: BaseException,
    *,
    operation: str | None = None,
    bootstrap: bool = False,
) -> dict[str, Any]:
    """Classify a worker exception into a structured ``error_detail`` envelope."""
    err_type = type(error).__name__
    msg = str(error)
    lower = msg.lower()
    if bootstrap:
        category = "infrastructure_error"
        code = (
            "robot_gpu_out_of_memory"
            if ("out of memory" in lower or "cuda error 2" in lower or err_type == "OutOfMemoryError")
            else "robot_worker_bootstrap_failed"
        )
    elif isinstance(error, ToolInputError):
        category = "input_error"
        code = error.code
    elif isinstance(error, TimeoutError):
        category = "unknown_execution" if operation == "execute" else "infrastructure_error"
        code = "execution_status_unknown" if operation == "execute" else (extract_domain_code(msg) or "worker_operation_timeout")
    elif (
        err_type in INFRASTRUCTURE_TYPE_NAMES
        or isinstance(error, OSError)
        or "out of memory" in lower
        or "cuda error 2" in lower
    ):
        category = "infrastructure_error"
        if "out of memory" in lower or "cuda error 2" in lower or err_type == "OutOfMemoryError":
            code = "robot_gpu_out_of_memory"
        else:
            code = extract_domain_code(msg) or _camel_to_snake(err_type)
    elif isinstance(error, _WORKER_INTERNAL_BUG_TYPES):
        category = "internal_runtime_error"
        code = "internal_runtime_error"
    elif isinstance(error, ValueError):
        domain_code = extract_domain_code(msg)
        if domain_code in INFRASTRUCTURE_DOMAIN_CODES:
            category = "infrastructure_error"
            code = domain_code
        elif domain_code is not None:
            category = "input_error"
            code = domain_code
        elif msg == "unsupported operation":
            category = "input_error"
            code = "unsupported_worker_operation"
        elif msg == "seed and layout_id must be nonnegative integers":
            category = "input_error"
            code = "invalid_reset_arguments"
        else:
            category = "internal_runtime_error"
            code = "internal_runtime_error"
    else:
        domain_code = extract_domain_code(msg)
        if domain_code in INFRASTRUCTURE_DOMAIN_CODES:
            category = "infrastructure_error"
            code = domain_code
        else:
            category = "internal_runtime_error"
            code = "internal_runtime_error"

    detail: dict[str, Any] = {
        "type": err_type,
        "module": type(error).__module__,
        "category": category,
        "code": code,
        "message": msg,
        "operation": operation,
        "bootstrap": bootstrap,
    }
    if isinstance(error, ToolInputError):
        detail["field_path"] = error.field_path
        detail["suggestion"] = error.suggestion
        detail["minimal_example"] = error.minimal_example
    return detail


def _error_frame(
    request_id: Any,
    error: BaseException,
    *,
    operation: str | None = None,
    bootstrap: bool = False,
) -> dict[str, Any]:
    err_type = type(error).__name__
    msg = str(error)
    prefix = "bootstrap_failed: " if bootstrap else ""
    return {
        "id": request_id,
        "error": f"{prefix}{err_type}: {msg}",
        "error_detail": classify_worker_exception(error, operation=operation, bootstrap=bootstrap),
    }


def serve_worker(
    adapter_factory: Callable[[Any], Any],
    arguments: Any,
    emit: Callable[[dict[str, Any]], None],
    *,
    preemptive_operator_commands: Collection[str] = (),
    on_eof: Callable[[Any], None] | None = None,
    operations: Mapping[str, Callable[[Any, dict[str, Any], Callable[[dict[str, Any]], None]], Any]] | None = None,
) -> None:
    """Run the single-owner JSONL request loop for a simulator or hardware worker process.

    Parameters:
        adapter_factory: Callable ``(arguments) -> adapter`` invoked once on the main thread.
        arguments: Parsed CLI namespace or config object passed to ``adapter_factory``.
        emit: Thread-safe frame emitter ``emit(message_dict)`` writing ``WIRE_PREFIX + json``.
        preemptive_operator_commands: Collection of ``operator`` command names (e.g., ``"hold"``,
            ``"relax"``) answered immediately off the main queue without waiting on motion.
        on_eof: Optional cleanup hook ``on_eof(adapter)`` invoked when owner stdin closes before
            ``os._exit(0)`` (e.g., engaging a hardware safety ``hold``).
        operations: Optional extra ``{op_name: handler(adapter, request, emit)}`` dispatch entries.
    """
    requests: queue.Queue[dict[str, Any]] = queue.Queue()
    cancellations: dict[str, threading.Event] = {}
    ready = threading.Event()
    lock = threading.Lock()
    holder: dict[str, Any] = {}
    preemptive_set = frozenset(preemptive_operator_commands)
    custom_ops = dict(operations) if operations is not None else {}

    def operator_now(request: dict[str, Any]) -> None:
        try:
            if not ready.wait(timeout=0) or "adapter" not in holder:
                raise RuntimeError("worker_not_ready")
            payload = dict(request.get("payload") or {})
            command = payload.pop("command")
            payload.pop("progress", None)
            emit({"id": request["id"], "result": holder["adapter"].operator(command, **payload)})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit(_error_frame(request["id"], error, operation="operator"))

    def read_input() -> None:
        for line in sys.stdin:
            request: Any = None
            try:
                request = json.loads(line)
                op = request["op"]
                if op == "cancel":
                    with lock:
                        event = cancellations.get(request.get("target"))
                        if event:
                            event.set()
                    emit({"id": request["id"], "result": {"stop_requested": event is not None, "confirmed": False}})
                elif op == "operator" and (request.get("payload") or {}).get("command") in preemptive_set:
                    threading.Thread(target=operator_now, args=(request,), daemon=True).start()
                else:
                    if op == "execute":
                        with lock:
                            cancellations[request["id"]] = threading.Event()
                    requests.put(request)
            except Exception as error:
                traceback.print_exc(file=sys.stderr)
                if isinstance(request, dict) and "id" in request:
                    req_id = request["id"]
                    op_val = request.get("op") if isinstance(request.get("op"), str) else None
                    with lock:
                        cancellations.pop(req_id, None)
                    try:
                        emit(_error_frame(req_id, error, operation=op_val))
                    except Exception:
                        traceback.print_exc(file=sys.stderr)
        with lock:
            for event in cancellations.values():
                event.set()
        adapter = holder.get("adapter")
        if on_eof is not None and adapter is not None:
            try:
                on_eof(adapter)
            except Exception:
                traceback.print_exc(file=sys.stderr)
        os._exit(0)

    threading.Thread(target=read_input, daemon=True).start()
    try:
        holder["adapter"] = adapter = adapter_factory(arguments)
        ready.set()
    except Exception as error:
        traceback.print_exc(file=sys.stderr)
        while not requests.empty():
            request = requests.get_nowait()
            emit(_error_frame(request["id"], error, operation=request.get("op"), bootstrap=True))
        sys.stderr.flush()
        os._exit(1)

    while True:
        request = requests.get()
        request_id, operation = request["id"], request["op"]
        payload = request.get("payload")
        try:
            if operation in custom_ops:
                result = custom_ops[operation](adapter, request, emit)
            elif operation == "ready":
                result = adapter.capabilities()
            elif operation == "observe":
                result = adapter.observe()
            elif operation == "preview_motion" and hasattr(adapter, "preview_motion"):
                result = adapter.preview_motion(payload)
            elif operation == "preview_batch" and hasattr(adapter, "preview_batch"):
                result = adapter.preview_batch(payload if payload is not None else {})
            elif operation == "reach_batch" and hasattr(adapter, "reach_batch"):
                result = adapter.reach_batch(payload if payload is not None else {})
            elif operation == "reset" and hasattr(adapter, "reset"):
                result = adapter.reset(**(payload or {}))
            elif operation == "evaluate" and hasattr(adapter, "evaluate"):
                result = adapter.evaluate(**(payload or {}))
            elif operation == "observer_snapshot" and hasattr(adapter, "observer_snapshot"):
                result = adapter.observer_snapshot()
            elif operation == "record":
                if hasattr(adapter, "record"):
                    result = adapter.record(**(payload or {}))
                elif hasattr(adapter, "motion_video"):
                    result = adapter.motion_video.record(**(payload or {}))
                else:
                    raise ValueError("unsupported operation")
            elif operation == "save_video":
                if hasattr(adapter, "save_video"):
                    result = adapter.save_video()
                elif hasattr(adapter, "motion_video"):
                    result = adapter.motion_video.save()
                else:
                    raise ValueError("unsupported operation")
            elif operation == "fast_forward_slice" and hasattr(adapter, "fast_forward_slice"):
                result = adapter.fast_forward_slice(payload)
            elif operation == "operator" and hasattr(adapter, "operator"):
                op_payload = dict(payload or {})
                op_payload.pop("progress", None)
                result = adapter.operator(
                    op_payload.pop("command"),
                    progress=lambda update: emit({"id": request_id, "progress": update}),
                    **op_payload,
                )
            elif operation == "execute":
                with lock:
                    cancellation = cancellations.setdefault(request_id, threading.Event())
                try:
                    result = adapter.execute(
                        payload,
                        lambda update: emit({"id": request_id, "progress": update}),
                        cancellation,
                    )
                finally:
                    with lock:
                        cancellations.pop(request_id, None)
            elif operation == "close":
                adapter.close(lambda update: emit({"id": request_id, "progress": update}))
                emit({"id": request_id, "result": {"closed": True}})
                sys.stderr.flush()
                os._exit(0)
            else:
                raise ValueError("unsupported operation")
            emit({"id": request_id, "result": result})
        except Exception as error:
            traceback.print_exc(file=sys.stderr)
            emit(_error_frame(request_id, error, operation=operation))
