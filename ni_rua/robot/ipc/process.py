"""Backend-neutral JSONL worker subprocess lifecycle, cancellation, and exit evidence.

This module owns subprocess supervision, generation-scoped stdout framing,
non-blocking stdin writes with strict deadlines, and graceful/forced shutdown
bookkeeping. Backend-specific native failure classification (such as CUDA/Warp
OOM log signatures on GPU simulators) plugs in via ``log_failure_classifier``
and ``reply_failure_classifier`` without coupling hardware backends to simulator
log rules.
"""
from __future__ import annotations

from collections.abc import Callable
import json
import os
from pathlib import Path
import select
import subprocess
import threading
import time
from typing import Any
import uuid

from ...core.tool_errors import (
    INFRASTRUCTURE_DOMAIN_CODES,
    WorkerRemoteError,
    extract_domain_code,
)
from .protocol import WIRE_PREFIX

_MAX_ORPHANED = 128


def _remote_error_from_wire(
    wire_error: str,
    *,
    detail: dict[str, Any] | None = None,
    operation: str | None = None,
) -> WorkerRemoteError:
    if isinstance(detail, dict):
        category = detail.get("category") or "internal_runtime_error"
        code = detail.get("code") or "internal_runtime_error"
        if category == "input_error" and code in INFRASTRUCTURE_DOMAIN_CODES:
            category = "infrastructure_error"
        return WorkerRemoteError(
            wire_error,
            remote_type=str(detail.get("type") or "RuntimeError"),
            remote_module=detail.get("module"),
            category=category,
            code=str(code),
            remote_message=str(detail.get("message") or ""),
            operation=detail.get("operation") or operation,
            bootstrap=bool(detail.get("bootstrap", False)),
            physics_advanced=detail.get("physics_advanced"),
            executed_steps=detail.get("executed_steps"),
            field_path=str(detail.get("field_path") or ""),
            suggestion=detail.get("suggestion"),
            minimal_example=detail.get("minimal_example"),
        )
    lower = wire_error.lower()
    if wire_error.startswith("bootstrap_failed"):
        code = "robot_gpu_out_of_memory" if ("out of memory" in lower or "cuda error 2" in lower) else "robot_worker_bootstrap_failed"
        return WorkerRemoteError(
            wire_error,
            remote_type="RuntimeError",
            category="infrastructure_error",
            code=code,
            remote_message=wire_error.removeprefix("bootstrap_failed:").strip(),
            operation=operation,
            bootstrap=True,
        )
    if "out of memory" in lower or "cuda error 2" in lower:
        return WorkerRemoteError(
            wire_error,
            remote_type="OutOfMemoryError",
            category="infrastructure_error",
            code="robot_gpu_out_of_memory",
            remote_message=wire_error,
            operation=operation,
        )
    if "robot status unknown" in lower:
        return WorkerRemoteError(
            wire_error,
            remote_type="TimeoutError",
            category="unknown_execution",
            code="execution_status_unknown",
            remote_message=wire_error,
            operation=operation,
        )
    if wire_error.startswith("robot_worker_disconnected") or "robot worker disconnected" in lower:
        return WorkerRemoteError(
            wire_error,
            remote_type="WorkerDisconnected",
            category="unknown_execution" if operation == "execute" else "infrastructure_error",
            code="execution_status_unknown" if operation == "execute" else "robot_worker_disconnected",
            remote_message=wire_error,
            operation=operation,
        )
    head, sep, tail = wire_error.partition(":")
    head_clean = head.strip()
    tail_clean = tail.strip()
    if sep and head_clean == "ValueError":
        domain = extract_domain_code(tail_clean)
        if domain in INFRASTRUCTURE_DOMAIN_CODES:
            return WorkerRemoteError(
                wire_error,
                remote_type="ValueError",
                category="infrastructure_error",
                code=domain,
                remote_message=tail_clean,
                operation=operation,
            )
        if domain is not None or tail_clean == "unsupported operation":
            return WorkerRemoteError(
                wire_error,
                remote_type="ValueError",
                category="input_error",
                code=domain or "unsupported_worker_operation",
                remote_message=tail_clean,
                operation=operation,
            )
    domain = extract_domain_code(wire_error)
    if domain in INFRASTRUCTURE_DOMAIN_CODES:
        return WorkerRemoteError(
            wire_error,
            remote_type=head_clean if sep else "RuntimeError",
            category="infrastructure_error",
            code=domain,
            remote_message=wire_error,
            operation=operation,
        )
    return WorkerRemoteError(
        wire_error,
        remote_type=head_clean if sep and head_clean.isidentifier() else "RuntimeError",
        category="internal_runtime_error",
        code="internal_runtime_error",
        remote_message=tail_clean if sep else wire_error,
        operation=operation,
    )


class WorkerProcess:
    """Supervise one long-lived JSONL child worker process."""

    _MAX_ORPHANED = _MAX_ORPHANED
    log_failure_classifier: Callable[[str], str | None] | None = None
    reply_failure_classifier: Callable[[dict[str, Any]], str | None] | None = None
    on_late_reply: Callable[[dict[str, Any], dict[str, Any]], None] | None = None

    def __init__(
        self,
        run_dir: str | Path,
        *,
        log_failure_classifier: Callable[[str], str | None] | None = None,
        reply_failure_classifier: Callable[[dict[str, Any]], str | None] | None = None,
        on_late_reply: Callable[[dict[str, Any], dict[str, Any]], None] | None = None,
    ) -> None:
        self.run_dir = Path(run_dir).resolve()
        if log_failure_classifier is not None:
            self.log_failure_classifier = log_failure_classifier
        if reply_failure_classifier is not None:
            self.reply_failure_classifier = reply_failure_classifier
        if on_late_reply is not None:
            self.on_late_reply = on_late_reply
        self.process: subprocess.Popen[str] | None = None
        self.pending: dict[str, tuple[threading.Event, dict[str, Any], Callable[[Any], None] | None]] = {}
        self.orphaned: dict[str, dict[str, Any]] = {}
        self.lock = threading.Lock()
        self.write_lock = threading.Lock()
        self.operation_lock = threading.Lock()
        self.failure: str | None = None
        self.failure_log_path: Path = self.run_dir / "robot-worker.log"
        self.capabilities: dict[str, Any] | None = None
        self.reader: threading.Thread | None = None
        self.log_start = 0
        self._process_generation = 0
        self.last_activity_at = time.monotonic()

    def start_process(self, command: list[str], *, cwd: str | Path, environment: dict[str, str]) -> dict[str, Any]:
        if self.process is not None and self.capabilities is not None and self.process.poll() is None:
            return self.capabilities
        if self.process is not None:
            cleanup = self.close()
            if cleanup.get("process_exit_confirmed") is not True or cleanup.get("reader_exit_confirmed", True) is not True:
                raise RuntimeError("previous_worker_exit_unconfirmed")
            self.process, self.failure, self.capabilities = None, None, None
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.failure_log_path = self.run_dir / "robot-worker.log"
        self.log = self.failure_log_path.open("a", buffering=1)
        self.log_start = self.log.tell()
        self.startup_forced_termination = None
        try:
            self.process = subprocess.Popen(
                command,
                cwd=cwd,
                env=environment,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=self.log,
                text=True,
                bufsize=1,
            )
        except BaseException:
            self.log.close()
            raise
        with self.lock:
            self.orphaned.clear()
        self._process_generation += 1
        self.close_report = None
        # Capture identity while this Popen still owns the live child. Never
        # infer its start time later from a potentially reused PID during cleanup.
        try:
            fields = Path(f"/proc/{self.process.pid}/stat").read_text().rsplit(")", 1)[1].split()
            self.process_start_ticks: int | None = int(fields[19])
        except (OSError, ValueError, IndexError):
            self.process_start_ticks = None
        self.reader = threading.Thread(
            target=self._read,
            args=(self.process, self.log, self._process_generation),
            daemon=True,
        )
        try:
            self.reader.start()
            self.capabilities = self._request("ready", timeout=600)
        except BaseException:
            cleanup = self.close()
            if cleanup.get("process_exit_confirmed") is True and cleanup.get("reader_exit_confirmed", True) is True:
                self.process = None
                self.capabilities = None
                self.failure = None
            raise
        return self.capabilities

    def _read(self, process=None, log=None, generation=None) -> None:
        process = self.process if process is None else process
        log = self.log if log is None else log
        generation = self._process_generation if generation is None else generation

        def owns_generation() -> bool:
            return self.process is process and self._process_generation == generation

        progress_seen: dict[Any, dict[str, Any]] = {}
        for line in process.stdout:
            if not owns_generation():
                return
            if not line.startswith(WIRE_PREFIX):
                log.write(line)
                continue
            try:
                message = json.loads(line[len(WIRE_PREFIX):])
            except ValueError:
                log.write(line)
                continue
            req_id = message.get("id")
            orphan = None
            is_progress = "progress" in message
            with self.lock:
                entry = self.pending.get(req_id)
                if entry is not None and not is_progress:
                    self.last_activity_at = time.monotonic()
                    entry[1].update(message)
                    entry[0].set()
                elif entry is None and not is_progress and isinstance(req_id, str):
                    orphan = self.orphaned.pop(req_id, None)
            if entry is None:
                if orphan is not None:
                    self.last_activity_at = time.monotonic()
                    reply_failure = self._classify_reply(message)
                    if reply_failure:
                        self._fail(reply_failure)
                    callback = self.on_late_reply
                    if callable(callback):
                        try:
                            callback(orphan, message)
                        except Exception as error:
                            if log is not None:
                                log.write(f"late reply callback failed: {error}\n")
                continue
            if is_progress:
                progress = message["progress"]
                if isinstance(progress, dict):
                    # Only worker execution milestones count, never log/heartbeat traffic.
                    milestone = {
                        key: progress[key]
                        for key in ("phase", "waypoint_index", "executed_steps", "sim_step", "arm_context")
                        if key in progress
                    }
                    if milestone and milestone != progress_seen.get(message.get("id")):
                        progress_seen[message.get("id")] = milestone
                        self.last_activity_at = time.monotonic()
                if entry[2]:
                    try:
                        entry[2](message["progress"])
                    except Exception as error:
                        log.write(f"progress callback failed: {error}\n")
            else:
                progress_seen.pop(req_id, None)
        if owns_generation():
            self._check_native_failure()
            self._fail(f"robot_worker_disconnected: exit={process.poll()}")

    def _fail(self, reason: str) -> None:
        with self.lock:
            self.failure = self.failure or reason
            for event, result, _ in self.pending.values():
                if not event.is_set():
                    result.setdefault("error", self.failure)
                    event.set()

    def _classify_log_line(self, line: str) -> str | None:
        classifier = self.log_failure_classifier
        return classifier(line) if classifier is not None else None

    def _classify_reply(self, result: dict[str, Any]) -> str | None:
        classifier = self.reply_failure_classifier
        return classifier(result) if classifier is not None else None

    def _check_native_failure(self) -> None:
        if self.log_failure_classifier is None and type(self)._classify_log_line is WorkerProcess._classify_log_line:
            return
        try:
            with (self.run_dir / "robot-worker.log").open("rb") as stream:
                end = stream.seek(0, os.SEEK_END)
                stream.seek(max(self.log_start, end - 1024 * 1024))
                lines = stream.read(1024 * 1024).decode("utf-8", errors="replace").splitlines()
        except OSError:
            return
        for line in lines:
            classified = self._classify_log_line(line)
            if classified:
                self._fail(classified)
                return

    def _send(self, message: dict[str, Any], deadline: float) -> None:
        data = memoryview((json.dumps(message, allow_nan=False) + "\n").encode())
        acquired = False
        try:
            acquired = self.write_lock.acquire(timeout=max(0, deadline - time.monotonic()))
            if not acquired:
                raise TimeoutError("robot status unknown: worker request write deadline; do not replay action")
            descriptor = self.process.stdin.fileno()  # type: ignore[union-attr]
            os.set_blocking(descriptor, False)
            while data:
                try:
                    data = data[os.write(descriptor, data):]
                except BlockingIOError:
                    pass
                if data:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0 or not select.select([], [descriptor], [], remaining)[1]:
                        raise TimeoutError("robot status unknown: worker request write deadline; do not replay action")
        except (TimeoutError, OSError, ValueError) as error:
            self._fail(str(error))
            raise
        finally:
            if acquired:
                self.write_lock.release()

    def _request(self, operation: str, payload: Any = None, timeout: float = 180, progress=None, cancel=None) -> Any:
        deadline = time.monotonic() + timeout
        request_id = str(uuid.uuid4())
        done, result = threading.Event(), {}
        with self.lock:
            if self.failure:
                raise _remote_error_from_wire(self.failure, operation=operation)
            if self.process is None:
                raise WorkerRemoteError(
                    "robot_worker_not_running",
                    remote_type="RuntimeError",
                    category="infrastructure_error",
                    code="robot_worker_not_running",
                    remote_message="robot_worker_not_running",
                    operation=operation,
                )
            self.pending[request_id] = (done, result, progress)
        timed_out = False
        try:
            self._send({"id": request_id, "op": operation, "payload": payload}, deadline)
            cancellation_sent = False
            next_log_check = time.monotonic() + 1
            while not done.wait(min(0.1, max(0, deadline - time.monotonic()))):
                if time.monotonic() >= next_log_check:
                    self._check_native_failure()
                    next_log_check = time.monotonic() + 1
                if cancel is not None and cancel.is_set() and not cancellation_sent:
                    try:
                        self._send({"id": str(uuid.uuid4()), "op": "cancel", "target": request_id}, deadline)
                    except (TimeoutError, OSError, ValueError):
                        pass
                    cancellation_sent = True
                if time.monotonic() >= deadline:
                    if not cancellation_sent:
                        try:
                            self._send({"id": str(uuid.uuid4()), "op": "cancel", "target": request_id}, deadline)
                        except (TimeoutError, OSError, ValueError):
                            pass
                    timed_out = True
                    raise TimeoutError("robot status unknown: worker response deadline; do not replay action")
            self._check_native_failure()
            reply_failure = self._classify_reply(result)
            if reply_failure:
                self._fail(reply_failure)
            if (self.failure or "").startswith("robot_gpu_out_of_memory") or (reply_failure and self.failure):
                oom_text = self.failure or reply_failure or "robot_gpu_out_of_memory"
                raise WorkerRemoteError(
                    oom_text,
                    remote_type="OutOfMemoryError",
                    category="infrastructure_error",
                    code="robot_gpu_out_of_memory",
                    remote_message=oom_text,
                    operation=operation,
                )
            if "error" in result:
                detail = result.get("error_detail") if isinstance(result.get("error_detail"), dict) else None
                wire_text = result["error"] if detail is not None else (self.failure or result["error"])
                raise _remote_error_from_wire(wire_text, detail=detail, operation=operation)
            return result["result"]
        finally:
            late_message = None
            orphan_record = None
            with self.lock:
                popped = self.pending.pop(request_id, None)
                if timed_out and popped is not None and not self.failure:
                    orphan_record = {
                        "request_id": request_id,
                        "method": operation,
                        "op": operation,
                        "params": payload,
                        "payload": payload,
                        "timed_out_at_ns": time.monotonic_ns(),
                    }
                    if done.is_set() and "id" in result:
                        late_message = dict(result)
                    else:
                        while len(self.orphaned) >= self._MAX_ORPHANED:
                            self.orphaned.pop(next(iter(self.orphaned)))
                        self.orphaned[request_id] = orphan_record
            if late_message is not None and orphan_record is not None:
                callback = self.on_late_reply
                if callable(callback):
                    try:
                        callback(orphan_record, late_message)
                    except Exception as error:
                        if getattr(self, "log", None) is not None:
                            try:
                                self.log.write(f"late reply callback failed: {error}\n")
                            except Exception:
                                pass

    def close(self) -> dict[str, Any]:
        if self.process is None:
            with self.lock:
                self.orphaned.clear()
            return {
                "native_close_acknowledged": False,
                "forced_termination": None,
                "process_exit_confirmed": True,
                "reader_exit_confirmed": True,
                "not_started": True,
                "errors": [],
            }
        previous = getattr(self, "close_report", None)
        if (
            previous
            and previous.get("pid") == self.process.pid
            and previous["process_exit_confirmed"]
            and previous.get("reader_exit_confirmed") is True
        ):
            return previous
        report: dict[str, Any] = {
            "pid": self.process.pid,
            "start_ticks": getattr(self, "process_start_ticks", None),
            "native_close_acknowledged": False,
            "environment_close_completed": False,
            "application_close_started": False,
            "application_close_completed": False,
            "application_fast_shutdown": None,
            "forced_termination": getattr(self, "startup_forced_termination", None),
            "process_exit_confirmed": False,
            "reader_exit_confirmed": False,
            "errors": [],
        }
        self.close_report = report

        def close_progress(value: dict[str, Any]) -> None:
            if value.get("phase") == "environment_closed":
                report["environment_close_completed"] = True
            elif value.get("phase") == "application_close_started":
                report["application_close_started"] = True
                report["application_fast_shutdown"] = value.get("fast_shutdown")
            elif value.get("phase") == "application_closed":
                report["application_close_completed"] = True

        try:
            if self.process.poll() is None:
                if self.failure:
                    raise RuntimeError(self.failure)
                result = self._request("close", timeout=180, progress=close_progress)
                report["native_close_acknowledged"] = result.get("closed") is True
                if not report["native_close_acknowledged"]:
                    raise RuntimeError("native_close_acknowledgement_missing")
                report.update(
                    environment_close_completed=True,
                    application_close_started=True,
                    application_close_completed=True,
                )
            self.process.wait(timeout=30)
        except (RuntimeError, TimeoutError, subprocess.TimeoutExpired, OSError) as error:
            report["errors"].append(f"{type(error).__name__}: {error}")
            try:
                self.process.wait(timeout=2)
            except subprocess.TimeoutExpired:
                try:
                    if self.process.poll() is None:
                        self.process.terminate()
                        report["forced_termination"] = "SIGTERM"
                    self.process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        if self.process.poll() is None:
                            self.process.kill()
                            report["forced_termination"] = "SIGKILL"
                        self.process.wait(timeout=10)
                    except (subprocess.TimeoutExpired, OSError) as error:
                        report["errors"].append(f"{type(error).__name__}: {error}")
                except OSError as error:
                    report["errors"].append(f"{type(error).__name__}: {error}")
            except OSError as error:
                report["errors"].append(f"{type(error).__name__}: {error}")
        finally:
            orphaned = getattr(self, "orphaned", None)
            lock = getattr(self, "lock", None)
            if isinstance(orphaned, dict):
                if lock is not None:
                    with lock:
                        orphaned.clear()
                else:
                    orphaned.clear()
            report["process_exit_confirmed"] = self.process.poll() is not None
            report["returncode"] = self.process.returncode
            report["exit_mode"] = (
                "unconfirmed"
                if not report["process_exit_confirmed"]
                else "after_forced_signal"
                if report["forced_termination"]
                else "after_native_close_ack"
                if report["native_close_acknowledged"]
                else "self_exit_without_close_ack"
            )
            if report["process_exit_confirmed"]:
                if self.reader is not None:
                    self.reader.join(timeout=5)
                report["reader_exit_confirmed"] = self.reader is None or not self.reader.is_alive()
                if report["reader_exit_confirmed"]:
                    for stream in (self.process.stdin, self.process.stdout):
                        try:
                            if stream is not None:
                                stream.close()
                        except OSError as error:
                            report["errors"].append(f"{type(error).__name__}: {error}")
                    try:
                        self.log.close()
                    except OSError as error:
                        report["errors"].append(f"{type(error).__name__}: {error}")
                else:
                    report["errors"].append("robot_reader_stop_unconfirmed")
        return report
