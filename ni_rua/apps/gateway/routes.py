"""Core HTTP, SSE, and WebSocket routes of one workbench, plus capability-extension mounting."""
from __future__ import annotations

import asyncio
import base64
from functools import wraps
import json

from aiohttp import ClientError, WSMsgType, web
try:
    from openai_codex.errors import CodexError, JsonRpcError, ServerBusyError
except ModuleNotFoundError:
    class CodexError(Exception):
        pass

    class JsonRpcError(CodexError):
        pass

    class ServerBusyError(CodexError):
        pass

from ...core.contracts import IDENTITY, canonical, digest
from ...core.storage.read_models import artifact_response, record_updates
from ...core.storage.store import StoreClosed
from ...cortex.agents.base import AgentRejected, AgentTransportUnknown
from ...perception.scene.scene_service import TERMINAL as JOB_TERMINAL
from ...robot.contracts import ClockDomain, EvaluatorBackend, OperatorBackend, VideoBackend
from ...robot.model.embodiments import ARX_X5
from ..library.static_assets import static_directory
from .extensions.operator_console import LEASE_BUSY_CODE, LEASE_COMMANDS, PREEMPTIVE_COMMANDS
from .read_router import RunReadRouter


class GatewayRouter:
    """Core gateway router composed into :class:`Workbench` alongside capability extensions."""

    remote = None

    def __init__(self, workbench=None):
        self.workbench = workbench

    def _owner(self):
        return self.workbench if self.workbench is not None else self

    def _is_operator_console_home(self):
        wb = self._owner()
        if wb.replay:
            return False
        robot = wb.robot
        has_operator = isinstance(robot, OperatorBackend) or callable(getattr(robot, "operator", None))
        is_evaluator = isinstance(robot, EvaluatorBackend) or callable(getattr(robot, "evaluate", None))
        if not has_operator or is_evaluator:
            return False
        if callable(getattr(robot, "embodiment_profile", None)):
            return wb._embodiment().clock == ClockDomain.REAL_TIME
        model_id = (wb.harness.capabilities or {}).get("robot_model_id") or wb.config.backend.robot_model_id
        return bool(model_id) and model_id != ARX_X5.robot_model_id

    async def control_dispatch(self, value, origin):
        """One control request: idempotent by request_id, serialized by the control (or safety, or lease) lock."""
        wb = self._owner()
        if not isinstance(value, dict) or not isinstance(value.get("request_id"), str) or not 0 < len(value["request_id"]) <= 256:
            raise ValueError("control_requires_request_id")
        identifier = value["request_id"]
        command = value.get("command")
        if not isinstance(command, str) or not command:
            raise ValueError("control_requires_command")
        lock = wb.safety_lock if command in PREEMPTIVE_COMMANDS else wb.lease_lock if command in LEASE_COMMANDS else wb.control_lock
        async with lock:
            wb.store.save("control_request", identifier, value)
            saved = wb.store.read_state("control_response", identifier)
            if saved:
                return web.json_response(
                    saved,
                    status=saved.get("http_status", 409) if saved.get("status") == "unknown" else saved.get("http_status", 200),
                )
            wb.store.set_state(
                "control_response",
                identifier,
                {
                    "request_id": identifier,
                    "status": "unknown",
                    "reason": "dispatch_acknowledgement_missing_do_not_resend",
                },
            )
            try:
                result = await wb.execute_control(value, origin)
            except (JsonRpcError, AgentRejected) as error:
                uncertain = wb.session.state in ("running", "disconnected", "unknown")
                if uncertain:
                    wb.status = "unknown"
                busy = isinstance(error, ServerBusyError) or getattr(error, "busy", False) is True
                result = dict(
                    request_id=identifier,
                    status="unknown" if uncertain else "rejected",
                    reason=getattr(error, "message", str(error)),
                    native_rpc_code=getattr(error, "code", None),
                    http_status=409 if uncertain else 429 if busy else 400,
                )
                wb.store.event("control.status_unknown" if uncertain else "control.rejected", result)
            except (CodexError, AgentTransportUnknown) as error:
                wb.status = "unknown"
                result = dict(request_id=identifier, status="unknown", reason=str(error), http_status=409)
                wb.store.event("control.status_unknown", result)
            except (RuntimeError, TimeoutError, OSError) as error:
                wb.status = "unknown"
                result = {"request_id": identifier, "status": "unknown", "reason": str(error), "http_status": 409}
                wb.store.event("control.status_unknown", result)
            except (ValueError, KeyError) as error:
                conflict = "conflict" in str(error) or str(error).startswith(LEASE_BUSY_CODE)
                result = {"request_id": identifier, "error": str(error), "http_status": 409 if conflict else 400}
            except Exception as error:
                wb.status = "unknown"
                result = {
                    "request_id": identifier,
                    "status": "unknown",
                    "reason": f"{type(error).__name__}: {error}",
                    "http_status": 500,
                }
                wb.store.event("control.status_unknown", result)
            wb.store.set_state("control_response", identifier, result)
            return web.json_response(result, status=result.get("http_status", 200))

    def app(self):
        wb = self._owner()

        @web.middleware
        async def errors(request, handler):
            try:
                if request.host.split(":")[0] not in ("localhost", "127.0.0.1"):
                    raise web.HTTPForbidden(text="loopback_host_required")
                if request.headers.get("X-Ni-Rua-Run-ID", wb.identity["run_id"]) != wb.identity["run_id"]:
                    raise web.HTTPNotFound(text="run_identity_mismatch")
                if request.match_info.get("run_id", wb.identity["run_id"]) != wb.identity["run_id"]:
                    raise web.HTTPNotFound(text="run_identity_mismatch")
                if request.method == "POST" and request.headers.get("Origin") not in (None, f"http://{request.host}"):
                    raise web.HTTPForbidden(text="same_origin_required")
                return await handler(request)
            except FileNotFoundError:
                return web.json_response({"error": "artifact_original_unavailable"}, status=404)
            except KeyError as error:
                return web.json_response({"error": f"missing:{error}"}, status=404)
            except ValueError as error:
                return web.json_response({"error": str(error)}, status=409 if "conflict" in str(error) else 400)

        app = web.Application(middlewares=[errors], client_max_size=1024 * 1024)
        stream_tasks = set()

        def stream_handler(handler):
            @wraps(handler)
            async def tracked(request):
                task = asyncio.current_task()
                stream_tasks.add(task)
                try:
                    return await handler(request)
                finally:
                    stream_tasks.discard(task)

            return tracked

        async def stop_streams(_app):
            tasks = tuple(stream_tasks)
            for task in tasks:
                task.cancel()
            await asyncio.gather(*tasks, return_exceptions=True)

        app.on_shutdown.append(stop_streams)

        read_router = RunReadRouter(wb.config.library.task_source_root, config=wb.config)
        read_router.mount_gateway_routes(
            app,
            root=wb.root,
            task_supplier=lambda: wb.task,
            status_supplier=lambda: wb.status,
        )

        async def run(request):
            return web.json_response(wb.snapshot())

        async def remote_status(request):
            if wb.remote is not None:
                return web.json_response(wb.remote.status(), headers={"Cache-Control": "no-store"})
            from ...core.storage.read_models import recorded_remote_status
            return web.json_response(recorded_remote_status(wb.store.root), headers={"Cache-Control": "no-store"})

        async def observation_perception(request):
            identifier = request.match_info.get("id")
            obs_id = identifier
            if identifier == "latest":
                latest = getattr(wb.harness, "latest", None) if wb.harness is not None else None
                obs_id = latest.get("observation_id") if isinstance(latest, dict) else None
            if not obs_id:
                raise web.HTTPNotFound(text="observation_not_found")
            from ...core.storage.read_models import observation_perception as read_perception
            try:
                value = read_perception(wb.store, obs_id)
            except KeyError:
                raise web.HTTPNotFound(text="perception_not_found") from None
            return web.json_response(value, headers={"Cache-Control": "no-store"})

        async def observation(request):
            identifier = request.match_info.get("id")
            target = wb.harness.latest if identifier == "latest" else wb.store.read("observation", identifier)
            if target is None:
                raise web.HTTPNotFound(text="observation_not_found")
            return web.json_response(wb.store.observation_view(target))

        async def artifact(request):
            reference = "artifact:" + request.match_info["id"]
            try:
                return await artifact_response(wb.store, reference, image=request.query.get("view") == "image")
            except KeyError:
                if not wb.scene_service:
                    raise
                return await artifact_response(
                    wb.scene_service.store,
                    reference,
                    allowed_scopes=("actor", "internal"),
                )

        async def action(request):
            return web.json_response(wb.harness.action(request.match_info["id"]))

        @stream_handler
        async def events(request):
            after = max(int(request.query.get("after") or "0"), int(request.headers.get("Last-Event-ID") or "0"))
            if after < 0:
                raise ValueError("invalid_event_cursor")
            response = web.StreamResponse(headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"})
            await response.prepare(request)
            try:
                if after > wb.store.event_cursor():
                    after = wb.store.event_cursor()
                    await response.write(
                        f'event: gap\ndata: {canonical({"reason": "cursor_ahead_or_run_reset", "current_seq": after})}\n\n'.encode()
                    )
                while True:
                    batch = wb.store.events(after)
                    for event in batch:
                        await response.write(f'id: {event["server_seq"]}\ndata: {canonical(event)}\n\n'.encode())
                        after = event["server_seq"]
                    if not batch:
                        await response.write(b": live\n\n")
                        await asyncio.sleep(0.2)
            except (ConnectionError, ClientError, asyncio.CancelledError, StoreClosed):
                pass
            return response

        @stream_handler
        async def terminal(request):
            if not wb.terminal:
                raise web.HTTPServiceUnavailable(text="terminal_not_started")
            if request.headers.get("Origin") not in (None, f"http://{request.host}"):
                raise web.HTTPForbidden(text="same_origin_required")
            after = int(request.query.get("after") or "0")
            ws = web.WebSocketResponse(max_msg_size=4096)
            await ws.prepare(request)
            prior_session = request.query.get("session_id")
            if prior_session and prior_session != wb.terminal.session_id:
                await ws.send_json(dict(type="gap", reason="terminal_session_changed", current_cursor=wb.terminal.cursor))
                after = 0
            await ws.send_json(dict(type="ready", status=wb.terminal.status()))
            if after < 0 or after > wb.terminal.cursor:
                await ws.send_json(
                    dict(type="gap", reason="terminal_cursor_outside_retained_stream", current_cursor=wb.terminal.cursor)
                )
                after = 0

            async def output():
                try:
                    async for data, cursor in wb.terminal.stream(after):
                        await ws.send_json(
                            dict(type="output", cursor=cursor, after=cursor - len(data), data=base64.b64encode(data).decode())
                        )
                    await ws.send_json(dict(type="status", status=wb.terminal.status()))
                except (ConnectionError, ClientError, asyncio.CancelledError):
                    pass
                except Exception as error:
                    try:
                        wb.store.event("terminal.stream_failed", {"error": f"{type(error).__name__}: {error}"})
                    except Exception:
                        pass
                    try:
                        if not ws.closed:
                            await ws.send_json(
                                dict(
                                    type="error",
                                    terminal=True,
                                    message=f"terminal_stream_failed: {type(error).__name__}",
                                )
                            )
                            await ws.close()
                    except Exception:
                        pass

            task = asyncio.create_task(output())
            try:
                async for message in ws:
                    if message.type != WSMsgType.TEXT:
                        continue
                    try:
                        value = json.loads(message.data)
                        if not isinstance(value, dict):
                            raise ValueError("terminal_input_is_read_only")
                        msg_type = value.get("type")
                        if msg_type == "resize":
                            if type(value.get("cols")) is not int or type(value.get("rows")) is not int:
                                raise ValueError("invalid_terminal_size")
                            wb.terminal.resize(value["cols"], value["rows"])
                        elif msg_type == "scroll":
                            if wb.replay:
                                raise ValueError("replay_is_read_only")
                            wb.terminal.scroll(
                                lines=value["lines"] if "lines" in value else 0,
                                action=value.get("action"),
                            )
                        else:
                            raise ValueError("terminal_input_is_read_only")
                    except (ValueError, KeyError, TypeError) as error:
                        await ws.send_json(dict(type="error", message=str(error)))
            finally:
                task.cancel()
                await asyncio.gather(task, return_exceptions=True)
            return ws

        async def control(request):
            return await self.control_dispatch(await request.json(), request.headers.get("Origin"))

        async def observations(request):
            return web.json_response({"observations": [wb.store.observation_view(o) for o in wb.store.records("observation")]})

        async def records(request):
            return web.json_response(record_updates(wb.store, request.query))

        async def items(request):
            identifier = request.match_info.get("id")
            if identifier:
                item = wb.store.read_state("codex_item", identifier)
                if item is None:
                    raise KeyError(identifier)
                return web.json_response(item)
            return web.json_response({"items": wb.store.states("codex_item"), "cursor": wb.store.event_cursor()})

        async def requests(request):
            return web.json_response({"requests": wb.store.states("native_request")})

        async def scene_get(request):
            kind = request.match_info["kind"]
            if request.match_info.get("version"):
                result = wb.harness.scene.version(wb.identity, int(request.match_info["version"]))
            elif kind == "head":
                result = wb.harness.scene.head(wb.identity)
            elif kind == "versions":
                result = {"versions": wb.harness.scene.versions(wb.identity)}
            else:
                result = wb.harness.scene.version(wb.identity, int(request.match_info["version"]))
            return web.json_response(result)

        async def scene_post(request):
            if wb.replay:
                raise ValueError("replay_is_read_only")
            if not wb.scene_service:
                raise ValueError("scene_service_not_started")
            value = await request.json()
            if not isinstance(value, dict):
                raise ValueError("scene_request_requires_object")
            if any(key in value and value[key] != wb.identity[key] for key in IDENTITY):
                raise ValueError("scene_identity_conflict")
            identifier = value.get("request_id") or request.headers.get("Idempotency-Key")
            if not isinstance(identifier, str) or not 0 < len(identifier) <= 256:
                raise ValueError("scene_requires_request_id")
            scope = "scene-" + digest([request.path, identifier])
            wb.store.save("scene_request", scope, value)
            old = wb.store.read_state("scene_response", scope)
            if old:
                return web.json_response(old)
            body = {key: item for key, item in value.items() if key not in {"request_id", "request"}}
            body.update(wb.identity, schema_version="0.1")
            operation = request.match_info["operation"]
            if operation == "query":
                result = wb.harness.scene.query(wb.identity, value["scene_version"], value.get("request", value))
            elif operation == "snapshot":
                result = wb.harness.scene.snapshot(wb.identity, value["scene_version"], value["anchor_observation_id"])
            else:
                req_sub = value.get("request", {})
                if not isinstance(req_sub, dict):
                    raise ValueError("scene_request_requires_object")
                body.update(req_sub)
                body.update(wb.identity)
                result = wb.scene_service.submit(
                    body,
                    "alignment" if operation == "alignment" else "build_scene",
                    identifier="job-" + scope,
                )
            wb.store.set_state("scene_response", scope, result)
            return web.json_response(result)

        async def job(request):
            if wb.replay and request.method == "POST":
                raise ValueError("replay_is_read_only")
            identifier = request.match_info["id"]
            if identifier.startswith("program-"):
                if wb.replay:
                    result = wb.store.read_state("program", identifier)
                    if result is None:
                        raise KeyError(identifier)
                else:
                    result = (
                        wb.harness.programs.cancel(identifier)
                        if request.method == "POST"
                        else wb.harness.programs.status(identifier)
                    )
            else:
                result = wb.scene_service.job(identifier)
                if any(result[key] != wb.identity[key] for key in IDENTITY):
                    raise ValueError("job_identity_conflict")
                if request.method == "POST" and result["status"] not in JOB_TERMINAL:
                    result = wb.scene_service.transition(
                        result,
                        result["status"],
                        cancel_requested=True,
                        cancel_reason="web_request",
                    )
            return web.json_response(result)

        def unsupported(operation, capability):
            return web.json_response(
                {
                    **wb.identity,
                    "status": "unsupported",
                    "operation": operation,
                    "reason": f"{capability}_unsupported_by_robot_backend",
                    "audience": "observer_only",
                },
                status=501,
            )

        async def evaluate(request):
            if not (isinstance(wb.robot, EvaluatorBackend) or callable(getattr(wb.robot, "evaluate", None))):
                return unsupported("observer_evaluate", "native_evaluator")
            return web.json_response(await asyncio.to_thread(wb.observer_evaluate))

        async def observer_snapshot(request):
            if not (isinstance(wb.robot, EvaluatorBackend) or callable(getattr(wb.robot, "observer_snapshot", None))):
                return unsupported("observer_snapshot", "observer_snapshot")

            def capture():
                with wb.harness.robot_lock:
                    wb.require_observer_idle()
                    result = wb.robot.observer_snapshot()
                    evidence = {**wb.identity, "audience": "observer_only", "snapshot": result}
                    evidence["snapshot_ref"] = wb.store.put_artifact(
                        canonical(evidence).encode(),
                        "observer_scene_snapshot",
                        "application/json",
                        "observer",
                    )
                    return evidence

            return web.json_response(await asyncio.to_thread(capture))

        async def record_video(request):
            if not (isinstance(wb.robot, VideoBackend) or callable(getattr(wb.robot, "record", None))):
                return unsupported("observer_record", "observer_video")
            value = await request.json()
            if not isinstance(value, dict) or type(value.get("enabled", True)) is not bool:
                raise ValueError("invalid_recording_enabled")

            def record():
                with wb.harness.robot_lock:
                    wb.require_observer_idle()
                    result = {**wb.identity, **wb.robot.record(enabled=value.get("enabled", True))}
                    wb.store.event("observer.recording", result, source="private_evaluator")
                    return result

            return web.json_response(await asyncio.to_thread(record))

        async def video(request):
            if not (isinstance(wb.robot, VideoBackend) or callable(getattr(wb.robot, "save_video", None))):
                return unsupported("observer_video", "observer_video")
            return web.json_response(await asyncio.to_thread(wb.observer_video))

        def observer_guard(handler):
            async def guarded(request):
                async with wb.control_lock:
                    wb.require_observer_idle()
                    return await handler(request)

            return guarded

        evaluate = observer_guard(evaluate)
        observer_snapshot = observer_guard(observer_snapshot)
        record_video = observer_guard(record_video)
        video = observer_guard(video)

        app.router.add_get("/v1/observations", observations)
        app.router.add_get("/v1/records", records)
        app.router.add_get("/v1/items", items)
        app.router.add_get("/v1/items/{id}", items)
        app.router.add_get("/v1/requests", requests)
        app.router.add_get("/v1/scene/{kind:head|versions}", scene_get)
        app.router.add_get("/v1/scene/{kind:versions}/{version}", scene_get)
        app.router.add_post("/v1/scene/{operation:query|snapshot|alignment|update}", scene_post)
        app.router.add_get("/v1/jobs/{id}", job)
        app.router.add_post("/v1/jobs/{id}/cancel", job)
        app.router.add_get("/v1/observer/snapshot", observer_snapshot)
        app.router.add_post("/v1/observer/evaluate", evaluate)
        app.router.add_post("/v1/observer/video", video)
        app.router.add_post("/v1/observer/record", record_video)
        app.router.add_get("/v1/run", run)
        app.router.add_get("/v1/remote/status", remote_status)
        app.router.add_get("/v1/runs/{run_id}", run)
        app.router.add_get("/v1/events", events)
        app.router.add_get("/v1/runs/{run_id}/events", events)
        app.router.add_get("/v1/observations/{id}", observation)
        app.router.add_get("/v1/observations/{id}/perception", observation_perception)
        app.router.add_get("/v1/artifacts/{id}", artifact)
        app.router.add_get("/v1/actions/{id}", action)
        app.router.add_get("/v1/terminal", terminal)
        app.router.add_post("/v1/control", control)
        app.router.add_post("/v1/runs/{run_id}/control", control)

        mounted = []
        for extension in getattr(wb, "extensions", ()):
            if extension.supports(wb):
                extension.mount(app)
                mounted.append(extension.name)
        wb.mounted_extensions = tuple(mounted)

        static = static_directory()

        async def index(request):
            if self._is_operator_console_home():
                raise web.HTTPFound("/console", headers={"Cache-Control": "no-store"})
            return web.FileResponse(static / "index.html", headers={"Cache-Control": "no-store"})

        app.router.add_get("/", index)
        if static.exists():
            app.router.add_static("/", static)
        return app


WorkbenchRoutes = GatewayRouter

__all__ = ["GatewayRouter", "WorkbenchRoutes", "static_directory"]
