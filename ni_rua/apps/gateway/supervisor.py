"""Process-level run supervision: signals, lifecycle, serving, and handoff chaining."""
from __future__ import annotations

import asyncio
import json
import os
import signal
import sys
import time
from pathlib import Path

from aiohttp import web

from ... import __file__ as NI_RUA_PACKAGE_FILE
from ...core.config import Config
from ...core.contracts import canonical
from ...core.presentation.actor_payload import DEFAULT_ACTOR_PAYLOAD_PROFILE
from ...cortex.agents.base import session_backend_kind
from ...cortex.agents.factory import backend_requirements
from ...cortex.session.usage import load_price_card
from ...runtime.supervision import (
    INACTIVITY_REASON,
    model_budget_exhausted,
    robot_gpu_oom,
    start_with_shutdown,
    wait_turn,
)
from ..evaluation.episode import episode_summary, latest_observer_evaluation, settle_model_records, wait_benchmark
from ..library.experiments import allocate_run, finish_run, register_run, write_runtime
from ..library.scoring import classify_episode_result
from ..remote.manager import RemoteManager, start_services
from ..remote.tunnel import write_record
from .extensions import handoff
from .workbench import DEFAULT_PROMPT, Workbench, _UNSET


def shutdown_signals(*, signal_module=signal):
    """Signals that request an orderly shutdown.

    SIGHUP is the controlling terminal closing; without a handler Python's
    default action kills the process before the archive is finalised. A SIGHUP
    that was deliberately ignored at launch (`nohup`, a detached launcher) stays
    ignored, so a detached gateway survives its terminal as intended.
    """
    result = [signal_module.SIGINT, signal_module.SIGTERM]
    if hasattr(signal_module, "SIGHUP") and signal_module.getsignal(signal_module.SIGHUP) is not signal_module.SIG_IGN:
        result.append(signal_module.SIGHUP)
    return result


def _say(text, *, os_module=os):
    """Operator-facing console line.

    After a hangup stdout may be a dead tty or a closed pipe; losing a console
    line must never abort shutdown, so on the first failure fds 1 and 2 are
    pointed at /dev/null for the rest of the run.
    """
    try:
        print(text, flush=True)
    except (OSError, ValueError):
        try:
            null = os_module.open(os_module.devnull, os_module.O_RDWR)
            for fd in (1, 2):
                os_module.dup2(null, fd)
            os_module.close(null)
        except OSError:
            pass


class ProcessSignals:
    """Shutdown signals of the gateway process across every run of one ``serve``."""

    def __init__(self):
        self.signals = None
        self.received = None
        self._done = None

    def bind(self, done):
        self._done = done
        if self.signals is None:
            self.signals = shutdown_signals()
            loop = asyncio.get_running_loop()
            for sig in self.signals:
                loop.add_signal_handler(sig, self._handle, sig)
        if self.received is not None:
            done.set()

    def _handle(self, sig):
        self.received = sig
        if self._done is not None:
            self._done.set()


def _turn_completed_after_reset(store) -> bool:
    """确认当前回合的 ``agent.turn_completed(completed)`` 未被后续环境重置作废。"""
    rows = store.events_by_type(
        ("agent.turn_completed", "environment.reset_requested", "environment.recovery_reset"),
        limit=1,
        newest_first=True,
    )
    return bool(
        rows
        and rows[0].get("type") == "agent.turn_completed"
        and (rows[0].get("payload") or {}).get("status") == "completed"
    )


def _format_internal_runtime_error(tripped) -> str:
    if isinstance(tripped, dict):
        return (
            f"internal_runtime_error: {tripped.get('tool', 'unknown')}: "
            f"{tripped.get('error_type', 'Exception')}: {tripped.get('message', '')}"
        )[:4096]
    return "internal_runtime_error"


def rebind_remote_record(remote, root):
    """Point a handoff-born run's shared remote services at its own ``private/remote_services.json``."""
    if remote.config.record:
        return
    remote.record_path = str(root / "private" / "remote_services.json")
    supervisor = getattr(remote, "supervisor", None)
    if supervisor is not None:
        supervisor.record_path = remote.record_path
        try:
            write_record(remote.record_path, supervisor.record)
        except (OSError, TypeError, ValueError):
            pass


class RunSupervisor:
    """Owns process-level run registration, HTTP serving, wait/evaluation, and handoff chaining."""

    def __init__(
        self,
        *,
        workbench_factory=Workbench,
        remote_manager_cls=RemoteManager,
        start_services_fn=start_services,
        say_fn=_say,
        serve_run_fn=None,
        allocate_run_fn=allocate_run,
        register_run_fn=register_run,
        finish_run_fn=finish_run,
        write_runtime_fn=write_runtime,
    ):
        self.workbench_factory = workbench_factory
        self.remote_manager_cls = remote_manager_cls
        self.start_services_fn = start_services_fn
        self.say = say_fn
        self._serve_run_override = serve_run_fn
        self.allocate_run = allocate_run_fn
        self.register_run = register_run_fn
        self.finish_run = finish_run_fn
        self.write_runtime = write_runtime_fn

    async def serve(self, args):
        """Serve one run, then the successor run of every handoff it accepts."""
        remote_config = getattr(getattr(args, "resolved_config", None), "remote", None)
        remote = (
            self.remote_manager_cls(remote_config)
            if remote_config is not None and remote_config.enabled and not args.replay
            else None
        )
        serve_run = self._serve_run_override or self.serve_run
        try:
            outcome = await serve_run(args, remote)
            while isinstance(outcome, handoff.Successor):
                outcome = await self.serve_successor(outcome, remote)
        finally:
            if remote is not None:
                await asyncio.to_thread(remote.stop)

    async def serve_successor(self, successor, remote):
        """Serve a handoff successor, falling back once on startup failure."""
        serve_run = self._serve_run_override or self.serve_run
        try:
            return await serve_run(successor.args, remote, launch=successor.launch)
        except Exception as error:
            fallback = successor.fallback(error)
            if fallback is None:
                raise
            self.say(
                f"Handoff: successor run {successor.launch.request.successor_run_id} failed to start "
                f"({handoff.failure_reason(error)}); starting one fallback run with the previous settings"
            )
            return await serve_run(fallback.args, remote, launch=fallback.launch)

    async def serve_run(self, args, remote, launch=None):
        """Serve one run and return the next ``handoff.Successor`` if accepted, else None."""
        signals = launch.signals if launch is not None else ProcessSignals()
        born = launch is not None and launch.born
        if born and signals.received is not None:
            handoff.finish_unstarted(args.run_dir, launch, status="cancelled", reason="gateway_shutdown_before_start")
            return None
        if (args.resume or args.replay) and not args.run_dir:
            raise ValueError("resume_and_replay_require_run_dir")
        try:
            resolved = getattr(args, "resolved_config", None)
            if resolved is not None and not args.replay and not args.resume:
                backend_requirements(resolved)
            card_path = getattr(args, "price_card", None)
            price_card = load_price_card(card_path) if card_path and not (args.resume or args.replay) else None
            root = (
                Path(args.run_dir).expanduser().resolve()
                if args.run_dir
                else self.allocate_run(
                    args.experiment_root,
                    category=args.category,
                    task=args.task,
                    title=args.title,
                    tags=args.tag,
                    kind="workbench",
                )
            )
            if not args.replay:
                self.register_run(
                    root,
                    category=args.category,
                    task=args.task,
                    title=args.title,
                    tags=args.tag,
                    kind="workbench",
                )
            workbench = self.workbench_factory(
                root,
                args.task,
                args.gpu,
                args.max_actions,
                resume=args.resume,
                replay=args.replay,
                observation_profile=args.observation_profile,
                seed=args.seed,
                layout_id=args.layout_id,
                price_card=price_card,
                pi05=getattr(args, "pi05", False),
                visual_review_model=getattr(args, "visual_review_model", None),
                actor_payload_profile=getattr(args, "actor_payload_profile", DEFAULT_ACTOR_PAYLOAD_PROFILE),
                enable_wait=getattr(args, "enable_wait", True),
                tracking_guard=getattr(args, "tracking_guard", False),
                joint_replan=getattr(args, "joint_replan", _UNSET),
                completion_guidance=getattr(args, "completion_guidance", _UNSET),
                default_task_notes=getattr(args, "default_task_notes", True),
                vision_compare=getattr(args, "vision_compare", True),
                motion_review=getattr(args, "motion_review", True),
                config=getattr(args, "resolved_config", None),
            )
        except Exception as error:
            if born:
                handoff.finish_unstarted(args.run_dir, launch, status="failed", reason=handoff.failure_reason(error))
            raise
        done = asyncio.Event()
        runner = None
        heartbeat_task = budget_task = None
        final_status = "stopped"
        infrastructure_error = None
        bound_port = args.port

        async def heartbeat():
            while True:
                self.write_runtime(
                    root,
                    {
                        **workbench.snapshot(),
                        "endpoint": f"http://127.0.0.1:{bound_port}",
                        "pid": os.getpid(),
                        "kind": "workbench",
                        **({"remote": remote.status()} if remote is not None else {}),
                    },
                )
                await asyncio.sleep(2)

        started_at = time.monotonic()
        try:
            if args.replay:
                args.benchmark, args.exit_after_turn, args.wall_budget_s = False, False, None
            handoff.adopt(workbench, launch)
            if args.resume:
                for key in (
                    "task",
                    "seed",
                    "layout_id",
                    "gpu",
                    "max_actions",
                    "wall_budget_s",
                    "benchmark",
                    "prompt",
                    "exit_after_turn",
                    "task_notes",
                    "task_notes_source",
                ):
                    setattr(args, key, getattr(workbench.config.run, key))
            workbench.remote = remote
            if remote is not None and not remote.record_path:
                remote.record_path = remote.config.record or str(root / "private" / "remote_services.json")
            elif remote is not None and born:
                rebind_remote_record(remote, root)
            runner = web.AppRunner(workbench.app())
            signals.bind(done)
            await runner.setup()
            site = web.TCPSite(runner, "127.0.0.1", args.port)
            await site.start()
            bound_port = site._server.sockets[0].getsockname()[1]
            if not args.replay:
                heartbeat_task = asyncio.create_task(heartbeat())
            self.say(f"Workbench http://127.0.0.1:{bound_port}; run {root}")
            # Emitted after the Workbench line, which launchers read first; workers log the same startup facts.
            self.say(f"gateway_process pid={os.getpid()} python={sys.executable} ni_rua={Path(NI_RUA_PACKAGE_FILE).resolve()}")
            if remote is not None and not born:
                await self.start_services_fn(remote, done)
            if done.is_set():
                return None
            if not (args.replay or args.exit_after_turn or args.benchmark):
                base = handoff.base_config(args, workbench)
                if base is not None:
                    workbench.handoff_host = handoff.HandoffHost(base_config=base, port=bound_port, done=done)
            try:
                await start_with_shutdown(
                    workbench,
                    done,
                    prompt=args.prompt or DEFAULT_PROMPT,
                    benchmark=args.benchmark,
                    task_notes=args.task_notes,
                    task_notes_source=args.task_notes_source,
                    wall_budget_s=args.wall_budget_s,
                )
            except asyncio.TimeoutError:
                await workbench.stop("official_startup_timeout")
                raise RuntimeError("official_startup_timeout") from None
            if launch is not None:
                launch.started = True
            client_view = getattr(workbench.session, "client_view", None)
            external = client_view() if callable(client_view) else None
            if external and external.get("endpoint"):
                self.say(
                    f"External agent MCP endpoint {external['endpoint']} (no bearer; the agent's own model and settings "
                    "apply: --model, --effort and agent.evidence are not used)"
                )
            self.say(canonical(workbench.snapshot()))
            if args.exit_after_turn or args.benchmark:
                finished = (
                    await wait_benchmark(workbench, done)
                    if args.benchmark
                    else await wait_turn(workbench.session, done, args.wall_budget_s, workbench.harness)
                )
                if not finished or workbench.has_active_programs():
                    await workbench.stop(
                        "actor_finished_with_active_programs"
                        if finished
                        else "shutdown-signal"
                        if done.is_set()
                        else getattr(workbench.harness, "wait_stop_reason", None) or "wall-budget"
                    )
                tripped = getattr(workbench.harness, "internal_runtime_tripped", None)
                internal_stopped = (
                    tripped is not None
                    or getattr(workbench.harness, "wait_stop_reason", None) == "internal_runtime_error"
                )
                if getattr(workbench.harness, "wait_stop_reason", None) == INACTIVITY_REASON:
                    infrastructure_error = INACTIVITY_REASON
                elif robot_gpu_oom(workbench.harness):
                    infrastructure_error = workbench.harness.robot.failure
                elif internal_stopped:
                    infrastructure_error = _format_internal_runtime_error(tripped)
                native_terminal = getattr(workbench.harness, "episode_terminal", None)
                unresolved = (
                    workbench.current_unknown_actions()
                    or workbench.harness.active_action
                    or workbench.has_active_programs()
                    or (not native_terminal and (workbench.session.state == "running" or workbench.status == "unknown"))
                )
                final_status = (
                    "unknown"
                    if unresolved
                    else "completed"
                    if native_terminal or (not args.benchmark and finished and workbench.session.state == "completed")
                    else "failed"
                    if internal_stopped
                    else "stopped"
                )
                if args.benchmark and unresolved:
                    workbench.store.event("observer.finalization_skipped", {"reason": "motion_or_turn_stop_unconfirmed"})
                elif args.benchmark:
                    async with workbench.control_lock:
                        evaluation = await asyncio.to_thread(workbench.observer_evaluate)
                        (root / "evaluation.json").write_text(json.dumps(evaluation, indent=2))
                        if (
                            getattr(workbench, "config", Config()).features.export_video
                            and getattr(workbench, "config", Config()).features.record_video
                        ):
                            video = await asyncio.to_thread(workbench.observer_video)
                            (root / "video.json").write_text(json.dumps(video, indent=2))
                self.say(canonical({"turn_result": workbench.session.turn_result}))
                done.set()
            else:
                async def budget():
                    await asyncio.sleep(args.wall_budget_s)
                    await workbench.stop("wall-budget")
                    await asyncio.to_thread(workbench.robot.close)

                if not args.replay and args.wall_budget_s is not None:
                    budget_task = asyncio.create_task(budget())
                await done.wait()
                # 交互模式停机前先关闭未结束的外部/托管回合，确保 episode_summary() 与事件流中的 interrupted 状态一致
                if not args.replay and (
                    workbench.session.state == "running"
                    or workbench.harness.active_action
                    or workbench.has_active_programs()
                ):
                    await workbench.stop("shutdown-signal")
                if not args.replay:
                    tripped = getattr(workbench.harness, "internal_runtime_tripped", None)
                    internal_stopped = (
                        tripped is not None
                        or getattr(workbench.harness, "wait_stop_reason", None) == "internal_runtime_error"
                    )
                    if robot_gpu_oom(workbench.harness):
                        infrastructure_error = workbench.harness.robot.failure
                    elif internal_stopped:
                        infrastructure_error = _format_internal_runtime_error(tripped)
                    native_terminal = getattr(workbench.harness, "episode_terminal", None)
                    external_finished = (
                        session_backend_kind(workbench.session) == "external"
                        and workbench.session.state == "completed"
                        and _turn_completed_after_reset(workbench.store)
                    )
                    unresolved = (
                        workbench.current_unknown_actions()
                        or workbench.harness.active_action
                        or workbench.has_active_programs()
                        or (
                            not native_terminal
                            and (workbench.session.state == "running" or workbench.status == "unknown")
                        )
                    )
                    final_status = (
                        "unknown"
                        if unresolved
                        else "completed"
                        if native_terminal
                        or (
                            external_finished
                            and not internal_stopped
                            and infrastructure_error is None
                            and getattr(workbench, "successor", None) is None
                        )
                        else "failed"
                        if internal_stopped
                        else "stopped"
                    )
        except Exception as error:
            final_status = "failed"
            infrastructure_error = str(error)
            workbench.store.event("run.infrastructure_failed", {"reason": infrastructure_error})
            raise
        finally:
            for task in (heartbeat_task, budget_task):
                if task:
                    task.cancel()
                    await asyncio.gather(task, return_exceptions=True)
            try:
                if runner is not None:
                    await runner.cleanup()
            finally:
                try:
                    try:
                        if not args.replay:
                            # The relay's records trail its delivered answers; wait for them off the event loop (bounded
                            # by ModelRequestProxy.SETTLE_TIMEOUT_S). First, so that no await separates the status checks
                            # below from the summary that reads the same state.
                            settled = await asyncio.to_thread(settle_model_records, workbench.session)
                            native_turn = workbench.session.turn_result or {}
                            codex_backend = session_backend_kind(workbench.session) == "codex"
                            if native_turn.get("status") == "failed":
                                expected_limit_stop = model_budget_exhausted(workbench.session)
                                if expected_limit_stop:
                                    workbench.harness.wait_stop_reason = "model_request_limit"
                                    workbench.store.event(
                                        "codex.expected_budget_stop" if codex_backend else "agent.expected_budget_stop",
                                        {"turn_error": native_turn.get("error")},
                                    )
                                elif final_status != "unknown" and not getattr(workbench.harness, "episode_terminal", None):
                                    final_status = "failed"
                                if infrastructure_error is None and not expected_limit_stop:
                                    error = native_turn.get("error") or {}
                                    detail = error.get("message") if isinstance(error, dict) else str(error)
                                    infrastructure_error = (
                                        ("codex_turn_failed: " if codex_backend else "agent_turn_failed: ")
                                        + str(detail or canonical(error))[:4096]
                                    )
                            if (
                                workbench.current_unknown_actions()
                                or workbench.harness.active_action
                                or workbench.has_active_programs()
                            ):
                                final_status = "unknown"
                            # 交互模式下若已通过 POST /v1/observer/evaluate 记录最新有效评测，停机时同步落盘 evaluation.json
                            if not args.benchmark and final_status != "unknown":
                                latest_eval = latest_observer_evaluation(
                                    workbench.store, workbench.identity, final_status
                                )
                                if latest_eval is not None:
                                    (root / "evaluation.json").write_text(json.dumps(latest_eval, indent=2))
                            result_summary = episode_summary(
                                workbench.store,
                                workbench.identity,
                                workbench.session,
                                workbench.harness,
                                benchmark=args.benchmark,
                                final_status=final_status,
                                started_at=started_at,
                                infrastructure_error=infrastructure_error,
                                model_records_settled=settled,
                            )
                    finally:
                        cleanup_report = await workbench.close()
                        if isinstance(cleanup_report, dict):
                            result_summary = locals().get(
                                "result_summary", {"task_success": None, "normalized_score": None}
                            )
                            result_summary["cleanup"] = cleanup_report
                            robot_cleanup = cleanup_report.get("robot") or {}
                            if (
                                cleanup_report.get("errors")
                                or robot_cleanup.get("process_exit_confirmed") is False
                                or robot_cleanup.get("reader_exit_confirmed") is False
                            ):
                                result_summary["cleanup_status"] = "failed"
                                final_status = "unknown"
                            elif robot_cleanup.get("forced_termination"):
                                result_summary["cleanup_status"] = "forced_termination"
                            elif robot_cleanup.get("returncode") not in (None, 0):
                                result_summary["cleanup_status"] = "worker_exit_failed"
                            elif (
                                robot_cleanup
                                and robot_cleanup.get("native_close_acknowledged") is not True
                                and not robot_cleanup.get("not_started")
                            ):
                                result_summary["cleanup_status"] = "native_close_unconfirmed"
                            else:
                                result_summary["cleanup_status"] = "completed"
                finally:
                    if not args.replay:
                        final_summary = locals().get(
                            "result_summary", {"task_success": None, "normalized_score": None}
                        )
                        final_summary = classify_episode_result(final_summary, final_status)
                        final_summary = handoff.summary_links(final_summary, workbench, launch, final_status)
                        self.finish_run(root, status=final_status, summary=final_summary)
        successor = handoff.next_launch(workbench, signals, final_status)
        if successor is not None:
            self.say(
                f"Handoff: run {root.name} finalised ({final_status}); starting successor run "
                f"{successor.launch.request.successor_run_id} on port {bound_port}"
            )
        return successor


__all__ = [
    "ProcessSignals",
    "RunSupervisor",
    "_say",
    "rebind_remote_record",
    "shutdown_signals",
]
