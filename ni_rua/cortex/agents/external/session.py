"""External agent backend (``agent.backend = "external"``; design 20 section 2): NI-RUA launches no agent.

The run's registry tools are served on the fixed loopback MCP endpoint
``http://127.0.0.1:<agent.external.mcp_port>/mcp``
(``NiRuaMcpServer`` with ``serverInfo.name`` ``ni_rua``, no bearer, and the short fixed ``EXTERNAL_MCP_INSTRUCTIONS``
in ``initialize``), and the user's own Claude Code / Codex / OpenCode drives the run through it. There is no native
process, no model relay, and no model, provider, effort, price card or token usage: each is recorded as unknown
(null), never as a host-mode default or 0 (``manifest_null_fields``, ``summary_null_fields``).

Turns: the session bracket. An external agent's turn is opened by the ``run_begin`` tool and closed by ``run_finish``
(registry ``ToolSpec`` objects with the declared effect ``"session"``, ``ni_rua/runtime/tools/plugins/run_bracket.py``).
The transport tells them apart only by that declared effect (``tool_effect`` = ``ToolRegistry.effect_of``), never by
name:

* admission (the MCP request's own handler thread): a session-effect call opens a turn when none is open (``agent.
  turn_started {origin: "external"}``), so its ``tool_call`` record and transcript row carry a turn id; every other
  call is admitted with no bracket check. Nothing is admitted on a fresh run before ``run_turn`` set the task (the
  Workbench calls it right after ``start``); a resumed run has its task from ``start``.
* dispatch (``ToolDispatcher``, arrival order): every other call is served only when the open turn was briefed (a
  ``run_begin`` of that turn returned its briefing). Otherwise it is refused, nothing executed and no transcript row
  written (the MCP raw log has it): ``external_briefing_required`` while no briefing was delivered in this turn or
  session, ``external_turn_closed`` with one fixed sentence per reason after a briefed turn closed (``run_finish``,
  operator stop, operator reset). A session-effect call that opened a turn and failed (bad arguments, a handler error)
  closes that turn again (``interrupted``, error ``external_bracket_call_failed``). The gate runs inside the dispatched
  function, after every earlier ordered call finished, so it follows dispatch order, never HTTP arrival: a call that
  reached the dispatcher after ``run_begin`` waits for it (``run_begin`` is ordered) and is served once it returned
  the briefing; one that reached
  it earlier, also one sent in the same assistant message whose request arrived first, is refused with nothing
  executed and can simply be repeated; a call dispatched after ``run_finish`` is refused.

Any caller is admitted while a turn is open: the host's own subagents share the run's turn. Host ``_meta`` ids stay in
the MCP raw log only. ``initialize`` records ``agent.client_attached`` with the client's self-reported ``clientInfo``;
a refused call records ``agent.tool_refused {code, tool}``.

Operator authority: ``interrupt()`` (console stop / estop, wall budget, gateway shutdown) and ``release_turn``
(console reset, ``operator_turn_release``) close an open turn as ``interrupted``; the agent's next call gets
``external_turn_closed`` naming the operator's action. An open turn never blocks an operator command
(``base.session_turn_blocks_operator``): motion stays protected by motion admission, the robot lock and the active
action. Cancelling a tool call in the agent does not stop the robot: the admitted action keeps running
(``NiRuaMcpServer`` delivery semantics), which ``EXTERNAL_MCP_INSTRUCTIONS`` tells the model. Console commands that
need an agent NI-RUA runs are refused (``console_commands_unsupported``).

Everything recorded here (identity record, events, transcript rows, the console view) is observer data; the only
model-visible text this module produces is ``EXTERNAL_MCP_INSTRUCTIONS``, the briefing (the run's own startup
instructions after the presentation scrub) and the refusals.
"""
import errno
import threading
import time
from pathlib import Path

from ....core.contracts import digest, new_id, utc_now
from ....core.tool_errors import ToolInputError
from ..base import (AgentCapabilities, AgentCapabilityError, AgentSession, STATE_IDLE, STATE_NOT_STARTED, STATE_REPLAY,
                    STATE_RUNNING, UNCONNECTED_STATES, read_agent_identity)
from ..identity import write_identity
from ..mcp_server import PostExecutionDeliveryError, safe_present_tool_result, tool_bridge_error_response
from ..mcp_session import STALE_TOOL_STATES, McpSessionMixin, McpToolBridge
from .protocol import BACKEND_KIND, DOCTOR_CLIENT_NAME, MCP_SERVER_NAME, RUN_BEGIN, RUN_FINISH, SESSION_EFFECT

__all__ = ["CONSOLE_COMMANDS_UNSUPPORTED", "EXTERNAL_MCP_INSTRUCTIONS", "INFORMATION_BOUNDARY", "MANIFEST_NULL_FIELDS",
           "RUN_FINISH_ORIGIN", "SUMMARY_NULL_FIELDS", "TURN_CLOSED", "BracketRefusal", "ExternalSession",
           "ExternalToolBridge", "port_unavailable"]

# manifest agent_backend.information_boundary: the agent sees only what crosses the MCP endpoint (tool catalogue,
# initialize instructions, tool results); NI-RUA sees none of its prompts, model, reasoning or other tools.
INFORMATION_BOUNDARY = "mcp_boundary_only"
# Transcript origin of the run_finish summary row (TranscriptWriter.assistant_text origin): not native prose.
RUN_FINISH_ORIGIN = "run_finish_summary"
# Longest self-reported clientInfo name / version kept in records (characters); longer values are cut.
CLIENT_FIELD_MAX_CHARS = 256
# 外部宿主工具输出截断保守阈值（clientInfo.name -> (max_bytes, max_lines, text_only_without_images)）
_HOST_TRUNCATION_LIMITS = {
    "claude-code": (500_000, None, True),
    "codex-mcp-client": (400_000, None, False),
    "opencode": (16_000_000, 100_000, False),
}

# MCP initialize.instructions (design 20 section 2.4): shown by every host to the model next to the tool catalogue.
EXTERNAL_MCP_INSTRUCTIONS = (
    "NI-RUA robot tools. Call run_begin first, and call it again after any context compaction, a new session or a "
    "resume: it returns the operating instructions and the task, and the other NI-RUA tools are refused until it has "
    "opened a turn. Tool results are the only source of robot facts. Cancelling a tool call in the agent does not "
    "stop robot motion; use the NI-RUA console stop, hold or estop. Call run_finish with an evidence-based summary "
    "when the task is done or cannot continue. If NI-RUA tools are missing or every call fails after the gateway was "
    "restarted with another robot, ask the user to reconnect the ni_rua MCP server or start a new session: the tool "
    "list is not refreshed by itself.")

# Console control commands that need an agent NI-RUA runs itself (a prompt to send, a model to switch, a relay to
# unblock, a session to hand off): refused with AgentCapabilityError("<command>_unsupported_by_backend").
CONSOLE_COMMANDS_UNSUPPORTED = frozenset({
    "prompt", "steer", "reply", "set_model", "set_effort", "set_service_tier", "apply_agent_settings", "handoff_run",
    "retry_model_requests"})
# Manifest keys whose host-mode value describes NI-RUA's model relay or its inactivity supervisor, neither of which
# exists here (no model request budget, retries, retry policy, no-progress timeout): written as null.
MANIFEST_NULL_FIELDS = ("model", "provider", "effort", "price_card", "model_request_limit", "model_request_count_basis",
                        "model_attempt_max_retries", "model_auto_retries", "model_retry_policy",
                        "no_progress_timeout_s", "progress_clock")
# Summary keys NI-RUA cannot measure without a relay (served models, token usage and its accounting): null, never 0.
SUMMARY_NULL_FIELDS = ("token_usage", "token_accounting", "served_models", "model_usage_relay", "model_requests")


def port_unavailable(port, error):
    """The one actionable sentence (``ValueError`` text) for an MCP port that cannot be bound: ``port`` (int) and the
    bind ``error`` (OSError)."""
    if getattr(error, "errno", None) == errno.EADDRINUSE:
        return (f"external_mcp_port_in_use: 127.0.0.1:{port} is already in use; another external gateway may be "
                "running (stop it first), or set agent.external.mcp_port to a free port")
    return (f"external_mcp_port_unavailable: 127.0.0.1:{port} cannot be bound ({type(error).__name__}: "
            f"{getattr(error, 'strerror', None) or error}); set agent.external.mcp_port to a free port")


class BracketRefusal(ToolInputError):
    """A non-session tool dispatched while no briefed turn is open: nothing was executed (model-visible error result).

    ``code``: ``external_briefing_required`` or ``external_turn_closed``; ``str(error)`` is ``"<code>: <sentence>"``,
    the minimal example names ``run_begin``. ``turn_open`` (bool): whether a turn was open at that moment (a turn opened
    by a ``run_begin`` that has not returned its briefing yet counts as open); reported as ``actual.turn_open`` together
    with ``actual.briefed = false``."""

    def __init__(self, code, sentence, *, turn_open):
        super().__init__(code, field_path="/tool", actual=dict(turn_open=bool(turn_open), briefed=False),
                         suggestion=("Nothing was executed. Call run_begin (no arguments), read the operating "
                                     "instructions and the task it returns, then repeat this call if it is still "
                                     "needed."),
                         minimal_example={"tool": RUN_BEGIN, "arguments": {}})
        self.args = (f"{code}: {sentence}",)


BRIEFING_REQUIRED = ("external_briefing_required",
                     "call run_begin first; it returns the operating instructions and the task")
# Why the last briefed turn closed -> the fixed sentence of its external_turn_closed refusal.
TURN_CLOSED = {
    "finished": "run_finish closed the turn; call run_begin to start the next task",
    "operator_stop": ("the operator stopped the run; motion stays refused until the operator resumes; call run_begin "
                      "to re-read the task before continuing"),
    "operator_reset": ("the operator reset the environment; earlier observations and plans no longer describe it; "
                       "call run_begin to re-read the task before continuing"),
}


class ExternalToolBridge(McpToolBridge):
    """``McpToolBridge`` whose admission step is the session bracket (module docstring).

    Admission (``_call_tool``, on the MCP request's handler thread): refused with ``native_tool_call_stale_connection``
    only while the session cannot own a call (not started, closing, disconnected, closed, replay); a session-effect
    call opens a turn when none is open. Dispatch (``_dispatched``, the ``ToolDispatcher`` handler, in arrival order):
    the bracket gate, then the transcript row and ``Harness.handle_tool`` with the turn id the call runs in. Delivery
    failure handling, the second presentation scrub and the shutdown order are the inherited ones.
    """

    def __init__(self, session, tools, *, parallel_hints=None, server_options=None):
        super().__init__(session, tools, parallel_hints=parallel_hints, server_name=MCP_SERVER_NAME,
                         handler=self._dispatched, server_options=server_options)

    def _call_tool(self, name, arguments, *, call_id, native_call_id, rpc_id, native_thread_id=None,
                   is_abandoned=None):
        # No relay, so no model step is ever announced: calls are dispatched in arrival order.
        session = self.session
        bracket = session.is_bracket_call(name, arguments)
        with session.lock:
            if (session.closing or session.state in STALE_TOOL_STATES or session.thread_id is None
                    or not session._task_ready):
                raise ValueError("native_tool_call_stale_connection")
            turn_id = session._ensure_turn(call_id) if bracket else None
            session.last_call_at = utc_now()
            with self._idle:
                self._active += 1
        job = dict(tool=name, arguments=arguments, call_id=call_id, native_call_id=native_call_id,
                   native_thread_id=native_thread_id, is_abandoned=is_abandoned,
                   thread_id=session.thread_id, turn_id=turn_id, bracket=bracket, started=False)
        try:
            return self._run_job(job)
        finally:
            with self._idle:
                self._active -= 1
                self._idle.notify_all()

    def _run_job(self, job):
        session = self.session
        session.last_activity_at = time.monotonic()
        executed = False
        result = None
        try:
            result = self.dispatcher.submit(job).result()
            executed = True
        except Exception as error:  # noqa: BLE001 - answered like the Codex transport (RecordedClient._approval)
            if isinstance(error, BracketRefusal):
                try:
                    session.transcript.event("agent.tool_refused", dict(code=error.code, tool=job["tool"]),
                                             tool_call_id=job["call_id"])
                except Exception as log_error:  # noqa: BLE001 - must never skip _close_failed_opening
                    self._record_bridge_error("tool_refused", job["call_id"], log_error)
            result = tool_bridge_error_response(error, tool=job["tool"], arguments=job["arguments"],
                                                store=session.store, call_id=job["call_id"],
                                                errors_sink=self.server.errors)
        finally:
            if job["bracket"] and not (executed and isinstance(result, dict) and result.get("success")):
                session._close_failed_opening(job["call_id"])
        if executed:
            try:
                result = session.presentation.present(result)
            except Exception as present_error:  # noqa: BLE001 - never claim "no motion" after tool execution
                self._record_bridge_error("present_tool_result", job["call_id"], present_error)
                raise PostExecutionDeliveryError("invalid_tool_result") from present_error
        else:
            result = safe_present_tool_result(session.presentation, result)
        if job["started"]:
            items = result.get("contentItems") if isinstance(result, dict) else None
            try:
                session.transcript.tool_completed(job["call_id"], success=bool(isinstance(result, dict)
                                                                               and result.get("success")),
                                                  content_items=items)
            except Exception as log_error:  # noqa: BLE001 - never discard an executed physical ToolResponse receipt
                self._record_bridge_error("tool_completed", job["call_id"], log_error)
        session.last_activity_at = time.monotonic()
        return result

    def _dispatched(self, job):
        """``ToolDispatcher`` handler: runs on a tool worker after every earlier ordered call finished."""
        session = self.session
        turn_id = session._dispatch_turn(job)
        context_scope = "isolated"
        context_epoch = None
        if job["bracket"]:
            # run_begin / run_finish 始终标记 main，确保重简报 decision_reminder 与能力预播种在主链生效
            context_scope = "main"
        else:
            native_thread_id = job.get("native_thread_id")
            with session.lock:
                main_thread_id = session._main_native_thread_id
            caller_is_main = native_thread_id is None or native_thread_id == main_thread_id
            if caller_is_main and session.context_delivery == "delta":
                abandoned_check = job["is_abandoned"] if callable(job.get("is_abandoned")) else (lambda: False)
                with self._lease_lock:
                    if self._lease_call_id is None:
                        candidate_epoch = self._current_context_epoch()
                        if not abandoned_check():
                            self._lease_call_id = job["call_id"]
                            context_scope = "main"
                            context_epoch = candidate_epoch
        params = dict(
            threadId=job["thread_id"],
            turnId=turn_id,
            callId=job["call_id"],
            tool=job["tool"],
            arguments=job["arguments"],
            callIdSource="native_tool_use_id" if job["native_call_id"] else "mcp_request",
            contextScope=context_scope,
        )
        if context_epoch is not None:
            params["contextEpoch"] = context_epoch
        session.transcript.tool_started(job["call_id"], job["tool"], job["arguments"], native_id=job["native_call_id"],
                                        turn_id=turn_id)
        job["started"] = True
        return session.tool_handler(params)


class ExternalSession(McpSessionMixin, AgentSession):
    """External agent backend (``agent.backend = "external"``). Public contract: ``AgentSession``; see the module doc.

    Constructor (called by ``factory.create_session``): ``run_dir`` (run directory), ``store`` (run Store),
    ``tool_handler`` (``Harness.handle_tool``), ``price_card`` (must be None: there is no token usage to price; the
    launch gate refuses ``model.price_card``), ``config`` (``Config``; default ``Config.load()``) and ``task_planning``
    (unused: the decision reminder reaches the agent through ``on_context_reset`` and the next tool result).

    Members the Workbench reads besides ``AgentSession``'s: ``task`` (``{"text": str | None, "source":
    "operator_prompt" | "none"}``, set by ``run_turn``), ``briefing_text`` (the startup instructions after the
    presentation scrub), ``client_info`` and ``last_call_at`` (``client_view``), and the bracket handlers ``run_begin``
    / ``run_finish`` the two registry tools are bound to.
    """
    backend_kind = BACKEND_KIND
    counting = None
    # Every flag False: no steer, native requests, terminal, fork, sandbox, live effort or native wake-up, and no
    # compaction signal (the agent's history is invisible to NI-RUA; the briefing rule covers context loss).
    capabilities = AgentCapabilities(compaction_signal="none")
    context_delivery = "delta"
    wire_api = None
    console_commands_unsupported = CONSOLE_COMMANDS_UNSUPPORTED
    operator_turn_release = True
    manifest_null_fields = MANIFEST_NULL_FIELDS
    summary_null_fields = SUMMARY_NULL_FIELDS

    def host_tool_output_truncation_limit(self):
        """按自报 ``clientInfo.name`` 返回保守宿主截断阈值，用于 ``delta`` 模式超限基线失效保护。"""
        with self.lock:
            name = self.client_info.get("name") if isinstance(self.client_info, dict) else None
        return _HOST_TRUNCATION_LIMITS.get(name, (None, None, False))

    def __init__(self, run_dir, store, tool_handler, *, price_card=None, config=None, task_planning=None):
        from ....core.config import Config
        from ....core.presentation.actor_presentation import ActorPresentation
        if price_card is not None:
            raise ValueError("external_backend_has_no_model_relay: set model.price_card=null; an external agent's "
                             "token usage is not recorded")
        self.config = config if config is not None else Config.load()
        self.mcp_port = self.config.agent.external.mcp_port
        self.context_delivery = self.config.agent.external.context_delivery
        self.root = Path(run_dir).resolve()
        self.store, self.tool_handler, self.task_planning = store, tool_handler, task_planning
        self.presentation = ActorPresentation(store)
        self.model = self.provider = self.effort = self.price_card = None
        self.active_turn_model = None
        self.sandbox_enabled, self.isolation = False, "none"
        self.actor_dir, self.private = self.root / "actor", self.root / "private"
        self.lock = threading.RLock()
        self._close_lock = threading.Lock()
        self.state, self.thread_id, self.turn_id = STATE_NOT_STARTED, None, None
        self.turn_done, self.turn_result = threading.Event(), None
        self.turn_done.set()
        self.last_activity_at = time.monotonic()
        self.model_limit_reached = threading.Event()
        self.model_proxy = self.bridge = self.transcript = None
        self.closing, self.close_report = False, None
        self.tool_effect = None
        self.task = dict(text=None, source="none")
        self.native_task = None
        self.task_notes = None
        self.briefing_text = None
        self.client_info = self.last_call_at = None
        self.turn_origin = None
        self._started = False
        self._identity = None
        self._instance_id = None
        self._teardown_confirmed = True
        self._interrupt_requested = False
        self._task_ready = False         # run_turn set the task (fresh run) or start read it (resume): calls admitted
        self._turn_opening_call = None   # call id of the session-effect call that opened the current turn
        self._briefed_turn = None        # turn id whose run_begin returned its briefing (the gate serves only it)
        # Why the last briefed turn closed (TURN_CLOSED key), or None: no briefed turn closed yet in this session.
        self._close_reason = None
        # 跨回合 / 跨客户端 delta 基线刷新与主会话识别状态
        self._has_opened_briefed_turn = False
        self._seen_agent_client = False
        self._opening_native_thread_id = None
        self._main_native_thread_id = None
        # 首次 run_begin 内联首轮接地的一次性标志与回调
        self._startup_grounding_delivered = False
        self._startup_grounding_provider = None

    # ---- start ---------------------------------------------------------------------------------------------
    def start(self, tools, instructions, resume_thread_id=None):
        """Open the MCP endpoint on ``agent.external.mcp_port`` and wait for an agent; return the thread id.

        ``tools``: the actor catalogue; ``instructions``: the run's startup instructions (kept after the presentation
        scrub as the ``run_begin`` briefing); ``resume_thread_id``: the recorded thread id of a resumed run (the
        task is then the archived ``run.prompt``). A fresh thread id is host-minted (there is no native thread).
        A port that cannot be bound raises ``ValueError(port_unavailable(...))``. Any failure closes the session.
        """
        with self.lock:
            if self._started or self.closing:
                raise ValueError("session_already_started")
            self._started = True
        try:
            return self._start(tools, instructions, resume_thread_id)
        except BaseException:
            try:
                self.close()
            except Exception:  # noqa: BLE001 - the start failure is the error to report
                pass
            raise

    def _start(self, tools, instructions, resume_thread_id):
        from ..transcript import TranscriptWriter
        resume = resume_thread_id is not None
        previous = read_agent_identity(self.store) if resume else None
        if resume and (previous or {}).get("backend") != BACKEND_KIND:
            raise ValueError("resume_agent_identity_missing: the run has no recorded external agent session")
        thread_id = resume_thread_id if resume else new_id("thread")
        self._instance_id = new_id("agent")
        self.thread_id = thread_id
        self.transcript = TranscriptWriter(self.store, backend=self.backend_kind, thread_id=thread_id)
        self.briefing_text = self.presentation.present_text(instructions)
        if resume:
            self.task = self._task(self.config.run.prompt)
        self.bridge = ExternalToolBridge(self, tools, parallel_hints=self.tool_parallel_hints, server_options=dict(
            port=self.mcp_port, require_bearer=False,
            instructions=self.presentation.present_text(EXTERNAL_MCP_INSTRUCTIONS),
            on_initialize=self._client_initialized))
        try:
            self.bridge.start()
        except OSError as error:
            raise ValueError(port_unavailable(self.mcp_port, error)) from None
        identity = write_identity(
            self.store,
            backend=self.backend_kind, thread_id=thread_id, instance_id=self._instance_id, pid=None,
            model=None, provider=None, effort=None, isolation=self.isolation, tools_digest=digest(tools),
            instructions_digest=digest(self.briefing_text), counting=None, wire_api=None,
            mcp=dict(url=self.bridge.url, auth="none"), resumed=resume, clean_shutdown=False)
        self._identity = identity
        if resume:
            clean = (previous or {}).get("clean_shutdown") is True
            if not clean:
                # Whatever the agent received before the crash can no longer be assumed in its context.
                self._invalidate_context("agent_resume_after_unclean_exit", marker=new_id("resume"))
            self.transcript.event("agent.history_recovered", dict(clean=clean, delta_history_complete=False))
        self.transcript.event("agent.endpoint_ready", dict(url=self.bridge.url, auth="none"))
        with self.lock:
            self.state = STATE_IDLE
            self._task_ready = resume   # a fresh run admits calls once run_turn set its task
        return thread_id

    def _task(self, text):
        text = self.presentation.present_text(text) if isinstance(text, str) and text.strip() else None
        return dict(text=text, source="operator_prompt" if text is not None else "none")

    # ---- turn protocol ---------------------------------------------------------------------------------------
    def run_turn(self, prompt):
        """Keep the operator's task for the next ``run_begin`` and return None: nothing is sent and no turn is
        admitted (the agent opens it). ``prompt``: the gateway ``--prompt`` / ``run.prompt`` text, or None when the
        operator gave none (``task.source`` ``"none"``; the development-acceptance default prompt is never used).
        ``ValueError("session_not_connected")`` before start, while closing, in replay or when unconnected."""
        with self.lock:
            if (self.closing or self.state in UNCONNECTED_STATES
                    or self.state in (STATE_NOT_STARTED, STATE_REPLAY)):
                raise ValueError("session_not_connected")
            self.task = self._task(prompt)
            self._task_ready = True
        return None

    def interrupt(self):
        """Operator stop (console stop / estop, wall budget, shutdown): close an open turn as ``interrupted``
        (``agent.turn_closed_by_operator {reason: "operator_stop"}``). The agent itself cannot be interrupted; motion
        admission was already closed by the caller (``Harness.request_stop``). Returns None."""
        self._close_turn_by_operator("operator_stop")

    def release_turn(self, reason):
        """Console ``reset`` (``reason`` ``"operator_reset"``, a ``TURN_CLOSED`` key): close an open turn as
        ``interrupted`` on operator authority (``agent.turn_closed_by_operator {reason}``); the agent's next call gets
        ``external_turn_closed`` with that reason's sentence. Returns True when a turn closed."""
        if not isinstance(reason, str) or reason not in TURN_CLOSED:
            raise ValueError(f"unknown_turn_close_reason:{reason}")
        return self._close_turn_by_operator(reason)

    def _close_turn_by_operator(self, reason):
        with self.lock:
            if self.state != STATE_RUNNING or self.closing:
                return False
            self.transcript.event("agent.turn_closed_by_operator", dict(reason=reason), turn_id=self.turn_id)
            self._closed(reason)
            return self._finish_turn("interrupted")

    def _closed(self, reason):
        """With ``lock`` held, before the open turn closes: remember why, if that turn was briefed."""
        if self._briefed_turn == self.turn_id:
            self._close_reason = reason

    def _close_failed_opening(self, call_id):
        """A session-effect call ``call_id`` failed: if it opened the open turn and no briefing was delivered in it,
        close that turn (``interrupted``, error ``external_bracket_call_failed``): no turn stays open without a
        briefing. Never raises."""
        try:
            with self.lock:
                if (self.state == STATE_RUNNING and self._turn_opening_call == call_id
                        and self._briefed_turn != self.turn_id):
                    # interrupted, not failed: the agent session did not fail and the run must not be classed failed
                    self._finish_turn("interrupted", error="external_bracket_call_failed")
        except Exception:  # noqa: BLE001 - the call's own error is what the agent receives
            pass

    def is_bracket_call(self, name, arguments):
        """True when the registry declares this call's effect ``"session"`` (``tool_effect``); never raises."""
        classify = self.tool_effect
        try:
            return classify is not None and classify(name, arguments) == SESSION_EFFECT
        except Exception:  # noqa: BLE001 - an unclassifiable call takes the gated path
            return False

    def _ensure_turn(self, call_id):
        """With ``lock`` held: the open turn's id, after opening one for the session-effect call ``call_id`` when none
        is open (``agent.turn_started {origin: "external"}``)."""
        if self.state != STATE_RUNNING:
            self._begin_turn(origin="external")
            self._turn_opening_call = call_id
        return self.turn_id

    def _dispatch_turn(self, job):
        """In dispatch order: the turn ``job`` runs in. A session-effect call re-opens a turn when an earlier call
        (``run_finish``, an operator stop) closed the one it was admitted in; any other call is refused
        (``BracketRefusal``) unless the open turn was briefed (module docstring)."""
        with self.lock:
            if self.closing or self.state in STALE_TOOL_STATES:
                raise ValueError("native_tool_call_stale_connection")
            if job["bracket"]:
                turn_id = self._ensure_turn(job["call_id"])
                if self._turn_opening_call == job["call_id"]:
                    self._opening_native_thread_id = job.get("native_thread_id")
                return turn_id
            if self.state == STATE_RUNNING and self._briefed_turn == self.turn_id:
                return self.turn_id
            if self.state != STATE_RUNNING and self._close_reason is not None:
                raise BracketRefusal("external_turn_closed", TURN_CLOSED[self._close_reason], turn_open=False)
            raise BracketRefusal(*BRIEFING_REQUIRED, turn_open=self.state == STATE_RUNNING)

    def _require_open_turn(self, correlation):
        if self.state != STATE_RUNNING or self.turn_id != correlation.get("turn_id"):
            raise ValueError("external_turn_not_open: the turn this call was admitted in has closed; call run_begin")

    def _refresh_context_epoch(self, source, **fields):
        """刷新 ``runtime/image_context_epoch`` 清空 delta 基线，但不触发 ``on_context_reset`` 决策提醒。"""
        record = dict(source=source, thread_id=self.thread_id, **fields)
        record.setdefault("marker", new_id("context-epoch"))
        self.store.set_state("runtime", "image_context_epoch", record)

    def set_startup_grounding_provider(self, provider):
        """注册首次 ``run_begin`` 按 5 重门禁生成内联首轮接地的回调（仅读 ``harness.latest``）。"""
        with self.lock:
            self._startup_grounding_provider = provider

    def set_briefing_context(self, *, native_task=None, task_notes=None):
        """设置 RoboDojo 等原生环境的任务指令、公开任务规则与操作笔记，供 ``run_begin`` 脱敏后返回。"""
        scrubbed_native = None
        if isinstance(native_task, dict):
            instruction = native_task.get("instruction")
            rules_text = native_task.get("public_task_rules")
            scrubbed_native = dict(
                task=self.presentation.present_text(native_task["task"]) if isinstance(native_task.get("task"), str) else None,
                instruction=self.presentation.present_text(instruction) if isinstance(instruction, str) and instruction.strip() else None,
                public_task_rules=self.presentation.present_text(rules_text) if isinstance(rules_text, str) and rules_text.strip() else None,
            )
            if scrubbed_native["instruction"] is None and scrubbed_native["public_task_rules"] is None:
                scrubbed_native = None
        scrubbed_notes = (
            self.presentation.present_text(task_notes)
            if isinstance(task_notes, str) and task_notes.strip()
            else None
        )
        with self.lock:
            self.native_task = scrubbed_native
            self.task_notes = scrubbed_notes

    def run_begin(self, correlation, budget):
        """``run_begin`` handler (bound by the Workbench): the briefing data ``{instructions, task, budget, turn:
        {opened}}`` (plus optional ``native_task``, ``task_notes`` and ``startup_grounding`` when present).
        ``correlation``: the call's tool correlation (``turn_id``, ``tool_call_id``); ``budget``:
        ``Harness.action_budget()``. Marks the turn briefed (the gate then serves the other tools in it). A call inside
        a turn it did not open re-reads the briefing and reports possible context loss
        (``Harness.model_context_reset("external_rebriefing")``, through ``on_context_reset``), so the decision
        reminder rides on the next actor tool result, normally this one (another call finishing first takes it); the
        call that opened the turn does not."""
        with self.lock:
            self._require_open_turn(correlation)
            opened = self._turn_opening_call == correlation.get("tool_call_id")
            prior_opened = self._has_opened_briefed_turn
            if opened:
                self._has_opened_briefed_turn = True
                self._main_native_thread_id = self._opening_native_thread_id
            task, text = dict(self.task), self.briefing_text
            native_task = dict(self.native_task) if self.native_task is not None else None
            task_notes = self.task_notes
            self._briefed_turn = self.turn_id
            grounding_provider = (
                self._startup_grounding_provider
                if opened and not prior_opened and not self._startup_grounding_delivered
                else None
            )
        if not opened:
            self._invalidate_context("external_rebriefing", marker=new_id("rebriefing"))
        elif prior_opened:
            # 非首次开回合（如 run_finish / operator_stop / operator_reset 后）刷新 delta 基线纪元，但不触发决策提醒
            self._refresh_context_epoch("external_new_turn", marker=new_id("turn-epoch"))
        result = dict(instructions=text, task=task, budget=budget, turn=dict(opened=opened))
        if native_task is not None:
            result["native_task"] = native_task
        if task_notes is not None:
            result["task_notes"] = task_notes
        if grounding_provider is not None:
            grounded = grounding_provider(correlation, opened=opened)
            if grounded is not None:
                with self.lock:
                    self._startup_grounding_delivered = True
                record = grounded.record
                result["startup_grounding"] = dict(
                    observation_id=record["actor_observation_alias"],
                    overview_camera_id=record["overview_camera_id"],
                    image_kind=record["image_kind"],
                    image_count=record["image_count"],
                    obs3d_included=record["obs3d_included"],
                )
                result["_startup_grounding"] = grounded
        return result

    def run_finish(self, summary, correlation):
        """``run_finish`` handler (bound by the Workbench): record ``summary`` as the turn's final answer (transcript
        ``agentMessage`` with ``origin`` ``RUN_FINISH_ORIGIN``) and close the turn as ``completed``; returns
        ``{"turn_closed": True}``."""
        with self.lock:
            self._require_open_turn(correlation)
            self.transcript.assistant_text(summary, turn_id=self.turn_id, origin=RUN_FINISH_ORIGIN)
            self._closed("finished")
            if not self._finish_turn("completed"):
                raise ValueError("external_turn_not_open: the turn could not be closed; call run_begin")
        return dict(turn_closed=True)

    # ---- client view -----------------------------------------------------------------------------------------
    def _client_initialized(self, params):
        """``NiRuaMcpServer.on_initialize``: record the client's self-reported ``clientInfo`` (never raises)."""
        info = params.get("clientInfo") if isinstance(params, dict) else None
        info = info if isinstance(info, dict) else {}

        def bounded(value):
            return value[:CLIENT_FIELD_MAX_CHARS] if isinstance(value, str) and value.strip() else None
        client = dict(name=bounded(info.get("name")), version=bounded(info.get("version")), self_reported=True)
        is_doctor = client["name"] == DOCTOR_CLIENT_NAME
        refresh_epoch = False
        with self.lock:
            # doctor 探测不覆盖已附着的真实 Agent 展示信息；若此前尚无真实 Agent 附着则正常记录
            if not (is_doctor
                    and self.client_info is not None
                    and self.client_info.get("name") != DOCTOR_CLIENT_NAME):
                self.client_info = client
            # 新非 doctor 客户端接入时刷新 delta 基线，避免继承上一客户端上下文（不触发 on_context_reset）
            if not is_doctor:
                if self._seen_agent_client:
                    refresh_epoch = True
                else:
                    self._seen_agent_client = True
        if refresh_epoch:
            self._refresh_context_epoch("external_client_attached", client_name=client["name"],
                                        marker=new_id("client-epoch"))
        self.transcript.event("agent.client_attached", dict(
            client_info=dict(client), protocol_version=bounded(params.get("protocolVersion")) if isinstance(
                params, dict) else None, via="http"))

    def client_view(self):
        """The console's read-only Agent bar data (operator data): ``{"endpoint": the MCP URL or None before start,
        "client": {"name", "version", "self_reported": True} of the newest ``initialize`` or None ("waiting for an
        agent"), "last_call_at": UTC time of the newest admitted tool call or None, "replay": True for a read-only replay
        of an archived run (``STATE_REPLAY``: no MCP server, no live agent; the bar says "archived run")}``."""
        with self.lock:
            bridge = self.bridge
            return dict(endpoint=bridge.url if bridge is not None else None,
                        client=dict(self.client_info) if self.client_info else None, last_call_at=self.last_call_at,
                        replay=self.state == STATE_REPLAY)

    # ---- features an agent NI-RUA runs would have ---------------------------------------------------------------
    def set_model(self, model):
        raise AgentCapabilityError("set_model_unsupported_by_backend")

    def set_effort(self, effort):
        raise AgentCapabilityError("set_effort_unsupported_by_backend")

    def evidence_agent(self, **options):
        raise AgentCapabilityError("evidence_unsupported_by_backend")

    def model_options(self):
        return None

    def _actor_upstream(self):
        return None

    def provenance(self):
        """Manifest ``agent_backend`` block (observer data only)."""
        bridge = self.bridge
        return dict(AgentSession.provenance(self), information_boundary=INFORMATION_BOUNDARY, model_relay=None,
                    native_tools="unrecorded", sandbox="unsupported", isolation=self.isolation,
                    context_delivery=self.context_delivery,
                    context_delivery_policy=dict(effective=self.context_delivery,
                                                 configured=self.config.agent.external.context_delivery),
                    evidence_client=None,
                    tool_transport=dict(kind="mcp_streamable_http", port=self.mcp_port, auth="none",
                                        initialize_instructions=True,
                                        read_only_tools=bridge.read_only_tools if bridge is not None else None))
