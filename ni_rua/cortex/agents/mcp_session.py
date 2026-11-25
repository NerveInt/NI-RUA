"""Shared skeleton of the MCP-tool agent backends (Claude Code, OpenCode): tool bridge, turn state, relay, close.

Design 04 §2.2-§2.3 (state and turn contract), §3.3 (session-side ``call_tool`` closure), §3.7 (shutdown ordering),
§4.6 (counting), §5.2-§5.5 (identity record, ``agent.*`` events, relay usage); plan 07 §1. Both backends drive one
native agent runtime whose registry tools come through the host-owned ``NiRuaMcpServer`` and whose model requests
go through the run's ``ModelRequestProxy``; everything here is backend-neutral and imports no agent SDK.

* ``McpToolBridge`` owns the ``NiRuaMcpServer`` and the ``ToolDispatcher`` of one session. Its ``call_tool`` closure
  is the only path from an MCP ``tools/call`` to ``Harness.handle_tool`` (through ``ToolDispatcher``, so the registry
  ``ToolSpec.parallel`` ordering applies); it writes the ``dynamicToolCall`` transcript row whose id is the host call
  id (== ``callId`` == ``correlation.item_id``) and scrubs the result a second time with ``ActorPresentation`` exactly
  as the Codex transport does (``RecordedClient._approval``).
* ``McpSessionMixin`` holds the turn helpers every backend uses under its session lock (``_begin_turn``,
  ``_reject_turn``, ``_finish_turn``, ``_disconnected``), the context-epoch helpers that report a possible loss of
  earlier model context to the Harness (``on_context_reset``), the relay wiring (budget resume rules,
  ``runtime/model_requests``, relay usage -> ``runtime/token_usage``) and the common ``close()`` order.

Nothing written here is model-visible: transcript rows, events, identity and close reports are observer data.
"""
import threading
import time

from ...core.contracts import new_id
from ...core.presentation.image_reuse import IMAGE_REUSE_POLICY
from ..relay.guard import DEFAULT_MAX_RETRIES, RETRY_POLICY
from ..relay.server import ModelRequestProxy
from ..session.usage import add_served_model
from .base import (AgentRejected, STATE_CLOSED, STATE_DISCONNECTED, STATE_IDLE, STATE_NOT_STARTED, STATE_REPLAY,
                   STATE_RUNNING, TURN_TERMINAL_STATES, UNCONNECTED_STATES)
from .dispatch import EMITTED_ORDER_WAIT_S, EmittedOrder, ToolDispatcher
from .identity import update_identity_shutdown
from .mcp_server import (NiRuaMcpServer, PostExecutionDeliveryError, safe_present_tool_result,
                         tool_bridge_error_response)
from .model_listing import ModelListingMixin

__all__ = ["GENERATION_COUNTING", "MODEL_REQUEST_LIMIT", "STALE_TOOL_STATES", "TURN_ORIGINS", "McpSessionMixin",
           "McpToolBridge"]

# Model-call counting semantics of the new backends (design §4.6): every delivered usable completion on a generation
# route counts toward the episode budget, whatever its purpose (actor, helper, compaction, set_model probe).
GENERATION_COUNTING = "delivered_usable_generation_completions_v1"
# Episode budget of delivered usable model completions (same 180 as Codex; manifest model_request_limit).
MODEL_REQUEST_LIMIT = 180
# Session states in which no native session can own a tool call. ``McpToolBridge`` refuses every call made while the
# session is not ``running`` (these and the terminal/idle states), so a registry tool only ever runs inside a host
# turn whose id it carries; this constant names the states that can never own one.
STALE_TOOL_STATES = UNCONNECTED_STATES | {STATE_NOT_STARTED, STATE_REPLAY}
# ``agent.turn_started`` origins: "host" = run_turn(); "native_wakeup" = the native runtime started a turn by itself
# while the host turn was terminal or idle and the session adopted it (design R4.2, capabilities.native_wakeup);
# "external" = an external agent's session-bracket call (run_begin) opened it (agent.backend "external").
TURN_ORIGINS = ("host", "native_wakeup", "external")


class McpToolBridge:
    """The registry-tool path of one MCP-backend session: ``NiRuaMcpServer`` + ``ToolDispatcher`` (plan 07 §1).

    session: the owning session. Read attributes: ``store``, ``presentation`` (ActorPresentation), ``tool_handler``
        (``Harness.handle_tool``), ``tool_parallel_safe`` (``ToolRegistry.parallel_safe`` or None = every call
        ordered), ``transcript`` (TranscriptWriter), ``closing``, ``state``, ``thread_id``, ``turn_id``; written:
        ``last_activity_at``; called: ``_invalidate_context(source, reason=..., **fields)``.
    tools: registry tool definitions (``ToolRegistry.catalog(runtime, role="actor")``) as passed to ``start()``.
    parallel_hints: ``{tool name: ToolSpec.parallel}`` from ``ToolRegistry.manifest()`` (``session.tool_parallel_hints``);
        drives MCP ``readOnlyHint``. None = every tool ``readOnlyHint: false``.
    server_name: MCP server name, i.e. the agent-side tool prefix (``mcp__ni_rua__`` / ``ni_rua_``); default "ni_rua".
    schema_max_depth: ``UpstreamConfig.tool_schema_max_depth`` or None; folds the published input schemas (see
        ``NiRuaMcpServer``); ``folded_schemas`` reports what was folded.
    handler: the function the ``ToolDispatcher`` runs for each admitted call, ``handler(params) -> result``; None
        (default; every host-mode backend) = ``session.tool_handler`` (``Harness.handle_tool``). A subclass that must
        decide something in dispatch order (the external backend's bracket gate) passes its own.
    server_options: extra keyword arguments of ``NiRuaMcpServer`` (``port``, ``require_bearer``, ``instructions``,
        ``on_initialize``); None (default; every host-mode backend) = the server defaults.
    order_wait_s: bound, in seconds (> 0), of ``EmittedOrder.wait``: how long a call is held for a call the model
        emitted before it in the same step that has not arrived; default ``EMITTED_ORDER_WAIT_S`` (5.0, a documented
        constant: no session passes another value and there is no setting).

    Order of one model step's calls (``order``, an ``EmittedOrder``): the session's relay announces the tool calls of
    every model response in emitted order before the response reaches the client (``McpSessionMixin.
    _model_tool_calls`` -> ``announce_model_step``), and ``_call_tool`` hands each arrived call to the
    ``ToolDispatcher`` in that order, whatever order the client's concurrent ``tools/call`` requests arrived in.
    Observer records (never model context), each correlated with the call id:

    * ``agent.tool_order_unmatched {tool, reason}``: the call has no exact place in an announced step and runs in
      arrival order; ``reason`` is one of ``dispatch.UNMATCHED_REASONS`` (``no_step_known``, ``call_id_unknown``,
      ``no_session_key``, ``session_key_mismatch``, ``arguments_differ``, ``not_in_step``).
    * ``agent.tool_order_wait_expired {tool, position, step_size, matched_by, waited_s, missing_positions}``: the
      bound ran out and the call went ahead. Never a refusal.

    After ``start()``: ``url`` and ``headers`` (per-run bearer) configure the agent's MCP client.
    Shutdown (design §3.7): ``stop_accepting()`` after the agent process is stopped, ``join()`` last (the robot is
    already closed by then, so in-flight handlers return).
    """

    def __init__(self, session, tools, *, parallel_hints=None, server_name="ni_rua", schema_max_depth=None,
                 handler=None, server_options=None, order_wait_s=EMITTED_ORDER_WAIT_S):
        self.session = session
        self.order = EmittedOrder(wait_s=order_wait_s)
        # 非阻塞单主写回租约：任一时刻仅允许一个主链在途调用写入 delta 基线，重叠调用一律降级 isolated 全速并行
        self._lease_lock = threading.Lock()
        self._lease_call_id = None
        self.dispatcher = ToolDispatcher(handler if handler is not None else session.tool_handler,
                                         session.tool_parallel_safe)
        self.server = NiRuaMcpServer(
            tools=tools, presentation=session.presentation, call_tool=self._call_tool,
            raw_log=lambda direction, message: session.store.raw_agent(direction, message, channel="mcp"),
            on_delivery_failure=self._delivery_failure, on_call_settled=self._on_call_settled,
            parallel_hints=parallel_hints, server_name=server_name,
            schema_max_depth=schema_max_depth, **(server_options or {}))
        self.url = self.headers = None
        # call_tool invocations between their admission and their return (their transcript writes included).
        self._active = 0
        self._idle = threading.Condition()

    @property
    def folded_schemas(self):
        """``{tool name: [JSON pointer, ...]}`` of the published input schemas folded for the provider's depth limit
        (provenance; empty without ``schema_max_depth``)."""
        return dict(self.server.folded_schemas)

    @property
    def read_only_tools(self):
        """Tool names published with ``readOnlyHint: true`` (provenance)."""
        return self.server.read_only_tools

    def start(self):
        """Start the loopback endpoint; sets ``url``/``headers``. Returns self."""
        self.server.start()
        self.url, self.headers = self.server.url, dict(self.server.headers)
        return self

    def stop_accepting(self):
        """Close the endpoint (idempotent); in-flight calls are abandoned and reported uncertain by the server."""
        self.server.close()

    def join(self, timeout_s=30.0):
        """Cancel queued tool calls and join running ones (call after the robot is closed), then wait until every
        admitted ``call_tool`` has also finished its own transcript writes on its MCP handler thread: after ``join``
        returns, nothing of the bridge touches the session's Store again (the gateway closes the Store after the
        session). ``timeout_s``: seconds to wait for those handler threads (default 30); exceeded ->
        ``RuntimeError("tool_handlers_still_running")`` (recorded by ``close()`` as a failed step)."""
        self.dispatcher.shutdown(wait=True, cancel_futures=True)
        with self._idle:
            if not self._idle.wait_for(lambda: self._active == 0, timeout=timeout_s):
                raise RuntimeError("tool_handlers_still_running")

    def announce_model_step(self, thread_key, calls, *, tool_name_prefix="", purpose=None):
        """Record the host tool calls of one model response, in emitted order, for ``order``.

        thread_key: the client's session key of the model request (str or None). calls: ``(call_id or None, name,
        arguments)`` as ``WireAdapter.emitted_tool_calls`` returns them. tool_name_prefix: what the agent's client
        puts before this endpoint's tool names (``mcp__ni_rua__``, ``ni_rua_``). purpose: relay call purpose
        (e.g. ``"main"``). Calls of other tools (the agent's native tools, unknown names) are not this endpoint's
        and are left out: they never arrive here.
        """
        published = {tool["name"] for tool in self.server.catalog}
        own = []
        for call_id, name, arguments in calls:
            if isinstance(name, str) and name.startswith(tool_name_prefix) and name[len(tool_name_prefix):] in published:
                own.append((call_id, name[len(tool_name_prefix):], arguments))
        self.order.announce(thread_key, own, purpose=purpose)

    def _call_tool(self, name, arguments, *, call_id, native_call_id, rpc_id, native_thread_id=None,
                   is_abandoned=None):
        """``NiRuaMcpServer.call_tool``: one registry tool call -> ``{"success", "contentItems"}`` (design §3.3).

        Refused with ``ValueError("native_tool_call_stale_connection")`` (answered by the server as an error result,
        nothing dispatched) unless the session is ``running`` a host turn with a thread and a turn id and is not
        closing: a call outside a running turn (after its terminal, while idle, disconnected, closed, not started or
        a replay viewer) would run with no turn that owns it. The state and turn id are read together under the
        session lock. ``params`` are what ``Harness.handle_tool`` receives: ``threadId``, ``turnId``, ``callId``
        (the host call id), ``tool``, ``arguments``, ``callIdSource`` (``"native_tool_use_id"`` when the client
        sent its own id, else ``"mcp_request"``), ``contextScope`` (``"main"`` or ``"isolated"``) and optional
        ``contextEpoch``. A handler exception becomes ``core.tool_errors.error_response``;
        the result is scrubbed once more with ``ActorPresentation.present``.

        The call is handed to the ``ToolDispatcher`` in the model's emitted order among the calls of its step (class
        docstring): it is first correlated with the announced step (``native_call_id``, else ``native_thread_id`` +
        tool + arguments), a refused call resolves its place at once, and an admitted one is held (bounded) until
        every earlier call of the step was submitted or refused. The hold comes before the transcript row, so rows
        start in execution order. After a hold the admission is checked again: a call whose turn ended, changed or
        whose session started closing meanwhile is refused as stale exactly like one arriving then; nothing runs
        and no transcript row is written.
        """
        session = self.session
        ticket, unmatched = self.order.claim(name, arguments, native_call_id=native_call_id,
                                             thread_id=native_thread_id)
        try:
            thread_id, turn_id = self._admit()
            with self._idle:
                self._active += 1
        except BaseException:
            self.order.resolve(ticket)
            raise
        try:
            if ticket is None:
                self._order_event("agent.tool_order_unmatched", call_id, dict(tool=name, reason=unmatched))
            else:
                self.order.wait(ticket)
                if ticket.expired:
                    self._order_event("agent.tool_order_wait_expired", call_id, dict(
                        tool=name, position=ticket.position, step_size=ticket.step_size,
                        matched_by=ticket.matched_by, waited_s=round(ticket.waited_s, 3),
                        missing_positions=list(ticket.skipped)))
                if self._admit() != (thread_id, turn_id):
                    raise ValueError("native_tool_call_stale_connection")
            return self._run_admitted(name, arguments, call_id, native_call_id, thread_id, turn_id, ticket,
                                      native_thread_id=native_thread_id, is_abandoned=is_abandoned)
        finally:
            self.order.resolve(ticket)
            with self._idle:
                self._active -= 1
                self._idle.notify_all()

    def _current_context_epoch(self):
        """读取宿主 Harness 当前模型上下文纪元；若测试桩未挂接 Harness 则返回 None。"""
        session = self.session
        for owner in (
            getattr(session, "harness", None),
            getattr(session.tool_handler, "__self__", None),
            getattr(self.dispatcher.handler, "__self__", None),
        ):
            if owner is None:
                continue
            dc = getattr(owner, "decision_context", None)
            epoch_fn = getattr(dc, "model_context_epoch", None)
            if callable(epoch_fn):
                return epoch_fn()
            legacy_fn = getattr(owner, "_model_context_epoch", None)
            if callable(legacy_fn):
                return legacy_fn()
        return None

    def _release_lease(self, call_id):
        with self._lease_lock:
            if self._lease_call_id == call_id:
                self._lease_call_id = None

    def _on_call_settled(self, call_id, delivered):
        """``NiRuaMcpServer.on_call_settled``：HTTP 写回结束或失败退出后释放单主写回租约。"""
        self._release_lease(call_id)

    def _admit(self):
        """``(thread_id, turn_id)`` of the running host turn a call may run in, read together under the session
        lock; ``ValueError("native_tool_call_stale_connection")`` when there is none."""
        session = self.session
        with session.lock:
            thread_id, turn_id = session.thread_id, session.turn_id
            if (session.closing or session.state != STATE_RUNNING or thread_id is None or turn_id is None):
                raise ValueError("native_tool_call_stale_connection")
            return thread_id, turn_id

    def _record_bridge_error(self, where, call_id, error):
        """Record an observer-side bridge error without raising into tool delivery."""
        self.server.errors.append((f"{where}:{call_id}", repr(error)))
        handler_error = getattr(self.session, "_handler_error", None)
        if callable(handler_error):
            try:
                handler_error(where, error)
            except Exception:  # noqa: BLE001
                pass

    def _run_admitted(self, name, arguments, call_id, native_call_id, thread_id, turn_id, ticket=None,
                      *, native_thread_id=None, is_abandoned=None):
        session = self.session
        caller_is_main = (
            ticket is not None
            and ticket.purpose == "main"
            and (native_thread_id is None or native_thread_id == session.thread_id)
        )
        context_scope = "isolated"
        context_epoch = None
        if caller_is_main and getattr(session, "context_delivery", "delta") == "delta":
            abandoned_check = is_abandoned if callable(is_abandoned) else (lambda: False)
            with self._lease_lock:
                if self._lease_call_id is None:
                    candidate_epoch = self._current_context_epoch()
                    if not abandoned_check():
                        self._lease_call_id = call_id
                        context_scope = "main"
                        context_epoch = candidate_epoch
        params = dict(
            threadId=thread_id,
            turnId=turn_id,
            callId=call_id,
            tool=name,
            arguments=arguments,
            callIdSource="native_tool_use_id" if native_call_id else "mcp_request",
            contextScope=context_scope,
        )
        if context_epoch is not None:
            params["contextEpoch"] = context_epoch
        session.transcript.tool_started(call_id, name, arguments, native_id=native_call_id, turn_id=turn_id)
        session.last_activity_at = time.monotonic()
        executed = False
        try:
            try:
                future = self.dispatcher.submit(params)
            finally:
                self.order.resolve(ticket)
            result = future.result()
            executed = True
        except Exception as error:  # noqa: BLE001 - answered like the Codex transport (RecordedClient._approval)
            result = tool_bridge_error_response(error, tool=name, arguments=arguments, store=session.store,
                                                call_id=call_id, errors_sink=self.server.errors)
        if executed:
            try:
                result = session.presentation.present(result)
            except Exception as present_error:  # noqa: BLE001 - never claim "no motion" after tool execution
                self._record_bridge_error("present_tool_result", call_id, present_error)
                raise PostExecutionDeliveryError("invalid_tool_result") from present_error
        else:
            result = safe_present_tool_result(session.presentation, result)
        items = result.get("contentItems") if isinstance(result, dict) else None
        try:
            session.transcript.tool_completed(call_id, success=bool(isinstance(result, dict) and result.get("success")),
                                              content_items=items)
        except Exception as log_error:  # noqa: BLE001 - never discard an executed physical ToolResponse receipt
            self._record_bridge_error("tool_completed", call_id, log_error)
        session.last_activity_at = time.monotonic()
        return result

    def _order_event(self, kind, call_id, payload):
        """Record one order observer event (class docstring) for ``call_id``. Never raises: the call must still run."""
        try:
            self.session.transcript.event(kind, payload, tool_call_id=call_id, item_id=call_id)
        except Exception as error:  # noqa: BLE001
            self._record_bridge_error("order_event", call_id, error)

    def _delivery_failure(self, call_id, reason):
        """``NiRuaMcpServer.on_delivery_failure``: the agent may not have received this tool result (never raises).

        The tool may have taken effect, so evidence delivered with it can no longer be assumed in model context:
        ``runtime/image_context_epoch`` changes, ``agent.tool_reply_unknown`` is recorded and the Harness is told
        (``on_context_reset``), exactly once per call (the server guarantees once).
        """
        session = self.session
        try:
            try:
                session._invalidate_context(reason, reason=reason, tool_call_id=call_id,
                                            marker=new_id("image-delivery-failure"))
            except Exception as error:  # noqa: BLE001 - must not raise into the MCP server
                self._record_bridge_error("delivery_failure_invalidate_context", call_id, error)
        finally:
            self._release_lease(call_id)
        try:
            session.transcript.event("agent.tool_reply_unknown", dict(reason=reason), tool_call_id=call_id,
                                     item_id=call_id)
        except Exception as error:  # noqa: BLE001
            self._record_bridge_error("delivery_failure_event", call_id, error)


class McpSessionMixin(ModelListingMixin):
    """Turn state, context-epoch, relay and close helpers shared by the MCP-backend sessions (plan 07 §1).

    The concrete session sets (instance attributes): ``lock`` (``threading.RLock``; every helper here takes it),
    ``state``, ``thread_id``, ``turn_id``, ``turn_done`` (Event), ``turn_result``, ``transcript`` (TranscriptWriter,
    created in ``start()``), ``closing``, ``store``, ``private`` (run ``private/`` dir), ``config``, ``model``,
    ``provider``, ``wire`` (WireAdapter), ``model_proxy``, ``model_limit_reached`` (Event), ``bridge``
    (McpToolBridge or None), ``last_activity_at``, ``price_card``, ``on_disconnect``, ``on_context_reset``,
    ``transport_recovery_gate``. It implements ``_stop_backend()`` (stop the native process tree;
    called by ``close()`` between relay cancellation and relay close) and may set ``_teardown_confirmed`` (bool).
    Class attribute ``counting`` (``GENERATION_COUNTING``).

    Lock order: the session lock may be taken while holding no other lock; callers never hold it across a blocking
    wait on the backend's own I/O thread (that thread takes the lock to deliver messages).
    """

    # Delivered completions per served model; rebound (never mutated) by ``_model_event``, seeded by ``_start_relay``.
    _served_models = {}
    # Provider model listing (``model_options`` suggestions; OpenCode also reads limits and the image flag of a model
    # it declares from it): ``ModelListingMixin`` (``model_catalogue_sources``, ``model_catalogue_timeout_s``,
    # ``catalogue_cache``, ``_catalogue``, ``_start_model_catalogue``, ``_await_model_catalogue``, ``model_options``).

    # ---- turn helpers (call with no lock or with self.lock held; RLock) -----------------------------------
    def _check_turn_admissible(self):
        """Raise ``ValueError`` unless a new turn may start now (``turn_already_active`` while running,
        ``session_not_connected`` before start, in replay or in ``UNCONNECTED_STATES``)."""
        with self.lock:
            if self.state == STATE_RUNNING:
                raise ValueError("turn_already_active")
            if (self.state in UNCONNECTED_STATES or self.state in (STATE_NOT_STARTED, STATE_REPLAY)
                    or self.closing):
                raise ValueError("session_not_connected")

    def _begin_turn(self, origin="host"):
        """Enter ``running`` with a new host-minted turn id (admission step 1 for ``run_turn``, or adoption).

        origin: ``"host"`` (``run_turn``; refused like ``_check_turn_admissible``), ``"native_wakeup"`` (the
        native runtime started a turn while the host turn was terminal or idle) or ``"external"`` (an external
        agent's session-bracket call); the last two have no admissibility check, the caller decides. The
        ``agent.turn_started {origin}`` event is written FIRST: if the Store refuses it, the exception propagates
        and nothing changed (previous state, ``turn_id``, ``turn_result`` and a set ``turn_done`` are kept, so no
        waiter can hang). Only then ``turn_done`` and ``turn_result`` are cleared, ``active_turn_model``
        is set to ``model`` and ``turn_origin`` to ``origin``. Returns the previous ``(state, turn_id, turn_result,
        active_turn_model, transcript turn id, turn_origin)`` for ``_reject_turn``.
        """
        if origin not in TURN_ORIGINS:
            raise ValueError("invalid_turn_origin")
        with self.lock:
            if origin == "host":
                self._check_turn_admissible()
            previous = (self.state, self.turn_id, self.turn_result, getattr(self, "active_turn_model", None),
                        self.transcript.turn_id, getattr(self, "turn_origin", None))
            turn_id = new_id("turn")
            self.transcript.event("agent.turn_started", dict(origin=origin), turn_id=turn_id)
            with self.transcript.lock:
                self.transcript.turn_id = turn_id
            self.turn_id = turn_id
            self.turn_result = None
            self.active_turn_model = self.model
            self.turn_origin = origin
            self.state = STATE_RUNNING
            self._interrupt_requested = False
            self.turn_done.clear()
            return previous

    def _reject_turn(self, previous, code, message, *, busy=False):
        """Definite rejection after ``_begin_turn``: nothing was admitted. Emits ``agent.turn_start_rejected``
        (correlated with the rejected turn id), restores the previous state, sets ``turn_done`` and raises
        ``AgentRejected(code, message, busy)``."""
        with self.lock:
            rejected = self.turn_id
            try:
                self.transcript.event("agent.turn_start_rejected", dict(code=code, message=message, busy=bool(busy),
                                                                        state_restored=True), turn_id=rejected)
            finally:
                (self.state, self.turn_id, self.turn_result, self.active_turn_model,
                 self.transcript.turn_id, self.turn_origin) = previous
                self.turn_done.set()
        raise AgentRejected(code, message, busy=busy)

    def _finish_turn(self, status, error=None):
        """Native terminal of the current turn: ``status`` completed|interrupted|failed, ``error`` a host-minted
        message (no provider text) or None. Idempotent per turn (no-op once ``turn_done`` is set, after a
        disconnect or close). Stamps final-answer phases (completed only), emits ``agent.turn_completed``, sets
        ``turn_result = {"id", "status", "error": {"message"} | None}``, ``state = status``, ``turn_done``.
        Returns True when this call finished the turn."""
        with self.lock:
            if (self.turn_done.is_set() or self.state != STATE_RUNNING or self.closing
                    or status not in TURN_TERMINAL_STATES):
                return False
            try:
                self.transcript.turn_completed(status, error=error, turn_id=self.turn_id)
            finally:
                self.turn_result = dict(id=self.turn_id, status=status, error=dict(message=error) if error else None)
                self.state = status
                self.turn_done.set()
            return True

    def _disconnected(self, error):
        """Transport loss or unknown admission (same semantics as ``CodexSession._disconnected``; design §2.2).

        Once per session and never while closing: ``state = "disconnected"``; a running turn gets the synthetic
        ``turn_result {"status": "failed", "error": {"message": "upstream_connection_lost: <ErrorType>"}}``;
        ``agent.connection_lost {reason, detail, history_complete: false}``; ``on_disconnect()`` (Harness.request_stop:
        motion admission closes) and finally ``turn_done.set()``. Never retried.
        """
        with self.lock:
            if self.closing or self.state in (STATE_DISCONNECTED, STATE_CLOSED):
                return
            self.state = STATE_DISCONNECTED
            if not self.turn_done.is_set():
                self.turn_result = dict(id=self.turn_id, status="failed",
                                        error=dict(message="upstream_connection_lost: " + type(error).__name__))
            try:
                if self.transcript is not None:
                    self.transcript.event("agent.connection_lost", dict(
                        reason=type(error).__name__, detail=str(error)[:200], history_complete=False))
            except Exception:  # noqa: BLE001 - the stop below must still happen
                pass
            finally:
                try:
                    if self.on_disconnect:
                        self.on_disconnect()
                finally:
                    self.turn_done.set()

    # ---- context epoch (design §4.3/§4.5) -----------------------------------------------------------------
    def _context_reset(self, reason):
        """Tell the Harness the model may have lost earlier context (``on_context_reset``, contract C2). The caller
        has already written ``runtime/image_context_epoch``. Never raises."""
        callback = self.on_context_reset
        if callback is None:
            return
        try:
            callback(reason)
        except Exception:  # noqa: BLE001 - a listener failure must not break the agent transport
            pass

    def _invalidate_context(self, source, *, reason=None, **fields):
        """Write a fresh ``runtime/image_context_epoch = {source, thread_id, marker, **fields}`` (so later tool
        results re-deliver full evidence) and always call ``_context_reset(reason or source)``.

        source: why context may be gone (``native_context_compaction``, ``relay_context_discontinuity``,
        ``tool_reply_delivery_uncertain``, ``tool_reply_encoding_failed``, ``agent_resume_after_unclean_exit``).
        fields: observer-only details (e.g. ``signal``, ``tool_call_id``); ``marker`` defaults to a new id."""
        record = dict(source=source, thread_id=self.thread_id, **fields)
        record.setdefault("marker", new_id("context-epoch"))
        try:
            self.store.set_state("runtime", "image_context_epoch", record)
        finally:
            self._context_reset(reason or source)

    def _relay_discontinuity(self, reason):
        """``ModelRequestProxy.on_context_discontinuity(reason)``: the relay's continuity tracker proved that the
        model no longer holds context it was given (``history_shortened``, ``history_rewritten``, ``images_removed``,
        ``system_changed``, ``provider_context_edit``). Same effect as a native compaction signal with
        ``source="relay_context_discontinuity"``: new ``runtime/image_context_epoch``, ``agent.compaction`` and
        ``on_context_reset``. Called on a relay handler thread before that request is forwarded, outside the proxy
        lock; ``PayloadMixin._context_discontinuity`` records ``model_context_discontinuity_callback_failed`` if
        ``_invalidate_context`` fails."""
        self._invalidate_context("relay_context_discontinuity", signal=str(reason)[:120],
                                 marker=new_id("compaction"))
        try:
            if self.transcript is not None:
                self.transcript.event("agent.compaction", dict(signal=str(reason)[:120], source="relay"))
        except Exception as error:  # noqa: BLE001 - optional observer event
            handler_error = getattr(self, "_handler_error", None)
            if callable(handler_error):
                try:
                    handler_error("relay_compaction_event", error)
                except Exception:  # noqa: BLE001
                    pass

    # ---- relay -------------------------------------------------------------------------------------------------
    def _reopen_relay_thread(self):
        """Before a new turn: lift the relay's cancellation tombstone on this session's thread.

        For non-Codex client profiles ``ModelRequestProxy.cancel_pending`` (called by ``interrupt()``) tombstones the
        thread of every cancelled call, so a client re-send of the interrupted turn is denied instead of charged;
        the session's next host turn must reopen it or every request of that turn is denied ``thread_cancelled``.
        Returns True when a tombstone was lifted (the proxy records ``model_thread_reopened``)."""
        proxy = self.model_proxy
        if proxy is None or self.thread_id is None:
            return False
        return proxy.reopen_thread(self.thread_id)

    def _transport_recovery_block(self):
        """``ModelRequestProxy.recovery_gate``: reason a model resend is unsafe now, or None (Workbench's gate)."""
        gate = self.transport_recovery_gate
        return "recovery_gate_unavailable" if gate is None else gate()

    def _relay_budget(self):
        """Budget resume rules (copied from ``CodexSession._prepare``): returns ``(used, previous_attempts,
        transport_recovery_policy)``. Raises ``resume_model_budget_semantics_changed`` when the archived
        ``runtime/model_requests.counting`` differs and ``episode_model_request_limit_exhausted`` at 180 used calls.
        A resumed run keeps the recovery policy it started with."""
        previous = self.store.read_state("runtime", "model_requests", {}) or {}
        if previous and previous.get("counting") != self.counting:
            raise ValueError("resume_model_budget_semantics_changed: choose a fresh run")
        used = previous.get("completed_calls", 0)
        if used >= MODEL_REQUEST_LIMIT:
            raise ValueError("episode_model_request_limit_exhausted")
        recovery = ((previous.get("transport_recovery") or {}).get("policy") if previous
                    else self.config.model.transport_recovery.to_dict())
        return used, previous.get("http_attempt_count", 0), recovery

    def provider_compat_block(self, upstream):
        """``agent_backend.provider_compat`` provenance (operator data, never model context) for the session's actor
        ``upstream`` (``UpstreamConfig``): ``tool_result_images`` (``"inline"`` | ``"user_blocks"``) and
        ``tool_schema_max_depth`` (int or None) as configured, the depth measure, and ``folded_schemas`` (``{tool:
        [JSON pointer, ...]}``) published by the MCP bridge."""
        from .mcp_server import SCHEMA_DEPTH_MEASURE
        bridge = getattr(self, "bridge", None)
        return dict(tool_result_images=upstream.tool_result_images, tool_schema_max_depth=upstream.tool_schema_max_depth,
                    schema_depth_measure=SCHEMA_DEPTH_MEASURE if upstream.tool_schema_max_depth is not None else None,
                    folded_schemas=bridge.folded_schemas if bridge is not None else None)

    # What the agent's client puts before the host MCP endpoint's tool names in model requests (``mcp__ni_rua__``,
    # ``ni_rua_``); the concrete session sets it.
    tool_name_prefix = ""

    def _model_tool_calls(self, thread_key, calls, purpose=None):
        """``ModelRequestProxy.on_tool_calls``: the tool calls of one model response in emitted order, before the
        response reaches the agent. Passed to the bridge so one step's concurrent calls run in that order
        (``McpToolBridge.announce_model_step``). Relay handler thread; takes no session lock."""
        bridge = getattr(self, "bridge", None)
        if bridge is not None:
            bridge.announce_model_step(thread_key, calls, tool_name_prefix=self.tool_name_prefix, purpose=purpose)

    on_tool_calls = _model_tool_calls

    def _start_relay(self, upstream_url, *, upstream_auth_headers, request_projection, default_thread_id,
                     image_settings, tool_result_images="inline", request_context=None):
        """Create and start the run's ``ModelRequestProxy`` for this backend's ``wire`` (plan 07 §1 relay wiring).

        upstream_url: provider base URL; upstream_auth_headers: host credential headers (never seen by the agent);
        request_projection: egress scrub ``request -> request``; default_thread_id: thread attributed to requests
        without a registered thread key; image_settings: the run's ``store_image_settings`` (existing relay image
        trim/reuse logic, unchanged); tool_result_images: ``UpstreamConfig.tool_result_images`` (``"inline"`` |
        ``"user_blocks"``, see ``ModelRequestProxy``); request_context: optional ``(request, identity) -> request``
        hook run once per actor generation call before the egress projection (``ModelRequestProxy``; default None).
        Budget = ``180 - used``; events -> ``_model_event``; budget exhaustion sets ``model_limit_reached``. Sets and
        returns ``self.model_proxy``."""
        used, attempts, recovery = self._relay_budget()
        self._relay_used, self._relay_attempts = used, attempts
        self._attempt_usage = {}
        # Delivered completions per provider-reported (served) model, carried across resumes.
        self._served_models = dict((self.store.read_state("runtime", "model_requests", {}) or {}).get("served_models") or {})
        self.model_limit_reached.clear()
        self.store.set_state("runtime", "model_image_policy", dict(
            enabled=image_settings["reuse_exact_images"], policy=IMAGE_REUSE_POLICY, compaction="unchanged",
            oversized_trim=image_settings["trim_oversized_requests"],
            max_inline_images=image_settings["trim_max_inline_images"]))
        proxy = ModelRequestProxy(
            upstream_url, upstream_auth_headers=upstream_auth_headers, wire=self.wire,
            default_thread_id=default_thread_id, max_requests=MODEL_REQUEST_LIMIT - used, request_timeout_s=None,
            max_retries=DEFAULT_MAX_RETRIES, on_event=self._model_event,
            on_limit=lambda _event: self.model_limit_reached.set(),
            on_context_discontinuity=self._relay_discontinuity,
            image_reuse=image_settings["reuse_exact_images"], image_trim=image_settings["trim_oversized_requests"],
            max_inline_images=image_settings["trim_max_inline_images"], transport_recovery=recovery,
            recovery_gate=self._transport_recovery_block, request_projection=request_projection,
            metrics_path=self.private / (new_id("model-requests") + ".jsonl"), tool_result_images=tool_result_images,
            request_context=request_context, on_tool_calls=self._model_tool_calls)
        self.model_proxy = proxy
        proxy.start()
        return proxy

    def _model_event(self, event):
        """``ModelRequestProxy.on_event`` (relay callback thread): ``runtime/model_requests`` + ``model.<kind>``
        event as Codex writes them, and relay usage accounting (design §5.5).

        Usage: the ``usage`` of an attempt's ``model_request_finished`` is kept per logical call and added once,
        normalised by the wire, when that call is ``model_call_completed`` (delivered and charged) on this session's
        thread; ``runtime/token_accounting`` is re-derived each time (source ``relay.<wire>.usage.sum``). A delivered
        call without usage still counts (its counters become unknown, never zero)."""
        proxy = self.model_proxy
        if proxy is None:
            return
        kind, call = event.get("kind"), event.get("logical_call_id")
        if kind == "model_request_finished" and call is not None:
            self._attempt_usage[("served", call)] = event.get("served_model")
        elif kind == "model_call_completed" and call is not None:
            self._served_models = add_served_model(self._served_models, self._attempt_usage.pop(("served", call), None))
        elif kind == "model_call_finished" and call is not None:
            self._attempt_usage.pop(("served", call), None)
        self.store.set_state("runtime", "model_requests", dict(
            completed_calls=self._relay_used + proxy.request_count,
            http_attempt_count=self._relay_attempts + proxy.attempt_count, counting=self.counting,
            limit=MODEL_REQUEST_LIMIT, timeout_s=None, max_retries=DEFAULT_MAX_RETRIES, retry_policy=RETRY_POLICY,
            transport_blocked=proxy.transport_blocked, transport_error=proxy.blocked_error,
            transport_recovery=proxy.transport_recovery_summary, served_models=dict(self._served_models),
            last_event=event))
        self.store.event("model." + event["kind"], event, source="private_model_transport")
        if kind == "model_request_finished" and call is not None:
            self._attempt_usage[call] = event.get("usage") if event.get("usage_known") else None
        elif kind == "model_call_completed" and call is not None:
            usage = self._attempt_usage.pop(call, None)
            if event.get("native_thread_id") == self.thread_id and self.transcript is not None:
                self.transcript.add_usage(self.wire.normalize_usage(usage))
                self.transcript.write_accounting(source=f"relay.{self.wire.name}.usage.sum",
                                                 model=getattr(self, "active_turn_model", None) or self.model,
                                                 provider=self.provider, price_card=self.price_card)
                self.last_activity_at = time.monotonic()
        elif kind == "model_call_finished" and call is not None:
            self._attempt_usage.pop(call, None)

    # ---- previous native processes (resume) -------------------------------------------------------------------
    def _stop_previous_instance(self, previous):
        """Resume: stop what a previous session of this run left running (a crashed gateway's CLI exits with its
        stdin, but a detached server or tool child keeps running and could act on the run's files or native state).
        Finds the processes carrying the previous instance's marker (``isolation.AGENT_MARKER_ENV``) or orphaned in
        the actor dir, kills and verifies them (``isolation.terminate_agent_processes``); ``agent.previous_process_
        stopped`` records it when anything was found. Raises ``RuntimeError("previous_agent_process_stop_unconfirmed")``
        when it cannot verify that they are gone (the resumed session would share the run with them)."""
        from . import isolation
        instance = (previous or {}).get("instance_id")
        instance = instance if isinstance(instance, str) and isolation._MARKER_VALUE.match(instance) else None
        result = isolation.terminate_agent_processes(isolation.TreeSnapshot(0, ()), instance,
                                                     cwd_roots=[self.actor_dir])
        stopped = [item["pid"] for item in result.marker.get("found", [])]
        if stopped or not result.confirmed:
            self.store.event("agent.previous_process_stopped", dict(pids=stopped, stop_confirmed=result.confirmed,
                                                                    scan_errors=list(result.errors)),
                             dict(thread_id=(previous or {}).get("thread_id")), source="agent_backend")
        if not result.confirmed:
            raise RuntimeError("previous_agent_process_stop_unconfirmed")

    # ---- close (design §3.7) ----------------------------------------------------------------------------------
    def _stop_backend(self):
        """Backend hook: stop the native agent process tree (called once by ``close()``). Default: nothing."""

    def _native_active(self):
        """Backend hook: True while the native runtime reports a turn in progress (whatever the host state), so a
        close at that moment is not a clean boundary. Default: False."""
        return False

    def close(self):
        """Release every owned resource; idempotent and serialised (design §3.7, plan 07 §1).

        Order (the Workbench has already closed the robot): ``_close_model_catalogue()`` (a listing read still
        running records nothing any more) -> ``closing = True`` (new tool calls are stale) ->
        ``model_proxy.cancel_pending("session_closed")`` -> ``_stop_backend()`` -> ``model_proxy.close()`` ->
        MCP endpoint closed -> tool workers joined. Each step runs in its own try; outcomes and errors are kept in
        ``close_report`` (``{"steps": {name: "completed"|"failed"}, "errors": [...], "backend": {...}}``). Then
        ``runtime/agent.clean_shutdown`` is written (True only when, at close, no host turn was running, the session
        was not disconnected, the native runtime reported no turn in progress (``_native_active``), every step
        succeeded and the backend confirmed teardown), ``agent.session_closed`` is recorded,
        ``state = "closed"`` and ``turn_done`` is set.

        Late-writer record (observer data, never model context): ``close_report["model_listing_pending"]`` is True
        when the provider model listing was still being read, and a backend names its reader threads that outlived
        their bounded join in ``close_report["backend"]["threads_alive"]``. What such a thread delivers after the
        run's Store is closed is refused (``Store.refused_after_close``), so ``agent.session_closed`` carries
        ``model_listing_pending: true`` and ``threads_alive: [names]`` while the Store is still open; both keys are
        absent on a close without them.
        """
        close_lock = self.__dict__.setdefault("_close_lock", threading.Lock())
        listing_pending = self._close_model_catalogue()
        with close_lock:
            with self.lock:
                if self.state == STATE_CLOSED:
                    return self.close_report
                self.closing = True
                at_close = self.state
                native_active = bool(self._native_active())
            report = self.close_report = dict(steps={}, errors=[], state_at_close=at_close,
                                              model_listing_pending=listing_pending)

            def step(name, operation):
                try:
                    operation()
                    report["steps"][name] = "completed"
                except Exception as error:  # noqa: BLE001 - every step is independent
                    report["steps"][name] = "failed"
                    report["errors"].append(dict(step=name, error=f"{type(error).__name__}: {error}"[:300]))

            proxy, bridge = self.model_proxy, getattr(self, "bridge", None)
            if proxy is not None:
                step("relay_cancel", lambda: proxy.cancel_pending("session_closed"))
            step("backend", self._stop_backend)
            if proxy is not None:
                step("relay", proxy.close)
            if bridge is not None:
                step("mcp", bridge.stop_accepting)
                step("tool_workers", bridge.join)
            identity = getattr(self, "_identity", None)
            clean = (at_close in (STATE_IDLE, *TURN_TERMINAL_STATES) and not native_active
                     and getattr(self, "_teardown_confirmed", True) is True and not report["errors"])
            report["clean_shutdown"] = clean
            report["native_active_at_close"] = native_active
            if identity is not None:
                step("identity", lambda: update_identity_shutdown(self.store, identity, clean_shutdown=clean))
            if self.transcript is not None:
                late = dict(model_listing_pending=True) if listing_pending else {}
                threads_alive = (report.get("backend") or {}).get("threads_alive")
                if threads_alive:
                    late["threads_alive"] = list(threads_alive)
                step("event", lambda: self.transcript.event("agent.session_closed", dict(
                    clean_shutdown=clean, state_at_close=at_close, steps=dict(report["steps"]),
                    errors=len(report["errors"]), **late)))
            with self.lock:
                self.state = STATE_CLOSED
                self.turn_done.set()
            return report
