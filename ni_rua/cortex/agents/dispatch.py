"""Ordered concurrent dispatch of agent tool calls.

Agent backends may deliver several tool calls at once. Read-only calls may overlap, but a call that can change
Harness or world state must observe every earlier call's effect and must not race a later read. The policy is
defined once here and the per-call verdict comes from the central tool registry contract
(``ToolSpec.parallel`` via ``ToolRegistry.parallel_safe``); there is no name list in the transport.

Ordering rule of ``OrderedBarrier`` / ``ToolDispatcher`` (unchanged from the original Codex ``RecordedClient``
dispatcher), in SUBMISSION order:

* a parallel-safe call starts after the most recent ordered call has finished and may overlap other parallel-safe
  calls;
* any other (ordered) call starts only after every parallel-safe call admitted before it and the previous ordered
  call have finished.

So execution order equals submission order for every ordered call. Without a classifier every call is ordered, which
is the safe direction.

Submission order is the caller's. A transport that reads one ordered stream (Codex dynamic tools) submits in the
order the model emitted the calls. An MCP client may send the ``tools/call`` requests of one model step concurrently
over separate connections, so their arrival order is decided by scheduling (OpenCode 2.2.7: 4 of 96 steps with three
ordered calls arrived inverted under load). ``EmittedOrder`` restores the model's order for such a transport when
the order is known (the run's relay saw the model response): the bridge passes each call through it before
``ToolDispatcher.submit``. Where no order is known (no relay: external / plugin mode) submission order stays arrival
order.
"""
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait as wait_futures


def is_parallel_call(parallel_safe, params):
    """Return True only when ``parallel_safe(tool, arguments)`` explicitly returns True for ``params``.

    ``parallel_safe`` is a callable ``(tool_name: str, arguments: dict) -> bool`` such as
    ``ToolRegistry.parallel_safe``, or ``None`` (every call ordered). ``params`` is the native tool-call payload
    with ``tool`` and ``arguments`` keys. A missing classifier, a non-object payload, a non-``True`` verdict or a
    classifier exception all yield False, so a malformed call can never stop the transport's reader.
    """
    if parallel_safe is None or not isinstance(params, dict):
        return False
    try:
        return parallel_safe(params.get("tool"), params.get("arguments")) is True
    except Exception:
        return False


def _run_after(deps, fn, args):
    if deps:
        wait_futures(deps)
    return fn(*args)


class OrderedBarrier:
    """Pool-agnostic ordering barrier between parallel-safe and ordered calls.

    The barrier never runs work itself: ``schedule`` submits a wrapper to the caller's executor that first waits for
    the call's dependencies (see the module docstring for the rule), then calls ``fn(*args)``. A dependency's
    exception or cancellation only ends the wait; it is never re-raised in a later call, so one failed call cannot
    deadlock or poison the chain. Dependencies are always submitted earlier, so a FIFO executor such as
    ``ThreadPoolExecutor`` cannot deadlock on them. Thread-safe; the arrival order is the order in which
    ``schedule`` acquires the internal lock.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._active_parallel = []
        self._last_ordered = None

    def schedule(self, pool, parallel, fn, *args):
        """Submit ``fn(*args)`` to ``pool`` behind its dependencies and return the executor's ``Future``.

        ``pool`` is any executor with ``submit`` (e.g. ``ThreadPoolExecutor``). ``parallel`` is the call's
        classification: True for a parallel-safe call, anything else for an ordered call. The returned future
        carries ``fn``'s result or exception. Errors raised by ``pool.submit`` (e.g. after shutdown) propagate.
        """
        with self._lock:
            self._active_parallel = [future for future in self._active_parallel if not future.done()]
            ordered_dep = ([self._last_ordered]
                           if self._last_ordered is not None and not self._last_ordered.done() else [])
            if parallel is True:
                future = pool.submit(_run_after, ordered_dep, fn, args)
                self._active_parallel.append(future)
                return future
            deps = [*self._active_parallel, *ordered_dep]
            self._active_parallel = []
            future = pool.submit(_run_after, deps, fn, args)
            self._last_ordered = future
            return future


class ToolDispatcher:
    """Own a worker pool that runs native tool calls under the ``OrderedBarrier`` policy.

    ``handler(params) -> result`` executes one tool call (normally ``Harness.handle_tool`` behind the session's
    stale-connection checks); it runs on a worker thread, never on the caller's thread. ``parallel_safe`` is a
    ``(tool_name, arguments) -> bool`` classifier (``ToolRegistry.parallel_safe``) or ``None`` to order every call.
    ``max_workers`` bounds concurrently running calls (default 8, as the Codex transport); ``thread_name_prefix``
    names the worker threads for diagnostics.
    """

    def __init__(self, handler, parallel_safe, *, max_workers=8, thread_name_prefix="agent-tools"):
        self.handler = handler
        self.parallel_safe = parallel_safe
        self.pool = ThreadPoolExecutor(max_workers=max_workers, thread_name_prefix=thread_name_prefix)
        self.barrier = OrderedBarrier()

    def submit(self, params):
        """Schedule ``handler(params)`` and return its ``Future``.

        ``params`` is the native tool-call payload (at least ``tool`` and ``arguments``); it is classified with
        ``is_parallel_call`` at submission, so submission order is the arrival order. Raises ``RuntimeError`` after
        ``shutdown``.
        """
        return self.barrier.schedule(self.pool, is_parallel_call(self.parallel_safe, params), self.handler, params)

    def shutdown(self, *, wait=True, cancel_futures=True):
        """Stop accepting calls; by default cancel queued calls and join the running ones.

        Join only after the robot is closed (session close ordering), so in-flight handlers can return.
        """
        self.pool.shutdown(wait=wait, cancel_futures=cancel_futures)


# Bound, in seconds, of ``EmittedOrder.wait``: how long a call is held for an earlier call of its model step that
# has not arrived at all. It is a liveness bound, not a task decision, and deliberately a constant (no setting).
# Rationale: a client that sends one step's calls concurrently delivers them together. Measured with OpenCode 2.2.7,
# three ordered calls per step: the three ``tools/call`` requests were logged 0.6 to 8.3 ms apart, and at most 10 ms
# apart under 8-way load (review 29). 5 s is 500 times that spread, so the bound never decides between calls that
# are in flight; it only limits the wait for an earlier sibling the client never sends (it refused the call locally:
# unknown tool, arguments its own schema check rejected, a permission rule, an interrupted step). Expiry is recorded
# by the caller and is never a refusal. Worst case for one call: see ``EmittedOrder`` ("Bound").
EMITTED_ORDER_WAIT_S = 5.0
# Announced model steps kept for correlation (oldest dropped first). A step is needed only while its calls are
# arriving, i.e. while it is the newest step of its client session (the main agent and each running subagent have
# one); 64 is far above the number of sessions one run has open at once, and a call of a dropped step is merely
# unmatched (arrival order, recorded).
EMITTED_ORDER_STEPS = 64

_PENDING, _ARRIVED, _RESOLVED, _SKIPPED = "pending", "arrived", "resolved", "skipped"

# Why ``EmittedOrder.claim`` found no place for a call (it then runs in arrival order).
UNMATCHED_NO_STEP = "no_step_known"                  # no model step was announced at all
UNMATCHED_CALL_ID = "call_id_unknown"                # the client's call id is not an open call of any kept step
UNMATCHED_NO_SESSION_KEY = "no_session_key"          # the client sent neither a call id nor a session key
UNMATCHED_SESSION_KEY = "session_key_mismatch"       # no kept step was announced under the client's session key
UNMATCHED_ARGUMENTS = "arguments_differ"             # the step has an open call of this tool, with other arguments
UNMATCHED_NOT_IN_STEP = "not_in_step"                # the step has no open call of this tool (none emitted, or taken)
UNMATCHED_REASONS = (UNMATCHED_NO_STEP, UNMATCHED_CALL_ID, UNMATCHED_NO_SESSION_KEY, UNMATCHED_SESSION_KEY,
                     UNMATCHED_ARGUMENTS, UNMATCHED_NOT_IN_STEP)


def _same_arguments(left, right):
    """True when two decoded JSON argument values are the same JSON value (key order ignored; ``1`` and ``1.0``
    equal, as a client that re-serialises numbers may send either; ``true`` never equals ``1``)."""
    try:
        return left == right and _bool_shape(left) == _bool_shape(right)
    except (TypeError, ValueError, RecursionError):
        return False


def _bool_shape(value):
    if isinstance(value, dict):
        return {key: _bool_shape(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_bool_shape(item) for item in value]
    return isinstance(value, bool)


class OrderTicket:
    """One arrived call's place in its model step (``EmittedOrder.claim``); read-only for the caller.

    matched_by: how the call was correlated with the model's emitted call: ``"native_call_id"`` (the client sent the
        model's tool-call id) or ``"arguments"`` (same client session key, tool name and JSON arguments).
    position: 0-based index of the call among the host tool calls of its step, in the model's emitted order.
    step_size: number of host tool calls in that step.
    After ``EmittedOrder.wait``: ``waited_s`` (seconds held, float), ``expired`` (True when the bound ran out) and
    ``skipped`` (positions of earlier calls that had not arrived when it did; they are no longer waited for).
    """
    __slots__ = ("step", "position", "matched_by", "step_size", "waited_s", "expired", "skipped")

    def __init__(self, step, position, matched_by):
        self.step, self.position, self.matched_by = step, position, matched_by
        self.step_size = len(step["entries"])
        self.waited_s, self.expired, self.skipped = 0.0, False, []

    @property
    def purpose(self):
        """该模型步骤的调用用途（如 ``"main"`` 或非主链用途，未知时为 ``None``）。"""
        return self.step.get("purpose")


class EmittedOrder:
    """Hand the calls of one model step to the dispatcher in the order the model emitted them.

    The owner of the model transport calls ``announce(thread_id, calls)`` with the host tool calls of each model
    response, in emitted order, BEFORE the response is released to the agent client, so every call the client can
    send is already known. For each arriving call the bridge then does::

        ticket, reason = order.claim(tool, arguments, native_call_id=..., thread_id=...)   # never blocks
        order.wait(ticket)                 # until every earlier call of the step is resolved, bounded
        try:
            future = dispatcher.submit(params)
        finally:
            order.resolve(ticket)          # also when the call is refused instead of submitted

    A call is "resolved" once it was submitted to the dispatcher or refused; the dispatcher's own rule then orders
    execution, so parallel-safe calls still overlap and an ordered call runs after every call emitted before it.

    Correlation (``claim``) is exact or absent; nothing is guessed:

    1. The client sent the model's call id (``native_call_id``; Claude Code sends the ``tool_use`` id as
       ``_meta["claudecode/toolUseId"]``): the announced call with that id that has not arrived yet, in any kept
       step. Nothing else is tried for such a call.
    2. The client sent no call id (OpenCode) but its session key (``thread_id``, OpenCode
       ``_meta["ai.opencode/sessionID"]``): in the newest step announced under that key, the first call in emitted
       order that has not arrived and has the same tool name and the same JSON arguments. Calls with equal name and
       arguments are interchangeable.
    3. Otherwise the call has no place: ``claim`` returns ``(None, reason)`` with a reason from
       ``UNMATCHED_REASONS`` and the call is not sequenced (arrival order, as without this class). The caller
       records the reason, so a correlation that stopped working is visible.

    Bound: ``wait`` holds a call at most ``wait_s`` seconds for earlier calls that have not ARRIVED. When that runs
    out the missing calls are written off (``skipped``; every later call of the step stops waiting for them too),
    with ``ticket.expired = True`` and ``ticket.skipped`` naming them for the caller to record. The earlier calls
    that did arrive are released by the same write-off and go first, so the emitted order holds among the calls
    that arrived; each of them only has to finish its own hand-over (the caller's bookkeeping between ``wait`` and
    ``resolve``), which takes no timed wait. That hand-over is given a second window of at most ``wait_s``, purely
    so that a caller thread stuck there cannot hold the step forever; only then does the call go ahead of an
    arrived one (``expired`` with empty ``skipped``). So a call is held ``wait_s`` when an earlier sibling is never
    sent, and 2 x ``wait_s`` at the very most. The second window cannot be removed without either an unbounded
    wait or letting a call overtake an arrived earlier one. Nothing is ever dropped or refused here.

    wait_s: the bound in seconds (> 0; default ``EMITTED_ORDER_WAIT_S`` = 5.0).
    max_steps: announced steps kept (default ``EMITTED_ORDER_STEPS`` = 64); a call of an older step is not sequenced.
    Thread-safe. ``held`` is the number of calls currently inside ``wait``.
    """

    def __init__(self, *, wait_s=EMITTED_ORDER_WAIT_S, max_steps=EMITTED_ORDER_STEPS):
        if isinstance(wait_s, bool) or not isinstance(wait_s, (int, float)) or not wait_s > 0:
            raise ValueError("invalid_emitted_order_wait_s")
        if type(max_steps) is not int or max_steps <= 0:
            raise ValueError("invalid_emitted_order_max_steps")
        self.wait_s, self.max_steps = float(wait_s), max_steps
        self._condition = threading.Condition()
        self._steps = []
        self.held = 0

    def announce(self, thread_id, calls, *, purpose=None):
        """Record one model step. ``thread_id``: the client's session key of the model request (str or None).
        ``calls``: iterable of ``(native_call_id or None, tool name, arguments)`` in emitted order, host tools only
        and named as the MCP endpoint names them (no client prefix). ``purpose``: optional relay call purpose
        (e.g. ``"main"``). An empty step is ignored (a response without host tool calls says nothing about the
        calls of an earlier step)."""
        entries = [dict(id=call_id if isinstance(call_id, str) and call_id else None, tool=tool, arguments=arguments,
                        state=_PENDING) for call_id, tool, arguments in calls]
        if not entries:
            return
        with self._condition:
            self._steps.append(dict(thread=thread_id, purpose=purpose, entries=entries))
            del self._steps[:-self.max_steps]

    def claim(self, tool, arguments, *, native_call_id=None, thread_id=None):
        """Correlate an arrived call with an announced one (class docstring, "Correlation") and mark it arrived.
        Returns ``(OrderTicket, None)``, or ``(None, reason)`` with a reason from ``UNMATCHED_REASONS`` when the call
        has no exact place. Never blocks."""
        with self._condition:
            step = position = None
            if not self._steps:
                reason = UNMATCHED_NO_STEP
            elif native_call_id is not None:
                matched_by, reason = "native_call_id", UNMATCHED_CALL_ID
                for candidate in reversed(self._steps):
                    for index, entry in enumerate(candidate["entries"]):
                        if entry["id"] == native_call_id and entry["state"] in (_PENDING, _SKIPPED):
                            step, position = candidate, index
                            break
                    if step is not None:
                        break
            elif thread_id is None:
                reason = UNMATCHED_NO_SESSION_KEY
            else:
                matched_by = "arguments"
                step = next((candidate for candidate in reversed(self._steps) if candidate["thread"] == thread_id), None)
                if step is None:
                    reason = UNMATCHED_SESSION_KEY
                else:
                    open_calls = [(index, entry) for index, entry in enumerate(step["entries"])
                                  if entry["tool"] == tool and entry["state"] in (_PENDING, _SKIPPED)]
                    position = next((index for index, entry in open_calls
                                     if _same_arguments(entry["arguments"], arguments)), None)
                    reason = UNMATCHED_ARGUMENTS if open_calls else UNMATCHED_NOT_IN_STEP
            if position is None:
                return None, reason
            step["entries"][position]["state"] = _ARRIVED
            self._condition.notify_all()
            return OrderTicket(step, position, matched_by), None

    def wait(self, ticket):
        """Block until every call emitted before ``ticket``'s call in its step is resolved, within the bound (class
        docstring). ``ticket`` None returns at once. Returns ``ticket`` with ``waited_s`` / ``expired`` / ``skipped``."""
        if ticket is None:
            return None
        earlier = ticket.step["entries"][:ticket.position]
        started = time.monotonic()
        with self._condition:
            self.held += 1
            try:
                deadline, handing_over = started + self.wait_s, False
                while True:
                    if not any(entry["state"] in (_PENDING, _ARRIVED) for entry in earlier):
                        break
                    remaining = deadline - time.monotonic()
                    if remaining > 0:
                        self._condition.wait(remaining)
                        continue
                    if handing_over:
                        ticket.expired = True      # an arrived earlier call was not handed over within the bound
                        break
                    # Write off the earlier calls that never arrived (possibly none left: another held call of the
                    # step did it a moment ago), then let the arrived ones go first.
                    missing = [position for position, entry in enumerate(earlier) if entry["state"] == _PENDING]
                    if missing:
                        ticket.expired, ticket.skipped = True, missing
                        for position in missing:
                            earlier[position]["state"] = _SKIPPED
                        self._condition.notify_all()
                    deadline, handing_over = time.monotonic() + self.wait_s, True
            finally:
                self.held -= 1
                ticket.waited_s = time.monotonic() - started
        return ticket

    def resolve(self, ticket):
        """Mark ``ticket``'s call resolved (submitted to the dispatcher, or refused); wakes the calls held for it.
        ``ticket`` None is a no-op. Idempotent."""
        if ticket is None:
            return
        with self._condition:
            ticket.step["entries"][ticket.position]["state"] = _RESOLVED
            self._condition.notify_all()
