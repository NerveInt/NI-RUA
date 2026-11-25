"""Pure turn verdict for OpenCode 2.x (design 11 sections 1.6-1.8; pinned 2.2.7).

OpenCode 2.x writes an explicit end marker into the session's message list: one ``idle`` message
``{type: "idle", outcome: succeeded|failed|interrupted, error?}`` per finished execution. A NI-RUA turn has a
**boundary**: the id of the latest message the host submitted in it (``POST /api/session/{id}/prompt`` returns it), or,
for an adopted turn, ``adoption_boundary`` of the list seen before adoption. One snapshot = the main session's
``GET /api/session/{id}/message`` list (every page, ascending) + ``GET /api/session/active`` + the main session's
``GET /api/session/{id}/inbox``. ``turn_verdict`` is a pure function of one snapshot and host facts; it never performs
I/O and never mutates its input. Two small guards hold the caller's clock-dependent state: ``SettleGuard`` (rule 5)
and ``IncompleteTurnGuard`` (a turn that never ends).

The list is a **timeline**: its order is the API's (positional cursor pagination), never id order (a background
subagent's report has a smaller id than the parent's ``idle`` it follows). Pages are read one after another, so a list
can be non-atomic: later pages are fresher and may hold messages appended after an earlier page was read, and a
message may appear twice across pages. ``timeline`` keeps the first position and the freshest copy of each id.

Rules, in order (the first that applies decides):

1. Any session of the run's tree is in ``active`` -> ``running`` (``active``). The server is private to the run, so by
   default every active session counts (``sessions=None``); a background subagent keeps the turn open because
   ``McpToolBridge`` refuses registry calls outside a running turn and the child may call ``ni_rua_*``.
2. The boundary is not in the list -> ``running``: ``queued`` when it waits in the inbox (a steer before the next step
   boundary), else ``user_message_pending``.
3. No ``idle`` message after the boundary:
   a. the inbox is not empty -> ``running`` (``inbox_pending``);
   b. the last ``user``/``synthetic``/``assistant``/``compaction`` message after the boundary is a FINISHED assistant
      step whose ``error.type`` is ``aborted`` -> terminal: the execution ended without an ``idle``. OpenCode 2.2.7 does
      this when a permission request is rejected (the tool ends ``aborted`` "The user declined this tool call", the
      step ``aborted``, ``session.execution.interrupted`` with ``reason: shutdown``, no ``idle``; probes b07 and
      OC2-C ``permission``). ``interrupt_requested`` -> ``interrupted``; ``permission_rejected`` (host fact: the
      session auto-rejected a permission request of this turn, design 11 section 1.3) -> ``failed``
      ``opencode_permission_rejected`` at once; otherwise ``failed`` ``opencode_execution_interrupted`` with
      ``settle_key`` always set, so ``SettleGuard`` confirms it (a host interrupt's ``idle`` may still be on its way);
   c. otherwise -> ``running`` (``awaiting_idle``). A server that died mid-turn leaves the tool ``running``, the step
      unfinished and writes no ``idle``; ``IncompleteTurnGuard`` turns that into ``failed``.
4. Inbox not empty -> ``running`` (``inbox_pending``: a subagent report or a steer is about to start an execution).
   A ``user``/``synthetic``/``assistant``/``compaction`` message listed after the last ``idle`` -> ``running``
   (``execution_pending``: a delivered report or steer whose execution was not seen in ``active`` because the reads
   are not atomic). ``model-switched`` and the other row-less types do not count. An assistant step between the
   boundary and the deciding ``idle`` that is not finished, or a tool of it that is not ``completed``/``error``, ->
   ``running`` (``stale_snapshot``: an earlier page was read before the step ended; an execution never ends with an
   open step).
5. With subagents (``children=True``) a terminal candidate carries ``settle_key``; ``SettleGuard`` confirms it only
   after the same quiescent snapshot was seen twice at least ``SETTLE_MIN_GAP_S`` apart (a child's report is
   enqueued to the parent just after the child leaves ``active``).
6. ``interrupt_requested`` -> ``interrupted``. Otherwise the last ``idle`` after the boundary decides, except that
   empty successful executions preserve the preceding failure or interrupt. A new user, synthetic, assistant or
   compaction message starts fresh work. This also applies when the host adopted the empty execution after the
   preceding idle was already recorded. The deciding outcome maps as follows: ``interrupted`` -> ``interrupted``; ``failed`` ->
   ``failed`` with ``opencode_turn_failed:<error.type>[:<error.status>]`` (``idle.error``, else the error of the last
   assistant message; never provider text); ``succeeded`` -> ``completed`` with ``final_text`` = the non-empty text
   items of the last assistant message between the boundary and that idle, joined by ``"\\n"`` (None when there is
   none), the same join as ``AgentSession.completed_final_answer``.

Every running verdict except ``active`` has ``idle=True`` (no tree session active); only those feed
``IncompleteTurnGuard``. Nothing here is model-visible: verdicts are observer facts for the session.
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Collection, Mapping, Sequence

__all__ = ["TurnVerdict", "turn_verdict", "timeline", "status_type", "active_sessions", "inbox_ids", "last_idle_id",
           "adoption_boundary", "final_text_of", "error_code", "SettleGuard", "IncompleteTurnGuard", "TURN_FAILED",
           "STEP_FAILED", "COMPACTION_FAILED", "PERMISSION_REJECTED", "EXECUTION_INTERRUPTED", "ABORTED_ERROR_TYPE",
           "INCOMPLETE_ERROR", "INCOMPLETE_MIN_GAP_S", "SETTLE_MIN_GAP_S"]

TURN_FAILED = "opencode_turn_failed"           # prefix of the failed-turn error code
STEP_FAILED = "opencode_step_failed"           # prefix of an assistant message (one model step) error row
COMPACTION_FAILED = "opencode_compaction_failed"   # prefix of a failed compaction's error code
PERMISSION_REJECTED = "opencode_permission_rejected"       # turn ended by an auto-rejected permission request
EXECUTION_INTERRUPTED = "opencode_execution_interrupted"   # execution ended aborted without an idle, cause unknown
ABORTED_ERROR_TYPE = "aborted"                 # ``error.type`` of an interrupted step or tool (interrupt, rejection)
INCOMPLETE_ERROR = "opencode_turn_incomplete"
INCOMPLETE_MIN_GAP_S = 2.0     # IncompleteTurnGuard: two idle observations at least this far apart (s)
SETTLE_MIN_GAP_S = 0.5         # SettleGuard (rule 5): the same quiescent snapshot seen twice this far apart (s)
_ERROR_TYPE = re.compile(r"^[A-Za-z0-9_.:-]{1,64}$")
# message types that belong to an execution: listed after the last idle, they mean another execution has begun
_EXECUTION_TYPES = frozenset({"user", "synthetic", "assistant", "compaction"})
_SETTLED_TOOL = frozenset({"completed", "error"})


@dataclass(frozen=True)
class TurnVerdict:
    """Outcome of one reconciliation.

    status: ``completed`` | ``interrupted`` | ``failed`` | ``running`` (not terminal yet).
    final_text: final answer text (``completed`` only; None when the answer has no text).
    error: stable error string for ``failed`` (no provider text), else None.
    message_id: id of the last assistant message between the boundary and the deciding idle, the step of a previous
        execution whose error was preserved across an empty wake-up, or the aborted step that ended the execution
        without an idle (None when there is none).
    idle_id: id of the deciding ``idle`` message (terminal from an idle only, else None).
    reason: running -> ``active`` | ``user_message_pending`` | ``queued`` | ``awaiting_idle`` | ``inbox_pending`` |
        ``execution_pending`` | ``stale_snapshot`` | ``settling``; terminal -> ``succeeded`` | ``failed`` |
        ``interrupted`` (the idle outcome) | ``interrupt_requested`` | ``permission_rejected`` |
        ``execution_interrupted`` | ``incomplete``.
    idle: True when no session of the tree was active (only idle non-terminal verdicts feed ``IncompleteTurnGuard``).
    settle_key: set on a terminal candidate that ``SettleGuard`` must confirm, else None.
    """
    status: str
    final_text: str | None = None
    error: str | None = None
    message_id: str | None = None
    idle_id: str | None = None
    reason: str | None = None
    idle: bool = False
    settle_key: tuple | None = None

    @property
    def terminal(self) -> bool:
        return self.status != "running"


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def timeline(messages) -> list:
    """One session's messages in API (timeline) order, never re-sorted: a list, or a ``{data: [...], cursor}`` page.
    Entries without a string ``id`` are dropped; an id that appears twice (overlapping pages of a non-atomic read)
    keeps its first position and its last (freshest) content."""
    if isinstance(messages, Mapping):
        messages = messages.get("data")
    order, latest = [], {}
    for message in messages or ():
        if isinstance(message, Mapping) and isinstance(message.get("id"), str):
            if message["id"] not in latest:
                order.append(message["id"])
            latest[message["id"]] = message
    return [latest[identifier] for identifier in order]


def status_type(entry, session_id: str | None = None) -> str:
    """Activity of one session from ``GET /api/session/active``: ``"running"`` (or another native type) / ``"idle"``.

    Accepts the session's entry (``{"type": "running"}``), None (absent from the map = idle), a type string, or the
    whole answer / its ``data`` map (looked up by ``session_id``)."""
    if entry is None:
        return "idle"
    if isinstance(entry, str):
        return entry or "idle"
    if isinstance(entry, Mapping):
        if "type" in entry:
            return str(entry.get("type") or "idle")
        if isinstance(entry.get("data"), Mapping):
            entry = entry["data"]
        return status_type(entry.get(session_id) if session_id else None)
    raise TypeError("opencode_status_must_be_mapping_or_str")


def active_sessions(active) -> frozenset:
    """Ids of the active sessions in a ``GET /api/session/active`` answer (``{"data": {<id>: {"type": ...}}}``), its
    ``data`` map, None (nothing active), or an iterable of session ids."""
    if active is None:
        return frozenset()
    if isinstance(active, Mapping):
        entries = active.get("data") if isinstance(active.get("data"), Mapping) else active
        return frozenset(session for session, entry in entries.items() if status_type(entry) != "idle")
    if isinstance(active, str):
        raise TypeError("opencode_active_must_be_mapping_or_ids")
    return frozenset(session for session in active if isinstance(session, str))


def inbox_ids(inbox) -> list:
    """Message ids waiting in a ``GET /api/session/{id}/inbox`` answer (``{"data": [...]}``), its list, or None."""
    if isinstance(inbox, Mapping):
        inbox = inbox.get("data")
    return [item["id"] for item in inbox or () if isinstance(item, Mapping) and isinstance(item.get("id"), str)]


def _finished(message) -> bool:
    """An assistant step that OpenCode closed (``time.completed``, ``finish`` or ``error`` set)."""
    return bool(_mapping(message.get("time")).get("completed") or message.get("finish") or message.get("error"))


def _settled(message) -> bool:
    """A finished assistant step whose tools are all ``completed``/``error``."""
    return _finished(message) and all(
        _mapping(item.get("state")).get("status") in _SETTLED_TOOL for item in message.get("content") or ()
        if isinstance(item, Mapping) and item.get("type") == "tool")


def _aborted_step(message) -> bool:
    return (message.get("type") == "assistant" and _settled(message)
            and _mapping(message.get("error")).get("type") == ABORTED_ERROR_TYPE)


def last_idle_id(messages) -> str | None:
    """Id of the last ``idle`` message of a message list, or None."""
    idles = [message["id"] for message in timeline(messages) if message.get("type") == "idle"]
    return idles[-1] if idles else None


def adoption_boundary(messages) -> str | None:
    """Boundary of a turn adopted from the list seen before adoption (design R4.2): the id of the last message that
    ended an execution - the last ``idle``, or a later finished ``aborted`` step that no ``idle`` follows (an execution
    ended by a rejected permission). None when the list has neither (adopt from the start of the list)."""
    boundary = None
    for message in timeline(messages):
        if message.get("type") == "idle" or _aborted_step(message):
            boundary = message["id"]
    return boundary


def final_text_of(message) -> str | None:
    """Non-empty text items of one assistant message joined by ``"\\n"``; None when there is none."""
    content = _mapping(message).get("content")
    texts = [item["text"] for item in content if isinstance(item, Mapping) and item.get("type") == "text"
             and isinstance(item.get("text"), str) and item["text"].strip()] if isinstance(content, list) else []
    return "\n".join(texts) if texts else None


def error_code(error, prefix: str = TURN_FAILED) -> str:
    """``"<prefix>:<error.type>[:<error.status>]"`` of an OpenCode structured error (``{type, message, status?,
    response?}``). Only OpenCode's error classification and the HTTP status are kept; ``message`` and
    ``response.body`` are provider text and never copied. A type outside ``[A-Za-z0-9_.:-]{1,64}`` becomes
    ``unknown``."""
    error = _mapping(error)
    kind = error.get("type")
    kind = kind if isinstance(kind, str) and _ERROR_TYPE.match(kind) else "unknown"
    status = error.get("status")
    suffix = f":{status}" if isinstance(status, int) and not isinstance(status, bool) else ""
    return f"{prefix}:{kind}{suffix}"


def _deciding_idle(messages, last, boundary):
    """2.2.7 can append an empty successful execution immediately after a failed one."""
    if messages[last].get("outcome") == "succeeded":
        # A host submission starts a new turn. A native wake-up can start at an already accounted idle.
        first = 0 if messages[boundary].get("type") == "idle" else boundary + 1
        for index in range(last - 1, first - 1, -1):
            message = messages[index]
            if message.get("type") in _EXECUTION_TYPES:
                break
            if message.get("type") == "idle" and message.get("outcome") != "succeeded":
                return index
    return last


def turn_verdict(messages: Sequence, *, boundary: str | None, active=None, inbox=None,
                 sessions: Collection[str] | None = None, children: bool = False,
                 interrupt_requested: bool = False, permission_rejected: bool = False) -> TurnVerdict:
    """Decide the current turn from one snapshot (module rules 1-6).

    messages: the main session's ``GET /api/session/{id}/message`` timeline (API order, every page) or one page
        mapping; see ``timeline``.
    boundary: id of the latest message the host submitted in this turn (the ``POST .../prompt`` answer's ``data.id``;
        ``OpenCodeProjector.submissions`` reports it), or ``adoption_boundary`` for an adopted turn; None while the
        submission has no id yet.
    active: ``GET /api/session/active`` answer, its ``data`` map, or session ids (``active_sessions``); None = nothing.
    inbox: the main session's ``GET /api/session/{id}/inbox`` answer or list (``inbox_ids``); None = empty.
    sessions: ids whose activity holds the turn (main + known children); None = every active session of the private
        server (recommended: it also covers a child whose ``session.created`` frame was missed).
    children: True when the tree has (or had during this turn) subagent sessions -> rule 5 via ``SettleGuard``.
    interrupt_requested: True once the host requested an interrupt of this turn.
    permission_rejected: True once the host auto-rejected a permission request of this turn's tree (rule 3b).
    """
    messages = timeline(messages)
    running = active_sessions(active)
    if sessions is not None:
        running &= frozenset(sessions)
    if running:
        return TurnVerdict("running", reason="active")
    waiting = inbox_ids(inbox)
    ids = [message["id"] for message in messages]
    if boundary is None or boundary not in ids:
        return TurnVerdict("running", reason="queued" if boundary is not None and boundary in waiting
                           else "user_message_pending", idle=True)
    boundary_index = ids.index(boundary)
    after = messages[boundary_index + 1:]
    key = (len(messages), ids[-1])
    idles = [index for index, message in enumerate(after) if message.get("type") == "idle"]
    if not idles:
        if waiting:
            return TurnVerdict("running", reason="inbox_pending", idle=True)
        tail = [message for message in after if message.get("type") in _EXECUTION_TYPES]
        if not tail or not _aborted_step(tail[-1]):
            return TurnVerdict("running", reason="awaiting_idle", idle=True)
        common = dict(message_id=tail[-1]["id"], idle=True)
        if interrupt_requested:
            return TurnVerdict("interrupted", reason="interrupt_requested", settle_key=key if children else None,
                               **common)
        if permission_rejected:
            return TurnVerdict("failed", error=PERMISSION_REJECTED, reason="permission_rejected",
                               settle_key=key if children else None, **common)
        return TurnVerdict("failed", error=EXECUTION_INTERRUPTED, reason="execution_interrupted", settle_key=key,
                           **common)
    if waiting:
        return TurnVerdict("running", reason="inbox_pending", idle=True)
    if any(message.get("type") in _EXECUTION_TYPES for message in after[idles[-1] + 1:]):
        return TurnVerdict("running", reason="execution_pending", idle=True)
    answers = [message for message in after[:idles[-1]] if message.get("type") == "assistant"]
    if not all(_settled(message) for message in answers):
        return TurnVerdict("running", reason="stale_snapshot", idle=True)
    deciding = _deciding_idle(messages, boundary_index + 1 + idles[-1], boundary_index)
    idle, answer = messages[deciding], answers[-1] if answers else None
    if deciding <= boundary_index:  # an adopted, empty wake-up: its effective terminal belongs to the prior execution
        for index in range(deciding - 1, -1, -1):
            message = messages[index]
            if message.get("type") in {"user", "idle"}:
                break
            if message.get("type") == "assistant":
                answer = message
                break
    common = dict(message_id=answer["id"] if answer else None, idle_id=idle["id"], idle=True,
                  settle_key=key if children else None)
    outcome = idle.get("outcome")
    if interrupt_requested:
        return TurnVerdict("interrupted", reason="interrupt_requested", **common)
    if outcome == "interrupted":
        return TurnVerdict("interrupted", reason="interrupted", **common)
    if outcome == "succeeded":
        return TurnVerdict("completed", final_text=final_text_of(answer), reason="succeeded", **common)
    error = idle.get("error") if isinstance(idle.get("error"), Mapping) else _mapping(answer).get("error")
    return TurnVerdict("failed", error=error_code(error), reason="failed", **common)


class SettleGuard:
    """Rule 5 (and rule 3b without a host fact): a terminal candidate with ``settle_key`` is confirmed only once the
    same snapshot key was seen again at least ``min_gap_s`` after it was first seen; until then ``observe`` returns
    ``running``/``settling`` (``idle=True``). A verdict without ``settle_key`` passes unchanged; any non-terminal
    verdict resets the guard. ``now`` is a monotonic time in seconds supplied by the caller (no clock is read).

    min_gap_s: minimum time (s) between the two observations of the same quiescent snapshot; default 0.5.
    """

    def __init__(self, *, min_gap_s: float = SETTLE_MIN_GAP_S):
        if not min_gap_s >= 0:
            raise ValueError("min_gap_s_must_be_non_negative")
        self.min_gap_s = float(min_gap_s)
        self.key: tuple | None = None
        self.first_seen_at: float | None = None

    def reset(self) -> None:
        self.key = self.first_seen_at = None

    def observe(self, verdict: TurnVerdict, now: float) -> TurnVerdict:
        if not verdict.terminal:
            self.reset()
            return verdict
        if verdict.settle_key is None:
            return verdict
        if verdict.settle_key != self.key:
            self.key, self.first_seen_at = verdict.settle_key, now
        elif now - self.first_seen_at >= self.min_gap_s:
            return verdict
        return TurnVerdict("running", message_id=verdict.message_id, reason="settling", idle=True)


class IncompleteTurnGuard:
    """A turn whose verdict stays ``running`` while no session of the tree is active is failed after two idle
    observations at least ``min_gap_s`` apart (a server killed mid-turn: no ``idle`` is ever written, the tool stays
    ``running``; a submission that never materialised).

    min_gap_s: minimum time (s) between the first and the confirming idle non-terminal observation; default 2.0.
    Observations while active, and any terminal verdict, reset the guard. Use one guard per turn (or ``reset()``).
    ``observe(verdict, now, interrupt_requested=False)`` returns the verdict unchanged, or the guard verdict:
    ``failed`` / ``opencode_turn_incomplete`` (``interrupted`` when the host requested an interrupt).
    ``now`` is a monotonic time in seconds supplied by the caller (the guard never reads a clock).
    """

    def __init__(self, *, min_gap_s: float = INCOMPLETE_MIN_GAP_S):
        if not min_gap_s >= 0:
            raise ValueError("min_gap_s_must_be_non_negative")
        self.min_gap_s = float(min_gap_s)
        self.first_idle_at: float | None = None

    def reset(self) -> None:
        self.first_idle_at = None

    def observe(self, verdict: TurnVerdict, now: float, *, interrupt_requested: bool = False) -> TurnVerdict:
        if verdict.terminal or not verdict.idle:
            self.first_idle_at = None
            return verdict
        if self.first_idle_at is None:
            self.first_idle_at = now
            return verdict
        if now - self.first_idle_at < self.min_gap_s:
            return verdict
        if interrupt_requested:
            return TurnVerdict("interrupted", message_id=verdict.message_id, reason="interrupt_requested", idle=True)
        return TurnVerdict("failed", error=INCOMPLETE_ERROR, message_id=verdict.message_id, reason="incomplete",
                           idle=True)
