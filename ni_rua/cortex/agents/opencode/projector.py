"""Projects OpenCode 2.x messages and events onto the neutral transcript (design 11 section 1.6; pinned 2.2.7).

OpenCode 2.x keeps one flat, ordered list of typed messages per session (``GET /api/session/{id}/message``):
``user``, ``synthetic``, ``assistant`` (``content[]`` of ``text``/``reasoning``/``tool`` items), ``compaction``,
``idle``, ``model-switched`` and a few that carry no transcript meaning. Two inputs, one result:

* ``sync(messages)`` takes one session's full list and upserts every row it implies. It is the authoritative path:
  ``/api/event`` has no replay, so after any gap (and before every turn verdict) the session re-projects the whole
  list. Re-projection is idempotent: row ids are derived from native ids, unchanged rows are not rewritten
  (``Store.set_state`` skips identical bodies) and nothing is appended. The list is taken in API (timeline) order and
  never sorted by id (``verdict.timeline``: a background subagent's report has a smaller id than the ``idle`` it
  follows); new rows are written in that order. A non-atomic paged read (a later page fresher than an earlier one, an
  id on two pages) is fine: rows are upserts and never deleted, so the next ``sync`` completes them.
* ``project_event(frame)`` applies one ``/api/event`` frame (``{id, created, type, data: {sessionID, ...}}``) for live
  display. It is advisory; the next ``sync`` overwrites whatever it wrote.

The projector never decides turn state (``verdict.turn_verdict`` does) and never writes usage (the relay is the
accounting source). Rows are the ``codex_item`` shapes ``TranscriptWriter`` writes (the replay, the web readers,
``Store.completed_final_texts`` and ``native_evidence`` cannot tell them from Codex rows); every row carries
``backend``, ``native_id`` (the OpenCode/provider id) and, for assistant content, ``message_id`` (the assistant
message, so the final-answer rule sees the tool calls of the same message). Mapping:

* ``user``                  -> the host writes the ``userMessage`` row (``run_turn``/``steer``) BEFORE the prompt
  request and calls ``bind_submission(row, <id the request returned>)``: the native message then has a row and no
  text is matched. A ``user`` message with no bound row while a host submission of the current turn is still unbound
  is deferred (``deferred_users``); otherwise (a resumed history, a subagent's first prompt) it becomes its own
  ``userMessage`` row (id = message id, ``source: "native"``, ``synthetic: false``);
* ``synthetic``             -> ``userMessage`` (id = message id, ``source: "native"``, ``synthetic: true``): the
  "previous response was interrupted" / "hit the output token limit" continuations and subagent reports;
* assistant ``text``        -> ``agentMessage`` id ``<message id>:text:<k>`` (``k`` = the item's ordinal among the
  message's text items: the ``ordinal`` of ``session.text.*`` frames, counted per kind and step by the binary); ``inProgress`` until the message finished
  (``time.completed``, ``finish`` or ``error``); empty text writes no row; the writer stamps ``phase`` at turn end.
  A finished row (text, tool, compaction) is never set back to ``inProgress`` by a stale read;
* assistant ``reasoning``   -> ``reasoning`` id ``<message id>:reasoning:<k>`` (text only; ``native`` signatures and
  encrypted content are never stored);
* assistant ``tool``        -> row id ``<message id>:<tool id>``, ``native_id`` = the tool id (provider call id):
  ``shell`` -> ``commandExecution`` (command, joined text output, ``metadata.exit`` as exit code); ``edit``/``write``/
  ``patch`` -> ``fileChange`` (``changes = [{path, kind add|update|delete|write, diff}]`` from ``metadata.files`` once
  finished, else from the input); every other native tool (``read``, ``glob``, ``grep``, ``subagent``, ``execute``,
  ...) -> generic ``mcpToolCall`` (tool, arguments, result = joined text output, error ``{message, type}`` of
  OpenCode's own error, e.g. ``aborted`` "The user declined this tool call" after a rejected permission). Tool images
  (``file`` content) are never copied. ``streaming`` (input not complete) writes no row;
* registry tools ``ni_rua_*`` -> NO row of their own: the MCP transport wrote the ``dynamicToolCall`` row under the
  host call id (OpenCode sends no call id over MCP). The projector attaches ``native_id`` (the tool id) and
  ``message_id`` to the first unmatched transport row of the current turn with the same tool name and canonical
  arguments (best effort, never required for correctness). A registry call that ended in ``error`` while no transport
  row of that tool is unmatched (refused before MCP) becomes one failed generic row under its row id;
* assistant ``error``       -> ``error`` row ``<message id>:error`` with ``opencode_step_failed:<type>[:<status>]``
  unless the step was ``aborted`` (an interrupt or a rejected permission); a step error is not a turn failure in 2.x (a mid-stream error is
  continued), the turn outcome is the ``idle`` message's. ``retry`` -> ``error`` row ``<message id>:retry:<n>`` with
  ``opencode_retry:<n>:<type>[:<status>]``. Provider text (``message``, ``response.body``) is never copied;
* ``compaction``            -> ``contextCompaction`` (id = message id; ``inProgress``/``completed``/``failed``;
  ``reason`` auto|manual; ``summary`` as a field of that row, never final text; ``error`` code when failed);
* ``model-switched``        -> no row; reported once in ``model_switches`` (the session emits ``agent.model_switched``);
* ``idle``, ``agent-switched``, ``location-switched``, ``system``, ``skill``, ``shell`` -> no row.

Turn attribution: a row gets the writer's current turn when it is first written; existing rows keep theirs. Call
``sync`` once in ``start()`` before the first turn (a resumed session adopts its history without moving it into the
new turn), and begin an adopted turn before projecting the execution it adopts. Nothing written here is
model-visible.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Mapping, Sequence

from ....core.contracts import canonical
from .verdict import ABORTED_ERROR_TYPE, COMPACTION_FAILED, STEP_FAILED, error_code, timeline

__all__ = ["Projected", "OpenCodeProjector", "TOOL_NAME_PREFIX", "INTERRUPTED_TOOL_ERROR", "RETRY_ERROR",
           "compaction_message_id", "subagent_links"]

TOOL_NAME_PREFIX = "ni_rua_"       # model-visible registry tool prefix (MCP server key "ni_rua" + "_")
INTERRUPTED_TOOL_ERROR = "opencode_tool_interrupted"
RETRY_ERROR = "opencode_retry"
_SHELL, _SUBAGENT = "shell", "subagent"
_FILE_TOOLS = ("edit", "write", "patch")
_STREAMED = ("text", "reasoning")
_TOOL_STATUS = {"running": "inProgress", "completed": "completed", "error": "failed"}
_COMPACTION_STATUS = {"running": "inProgress", "completed": "completed", "failed": "failed"}
_DIFF_KIND = {"added": "add", "modified": "update", "deleted": "delete"}
_PATCH_FILE = re.compile(r"^\*\*\* (Add|Update|Delete) File: (.+)$", re.MULTILINE)
_PATCH_KIND = {"Add": "add", "Update": "update", "Delete": "delete"}


@dataclass
class Projected:
    """What ``sync``/``project_event`` found: ``kind`` plus facts for the session.

    kinds (``data`` keys): ``snapshot`` (messages, boundary, pending_submissions, compactions, unmatched_registry,
    subagents, model_switches, deferred_users); ``step`` (message_id, phase started|ended|failed); ``text`` (item_id,
    kind text|reasoning, applied); ``tool`` (item_id, message_id, status, applied); ``subagent`` (session_id, item_id,
    message_id); ``retry`` (item_id); ``compaction`` (message_id, phase started|ended|failed, reason, new);
    ``execution`` (phase started|succeeded|failed|interrupted); ``deferred`` (item_id: a tool frame whose call this
    projector has not seen; the next ``sync`` writes it); ``ignored`` (another session); ``other`` (type).
    """
    kind: str
    data: dict = field(default_factory=dict)


def _text(value) -> str:
    return value if isinstance(value, str) else ""


def _mapping(value) -> Mapping:
    return value if isinstance(value, Mapping) else {}


def _canonical(arguments):
    try:
        return canonical(dict(arguments) if isinstance(arguments, Mapping) else {})
    except (TypeError, ValueError):
        return None


def _content_text(content) -> str | None:
    """Text items of a tool result joined by ``"\\n"`` (``file`` items, e.g. images, are never copied); None if none."""
    texts = [item["text"] for item in content if isinstance(item, Mapping) and item.get("type") == "text"
             and isinstance(item.get("text"), str)] if isinstance(content, list) else []
    return "\n".join(texts) if texts else None


def _finished(message) -> bool:
    return bool(_mapping(message.get("time")).get("completed") or message.get("finish") or message.get("error"))


def compaction_message_id(event_id) -> str | None:
    """Id of the ``compaction`` message a ``session.compaction.started`` frame creates: OpenCode 2.2.7 mints both
    from one id (``evt_<x>`` -> ``msg_<x>``; verified on every captured compaction, s05/s06). None for another form."""
    return "msg_" + event_id[4:] if isinstance(event_id, str) and event_id.startswith("evt_") and event_id[4:] else None


def subagent_links(messages) -> list:
    """``[{session_id, item_id, message_id, background}]`` of every ``subagent`` tool item of one message list whose
    ``state.metadata.sessionID`` names the child (set once the call settled; while a foreground child runs the link
    comes from ``session.tool.progress`` / ``session.created``). ``item_id`` is the tool's row id (the child's rows
    hang under it)."""
    links = []
    for message in timeline(messages):
        if message.get("type") != "assistant":
            continue
        for item in message.get("content") or ():
            state = _mapping(_mapping(item).get("state"))
            child = _mapping(state.get("metadata")).get("sessionID")
            if _mapping(item).get("type") == "tool" and item.get("name") == _SUBAGENT and isinstance(child, str):
                links.append(dict(session_id=child, item_id=f"{message['id']}:{item.get('id')}",
                                  message_id=message["id"], background=_mapping(state.get("input")).get(
                                      "background") is True))
    return links


class _Rows:
    """This thread's ``userMessage``/``dynamicToolCall`` rows (the only ones the projector looks up) in store order,
    read once on first use (one query; ``contentItems`` - tool images - are not decoded) and kept current by
    the projector's own annotations."""

    KINDS = ("userMessage", "dynamicToolCall")

    def __init__(self, store, thread_id):
        self.store, self.thread_id, self._rows = store, thread_id, None

    @property
    def rows(self):
        if self._rows is None:
            with self.store.lock:
                bodies = self.store.db.execute(
                    "SELECT json_remove(body, '$.contentItems') FROM state WHERE kind=? "
                    "AND json_extract(body,'$.thread_id')=? AND json_extract(body,'$.type') IN (?,?) ORDER BY rowid",
                    (self.store.TRANSCRIPT_KIND, self.thread_id, *self.KINDS)).fetchall()
            self._rows = [json.loads(body) for body, in bodies]
        return self._rows

    def find(self, kind, native_id, message_id=None):
        return next((row for row in self.rows if row.get("type") == kind and row.get("native_id") == native_id
                     and (message_id is None or row.get("message_id") == message_id)), None)

    def put(self, row):
        if row is None or self._rows is None or row.get("type") not in self.KINDS:
            return
        for index, existing in enumerate(self._rows):
            if existing["id"] == row["id"]:
                self._rows[index] = row
                return
        self._rows.append(row)


def _facts():
    return dict(compactions=[], unmatched_registry=[], subagents=[], model_switches=[], deferred_users=[])


class OpenCodeProjector:
    """Message/event-to-transcript projector of one OpenCode 2.x session (one ``TranscriptWriter`` whose ``thread_id``
    is the OpenCode session id; a subagent gets its own projector on the session's ``_SubagentTranscript``). Single
    consumer: the session calls ``sync``/``project_event`` from one thread at a time; the writer is thread-safe.

    transcript: the session's ``TranscriptWriter``. prefix: model-visible registry tool prefix (default ``ni_rua_``).
    """

    def __init__(self, transcript, *, prefix=TOOL_NAME_PREFIX):
        self.transcript, self.prefix = transcript, prefix
        self.store = transcript.store
        self._registry = {}        # (message id, tool id) -> transport row id (or the failed generic row id)
        self._native_rows = set()  # native tool row ids written by this projector
        self._finished = set()     # assistant message ids seen finished (event path: step.ended/failed)
        # Event path only, dropped when the step ends: (message id, kind, ordinal) -> text accumulated since its
        # *.started frame; (message id, tool id) -> latest tool item of an unfinished step (frames + sync).
        self._texts = {}
        self._tools = {}
        self._switches = set()     # model-switched message ids already reported
        self._compaction = None    # compaction message id the event path follows (started, not ended yet)

    # ----- host-side helpers -----

    def bind_submission(self, item_id, native_id):
        """Attach the id ``POST /api/session/{id}/prompt`` returned (``data.id``) to the host ``userMessage`` row
        ``item_id`` written before the request (prompt or steer). Returns False when the row does not exist."""
        if not isinstance(native_id, str) or not native_id:
            raise ValueError("opencode_native_id_invalid")
        return self._annotate(item_id, native_id=native_id)

    def withdraw_submission(self, item_id):
        """Mark a host ``userMessage`` row whose prompt request was definitely rejected (4xx): it never becomes the
        boundary and is not a pending submission. Returns False when the row does not exist."""
        return self._annotate(item_id, withdrawn=True)

    def submissions(self, turn_id=None):
        """``(boundary, pending)`` of ``turn_id`` (default: the writer's current turn): the native id of the latest
        non-withdrawn host prompt/steer row of that turn (None while it is unbound, or when the turn has no host row,
        e.g. an adopted turn) and how many of its host rows are still unbound. ``boundary`` is what
        ``verdict.turn_verdict`` needs; re-read it under the lock ``steer()`` takes before finishing a turn, so a steer
        written after the snapshot keeps the turn running."""
        hosts = self._hosts(_Rows(self.store, self.transcript.thread_id), turn_id)
        return (hosts[-1].get("native_id") if hosts else None), sum(1 for row in hosts if not row.get("native_id"))

    def _hosts(self, rows, turn_id=None):
        turn = turn_id or self.transcript.turn_id
        return [row for row in rows.rows if row.get("type") == "userMessage" and row.get("turn_id") == turn
                and row.get("source") != "native" and not row.get("withdrawn")]

    def abandon_open_items(self, status="failed"):
        """Set every native tool row this projector wrote that is still ``inProgress`` to ``status`` (``failed`` by
        default) with ``interrupted: true``; returns the row ids changed. Registry rows are never touched; a later
        native ``error`` for an abandoned row does not reopen it."""
        changed = []
        with self.transcript.lock:
            for row_id in sorted(self._native_rows):
                row = self.store.transcript_item(row_id, None)
                if row and row.get("status") == "inProgress":
                    self.store.set_transcript_item(row_id, dict(row, status=status, interrupted=True,
                                                                error=dict(message=INTERRUPTED_TOOL_ERROR)))
                    changed.append(row_id)
        return changed

    # ----- snapshot -----

    def sync(self, messages: Sequence) -> Projected:
        """Idempotently upsert all rows implied by one session's message list (ascending; a page mapping is accepted).

        Returns ``Projected("snapshot", ...)``: ``boundary``/``pending_submissions`` (``submissions()``, read fresh
        after the pass), ``compactions`` (compaction message ids written for the first time), ``unmatched_registry``
        (registry tool ids with no transport row yet), ``subagents`` (``subagent_links``), ``model_switches``
        (``{message_id, model, previous}`` not reported before), ``deferred_users`` (user message ids waiting for
        ``bind_submission``)."""
        messages = timeline(messages)
        rows = _Rows(self.store, self.transcript.thread_id)
        facts = _facts()
        unbound = any(not row.get("native_id") for row in self._hosts(rows))
        for message in messages:
            kind = message.get("type")
            if kind == "user":
                self._user(message, rows, facts, unbound)
            elif kind == "synthetic":
                self._synthetic(message)
            elif kind == "assistant":
                self._assistant(message, rows, facts)
            elif kind == "compaction":
                self._compaction_message(message, facts)
            elif kind == "model-switched" and message["id"] not in self._switches:
                self._switches.add(message["id"])
                facts["model_switches"].append(dict(message_id=message["id"], model=_model_ref(message.get("model")),
                                                    previous=_model_ref(message.get("previous"))))
        facts["subagents"] = subagent_links(messages)
        boundary, pending = self.submissions()
        return Projected("snapshot", dict(messages=len(messages), boundary=boundary, pending_submissions=pending,
                                          **facts))

    # ----- events -----

    def project_event(self, frame) -> Projected:
        """Apply one ``/api/event`` frame (advisory; ``sync`` stays authoritative). Never raises for unknown frames."""
        kind = frame.get("type") if isinstance(frame, Mapping) else None
        data = _mapping(frame.get("data")) if isinstance(frame, Mapping) else {}
        session = data.get("sessionID")
        if session and session != self.transcript.thread_id:
            return Projected("ignored", dict(type=kind))
        message_id = data.get("assistantMessageID")
        if kind in ("session.step.started", "session.step.ended", "session.step.failed") and isinstance(
                message_id, str):
            phase = kind.rsplit(".", 1)[1]
            if phase != "started":
                self._finished.add(message_id)
                if phase == "failed":
                    self._step_error(message_id, data.get("error"))
                for key in [key for key in self._texts if key[0] == message_id]:
                    self._stream_row(*key, self._texts.pop(key), finished=True)   # the step's items are final now
                for key in [key for key in self._tools if key[0] == message_id]:
                    del self._tools[key]
            return Projected("step", dict(message_id=message_id, phase=phase))
        if kind and kind.startswith(("session.text.", "session.reasoning.")) and isinstance(message_id, str):
            return self._event_text(kind, message_id, data)
        if kind and kind.startswith("session.tool.") and isinstance(message_id, str):
            return self._event_tool(kind, message_id, data)
        if kind == "session.retry.scheduled" and isinstance(message_id, str):
            return Projected("retry", dict(item_id=self._retry(message_id, data)))
        if kind and kind.startswith("session.compaction."):
            return self._event_compaction(kind, frame.get("id"), data)
        if kind and kind.startswith("session.execution."):
            return Projected("execution", dict(phase=kind.rsplit(".", 1)[1]))
        return Projected("other", dict(type=kind))

    def _event_text(self, kind, message_id, data):
        item, phase = kind.split(".")[1], kind.rsplit(".", 1)[1]
        ordinal = data.get("ordinal")
        if type(ordinal) is not int:
            return Projected("other", dict(type=kind))
        key, item_id = (message_id, item, ordinal), f"{message_id}:{item}:{ordinal}"
        if phase == "started":
            self._texts[key] = ""
            return Projected("text", dict(item_id=item_id, kind=item, applied=True))
        if phase == "delta":
            if key not in self._texts or not isinstance(data.get("delta"), str):
                return Projected("text", dict(item_id=item_id, kind=item, applied=False))
            self._texts[key] += data["delta"]
        elif phase == "ended" and isinstance(data.get("text"), str):
            self._texts[key] = data["text"]
        else:
            return Projected("other", dict(type=kind))
        self._stream_row(message_id, item, ordinal, self._texts[key], finished=message_id in self._finished)
        return Projected("text", dict(item_id=item_id, kind=item, applied=True))

    def _event_tool(self, kind, message_id, data):
        tool_id, phase = data.get("id"), kind[len("session.tool."):]
        if not isinstance(tool_id, str):
            return Projected("other", dict(type=kind))
        key, item_id = (message_id, tool_id), f"{message_id}:{tool_id}"
        known = self._tools.get(key)
        if phase == "input.started" and isinstance(data.get("name"), str):
            if known is None:
                self._tools[key] = dict(type="tool", id=tool_id, name=data["name"],
                                        state=dict(status="streaming", input=""))
            return Projected("tool", dict(item_id=item_id, message_id=message_id, status="streaming", applied=True))
        if known is None or not isinstance(known.get("name"), str):
            return Projected("deferred", dict(item_id=item_id))
        state = dict(_mapping(known.get("state")))
        if phase == "called" and isinstance(data.get("input"), Mapping):
            state.update(status="running", input=dict(data["input"]), metadata=dict(_mapping(state.get("metadata"))))
        elif phase == "progress" and isinstance(data.get("metadata"), Mapping):
            state["metadata"] = {**_mapping(state.get("metadata")), **data["metadata"]}
        elif phase in ("success", "failed"):
            state.update(status="completed" if phase == "success" else "error")
            for name in ("content", "metadata", "error"):
                if data.get(name) is not None:
                    state[name] = data[name]
        else:
            return Projected("other", dict(type=kind))
        item = dict(known, state=state)
        self._tools[key] = item
        if state.get("status") in _TOOL_STATUS:
            self._tool(item, message_id, _Rows(self.store, self.transcript.thread_id), _facts())
        child = _mapping(state.get("metadata")).get("sessionID")
        if item.get("name") == _SUBAGENT and isinstance(child, str):
            return Projected("subagent", dict(session_id=child, item_id=item_id, message_id=message_id))
        return Projected("tool", dict(item_id=item_id, message_id=message_id, status=state.get("status"),
                                      applied=state.get("status") in _TOOL_STATUS))

    def _event_compaction(self, kind, event_id, data):
        phase = kind.rsplit(".", 1)[1]
        if phase == "started":
            message_id = compaction_message_id(event_id)
            if message_id is None:
                return Projected("other", dict(type=kind))
            self._compaction = message_id
            facts = _facts()
            if self.store.transcript_item(message_id, None) is None:
                self._compaction_message(dict(id=message_id, type="compaction", status="running",
                                              reason=data.get("reason"), summary=""), facts)
            return Projected("compaction", dict(message_id=message_id, phase=phase, reason=data.get("reason"),
                                                new=bool(facts["compactions"])))
        if phase in ("ended", "failed") and self._compaction is not None:
            message_id, self._compaction = self._compaction, None
            message = dict(id=message_id, type="compaction", reason=data.get("reason"),
                           status="completed" if phase == "ended" else "failed")
            if phase == "ended":
                message["summary"] = _text(data.get("text"))
            else:
                message["error"] = data.get("error")
            self._compaction_message(message, _facts())
            return Projected("compaction", dict(message_id=message_id, phase=phase, reason=data.get("reason"),
                                                new=False))
        return Projected("other", dict(type=kind))

    # ----- rows -----

    def _annotate(self, row_id, rows=None, **fields):
        """Add store-only fields to an existing row (``None`` values are skipped); False when the row does not exist."""
        with self.transcript.lock:
            row = self.store.transcript_item(row_id, None)
            if not row:
                return False
            update = {key: value for key, value in fields.items() if value is not None and row.get(key) != value}
            if update:
                row = dict(row, **update)
                self.store.set_transcript_item(row_id, row)
        if rows is not None:
            rows.put(row)
        return True

    def _user(self, message, rows, facts, unbound):
        message_id = message["id"]
        if rows.find("userMessage", message_id) is not None or self.store.transcript_item(message_id, None) is not None:
            return
        if unbound:                                   # the host row of this message may be waiting for its id
            facts["deferred_users"].append(message_id)
            return
        self._native_user(message, synthetic=False, rows=rows)

    def _synthetic(self, message):
        if self.store.transcript_item(message["id"], None) is None:
            self._native_user(message, synthetic=True)

    def _native_user(self, message, *, synthetic, rows=None):
        """``userMessage`` row of a native user/synthetic message, written in ONE store write (the shape of
        ``TranscriptWriter.user_message`` plus ``source``/``synthetic``) so ``submissions()`` never sees it as a host
        row."""
        message_id = message["id"]
        row = self.transcript._row(message_id, dict(type="userMessage", content=[dict(
            type="text", text=_text(message.get("text")))], source="native", synthetic=synthetic), native_id=message_id)
        if rows is not None:
            rows.put(row)

    def _assistant(self, message, rows, facts):
        message_id, finished = message["id"], _finished(message)
        if finished:
            self._finished.add(message_id)
        ordinals = dict.fromkeys(_STREAMED, 0)
        for item in message.get("content") or ():
            item = _mapping(item)
            kind = item.get("type")
            if kind in _STREAMED:
                self._stream_row(message_id, kind, ordinals[kind], _text(item.get("text")), finished=finished)
                ordinals[kind] += 1
            elif kind == "tool" and isinstance(item.get("id"), str) and isinstance(item.get("name"), str):
                if not finished:                           # live call: later tool frames build on this state
                    self._tools[(message_id, item["id"])] = dict(item)
                self._tool(item, message_id, rows, facts)
        self._step_error(message_id, message.get("error"))
        retry = _mapping(message.get("retry"))
        if retry:
            self._retry(message_id, retry)

    def _stream_row(self, message_id, kind, ordinal, text, *, finished):
        """agentMessage/reasoning row of one text/reasoning item (no row until it has text). ``finished``: the step
        was closed in the read (or frame) this text comes from; a row written ``completed`` is never rewritten from an
        unfinished read."""
        row_id = f"{message_id}:{kind}:{ordinal}"
        existing = self.store.transcript_item(row_id, None)
        if not text.strip() and existing is None:
            return
        common = dict(message_id=message_id, item_id=row_id, native_id=message_id)
        if kind == "text":
            status = "completed" if finished else "inProgress"
            if status == "inProgress" and existing is not None and existing.get("status") == "completed":
                return                                   # a stale read never reopens a finished step
            self.transcript.assistant_text(text, status=status, **common)
        else:
            self.transcript.reasoning(text, **common)

    def _step_error(self, message_id, error):
        error = _mapping(error)
        if error and error.get("type") != ABORTED_ERROR_TYPE:
            self.transcript.error(error_code(error, STEP_FAILED), item_id=f"{message_id}:error", native_id=message_id)

    def _retry(self, message_id, retry):
        attempt = retry.get("attempt")
        attempt = attempt if type(attempt) is int else "unknown"
        code = error_code(retry.get("error"), f"{RETRY_ERROR}:{attempt}")
        row_id = f"{message_id}:retry:{attempt}"
        self.transcript.error(code, item_id=row_id, native_id=message_id)
        return row_id

    def _compaction_message(self, message, facts):
        row_id = message["id"]
        status = _COMPACTION_STATUS.get(message.get("status"), "inProgress")
        error = message.get("error")
        fields = dict(type="contextCompaction", reason=message.get("reason") if isinstance(message.get("reason"), str)
                      else None, summary=message.get("summary") if isinstance(message.get("summary"), str) and
                      message["summary"] else None,
                      error=dict(message=error_code(error, COMPACTION_FAILED)) if isinstance(error, Mapping) else None)
        existing = self.store.transcript_item(row_id, None)
        if existing is not None and ((existing.get("status") == status and all(
                value is None or existing.get(key) == value for key, value in fields.items()))
                or (status == "inProgress" and existing.get("status") in ("completed", "failed"))):
            return                                       # unchanged, or a stale read of a finished compaction
        # TranscriptWriter.compaction() has no status parameter; its _row writes the same row shape (incl. the
        # subagent writer's parent link) with the status in one store write.
        self.transcript._row(row_id, fields, status=status, native_id=row_id)
        if existing is None:
            facts["compactions"].append(row_id)

    # ----- tools -----

    def _tool(self, item, message_id, rows, facts):
        name = _text(item.get("name"))
        if name.startswith(self.prefix):
            self._registry_tool(item, message_id, rows, facts)
        else:
            self._native_tool(item, message_id)

    def _native_tool(self, item, message_id):
        row_id, name = f"{message_id}:{item['id']}", _text(item.get("name"))
        state = _mapping(item.get("state"))
        status = _TOOL_STATUS.get(state.get("status"))
        existing = self.store.transcript_item(row_id, None)
        if status is None or (existing and (existing.get("interrupted") or (
                status == "inProgress" and existing.get("status") in ("completed", "failed")))):
            return                       # streaming: input incomplete; an interrupted or settled row is never reopened
        arguments, metadata = dict(_mapping(state.get("input"))), _mapping(state.get("metadata"))
        output = _content_text(state.get("content"))
        error = _mapping(state.get("error"))
        error = {key: error[key] for key in ("message", "type") if isinstance(error.get(key), str)} or None
        common = dict(status=status, message_id=message_id, native_id=item["id"])
        if name == _SHELL:
            exit_code = metadata.get("exit")
            self.transcript.command(row_id, arguments.get("command"), aggregated_output=output,
                                    exit_code=exit_code if type(exit_code) is int else None, **common)
        elif name in _FILE_TOOLS:
            self.transcript.file_change(row_id, self._changes(name, arguments, metadata), **common)
        else:
            self.transcript.native_tool(row_id, name, arguments, result=output, error=error, **common)
        self._native_rows.add(row_id)
        if error is not None and (name == _SHELL or name in _FILE_TOOLS):
            self._annotate(row_id, error=error)

    @staticmethod
    def _changes(name, arguments, metadata):
        """``fileChange.changes`` (``[{path, kind, diff}]``) from ``metadata.files`` once finished, else the input."""
        files = metadata.get("files")
        if isinstance(files, list) and files:
            return [dict(path=f.get("file"), kind=_DIFF_KIND.get(f.get("status"), f.get("status")), diff=f.get("patch"))
                    for f in files if isinstance(f, Mapping)]
        if name == "patch":
            return [dict(path=path.strip(), kind=_PATCH_KIND[kind], diff=None)
                    for kind, path in _PATCH_FILE.findall(_text(arguments.get("patchText")))]
        return [dict(path=arguments.get("path"), kind="update" if name == "edit" else "write", diff=None)]

    def _registry_tool(self, item, message_id, rows, facts):
        call_id, row_id = item["id"], f"{message_id}:{item['id']}"
        state = _mapping(item.get("state"))
        name = _text(item.get("name"))[len(self.prefix):]
        if (message_id, call_id) in self._registry:
            return                                       # matched and annotated earlier (rows keep their linkage)
        if state.get("status") not in _TOOL_STATUS:
            return                                       # streaming: no input yet, nothing to match
        # a provider may reuse call ids across messages (e.g. "call_0"): a linkage is (message id, tool id)
        found = rows.find("dynamicToolCall", call_id, message_id)
        target = found["id"] if found else None
        if target is None and self.store.transcript_item(row_id, None) is not None:
            target = row_id                              # failed generic row written earlier
        if target is None:
            turn = self.transcript.turn_id
            scope = [row for row in rows.rows if row.get("type") == "dynamicToolCall" and not row.get("native_id")
                     and row.get("tool") == name and (turn is None or row.get("turn_id") == turn)]
            key = _canonical(state.get("input"))
            match = next((row for row in scope if key is not None and _canonical(row.get("arguments")) == key), None)
            if match is not None:
                target = match["id"]
            elif state.get("status") == "error" and not scope:
                error = _mapping(state.get("error"))
                self.transcript.native_tool(row_id, self.prefix + name, dict(_mapping(state.get("input"))),
                                            error={key: error[key] for key in ("message", "type")
                                                   if isinstance(error.get(key), str)} or None,
                                            status="failed", message_id=message_id, native_id=call_id)
                target = row_id
        if target is None:
            facts["unmatched_registry"].append(call_id)
            return
        self._registry[(message_id, call_id)] = target
        self._annotate(target, rows, native_id=call_id, message_id=message_id)


def _model_ref(value):
    """``{id, providerID, variant?}`` of a native model reference (string fields only), or None."""
    value = _mapping(value)
    ref = {key: value[key] for key in ("id", "providerID", "variant") if isinstance(value.get(key), str)}
    return ref or None
