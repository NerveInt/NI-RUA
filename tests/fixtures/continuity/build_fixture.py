"""Build ``continuity_fixture.json``, the ContinuityTracker acceptance data (design §4.3, P1-3a; subagent chains:
design Revision 4 R4.3, R4-4).

Sources are the generation requests captured by the Claude Code spike, by probe R4-0 and by the OpenCode 2.2.7
capture of work package OC2-E1:

* S-B, Claude Code CLI 2.1.321 (report 05): every ``<evidence>/*/model-requests*.jsonl`` with at least
  two generation requests that carry ``messages``. The thread key is the ``X-Claude-Code-Session-Id`` header.
* OpenCode 2.2.7: every conversation scenario of ``tests/fixtures/opencode/capture_opencode.py`` (``basic``,
  ``steer``, ``subagent``, ``compact``) on the ``anthropic_messages``, ``openai_responses`` and ``openai_chat``
  wires, read from ``<evidence>/<scenario>-<wire>/requests.json``. The thread key is the ``x-opencode-session-id``
  header (a child session's own id on its requests). The capture launches the shipped config, so the NI-RUA
  instructions are the config ``instructions`` file, part of the system text of every request (constant across a
  compaction and identical in a ``general`` child session). These replace the opencode 1.19.5 spike sequences
  (report 06), which no longer match what the pinned client sends.
* R4-0, Claude Code CLI 2.1.321 with subagents on (probe R4-0, ``scratch/r4-0``): every
  ``<evidence>/*/model-requests.jsonl`` of a finished set (``result.json`` present) with at least one
  generation request that carries the ``x-claude-code-agent-id`` header, i.e. one made by a ``Task``/``Agent``
  subagent. Main and subagent requests share the session id. These sequences carry ``"source": "r4-0"``
  so that the spike sequences keep their own expected counts.

Every request is kept, in capture order, with what the tracker and its replay need:

* ``thread``: the session id, mapped to ``t1``, ``t2``, ... per sequence.
* ``status``: the HTTP status the fake model answered, or null.
* ``delivered``: true when the completion reached the agent as a usable answer. That means HTTP 200,
  no client disconnect, and not in ``UNDELIVERED`` (aborted model calls documented by the spike
  reports). OpenCode rows take the capture's own ``delivered`` and ``note`` (false for the model call
  the capture interrupted).
* ``label``: ground truth from content markers, never from any tracker rule.
  ``helper`` = no tools declared (Claude Code title call; checked against its prompt text). OpenCode 2.2.7
  sent none in these scenarios (the session has an explicit title, a child session's title is the subagent
  description, and the compaction summary request carries the tools); the builder refuses one.
  ``compaction_request`` = a summary request that carries tools and only appends to the previous history:
  Claude Code's own (``CLAUDE_SUMMARY_MARK``), or OpenCode's (``OPENCODE_SUMMARY_MARKS`` in its last
  conversation item: the first summary, a checkpoint update, and the one retry after a summary that missed
  the template).
  ``context_lost`` = the first main request on a thread after a compaction (a summary request), or a
  Claude Code main request that holds more ``[Old tool result content cleared]`` markers than the
  previous main request on its thread (a prune rewrite; OpenCode 2.x has no prune).
  ``continuous`` = everything else.
  ``subagent`` = a request with tools that carries the client's own attribution of the request to a
  subagent (the tracker never reads these headers): ``x-claude-code-agent-id`` (Claude Code, same session
  id as the main chain) or ``x-opencode-parent-session-id`` (an OpenCode child session, own thread). The
  other labels of such a sequence are computed over its other requests only.
* ``body``: the request body reduced to ``system`` / ``instructions`` / ``messages`` / ``input`` /
  ``tools``. Tools keep only ``type`` and ``name``. Conversation items and system values are stored once
  per sequence in ``values`` and referenced by index (a list field holds a list of indexes). Every string
  longer than ``LONG_TEXT`` characters, and every string with a local path or a credential-like
  marker, is replaced by ``<sha256:16 hex len=N>``, which removes media bytes, environment dumps and
  long tool output. The replacement is a deterministic function of the string, so equality between
  any two strings, and with it every continuity relation, is preserved.

The evidence directories (the spike scratch areas of report 05 and probe R4-0, and the OC2-E1 capture
directory) are not checked in, and the fixture is the durable copy. Usage, from the worktree root::

    mamba run -n ni-rua python tests/fixtures/continuity/build_fixture.py \
        --claude /tmp/ni-rua-agent-harness/scratch/s-b/evidence \
        --opencode <capture_opencode.py evidence directory> \
        --subagents /tmp/ni-rua-agent-harness/scratch/r4-0/evidence
"""
import argparse
import glob
import hashlib
import json
import os
from pathlib import Path

OUT = Path(__file__).resolve().with_name("continuity_fixture.json")
LONG_TEXT = 48
PRIVATE_MARKERS = ("/Users/", "/private/", "/tmp/", "/home/", "/var/", "API_KEY", "TOKEN", "SECRET", "PASSWORD")
CLAUDE_SUMMARY_MARK = "create a detailed summary"
CLAUDE_TITLE_MARK = "Write the title"
CLAUDE_AGENT_HEADER = "x-claude-code-agent-id"
OPENCODE_PARENT_HEADER = "x-opencode-parent-session-id"
# OpenCode 2.2.7 compaction instructions (last conversation item of its summary request): the first summary and a
# checkpoint update share the template intro; the second mark is the one retry after a summary without the template.
OPENCODE_SUMMARY_MARKS = ("You MUST use this format for your response", "did not fill in the required summary template")
PRUNE_MARK = "[Old tool result content cleared]"
# HTTP 200 answers the agent never received as a usable completion (index = position among the
# sequence's generation requests). OpenCode rows carry the capture's own ``delivered`` / ``note``.
UNDELIVERED = {
    ("claude_code/s3-gated-model-p_now/model-requests", 1):
        "report 05 S3: steer with priority now while the model call was gated -> terminal_reason aborted_streaming",
}
# OpenCode 2.2.7 capture: scenarios and wires read from the evidence directory, and the short wire names of the ids.
OPENCODE_SCENARIOS = ("basic", "steer", "subagent", "compact")
WIRES = {"anthropic": "anthropic_messages", "responses": "openai_responses", "chat": "openai_chat"}


def scrub(node):
    if isinstance(node, dict):
        return {k: scrub(v) for k, v in node.items()}
    if isinstance(node, list):
        return [scrub(v) for v in node]
    if isinstance(node, str) and (len(node) > LONG_TEXT or any(m in node for m in PRIVATE_MARKERS)):
        return "<sha256:%s len=%d>" % (hashlib.sha256(node.encode("utf-8", "surrogatepass")).hexdigest()[:16], len(node))
    return node


def reduce_tools(tools):
    out = []
    for tool in tools or []:
        kept = {k: tool[k] for k in ("type", "name") if k in tool}
        if isinstance(tool.get("function"), dict):
            kept["function"] = {"name": tool["function"].get("name")}
        out.append(kept)
    return out


def wire_of(body):
    if "input" in body:
        return "openai_responses"
    if "system" in body or not (body.get("messages") and body["messages"][0].get("role") == "system"):
        return "anthropic_messages"
    return "openai_chat"


class Sequence:
    def __init__(self, sid, client, wire, thread_header):
        self.data = {"id": sid, "client": client, "wire": wire, "thread_header": thread_header,
                     "values": [], "requests": []}
        self._index, self._threads = {}, {}

    def ref(self, value):
        value = scrub(value)
        key = json.dumps(value, ensure_ascii=True)
        if key not in self._index:
            self._index[key] = len(self.data["values"])
            self.data["values"].append(value)
        return self._index[key]

    def add(self, headers, body, status, delivered, label, note=None):
        headers = {k.lower(): v for k, v in (headers or {}).items()}
        thread = self._threads.setdefault(headers.get(self.data["thread_header"], ""), "t%d" % (len(self._threads) + 1))
        reduced = {}
        for key in ("system", "instructions"):
            if key in body:
                reduced[key] = self.ref(body[key])
        for key in ("messages", "input"):
            if isinstance(body.get(key), list):
                reduced[key] = [self.ref(item) for item in body[key]]
            elif key in body:
                reduced[key] = self.ref(body[key])
        if "tools" in body:
            reduced["tools"] = self.ref(reduce_tools(body["tools"]))
        row = {"thread": thread, "status": status, "delivered": delivered, "label": label, "body": reduced}
        if note:
            row["note"] = note
        self.data["requests"].append(row)


def _agent_id(row):
    return {k.lower(): v for k, v in (row.get("headers") or {}).items()}.get(CLAUDE_AGENT_HEADER)


def claude_sequences(root, *, source=None):
    """S-B sequences; with ``source="r4-0"`` the R4-0 subagent sequences (see the module docstring)."""
    for path in sorted(glob.glob(os.path.join(root, "*", "model-requests*.jsonl"))):
        rows = [json.loads(line) for line in open(path)]
        rows = [r for r in rows if r.get("is_generation") and isinstance(r.get("json"), dict) and r["json"].get("messages")]
        if len(rows) < 2:
            continue
        if source is not None and not (os.path.exists(os.path.join(os.path.dirname(path), "result.json"))
                                       and any(_agent_id(r) and r["json"].get("tools") for r in rows)):
            continue
        prefix = "claude_code/" if source is None else "claude_code/%s/" % source
        sid = prefix + "%s/%s" % (os.path.basename(os.path.dirname(path)), Path(path).stem)
        seq = Sequence(sid, "claude_code", "anthropic_messages", "x-claude-code-session-id")
        if source is not None:
            seq.data["source"] = source
        previous_summary, pruned = False, 0
        for k, r in enumerate(rows):
            body = r["json"]
            tail = json.dumps(body["messages"][-1:])
            if _agent_id(r) and body.get("tools"):
                label = "subagent"
            elif not body.get("tools"):
                assert CLAUDE_TITLE_MARK in tail, (sid, k, "tool-less request that is not a title call")
                label = "helper"
            elif CLAUDE_SUMMARY_MARK in tail:
                label = "compaction_request"
            else:
                marks = json.dumps(body["messages"]).count(json.dumps(PRUNE_MARK)[1:-1])
                label = "context_lost" if previous_summary or marks > pruned else "continuous"
                pruned = marks
            if label not in ("helper", "subagent"):
                previous_summary = label == "compaction_request"
            status = r.get("response_status")
            note = UNDELIVERED.get((sid, k))
            delivered = status == 200 and not r.get("client_disconnected") and note is None
            if r.get("client_disconnected"):
                note = "client disconnected before the answer was delivered"
            seq.add(r.get("headers"), body, status, delivered, label, note)
        yield seq.data


def _last_item(body):
    items = body.get("messages") if isinstance(body.get("messages"), list) else body.get("input")
    return json.dumps(items[-1:]) if isinstance(items, list) else ""


def opencode_sequences(root):
    """OpenCode 2.2.7 sequences of the OC2-E1 capture (see the module docstring), ids ``opencode/<scenario>-<wire>``
    with the short wire names of ``WIRES``."""
    for scenario in OPENCODE_SCENARIOS:
        for short, wire in WIRES.items():
            rows = json.load(open(os.path.join(root, "%s-%s" % (scenario, wire), "requests.json")))
            sid = "opencode/%s-%s" % (scenario, short)
            seq = Sequence(sid, "opencode", wire, "x-opencode-session-id")
            assert wire_of(rows[0]["body"]) == wire, sid
            after_summary = set()
            for k, r in enumerate(rows):
                body, headers = r["body"], {kk.lower(): v for kk, v in r["headers"].items()}
                thread = headers.get("x-opencode-session-id", "")
                assert body.get("tools"), (sid, k, "tool-less OpenCode 2.2.7 request")
                if headers.get(OPENCODE_PARENT_HEADER):
                    label = "subagent"
                elif any(mark in _last_item(body) for mark in OPENCODE_SUMMARY_MARKS):
                    label = "compaction_request"
                    after_summary.add(thread)
                else:
                    label = "context_lost" if thread in after_summary else "continuous"
                    after_summary.discard(thread)
                seq.add(r["headers"], body, r["status"], bool(r["delivered"]), label, r.get("note"))
            yield seq.data


def main():
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--claude", required=True, help="S-B evidence directory (Claude Code spikes, report 05)")
    parser.add_argument("--opencode", required=True,
                        help="OpenCode 2.2.7 capture directory (tests/fixtures/opencode/capture_opencode.py)")
    parser.add_argument("--subagents", required=True, help="R4-0 evidence directory (Claude Code with subagents)")
    parser.add_argument("--out", default=str(OUT))
    args = parser.parse_args()
    sequences = (list(claude_sequences(args.claude)) + list(opencode_sequences(args.opencode))
                 + list(claude_sequences(args.subagents, source="r4-0")))
    fixture = {
        "format": "ni_rua.continuity_fixture.v1",
        "about": ("Generation requests captured by the Claude Code spike (S-B, CLI 2.1.321), probe R4-0 (Claude Code "
                  "with subagents) and the OpenCode 2.2.7 capture (OC2-E1: steer, interrupts, foreground subagent, "
                  "auto-compaction) against loopback fake models, reduced for the ContinuityTracker replay; built by "
                  "build_fixture.py (see its docstring for the reduction, the labels and UNDELIVERED)."),
        "sequences": sequences,
    }
    text = json.dumps(fixture, ensure_ascii=True, separators=(",", ":"))
    # One sequence per line keeps diffs reviewable.
    lines = ['{"format":%s,"about":%s,"sequences":[' % (json.dumps(fixture["format"]), json.dumps(fixture["about"]))]
    lines += [json.dumps(s, ensure_ascii=True, separators=(",", ":")) + ("," if i < len(sequences) - 1 else "")
              for i, s in enumerate(sequences)]
    lines.append("]}")
    out = "\n".join(lines) + "\n"
    assert json.loads(out) == json.loads(text)
    Path(args.out).write_text(out)
    print("wrote %s: %d sequences, %d requests, %d bytes" % (
        args.out, len(sequences), sum(len(s["requests"]) for s in sequences), len(out.encode())))


if __name__ == "__main__":
    main()
