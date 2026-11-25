"""Builds the OpenCode 2.2.7 fixtures of this directory from key-free probe captures (design 11 sections 1.6-1.9).

Every capture is a real ``opencode serve`` 2.2.7 (darwin-arm64) run with a fake model server on loopback
(``tests/support/fake_model_servers.py``), the real ``NiRuaMcpServer`` and isolated HOME/XDG/TMPDIR; no real model and
no network. Two probe families:

* ``/tmp/ni-rua-agent-harness/oc2/runs/<scenario>/out/`` (design 11 probes s01-s06, scripts
  ``/tmp/ni-rua-agent-harness/oc2/bin/sNN_*.py``): the final ``GET /api/session/{id}/message`` list, the ``/api/event``
  frames, and for subagents the child's list (``child-<suffix>.json``). ``active``/``inbox`` were not read with those
  snapshots and are recorded as ``null`` (unknown).
* ``/tmp/oc2c-probe/runs/oc2c-<scenario>/out/capture.json`` (OC2-C probe, same lab): ordered snapshots
  ``{label, messages, active, inbox}`` read at the labelled moments, the prompt responses and the frames.

Output, one file per behaviour: ``{capture, session_id, prompts, snapshots, children, events}`` where ``prompts`` are
the user message ids ``POST /api/session/{id}/prompt`` returned (``session.inbox.enqueued`` of a user item for the
s01-s06 family), ``snapshots[-1]`` is the final state, and ``events`` keeps the ``session.*``/``permission.*`` frames
as ``{id, created, type, data}``. Sanitised: run directories become ``/runs/<scenario>``; ``location``, ``durable``,
``projectID``, ``slug`` and the instruction digests of ``session.instructions.updated`` are dropped (not used by the
projector or the verdict). Provider error bodies, fake reasoning signatures and the small probe PNG stay on purpose:
the tests assert that none of them reaches a transcript row.

Run (from the worktree root, after the probes): ``mamba run -n ni-rua python tests/fixtures/opencode/v2/build_v2_fixtures.py``
"""
import json
import re
from pathlib import Path

HERE = Path(__file__).resolve().parent
DESIGN_RUNS = Path("/tmp/ni-rua-agent-harness/oc2/runs")
OC2C_RUNS = Path("/tmp/oc2c-probe/runs")
BINARY = "opencode serve 2.2.7 (darwin-arm64)"
WRITTEN = "2026-11-13"
DROP_EVENT_TYPES = {"session.instructions.updated", "session.usage.updated"}
DROP_DATA_KEYS = {"projectID", "location", "slug", "version", "subpath"}
PATH = re.compile(r"/(?:private/)?tmp/(?:ni-rua-agent-harness/oc2|oc2c-probe)/runs/(?:oc2c-)?([A-Za-z0-9_.-]+)")

# fixture name -> (family, scenario, what the probe did)
SOURCES = {
    "tool_turn": ("design", "s01-anthropic_messages", "Anthropic Messages wire: reasoning + commentary text + one "
                  "ni_rua_observe call returning text and a PNG, then the final text"),
    "tool_turn_responses": ("design", "s01-openai_responses", "same turn on the OpenAI Responses wire (native "
                            "itemId/responseId/encrypted reasoning fields)"),
    "steer": ("oc2c", "steer", "P1 -> ni_rua_wait_s 3 s; STEER-1 posted during the tool call (snapshot "
              "steer_in_inbox read right after the steer's 200); answer after the steer"),
    "steer_late": ("design", "s03-steer-late", "P1 -> FIRST-FINAL held at a gate; LATE-STEER posted while the final "
                   "text streams; one more model call in the same execution; one idle"),
    "interrupt_tool": ("design", "s03-interrupt-tool", "P1 -> ni_rua_wait_s 8 s; POST /interrupt during the call; "
                       "then a second prompt P2 answered normally"),
    "interrupt_stream": ("design", "s03-interrupt-stream", "P1 -> the model response is held at a gate; POST "
                         "/interrupt during the stream"),
    "error_400": ("design", "s03-err400", "the provider answers 400 (no retry)"),
    "error_500_retried": ("oc2c", "retry", "the provider answers 500 twice, then text (snapshot retrying read during "
                          "the retry wait)"),
    "error_500_exhausted": ("design", "s03-err500", "the provider answers 500 on every attempt (11 attempts)"),
    "stream_error_continuation": ("design", "s03-stream-error", "8 responses fail mid-stream after partial text; "
                                  "OpenCode continues each with a synthetic message; the 9th succeeds"),
    "length_continuation": ("design", "s03-length", "the first response stops with max_tokens; OpenCode continues "
                            "with a synthetic message"),
    "subagent_foreground": ("oc2c", "foreground", "main calls subagent (foreground); the child runs ni_rua_wait_s 2 s "
                            "and answers; snapshot child_running read while both sessions are active"),
    "subagent_background": ("oc2c", "background", "main calls subagent with background:true, answers and goes idle "
                            "while the child runs ni_rua_wait_s 3 s; the child's report is enqueued as a synthetic "
                            "message and the main session runs again"),
    "interrupt_foreground_subagent": ("design", "s04-interrupt-fg-sub", "main calls a foreground subagent whose "
                                      "child waits 6 s in ni_rua_wait_s; POST /interrupt on the main session cascades "
                                      "to the child"),
    "compaction": ("design", "s06-compaction", "two ni_rua_observe calls push the context over the limit; auto "
                   "compaction succeeds; the turn continues and finishes"),
    "compaction_failed": ("design", "s05-compaction", "auto compaction whose summary misses the template twice; the "
                          "turn fails compaction.failed"),
    "crash_without_idle": ("oc2c", "crash", "T1 answered; T2 -> ni_rua_wait_s 30 s; the server is killed with "
                           "SIGKILL during the call and restarted (snapshots after_crash/after_crash_later: tool "
                           "still running, no idle); T3 afterwards"),
    "native_tools": ("oc2c", "native", "write, edit, failing edit, shell with exit 3, glob, read, grep, failing read, "
                     "then the final text"),
    "patch_tool": ("oc2c", "patch", "GPT model id on the Responses wire: one patch call (add, update, delete)"),
    "shell_running": ("oc2c", "shell-running", "a 3 s shell command; snapshot shell_running read during it"),
    "model_switch": ("oc2c", "switch", "T1 answered; POST /api/session/{id}/model to second-model while idle; T2"),
    "permission_rejected": ("oc2c", "permission", "default permissions (read *.env -> ask); P1 -> read .env; the "
                            "permission request is answered POST .../permission/{id}/reply {decision: reject} "
                            "(snapshots permission_asked, after_reject, after_reject_later: no idle is written); P2 "
                            "answered normally"),
}


def scrub(value):
    return json.loads(PATH.sub(lambda match: "/runs/" + match.group(1), json.dumps(value)))


def clean_event(frame):
    data = {key: value for key, value in (frame.get("data") or {}).items() if key not in DROP_DATA_KEYS}
    return dict(id=frame.get("id"), created=frame.get("created"), type=frame.get("type"), data=data)


def keep_event(frame):
    kind = frame.get("type") or ""
    return kind.startswith(("session.", "permission.")) and kind not in DROP_EVENT_TYPES


def design_capture(scenario):
    out = DESIGN_RUNS / scenario / "out"
    frames = [record["data"] for record in map(json.loads, (out / "events.jsonl").read_text().splitlines())
              if isinstance(record.get("data"), dict)]
    main = next(f["data"]["sessionID"] for f in frames if f.get("type") == "session.created"
                and not f["data"].get("parentID"))
    prompts = [f["data"]["inboxID"] for f in frames if f.get("type") == "session.inbox.enqueued"
               and f["data"].get("sessionID") == main and (f["data"].get("item") or {}).get("type") == "user"]
    children = {}
    for path in sorted(out.glob("child-*.json")):
        child = next(f["data"]["sessionID"] for f in frames if f.get("type") == "session.created"
                     and f["data"]["sessionID"].endswith(path.stem.split("-", 1)[1]))
        children[child] = dict(messages=json.loads(path.read_text()))
    messages = json.loads((out / "messages.json").read_text())
    return dict(session_id=main, prompts=prompts, children=children, events=frames,
                snapshots=[dict(label="final", messages=messages, active=None, inbox=None)])


def oc2c_capture(scenario):
    capture = json.loads((OC2C_RUNS / f"oc2c-{scenario}" / "out" / "capture.json").read_text())
    prompts = [p["response"]["data"]["id"] for p in capture["prompts"] if "data" in (p.get("response") or {})]
    snapshots, previous = [], None
    for snapshot in capture["snapshots"]:
        key = json.dumps([snapshot["messages"], snapshot["active"], snapshot["inbox"]], sort_keys=True)
        if key == previous:                     # polls that saw nothing new
            continue
        previous = key
        snapshots.append(dict(label=snapshot["label"], messages=snapshot["messages"], active=snapshot["active"],
                              inbox=snapshot["inbox"]))
    children = {child: dict(messages=entry["messages"]) for child, entry in capture["children"].items()}
    return dict(session_id=capture["session_id"], prompts=prompts, children=children, events=capture["events"],
                snapshots=snapshots)


def build(name):
    family, scenario, description = SOURCES[name]
    data = design_capture(scenario) if family == "design" else oc2c_capture(scenario)
    probe = (f"/tmp/ni-rua-agent-harness/oc2/bin/{scenario.split('-')[0]}_*.py:{scenario}" if family == "design"
             else f"OC2-C probe (lab.py of the design probes):{scenario}")
    fixture = dict(
        capture=dict(binary=BINARY, probe=probe, scenario=description, written=WRITTEN,
                     sanitised="run directories -> /runs/<scenario>; location/durable/projectID/slug and instruction "
                               "digests dropped; fake model, fake signatures, no credential"),
        session_id=data["session_id"], prompts=data["prompts"], snapshots=data["snapshots"],
        children=data["children"], events=[clean_event(f) for f in data["events"] if keep_event(f)])
    (HERE / f"{name}.json").write_text(json.dumps(scrub(fixture), indent=1) + "\n")
    return name


if __name__ == "__main__":
    for fixture_name in SOURCES:
        print(build(fixture_name))
