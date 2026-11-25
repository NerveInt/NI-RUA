"""Shortening of a finished actor reply: lossless compaction, fixed notes stated once, one detail entry.

The first part (``same_as``, ``float32``):

Two independent pure transforms over the JSON text items of one presented reply. Neither removes a fact, rounds a
measurement or reorders keys; images and non-JSON text pass through untouched, and program callers never reach this
module (it is applied in ``Harness.handle_tool`` only).

``same_as`` (``features.reply_same_as``)
    Inside one text item, a JSON object/array or a long string that is byte-identical to a value printed earlier in the
    same item is printed once. The later copy becomes ``{"same_as": "<JSON Pointer>"}`` (objects, arrays) or the
    string ``"same_as:<JSON Pointer>"`` (strings), where the pointer (RFC 6901, rooted at the item) names the first
    copy. Example: ``"receipt": {"action_budget": {"same_as": "/action_budget"}}`` states that the receipt's budget
    equals the top-level ``action_budget`` object in the same text item. A copy is replaced only when that saves at
    least ``MIN_SAVING`` characters.

``float32`` (``features.reply_float32``)
    A JSON number whose value is exactly representable as an IEEE-754 float32 (it was a float32 measurement widened to
    float64 before printing, e.g. ``1.0667834281921387``) is printed with the shortest decimal that parses back to the
    same float32 (``1.0667834``). Numbers that are not float32-exact (float64 results) are printed unchanged, so no
    value loses precision relative to its source representation.
"""
from __future__ import annotations

import json
import struct
from typing import NamedTuple

from ..contracts import INSPECT_STEP_LIMIT, canonical

# How an actor reads a stand-in; stated once in the catalogue while ``features.reply_same_as`` is on.
SAME_AS_CATALOGUE_NOTE = (
    ' Repeated values: within one reply text part a value that occurs again is printed once; the later place holds '
    '{"same_as": "<JSON Pointer>"} (or the string "same_as:<JSON Pointer>" for a repeated string), and the pointer '
    "(RFC 6901, from the root of that text part) names the place that holds the value."
)
SAME_AS_KEY = "same_as"
SAME_AS_PREFIX = "same_as:"
MIN_CONTAINER_CHARS = 80
MIN_STRING_CHARS = 60
MIN_SAVING = 40


def _pointer_token(key):
    return str(key).replace("~", "~0").replace("/", "~1")


def dedup_same_as(value):
    """Return ``value`` with later byte-identical copies replaced by pointers to the first copy (document order)."""
    seen = {}

    def visit(node, path):
        if isinstance(node, (dict, list)):
            text = canonical(node)
            if len(text) >= MIN_CONTAINER_CHARS:
                first = seen.get(text)
                if first is not None:
                    pointer = {SAME_AS_KEY: first}
                    if len(text) - len(canonical(pointer)) >= MIN_SAVING:
                        return pointer
                else:
                    seen[text] = path
            if isinstance(node, dict):
                # canonical() prints keys sorted; visit in that order so "earlier" matches the printed text
                return {key: visit(node[key], path + "/" + _pointer_token(key)) for key in sorted(node)}
            return [visit(item, f"{path}/{index}") for index, item in enumerate(node)]
        if isinstance(node, str) and len(node) >= MIN_STRING_CHARS:
            first = seen.get(node)
            if first is not None:
                pointer = SAME_AS_PREFIX + first
                if len(node) - len(pointer) >= MIN_SAVING:
                    return pointer
            else:
                seen[node] = path
        return node

    return visit(value, "")


def _float32_exact(value):
    try:
        return struct.unpack("<f", struct.pack("<f", value))[0] == value
    except (OverflowError, struct.error):
        return False


def _shortest_float32(token):
    """Shortest JSON number that parses to the same float32 as ``token``; ``token`` itself when nothing is gained."""
    value = float(token)
    if value != value or value in (float("inf"), float("-inf")) or not _float32_exact(value):
        return token
    for digits in range(1, 10):
        candidate = repr(float(f"{value:.{digits}g}"))
        if _float32_exact_same(candidate, value):
            return candidate if len(candidate) < len(token) else token
    return token


def _float32_exact_same(candidate, value):
    try:
        return struct.unpack("<f", struct.pack("<f", float(candidate)))[0] == value
    except (OverflowError, struct.error):
        return False


_NUMBER_START = frozenset("-0123456789")
_NUMBER_BODY = frozenset("+-0123456789.eE")


def shorten_float32_numbers(text):
    """Rewrite float32-exact JSON numbers of serialized JSON ``text``; strings, keys and layout stay byte-identical."""
    out = []
    index, size = 0, len(text)
    while index < size:
        char = text[index]
        if char == '"':
            end = index + 1
            while end < size:
                if text[end] == "\\":
                    end += 2
                    continue
                if text[end] == '"':
                    break
                end += 1
            out.append(text[index:end + 1])
            index = end + 1
        elif char in _NUMBER_START:
            end = index + 1
            while end < size and text[end] in _NUMBER_BODY:
                end += 1
            token = text[index:end]
            out.append(_shortest_float32(token) if ("." in token or "e" in token or "E" in token) else token)
            index = end
        else:
            out.append(char)
            index += 1
    return "".join(out)


def compact_reply_text(text, *, same_as=False, float32=False):
    """One JSON text item -> its compacted text. Text that is not canonical JSON of an object/array is returned as is."""
    if not (same_as or float32) or not isinstance(text, str):
        return text
    try:
        value = json.loads(text)
    except ValueError:
        return text
    if not isinstance(value, (dict, list)) or canonical(value) != text:
        return text
    if same_as:
        text = canonical(dedup_same_as(value))
    if float32:
        text = shorten_float32_numbers(text)
    return text


def compact_content_items(items, *, same_as=False, float32=False):
    """Content items of one actor reply -> compacted items. With both switches off the same list object is returned."""
    if not (same_as or float32):
        return items
    result = []
    for item in items:
        if isinstance(item, dict) and item.get("type") in ("inputText", "input_text") and isinstance(item.get("text"), str):
            compacted = compact_reply_text(item["text"], same_as=same_as, float32=float32)
            item = item if compacted == item["text"] else dict(item, text=compacted)
        result.append(item)
    return result


def same_as_pointers(value, path=""):
    """``(where, target)`` JSON Pointers of every ``same_as`` stand-in inside the parsed text item ``value``."""
    if isinstance(value, dict):
        if set(value) == {SAME_AS_KEY} and isinstance(value[SAME_AS_KEY], str):
            yield path, value[SAME_AS_KEY]
            return
        for key, item in value.items():
            yield from same_as_pointers(item, path + "/" + _pointer_token(key))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            yield from same_as_pointers(item, f"{path}/{index}")
    elif isinstance(value, str) and value.startswith(SAME_AS_PREFIX):
        yield path, value[len(SAME_AS_PREFIX):]


def dangling_same_as(items):
    """``[(item index, where, target)]`` of every ``same_as`` pointer in ``items`` that does not name a value printed
    in full in its own text item: the path is missing, or it ends at another pointer or at a block the reply budget
    replaced (``{"omitted": "by_reference", ...}``). An empty list is the invariant of every delivered reply."""
    missing = object()
    found = []
    for index, item in enumerate(items):
        if not (isinstance(item, dict) and item.get("type") in ("inputText", "input_text")):
            continue
        try:
            root = json.loads(item.get("text"))
        except (TypeError, ValueError):
            continue
        for where, target in same_as_pointers(root):
            node = root
            for token in target.split("/")[1:]:
                token = token.replace("~1", "/").replace("~0", "~")
                try:
                    node = node[int(token)] if isinstance(node, list) else node[token]
                except (KeyError, IndexError, ValueError, TypeError):
                    node = missing
                    break
            if (not target.startswith("/") or node is missing or (isinstance(node, str) and node.startswith(SAME_AS_PREFIX))
                    or (isinstance(node, dict) and (set(node) == {SAME_AS_KEY} or node.get("omitted") == "by_reference"))):
                found.append((index, where, target))
    return found


# --- features.reply_static_once -------------------------------------------------------------------------------------
#
# Some explanatory strings of a reply never change: they describe what a field means or what a check covers, not what
# happened. ``drop_static_once`` leaves them out of routine replies. A string is left out only when
#   * its key and its exact text are listed below (a producer that changes its wording is delivered again), and
#   * the same fact is stated where the model always has it: in the tool description of ``program_run`` (the two
#     program entries) or in the note that ``catalogue_with_reply_notes`` puts into the ``robot_action`` description
#     while the switch is on, and
#   * the run's catalogue shows that the note applies to this robot (``static_once_plan``): a group of notes is stated,
#     and its strings left out, only when the ``robot_action`` description already speaks of the thing the group is
#     about (``requires``), so a robot without native step budgets, joint-travel admission or grasp-contact fields
#     gets neither the sentences nor the omissions, and
#   * the reply is routine. A reply that reports a failure, a stop by a guard, an unknown outcome, a refused plan or
#     a pending review keeps every string, so the scope notes that qualify an exception are next to it. A stop the
#     model asked for (an ``observe`` stage or ``checkpoint_after_gripper``; ``receipt.checkpoint_reason_code``, see
#     ``feedback_delivery.MODEL_REQUESTED_CHECKPOINT_CODES``) is routine.
# Numbers, states, residuals, effective tolerances, reasons and every other string are never touched.

_BUDGET_NOTE = "official native steps; separate from tool requests and model-call budget; reserves are not included"
_COLLISION_NOTE = "native robot self-collision and configured static planner table; no task-object or other-arm collision proof"
_SHORTFALL_NOTE = (
    "Positive approach_shortfall_m means the selected reference remains behind its requested final target along the "
    "requested approach axis. The final target may not have been attempted after failure/stop. Robot geometry and pose "
    "feedback do not establish object contact or retention.")
_TRAVEL_POLICY_NOTE = (
    'admission policy only; periodic delta requires a continuous joint or a shortest equivalent goal within declared '
    'limits; set joint_travel_preset ("extended" or "unconstrained") or max_joint_travel_rad/max_backtracking_rad on '
    'the stage or batch to override travel thresholds; this is not a substitute path; no commands changed or '
    'replanning; no collision or arrival guarantee')
_SAFETY_SOURCES = ("public measured named joint positions", "public robot URDF joint types and declared limits",
                   "compiled candidate joint commands", "controller request and episode action budgets")
_PROGRAM_STATED = "does not prove robot/task success"  # the program_run description sentence the program notes rest on


class _Segment(NamedTuple):
    """One group of fixed notes. ``requires``: substrings that must all occur in the run's ``robot_action``
    description (before any note) for the group to apply; ``feature``: a boolean ``features`` field that must be on
    (the producer runs only then); ``note``: its sentence(s) in the catalogue note; ``strings`` / ``lists`` /
    ``flags``: the ``(key, exact value)`` pairs left out of routine replies when it applies."""
    name: str
    requires: tuple
    note: str
    strings: tuple = ()
    lists: tuple = ()
    flags: tuple = ()
    feature: str = None


_SEGMENTS = (
    _Segment("budget", ("native step",), "action_budget: " + _BUDGET_NOTE + "; a null count means unknown. ",
             strings=(("semantics", _BUDGET_NOTE),), flags=(("unknown_values_are_null", True),)),
    # collision_scope: the text is the one the robot backend publishes in its capabilities (see static_once_plan).
    _Segment("collision", ("native step",), "collision_scope: {collision_scope}. ",
             strings=(("collision_scope", "{collision_scope}"),)),
    _Segment("steps", ("native step",),
             "Step counts: executed_steps is the official native action count delta and includes a partially executed "
             "faulting call; completed_native_steps counts native take_action calls that returned successfully; "
             "executed_control_steps counts physics steps within official action interpolation. ",
             strings=(("executed_steps", "official native action count delta; includes a partially executed faulting call"),
                      ("completed_native_steps", "native take_action calls that returned successfully"),
                      ("executed_control_steps", "physics steps within official action interpolation"))),
    _Segment("grasp_contact", ("contact_region",),
             "grasp_contact_feedback: " + _SHORTFALL_NOTE[0].lower() + _SHORTFALL_NOTE[1:] + " ",
             strings=(("interpretation", _SHORTFALL_NOTE),)),
    _Segment("joint_travel", ("joint_travel_preset",),
             "motion_safety_checks: information_sources are " + ", ".join(_SAFETY_SOURCES) + " unless a row lists its "
             "own; joint-travel-admission-v1 is an " + _TRAVEL_POLICY_NOTE + ". ",
             strings=(("semantics", _TRAVEL_POLICY_NOTE),), lists=(("information_sources", _SAFETY_SOURCES),)),
    _Segment("trajectory", (), feature="trajectory_preview", note=
             "trajectory_feedback: completed commands only, grasp retention, placement and task success are not "
             "assessed; planned rows are stored native command rows (independent arms include scheduled merged rows "
             "only, unexecuted local candidates are separate; a row not recorded completed may have executed partially "
             "in a faulting call; solver endpoints are not executed); measured rows are recorded measurement boundaries "
             "with their original sim_step, and the final boundary may represent a partial faulting call; derivatives "
             "are finite differences over recorded boundary intervals (acceleration at interval midpoints), not "
             "high-frequency extrema or contact evidence; measured_command_error is encoder minus stored target at "
             "completed native boundaries, no contact or task-success inference; geometry uses the declared common "
             "measured frame (the inferred base fit is not independent calibration validation); the automatic overlay "
             "shows the measured route (planned joint samples and tracking comparison are in the JSON, the planned "
             "image is available on demand) on the RGB and camera pose of the action start; no object/contact truth or "
             "replanning, and scene changes, held objects and swept-volume collision are unknown. "
             "trajectory_feedback_ref is the artifact with every exact row, public FK, full overlay metadata and "
             "source traces (program_run: NiRuaClient().artifact(ref)). ",
             strings=(
                 ("semantics", "Completed commands only; grasp retention, placement and task success are not assessed."),
                 ("semantics", "Encoder minus stored target at completed native boundaries; no contact or task-success inference."),
                 ("timing", "Stored native command rows. Independent arms include scheduled merged rows only; unexecuted local candidates are separate."),
                 ("timing", "Recorded measurement boundary index, with original sim_step; final boundary may represent a partial faulting call."),
                 ("derivatives", "Finite differences over recorded boundary intervals; acceleration of interval velocities uses interval midpoints. Not high-frequency extrema or contact evidence."),
                 ("unexecuted_scope", "Not recorded completed; partial faulting execution is possible. Solver endpoints are explicitly not executed."),
                 ("calibration_scope", "Declared common measured frame; inferred base fit is not independent calibration validation."),
                 ("image_scope", "Measured route; planned joint samples and tracking comparison included. Planned image is available on demand."),
                 ("limitations", "No object/contact truth or replanning. RGB/camera pose belongs to action start; scene changes, held objects and swept-volume collision are unknown."))),
    _Segment("trajectory_link6", ("link6",), feature="trajectory_preview", note=
             "trajectory_feedback.geometry includes measured jaws only where retained, and command geometry stops at "
             "link6. ",
             strings=(("jaw_scope", "Measured jaws included only where retained; command geometry stops at link6."),)),
    # Stated by the program_run description itself (no sentence added): applies when that tool is in the catalogue.
    _Segment("program", (), "", strings=(
        ("result_semantics", "Program status describes process execution only; stdout/stderr are program-authored text, not verified robot or task outcomes. action_summary and final_observation come from broker tool replies. awaiting_review retains the paused program; approve or reject with program_run resume, without executing the plan separately."),
        ("semantics", "Recorded action replies only. Process exit and stdout do not establish motion, contact, retention or task success."))),
)
_NOTE_HEAD = (
    " Fixed reply notes (stated once here; routine replies, including a stop at a checkpoint you requested, omit "
    "them; replies that report a failure, a guard pause, a slip, an unknown outcome or a refused plan carry them "
    "inline). ")
_NOTE_DETAIL = (
    ' detail: {"tool", "arguments"} in a reply is a ready call of that tool with exactly those arguments; it returns '
    "the detail omitted at that place.")
STATIC_ONCE_NOTE_TOOL = "robot_action"


class StaticOncePlan(NamedTuple):
    """What ``reply_static_once`` states and leaves out in one run: ``note`` (the text for the ``robot_action``
    description), ``segments`` (names of the groups that apply) and the ``(key, exact value)`` sets ``strings`` /
    ``lists`` / ``flags`` that ``drop_static_once`` leaves out of routine replies."""
    note: str
    segments: tuple
    strings: frozenset
    lists: frozenset
    flags: frozenset


def static_once_plan(tools=None, capabilities=None, features=None):
    """The ``StaticOncePlan`` of a run.

    features: the run's ``FeaturesConfig``; a group whose producer is behind a feature switch
    (``trajectory_preview``) applies only while it is on. None = on.

    tools: the actor catalogue before any reply note (``ToolRegistry`` definitions), or None for the plan with every
    group (ARX-X5 / RoboDojo wording; what the module constants below hold). A group applies when every ``requires``
    substring occurs in the ``robot_action`` description; the program group when ``program_run`` is in the catalogue
    and its description carries the sentence the two program notes repeat.
    capabilities: the robot backend's published capabilities; ``collision_scope`` (a string) is the text the
    collision group states and leaves out. Without it the group does not apply (None with ``tools=None``: the
    RoboDojo text).
    """
    if tools is None:
        action, program, collision = None, True, _COLLISION_NOTE
    else:
        by_name = {tool.get("name"): tool.get("description") or "" for tool in tools}
        action, program = by_name.get(STATIC_ONCE_NOTE_TOOL), _PROGRAM_STATED in by_name.get("program_run", "")
        collision = (capabilities or {}).get("collision_scope")
        collision = collision if isinstance(collision, str) and collision else None
    applied, note, strings, lists, flags = [], "", set(), set(), set()
    for segment in _SEGMENTS:
        if segment.name == "program":
            if not program:
                continue
        elif tools is not None and (action is None or not all(token in action for token in segment.requires)):
            continue
        if segment.feature is not None and features is not None and not getattr(features, segment.feature, True):
            continue
        if segment.name == "collision" and collision is None:
            continue
        fill = (lambda text: text.replace("{collision_scope}", collision)) if segment.name == "collision" else (lambda text: text)
        applied.append(segment.name)
        note += fill(segment.note)
        strings.update((key, fill(text)) for key, text in segment.strings)
        lists.update(segment.lists)
        flags.update(segment.flags)
    note = (_NOTE_HEAD + note.rstrip() if note else "") + _NOTE_DETAIL
    return StaticOncePlan(note, tuple(applied), frozenset(strings), frozenset(lists), frozenset(flags))


_FULL_PLAN = static_once_plan()
# The plan with every group: (key, exact value) of the strings / lists / flags left out, and the full note.
STATIC_ONCE_STRINGS = _FULL_PLAN.strings
STATIC_ONCE_LISTS = _FULL_PLAN.lists
STATIC_ONCE_FLAGS = _FULL_PLAN.flags
STATIC_ONCE_CATALOGUE_NOTE = _FULL_PLAN.note

_TEXT_ITEM_TYPES = ("inputText", "input_text")
_RECEIPT_KEYS = ("completed_native_steps", "executed_steps", "completed_steps", "batch_status")
_EXCEPTION_KEYS = ("failure_feedback", "anomaly_reason", "blocking_risks", "pending_review", "error")
_MISSING = object()


def _parsed(item):
    """The JSON object/array of a canonical JSON text item, else ``_MISSING`` (images, prose, non-canonical text)."""
    if not (isinstance(item, dict) and item.get("type") in _TEXT_ITEM_TYPES and isinstance(item.get("text"), str)):
        return _MISSING
    try:
        value = json.loads(item["text"])
    except ValueError:
        return _MISSING
    if not isinstance(value, (dict, list)) or canonical(value) != item["text"]:
        return _MISSING
    return value


def reports_exception(value):
    """True when the parsed reply value ``value`` reports anything but a routine outcome, judged from typed facts only:
    a receipt or stage entry (a dict with ``status`` and a step counter) that is none of (a) a clean ``completed``
    one with ``batch_status`` completed, (b) a stop the model asked for (``is_requested_checkpoint_receipt``: an
    ``observe`` stage or ``checkpoint_after_gripper``, told from a guard by ``checkpoint_reason_code``) or (c) a
    ``pending`` stage that never ran; a program result (``result_semantics``, or ``job_id`` with
    ``action_summary``) whose ``status`` is not ``succeeded``; trajectory feedback with ``available: false``; or a non-empty ``failure_feedback``,
    ``anomaly_reason``, ``blocking_risks``, ``pending_review`` or ``error`` anywhere."""
    from .feedback_delivery import is_clean_completed_receipt, is_requested_checkpoint_receipt, is_unexecuted_stage
    if isinstance(value, list):
        return any(reports_exception(item) for item in value)
    if not isinstance(value, dict):
        return False
    if any(value.get(key) for key in _EXCEPTION_KEYS):
        return True
    # A program result: its fixed sentence, or (the lean_v2 display leaves that sentence out) its job fields.
    if (("result_semantics" in value or ("job_id" in value and "action_summary" in value))
            and value.get("status") != "succeeded"):
        return True
    if isinstance(value.get("status"), str) and any(key in value for key in _RECEIPT_KEYS):
        clean = is_clean_completed_receipt(value) and value.get("batch_status", "completed") == "completed"
        if not (clean or is_requested_checkpoint_receipt(value) or is_unexecuted_stage(value)):
            return True
    if isinstance(value.get("trajectory_feedback"), dict) and value["trajectory_feedback"].get("available") is False:
        return True
    return any(reports_exception(item) for item in value.values())


def _without_static(node, key, dropped, plan):
    """``node`` without the fixed values of ``plan``; ``_MISSING`` when ``node`` itself is one (its key is then left
    out). A container that held nothing but fixed values is left out with them."""
    if isinstance(node, str):
        if (key, node) in plan.strings:
            dropped[key] = dropped.get(key, 0) + 1
            return _MISSING
        return node
    if isinstance(node, dict):
        if not node:
            return node
        result = {}
        for name, item in node.items():
            if type(item) is bool and (name, item) in plan.flags:
                dropped[name] = dropped.get(name, 0) + 1
                continue
            kept = _without_static(item, name, dropped, plan)
            if kept is not _MISSING:
                result[name] = kept
        return result if result else _MISSING
    if isinstance(node, list):
        if not node:
            return node
        if all(isinstance(item, str) for item in node):
            if (key, tuple(node)) in plan.lists:
                dropped[key] = dropped.get(key, 0) + 1
                return _MISSING
            kept = [item for item in node if (key, item) not in plan.strings]
            if len(kept) != len(node):
                dropped[key] = dropped.get(key, 0) + len(node) - len(kept)
            return kept if kept else _MISSING
        # Rows keep their position: an entry is never removed from a list of objects.
        return [item if (kept := _without_static(item, key, dropped, plan)) is _MISSING else kept for item in node]
    return node


def drop_static_once_report(items, *, routine_only=True, plan=None):
    """``drop_static_once`` plus what it did: ``(items, report)`` with ``report = {"exception": bool, "dropped":
    {key: count}}`` (``exception`` True: the reply reports a non-routine outcome and, with ``routine_only``, was
    returned unchanged). ``routine_only=False`` is for measuring which listed notes a reply carries; delivery never
    uses it."""
    plan = _FULL_PLAN if plan is None else plan
    values = [_parsed(item) for item in items]
    report = dict(exception=any(value is not _MISSING and reports_exception(value) for value in values), dropped={})
    if report["exception"] and routine_only:
        return items, report
    result = []
    for item, value in zip(items, values):
        if value is not _MISSING:
            kept = _without_static(value, None, report["dropped"], plan)
            text = canonical(type(value)() if kept is _MISSING else kept)
            if text != item["text"]:
                item = dict(item, text=text)
        result.append(item)
    return (result if report["dropped"] else items), report


def drop_static_once(items, plan=None):
    """Content items of one presented reply without the fixed explanatory values of ``plan``
    (``features.reply_static_once``).

    plan: the run's ``static_once_plan(tools, capabilities)``, i.e. what its catalogue states; None = the plan with
    every group (``STATIC_ONCE_STRINGS`` / ``STATIC_ONCE_LISTS`` / ``STATIC_ONCE_FLAGS``).

    Pure. Only canonical JSON text items are read; images and other text pass through. A value is left out only when
    both its key and its exact value are in the plan, so a diagnostic that merely shares a key name (a failure
    ``interpretation``, a different ``collision_scope``) stays. A reply for which ``reports_exception`` holds in any
    text item is returned unchanged. Nothing is renamed, reordered or rounded.
    """
    return drop_static_once_report(items, plan=plan)[0]


DETAIL_KEY = "detail"


def _detail_call(tool, arguments, direct_tools, at=None):
    """``({"tool", "arguments"}, at)`` for one read, or None when the actor has not loaded the tool (``direct_tools``;
    None = every tool). The entry names the tool itself."""
    if direct_tools is None or tool in direct_tools:
        return dict(tool=tool, arguments=arguments), at
    return None


def _artifact_call(reference, at, direct_tools):
    if direct_tools is not None and "program_run" not in direct_tools:
        return None
    selected = "".join(f"[{part!r}]" for part in at.split(".")) if at else ""
    source = ("import json\nfrom ni_rua_client import NiRuaClient\n"
              f"document = json.loads(NiRuaClient().artifact({reference!r}))\n"
              f"print(json.dumps(document{selected}))\n")
    return dict(tool="program_run", arguments=dict(source=source))


def _lookup(value):
    """``(tool, arguments)`` of a ``{"tool": name, **arguments}`` lookup as the producers write it, else None."""
    if isinstance(value, dict) and isinstance(value.get("tool"), str) and "arguments" not in value:
        return value["tool"], {key: item for key, item in value.items() if key != "tool"}
    return None


def _unified(node, action_id, direct_tools, counts):
    if isinstance(node, list):
        return [_unified(item, action_id, direct_tools, counts) for item in node]
    if not isinstance(node, dict):
        return node
    if isinstance(node.get("action_id"), str):
        action_id = node["action_id"]
    result = {key: _unified(item, action_id, direct_tools, counts) for key, item in node.items()}
    if DETAIL_KEY in result:
        return result  # the place already has a field of that name; its read entries stay as they are

    def replace(old, call, *also):
        if call is None:
            return
        for key in (old, *also):
            result.pop(key, None)
        result[DETAIL_KEY] = call
        counts[old] = counts.get(old, 0) + 1

    if node.get("omitted") == "by_reference" and "read" in node:
        read, at = node["read"], node.get("at")
        if isinstance(read, dict) and read.get("program_run") == "NiRuaClient.artifact" and isinstance(read.get("reference"), str):
            replace("read", _artifact_call(read["reference"], at, direct_tools), "at")
        elif _lookup(read):
            made = _detail_call(*_lookup(read), direct_tools, at)
            if made is not None:
                replace("read", made[0], *(() if made[1] else ("at",)))
        return result
    for key, at in (("full_result", "receipt.motion_safety_checks" if "pass_count" in node else None), ("read_full", None)):
        if _lookup(node.get(key)):
            made = _detail_call(*_lookup(node[key]), direct_tools, at)
            if made is not None:
                replace(key, made[0])
            return result
    if isinstance(node.get("detail_access"), str) and isinstance(node.get("trajectory_feedback_ref"), str):
        # ``detail_access`` names the artifact that holds every exact row; the entry reads that artifact, not an
        # inspect call (which returns three rows unless steps are named).
        call = _artifact_call(node["trajectory_feedback_ref"], None, direct_tools)
        if call is None:
            made = _detail_call("robot_action", dict(mode="inspect", trajectory_ref=node["trajectory_feedback_ref"],
                                                     path_kind="measured", start_native_step=0, max_samples=INSPECT_STEP_LIMIT),
                                direct_tools)
            call = None if made is None else made[0]
        replace("detail_access", call)
        return result
    following = node.get("next_action")
    if (isinstance(following, dict) and following.get("mode") == "inspect"
            and set(following) <= {"mode", "plan_id", "action_id", "trajectory_ref"}):
        made = _detail_call("robot_action", dict(following), direct_tools)
        if made is not None:
            replace("next_action", made[0])
    return result


def unify_detail_entries_report(items, direct_tools=None):
    """``unify_detail_entries`` plus ``{replaced field name: count}``."""
    counts, result = {}, []
    for item in items:
        value = _parsed(item)
        if value is not _MISSING:
            text = canonical(_unified(value, None, direct_tools, counts))
            if text != item["text"]:
                item = dict(item, text=text)
        result.append(item)
    return (result if counts else items), counts


def unify_detail_entries(items, direct_tools=None):
    """Rewrite the ways a reply names "read the rest here" into one field (``features.reply_static_once``):

        "detail": {"tool": "<tool name>", "arguments": {<the complete arguments of that call>}}

    placed where the old entry was. What the call returns is unchanged. Replaced entries:

    - ``full_result`` / ``read_full``: ``{"tool": name, **arguments}`` lookups (motion-safety summary, history
      entries, ``metadata_delivery``);
    - ``detail_access``: the sentence in ``trajectory_feedback`` (next to ``trajectory_feedback_ref``, which stays);
      the call reads that artifact through ``program_run``, or, where ``program_run`` is not loaded, is the
      paged ``robot_action`` ``mode="inspect"`` of that reference for the measured path;
    - ``next_action`` when it is nothing but the ``mode="inspect"`` read of a plan or action (a continuation
      ``next_action`` of a preview or of a paused batch is a command, not a read, and stays);
    - ``read`` (+ ``at``) of a ``{"omitted": "by_reference"}`` stand-in written by the reply budget.

    ``direct_tools``: names of the tools loaded in the actor's catalogue (not ``deferLoading``); None = every tool.
    An entry is only ever pointed at a loaded tool: a lookup of a tool that is deferred (loaded on demand by the host)
    is left as it is, and an artifact read becomes a ``program_run`` call printing that part of the artifact.
    ``robot_action_status`` and ``program_status`` are loaded up front, so their entries name them directly. A place
    that already has a ``detail`` field is left as it is. Pure; only canonical JSON text items are read.
    """
    return unify_detail_entries_report(items, direct_tools)[0]


def catalogue_with_reply_notes(tools, notes):
    """Tool definitions with ``notes`` (strings, in order) in the ``robot_action`` description, before its generated
    "Input field guide" (appended when there is none). Other tools, and a catalogue without ``robot_action``, are
    returned unchanged, as is every tool when ``notes`` is empty."""
    text = "".join(notes)
    if not text:
        return tools
    result = []
    for tool in tools:
        if tool.get("name") == STATIC_ONCE_NOTE_TOOL and isinstance(tool.get("description"), str):
            description, marker = tool["description"], "\nInput field guide"
            cut = description.find(marker)
            cut = len(description) if cut < 0 else cut
            tool = dict(tool, description=description[:cut] + text + description[cut:])
        result.append(tool)
    return result


def reply_catalogue_notes(features, tools=None, capabilities=None):
    """The notes the run's reply switches add to the catalogue, in a fixed order: reply layout
    (``action_reply_profile``), fixed notes (``reply_static_once``), encoding (``reply_compact_encoding``).

    tools / capabilities: the actor catalogue before any note and the robot backend's capabilities; they decide which
    sentences apply to this robot (``static_once_plan``; the layout note names the object table only when the
    catalogue does). None = every sentence."""
    from .reply_budget import summary_first_catalogue_note
    from .reply_encoding import COMPACT_ENCODING_CATALOGUE_NOTE
    notes = []
    if getattr(features, "action_reply_profile", "full") == "summary_first":
        action = next((tool.get("description") or "" for tool in tools or () if tool.get("name") == STATIC_ONCE_NOTE_TOOL), "")
        notes.append(summary_first_catalogue_note(
            object_table=tools is None or any("object_table" in (tool.get("description") or "") for tool in tools),
            # grasp_contact_feedback is produced for grasp-pose stages (contact_region), as in static_once_plan
            stage_detail=tools is None or "contact_region" in action))
    if getattr(features, "reply_static_once", False):
        notes.append(static_once_plan(tools, capabilities, features).note)
    if getattr(features, "reply_compact_encoding", False):
        notes.append(COMPACT_ENCODING_CATALOGUE_NOTE)
    if getattr(features, "reply_same_as", False):
        notes.append(SAME_AS_CATALOGUE_NOTE)
    return notes


def static_once_catalogue(tools):
    """``catalogue_with_reply_notes`` with ``STATIC_ONCE_CATALOGUE_NOTE`` alone."""
    return catalogue_with_reply_notes(tools, [STATIC_ONCE_CATALOGUE_NOTE])
