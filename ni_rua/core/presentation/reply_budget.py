"""Order one action reply by decision value and fit its text into the host's tool-output budget.

Used for every agent backend: the layout below is applied whenever ``features.budgeted_action_reply`` is on; blocks
are replaced by references only against a host limit that is known (``AgentSession.tool_output_text_budget_bytes``:
Codex and Claude Code state one, OpenCode and external hosts none; docs/development/tool-output-contract.md).

Why: an agent host may cap the text of one tool output. Codex (CLI 0.160.0, measured by forked probes against the
outbound ``/responses`` request) applies this rule to every tool output it records:

- budget = ``truncation_policy.limit`` tokens of the session model in the model catalogue x 1.2, i.e. 10,000 x 1.2 =
  12,000 tokens for every pinned model (``tool_output_token_limit`` in ``config.toml`` sets the same limit: 20,000
  gave 24,000 tokens on a live output);
- one text item costs ``ceil(UTF-8 bytes / 4)`` tokens (48,000 bytes of CJK text = 16,000 characters used the whole
  budget); image items cost nothing and are always kept;
- items are taken in order. The first text item that exceeds what is left is cut in its middle (head and tail kept,
  ``…N tokens truncated…`` inserted; one token over is enough) and uses up the budget; every later text item is
  dropped and one ``[omitted N text items ...]`` item is appended.

An ``robot_action`` execution reply is [result JSON, overlay images, observation JSON, camera images, object table
JSON]; its result JSON alone often exceeds 48,000 bytes, so the host cut it and dropped the post-action robot state
and the object table / object delta. ``fit_reply`` makes the Harness decide instead: the reply is laid out as

1. execution result: status, stage ids, terminal residuals, effective tolerances, guard/contact summaries, budgets
   (the result JSON without per-stage detail and trajectory feedback; stage entries that did not complete cleanly and
   motion-safety rows that did not pass stay here);
2. scene change: ``object_state_delta`` + ``object_table`` (+ its overlay image, when one follows it);
3. post-action observation: robot state and camera metadata, followed by its camera images (wait samples keep their
   place before it); never reduced, so ``metadata_delivery`` references stay valid;
4. per-stage detail of cleanly completed stages: ``{"action_id", "stages"}``;
5. trajectory feedback: ``{"action_id", "trajectory_feedback"}``, followed by the trajectory overlay images;

and, only while the text is over budget, the least valuable blocks are replaced by
``{"omitted": "by_reference", "read": {...}}`` naming the existing entry that returns them. Nothing is rounded,
renamed or deleted, and a block without such an entry is never replaced.
"""
import json

from ..contracts import INSPECT_STEP_LIMIT, canonical
from .feedback_delivery import is_clean_completed_receipt, is_requested_checkpoint_receipt, is_unexecuted_stage

BYTES_PER_TOKEN = 4  # the host's estimate: one token per started 4 UTF-8 bytes of a text item
_TEXT_TYPES = ("inputText", "input_text")
_IMAGE_TYPES = ("inputImage", "input_image")

# Replacement order, least valuable first. A cleanly completed action keeps its per-stage residuals longest; after a
# failure, pause or slip the measured path and the 3D process summary outrank the detail of the stages that did
# complete (the entries of the stages that did not are part of the execution result and are never replaced).
_ROUTINE_ORDER = ("motion_safety_pass_rows", "planned_samples", "measured_samples", "compact_3d_trajectory", "stages")
_EXCEPTION_ORDER = ("motion_safety_pass_rows", "planned_samples", "stages", "measured_samples", "compact_3d_trajectory")

# --- features.action_reply_profile = "summary_first" -----------------------------------------------------------------
#
# The reply has one fixed shape instead of "whatever fits": the blocks below are replaced by a reference in every
# reply, whether or not a host budget exists. If a host budget exists and the fixed shape still exceeds it, the
# replacement continues with the remaining blocks in the order above.
#
# | Reply                                   | Inline                                                   | By reference                                   |
# |-----------------------------------------|----------------------------------------------------------|------------------------------------------------|
# | routine (``_routine``: clean completed  | execution result; every stage entry (a cleanly completed | measured samples, planned samples, passing     |
# | receipt, batch completed, no            | one without ``STAGE_DETAIL_BY_REFERENCE``); scene        | motion-safety rows, the                        |
# | failure_feedback, no anomaly_reason,    | change; post-action observation and images; trajectory   | ``STAGE_DETAIL_BY_REFERENCE`` fields of        |
# | trajectory feedback available)          | feedback with ``compact_3d_trajectory``                  | cleanly completed stages                       |
# | routine, stop the model asked for       | as above; the stages that have not run yet (``pending``) | as above                                       |
# | (``_requested_checkpoint``: an          | are part of the execution result                         |                                                |
# | ``observe`` stage or                    |                                                          |                                                |
# | ``checkpoint_after_gripper``, nothing   |                                                          |                                                |
# | else reported)                          |                                                          |                                                |
# | exception: receipt failed or unknown    | the same, plus the measured samples of the affected step | planned samples, passing motion-safety rows,   |
# | (tracking_error ...), stop by a guard   | window: every sample whose native_step lies in the steps | the same stage fields of cleanly completed     |
# | (grip guard, carry slip guard), batch   | of the affected stages (``_affected_steps``) and the one | stages, measured samples outside the window    |
# | failed, failure_feedback,               | sample before and after it. Stages that did not complete |                                                |
# | anomaly_reason (slip), or trajectory    | cleanly and safety rows that did not pass are part of    |                                                |
# | feedback unavailable                    | the execution result in every profile.                   |                                                |
# | exception without an identifiable       | the same, plus all measured samples                      | planned samples, passing motion-safety rows,   |
# | window (no compact_3d_trajectory, or no |                                                          | the same stage fields of cleanly completed     |
# | step for any of the items below)        |                                                          | stages                                         |
#
# The window spans: the ``compact_3d_trajectory.stages`` rows of the affected stages (``receipt.failed_stage_id``,
# ``receipt.checkpoint_stage_id(s)`` and every stage entry that is not a clean completed one and is not ``pending``;
# a pending stage never ran), the ``slip_onset_step`` of a detected slip, and, when the receipt status is not
# ``completed``, the step execution stopped at (``receipt.completed_native_steps``).
#
# Which stop the model asked for is read from ``receipt.checkpoint_reason_code`` (the table next to
# ``feedback_delivery.MODEL_REQUESTED_CHECKPOINT_CODES``): ``observation_checkpoint`` and ``jaw_checkpoint`` are
# requested, every guard code and a missing code are exceptions. The ``full`` profile's replacement order is not
# affected by this: it still treats every stop as an exception.
ACTION_REPLY_PROFILES = ("full", "summary_first")
# Fields of a cleanly completed stage that summary_first gives by reference. Everything else of a stage entry stays
# inline on every robot: outcome, terminal residuals, the tolerances that were enforced, resolved poses, grip and guard
# facts, and any field this list does not know. ``grasp_contact_feedback`` (derived contact geometry of the reached
# pose, about 60 % of a clean stage entry) is the one block left out.
STAGE_DETAIL_BY_REFERENCE = ("grasp_contact_feedback",)
_SUMMARY_ROUTINE = ("motion_safety_pass_rows", "planned_samples", "measured_samples", "stage_detail")
_SUMMARY_EXCEPTION = ("motion_safety_pass_rows", "planned_samples", "stage_detail", "measured_samples_outside_window")


def summary_first_catalogue_note(object_table=True, stage_detail=True):
    """The sentence(s) the ``robot_action`` description carries under ``summary_first``. ``object_table``: name the
    object table and its delta among the inline parts; ``stage_detail``: name the stage fields given by reference
    (both False when the run's catalogue does not have them)."""
    scene = "object_state_delta and object_table, " if object_table else ""
    fields = " and ".join(STAGE_DETAIL_BY_REFERENCE)
    return (
        " Execution reply layout: inline are the execution result (receipt with status, reason, terminal residual, "
        "guard and contact summaries and budgets; every motion-safety row that did not pass), every stage entry"
        + (" (a cleanly completed stage without its " + fields + ")" if stage_detail else "") + ", " + scene
        + "the post-action observation with its images, and trajectory_feedback with compact_3d_trajectory. Given by "
        "reference, with the call that returns them at their place: measured and planned joint samples"
        + (", passing motion-safety rows and " + fields + " of cleanly completed stages" if stage_detail
           else " and passing motion-safety rows")
        + ". A stop at a checkpoint you requested (observe stage, checkpoint_after_gripper) has this same layout. "
        "After a failure, a guard pause, a slip or an unknown outcome the measured samples around the affected stage "
        "are inline as well.")


SUMMARY_FIRST_CATALOGUE_NOTE = summary_first_catalogue_note()


def text_cost_tokens(text):
    """Host cost of one text item: ``ceil(UTF-8 bytes / 4)`` tokens."""
    return -(-len(text.encode()) // BYTES_PER_TOKEN)


def reply_cost_tokens(items):
    """Host cost of a reply: the sum over its text items; images and unknown items cost nothing."""
    return sum(text_cost_tokens(item["text"]) for item in items
               if isinstance(item, dict) and item.get("type") in _TEXT_TYPES and isinstance(item.get("text"), str))


def _value(item):
    if not isinstance(item, dict) or item.get("type") not in _TEXT_TYPES:
        return None
    try:
        value = json.loads(item.get("text", ""))
    except (TypeError, ValueError):
        return None
    return value if isinstance(value, dict) else None


def _kind(value):
    if value is None:
        return "other"
    if isinstance(value.get("output_contract_error"), dict):
        return "contract_error"
    if isinstance(value.get("receipt"), dict):
        return "result"
    if value.get("temporal_sample") is True or (isinstance(value.get("bundle"), dict) and "robot_state" in value):
        return "observation"
    if any(key in value for key in ("object_table", "object_state_delta", "scene3d")):
        return "scene"
    return "other"


def _samples_read(action_id, path_kind, samples):
    """The ``robot_action`` inspect call that returns exactly ``samples`` (the rows a reply leaves out): their
    ``native_step`` values are named, because without a selector inspect returns only the first, middle and last row.
    More than ``INSPECT_STEP_LIMIT`` rows are read as pages from the first one (``next_native_step`` continues)."""
    read = dict(tool="robot_action", mode="inspect", action_id=action_id, path_kind=path_kind)
    steps = [sample.get("native_step") for sample in samples or () if isinstance(sample, dict)]
    if not steps or any(type(step) is not int for step in steps) or len(set(steps)) != len(steps):
        return read
    if len(steps) <= INSPECT_STEP_LIMIT:
        return dict(read, native_steps=steps)
    return dict(read, start_native_step=min(steps), max_samples=INSPECT_STEP_LIMIT)


def _note(read, at=None, **facts):
    """The one-line stand-in of a replaced block: ``read`` is the call (or artifact read) that returns it and ``at``
    the path of the block inside what that call returns."""
    note = dict(omitted="by_reference", **facts, read=read)
    if at:
        note["at"] = at
    return note


def _routine(result):
    """A completed action without any exception flag (same receipt test as the image policy)."""
    receipt, trajectory = result["receipt"], result.get("trajectory_feedback")
    return (is_clean_completed_receipt(receipt) and not receipt.get("failure_feedback")
            and receipt.get("batch_status", "completed") == "completed"
            and not (isinstance(trajectory, dict)
                     and (trajectory.get("anomaly_reason") or trajectory.get("available") is False)))


def _same_stage(first, second):
    """Stage ids name the same stage when equal or when one is a sub-stage of the other (``grasp`` / ``grasp/motion``)."""
    return (isinstance(first, str) and isinstance(second, str)
            and (first == second or first.startswith(second + "/") or second.startswith(first + "/")))


def _affected_steps(result):
    """``(first, last)`` native step of the stages an exception reply is about, or None when they cannot be placed
    on the step axis (see the table above)."""
    receipt, trajectory = result["receipt"], result.get("trajectory_feedback")
    compact = trajectory.get("compact_3d_trajectory") if isinstance(trajectory, dict) else None
    if not isinstance(compact, dict):
        return None
    ids = [receipt.get("failed_stage_id"), receipt.get("checkpoint_stage_id"), *(receipt.get("checkpoint_stage_ids") or [])]
    if isinstance(result.get("stages"), dict):
        ids += [name for name, entry in result["stages"].items()
                if not is_clean_completed_receipt(entry) and not (isinstance(entry, dict) and entry.get("status") == "pending")]
    steps = []
    for row in compact.get("stages") or []:
        if isinstance(row, dict) and any(_same_stage(row.get("stage_id"), name) for name in ids):
            steps += [row.get(key) for key in ("start_step", "end_step")]
    process = compact.get("process_trajectory")
    for arm in (process.get("co_motion") or {}).values() if isinstance(process, dict) else ():
        if isinstance(arm, dict):
            steps.append(arm.get("slip_onset_step"))
    if receipt.get("status") != "completed":
        # Execution stopped: the step it stopped at belongs to the window (a checkpoint ``observe`` stage has no
        # steps of its own).
        steps.append(receipt.get("completed_native_steps"))
    steps = [step for step in steps if type(step) is int]
    return (min(steps), max(steps)) if steps else None


def _window_samples(samples, window):
    """Indices of ``samples`` inside ``window`` plus the nearest one before and after it, or None when a sample has
    no integer ``native_step``."""
    steps = [sample.get("native_step") if isinstance(sample, dict) else None for sample in samples]
    if any(type(step) is not int for step in steps):
        return None
    keep = {index for index, step in enumerate(steps) if window[0] <= step <= window[1]}
    before = [index for index, step in enumerate(steps) if step < window[0]]
    after = [index for index, step in enumerate(steps) if step > window[1]]
    if before:
        keep.add(max(before, key=lambda index: steps[index]))
    if after:
        keep.add(min(after, key=lambda index: steps[index]))
    return sorted(keep)


def _requested_checkpoint(result):
    """The batch stopped at a checkpoint the model asked for and reports nothing else (summary_first: routine)."""
    receipt, trajectory, stages = result["receipt"], result.get("trajectory_feedback"), result.get("stages")
    return (is_requested_checkpoint_receipt(receipt) and not receipt.get("failure_feedback")
            and not result.get("failure_feedback")
            and not (isinstance(trajectory, dict)
                     and (trajectory.get("anomaly_reason") or trajectory.get("available") is False))
            and all(is_clean_completed_receipt(entry) or is_unexecuted_stage(entry)
                    for entry in (stages.values() if isinstance(stages, dict) else ())))


def _blocks(result):
    """``{name: apply(head, stages, trajectory)}`` for every block of this result that an existing entry returns."""
    action_id = result.get("action_id") or result["receipt"].get("action_id")
    trajectory = result.get("trajectory_feedback")
    blocks = {}
    if not isinstance(action_id, str) or not action_id:
        return blocks
    status = dict(tool="robot_action_status", action_id=action_id, detail="full")
    rows = result["receipt"].get("motion_safety_checks")
    if isinstance(rows, list) and any(isinstance(row, dict) and row.get("status") == "pass" for row in rows):
        def safety(head, stages, trajectory):
            kept = [row for row in rows if not (isinstance(row, dict) and row.get("status") == "pass")]
            receipt = dict(head["receipt"], motion_safety_checks_by_reference=_note(
                status, "receipt.motion_safety_checks", pass_rows=len(rows) - len(kept)))
            if kept:
                receipt["motion_safety_checks"] = kept
            else:
                receipt.pop("motion_safety_checks")
            head["receipt"] = receipt
        blocks["motion_safety_pass_rows"] = safety
    stages = result.get("stages")
    if isinstance(stages, dict) and any(is_clean_completed_receipt(entry) for entry in stages.values()):
        def stage_detail(head, stages, trajectory):
            ids = list(stages.pop("stages"))
            stages["stages_by_reference"] = _note(status, "receipt.stage_results", stage_ids=ids)
        blocks["stages"] = stage_detail
        clean = {name: entry for name, entry in stages.items() if is_clean_completed_receipt(entry)}
        detailed = [name for name, entry in clean.items() if set(entry) & set(STAGE_DETAIL_BY_REFERENCE)]
        if detailed:
            def stage_fields(head, stages, trajectory):
                stages["stages"] = {name: {key: item for key, item in entry.items() if key not in STAGE_DETAIL_BY_REFERENCE}
                                    for name, entry in stages["stages"].items()}
                stages["stages_by_reference"] = _note(status, "receipt.stage_results", stage_ids=detailed,
                                                      fields=list(STAGE_DETAIL_BY_REFERENCE))
            blocks["stage_detail"] = stage_fields
    if isinstance(trajectory, dict):
        for path_kind in ("planned", "measured"):
            path = trajectory.get(path_kind)
            if isinstance(path, dict) and isinstance(path.get("samples"), list) and path["samples"]:
                def samples(head, stages, trajectory, path_kind=path_kind):
                    kept = {key: value for key, value in trajectory["trajectory_feedback"][path_kind].items()
                            if key != "samples_by_reference"}
                    trajectory["trajectory_feedback"][path_kind] = dict(
                        kept, samples=_note(_samples_read(action_id, path_kind, kept.get("samples"))))
                blocks[path_kind + "_samples"] = samples
        window = None if _routine(result) else _affected_steps(result)
        if window is not None and "measured_samples" in blocks:
            keep = _window_samples(trajectory["measured"]["samples"], window)
            if keep is not None and len(keep) < len(trajectory["measured"]["samples"]):
                def outside(head, stages, trajectory):
                    measured = trajectory["trajectory_feedback"]["measured"]
                    trajectory["trajectory_feedback"]["measured"] = dict(
                        measured, samples=[measured["samples"][index] for index in keep],
                        samples_by_reference=_note(
                            _samples_read(action_id, "measured",
                                          [sample for index, sample in enumerate(measured["samples"]) if index not in keep]),
                            samples_outside_native_steps=list(window), count=len(measured["samples"]) - len(keep)))
                blocks["measured_samples_outside_window"] = outside
        reference = trajectory.get("trajectory_feedback_ref")
        if isinstance(trajectory.get("compact_3d_trajectory"), dict) and isinstance(reference, str) and reference:
            def compact(head, stages, trajectory):
                trajectory["trajectory_feedback"]["compact_3d_trajectory"] = _note(
                    dict(program_run="NiRuaClient.artifact", reference=reference), "summary.compact_3d_trajectory")
            blocks["compact_3d_trajectory"] = compact
    return blocks


def _render(result, text_type, replaced):
    """(head item, stage item | None, trajectory item | None) with the named blocks replaced."""
    head = {key: value for key, value in result.items() if key not in ("stages", "trajectory_feedback")}
    stages = trajectory = None
    action_id = result.get("action_id") or result["receipt"].get("action_id")
    if isinstance(result.get("stages"), dict):
        pinned = {key: value for key, value in result["stages"].items() if not is_clean_completed_receipt(value)}
        routine = {key: value for key, value in result["stages"].items() if key not in pinned}
        if pinned:
            head["stages"] = pinned
        if routine:
            stages = dict(action_id=action_id, stages=routine)
    elif "stages" in result:
        head["stages"] = result["stages"]
    if isinstance(result.get("trajectory_feedback"), dict):
        trajectory = dict(action_id=action_id, trajectory_feedback=dict(result["trajectory_feedback"]))
    elif "trajectory_feedback" in result:
        head["trajectory_feedback"] = result["trajectory_feedback"]
    blocks = _blocks(result)
    for name in replaced:
        blocks[name](head, stages, trajectory)
    return tuple(None if value is None else {"type": text_type, "text": canonical(value)}
                 for value in (head, stages, trajectory))


def fit_reply_report(items, budget_bytes, compact=None, profile="full", layout=False):
    """``fit_reply`` plus what it did: ``(items, report)``.

    ``layout``: with ``budget_bytes=None`` and the ``full`` profile, still lay the reply out by decision value
    (nothing is replaced: a host without a known limit). Default False: such a reply is returned unchanged.

    ``profile``: ``"full"`` replaces blocks only while the text is over budget; ``"summary_first"`` always replaces
    the blocks of the fixed shape (table above ``ACTION_REPLY_PROFILES``; ``report["fixed"]`` names them), with or
    without a budget, and continues with the usual order only if that shape is still over budget.

    ``compact``: optional ``items -> items`` shortening of text items (``reply_compaction.compact_content_items``
    with the run's switches). Pass the reply as presented, not an already compacted one: the layout is made from the
    presented values and ``compact`` is applied to the text items of every layout that is weighed and of the one
    returned, so the budget is spent on the shortened text and each ``same_as`` pointer is computed inside the text
    item it ends up in. A value whose first copy went to another item or was replaced by a reference is therefore
    printed again in full instead of pointing at text that is not there.

    ``report``: ``applied`` (False: not an action reply or no budget, items returned as given), ``budget_tokens``,
    ``cost_tokens_before`` / ``cost_tokens_after`` (host cost of all text items), ``fits`` (after <= budget; False
    only when the blocks that have a reading entry do not free enough, in which case the host cuts the reply at its
    least valuable end) and ``by_reference`` (names of the replaced blocks, in replacement order).
    """
    items = list(items)
    report = dict(applied=False, by_reference=[])
    if profile not in ACTION_REPLY_PROFILES:
        raise ValueError("unknown_action_reply_profile")
    if budget_bytes is None and profile == "full" and not layout:
        return items, report
    if budget_bytes is not None and (type(budget_bytes) is not int or budget_bytes < 0):
        raise ValueError("reply_budget_bytes_must_be_non_negative_int")
    groups = []  # [kind, value, [items]]: a text item owns the images that follow it
    for item in items:
        if isinstance(item, dict) and item.get("type") in _IMAGE_TYPES and groups:
            groups[-1][2].append(item)
        else:
            value = _value(item)
            groups.append([_kind(value), value, [item]])
    index = next((i for i, group in enumerate(groups) if group[0] == "result"), None)
    if index is None:
        return items, report
    _, result, (result_item, *result_images) = groups[index]
    rest = [group for i, group in enumerate(groups) if i != index]
    for group in rest:
        if group[0] == "result":
            group[0] = "other"
    budget = None if budget_bytes is None else budget_bytes // BYTES_PER_TOKEN
    shorten = compact if compact is not None else (lambda content: content)
    for group in rest:
        group[2] = list(shorten(group[2]))
    rest_cost = sum(reply_cost_tokens(group[2]) for group in rest)

    def render(replaced):
        return tuple(part and shorten([part])[0] for part in _render(result, result_item["type"], replaced))

    def cost(replaced):
        return rest_cost + reply_cost_tokens([part for part in render(replaced) if part])

    available = _blocks(result)
    routine = _routine(result) or (profile == "summary_first" and _requested_checkpoint(result))
    fixed = []
    if profile == "summary_first":
        fixed = [name for name in (_SUMMARY_ROUTINE if routine else _SUMMARY_EXCEPTION) if name in available]
    order = [name for name in (_ROUTINE_ORDER if routine else _EXCEPTION_ORDER) if name in available and name not in fixed]
    replaced = list(fixed)
    if budget is not None:
        for name in order:
            if cost(replaced) <= budget:
                break
            replaced.append(name)
        # Give back, most valuable first, whatever fits in the room the last replacement left.
        for name in reversed([name for name in replaced if name not in fixed]):
            trial = [other for other in replaced if other != name]
            if cost(trial) <= budget:
                replaced = trial
    head, stages, trajectory = render(replaced)
    # 锁死后置契约违例警告始终位于首项，避免被动作回执重排挤到末尾
    out = [item for group in rest if group[0] == "contract_error" for item in group[2]]
    out += [head] + ([] if trajectory else result_images)
    for kind in ("scene", "observation", "other"):
        for group in rest:
            if group[0] == kind:
                out += group[2]
    if stages:
        out.append(stages)
    if trajectory:
        out += [trajectory] + result_images
    after = reply_cost_tokens(out)
    report.update(applied=True, budget_tokens=budget, cost_tokens_before=reply_cost_tokens(items),
                  cost_tokens_after=after, fits=budget is None or after <= budget, by_reference=replaced)
    if profile != "full":
        report.update(profile=profile, routine=routine, fixed=fixed)
    return out, report


def fit_reply(items, budget_bytes, compact=None, profile="full", layout=False):
    """Return the content items of one action reply ordered by decision value and fitted to a host text budget.

    ``items``: the reply's content items in wire form, ``{"type": "inputText", "text": <str>}`` and
    ``{"type": "inputImage", "imageUrl": <data URL>}`` (the Codex rollout spelling ``input_text`` / ``input_image``
    is accepted as well; new text items use the spelling of the result item). Pure: the input list and its items
    are not modified, and the same input always gives the same output.

    ``budget_bytes``: the host's text budget for one tool output in UTF-8 bytes (Codex: 48,000; a text item is
    charged per started 4 bytes); ``None`` = the host has no such budget, the items are returned unchanged.

    A reply without an execution result (no JSON text item with a ``receipt`` object) is returned unchanged. Images
    are never dropped; each keeps following the text item that describes it. See the module docstring for the layout
    and the replacement rule, and ``fit_reply_report`` for what was replaced and for ``compact``.
    """
    return fit_reply_report(items, budget_bytes, compact, profile, layout)[0]
