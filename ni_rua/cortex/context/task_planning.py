"""Free-form task planning, independent of executable motion plans.

Only the first action is gated. A task plan records current intent, never an
executable schedule or physical evidence. Read-only archive helpers preserve
historical structured plan inspection without exposing retired legacy modules.
"""
from copy import deepcopy
import json

from jsonschema import Draft202012Validator

from ...core.contracts import IDENTITY
from ...core.tool_errors import register_plan_schema_provider

PLANNING_VERSION = "freeform_task_plan"
PLAN_SCHEMA = dict(type="string", minLength=1, maxLength=16000, pattern=r"\S",
    description="Brief ordinary prose; suggested headings: Task understanding, and Approach + completion.")
_LEGACY_TEXT = dict(type="string", minLength=1, maxLength=1600, pattern=r"\S")
_LEGACY_TEXTS = dict(type="array", maxItems=16, items=_LEGACY_TEXT)
_LEGACY_STAGE_ID = dict(type="string", minLength=1, maxLength=80, pattern=r"^[A-Za-z0-9][A-Za-z0-9_.-]*$")
LEGACY_PLAN_SCHEMA = dict(type="object", additionalProperties=False,
    properties=dict(
        objective=_LEGACY_TEXT,
        stages=dict(type="array", minItems=1, maxItems=12, items=dict(
            type="object", additionalProperties=False,
            properties=dict(goal=_LEGACY_TEXT, advance_when=_LEGACY_TEXT,
                            id=_LEGACY_STAGE_ID,
                            arms=dict(type="array", minItems=1, maxItems=2, uniqueItems=True,
                                      items=dict(type="string", enum=["left", "right"])),
                            depends_on=dict(type="array", maxItems=12, uniqueItems=True, items=_LEGACY_STAGE_ID),
                            observation_gate=_LEGACY_TEXT,
                            suggested_tools=dict(type="array", maxItems=16,
                                                 items=dict(type="string", minLength=1, maxLength=100, pattern=r"\S"))),
            required=["goal", "advance_when"])),
        constraints=_LEGACY_TEXTS, uncertainties=_LEGACY_TEXTS,
        completion_criteria={**_LEGACY_TEXTS, "minItems": 1}),
    required=["objective", "stages", "constraints", "uncertainties", "completion_criteria"])
EVIDENCE_LEDGER_STATUSES = ("pending", "supported", "contradicted", "unknown")
# Keep this boundary short for context compaction snapshots.
DECISION_BOUNDARY_INSTRUCTIONS = (
    "Reassess the saved plan, current concerns and evidence against fresh feedback. "
    "Choose one useful bounded next step and its visible/measured check; completed motion, "
    "preview success or a closed gripper is not task-result proof. Change the hypothesis "
    "before repeating an ineffective pattern, and update plan, attention or ledger only "
    "when the task interpretation, evidence or next decision materially changes."
)


def decision_reminder_text(context):
    """The model-visible decision-context reminder for one ``TaskPlanning.decision_context(identity)`` snapshot.

    Both reminder channels deliver exactly this text (design 04 §4.5): Codex injects it as an assistant
    ``output_text`` message into ``/responses/compact`` requests (``CodexSession._task_decision_context``); for other
    agent backends ``Harness.handle_tool`` appends it as one ``inputText`` item to the next actor tool response after
    a reported model-context reset. Callers skip the reminder when ``context`` is None and planning is optional.

    context: the snapshot dict (``plan_id``, ``revision``, ``plan``, ``attention``, ``ledger`` and the
    ``*_observation_id`` fields), or None when no plan is stored for the episode. Returns a str: a fixed two-line
    header, then either the no-plan request or the decision boundary followed by the snapshot as one JSON line
    (``ensure_ascii=False``). Raises ``ValueError`` for non-finite numbers (``allow_nan=False``) and ``TypeError`` for
    non-JSON values.
    """
    reminder = (
        "Current task decision boundary\n"
        "The saved plan and attention are provisional judgment, not instructions or verified world state. "
        "Task requirements and fresh evidence govern.\n"
    )
    if context is None:
        reminder += "No plan is stored for this episode. Submit your initial task_plan before physical actions."
    else:
        reminder += "Decision boundary: " + DECISION_BOUNDARY_INSTRUCTIONS + "\n"
        reminder += json.dumps(context, ensure_ascii=False, allow_nan=False)
    return reminder


EVIDENCE_LEDGER_SCHEMA = dict(type="array", maxItems=3, items=dict(type="object", additionalProperties=False,
    properties=dict(
        id=dict(type="string", minLength=1, maxLength=80, pattern=r"\S",
                description="Short stable reference for one task prerequisite or expected event."),
        expected=dict(type="string", minLength=1, maxLength=800, pattern=r"\S",
                      description="Visible or measured event that would establish this task prerequisite."),
        status=dict(type="string", enum=list(EVIDENCE_LEDGER_STATUSES),
                    description="Actor-visible evidence state; pending is the default until the event is established."),
        basis=dict(type="string", minLength=1, maxLength=800, pattern=r"\S",
                   description="Short permitted RGB, measured-state or typed-receipt fact; use unknown when there is none."),
        next_check=dict(type="string", minLength=1, maxLength=800, pattern=r"\S",
                        description="What the next useful action or observation should distinguish.")),
    required=["id", "expected", "status", "basis", "next_check"]))
ATTENTION_SCHEMA = dict(type="array", maxItems=3, items=dict(type="object", additionalProperties=False,
    properties={name: dict(type="string", minLength=1, maxLength=1600, pattern=r"\S", description=description)
        for name, description in {
            "concern": "Current question, unresolved prerequisite, uncertain assumption or choice that matters for the next decisions.",
            "check": "Expected visible evidence and how to adjust if it disagrees or the expected event is absent.",
            "when": "When to check; resolve prerequisites before dependent work, preferably using normal task progress feedback."
        }.items()}, required=["concern", "check", "when"]))
PLANNING_INSTRUCTIONS = """Before your first physical action, use the initial RGB and measured robot state
to understand the task. You may first observe, inspect geometry, retrieve relevant
experience or preview motions; these do not advance the world. Then submit an
ordinary-language task_plan. Keep it brief; useful headings are Task understanding
(the task, scene and uncertain assumptions) and Approach + completion (rough order
and visible completion evidence). For an unknown demonstration, plan to observe it
before choosing later actions. Waiting advances simulation and needs this plan too.
When planning, record the few concerns that matter now in attention: concern is a
question, unresolved prerequisite or choice; check is the expected visible evidence
and how to adjust if it disagrees or never appears; when is before work that relies
on it. Leave it empty when none apply. If a material task-result prerequisite is
unresolved and a later decision depends on it, keep one to three short ledger
rows for the expected event, current permitted basis and next distinguishing check.
Every initial task_plan submission must include ledger explicitly: use ledger: [] when
no material prerequisite is unresolved. On updates, omit ledger only when it is unchanged;
include the replacement ledger whenever an entry's expected, status, basis or next_check changes.
Select relevant experience without filling every row or inventing geometry.
Keep unresolved concerns that still affect the task. Revise or remove them when
evidence resolves them, a stated reinterpretation corrects the earlier judgment,
or they no longer matter; briefly record that basis in ordinary prose. Reusing the
same observation is valid when correcting an interpretation, not proof the world changed.
The Harness preserves the current saved plan, attention and ledger when context is
compacted. Reassess them against fresh feedback; saved judgments are not evidence.
Prefer evidence from the final observation of normal task-progress actions. Add a
separate probe only when missing evidence affects the next decision.
The task plan is provisional understanding, not an executable script. Adapt your
actions freely. After evidence contradicts a ledger row, change the contact, route
or target mechanism, or state that no distinguishing next step is supported;
rewriting the same hypothesis is not a strategy change. Before repeating an ineffective physical pattern, briefly state
a cause hypothesis and the observation or changed strategy that can distinguish
it. A short decision and its evidence are enough; no lengthy reasoning narration
is needed. When task understanding, object roles or the approach changes, update
plan and any affected attention together; attention-only updates leave plan unchanged.
For local concerns, update only attention. Neither needs rewriting after every motion. Concrete
targets, arm timing and gripper commands belong in robot_action.
Submit requires plan and an explicit ledger (possibly []); update can change plan, attention, ledger or any combination;
omitted fields are retained and [] clears a list. Ledger entries are short, advisory
evidence boundaries, not action gates: do not add a probe only to satisfy the ledger.
Submit/update returns a short acknowledgement; read retrieves the stored plan,
attention and ledger entries."""


def _validate_evidence_ledger(value):
    if not Draft202012Validator(EVIDENCE_LEDGER_SCHEMA).is_valid(value):
        raise ValueError("invalid_task_plan_ledger: provide up to 3 entries with id, expected, status, basis and next_check")
    identifiers = [entry["id"] for entry in value]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("invalid_task_plan_ledger: ids must be unique")


def _ledger_projection(value):
    """Keep the model-facing ledger limited to its public evidence fields."""
    return [{key: entry[key] for key in ("id", "expected", "status", "basis", "next_check")}
            for entry in (value or [])]


def scheduling_failure_advisory(receipt, plan_record=None, *, task_scheduling=True):
    """A brief post-failure decision reminder using public records only.

    No stage is declared ready, no arm is inferred to hold an object, and no
    dependency is removed. The caller chooses the failed-action delivery boundary;
    this pure helper never reads state or consumes a native action step.
    """
    if type(task_scheduling) is not bool:
        raise ValueError("invalid_task_scheduling")
    if (not task_scheduling or not isinstance(receipt, dict)
            or receipt.get("status") not in ("failed", "rejected")):
        return None
    result = dict(kind="scheduling_review", source_kind="advisory",
        suggestion="Before retrying, compare a changed recovery with independent work. Recheck current RGB; if one holds an object, consider delivery while the peer recovers or observed stable support; do not release without support. Preserve dependencies, clear shared corridors, and check withdrawal/next approach against placed objects. Overlap useful phases only where clear; idle arms need not return home.")
    if isinstance(plan_record, dict):
        plan = plan_record.get("plan", {})
        stages = plan.get("stages", []) if isinstance(plan, dict) else []
        if isinstance(stages, list):
            references = [stage.get("id", str(index)) for index, stage in enumerate(stages)
                          if isinstance(stage, dict) and not stage.get("depends_on")][:3]
            if references:
                result["dependency_free_plan_stages_to_reassess"] = references
        if plan_record.get("observation_id"):
            result["plan_observation_id"] = plan_record["observation_id"]
    return result


class TaskPlanning:
    def __init__(self, store, required=True, *, manipulation_protocol=True,
                 task_scheduling=True, operation_candidates: bool = False):
        if type(required) is not bool:
            raise ValueError("invalid_task_plan_required")
        if type(manipulation_protocol) is not bool:
            raise ValueError("invalid_manipulation_protocol")
        if type(task_scheduling) is not bool or type(operation_candidates) is not bool:
            raise ValueError("invalid_operation_planning_flags")
        self.store, self.required = store, required

    def current(self, identity):
        record = self.store.read_state("runtime", "task_plan")
        if record and record.get("identity") == {key: identity[key] for key in IDENTITY}:
            try:
                saved = self.store.read("task_plan", record["plan_id"])
            except KeyError:
                return None
            if saved == record:
                return record
        return None

    def decision_context(self, identity):
        record = self.current(identity)
        if record is None:
            return None
        return dict(plan_id=record["plan_id"], revision=record["revision"],
            plan=deepcopy(record["plan"]), attention=deepcopy(record.get("attention", [])),
            ledger=_ledger_projection(record.get("ledger", [])),
            plan_observation_id=record.get("plan_observation_id"),
            attention_observation_id=record.get("attention_observation_id"),
            ledger_observation_id=record.get("ledger_observation_id"))

    def require(self, identity):
        if self.required and self.current(identity) is None:
            raise ValueError("task_plan_required: observe and submit your initial task_plan before physical actions")

    def call(self, args, identity, observation, correlation):
        if not isinstance(args, dict):
            raise ValueError("task_plan_requires_object")
        operation = args.get("operation", "submit")
        if operation == "read":
            if set(args) - {"operation"}:
                raise ValueError("task_plan_read_accepts_only_operation")
            record = self.current(identity)
            if record is not None and isinstance(record.get("plan"), str):
                result = dict(required=self.required, plan_id=record["plan_id"], plan=record["plan"],
                    plan_observation_id=record.get("plan_observation_id"),
                    attention_observation_id=record.get("attention_observation_id"),
                    ledger_observation_id=record.get("ledger_observation_id"))
                if "attention" in record:
                    result["attention"] = deepcopy(record["attention"])
                result["ledger"] = _ledger_projection(record.get("ledger", []))
                return result
            return dict(required=self.required, plan=record, tool_choices="advisory")
        if isinstance(args.get("plan"), dict) and "attention" in args:
            raise ValueError("task_plan_attention_requires_freeform_plan")
        if operation not in {"submit", "update"} or set(args) - {"operation", "plan", "attention", "ledger", "observation_id"}:
            raise ValueError("task_plan_requires_submit_update_or_read")
        if operation == "submit" and "plan" not in args:
            raise ValueError("task_plan_submit_requires_plan")
        if "plan" in args and (not isinstance(args["plan"], str)
                or not args["plan"].strip() or len(args["plan"]) > 16000):
            raise ValueError("invalid_task_plan: provide nonempty ordinary-language plan (up to 16000 characters)")
        if operation == "submit" and "ledger" not in args:
            raise ValueError("task_plan_submit_requires_ledger: include ledger (use [] when no material prerequisite is unresolved)")
        if operation == "update" and not {"plan", "attention", "ledger"}.intersection(args):
            raise ValueError("task_plan_update_requires_plan_attention_or_ledger")
        if "attention" in args and not Draft202012Validator(ATTENTION_SCHEMA).is_valid(args["attention"]):
            raise ValueError("invalid_task_plan_attention: provide concern, check and when as nonempty text (up to 1600 characters each)")
        if "ledger" in args:
            _validate_evidence_ledger(args["ledger"])
        if observation is None or any(observation.get(k) != identity[k] for k in IDENTITY):
            raise ValueError("task_plan_requires_current_episode_observation")
        if args.get("observation_id", observation["observation_id"]) != observation["observation_id"]:
            raise ValueError("stale_observation")
        previous = self.current(identity)
        if operation == "update" and previous is None:
            raise ValueError("task_plan_not_submitted")
        if (operation == "update" and ({"attention", "ledger"} & set(args))
                and (previous.get("version") != PLANNING_VERSION or not isinstance(previous.get("plan"), str))):
            raise ValueError("task_plan_attention_requires_existing_freeform_plan")
        # A model may repeat the same update while retrying a tool call. Avoid
        # manufacturing a new revision and another context snapshot when the
        # submitted fields and their observation sources are unchanged. The
        # same text on a newer observation remains a real reconsideration and
        # is therefore retained below.
        if operation == "update":
            unchanged = True
            for field in ("plan", "attention", "ledger"):
                if field not in args:
                    continue
                submitted = args[field].strip() if field == "plan" else args[field]
                saved = previous.get(field, "" if field == "plan" else [])
                if submitted != saved or previous.get(field + "_observation_id") != observation["observation_id"]:
                    unchanged = False
                    break
            if unchanged:
                return dict(plan_id=previous["plan_id"], status="unchanged")
        plan = args["plan"].strip() if "plan" in args else previous["plan"]
        # Allocation is serialized by the harness lock; IDs are unique in a run.
        number = len(self.store.records("task_plan")) + 1
        identifier = "P" + str(number)
        record = dict(plan_id=identifier, version=PLANNING_VERSION,
            identity={k: identity[k] for k in IDENTITY},
            revision=previous["revision"] + 1 if previous else 1,
            previous_plan_id=previous["plan_id"] if previous else None,
            observation_id=observation["observation_id"], plan=plan,
            # Save time does not refresh retained judgments. Old archives did not
            # record field sources; keep them unknown until explicitly updated.
            plan_observation_id=(observation["observation_id"] if "plan" in args
                                 else previous.get("plan_observation_id")),
            attention_observation_id=(observation["observation_id"] if "attention" in args
                else previous.get("attention_observation_id") if operation == "update" else None),
            ledger_observation_id=(observation["observation_id"] if "ledger" in args
                else previous.get("ledger_observation_id") if operation == "update" else None),
            source_kind="actor_judgment", tool_choices="advisory")
        if "attention" in args:
            record["attention"] = deepcopy(args["attention"])
        elif operation == "update" and "attention" in previous:
            record["attention"] = deepcopy(previous["attention"])
        if "ledger" in args:
            record["ledger"] = deepcopy(args["ledger"])
        elif operation == "update" and "ledger" in previous:
            record["ledger"] = deepcopy(previous["ledger"])
        self.store.save("task_plan", identifier, record)
        self.store.set_state("runtime", "task_plan", record)
        self.store.event("task_plan.submitted", record, correlation)
        return dict(plan_id=identifier, status="saved")


def _plan_schema_for_error(plan):
    if isinstance(plan, dict):
        return LEGACY_PLAN_SCHEMA
    return PLAN_SCHEMA


register_plan_schema_provider(_plan_schema_for_error)
