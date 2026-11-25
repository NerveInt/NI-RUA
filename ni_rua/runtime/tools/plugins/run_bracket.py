"""Session-bracket tools of the external agent backend: ``run_begin`` and ``run_finish`` (design 20 section 2.2).

Registered only when ``agent.backend == "external"`` (``Workbench._configured_tool_specs``), roles ``{"actor"}``,
``parallel=False``, declared effect ``"session"`` (``SESSION_EFFECT``: changes only the agent-session bracket; no
observation, plan, memory or physical effect). The external transport admits calls by that declared effect
(``ToolRegistry.effect_of``); no tool-name list exists. The plugin receives two owner-bound callbacks and nothing
else: it cannot observe, execute or read the Store.

* ``begin(correlation) -> {"instructions", "task": {"text", "source"}, "budget", "turn": {"opened"}}``
  (``ExternalSession.run_begin`` with ``Harness.action_budget()``);
* ``finish(summary, correlation) -> {"turn_closed": True}`` (``ExternalSession.run_finish``).

Display: the briefing's instructions are delivered as one plain text item (the run's startup instructions, the text a
host-mode agent gets except the source-access paragraph, ``actor_instructions.EXTERNAL_AGENT_SOURCE_INSTRUCTIONS``),
followed by one JSON item with ``task``, ``turn`` and ``budget``; typed data carries all four.
Neither tool judges task success.
"""
from copy import deepcopy

from ....core.tool_output import StructuredText, ToolOutput, ToolResponse
from ....cortex.agents.external.protocol import RUN_BEGIN, RUN_FINISH, SESSION_EFFECT
from ..registry import ToolSpec

# Largest run_finish summary, in characters (Unicode code points). The same bound as an operator prompt.
SUMMARY_MAX_CHARS = 16000

RUN_BEGIN_DESCRIPTION = (
    "Start a NI-RUA task, or re-read its briefing. Call this first, before any other NI-RUA tool, and again whenever "
    "your earlier context may have been lost (after a context compaction, in a new session, or after a resume). It "
    "has no physical effect: it does not move the robot and captures no observation. No inputs.\n"
    "Turns: the other NI-RUA tools are served only after run_begin returned its briefing in the open turn; before "
    "that they are refused with external_briefing_required, and after the turn ended with external_turn_closed, whose "
    "text says why (run_finish, an operator stop, an operator reset). This call opens a turn when none is open; "
    "run_finish closes it. Wait for this result before other NI-RUA calls: a call that reaches NI-RUA before "
    "run_begin is refused with nothing executed and can simply be repeated. While the turn is open every NI-RUA tool "
    "call is served, also from your subagents.\n"
    "Result (kind briefing): instructions (text): the operating instructions of this run (robot description, "
    "coordinate frames, units and tool rules); read them before acting. task.text (text or null): the task the "
    "operator started the run with; null when none was given, then the user states the task to you directly. "
    "task.source: operator_prompt (task.text came from the operator) or none (no task was given). native_task "
    "(optional object, present on simulated environments such as RoboDojo): task (environment task id), instruction "
    "(native public task instruction with any public wording override applied), and public_task_rules (text or null: "
    "public task rules for that task). task_notes (optional text): versioned operation notes and any operator task "
    "notes when enabled for this run. budget: the "
    "current execution budget as every action receipt reports it: native_steps_used, native_steps_limit and "
    "native_steps_remaining are counts of native steps (null on an embodiment without a step budget), "
    "per_request_native_limit is the largest step count one request may use, semantics states what is counted, "
    "source names the accounting and unknown_values_are_null is true (null means unknown, never 0). turn.opened "
    "(boolean): true = this call opened a new turn; false = a turn was already open and the briefing was re-read, and "
    "the result then also ends with the current task-plan decision context. startup_grounding (optional object, "
    "present only on the first run_begin of a fresh simulated-clock run before any physical action when initial "
    "grounding is enabled): observation_id (short alias such as obs_1), overview_camera_id (attached overview camera "
    "id or null), image_kind (attached image kind), image_count (integer >= 0, number of attached initial camera "
    "images), and obs3d_included (boolean, true when structured 3D scene perception is attached); when present, the "
    "briefing display also appends the grounded initial observation text and images after the instructions and "
    "structured metadata. This tool does not judge task success.")

RUN_FINISH_DESCRIPTION = (
    "Declare that you have finished the current NI-RUA task, or that you cannot continue it. It has no physical "
    "effect: it does not move the robot, release torque, end the environment or judge task success. It closes the "
    "open turn as completed and records summary as the turn's final answer for the operator and replay; afterwards "
    "the other NI-RUA tools are refused (external_turn_closed) until run_begin opens the next turn. Called while no "
    "turn is open, it records the summary in a turn of its own and closes it. It does not stop helper programs that "
    "are still running (program_run jobs): read program_status first; a job paused for your review stops with "
    "program_run resume decision=reject, any other job runs until its work time ends or the operator stops the run.\n"
    f"Input summary (string, required, 1 to {SUMMARY_MAX_CHARS} characters, no default): your final statement: what "
    "was done, the evidence for the outcome (observation ids, measured values), and what is uncertain or why you "
    "stopped. Report what the evidence shows; never claim success from impression.\n"
    "Result (kind run_finish): turn_closed (boolean): true = the turn is closed.")


def _object(properties, required=None):
    return dict(type="object", properties=properties, required=list(properties if required is None else required),
                additionalProperties=False)


RUN_BEGIN_INPUT = _object({})
RUN_FINISH_INPUT = _object({"summary": dict(
    type="string", minLength=1, maxLength=SUMMARY_MAX_CHARS,
    description=(f"Final statement, 1 to {SUMMARY_MAX_CHARS} characters, required, no default: what was done, the "
                 "evidence for the outcome, and what is uncertain or why you stopped."))})

BRIEFING_OUTPUT = _object({
    "instructions": dict(type="string", minLength=1, description=(
        "Operating instructions of the run: robot description, frames, units and tool rules.")),
    "task": _object({
        "text": dict(type=["string", "null"], description="Operator task text; null when the operator gave none."),
        "source": dict(enum=["operator_prompt", "none"], description="Where task.text came from.")}),
    "native_task": _object({
        "task": dict(type=["string", "null"], description="Native environment task name."),
        "instruction": dict(type=["string", "null"], description="Native public task instruction."),
        "public_task_rules": dict(type=["string", "null"], description="Public task rules text, or null."),
    }),
    "task_notes": dict(type=["string", "null"], description="Operation notes and any custom task notes for this run."),
    "budget": dict(type="object", description="Execution budget: counts of native steps, null when unknown."),
    "turn": _object({"opened": dict(type="boolean", description="True when this call opened the turn.")}),
    "startup_grounding": _object({
        "observation_id": dict(type="string", minLength=1, description=(
            "Short observation alias of the inlined initial observation (e.g. obs_1).")),
        "overview_camera_id": dict(type=["string", "null"], description=(
            "Camera id of the attached overview view, or null.")),
        "image_kind": dict(type="string", minLength=1, description=(
            "Kind of initial grounding images attached (e.g. obs3d_overlay, raw_rgb, none).")),
        "image_count": dict(type="integer", minimum=0, description=(
            "Number of camera images attached to this briefing.")),
        "obs3d_included": dict(type="boolean", description=(
            "True when structured 3D scene perception (object_table/grasp_hints/scene3d) is included.")),
    })},
    required=["instructions", "task", "budget", "turn"])
RUN_FINISH_OUTPUT = _object({"turn_closed": dict(const=True, description="The turn is closed.")})


def make_run_bracket_tools(begin, finish):
    """``(run_begin ToolSpec, run_finish ToolSpec)`` bound to the owner callbacks ``begin`` / ``finish`` (module
    docstring). Raises ``TypeError("run_bracket_callbacks_required")`` unless both are callable."""
    if not callable(begin) or not callable(finish):
        raise TypeError("run_bracket_callbacks_required")

    def handle_begin(context, arguments, correlation):
        raw = begin(correlation)
        grounded = None
        if isinstance(raw, dict) and "_startup_grounding" in raw:
            data = {key: value for key, value in raw.items() if key != "_startup_grounding"}
            grounded = raw["_startup_grounding"]
        else:
            data = raw
        # 外部模式在 run_begin 真正将含 Robot runtime capabilities 的简报送达主链时，按当前 delta 纪元预播种能力基线
        services = getattr(context, "services", None)
        if (services is not None
                and getattr(services, "capabilities", None) is not None
                and not getattr(services, "context_epoch_volatile", True)
                and (correlation or {}).get("context_scope", "main") == "main"):
            cap_delivery = getattr(services, "_capability_delivery", None)
            epoch_fn = getattr(services, "_model_context_epoch", None)
            if cap_delivery is not None and callable(epoch_fn):
                cap_delivery.project(
                    services.capabilities,
                    epoch=epoch_fn(),
                    current_action_budget=data["budget"] if isinstance(data, dict) and "budget" in data else services.action_budget(),
                )
        display = {key: data[key] for key in ("task", "native_task", "task_notes", "turn", "budget", "startup_grounding")
                   if key in data and data[key] is not None}
        content = [dict(type="inputText", text=data["instructions"]), StructuredText(display)]
        if grounded is not None:
            content.append(dict(type="inputText", text=grounded.section_text))
            for img in grounded.image_items:
                content.append(dict(type="inputImage", imageUrl=img["url"]))
        return ToolResponse(content, data=data, kind="briefing")

    def handle_finish(context, arguments, correlation):
        return ToolOutput(finish(arguments["summary"], correlation), kind="run_finish").response()

    begin_definition = dict(type="function", name=RUN_BEGIN, description=RUN_BEGIN_DESCRIPTION,
                            inputSchema=deepcopy(RUN_BEGIN_INPUT))
    finish_definition = dict(type="function", name=RUN_FINISH, description=RUN_FINISH_DESCRIPTION,
                             inputSchema=deepcopy(RUN_FINISH_INPUT))
    return (ToolSpec(begin_definition, handle_begin, roles=frozenset({"actor"}), effect=SESSION_EFFECT,
                     api_version="1.0", output_schemas={"briefing": deepcopy(BRIEFING_OUTPUT)}, parallel=False),
            ToolSpec(finish_definition, handle_finish, roles=frozenset({"actor"}), effect=SESSION_EFFECT,
                     api_version="1.0", output_schemas={"run_finish": deepcopy(RUN_FINISH_OUTPUT)}, parallel=False))
