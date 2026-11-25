"""Actionable input diagnostics and honest 4-category runtime/infrastructure classification."""

from copy import deepcopy
import math
import re
from typing import Any, Literal

from .tool_output import ToolOutput, wire_response

_plan_schema_provider = None

ErrorCategory = Literal[
    "input_error",
    "unknown_execution",
    "infrastructure_error",
    "internal_runtime_error",
]

_SNAKE_CODE_RE = re.compile(r"\A([a-z][a-z0-9_]{1,95})(?::|\Z)")
_CAMEL_BOUNDARY_RE = re.compile(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])")
_HOST_PATH_RE = re.compile(
    r"(?<![A-Za-z0-9_.-])"
    r"(?:/(?:Users|home|private|var|tmp|opt|mnt|srv|etc|root|proc|sys|dev)/[^\s\"'`:;,)\]]+)"
)

INTERNAL_BUG_TYPES: tuple[type[BaseException], ...] = (
    TypeError,
    AttributeError,
    IndexError,
    NameError,
    UnboundLocalError,
    ImportError,
    ArithmeticError,
    AssertionError,
    NotImplementedError,
    RecursionError,
    SyntaxError,
    BufferError,
)

_WIRE_INTERNAL_TYPE_NAMES: frozenset[str] = frozenset({
    "TypeError",
    "AttributeError",
    "KeyError",
    "IndexError",
    "NameError",
    "UnboundLocalError",
    "ImportError",
    "ModuleNotFoundError",
    "ArithmeticError",
    "ZeroDivisionError",
    "OverflowError",
    "FloatingPointError",
    "AssertionError",
    "NotImplementedError",
    "RecursionError",
    "SyntaxError",
    "BufferError",
    "internal_runtime_error",
})

INFRASTRUCTURE_TYPE_NAMES: frozenset[str] = frozenset({
    "IKServiceError",
    "DriverFault",
    "CameraBindingError",
    "OutOfMemoryError",
})

INFRASTRUCTURE_DOMAIN_CODES: frozenset[str] = frozenset({
    "ik_service_unreachable",
    "robot_gpu_out_of_memory",
    "robot_worker_not_running",
    "robot_worker_disconnected",
    "robot_worker_bootstrap_failed",
    "worker_operation_timeout",
    "simulator_nonfinite_robot_state",
    "worker_not_ready",
})


def _camel_to_snake(name: str) -> str:
    cleaned = _CAMEL_BOUNDARY_RE.sub("_", name or "error").lower()
    cleaned = re.sub(r"[^a-z0-9_]+", "_", cleaned).strip("_")
    return cleaned or "infrastructure_error"


def scrub_host_paths(message: str) -> str:
    """Scrub absolute host filesystem paths so storage/build layout never enters Actor context."""
    if not isinstance(message, str) or "/" not in message:
        return message
    return _HOST_PATH_RE.sub("<host-path>", message)


def extract_domain_code(message: str) -> str | None:
    """Return leading snake_case domain code from a message, else None."""
    if not isinstance(message, str):
        return None
    match = _SNAKE_CODE_RE.match(message.strip())
    return match.group(1) if match else None


class WorkerRemoteError(RuntimeError):
    """Structured worker failure across JSONL IPC; subclasses RuntimeError for caller compatibility."""

    def __init__(
        self,
        wire_error: str,
        *,
        remote_type: str = "RuntimeError",
        remote_module: str | None = None,
        category: ErrorCategory = "internal_runtime_error",
        code: str = "internal_runtime_error",
        remote_message: str = "",
        operation: str | None = None,
        bootstrap: bool = False,
        physics_advanced: bool | None = None,
        executed_steps: int | None = None,
        field_path: str = "",
        suggestion: str | None = None,
        minimal_example: Any = None,
    ) -> None:
        super().__init__(wire_error)
        self.wire_error = wire_error
        self.remote_type = remote_type
        self.remote_module = remote_module
        self.category: ErrorCategory = category
        self.code = code
        self.remote_message = remote_message
        self.operation = operation
        self.bootstrap = bootstrap
        self.physics_advanced = physics_advanced
        self.executed_steps = executed_steps
        self.field_path = field_path
        self.suggestion = suggestion
        self.minimal_example = minimal_example


class InternalToolRuntimeError(RuntimeError):
    """Structured wrapper for unexpected internal software bugs in tool execution."""

    def __init__(
        self,
        message: str,
        *,
        tool: str | None = None,
        error_type: str = "RuntimeError",
        phase: str = "handler",
        retryable: bool = False,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.tool = tool
        self.error_type = error_type
        self.phase = phase
        self.retryable = retryable
        self.category: ErrorCategory = "internal_runtime_error"
        self.code = "internal_runtime_error"


def _is_known_domain_value_message(text: str) -> bool:
    stripped = text.strip()
    if extract_domain_code(stripped) is not None:
        return True
    return stripped in (
        "unsupported operation",
        "scene_version must identify an existing positive version",
        "seed and layout_id must be nonnegative integers",
    )


def _is_infrastructure_message(text: str) -> tuple[bool, str | None]:
    stripped = text.strip()
    lower = stripped.lower()
    if stripped.startswith("bootstrap_failed"):
        return True, "robot_worker_bootstrap_failed"
    if "out of memory" in lower or "cuda error 2" in lower or "error_out_of_device_memory" in lower:
        return True, "robot_gpu_out_of_memory"
    if "robot worker disconnected" in lower:
        return True, "robot_worker_disconnected"
    candidate = stripped.removeprefix("ValueError:").strip() if stripped.startswith("ValueError:") else stripped
    code = extract_domain_code(candidate)
    if code in INFRASTRUCTURE_DOMAIN_CODES:
        return True, code
    return False, None


def is_internal_runtime_error(error: BaseException) -> bool:
    """Return True iff ``error`` represents an unexpected software bug in tool/backend code."""
    if isinstance(error, (ToolInputError, TimeoutError)):
        return False
    _, validation_error_cls = _jsonschema()
    if validation_error_cls and isinstance(error, validation_error_cls):
        return False
    if isinstance(error, InternalToolRuntimeError):
        return True
    if isinstance(error, WorkerRemoteError):
        return error.category == "internal_runtime_error"
    explicit_category = getattr(error, "category", None)
    if explicit_category in ("input_error", "unknown_execution", "infrastructure_error", "internal_runtime_error"):
        return explicit_category == "internal_runtime_error"
    err_type_name = type(error).__name__
    if err_type_name in INFRASTRUCTURE_TYPE_NAMES or err_type_name == "StoreClosed":
        return False
    if isinstance(error, OSError):
        return False
    if isinstance(error, KeyError):
        return False
    if isinstance(error, INTERNAL_BUG_TYPES):
        return True
    msg = str(error)
    if "robot status unknown" in msg.lower():
        return False
    is_infra, _ = _is_infrastructure_message(msg)
    if is_infra:
        return False
    if isinstance(error, ValueError):
        return not _is_known_domain_value_message(msg)
    if isinstance(error, RuntimeError):
        head, sep, tail = msg.partition(":")
        head_clean = head.strip()
        if head_clean in _WIRE_INTERNAL_TYPE_NAMES:
            return True
        if sep and head_clean == "ValueError":
            return not _is_known_domain_value_message(tail)
        if extract_domain_code(msg) is not None:
            return False
        return True
    return True


def _jsonschema():
    try:
        from jsonschema import Draft202012Validator, ValidationError
    except ImportError:
        return None, ()
    return Draft202012Validator, ValidationError


def register_plan_schema_provider(fn):
    """Register `(plan) -> dict` returning the JSON Schema used for ``invalid_task_plan`` diagnostics."""
    global _plan_schema_provider
    _plan_schema_provider = fn


class ToolInputError(ValueError):
    """Keep the established error string while carrying structured context."""
    def __init__(self, code, *, field_path="", actual=None, suggestion=None, minimal_example=None):
        super().__init__(code)
        self.code = code
        self.field_path = field_path
        self.actual = actual
        self.suggestion = suggestion
        self.minimal_example = minimal_example


def _pointer(parts):
    return "".join("/" + str(part).replace("~", "~0").replace("/", "~1") for part in parts)


def _bounded_actual(value):
    if isinstance(value, str):
        return value if len(value) <= 160 else dict(type="string", characters=len(value), prefix=value[:160])
    if isinstance(value, dict):
        if len(value) <= 8 and all(item is None or type(item) in (str, bool, int, float) for item in value.values()):
            return {key: _bounded_actual(item) for key, item in value.items()}
        return dict(type="object", keys=list(value)[:16], key_count=len(value))
    if isinstance(value, (list, tuple)):
        if len(value) <= 8 and all(item is None or type(item) in (str, bool, int, float) for item in value):
            return [_bounded_actual(item) for item in value]
        return dict(type="array", length=len(value))
    if type(value) is float and not math.isfinite(value):
        return str(value)
    if value is None or type(value) in (bool, int, float):
        return value
    return dict(type=type(value).__name__)


def _example_for(schema):
    if not isinstance(schema, dict):
        return "<value meeting the field schema>"
    if "enum" in schema:
        return "<one of: " + ", ".join(map(str, schema["enum"])) + ">"
    kind = schema.get("type")
    if kind == "object":
        return {key: _example_for(schema.get("properties", {}).get(key, {}))
                for key in schema.get("required", [])[:8]}
    if kind == "array":
        return "<array of " + str(schema.get("minItems", "permitted")) + "+ entries matching items schema>"
    if kind in ("number", "integer"):
        return "<caller-selected " + kind + " within the stated bounds>"
    if kind == "boolean":
        return "<true or false>"
    return "<current caller-provided value>"


def _schema_advice(error):
    schema = error.schema
    if not isinstance(schema, dict):
        return 'This field/value is not permitted by the schema.'
    advice = []
    if schema.get('description'):
        advice.append(schema['description'])
    if error.validator in ('required', 'additionalProperties'):
        fields = schema.get('properties', {})
        advice.append('Allowed fields here: ' + ', '.join(fields) + '.')
        if schema.get('required'):
            advice.append('Required fields: ' + ', '.join(schema['required']) + '.')
        kind_schema = fields.get('kind', {})
        kind = kind_schema.get('const') if isinstance(kind_schema, dict) else None
        if kind == 'gripper':
            advice.append('Use goal_normalized in [0,1] (0 closed, 1 open); gripper/goal/gripper_goal are not field names. Arm speed fields do not apply; use gripper_speed_scale or gripper_speed_preset.')
        elif kind == 'move':
            advice.append('Choose grasp_position_m + angles_deg OR position_m + quaternion_xyzw. Optional jaw target is gripper_goal_normalized. Routes are successive move stages, not via/joint fields.')
        elif kind == 'observe':
            advice.append('Supply a nonblank reason describing what to inspect; pose and jaw fields do not belong on observe.')
    elif error.validator == 'oneOf':
        choices = [item.get('required', []) if isinstance(item, dict) else [] for item in error.validator_value]
        if any(choices):
            advice.append('Choose exactly one complete required field group: ' + ' OR '.join(', '.join(fields) for fields in choices) + '.')
    elif error.validator == 'not':
        fields = error.validator_value.get('required', []) if isinstance(error.validator_value, dict) else []
        if fields:
            advice.append('Do not combine these fields: ' + ', '.join(fields) + '; choose one.')
    bounds = {key: schema[key] for key in ('type', 'enum', 'const', 'minimum', 'maximum',
        'exclusiveMinimum', 'exclusiveMaximum', 'minItems', 'maxItems', 'minLength', 'pattern') if key in schema}
    if bounds:
        advice.append('Field constraints: ' + str(bounds) + '.')
    return ' '.join(advice)


def schema_input_error(error, *, code="invalid_tool_arguments", prefix=()):
    """Convert a jsonschema error while retaining the exact offending path."""
    path = [*prefix, *error.absolute_path]
    actual = error.instance
    schema = error.schema
    if error.validator == "required" and isinstance(error.instance, dict):
        missing = [key for key in error.validator_value if key not in error.instance]
        if missing:
            path.append(missing[0])
            actual = dict(present=False)
            schema = error.schema.get("properties", {}).get(missing[0], {})
    elif error.validator == "additionalProperties" and isinstance(error.instance, dict):
        extras = [key for key in error.instance if key not in error.schema.get("properties", {})]
        if extras:
            path.append(extras[0])
            actual = error.instance[extras[0]]
    field_path = _pointer(path)
    return ToolInputError(code, field_path=field_path, actual=actual,
        suggestion=error.message[:512] + '. ' + _schema_advice(error) +
            (' ' + schema['description'] if schema is not error.schema and isinstance(schema, dict) and schema.get('description') else '') +
            " Correct this input field; no action is retried automatically.",
        minimal_example={"field_path": field_path, "value": _example_for(schema),
                         "example_kind": "structural_placeholder; choose values from current permitted evidence"})


def _lookup(arguments, path):
    value = arguments
    for key in path:
        if isinstance(value, list) and type(key) is int and 0 <= key < len(value):
            value = value[key]
            continue
        if not isinstance(value, dict) or key not in value:
            return dict(present=False)
        value = value[key]
    return value


_REACH_FIELD_CODES = ("robot_action_reach_has_unsupported_fields", "robot_action_reach_accepts_move_poses_only")


def _reach_field_path(code, arguments):
    """Path of the first field a robot_action mode=reach request may not carry (``unified_actions`` field sets), in the
    caller's order: a top-level field, or a stage that is not a move / a stage field outside the reach fields."""
    from .embodiment import JAW_GRASP_CENTRE_FIELDS, REACH_REQUEST_FIELDS, REACH_STAGE_FIELDS
    # Reach exists only on profiles that may also offer the grasp-centre pose fields; never point at one of those.
    stage_fields = REACH_STAGE_FIELDS | frozenset(JAW_GRASP_CENTRE_FIELDS)
    if code == "robot_action_reach_has_unsupported_fields":
        extra = [key for key in arguments if key not in REACH_REQUEST_FIELDS]
        return [extra[0]] if extra else ["mode"]
    stages = arguments.get("stages")
    for index, stage in enumerate(stages if isinstance(stages, list) else []):
        if not isinstance(stage, dict):
            return ["stages", index]
        if stage.get("kind") != "move":
            return ["stages", index, "kind"]
        extra = [key for key in stage if key not in stage_fields]
        if extra:
            return ["stages", index, extra[0]]
    return ["stages"]


_IK_SERVICE_UNREACHABLE_SUGGESTION = (
    "Planning got no IK answer: the GPU IK service (cuRobo) or the SSH tunnel to it is down or timed out. "
    "This is not a property of your pose, so changing the pose does not help. Retry at most once after a pause; "
    "if it fails again, stop issuing reach/preview calls and tell the operator that the IK service or its tunnel "
    "needs to be restarted."
)


def _known_error(error, tool, arguments):
    raw = error if isinstance(error, str) else str(error)
    if raw.startswith("ValueError:"):
        raw = raw.removeprefix("ValueError:").strip()
    code = raw
    if code.startswith("invalid_task_plan: provide") and _plan_schema_provider is not None:
        validator_cls, _ = _jsonschema()
        if validator_cls is not None:
            plan = arguments.get("plan")
            plan_schema = _plan_schema_provider(plan)
            if isinstance(plan_schema, dict):
                issue = next(validator_cls(plan_schema).iter_errors(plan), None)
                if issue is not None:
                    return schema_input_error(issue, code="invalid_task_plan", prefix=("plan",))
    domain_code = extract_domain_code(code) or code
    if domain_code == "operation_context_release_evidence_required":
        paths = []
        def locate(value, path):
            if isinstance(value, dict):
                if value.get("state") in ("retained", "released_unverified") and not value.get("release_observation_id"):
                    paths.append(path + ["release_observation_id"])
                for key, item in value.items():
                    locate(item, path + [key])
            elif isinstance(value, list):
                for index, item in enumerate(value):
                    locate(item, path + [index])
        locate(arguments, [])
        path = paths[0] if paths else ["operation_context", "release_observation_id"]
        return ToolInputError(domain_code, field_path=_pointer(path), actual=dict(present=False),
            suggestion="Released_unverified and retained states both need an existing same-episode release observation. Retained additionally needs visible review. If release evidence is unavailable, keep state unknown; do not fabricate an observation ID or physical claim.",
            minimal_example={"state": "unknown", "example_kind": "only when release evidence is unavailable; preserve actual current evidence and uncertainty"})
    rules = {
        "stale_observation": (["observation_id"], "Read robot_observe and reassess the intended action against the current observation. Preview a new batch if needed; do not merely relabel old poses with a newer ID or retry an uncertain execution.", {"tool": "robot_observe", "arguments": {}}),
        "stale_environment_identity": ([], "This evidence or plan belongs to another run/episode/environment generation. Read current observations and build a new plan from current permitted evidence; old IDs cannot be made current by renaming.", {"tool": "robot_observe", "arguments": {}}),
        "action_id_content_conflict": (["action_id"], "This action_id already identifies a different request or plan cursor. Inspect its stored status first. Only a genuinely new intended action may use a new ID; do not use a new ID to retry unknown physical execution.", {"tool": "robot_action_status", "action_id": "<conflicting existing action ID>"}),
        "unknown_program_job": (["job_id"], "Use the job_id actually returned by program_run in this run. An unknown job is not proof that execution did not start; inspect recorded calls before considering new execution.", {"tool": "program_status", "job_id": "<existing job ID>"}),
        "program_review_not_active": (["review_id"], "Read program_status for this job. Resume only when status is awaiting_review, using its current pending_review.review_id and decision=approve or reject. Do not execute the paused request separately.", {"tool": "program_status", "job_id": "<existing job ID>"}),
        "program_time_insufficient_for_action": (["timeout_s"], "The inner action timeout must leave at least 5 seconds in the program work-time budget. Inspect job/action status before choosing a feasible shorter action deadline or a new bounded program; never blindly repeat execution.", {"tool": "program_status", "job_id": "<existing job ID>"}),
        "robot_resource_id_required": (["resource_id"], "Read robot_information with section=resources, then copy an existing resource_id with section=resource.", {"tool": "robot_information", "section": "resources"}),
        "robot_resource_id_requires_resource_section": (["section"], "resource_id is accepted only with section=resource. Use section=resources to list available IDs.", {"section": "resources"}),
        "depth_requires_current_observation": (["request", "observation_id"], "Read robot_observe. Refresh an existing target with target_id, current observation_id and a newly verified RGB ROI, retaining its registered camera; or supply current explicit observation, camera and geometry fields. Do not relabel stale candidates.", {"request": {"kind": "help"}}),
        "depth_target_requires_refresh": (["request", "target_id"], "Refresh this target using refresh_target with target_id, current observation_id and a newly verified ROI before querying candidates. The target retains its original camera; omit camera_id.", {"request": {"kind": "refresh_target", "target_id": "<existing target ID>", "observation_id": "<current observation ID>", "roi": [0, 0, 8, 8]}}),
        "unknown_depth_query": (["request", "kind"], "Choose help, depth_points, region_geometry, grasp_candidates, place_candidates, compare_regions, register_target, refresh_target, target_status, forget_target or skill_help. help returns the required fields.", {"request": {"kind": "help"}}),
        "robot_action_requires_stages": (["stages"], "preview/direct require 1..32 stages. Each has a unique id and kind=move/gripper/observe. Use inspect with a returned reference to read status without creating an action.", {"mode": "inspect", "plan_id": "<existing plan ID>"}),
        "robot_action_requires_observation_id": (["observation_id"], "Call robot_observe, then supply its current observation_id with stages for preview/direct/reach.", {"observation_id": "<current observation ID>"}),
        "robot_action_reach_unavailable": (["mode"], "This robot backend answers no reach (IK-only) queries. Preview the candidate stages instead: a failed preview reports the IK residuals of the stage it could not plan.", {"mode": "preview", "observation_id": "<current observation ID>", "stages": "<intended stages>"}),
        "robot_action_reach_has_unsupported_fields": (["mode"], "mode=reach accepts only mode, observation_id, stages, timeout_s and the batch tolerance fields this robot publishes (tolerance_preset, position_tolerance_m, orientation_tolerance_rad and, where offered, approach_tolerance_rad, roll_tolerance_rad, max_upward_dz_m). It plans and moves nothing, so plan_update, auto_execute_if_clean, action_id, plan/continuation, speed, guard, obstacle and settle fields do not apply; preview the chosen poses with them afterwards.", {"mode": "reach", "observation_id": "<current observation ID>", "stages": [{"id": "c1", "kind": "move", "arm": "<arm ID>", "position_m": "<[x,y,z] from evidence>", "quaternion_xyzw": "<[x,y,z,w]>"}]}),
        "robot_action_reach_accepts_move_poses_only": (["stages"], "mode=reach answers move poses only: every stage needs kind=move, id, arm, position_m and quaternion_xyzw (or, where offered, grasp_centre_m + object_width_m with quaternion_xyzw or fixed_jaw_yaw_deg), plus optional tolerance fields (tolerance_preset, position_tolerance_m, orientation_tolerance_rad and, where offered, approach_tolerance_rad, roll_tolerance_rad, max_upward_dz_m). Remove gripper/observe stages and jaw, speed, guard and depends_on fields here; keep them for the preview of the full batch.", {"id": "c1", "kind": "move", "arm": "<arm ID>", "position_m": "<[x,y,z] from evidence>", "quaternion_xyzw": "<[x,y,z,w]>"}),
        "robot_action_reach_requires_idle_active_run": (["mode"], "A reach query reads the measured joints and needs an idle active run. Inspect the in-flight action and wait for its terminal status before querying. Do not repeat a physical request whose result is unknown.", {"mode": "inspect", "action_id": "<in-flight action ID>"}),
        "robot_action_reach_failed": (["mode"], "The backend returned no reach verdicts and no reason. Read robot_observe, then query once more or preview the candidate stages; do not loop on the same call.", {"tool": "robot_observe", "arguments": {}}),
        "ik_service_unreachable": ([], _IK_SERVICE_UNREACHABLE_SUGGESTION, {}),
        "robot_action_requires_plan_id": (["plan_id"], "Execute requires the plan_id returned by a successful preview. Review the plan first and use its next_action fields.", {"mode": "execute", "plan_id": "<reviewed existing plan ID>"}),
        "robot_action_preview_has_execute_or_inspect_fields": (["mode"], "preview/direct/execute_if_clean accept observation_id, action_id, stages, source_plan_id, start_from_stage_id, stage_overrides, timeout_s, camera_id, detail, settle, obstacles, obstacle_object_ids, obstacle_policy, ignore_obstacle_ids, target_exemption_preset, target_exemption_radius_m, clearance_preset, clearance_height_m, held_object/held_objects, contact_entity_id, initial_contact, placement_contact. Use execute for plan_id/cursor/revision and inspect for trajectory selection.", {"mode": "preview", "observation_id": "<current observation ID>", "stages": "<intended stages>"}),
        "cached_batch_commands_cannot_be_changed": (["mode"], "execute accepts only mode, plan_id, action_id, expected_cursor, expected_revision. To change commands, inspect current state and preview a new batch; do not replay an uncertain execution.", {"mode": "execute", "plan_id": "<reviewed existing plan ID>"}),
        "robot_action_inspect_has_command_fields": (["mode"], "inspect accepts mode, exactly one plan_id/action_id/trajectory_ref, and optional path_kind, camera_id, detail, native_steps OR start_native_step/max_samples. Omit command fields.", {"mode": "inspect", "action_id": "<existing action ID>"}),
        "inspect_requires_one_plan_action_or_trajectory_reference": (["mode"], "Supply exactly one existing plan_id, action_id, or trajectory_ref with mode=inspect.", {"mode": "inspect", "action_id": "<existing action ID>"}),
        "robot_action_plan_unknown_or_expired": (["plan_id"], "Only 16 recent plans in the active process are executable. Read current observation and preview a new batch if needed. Use an existing action_id or trajectory_ref to inspect archived execution; never replay merely because a plan expired.", {"mode": "inspect", "action_id": "<existing executed action ID>"}),
        "robot_action_preview_requires_idle_active_run": (["mode"], "Preview requires an idle active run. Inspect the in-flight action and wait for its terminal status before planning. Do not repeat a physical request whose result is unknown.", {"mode": "inspect", "action_id": "<in-flight action ID>"}),
        "trajectory_native_steps_and_pagination_are_mutually_exclusive": (["native_steps"], "Use native_steps (1..3 existing sample IDs) OR start_native_step/max_samples (up to 25 samples); do not combine them.", {"mode": "inspect", "action_id": "<existing action ID>", "max_samples": 25}),
        "trajectory_path_kind_requires_action_feedback": (["path_kind"], "path_kind selects measured/planned only on an executed action feedback artifact. Omit it when inspecting a preview plan or candidate trajectory.", {"mode": "inspect", "plan_id": "<existing plan ID>"}),
        "trajectory_start_native_step_must_exist": (["start_native_step"], "Use an existing native_step from the inspected trajectory, or follow returned next_native_step. First inspect without pagination to see available key samples.", {"mode": "inspect", "action_id": "<existing action ID>"}),
        "current_observation_unavailable": (["observation_id"], "A current observation is unavailable. Inspect stored action status and call robot_observe if available. If observation remains unavailable, report the blocker; do not invent an ID or repeat an uncertain physical action.", {"tool": "robot_observe", "arguments": {}}),
        "tool_arguments_must_be_object": ([], "Pass a JSON object of named tool arguments.", {}),
        "program_arguments_require_object": ([], "Pass a JSON object of named tool arguments.", {}),
        "operation_context_requires_current_evidence": (["operation_context", "evidence", "observation_id"], "Use the actual current observation supporting this claim; review it before revising the context.", {"evidence": {"observation_id": "<current existing observation ID>"}}),
        "invalid_output_detail": (["detail"], "Choose summary or full.", {"detail": "summary"}),
        "invalid_page_start": (["start"], "Use a non-negative integer sequence offset, not a boolean.", {"start": 0}),
        "invalid_page_limit": (["limit"], "Use an integer limit in the tool's documented range.", {"limit": 25}),
    }
    if domain_code in rules:
        path, suggestion, example = rules[domain_code]
        if domain_code in _REACH_FIELD_CODES:
            path = _reach_field_path(domain_code, arguments)
        return ToolInputError(domain_code, field_path=_pointer(path), actual=_lookup(arguments, path),
                              suggestion=suggestion, minimal_example=example)
    if tool in ('geometry_query', 'depth_query'):
        req = arguments.get('request')
        req_kind = (
            req.get('kind')
            if isinstance(req, dict) and isinstance(req.get('kind'), str) and req.get('kind')
            else '<supported query kind>'
        )
        return ToolInputError(domain_code, field_path='/request', actual=req,
            suggestion="Check this tool's published input field guide and schema for the selected request.kind and required fields. Preserve actual observation/camera/target IDs; use only capabilities and evidence enabled for this run. No geometry estimate proves contact or task success.",
            minimal_example={'request': {'kind': req_kind}})
    return None


def _contains_string_value(value: Any, target: str) -> bool:
    """递归检查入参结构中是否包含目标字符串值，用于区分调用方引用未命中与内部字典漏键。"""
    if isinstance(value, str):
        return value == target
    if isinstance(value, dict):
        return any(_contains_string_value(item, target) for item in value.values())
    if isinstance(value, (list, tuple)):
        return any(_contains_string_value(item, target) for item in value)
    return False


def _non_input_payload(
    *,
    error_text: str,
    code: str,
    category: ErrorCategory,
    error_type: str,
    suggestion: str,
    outcome_semantics: str,
    tool: str | None = None,
) -> dict[str, Any]:
    result: dict[str, Any] = dict(
        error=scrub_host_paths(error_text),
        code=code,
        error_category=category,
        error_type=error_type,
        retryable=False,
        field_path="",
        actual=None,
        suggestion=suggestion,
        minimal_example=None,
        outcome_semantics=outcome_semantics,
    )
    if tool:
        result["tool"] = tool
    return result


def _extract_wire_error_type(error: BaseException) -> str | None:
    if isinstance(error, RuntimeError):
        head, sep, _ = str(error).partition(":")
        if sep and head.strip() in _WIRE_INTERNAL_TYPE_NAMES:
            return head.strip()
    return None


def error_payload(error, *, tool=None, arguments=None):
    """Describe a rejected or failed call across the 4-category decision matrix."""
    _, validation_error_cls = _jsonschema()
    if validation_error_cls and isinstance(error, validation_error_cls):
        structured = schema_input_error(error)
        legacy_error = error.message
    elif isinstance(error, ToolInputError):
        structured, legacy_error = error, str(error)
    else:
        raw_text = str(error)
        if isinstance(error, WorkerRemoteError):
            remote_cat = error.category
            remote_code = error.code
            if remote_cat == "input_error" and remote_code in INFRASTRUCTURE_DOMAIN_CODES:
                remote_cat = "infrastructure_error"
            if remote_cat == "unknown_execution":
                return _non_input_payload(
                    error_text=raw_text,
                    code=remote_code or "execution_status_unknown",
                    category="unknown_execution",
                    error_type=error.remote_type or "TimeoutError",
                    suggestion=(
                        "Do NOT replay or modify arguments to retry; physical motion may have occurred or still be "
                        "stopping. Call robot_action_status or robot_observe to inspect actual state first."
                    ),
                    outcome_semantics="Physical execution outcome is UNKNOWN; motion may have occurred or still be stopping.",
                    tool=tool,
                )
            if remote_cat == "infrastructure_error":
                infra_suggestion = (
                    _IK_SERVICE_UNREACHABLE_SUGGESTION
                    if remote_code == "ik_service_unreachable"
                    else "Backend or infrastructure fault; tool arguments are NOT at fault. Changing tool arguments will not fix this fault; do not loop retrying."
                )
                return _non_input_payload(
                    error_text=raw_text,
                    code=remote_code or "infrastructure_error",
                    category="infrastructure_error",
                    error_type=error.remote_type or "RuntimeError",
                    suggestion=infra_suggestion,
                    outcome_semantics="Backend/infrastructure fault before motion; tool arguments are NOT at fault and no task outcome is inferred.",
                    tool=tool,
                )
            if remote_cat == "internal_runtime_error":
                return _non_input_payload(
                    error_text=raw_text,
                    code="internal_runtime_error",
                    category="internal_runtime_error",
                    error_type=error.remote_type or "RuntimeError",
                    suggestion=(
                        "Do NOT rewrite arguments to satisfy schema—your arguments passed schema validation and hit "
                        "an internal runtime bug. Report or bypass the broken path."
                    ),
                    outcome_semantics="Internal software bug in tool implementation; no motion or task outcome is inferred unless execution_uncertain is set.",
                    tool=tool,
                )
            # WorkerRemoteError with category == "input_error"
            legacy_error = error.remote_message or error.code or raw_text
            if error.suggestion is not None:
                structured = ToolInputError(
                    error.code,
                    field_path=error.field_path,
                    actual=_lookup(arguments, [p for p in error.field_path.split("/") if p])
                    if isinstance(arguments, dict) and error.field_path
                    else arguments,
                    suggestion=error.suggestion,
                    minimal_example=error.minimal_example,
                )
            else:
                try:
                    structured = _known_error(legacy_error, tool, arguments) if isinstance(arguments, dict) else None
                except (TypeError, ValueError, KeyError, AttributeError, ImportError):
                    structured = None
                if structured is None:
                    structured = ToolInputError(
                        error.code or legacy_error.split(":", 1)[0],
                        actual=arguments,
                        suggestion=(
                            "Check the named tool's argument schema and this error. If execution may have started, "
                            "inspect its stored status before considering a corrected request; no automatic retry is implied."
                        ),
                        minimal_example={
                            "tool": tool,
                            "arguments": "<correct only the field identified by the error; preserve valid caller choices>",
                        },
                    )
        elif isinstance(error, TimeoutError) or "robot status unknown" in raw_text.lower() or extract_domain_code(raw_text) == "execution_status_unknown":
            return _non_input_payload(
                error_text=raw_text,
                code="execution_status_unknown",
                category="unknown_execution",
                error_type=type(error).__name__,
                suggestion=(
                    "Do NOT replay or modify arguments to retry; physical motion may have occurred or still be "
                    "stopping. Call robot_action_status or robot_observe to inspect actual state first."
                ),
                outcome_semantics="Physical execution outcome is UNKNOWN; motion may have occurred or still be stopping.",
                tool=tool,
            )
        elif is_internal_runtime_error(error):
            err_type = getattr(error, "error_type", None) or _extract_wire_error_type(error) or type(error).__name__
            return _non_input_payload(
                error_text=raw_text,
                code="internal_runtime_error",
                category="internal_runtime_error",
                error_type=err_type,
                suggestion=(
                    "Do NOT rewrite arguments to satisfy schema—your arguments passed schema validation and hit "
                    "an internal runtime bug. Report or bypass the broken path."
                ),
                outcome_semantics="Internal software bug in tool implementation; no motion or task outcome is inferred unless execution_uncertain is set.",
                tool=tool,
            )
        elif isinstance(error, KeyError) and not (
            len(error.args) == 1
            and isinstance(error.args[0], str)
            and _contains_string_value(arguments, error.args[0])
        ):
            return _non_input_payload(
                error_text=f"internal_runtime_error: KeyError: {raw_text}",
                code="internal_runtime_error",
                category="internal_runtime_error",
                error_type="KeyError",
                suggestion=(
                    "Do NOT rewrite arguments to satisfy schema—your arguments passed schema validation and hit "
                    "an internal runtime bug. Report or bypass the broken path."
                ),
                outcome_semantics="Internal software bug in tool implementation; no motion or task outcome is inferred unless execution_uncertain is set.",
                tool=tool,
            )
        else:
            err_type_name = type(error).__name__
            is_infra_msg, infra_code = _is_infrastructure_message(raw_text)
            if is_infra_msg or err_type_name in INFRASTRUCTURE_TYPE_NAMES or err_type_name == "StoreClosed" or isinstance(error, OSError):
                resolved_code = infra_code or extract_domain_code(raw_text) or _camel_to_snake(err_type_name)
                infra_suggestion = (
                    _IK_SERVICE_UNREACHABLE_SUGGESTION
                    if resolved_code == "ik_service_unreachable"
                    else "Backend or infrastructure fault; tool arguments are NOT at fault. Changing tool arguments will not fix this fault; do not loop retrying."
                )
                return _non_input_payload(
                    error_text=raw_text,
                    code=resolved_code,
                    category="infrastructure_error",
                    error_type=err_type_name,
                    suggestion=infra_suggestion,
                    outcome_semantics="Backend/infrastructure fault before motion; tool arguments are NOT at fault and no task outcome is inferred.",
                    tool=tool,
                )
            # Remaining caller/domain ValueError, RuntimeError("ValueError: <domain_code>"), or direct KeyError miss
            unwrapped = raw_text.removeprefix("ValueError:").strip() if raw_text.startswith("ValueError:") else raw_text
            try:
                structured = _known_error(unwrapped, tool, arguments) if isinstance(arguments, dict) else None
            except (TypeError, ValueError, KeyError, AttributeError, ImportError):
                structured = None
            legacy_error = raw_text
            if structured is None:
                fallback_code = (
                    "unsupported_worker_operation"
                    if unwrapped == "unsupported operation"
                    else extract_domain_code(unwrapped) or unwrapped.split(":", 1)[0]
                )
                structured = ToolInputError(
                    fallback_code,
                    actual=arguments,
                    suggestion=(
                        "Check the named tool's argument schema and this error. If execution may have started, "
                        "inspect its stored status before considering a corrected request; no automatic retry is implied."
                    ),
                    minimal_example={
                        "tool": tool,
                        "arguments": "<correct only the field identified by the error; preserve valid caller choices>",
                    },
                )
    result = dict(
        error=legacy_error,
        code=structured.code,
        error_category="input_error",
        retryable=True,
        field_path=structured.field_path,
        actual=_bounded_actual(structured.actual),
        suggestion=structured.suggestion,
        minimal_example=deepcopy(structured.minimal_example),
        outcome_semantics="Input/runtime error only; no motion or task outcome is inferred.",
    )
    if tool:
        result["tool"] = tool
    return result


def error_response(error, *, tool=None, arguments=None):
    response = ToolOutput(error_payload(error, tool=tool, arguments=arguments)).response()
    response.success = False
    return wire_response(response)
