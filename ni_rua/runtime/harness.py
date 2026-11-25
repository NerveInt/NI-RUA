"""The only observed-world action writer, shared by dynamic tools and programs."""

from functools import cached_property
import inspect
import threading
import time
from copy import deepcopy

from ..core.contracts import IDENTITY, canonical, digest, new_id, validate_action, validate_identity
from ..core.episode_state import EpisodeStateLedger, slot_reset
from ..robot.model.embodiments import EMBODIMENT_RECORD_PREFIX, EmbodimentProfile, arx_x5_profile, profile_record
from ..robot.model.grasp_contact import contact_target_feedback
from .actions.action_state_ledger import ACTION_STATE_LEDGER_SLOT
from .actions.home_check import HomePoseCheck
from .actions.obstacle_registry import PERSISTENT_OBSTACLES_SLOT
from .actions.unified_actions import (
    UNIFIED_ACTION_CALL_IDS_SLOT,
    UNIFIED_ACTION_PLANS_SLOT,
    UNIFIED_ACTION_SEQ_SLOT,
    UNIFIED_PLAN_SEQ_SLOT,
    UNIFIED_TOOL_RESULTS_SLOT,
)
from .decision_context import DecisionContextBridge, REMINDER_IN_COMPACT_REQUEST, REMINDER_IN_TOOL_RESULT
from .observation_pipeline import ObservationPipeline, _action_focus_roi, _omitted_image_feedback
from ..robot.motion.motion_speed import SPEED_INPUT_FIELDS, SPEED_EFFECTIVE_FIELDS
from ..robot.motion.action_compilation import validate_backend_action
from ..cortex.context.task_planning import TaskPlanning
from ..perception.external.held_perception import HELD_PERCEPTION_SLOT
from ..perception.external.obs3d import (
    OBS3D_HISTORY_SLOT,
    OBS3D_MEMO_SLOT,
    PENDING_ACTION_OBS3D_SLOT,
    PENDING_OBSERVE_OBS3D_SLOT,
    VIRTUAL_FIELD_ERROR_SLOT,
    VIRTUAL_FIELD_SLOT,
)
from ..perception.external.pose_skill import POSE_SKILL_GRASPS_SLOT
from ..perception.external.wrist_progress_cache import WRIST_PROGRESS_CACHE_SLOT
from .tools.tool_search import TOOL_SEARCH_NAME, UNLOCKED_DEFERRED_TOOLS_SLOT
from ..perception.scene.store import SceneStore
from ..robot.model.geometry import GeometryTools
from ..core.observation_policy import LEGACY_ACTOR_PROFILE, require_actor_observation, actor_robot_state, reject_privileged, has_calibration, allows_depth
from ..perception.depth.depth_runtime import DepthRuntime
from ..core import tool_errors
from ..core.storage.images import image_url, store_image_settings
from ..core.storage.store import StoreClosed
from ..core.presentation.actor_payload import ACTOR_PAYLOAD_PROFILES
from ..core.presentation.feedback_delivery import execution_unconfirmed
from ..core.presentation.reply_compaction import compact_content_items, drop_static_once, unify_detail_entries
from ..core.presentation.reply_encoding import encode_compact_items
from ..core.tool_output import StructuredText, ToolOutput, ToolResponse, output_contract_error, validate_display, wire_response


_UNSET = object()

_LOCAL_INTERNAL_BUG_TYPES = (
    TypeError,
    AttributeError,
    ImportError,
    NameError,
    UnboundLocalError,
    IndexError,
    ArithmeticError,
    AssertionError,
    NotImplementedError,
    RecursionError,
    SyntaxError,
    BufferError,
)
_LOCAL_INTERNAL_PREFIXES = frozenset({
    "TypeError",
    "AttributeError",
    "ImportError",
    "ModuleNotFoundError",
    "NameError",
    "UnboundLocalError",
    "IndexError",
    "ZeroDivisionError",
    "ArithmeticError",
    "AssertionError",
    "NotImplementedError",
    "RecursionError",
    "SyntaxError",
    "BufferError",
    "internal_runtime_error",
})
_PASSIVE_STATUS_TOOLS = frozenset({"program_run", "program_status", "task_plan", "robot_action_status"})


class _FallbackInternalToolRuntimeError(RuntimeError):
    """Local fallback when ``core.tool_errors.InternalToolRuntimeError`` is not yet present."""

    def __init__(self, message, *, tool=None, error_type="RuntimeError"):
        super().__init__(message)
        self.tool = tool
        self.error_type = error_type
        self.category = "internal_runtime_error"
        self.code = "internal_runtime_error"
        self.retryable = False


def _extract_internal_error_info(error: BaseException) -> tuple[str, str]:
    remote_type = getattr(error, "remote_type", None) or getattr(error, "error_type", None)
    if isinstance(remote_type, str) and remote_type:
        msg = getattr(error, "remote_message", None) or str(error)
        return remote_type, str(msg)
    raw = str(error)
    if type(error) is RuntimeError and ":" in raw:
        head, tail = raw.split(":", 1)
        prefix = head.strip()
        if prefix in _LOCAL_INTERNAL_PREFIXES and prefix != "internal_runtime_error":
            return prefix, tail.strip()
        if prefix == "bootstrap_failed" and ":" in tail:
            sub_head, sub_tail = tail.split(":", 1)
            if sub_head.strip() in _LOCAL_INTERNAL_PREFIXES:
                return sub_head.strip(), sub_tail.strip()
    return type(error).__name__, raw


def is_internal_runtime_error_compat(error: BaseException) -> bool:
    if isinstance(error, (tool_errors.ToolInputError, StoreClosed)):
        return False
    _, validation_error_cls = tool_errors._jsonschema()
    if validation_error_cls and isinstance(error, validation_error_cls):
        return False
    checker = getattr(tool_errors, "is_internal_runtime_error", None)
    if callable(checker) and checker(error):
        return True
    if getattr(error, "category", None) == "internal_runtime_error":
        return True
    if isinstance(error, _LOCAL_INTERNAL_BUG_TYPES):
        return True
    if isinstance(error, RuntimeError):
        prefix = str(error).split(":", 1)[0].strip()
        if prefix in _LOCAL_INTERNAL_PREFIXES:
            return True
        if prefix == "bootstrap_failed":
            parts = str(error).split(":", 2)
            if len(parts) >= 2 and parts[1].strip() in _LOCAL_INTERNAL_PREFIXES:
                return True
    return False


def _wrap_internal_runtime_error(tool_name: str, error: BaseException) -> RuntimeError:
    cls = getattr(tool_errors, "InternalToolRuntimeError", None)
    if cls is not None and isinstance(error, cls):
        return error
    if isinstance(error, _FallbackInternalToolRuntimeError):
        return error
    orig_type, orig_msg = _extract_internal_error_info(error)
    wire_msg = f"internal_runtime_error: {orig_type}: {orig_msg}" if orig_msg else f"internal_runtime_error: {orig_type}"
    if cls is not None:
        for build in (
            lambda: cls(wire_msg, tool=tool_name, error_type=orig_type),
            lambda: cls(tool_name, orig_type, orig_msg),
            lambda: cls(wire_msg),
        ):
            try:
                inst = build()
                if not getattr(inst, "tool", None):
                    inst.tool = tool_name
                if not getattr(inst, "error_type", None):
                    inst.error_type = orig_type
                if not getattr(inst, "category", None):
                    inst.category = "internal_runtime_error"
                if not getattr(inst, "code", None):
                    inst.code = "internal_runtime_error"
                return inst
            except TypeError:
                continue
    return _FallbackInternalToolRuntimeError(wire_msg, tool=tool_name, error_type=orig_type)


def _internal_error_payload(wrapped: RuntimeError, tool_name: str, args) -> dict:
    payload = tool_errors.error_payload(
        wrapped,
        tool=tool_name,
        arguments=args if isinstance(args, dict) else None,
    )
    if payload.get("error_category") != "internal_runtime_error":
        payload = dict(
            payload,
            code="internal_runtime_error",
            error_category="internal_runtime_error",
            error_type=getattr(wrapped, "error_type", type(wrapped).__name__),
            retryable=False,
            field_path="",
            actual=None,
            minimal_example=None,
            suggestion=(
                "Internal harness or backend runtime defect; modifying tool arguments will not resolve this error."
            ),
            outcome_semantics=(
                "Internal software bug in tool implementation (no motion inferred unless execution_uncertain is set)."
            ),
        )
    return payload


class _BridgeAttr:
    """Data descriptor delegating instance state to ``harness.decision_context`` while keeping class defaults."""

    def __init__(self, attr_name, class_default=_UNSET):
        self.attr_name = attr_name
        self.class_default = class_default

    def __get__(self, instance, owner=None):
        if instance is None:
            return self if self.class_default is _UNSET else self.class_default
        return getattr(instance.decision_context, self.attr_name)

    def __set__(self, instance, value):
        setattr(instance.decision_context, self.attr_name, value)


class _SlotAttr:
    """Data descriptor routing legacy attribute reads/writes into ``harness.episode_state`` without ``__dict__`` stashing."""

    def __init__(self, slot, *, mode="get"):
        self.slot = slot
        self.mode = mode
        self.attr_name = slot.name

    def __set_name__(self, owner, name):
        self.attr_name = name

    def __get__(self, instance, owner=None):
        if instance is None:
            return self
        ledger = instance.episode_state
        ledger.sync_identity(getattr(instance, "identity", None))
        if self.mode == "attr":
            if not ledger.has(self.slot):
                raise AttributeError(self.attr_name)
            return ledger.peek(self.slot)
        if self.mode == "peek":
            return ledger.peek(self.slot, None)
        return ledger.get(self.slot)

    def __set__(self, instance, value):
        ledger = instance.episode_state
        ledger.sync_identity(getattr(instance, "identity", None))
        ledger.set(self.slot, value)

    def __delete__(self, instance):
        ledger = instance.episode_state
        ledger.sync_identity(getattr(instance, "identity", None))
        if self.mode == "attr" and not ledger.has(self.slot):
            raise AttributeError(self.attr_name)
        ledger.reset_slot(self.slot)


def resolve_embodiment_profile(robot, config):
    """The run's ``EmbodimentProfile`` (design 13 section 5.1).

    ``robot.embodiment_profile()`` when the backend defines that hook (looked up statically on the instance and its
    class, so Mock-like doubles that answer every attribute do not count), else the default ARX-X5 robot at the run's
    thresholds: ``arx_x5_profile(thresholds=config.features)``, equal to ``embodiments.ARX_X5_PROFILE`` at default
    thresholds (no RoboDojo worker binding, so ``per_request_step_limit`` is None). A hook returning anything else
    is refused with ``ValueError("invalid_embodiment_profile")``.
    """
    if inspect.getattr_static(robot, "embodiment_profile", None) is None:
        return arx_x5_profile(thresholds=config.features)
    profile = robot.embodiment_profile()
    if not isinstance(profile, EmbodimentProfile):
        raise ValueError("invalid_embodiment_profile: backend embodiment_profile() must return an EmbodimentProfile")
    return profile


def _execution_acknowledgement(raw):
    """Keep only validated execution facts if the remaining worker payload is unsafe."""
    status = raw.get("status")
    receipt = dict(status=status if isinstance(status, str) and status in {
        "completed", "partial", "failed", "stopped", "cancelled", "rejected", "unknown", "status_unknown"} else "unknown")
    for key in ("executed_steps", "executed_native_steps", "completed_native_steps", "executed_control_steps"):
        value = raw.get(key)
        if key in raw and (value is None or type(value) is int and value >= 0):
            receipt[key] = value
    if isinstance(raw.get("reason"), str):
        receipt["reason"] = raw["reason"]
    for key in ("execution_uncertain", "stop_confirmed"):
        if type(raw.get(key)) is bool:
            receipt[key] = raw[key]
    stop = raw.get("stop_confirmation")
    if stop is False:
        receipt["stop_confirmation"] = False
    elif isinstance(stop, dict) and type(stop.get("confirmed")) is bool:
        receipt["stop_confirmation"] = dict(confirmed=stop["confirmed"])
    elif (isinstance(stop, dict) and set(stop) <= {"clock", "actuator_target", "settling_steps"}
            and stop.get("clock") == "paused"
            and stop.get("actuator_target") == "last_native_command_retained"
            and type(stop.get("settling_steps")) is int and stop["settling_steps"] >= 0):
        receipt["stop_confirmation"] = deepcopy(stop)
    # Keep nested negative acknowledgements without copying unsafe arm/stage bodies.
    for field in ("arm_results", "gripper_execution", "stage_results"):
        if isinstance(raw.get(field), dict) and any(execution_unconfirmed(value) for value in raw[field].values()):
            receipt["execution_uncertain"] = True
    return receipt


class Harness:
    # Channel recorded on actor deliveries ("delivery" records): "codex_tool" for the Codex backend (unchanged),
    # "agent_tool" for other agent backends. Set by Workbench from the session's backend_kind.
    delivery_channel = _BridgeAttr("delivery_channel", "codex_tool")
    # Forced full delivery (design 04 §4.3). False (default; Codex): an image, static observation metadata field,
    # capability summary or program-log excerpt already delivered in the current model-context epoch may be left out
    # of a later actor tool result. True: _model_context_epoch() carries a fresh marker on every call, so every actor
    # tool result is complete and nothing delivered earlier is assumed to be in the model's context. Workbench sets it
    # right after session.start() from AgentSession.context_delivery ("full" -> True).
    context_epoch_volatile = _BridgeAttr("context_epoch_volatile", False)
    # Where the decision-context reminder goes after a model-context reset: REMINDER_IN_COMPACT_REQUEST (default;
    # Codex, the Harness never adds it) or REMINDER_IN_TOOL_RESULT (other backends: handle_tool sideband). Set by
    # Workbench from the session's backend_kind.
    decision_reminder_channel = _BridgeAttr("decision_reminder_channel", REMINDER_IN_COMPACT_REQUEST)
    # True while a reported model-context reset waits for its reminder: the next actor tool response returned by
    # handle_tool gains one inputText item with decision_reminder_text(...), then the flag clears. Set only by
    # model_context_reset() and only on the REMINDER_IN_TOOL_RESULT channel.
    decision_reminder_due = _BridgeAttr("decision_reminder_due", False)
    _decision_reminder_lock = _BridgeAttr("_decision_reminder_lock")
    # () -> int | None: UTF-8 byte budget the agent host applies to the text of one tool output
    # (AgentSession.tool_output_text_budget_bytes), set by Workbench for every backend; None here or returned = the
    # host has no known limit. Read per reply while features.budgeted_action_reply is on: every kind="action" reply
    # is laid out by decision value, and blocks are replaced by references only against a known limit
    # (core/presentation/reply_budget.fit_reply).
    action_reply_budget = None
    # () -> tuple[int | None, int | None, bool] | None: 宿主单条工具输出截断上限（字节/行数/带图豁免），由 Workbench 注入。
    host_truncation_limit = None
    # True (default): the host loads deferLoading tools on demand (native tool search), so a reply's detail entry
    # must not name them directly. False: the host publishes the whole catalogue eagerly (MCP hosts). Set by
    # Workbench from AgentCapabilities.deferred_tools.
    deferred_tool_loading = True
    require_tool_search_unlock = False
    _last_delivered_views = _BridgeAttr("last_delivered_views")
    _view_context_epoch = _BridgeAttr("view_context_epoch")
    _capability_delivery = _BridgeAttr("capability_delivery")
    _program_output_delivery = _BridgeAttr("program_output_delivery")
    _metadata_delivery = _BridgeAttr("metadata_delivery")

    # Typed EpisodeStateLedger-backed compatibility descriptors (zero ad-hoc setattr on Harness.__dict__).
    unified_action_plans = _SlotAttr(UNIFIED_ACTION_PLANS_SLOT, mode="get")
    _unified_plan_sequence = _SlotAttr(UNIFIED_PLAN_SEQ_SLOT, mode="attr")
    _unified_action_sequence = _SlotAttr(UNIFIED_ACTION_SEQ_SLOT, mode="attr")
    _unified_action_call_ids = _SlotAttr(UNIFIED_ACTION_CALL_IDS_SLOT, mode="get")
    _unified_tool_results = _SlotAttr(UNIFIED_TOOL_RESULTS_SLOT, mode="get")
    _action_state_ledger = _SlotAttr(ACTION_STATE_LEDGER_SLOT, mode="get")
    _persistent_obstacles = _SlotAttr(PERSISTENT_OBSTACLES_SLOT, mode="get")
    _obs3d_memo = _SlotAttr(OBS3D_MEMO_SLOT, mode="peek")
    _obs3d_history = _SlotAttr(OBS3D_HISTORY_SLOT, mode="get")
    _pending_observe_obs3d = _SlotAttr(PENDING_OBSERVE_OBS3D_SLOT, mode="peek")
    _pending_action_obs3d = _SlotAttr(PENDING_ACTION_OBS3D_SLOT, mode="peek")
    _virtual_spatial_field = _SlotAttr(VIRTUAL_FIELD_SLOT, mode="peek")
    _virtual_field_last_error = _SlotAttr(VIRTUAL_FIELD_ERROR_SLOT, mode="peek")
    _wrist_progress_cache = _SlotAttr(WRIST_PROGRESS_CACHE_SLOT, mode="attr")
    _held_perception = _SlotAttr(HELD_PERCEPTION_SLOT, mode="attr")
    _pose_skill_grasps = _SlotAttr(POSE_SKILL_GRASPS_SLOT, mode="get")
    unlocked_deferred_tools = _SlotAttr(UNLOCKED_DEFERRED_TOOLS_SLOT, mode="get")

    # Retained as None for read-only compatibility with observation_pipeline.py; dead safety-stop hooks removed.
    spatial_stream = None

    @cached_property
    def episode_state(self):
        return EpisodeStateLedger(getattr(self, "identity", None))

    @cached_property
    def decision_context(self):
        return DecisionContextBridge(self)

    @cached_property
    def observation_pipeline(self):
        return ObservationPipeline(self)

    @property
    def _persistent_obstacles_identity(self):
        ident = getattr(self, "identity", None)
        self.episode_state.sync_identity(ident)
        if not self.episode_state.has(PERSISTENT_OBSTACLES_SLOT) or not isinstance(ident, dict):
            return None
        return {k: ident.get(k) for k in IDENTITY}

    @_persistent_obstacles_identity.setter
    def _persistent_obstacles_identity(self, value):
        self.episode_state.sync_identity(value if isinstance(value, dict) else getattr(self, "identity", None))

    def __init__(self, store, robot, max_actions=None, actor_payload_profile=_UNSET, enable_wait=_UNSET,
                 observation_profile=None, motion_review=None, grasp_memory_store=None, vision_compare=_UNSET, config=None,
                 task_plan_required=False, tool_specs=(), max_internal_runtime_errors=3):
        from ..core.config import Config
        self.config = config if config is not None else Config.load()
        actor_payload_profile = self.config.features.actor_payload_profile if actor_payload_profile is _UNSET else actor_payload_profile
        enable_wait = self.config.features.enable_wait if enable_wait is _UNSET else enable_wait
        vision_compare = self.config.features.vision_compare if vision_compare is _UNSET else vision_compare
        self.disabled_tools = frozenset(self.config.features.disabled_tools)
        # The run's embodiment, resolved once from the backend hook before the tool registry exists, so every
        # consumer that receives this runtime (create_registry, actions, perception, programs, Workbench) reads the
        # same frozen profile as ``harness.embodiment``. Archived by start() as configuration record
        # ``embodiment_record_id`` (observer data; never model context).
        self.embodiment = resolve_embodiment_profile(robot, self.config)
        self.embodiment_record_id = None
        # finding C6: the configured default speed must lie in this embodiment's range (SO-101 [0.25, 1.0]), so a
        # wrong value fails at start instead of on every move.
        scale = self.embodiment.speed_scale
        if not scale.min <= self.config.features.default_motion_speed_scale <= scale.max:
            raise ValueError(f"invalid_default_motion_speed_scale: {self.config.features.default_motion_speed_scale} "
                             f"outside [{scale.min}, {scale.max}] of {self.embodiment.robot_model_id}")
        from .tools.builtins import create_registry
        self.tool_registry = create_registry(tool_specs, runtime=self)
        if TOOL_SEARCH_NAME in self.tool_registry.names:
            self.deferred_tool_loading = True
            self.require_tool_search_unlock = True
        from .tools.builtins import CONFIG_TOOL_NAMES
        unknown = self.disabled_tools - self.tool_registry.names - CONFIG_TOOL_NAMES
        if unknown: raise ValueError('unknown_disabled_tools:' + ','.join(sorted(unknown)))
        if actor_payload_profile not in ACTOR_PAYLOAD_PROFILES:
            raise ValueError("invalid_actor_payload_profile")
        if type(enable_wait) is not bool:
            raise ValueError("invalid_enable_wait")
        if type(vision_compare) is not bool:
            raise ValueError("invalid_vision_compare")
        self.vision_compare = vision_compare
        self.store, self.robot, self.max_actions = store, robot, max_actions
        from ..core.presentation.actor_presentation import ActorPresentation
        self.actor_presentation = ActorPresentation(store)
        from ..core.presentation.actor_output import ActorOutput
        self.actor_output = ActorOutput(store, self.actor_presentation, actor_payload_profile)
        # Actor sessions require a plan; direct backend clients can retain their
        # existing execution contract without pretending to be task actors.
        self.task_planning = TaskPlanning(store, required=task_plan_required,
                                          manipulation_protocol=self.config.features.manipulation_protocol,
                                          task_scheduling=self.config.features.task_scheduling,
                                          operation_candidates=self.config.features.operation_candidates)
        if task_plan_required and "task_plan" in self.disabled_tools:
            raise ValueError("required_task_plan_tool_cannot_be_disabled")
        # Workbench passes its explicit profile; unprofiled custom backends keep
        # their existing legacy interface. Native RobotBackend defaults calibrated.
        self.observation_profile = observation_profile or (config.features.observation_profile if config is not None else getattr(robot, "observation_profile", LEGACY_ACTOR_PROFILE))
        self.motion_review = (self.config.features.motion_review and hasattr(robot, "preview_motion") and has_calibration(self.observation_profile)) if motion_review is None else motion_review
        if type(self.motion_review) is not bool: raise ValueError("invalid_motion_review")
        if self.motion_review and not has_calibration(self.observation_profile):
            raise ValueError("motion_review_requires_calibration")
        self.depth_skills = self.config.features.depth_skills
        self.depth_auto_feedback = self.config.features.depth_auto_feedback
        self.depth_event_payload = self.config.features.depth_event_payload
        if any((self.depth_skills, self.depth_auto_feedback, self.depth_event_payload)) and not allows_depth(self.observation_profile):
            raise ValueError("depth_features_require_explicit_rgbd_profile")
        if self.depth_event_payload and not self.depth_auto_feedback:
            raise ValueError("depth_event_payload_requires_auto_feedback")
        from ..cortex.context.operation_memory import OperationMemory
        self.operation_memory = OperationMemory(store, shared_store=grasp_memory_store)
        self.actor_payload_profile, self.enable_wait = actor_payload_profile, enable_wait
        saved_image_settings = store.read_state("runtime", "image_settings")
        self.image_settings = store_image_settings(store, self.config.images.to_dict())
        if saved_image_settings is not None and "action_feedback_policy" not in saved_image_settings:
            # 兼容早于 action_feedback_policy 的历史归档，恢复其实际使用的 feedback_policy
            self.image_settings["action_feedback_policy"] = saved_image_settings.get("feedback_policy", "all")
        if hasattr(robot, "image_settings"):
            robot.image_settings = dict(self.image_settings)
        self.identity = store.identity
        self.episode_state = EpisodeStateLedger(self.identity)
        self.decision_context = DecisionContextBridge(self)
        self.observation_pipeline = ObservationPipeline(self)
        self.scene = SceneStore(store.root / "scene", store.resolve_artifact, store.put_artifact)
        self.geometry = GeometryTools(self.scene, settings=self.image_settings, profile=self.embodiment)
        self.depth = DepthRuntime(self)
        self.scene_artifacts = store
        self.programs = None
        self.action_policy = None
        self.visual_reviewer = None
        self.latest = None
        self.home_check = (HomePoseCheck(store, self.embodiment)
                           if self.config.features.completion_guidance in {"hook", "both"}
                           and self.embodiment.home_check is not None else None)
        self.capabilities = None
        self.lock = threading.RLock()
        self.robot_lock = threading.Lock()
        self.policy_lock = threading.Lock()
        self._late_reply_lock = threading.Lock()
        self.cancel = threading.Event()
        self.accepting = True
        self.stop_revision = 0
        self.batch_plan_revision = 0
        self.active_action = None
        self.active_context = None
        self.wall_deadline = None
        self.episode_done = threading.Event()
        self.episode_terminal = None
        self.on_episode_end = None
        self.started_at = self.last_native_progress_at = time.monotonic()
        self.model_wait_at_start = self.model_wait_at_native_progress = 0.0
        self.native_action_count = 0
        self.native_budget_limit = None
        self.native_budget_known = False
        self.failed_paths = set()
        self.action_count = store.accepted_action_count()
        self.states = {}
        self.thread_id = None
        self.call_items = {}
        self._closed_perception_sessions = set()
        self._in_flight_actions = set()
        self._action_stop_revisions = {}
        self._action_intervention_revisions = {}
        self._operator_intervention_revision = 0
        self._pending_late_replies = {}
        self._reconciled_late_actions = set()
        self._operator_acknowledged_actions = set()
        self._operator_acknowledged_programs = set()
        self.max_internal_runtime_errors = max_internal_runtime_errors
        self.max_total_internal_runtime_errors = 5
        self.internal_runtime_errors = []
        self.consecutive_internal_runtime_errors = 0
        self.consecutive_tool_internal_errors = {}
        self.internal_runtime_tripped = None
        self._admitted_action_calls = set()
        self.host_truncation_limit = None
        self._wire_late_reply_hook()
        # Lost worker acknowledgements are never automatically retried on recovery.
        for request in store.records("action_request"):
            key = request["action_id"]
            try:
                self.states[key] = store.read("action_receipt", key)
            except KeyError:
                self.states[key] = dict(status="unknown", reason="gateway_restarted_without_terminal_receipt")
                self.accepting = False
                store.event("action.status_unknown", self.states[key], {"action_id": key})
            if (all(request.get(name) == self.identity[name] for name in IDENTITY)
                    and execution_unconfirmed(self.states[key])):
                self.accepting = False
                self.episode_state.action_settled(continuous=False)

    def _wire_late_reply_hook(self):
        if self.robot is None or isinstance(self.robot, (int, float, str, bytes, bool)):
            return
        if getattr(self.robot, "on_late_reply", None) is None:
            try:
                self.robot.on_late_reply = self._on_late_worker_reply
            except (AttributeError, TypeError):
                pass
        for attr in ("process", "worker", "transport"):
            child = inspect.getattr_static(self.robot, attr, None)
            if child is None:
                child = getattr(self.robot, "__dict__", {}).get(attr)
            if child is not None and inspect.getattr_static(child, "on_late_reply", _UNSET) is not _UNSET:
                if getattr(child, "on_late_reply", None) is None:
                    try:
                        child.on_late_reply = self._on_late_worker_reply
                    except (AttributeError, TypeError):
                        pass

    def start(self):
        self._wire_late_reply_hook()
        self.capabilities = deepcopy(self.robot.start())
        from ..cortex.context.task_notes import handoff_policy
        if self.embodiment.handoff:
            # A cross-arm handoff policy describes nothing on an embodiment without one.
            self.capabilities["handoff_policy"] = {
                key: value for key, value in handoff_policy(self.config.features.airborne_handoff,
                    handoff_preference=self.config.features.handoff_preference).items() if key != "text"}
        limit = self.capabilities.get("action_budget", {}).get("native_step_limit")
        if type(limit) is int and limit > 0:
            self.native_budget_limit, self.native_budget_known = limit, True
        for key in ("arm_queue", "motion_cost_preview"):
            if key in self.capabilities:
                self.capabilities[key]["available"] = bool(self.capabilities[key].get("available")
                    and getattr(self.config.features, key))
        speed = self.capabilities.get("action_speed")
        if speed is not None:
            speed.setdefault("speed_scale", {})["default"] = self.config.features.default_motion_speed_scale
            # Only the shared presets inside this embodiment's speed_scale / gripper_speed_scale ranges.
            presets = dict(self.embodiment.speed_presets)
            speed.update(speed_presets=presets, gripper_speed_presets=dict(self.embodiment.gripper_speed_presets),
                         default_scope="harness_motion_request")
            if speed.get("waypoint_speed_scales", {}).get("available"):
                speed["waypoint_speed_presets"] = dict(available=True, presets=dict(presets),
                    order="via waypoints then final target; each controls its incoming segment",
                    null_semantics="inherit resolved request speed; omission of request speed uses run default",
                    mutually_exclusive_with="waypoint_speed_scales")
        if not self.enable_wait:
            self.capabilities.pop("bounded_wait", None)
        require_actor_observation(self.capabilities, self.observation_profile)
        if self.motion_review and not hasattr(self.robot, "preview_motion"):
            raise ValueError("motion_review_backend_unavailable")
        self.capabilities["motion_review_enabled"] = self.motion_review
        self.capabilities["vision_compare_enabled"] = self.vision_compare
        if self.capabilities.get("observer_video"): self.robot.record(enabled=self.config.features.record_video)
        self.store.save("configuration", "robot_capabilities-"+digest(self.capabilities), self.capabilities)
        # Audit record of the resolved embodiment, once per started run (a resume adds its own when it differs).
        record = profile_record(self.embodiment)
        self.embodiment_record_id = EMBODIMENT_RECORD_PREFIX + digest(record)
        self.store.save("configuration", self.embodiment_record_id, record)
        self.store.set_state("runtime","capabilities",self.capabilities)
        observation = self.observe()
        self.capture_home_reference(observation)
        return observation

    def capture_home_reference(self, observation):
        if self.home_check is not None:
            self.home_check.capture_initial(observation)
            self.store.event("completion_guidance.home_reference", self.home_check.reference_record())

    def action_budget(self):
        """Public execution accounting only; never score or evaluator success."""
        used = self.native_action_count if self.native_budget_known else None
        limit = self.native_budget_limit
        semantics = ("official native steps; separate from tool requests and model-call budget; reserves are not included"
                     if self.embodiment.clock == "simulated_steps"
                     else "no native step budget on this embodiment; values are null")
        return dict(native_steps_used=used, native_steps_limit=limit,
                    native_steps_remaining=max(0, limit-used) if limit is not None and used is not None else None,
                    source="native_execution_accounting", unknown_values_are_null=True,
                    per_request_native_limit=self.embodiment.per_request_step_limit,
                    semantics=semantics)

    def publish_observation(
        self,
        raw,
        correlation=None,
        last_action_id=None,
        make_current=True,
        *,
        identity=None,
        can_make_current=None,
    ):
        return self.observation_pipeline.publish_observation(
            raw,
            correlation=correlation,
            last_action_id=last_action_id,
            make_current=make_current,
            identity=identity,
            can_make_current=can_make_current,
        )

    def on_object_table_delta(self, before_observation_id, after_observation_id, delta):
        self.observation_pipeline.record_object_table_delta(before_observation_id, after_observation_id, delta)

    def observe(self, correlation=None):
        with self.robot_lock:
            return self.publish_observation(self.robot.observe(), correlation)

    def current_observation(self):
        if self.latest is None:
            raise ValueError('current_observation_unavailable')
        return self.latest

    def action(self, action_id):
        request = self.store.read("action_request", action_id)
        return dict(request=request, receipt=self.states.get(action_id, {"status": "unknown"}))

    def reset(self, seed=0, layout_id=0):
        with self.lock:
            if self.active_action: raise ValueError("reset_requires_confirmed_motion_stop")
            if not self.capabilities.get("reset"): raise ValueError("unsupported_capability:reset")
            if hasattr(self.robot,"validate_reset"): self.robot.validate_reset(seed,layout_id)
            self.accepting=False
            stop_revision=self.stop_revision
            self.cancel.clear()
        with self.robot_lock:
            with self.lock:
                previous=dict(self.identity)
                self.identity.update(episode_id=new_id("ep"),env_generation=self.identity["env_generation"]+1)
                self.episode_state.begin_episode(self.identity)
                self.latest=None
                if self.home_check is not None:
                    self.home_check.invalidate_current("environment_reset")
                self.episode_done.clear()
                self.episode_terminal = None
                self.native_action_count = 0
                self.native_budget_known = self.native_budget_limit is not None
                self.started_at = self.last_native_progress_at = time.monotonic()
                self.model_wait_at_start = self.model_wait_at_native_progress = getattr(getattr(self, "model_proxy", None), "model_wait_seconds", 0.0)
                self.failed_paths.clear()
                self._reconciled_late_actions.clear()
                self._action_stop_revisions.clear()
                self._action_intervention_revisions.clear()
                self._pending_late_replies.clear()
                self._operator_acknowledged_actions.clear()
                self._operator_acknowledged_programs.clear()
                self.internal_runtime_errors.clear()
                self.consecutive_internal_runtime_errors = 0
                self.consecutive_tool_internal_errors.clear()
                self.internal_runtime_tripped = None
                self._admitted_action_calls.clear()
            self._notify_perception_session_close(previous.get("episode_id"))
            self.store.set_state("runtime","identity",dict(self.identity))
            self.store.set_state("runtime","reset",dict(**self.identity,status="starting"))
            self.store.event("environment.reset_requested",{"previous_identity":previous,"seed":seed,"layout_id":layout_id})
            try:
                result=self.robot.reset(seed=seed,layout_id=layout_id)
                self._wire_late_reply_hook()
                if getattr(self.robot,"process",None):
                    self.store.set_state("runtime","robot",{"pid":self.robot.process.pid,"run_dir":str(self.robot.run_dir)})
                if self.capabilities.get("observer_video"): self.robot.record(enabled=self.config.features.record_video)
                observation=self.publish_observation(result["observation"])
                self.capture_home_reference(observation)
            except Exception as error:
                self.store.set_state("runtime","reset",dict(**self.identity,status="unknown",reason=str(error)))
                self.store.event("environment.reset_unknown",{"reason":str(error),"previous_identity":previous})
                raise RuntimeError("reset_acknowledgement_unknown:"+str(error)) from error
            with self.lock:
                self.accepting=self.stop_revision==stop_revision
                if self.accepting: self.cancel.clear()
            self.store.set_state("runtime","reset",dict(**self.identity,status="completed"))
            self.store.event("environment.reset",{"previous_identity":previous,"seed":seed,"layout_id":layout_id,"observation_id":observation["observation_id"]})
        return observation

    def execute(self, action, correlation=None):
        from ..robot.motion.motion_speed import resolve_action_speed
        original_speed = {key: action[key] for key in SPEED_INPUT_FIELDS if key in action}
        action = resolve_action_speed(action, self.config.features.default_motion_speed_scale, profile=self.embodiment)
        action_id = action.get("action_id")
        if not action_id:
            raise ValueError("missing_action_id")
        correlation = {**(correlation or {}), "action_id": action_id, "observation_id": action.get("based_on_observation_id")}
        with self.lock:
            try:
                old = self.store.read("action_request", action_id)
            except KeyError:
                old = None
            if old is not None:
                if digest(resolve_action_speed(old, self.config.features.default_motion_speed_scale,
                                                profile=self.embodiment)) != digest(action):
                    raise ValueError("action_id_content_conflict")
                if correlation.get("tool_call_id"):
                    self._admitted_action_calls.add((correlation.get("program_id"), correlation["tool_call_id"]))
                return self.action(action_id)
            if correlation.get("tool_call_id"):
                self._admitted_action_calls.add((correlation.get("program_id"), correlation["tool_call_id"]))
            self.store.save("action_request", action_id, action)
            self.store.event("action.requested", {"request": action}, correlation)
            self.store.event("action.speed_resolved", dict(action_id=action_id, input=original_speed,
                preview_plan_id=action.get("preview_plan_id"),
                default_motion_speed_scale=self.config.features.default_motion_speed_scale,
                effective={key: action[key] for key in SPEED_EFFECTIVE_FIELDS if key in action}), correlation)
            try:
                self.current_observation()
                if self.programs is not None:
                    unresolved_fn = getattr(self.programs, "unresolved_jobs", None)
                    if callable(unresolved_fn):
                        try:
                            has_unresolved_prog = bool(
                                unresolved_fn(identity=self.identity, acknowledged=self._operator_acknowledged_programs)
                            )
                        except TypeError:
                            has_unresolved_prog = bool(unresolved_fn())
                        if has_unresolved_prog:
                            raise ValueError("program_stop_unconfirmed")
                if action.get("type") == "ArmQueueAction" and not self.config.features.arm_queue:
                    raise ValueError("arm_queue_not_enabled_for_run")
                self.task_planning.require(self.identity)
                if action.get("type") == "JointPrefixAction":
                    validate_identity(action)
                    if any(action.get(key) != self.identity[key] for key in IDENTITY): raise ValueError("stale_environment_identity")
                    if action.get("world_kind") != "observed": raise ValueError("internal_action_cannot_execute_observed_world")
                    if action.get("based_on_observation_id") != self.latest["observation_id"]: raise ValueError("stale_observation")
                else:
                    validate_action(action, self.identity, self.latest["observation_id"], self.capabilities,
                                    profile=self.embodiment)
                validate_backend_action(self.robot, action, self.capabilities)
                if action.get("preview_plan_id"):
                    raise ValueError("motion_preview_unavailable")
                if not self.accepting:
                    raise ValueError("run_stopping_or_reconciliation_required")
                if correlation.get('program_id'):
                    if self.programs is None:
                        raise ValueError('program_channel_expired')
                    self.programs.admit_action(correlation['program_id'], action['timeout_s'])
                if self.active_action:
                    raise ValueError("action_writer_busy")
                if self.max_actions is not None and self.action_count >= self.max_actions:
                    raise ValueError("action_budget_exhausted")
                if action.get("path") and digest(action["path"]) in self.failed_paths:
                    self.store.event("action.prior_planner_failure", dict(action_id=action_id,
                        assessment_role="advisory", message="The same path previously failed planning; the actor chose to try again."), correlation)
                self.invalidate_scene(action_id)
            except ValueError as error:
                receipt = dict(status="rejected", reason=str(error), result_observation_id=None,
                               observation_unavailable_reason="action_not_executed")
                self.states[action_id] = receipt
                self.store.save("action_receipt", action_id, receipt)
                self.store.event("action.rejected", receipt, correlation)
                return self.action(action_id)
            self.cancel.clear()
            self.active_action = action_id
            self._in_flight_actions.add(action_id)
            self._action_stop_revisions[action_id] = self.stop_revision
            self._action_intervention_revisions[action_id] = self._operator_intervention_revision
            self.episode_state.action_admitted(action_id)
            if self.home_check is not None:
                self.home_check.invalidate_current("action_in_progress")
            self.active_context = correlation
            self.action_count += 1
            self.states[action_id] = dict(status="accepted")
            task_plan = self.task_planning.current(self.identity)
            self.store.event("action.accepted", {"task_plan_id": task_plan["plan_id"] if task_plan else None}, correlation)

        use_rollout_tracker = (
            self.embodiment.clock == "simulated_steps" and "grasp_angles" in self.embodiment.pose_inputs
        )
        wait_observations = []
        if use_rollout_tracker:
            from .actions.rollout_tracker import ProcessRolloutTracker
            rollout_tracker = ProcessRolloutTracker(action=action, runtime=self)
        else:
            rollout_tracker = None

        def progress(value):
            value = dict(value)
            self._native_progress(value.pop("episode_terminal", None))
            raw_observation = value.pop("observation", None)
            if raw_observation:
                # 运动中途 5Hz 过程帧仅作历史证据与多视角缓存，严禁覆写 self.latest 正式决策锚点
                published_progress = self.publish_observation(
                    raw_observation, correlation, action_id, make_current=False
                )
                value["observation_id"] = published_progress["observation_id"]
                if rollout_tracker is not None:
                    try:
                        rollout_tracker.record_progress_observation(value, published_progress)
                    except Exception:
                        pass
                steps = value.get("executed_steps", 0)
                if ("hold_native_steps" in action and steps and steps % 10 == 0
                        and steps < action["hold_native_steps"]):
                    wait_observations.append(value["observation_id"])
            stage = value.get("phase", "progress")
            self.states[action_id] = dict(status="running", progress=value)
            self.store.event("action.started" if stage == "started" else "action.progress", value, correlation)

        acknowledged_receipt = None
        try:
            with self.robot_lock:
                raw_receipt = self.robot.execute(action, progress_callback=progress, cancel_flag=self.cancel)
                episode_terminal = raw_receipt.pop("episode_terminal", None)
                terminal_raw = raw_receipt.pop("terminal_observation", None)
                before_raw = raw_receipt.pop("before_observation", None)
                acknowledged_receipt = _execution_acknowledgement(raw_receipt)
                self._native_progress(episode_terminal)
                from ..robot.motion.motion_failure import archive_collision_witnesses
                archive_collision_witnesses(self.store, raw_receipt, self.observation_profile)
                reject_privileged(raw_receipt, self.observation_profile,
                    stage_ids=[stage["id"] for stage in action["stages"]] if action.get("type") == "BatchAction" else None)
                validate_display(raw_receipt)
                receipt = dict(raw_receipt)
                # Execution is already acknowledged. Keep the public receipt
                # apart from evidence which can still fail to publish, and
                # never leave unprojected state/trace bodies in a fallback reply.
                details = {name: receipt.pop(name, None) for name in
                           ("planned_trace", "dispatched_trace", "actual_trace", "before_state")}
                acknowledged_receipt = receipt
                if "hold_native_steps" in action:
                    receipt["wait_observation_ids"] = wait_observations
                if details.get("before_state"):
                    details["before_state"] = actor_robot_state(details["before_state"], self.observation_profile,
                                                                 arms=self.embodiment.arms)
                for sample in details.get("actual_trace") or []:
                    if "state" in sample: sample["state"] = actor_robot_state(sample["state"], self.observation_profile,
                                                                                 arms=self.embodiment.arms)
                start_bundle = None
                if before_raw:
                    # Start-frame capture predates progress; keep it as historical evidence.
                    start_bundle = self.publish_observation(before_raw, correlation, make_current=False)
                    receipt["execution_start_observation_id"] = start_bundle["observation_id"]
                for name in ["planned_trace", "dispatched_trace", "actual_trace", "before_state"]:
                    value = details[name]
                    receipt[name + "_ref"] = self.store.put_artifact(canonical(value).encode(), name, "application/json") if value is not None else None
                terminal = None
                if terminal_raw:
                    terminal = self.publish_observation(terminal_raw, correlation, action_id)
                    receipt["result_observation_id"] = terminal["observation_id"]
                    if action.get("input_grasp_pose"):
                        try:
                            feedback = contact_target_feedback(action,
                                actor_robot_state(terminal_raw["robot_state"], self.observation_profile,
                                                  arms=self.embodiment.arms))
                            # Keep a diagnostic serialization failure from changing an executed action's status.
                            canonical(feedback)
                            if feedback:
                                receipt["grasp_contact_feedback"] = feedback
                        except Exception as error:
                            receipt["grasp_contact_feedback"] = dict(available=False,
                                reason="diagnostic_unavailable", error_type=type(error).__name__)
                    if self.home_check is not None:
                        self.home_check.update_current(terminal)
                else:
                    receipt.update(result_observation_id=None, observation_unavailable_reason="backend_terminal_capture_unavailable")
                    if receipt.get('status') != 'rejected' and receipt.get('physics_advanced') is not False:
                        self.latest = None
                    if self.home_check is not None:
                        self.home_check.invalidate_current("backend_terminal_capture_unavailable")
                # 独立保护运动期 4D 过程轨迹计算：绝不因可选诊断计算异常污染或吞没已确认的物理回执
                if rollout_tracker is not None:
                    try:
                        process_traj = rollout_tracker.finalize(
                            before_state=details.get("before_state"),
                            actual_trace=details.get("actual_trace"),
                            planned_trace=details.get("planned_trace"),
                            receipt=receipt,
                            start_observation=start_bundle,
                            terminal_observation=terminal,
                            store=self.store,
                        )
                        if process_traj is not None:
                            canonical(process_traj)
                            receipt["process_trajectory"] = process_traj
                    except Exception as error:
                        receipt["process_trajectory"] = dict(
                            available=False,
                            reason="diagnostic_unavailable",
                            error_type=type(error).__name__,
                        )
        except Exception as error:
            self.latest = None
            if self.home_check is not None:
                self.home_check.invalidate_current("post_execution_publication_failed"
                    if acknowledged_receipt is not None else "worker_acknowledgement_or_capture_failed")
            if acknowledged_receipt is not None:
                receipt = acknowledged_receipt
                receipt.update(result_observation_id=None,
                    observation_unavailable_reason="post_execution_observation_or_evidence_publication_failed",
                    post_execution_error=dict(phase="after_execution_acknowledgement",
                        error_type=type(error).__name__, message=str(error), action_replay_permitted=False))
                try:
                    self.store.event("action.evidence_publication_failed", receipt["post_execution_error"], correlation)
                except Exception:
                    pass
                # Missing current observation closes new-action admission until
                # fresh observe succeeds. It does not erase confirmed movement
                # or reopen a stop/unknown gate which was already closed.
            else:
                receipt = dict(status="unknown", reason=str(error), result_observation_id=None,
                               observation_unavailable_reason="worker_acknowledgement_or_capture_failed")
                self.accepting = False
        pending_late = None
        with self.lock:
            try:
                receipt["action_budget"] = self.action_budget()
                self.states[action_id] = receipt
                status = receipt.get("status", "unknown")
                if execution_unconfirmed(receipt):
                    self.accepting = False
                settled_continuous = (
                    not execution_unconfirmed(receipt)
                    and (
                        receipt.get("result_observation_id") is not None
                        or status == "rejected"
                        or receipt.get("physics_advanced") is False
                    )
                )
                self.episode_state.action_settled(
                    continuous=settled_continuous,
                    terminal_observation_id=receipt.get("result_observation_id"),
                )
                event_type = "action.status_unknown" if status in ("unknown", "status_unknown") else "action." + status
                try:
                    self.store.save("action_receipt", action_id, receipt)
                    self.store.event(event_type, receipt, correlation)
                except Exception as persist_error:
                    self.accepting = False
                    receipt["storage_persist_error"] = dict(
                        error_type=type(persist_error).__name__,
                        action_replay_permitted=False,
                    )
                    self.states[action_id] = receipt
                if action.get("path"):
                    if receipt.get("reason") == "path_infeasible":
                        self.failed_paths.add(digest(action["path"]))
                    elif receipt.get("status") == "completed":
                        self.failed_paths.clear()
            finally:
                self.active_action = None
                self.active_context = None
                self._in_flight_actions.discard(action_id)
                pending_late = self._pending_late_replies.pop(action_id, None)
        if pending_late is not None:
            try:
                self._reconcile_late_execute_reply(pending_late[0], pending_late[1])
            except Exception:
                pass
        try:
            return self.action(action_id)
        except Exception:
            return dict(request=deepcopy(action), receipt=deepcopy(self.states.get(action_id, receipt)))

    def _on_late_worker_reply(self, orphan, message):
        if not isinstance(orphan, dict):
            return None
        method = orphan.get("method") or orphan.get("op")
        params = orphan.get("params") if "params" in orphan else orphan.get("payload")
        if method != "execute" or not isinstance(params, dict):
            return None
        action_id = params.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            return None
        with self.lock:
            if action_id in self._in_flight_actions:
                self._pending_late_replies[action_id] = (deepcopy(orphan), deepcopy(message))
                return None
        return self._reconcile_late_execute_reply(orphan, message)

    on_late_reply = _on_late_worker_reply

    def _reconcile_late_execute_reply(self, orphan, message):
        params = orphan.get("params") if "params" in orphan else orphan.get("payload")
        if not isinstance(params, dict):
            return None
        action_id = params.get("action_id")
        if not isinstance(action_id, str) or not action_id:
            return None
        with self._late_reply_lock:
            with self.lock:
                state = self.states.get(action_id)
                if not isinstance(state, dict) or not execution_unconfirmed(state):
                    return None
                if isinstance(self.identity, dict):
                    try:
                        req = self.store.read("action_request", action_id)
                    except KeyError:
                        req = params
                    if not all(req.get(k) == self.identity.get(k) for k in IDENTITY):
                        return None
                started_stop_revision = self._action_stop_revisions.get(action_id, 0)
                started_intervention_revision = self._action_intervention_revisions.get(action_id, 0)
                current_identity = deepcopy(self.identity)

                def _un_superseded() -> bool:
                    return (
                        self.identity == current_identity
                        and action_id in self.states
                        and self.stop_revision == started_stop_revision
                        and self._operator_intervention_revision == started_intervention_revision
                        and self.active_action is None
                        and action_id not in self._operator_acknowledged_actions
                    )

                pre_can_current = _un_superseded()

            ok_reply = (
                isinstance(message, dict)
                and "error" not in message
                and (message.get("ok") is True if "ok" in message else "result" in message)
                and isinstance(message.get("result"), dict)
                and not getattr(self.robot, "failure", None)
            )
            result = message.get("result") if ok_reply else {}
            stop_confirmation = deepcopy(result.get("stop_confirmation")) if isinstance(result, dict) else None
            stop_confirmed_ok = (
                ok_reply
                and isinstance(stop_confirmation, dict)
                and stop_confirmation.get("confirmed") is True
                and not execution_unconfirmed(result)
            )

            obs_id = None
            if stop_confirmed_ok and isinstance(result.get("terminal_observation"), dict):
                try:
                    published = self.publish_observation(
                        deepcopy(result["terminal_observation"]),
                        correlation={"action_id": action_id, "source": "late_acknowledgement"},
                        last_action_id=action_id,
                        make_current=pre_can_current,
                        identity=current_identity,
                        can_make_current=_un_superseded,
                    )
                    obs_id = published["observation_id"]
                except Exception:
                    obs_id = None

            reopened = False
            with self.lock:
                if self.identity != current_identity or action_id not in self.states:
                    return None
                can_apply_owner_state = _un_superseded()
                if stop_confirmed_ok and obs_id is not None:
                    if can_apply_owner_state and isinstance(result.get("episode_terminal"), dict):
                        self._native_progress(deepcopy(result["episode_terminal"]))
                    self._reconciled_late_actions.add(action_id)
                    if can_apply_owner_state:
                        try:
                            self.invalidate_scene(action_id, reason="late_action_acknowledgement")
                        except ValueError:
                            pass
                        self.episode_state.action_settled(continuous=True, terminal_observation_id=obs_id)
                        if not self.episode_done.is_set() and not self.unresolved_execution():
                            self.accepting = True
                            self.cancel.clear()
                            reopened = True
                elif can_apply_owner_state:
                    self.accepting = False

                late_ack = {
                    "action_id": action_id,
                    "late_status": result.get("status") if isinstance(result, dict) else None,
                    "stop_confirmation": stop_confirmation,
                    "terminal_observation_id": obs_id,
                    "reopened_admission": reopened,
                }
                self.states[action_id]["late_acknowledgement"] = deepcopy(late_ack)
                self.store.event("action.late_acknowledgement", deepcopy(late_ack), {"action_id": action_id})
                return late_ack

    def unresolved_execution(self, *, acknowledged=()) -> bool:
        ack_set = set(acknowledged or ())
        if self.programs is not None:
            unresolved_fn = getattr(self.programs, "unresolved_jobs", None)
            if callable(unresolved_fn):
                try:
                    prog_unresolved = unresolved_fn(identity=self.identity, acknowledged=ack_set)
                except TypeError:
                    prog_unresolved = [
                        jid for jid in unresolved_fn()
                        if jid not in ack_set and f"program:{jid}" not in ack_set
                    ]
                if prog_unresolved:
                    return True
        reset = self.store.read_state("runtime", "reset")
        if (
            isinstance(reset, dict)
            and reset.get("status") == "unknown"
            and "runtime:reset" not in ack_set
            and (not isinstance(self.identity, dict) or all(reset.get(key) == self.identity.get(key) for key in IDENTITY))
        ):
            return True
        reconciled = getattr(self, "_reconciled_late_actions", ())
        for action_id, state in self.states.items():
            if action_id in reconciled or action_id in ack_set:
                continue
            if not execution_unconfirmed(state):
                continue
            if not isinstance(self.identity, dict):
                return True
            try:
                request = self.store.read("action_request", action_id)
            except KeyError:
                return True
            if all(request.get(key) == self.identity.get(key) for key in IDENTITY):
                return True
        return False

    def _native_progress(self, terminal):
        """Keep evaluator state private; publish only filtered execution budget."""
        if terminal is None:
            return
        with self.lock:
            count = terminal.get("native_action_count", 0)
            if type(count) is int and count >= 0:
                self.native_budget_known = True
            limit = terminal.get("native_step_limit")
            if type(limit) is int and limit > 0:
                self.native_budget_limit = limit
            if count > self.native_action_count:
                self.native_action_count = count
                self.last_native_progress_at = time.monotonic()
                self.model_wait_at_native_progress = getattr(getattr(self, "model_proxy", None), "model_wait_seconds", 0.0)
                self.failed_paths.clear()
            self.store.set_state("runtime", "native_episode", terminal)
            if terminal.get("ended") and not self.episode_done.is_set():
                self.episode_terminal = terminal
                self.accepting = False
                reference = self.store.put_artifact(canonical(terminal).encode(), "native_terminal", "application/json", "observer")
                self.store.event("environment.ended", {"result_ref": reference}, source="private_evaluator")
                self.episode_done.set()
                self._notify_perception_session_close(self.identity.get("episode_id"))
                if self.on_episode_end:
                    self.on_episode_end()

    def invalidate_scene(self, action_id=None, *, reason="observed_action_admitted"):
        head = self.scene.head(self.identity)
        if not head["scene_version"]:
            return
        current = self.scene.version(self.identity, head["scene_version"])
        operations = [
            dict(
                op="mark_dirty",
                entity_id=entity,
                source_kind="observation_inference",
                reason="Observed motion may change geometry; no contact extent is known.",
            )
            for entity, value in current["entities"].items()
            if value["state"] not in ("retired", "dirty")
        ]
        if not operations:
            return
        obs_ids = None
        if (
            isinstance(self.latest, dict)
            and isinstance(self.latest.get("observation_id"), str)
            and self.latest.get("calibration_id") == current.get("calibration_id")
            and self.latest.get("robot_model_id", current.get("robot_model_id")) == current.get("robot_model_id")
        ):
            try:
                imported = self.scene.observation(self.identity, self.latest["observation_id"])
                if "calib_rev" in current and imported.get("calibration_correction") is not None:
                    from ..robot.model.calibration_correction import CalibrationCorrection
                    rev = CalibrationCorrection.from_mapping(imported["calibration_correction"]).calib_rev
                    if rev >= current["calib_rev"]:
                        obs_ids = [self.latest["observation_id"]]
                else:
                    obs_ids = [self.latest["observation_id"]]
            except Exception:
                obs_ids = None
        if not obs_ids:
            recorded_ids = list(current.get("observation_ids") or [])
            loaded = []
            for oid in recorded_ids:
                try:
                    loaded.append(self.scene.observation(self.identity, oid))
                except Exception:
                    continue
            if loaded:
                best = max(loaded, key=lambda item: (isinstance(item.get("sim_step"), int), item.get("sim_step") or 0))
                obs_ids = [best["observation_id"]]
            else:
                obs_ids = recorded_ids[-1:]
        if not obs_ids:
            return
        base_version = head["scene_version"]
        result = None
        try:
            for start in range(0, len(operations), 64):
                patch = dict(
                    schema_version="0.1",
                    **self.identity,
                    world_kind="internal",
                    base_scene_version=base_version,
                    observation_ids=obs_ids,
                    builder_digest=digest("observed-action-invalidation-v1"),
                    operations=operations[start : start + 64],
                )
                result = self.scene.apply_patch(patch)
                base_version = result["scene_version"]
            if result is not None:
                correlation = {"action_id": action_id} if action_id is not None else {}
                self.store.event(
                    "scene.invalidated",
                    {"scene_version": result["scene_version"], "reason": reason},
                    correlation,
                    world_kind="internal",
                )
        except ValueError as error:
            # Admission fails closed if a concurrent scene commit prevents invalidation.
            raise ValueError("scene_invalidation_conflict:" + str(error)) from error

    def _clear_motion_ephemerals(self):
        for slot in (
            PERSISTENT_OBSTACLES_SLOT,
            OBS3D_MEMO_SLOT,
            OBS3D_HISTORY_SLOT,
            PENDING_OBSERVE_OBS3D_SLOT,
            PENDING_ACTION_OBS3D_SLOT,
            VIRTUAL_FIELD_SLOT,
            VIRTUAL_FIELD_ERROR_SLOT,
            WRIST_PROGRESS_CACHE_SLOT,
            HELD_PERCEPTION_SLOT,
            POSE_SKILL_GRASPS_SLOT,
        ):
            existing = self.episode_state.peek(slot, None)
            if isinstance(existing, (dict, list, set)):
                existing.clear()
            slot_reset(self, slot)
        self.failed_paths.clear()
        if self.home_check is not None:
            if callable(getattr(self.home_check, "clear", None)):
                self.home_check.clear()
            if callable(getattr(self.home_check, "invalidate_current", None)):
                self.home_check.invalidate_current("external_motion")

    def _can_observe(self) -> bool:
        if "observe" in getattr(self, "__dict__", {}) and callable(self.__dict__["observe"]):
            return True
        if type(self).observe is not Harness.observe:
            return True
        return callable(getattr(self.robot, "observe", None))

    def external_motion(self, phase: str, source: str, *, result=None, refresh_observation: bool = True):
        """Owner-side state transaction for operator/teleop/calibration motion outside Actor ToolSpec."""
        if phase not in ("begin", "end"):
            raise ValueError("invalid_external_motion_phase")
        if not isinstance(source, str) or not source:
            raise ValueError("invalid_external_motion_source")
        if phase == "begin":
            # Preemptive relax/hold MUST NEVER acquire robot_lock behind a wedged worker.
            with self.lock:
                self.accepting = False
                self._operator_intervention_revision += 1
                self.batch_plan_revision += 1
                plans = self.episode_state.peek(UNIFIED_ACTION_PLANS_SLOT, None)
                if isinstance(plans, dict):
                    for rec in plans.values():
                        if isinstance(rec, dict) and isinstance(rec.get("summary"), dict):
                            rec["summary"]["plan_revision"] = int(rec["summary"].get("plan_revision", 0)) + 1
                try:
                    self.invalidate_scene(reason=f"external_motion:{source}")
                except ValueError:
                    pass
                self._clear_motion_ephemerals()
                self.latest = None
                self.episode_state.action_settled(continuous=False)
            self.store.event("harness.external_motion", {"phase": "begin", "source": source})
            return {"phase": "begin", "source": source}

        with self.lock:
            self._operator_intervention_revision += 1
            try:
                self.invalidate_scene(reason=f"external_motion:{source}")
            except ValueError:
                pass
            self._clear_motion_ephemerals()
            self.latest = None
            self.episode_state.action_settled(continuous=False)
        observation = None
        if refresh_observation and not execution_unconfirmed(result or {}) and self._can_observe():
            observation = self.observe({"actor": "operator", "source": source})
        obs_id = observation.get("observation_id") if isinstance(observation, dict) else None
        self.store.event(
            "harness.external_motion",
            {"phase": "end", "source": source, "observation_id": obs_id},
        )
        return {"phase": "end", "source": source, "observation_id": obs_id, "observation": observation}

    def _review_controls(self, correlation):
        identifier = correlation.get('program_id')
        if identifier is None:
            return dict(cancel_event=self.cancel)
        if self.programs is None:
            raise ValueError('program_channel_expired')
        return self.programs.review_controls(identifier)

    def _model_context_epoch(self):
        return self.decision_context.model_context_epoch()

    def model_context_reset(self, reason):
        """Delegate model-context reset tracking and sideband reminder scheduling to ``decision_context``."""
        self.decision_context.model_context_reset(reason)

    def _deliver_decision_reminder(self, response, correlation):
        """Append the due decision-context reminder to one actor tool ``response`` (design 04 §4.5)."""
        return self.decision_context.deliver_decision_reminder(response, correlation)

    def observation_content(self, bundle, correlation, camera_id=None, roi=None, scale=1, numeric_only=False,
                            *, feedback_action=None, feedback_receipt=None, detail=None,
                            feedback_context=None, full_view_reason=None, metadata_delivery=True,
                            include_images=True, structured=False):
        return self.observation_pipeline.observation_content(
            bundle,
            correlation,
            camera_id=camera_id,
            roi=roi,
            scale=scale,
            numeric_only=numeric_only,
            feedback_action=feedback_action,
            feedback_receipt=feedback_receipt,
            detail=detail,
            feedback_context=feedback_context,
            full_view_reason=full_view_reason,
            metadata_delivery=metadata_delivery,
            include_images=include_images,
            structured=structured,
        )

    def _exceeds_host_truncation_limit(self, response):
        """检查已整形回复是否超出宿主本地截断阈值；超限时须立刻失效去重基线，防止后续调用引用被截断内容。"""
        limit_fn = self.host_truncation_limit
        if not callable(limit_fn):
            return False
        limit = limit_fn()
        if not limit:
            return False
        max_bytes, max_lines, text_only_without_images = limit
        if max_bytes is None and max_lines is None:
            return False
        items = response.get("contentItems") if isinstance(response, dict) else None
        if not isinstance(items, list):
            return False
        has_images = any(isinstance(item, dict) and item.get("type") == "inputImage" for item in items)
        texts = [
            item["text"]
            for item in items
            if isinstance(item, dict) and item.get("type") == "inputText" and isinstance(item.get("text"), str)
        ]
        if not texts:
            return False
        if max_bytes is not None and not (text_only_without_images and has_images):
            if sum(len(text.encode("utf-8")) for text in texts) > max_bytes:
                return True
        if max_lines is not None:
            if sum(len(text.splitlines()) for text in texts) > max_lines:
                return True
        return False

    def handle_tool(self, params):
        if params.get("threadId") != self.thread_id:
            raise ValueError("tool_call_thread_mismatch")
        name, args = params["tool"], params["arguments"]
        if not isinstance(args, dict):
            raise ValueError("tool_arguments_must_be_object")
        context_scope = params.get("contextScope", "main")
        if context_scope not in ("main", "isolated"):
            context_scope = "isolated"
        context_epoch = params.get("contextEpoch")
        correlation = dict(
            thread_id=params["threadId"],
            turn_id=params["turnId"],
            tool_call_id=params["callId"],
            item_id=self.call_items.get(params["callId"], params["callId"]),
            context_scope=context_scope,
        )
        if context_epoch is not None:
            correlation["context_epoch"] = context_epoch
        if name in {'program_run', 'program_status'}:
            correlation['actor_log_delivery'] = True
        self.store.save("tool_call", params["callId"], dict(params=params, correlation=correlation,
                        tool_call_id_source=params.get("callIdSource", "native_callId")))
        # 在 restore(args) 展开短引用前保留内联源码字面原文，供沙盒落盘且不污染持久化事件与模型可见载荷
        invoke_correlation = (
            {**correlation, "_raw_inline_source": args["source"]}
            if name == "program_run" and isinstance(args.get("source"), str)
            else correlation
        )
        try:
            result = self.invoke(name, self.actor_output.restore(args), invoke_correlation, role="actor")
            # A read of stored evidence in full (history_read_trace, any detail="full") is delivered with its
            # original numbers: no display rounding (lean_v2) and no compact encoding.
            exact = name == "history_read_trace" or args.get("detail") == "full"
            response = self.actor_output.present(result, tool=name, exact=exact)
        except Exception:
            self._invalidate_actor_delivery(params['callId'])
            raise
        response = self._shape_actor_reply(response, getattr(result, "kind", None), correlation, exact=exact)
        if not response.get('success', False) or (
            context_scope == "main" and self._exceeds_host_truncation_limit(response)
        ):
            self._invalidate_actor_delivery(params['callId'])
        return self._deliver_decision_reminder(response, correlation)

    def _shape_actor_reply(self, response, kind, correlation, exact=False):
        """Optional shaping of one presented actor reply; with every switch off the reply is returned untouched.

        Fixed order (``docs/development/tool-output-contract.md``, "Reply shaping order"):

        1. ``features.reply_static_once``: ``drop_static_once`` leaves the listed fixed notes out of a successful,
           routine reply. First, because it is decided on the presented values and frees budget.
        2. ``kind="action"`` only: ``fit_reply`` lays the reply out by decision value when
           ``features.budgeted_action_reply`` is on (every backend; blocks are replaced until step 3's text fits
           only when the host has a known text limit) or ``features.action_reply_profile="summary_first"`` (fixed
           shape, with or without a limit).
        3. On every text item of the final layout (and of each layout step 2 weighs): ``unify_detail_entries``
           (``reply_static_once``), ``features.reply_compact_encoding``, ``reply_same_as``, ``reply_float32``.
           Detail entries and the compact encoding change the structure, so they run before ``same_as``: a pointer
           is computed on the final structure of the text item a value ends up in and never names text that was
           moved to another item, replaced by a reference or re-encoded.

        ``exact`` (a full read of stored evidence): step 3 skips the compact encoding, whose 4-digit derived
        quantities are a display form; ``same_as`` and ``float32`` are lossless and still apply.

        Presentation only: a failure in a step delivers the reply as the previous step left it and never hides a
        receipt of an action that already ran.
        """
        features = self.config.features
        items = response.get('contentItems')
        if not isinstance(items, list):
            return response
        steps = []
        if features.reply_static_once:
            if response.get('success', False):
                try:
                    items = drop_static_once(items, self._static_once_plan())
                except Exception:
                    pass
            try:
                loaded = self.tool_registry.loaded_tool_names(self, role='actor')
            except Exception:
                loaded = None
            if loaded is not None:
                steps.append(lambda content: unify_detail_entries(content, loaded))
        if features.reply_compact_encoding and not exact:
            steps.append(encode_compact_items)
        if features.reply_same_as or features.reply_float32:
            steps.append(lambda content: compact_content_items(
                content, same_as=features.reply_same_as, float32=features.reply_float32))

        def finish(content):
            for step in steps:
                content = step(content)
            return content

        profile = features.action_reply_profile
        if kind == "action" and (features.budgeted_action_reply or profile != "full"):
            fitted = self._fit_action_reply(items, correlation, finish if steps else None, profile)
            if fitted is not None:
                return dict(response, contentItems=fitted)
        try:
            items = finish(items)
        except Exception:
            pass
        return response if items is response['contentItems'] else dict(response, contentItems=items)

    def _static_once_plan(self):
        """The fixed notes this run's catalogue states (``ToolRegistry.static_once_plan``), computed once the robot
        backend's capabilities are known."""
        plan = getattr(self, '_static_once_plan_cache', None)
        if plan is None:
            plan = self.tool_registry.static_once_plan(self, role='actor')
            if self.capabilities is not None:
                self._static_once_plan_cache = plan
        return plan

    def _fit_action_reply(self, items, correlation, compact=None, profile="full"):
        """``fit_reply`` of one action reply's content items, or None when it did not apply (no execution result or
        an error; the caller then delivers the items without layout). With ``features.budgeted_action_reply`` the
        reply is always laid out, and fitted to the host's text limit when the host has a known one."""
        try:
            from ..core.presentation.reply_budget import fit_reply_report
            budgeted = self.config.features.budgeted_action_reply
            budget = self.action_reply_budget() if budgeted and self.action_reply_budget is not None else None
            fitted, report = fit_reply_report(items, budget, compact=compact, profile=profile, layout=budgeted)
        except Exception as error:
            report, fitted = dict(applied=False, error_type=type(error).__name__), None
        if report.get("by_reference") or report.get("fits") is False or "error_type" in report:
            try:
                self.store.event("actor_reply.fitted", report, correlation, world_kind="internal")
            except Exception:
                pass
        return fitted if report["applied"] else None

    def _invalidate_actor_delivery(self, call_id):
        self.decision_context.invalidate_actor_delivery(call_id)

    def robot_information(self, section="default", robot_id=None):
        """One interface for the native geometry, static reference and calibration."""
        from ..robot.model.robot_information import robot_information, robot_resource_path
        from ..robot.model.robot_calibration_information import robot_calibration_information
        current = (self.capabilities or {}).get("robot_model_id") or "unknown"
        selected = robot_id if robot_id is not None else current
        if section in {"default", "calibration"}:
            calibration = (robot_calibration_information(self) if selected == current else
                           dict(available=False, reason="calibration_robot_mismatch"))
        if section == "calibration":
            result = dict(robot_id=selected, section=section,
                available=calibration["available"], calibration=calibration)
        else:
            result = robot_information(selected, section)
            if section == "default":
                result["calibration"] = calibration
                if result.get("available"):
                    geometry = result["reference"]["geometry"]
                    path = robot_resource_path(selected, geometry["resource_id"])
                    if path is None:
                        raise ValueError("robot_geometry_archive_unavailable")
                    geometry["resource_ref"] = self.store.put_artifact(
                        path.read_bytes(), "robot_geometry_model", "application/zip")
        # This API also serves startup directly, before the registry boundary.
        archived = self._archive_tool_result("robot_information", ToolOutput(result).response(),
            dict(robot_id=selected, section=section), role="internal")
        if archived.contract_error is not None:
            result = dict(result, output_contract_error=archived.contract_error)
        return self.actor_output.present(result, projection="robot_information")

    def _scheduling_advisory(self, receipt, correlation):
        if (correlation.get("program_id") or not self.config.features.task_scheduling
                or not self.config.features.manipulation_protocol):
            return None
        from ..cortex.context.task_planning import scheduling_failure_advisory
        # Do not bind an immutable action replay to a newer mutable plan. The
        # concise advice asks the actor to reassess its current commitments.
        return scheduling_failure_advisory(receipt)

    def unified_action_postprocess(self, result, correlation, *, settle=None):
        if settle is None or result["receipt"].get("status") != "completed":
            return result
        from .actions.stability import bounded_settle
        result = dict(result)
        settle_correlation = dict(
            correlation or {},
            internal_subaction="bounded_settle",
            parent_action_id=result["receipt"].get("action_id"),
        )
        result["settling"] = bounded_settle(self, settle, settle_correlation)
        result["current_observation_id"] = self.latest["observation_id"] if self.latest else None
        result["result_observation_is_current"] = (
            result["receipt"].get("result_observation_id") == result["current_observation_id"])
        return result

    def _reset_consecutive_internal_errors(self, name: str):
        with self.lock:
            self.consecutive_tool_internal_errors[name] = 0
            if name not in _PASSIVE_STATUS_TOOLS:
                self.consecutive_internal_runtime_errors = 0

    def record_internal_runtime_error(self, name: str, error: BaseException, correlation, *, role: str):
        orig_type, orig_msg = _extract_internal_error_info(error)
        corr = dict(correlation or {})
        entry = dict(
            tool=name,
            role=role,
            error_type=orig_type,
            message=(orig_msg or str(error))[:1024],
            native_action_count=self.native_action_count,
            call_id=corr.get("tool_call_id"),
            turn_id=corr.get("turn_id"),
            program_id=corr.get("program_id"),
        )
        tripped_now = None
        with self.lock:
            self.internal_runtime_errors.append(entry)
            self.consecutive_internal_runtime_errors += 1
            tool_streak = self.consecutive_tool_internal_errors.get(name, 0) + 1
            self.consecutive_tool_internal_errors[name] = tool_streak
            consec_limit = getattr(self, "max_internal_runtime_errors", 3)
            total_limit = max(consec_limit, getattr(self, "max_total_internal_runtime_errors", 5))
            if (
                type(consec_limit) is int
                and consec_limit > 0
                and self.internal_runtime_tripped is None
                and (
                    self.consecutive_internal_runtime_errors >= consec_limit
                    or tool_streak >= consec_limit
                    or len(self.internal_runtime_errors) >= total_limit
                )
            ):
                self.internal_runtime_tripped = dict(
                    entry,
                    consecutive_errors=self.consecutive_internal_runtime_errors,
                    tool_consecutive_errors=tool_streak,
                    total_errors=len(self.internal_runtime_errors),
                )
                self.accepting = False
                tripped_now = deepcopy(self.internal_runtime_tripped)
        try:
            self.store.event("tool.internal_runtime_error", entry, corr)
        except Exception:
            pass
        if tripped_now is not None:
            try:
                self.store.event("runtime.internal_error_circuit_tripped", tripped_now, corr)
            except Exception:
                pass
            if self.programs is not None:
                cancel_all = getattr(self.programs, "cancel_all", None)
                if callable(cancel_all):
                    try:
                        cancel_all()
                    except Exception:
                        pass
        return entry

    def _action_admitted_for_call(self, tool_call_id, program_id=None):
        if not isinstance(tool_call_id, str) or not tool_call_id:
            return False
        with self.lock:
            if (
                (program_id, tool_call_id) in self._admitted_action_calls
                or (None, tool_call_id) in self._admitted_action_calls
            ):
                return True
            if (self.active_context or {}).get("tool_call_id") == tool_call_id:
                return True
        try:
            cursor = 0
            while page := self.store.events_by_type(("action.requested",), after=cursor, limit=500):
                for ev in page:
                    corr = ev.get("correlation") or {}
                    if corr.get("tool_call_id") == tool_call_id and (
                        program_id is None or corr.get("program_id") == program_id
                    ):
                        return True
                cursor = page[-1]["server_seq"]
        except Exception:
            pass
        return False

    def enable_function_tool_search(self):
        """Enable provider-agnostic function ``tool_search`` and enforce unlock-before-call on deferred actor tools."""
        if TOOL_SEARCH_NAME not in self.tool_registry.names:
            from .tools.tool_search import make_tool_search_tool

            self.tool_registry.register(make_tool_search_tool())
        self.deferred_tool_loading = True
        self.require_tool_search_unlock = True
        return self

    def eager_actor_tools(self):
        """Initial actor ``tools`` array (excludes ``deferLoading=True`` tools when ``deferred_tool_loading`` is on)."""
        return self.tool_registry.eager_catalog(self, role="actor")

    def on_demand_tools_section(self):
        """Render ``## Tools loaded on demand`` section for deferred actor tools, or ``None`` if none are deferred."""
        from .tools.catalog_guide import on_demand_index

        return on_demand_index(self.tool_registry.catalog(self, role="actor"))

    def invoke(self, name, args, correlation, *, role):
        if (
            role == "actor"
            and self.deferred_tool_loading
            and self.require_tool_search_unlock
            and (correlation or {}).get("tool_call_id") != "initialize-capabilities"
        ):
            spec = self.tool_registry.spec(name, role=role)
            if self.tool_registry.is_deferred_tool(spec, self, role=role) and name not in self.unlocked_deferred_tools:
                raise tool_errors.ToolInputError(
                    "deferred_tool_not_loaded",
                    field_path="/tool",
                    actual=name,
                    suggestion=(
                        f"Tool {name!r} is loaded on demand. Call tool_search first "
                        f"(for example {{'names': [{name!r}]}}), then invoke {name!r}."
                    ),
                    minimal_example={"tool": TOOL_SEARCH_NAME, "arguments": {"names": [name]}},
                )
        try:
            result = self.tool_registry.invoke(name, args, correlation, runtime=self, role=role)
        except Exception as error:
            if is_internal_runtime_error_compat(error):
                wrapped = _wrap_internal_runtime_error(name, error)
                if role in {"actor", "program"}:
                    self.record_internal_runtime_error(name, error, correlation, role=role)
                failure = _internal_error_payload(wrapped, name, args)
                err_resp = ToolResponse([StructuredText(failure)], data=failure, kind="error", success=False)
                self._archive_tool_result(name, err_resp, correlation, role=role)
                raise wrapped from error
            raise
        finally:
            # 每次工具调用返回时清空同轮瞬态 obs3d 预取槽位，严防未消费槽位残留至后续调用
            slot_reset(self, PENDING_OBSERVE_OBS3D_SLOT)
            slot_reset(self, PENDING_ACTION_OBS3D_SLOT)
        if role in {"actor", "program"}:
            res_data = getattr(result, "data", None)
            if not (isinstance(res_data, dict) and res_data.get("code") == "internal_runtime_error"):
                self._reset_consecutive_internal_errors(name)
        return self._archive_tool_result(name, result, correlation, role=role)

    def _archive_tool_result(self, name, result, correlation, *, role):
        """Keep the unprojected reply locally before any actor presentation.

        Large native media stay in an immutable observer artifact. Each return
        has its own record, even when an idempotent call ID is read repeatedly.
        A storage defect after execution must never become a replayable input
        error or erase the business receipt.
        """
        record = dict(tool=name, role=role, correlation=correlation)
        payload, errors = {}, []
        original = getattr(result, "raw_response", result)
        fields = (dict(kind=original.kind, data=original.data, success=original.success)
            if isinstance(original, ToolResponse) else {})
        if isinstance(result, ToolResponse):
            fields["contract_error"] = result.contract_error
        for key, value in fields.items():
            try:
                validate_display(value)
                payload[key] = value
            except Exception as error:
                errors.append(dict(field_path="/" + key, reason="not_json_serializable",
                    error_type=type(error).__name__))
                if key == "data" and isinstance(value, dict):
                    # Keep valid top-level business fields, especially receipts,
                    # without inventing a representation for arbitrary objects.
                    payload[key] = {}
                    for field, part in value.items():
                        try:
                            validate_display({field: part})
                            payload[key][field] = part
                        except Exception:
                            errors.append(dict(field_path="/data/" + (field if isinstance(field, str) else "<non_string_key>"),
                                reason="not_json_serializable"))
        try:
            wire = wire_response(original)
            validate_display(wire)
            payload["wire_response"] = wire
        except Exception as error:
            errors.append(dict(field_path="/wire_response", reason="not_json_serializable",
                error_type=type(error).__name__))
        if errors:
            payload["archival_errors"] = errors
        record["archival_status"] = "partial" if errors else "complete"
        identifier = new_id("tool-result")
        try:
            record["raw_ref"] = self.store.put_artifact(canonical(payload).encode(),
                "tool_result", "application/json", "observer")
            self.store.save("tool_result", identifier, record)
        except Exception as error:
            record.update(archival_status="failed", error_type=type(error).__name__,
                reason="raw_result_archival_failed")
            try:
                self.store.save("tool_result", identifier, record)
            except Exception:
                pass  # The actor warning is still available when the Store is unwritable.
            return output_contract_error(result, name, "raw_result_archival_failed", "/archive")
        return result

    def dispatch(self, name, args, correlation):
        """Trusted in-process compatibility entry; actors/programs use invoke."""
        return wire_response(self.invoke(name, args, correlation, role="trusted_legacy"))

    def program_services(self):
        """Bind program permissions and atomic action ownership on the server."""
        from .programs.runner import ProgramServices
        from ..perception.external.scene3d import workspace_modules as scene3d_workspace_modules

        def validate_tool(name):
            self.tool_registry.spec(name, role='program')

        def validate_arguments(name, arguments):
            self.tool_registry.validate(name, arguments, role='program', runtime=self)

        def observation():
            with self.lock:
                return dict(observation_id=self.latest['observation_id'] if self.latest else None,
                    observation_profile=self.observation_profile)

        def active(identifier, *, cancel=False):
            with self.lock:
                owns_action = (self.active_context or {}).get('program_id') == identifier
                if owns_action and cancel:
                    self.cancel.set()
                return owns_action

        return ProgramServices(store=self.store, validate_tool=validate_tool,
            validate_arguments=validate_arguments,
            invoke=lambda name, arguments, correlation: self.invoke(name, arguments, correlation, role='program'),
            effect=lambda name, arguments: self.tool_registry.spec(name, role='program').effect_for(arguments),
            observation=observation, action_active=active,
            cancel_action=lambda identifier: active(identifier, cancel=True),
            sandbox_enabled=self.config.features.sandbox_enabled,
            seccomp_enabled=getattr(self.config.features, "program_seccomp_enabled", False),
            programs_dir=getattr(self.config.features, "programs_dir", None),
            workspace_modules=scene3d_workspace_modules(self), embodiment=self.embodiment,
            current_identity=lambda: deepcopy(self.identity) if isinstance(self.identity, dict) else None,
            action_admitted=self._action_admitted_for_call)

    def policy_step(self, args, correlation):
        if "dual_arm_progression" in args:
            raise ValueError("dual_arm_progression_requires_dual_arm_motion")
        request = dict(observation_id=args["observation_id"], prefix=args.get("prefix", 5),
                       prompt=args.get("prompt"), timeout_s=args.get("timeout_s", 120))
        with self.policy_lock:
            try:
                old = self.store.read("action_request", args["action_id"])
            except KeyError:
                old = None
            if old is not None:
                if old.get("type") != "JointPrefixAction" or old.get("policy_input") != request:
                    raise ValueError("action_id_content_conflict")
                return self.action(args["action_id"])
            self.task_planning.require(self.identity)
            if self.action_policy is None: raise ValueError("pi05_policy_not_enabled_for_run")
            if request["observation_id"] != self.current_observation()["observation_id"]: raise ValueError("stale_observation")
            prefix, prompt = request["prefix"], request["prompt"]
            if type(prefix) is not int or not 1 <= prefix <= 50: raise ValueError("invalid_policy_prefix")
            if not isinstance(prompt, str) or not prompt.strip() or len(prompt) > 2000:
                raise ValueError("pi05_local_operation_prompt_required")
            with self.lock:
                if not self.accepting: raise ValueError("run_stopping_or_reconciliation_required")
            prediction = self.action_policy.infer(self.store.root, request["observation_id"], prompt=prompt)
            provenance = self.store.put_artifact(canonical(prediction).encode(), "policy_prediction", "application/json", "observer")
            action = dict(schema_version="0.1", **self.identity, world_kind="observed", type="JointPrefixAction",
                action_id=args["action_id"], based_on_observation_id=request["observation_id"],
                actions=prediction["actions"][:prefix], timeout_s=request["timeout_s"], policy_input=request,
                policy_provenance=dict(prediction_ref=provenance, prefix=prefix))
            return self.execute(action, correlation)

    def _action_from_args(self, name, args):
        from ..robot.motion.action_compilation import compile_action
        kind = {"robot_wait": "wait"}[name]
        return compile_action(getattr(self, "robot", None), kind, args, self.identity, self.capabilities)

    def reopen_admission(self, correlation=None):
        """Operator resume after a stop: publish a fresh observation, then reopen.

        Only an explicit operator command calls this. Admission reopens only
        when no action is active and no newer stop arrived while observing, so
        a concurrent stop/hold always wins. The caller owns unknown-receipt
        acknowledgement; this method never rewrites any receipt.
        """
        with self.lock:
            if self.active_action:
                raise ValueError("resume_requires_confirmed_motion_stop")
            if self.episode_done.is_set():
                raise ValueError("native_episode_ended")
            stop_revision = self.stop_revision
        observation = self.observe(correlation)
        with self.lock:
            if self.stop_revision != stop_revision or self.active_action:
                raise ValueError("resume_conflict_stop_requested_during_observation")
            self._operator_intervention_revision += 1
            self.accepting = True
            self.cancel.clear()
        self.store.event("run.admission_reopened", {"observation_id": observation["observation_id"]}, correlation)
        return observation

    def request_stop(self):
        with self.lock:
            self.accepting = False
            self.stop_revision += 1
            self.cancel.set()
            if self.active_action:
                self.store.event("action.stop_requested", {}, {"action_id": self.active_action})
            episode_id = (self.identity or {}).get("episode_id")
        self._notify_perception_session_close(episode_id)

    def mark_perception_session_active(self, session_id):
        """当外部感知再次携带某 session_id 发起请求（如终态动作后的尾步观测）时，清除已关闭标记以便收尾彻底回收。"""
        if not isinstance(session_id, str) or not session_id:
            return
        with self.lock:
            self._closed_perception_sessions.discard(session_id)

    def _notify_perception_session_close(self, session_id):
        """当旧 Episode 结束且开启外部感知时，短超时非阻塞通知回收显存会话（失败静默忽略）。"""
        if not isinstance(session_id, str) or not session_id:
            return
        with self.lock:
            if session_id in self._closed_perception_sessions:
                return
            self._closed_perception_sessions.add(session_id)
        from .actions.rollout_tracker import external_perception_endpoint, notify_session_close
        endpoint = external_perception_endpoint(self)
        if endpoint:
            notify_session_close(endpoint, session_id)

    def close(self):
        """幂等关闭运行时，停止动作准入并通知外部感知服务释放当前 Episode 会话。"""
        self.request_stop()
