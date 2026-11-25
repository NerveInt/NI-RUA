"""Backend-neutral agent session contract (design 04 §2.2-§2.3, §5.2).

An agent session drives one native agent runtime (Codex today; Claude Code and OpenCode later) for one run. The
contract is the one ``CodexSession`` has always had:

* Every public method is **synchronous** and is called from the gateway's asyncio loop through
  ``asyncio.to_thread``; methods are mutually thread-safe. Tool handlers never run on the backend's reader thread,
  and ``interrupt()`` stays responsive while tools run.
* One native thread per session and at most one active turn. ``run_turn`` returns as soon as the backend has
  acknowledged admission, never at completion. Completion is observed through ``turn_done`` (a
  ``threading.Event``), ``state`` and ``turn_result``.
* ``state`` is a plain ``str`` (see the ``STATE_*`` constants); there is no Enum because sessions and test fakes
  assign strings.

Optional backend features are declared by ``AgentCapabilities`` flags; callers consult the flags instead of
probing backend-specific attributes. ``AgentSession`` defines **no ``__init__``** and only class-level defaults so
that subclasses (and tests building ``CodexSession.__new__`` objects) keep full control of instance state.

None of the provenance data defined here (identity record, ``provenance()`` block) is model-visible; it is written
to the run Store and ``manifest.json`` for observers only.
"""
from dataclasses import asdict, dataclass, fields

from .backend import _Hook
from .identity import _codex_identity, read_agent_identity  # noqa: F401 (historical re-export)

# ---- session states (plain strings, design §2.2) --------------------------------------------------------------
STATE_NOT_STARTED = "not_started"     # constructed; no native session yet
STATE_IDLE = "idle"                   # start() succeeded; no turn yet, or resumed at a clean boundary
STATE_RUNNING = "running"             # run_turn() before admission, native turn start, or recovered in-flight turn
STATE_COMPLETED = "completed"         # native terminal of the current turn
STATE_INTERRUPTED = "interrupted"     # native terminal of the current turn
STATE_FAILED = "failed"               # native terminal of the current turn
STATE_DISCONNECTED = "disconnected"   # transport loss or unknown admission; motion admission already closed
STATE_CLOSED = "closed"               # close() released every owned resource
STATE_REPLAY = "replay"               # read-only viewer (enter_replay); no native session
STATE_UNKNOWN = "unknown"             # reserved, never assigned; treated like "disconnected"

TURN_TERMINAL_STATES = frozenset({STATE_COMPLETED, STATE_INTERRUPTED, STATE_FAILED})
# run_turn refuses these: the native state is unknowable or released.
UNCONNECTED_STATES = frozenset({STATE_DISCONNECTED, STATE_UNKNOWN, STATE_CLOSED})


# ---- errors ---------------------------------------------------------------------------------------------------
class AgentError(Exception):
    """Base class of backend-neutral agent session errors."""


class AgentRejected(AgentError):
    """Definite rejection: the backend explicitly refused the request and nothing was admitted.

    ``code`` is the backend's own rejection code (JSON-RPC code, HTTP status, ...; any JSON scalar), ``message`` its
    human-readable reason and ``busy`` is True when the backend reported overload (the caller may retry later;
    gateway control maps it to HTTP 429). The session restores its previous state before raising, so a later
    request is safe. Codex keeps raising the SDK ``JsonRpcError`` instead (same ``code``/``message`` attributes).
    """

    def __init__(self, code, message, busy=False):
        super().__init__(f"{code}: {message}")
        self.code, self.message, self.busy = code, message, bool(busy)


class AgentTransportUnknown(AgentError):
    """The request may have reached the backend but no acknowledgement arrived (unknown admission or lost link).

    Never retried automatically: the session has already called ``_disconnected`` (state ``disconnected``, motion
    admission closed through ``on_disconnect``). Gateway control maps it to status ``unknown`` (HTTP 409).
    """


class AgentCapabilityError(ValueError):
    """The selected backend lacks a feature (for example ``steer_unsupported_by_backend``).

    A ``ValueError`` so the gateway answers it as a plain 400 refusal; ``str(error)`` is the stable error code
    followed by an optional explanation.
    """


# ---- capabilities ---------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentCapabilities:
    """Optional features of one agent backend. Callers gate behaviour on these flags, never on the backend name.

    Every flag defaults to False (feature absent); ``compaction_signal`` defaults to ``"relay"``.
    """
    steer: bool = False                  # operator text can join the ACTIVE turn (control command "steer")
    native_requests: bool = False        # operator reply() answers native approvals/questions
    terminal_attach: bool = False        # a live read-only native TUI can attach (features.terminal)
    native_terminal_replay: bool = False  # the library can re-render the native TUI offline
    fork: bool = False                   # native thread fork (apps/evaluation/checkpoint.py, turn_probe.py)
    deferred_tools: bool = False         # native on-demand discovery of deferLoading tools
    account_relay: bool = False          # ChatGPT account-control relay
    native_evidence_agent: bool = False  # the backend has its own agent-based evidence helper (Codex only)
    sandbox: bool = False                # features.sandbox_enabled=true is supported
    service_tier: bool = False           # operator service-tier knob (Responses service_tier)
    native_memories: bool = False        # native cross-turn memory extraction (model.native_memories)
    # True: set_effort() reaches the running native session and applies from its next turn (operator console reports
    # applies="next_codex_turn"). False: the native runtime fixes effort at session start, so a new value takes effect
    # only when the session is next started (resume); the console reports applies="next_session_start".
    live_effort: bool = False
    # True: the native runtime may start a turn by itself while the host turn is terminal or idle (a background shell
    # or monitor finishing, a scheduled wake-up). The session adopts it as a NI-RUA turn (agent.turn_started
    # {origin: "native_wakeup"}, new turn_id, turn_done cleared, ends at the next native idle), and a run_turn() issued
    # while such an adopted turn runs joins it (the prompt is delivered into it; returns its turn_id) instead of raising
    # turn_already_active. False: every turn is started by run_turn() (design R4.2).
    native_wakeup: bool = False
    # How context compaction is detected: "native+relay" (native event plus relay continuity check), "relay"
    # (relay continuity check only) or "none" (no usable signal; the run uses forced full delivery).
    compaction_signal: str = "relay"
    # 支持在 run_turn 中随首轮/回合文本一并附带内联多模态图片输入。
    turn_images: bool = False

    def to_dict(self):
        """Plain ``{flag: value}`` dict for provenance records."""
        return asdict(self)


def model_effort_choices():
    """Every ``model.effort`` value the config schema accepts (``ModelConfig.effort`` choices), as a tuple."""
    from ...core.config import ModelConfig
    return tuple(next(item.metadata["choices"] for item in fields(ModelConfig) if item.name == "effort"))


# ---- session base class ---------------------------------------------------------------------------------------
class AgentSession:
    """Base class of every agent backend session. No ``__init__``; class-level defaults only.

    Instance state a subclass sets (read by the gateway, supervision and evaluation): ``state`` (str),
    ``thread_id`` and ``turn_id`` (str | None), ``turn_done`` (threading.Event), ``turn_result`` (dict | None, at
    least ``{"id", "status", "error": {"message"} | None}``), ``last_activity_at`` (time.monotonic() seconds),
    ``model_limit_reached`` (threading.Event), ``model_proxy`` (the run's ModelRequestProxy or None), ``model``,
    ``effort``, ``provider``, ``price_card``, ``store``, ``root``, ``private``, ``actor_dir``, ``sandbox_enabled``
    and ``isolation``.
    """
    # Backend identifier recorded in runtime/agent, manifest and summary: "codex" | "claude_code" | "opencode" |
    # "external".
    backend_kind = "unknown"
    capabilities = AgentCapabilities()
    # Model-call counting semantics string of the 180-call budget (runtime/model_requests "counting"); None = unset.
    counting = _Hook(None)
    # model.effort values this backend accepts natively (subset of ModelConfig.effort choices); () = all of them.
    effort_choices = ()
    # How much of the earlier tool-result context the Harness may assume the model still holds (design §4.3):
    # "delta" (default; Codex) = images, static observation metadata, capability summaries and program logs already
    # delivered in the current model-context epoch may be left out of later tool results; "full" = every actor tool
    # result is self-contained, for backends whose context loss cannot (yet) be proven. Read by Workbench right after
    # start(): "full" sets Harness.context_epoch_volatile = True. Any other value counts as "delta".
    context_delivery = "delta"
    # Wired by Workbench after construction (descriptors so __new__ objects keep vars(session) == {}):
    on_disconnect = _Hook(None)              # () -> None; normally Harness.request_stop (closes motion admission)
    transport_recovery_gate = _Hook(None)    # () -> str | None; reason a model resend is unsafe, None when safe
    tool_parallel_safe = _Hook(None)         # (tool: str, arguments: dict) -> bool; ToolRegistry.parallel_safe
    # {tool name: True | False | "request_dependent"}: ToolSpec.parallel of every actor tool as published by
    # ToolRegistry.manifest() (NiRuaMcpServer(parallel_hints=...) derives MCP readOnlyHint from it); None = unset.
    tool_parallel_hints = _Hook(None)
    # (tool: str, arguments: dict) -> str | None; ToolRegistry.effect_of (the declared ToolSpec.effect of one call;
    # never raises). Read only by a transport that admits calls by their declared effect (the external backend's
    # session bracket); None = unset.
    tool_effect = _Hook(None)

    # ---- what a backend without an agent NI-RUA runs declares (defaults: every host-mode backend) ----------------
    # Console control commands this backend cannot serve; the Workbench refuses each with
    # AgentCapabilityError("<command>_unsupported_by_backend") before it reaches the session. Default: none.
    console_commands_unsupported = frozenset()
    # True: the operator console may close an open turn on its own authority (release_turn), because the turn belongs
    # to an agent NI-RUA does not run (console reset). False (default): a turn ends natively or through interrupt().
    operator_turn_release = False
    # Manifest / summary keys whose host-mode value does not describe this backend (no model relay, no model, no
    # inactivity supervisor); the Workbench and episode_summary write them as null (unknown), never as a host default
    # or 0. Default (): every key keeps its host-mode value.
    manifest_null_fields = ()
    summary_null_fields = ()
    # (reason: str) -> None; normally Harness.model_context_reset. A session calls it after the model may have lost
    # earlier context (native or relay-detected compaction, uncertain tool-reply delivery, resume after an unclean
    # exit), AFTER it has written runtime/image_context_epoch; the Harness then schedules the decision-context
    # reminder on the next actor tool response (design §4.5). Codex never calls it (it re-injects the reminder into
    # its own compaction request). None = nobody listens.
    on_context_reset = _Hook(None)
    # The run's EmbodimentProfile (robot/model/embodiments; Harness.embodiment). None = ARX-X5.
    embodiment = _Hook(None)
    # Model of the turn currently admitted (token accounting); None = use ``model``.
    active_turn_model = _Hook(None)

    def tool_output_text_budget_bytes(self):
        """UTF-8 byte budget this host applies to the text of ONE tool output before the model sees it, or None when
        the host has no known limit (default). Read per reply by the Harness while ``features.budgeted_action_reply``
        is on (``core/presentation/reply_budget``): action replies are laid out by decision value on every backend,
        and blocks are replaced by references only against a budget returned here.

        Per backend (rules and evidence: docs/development/tool-output-contract.md): Codex 48,000 bytes (catalogue
        ``truncation_policy`` x 1.2); Claude Code 500,000 (text results above it become a file pointer); OpenCode
        None (results up to 600,000 bytes arrived whole; no limit found); external None (the host is not known)."""
        return None

    def supports_turn_images(self) -> bool:
        """返回当前会话实例是否支持在 run_turn 中内联附带多模态图片。"""
        return bool(self.capabilities.turn_images)

    def host_tool_output_truncation_limit(self) -> tuple[int | None, int | None, bool]:
        """返回宿主对单条工具输出的截断阈值 ``(max_bytes, max_lines, text_only_exempt_with_images)``。"""
        return (self.tool_output_text_budget_bytes(), None, False)

    # -- lifecycle (backends implement) ------------------------------------------------------------------------
    def start(self, tools, instructions, resume_thread_id=None):
        """Create (or resume ``resume_thread_id``) the native thread with the actor ``tools`` catalog and
        ``instructions``; write the ``runtime/agent`` identity record; set ``state="idle"``; return ``thread_id``.
        Blocking (seconds)."""
        raise NotImplementedError

    def run_turn(self, prompt, *, images=()):
        """Submit ``prompt`` (and optional inline ``images``) as a new turn and return its ``turn_id`` once admitted (not at completion).

        Scrubs ``prompt`` with ``ActorPresentation.present_text`` (and ``images`` with ``ActorPresentation.present``),
        sets ``state="running"`` and clears
        ``turn_done`` first. Definite rejection restores ``(state, turn_id, turn_result, active_turn_model)``, sets
        ``turn_done`` and raises (``AgentRejected``); unknown admission calls ``_disconnected`` and re-raises.
        Raises ``ValueError("turn_already_active")`` while running and ``ValueError("session_not_connected")`` in
        ``UNCONNECTED_STATES``. Exception (``capabilities.native_wakeup``): while a turn the native runtime started by
        itself is running, the prompt joins that turn and its ``turn_id`` is returned.
        """
        raise NotImplementedError

    def interrupt(self):
        """Ask the native runtime to stop the active turn; returns before completion (the caller polls ``state``
        for up to 45 s). Expires pending native requests and cancels in-flight model calls."""
        raise NotImplementedError

    def close(self):
        """Release every owned resource (idempotent); sets ``state="closed"`` and ``turn_done``. Joins tool workers
        last, so the gateway closes the robot first."""
        raise NotImplementedError

    # -- optional features (defaults for backends without them) -----------------------------------------------
    def steer(self, text):
        """Add operator ``text`` to the ACTIVE turn and return its ``turn_id`` (``capabilities.steer``).

        Default: raises ``AgentCapabilityError("steer_unsupported_by_backend")``.
        """
        raise AgentCapabilityError("steer_unsupported_by_backend")

    def reply(self, request_id, response):
        """Answer the pending native request ``request_id`` with ``response`` and return its ``native_request``
        record (``capabilities.native_requests``). Default: no request is owned, so every id is refused with
        ``ValueError("native_request_expired_or_not_owned")``."""
        raise ValueError("native_request_expired_or_not_owned")

    def expire_requests(self, reason):
        """Mark every pending native request expired with ``reason`` (str). Default: nothing pending; returns None."""
        return None

    def release_turn(self, reason):
        """Close the open turn on operator authority (``operator_turn_release``); ``reason`` names the console
        command (``"operator_reset"``). Returns True when a turn was closed. Default: False (nothing to release)."""
        return False

    def client_view(self):
        """Read-only console data about an agent NI-RUA does not run (the external backend: its endpoint, the
        client's self-reported identity and the last call time), or None. Default: None (NI-RUA runs the agent)."""
        return None

    def requests(self):
        """Every ``native_request`` record of this run's Store (list of dicts, oldest first)."""
        return self.store.states("native_request")

    def completed_final_answer(self):
        """Completed final prose of the current turn (str joined by newlines), or None.

        None unless ``state == "completed"``, ``turn_done`` is set and ``thread_id``/``turn_id`` are known. Reads
        only transcript rows with ``type="agentMessage"``, ``status="completed"`` and ``phase="final_answer"``
        (``Store.completed_final_texts``), so commentary and tool output never count.
        """
        if self.state != STATE_COMPLETED or not self.turn_done.is_set() or not self.thread_id or not self.turn_id:
            return None
        texts = self.store.completed_final_texts(self.thread_id, self.turn_id)
        return "\n".join(texts) if texts else None

    def set_effort(self, effort):
        """Select the reasoning ``effort`` for the NEXT turn (an admitted turn keeps its own effort).

        With ``capabilities.live_effort`` False the running native session keeps the effort it started with and the
        new value applies from the next session start (resume). ``effort`` must be one of ``effort_choices`` (or of
        every ``ModelConfig.effort`` choice when the backend declares none); otherwise raises
        ``ValueError("invalid_effort")``.
        """
        if effort not in (tuple(self.effort_choices) or model_effort_choices()):
            raise ValueError("invalid_effort")
        self.effort = effort

    def set_model(self, model):
        """Select the model id (non-empty str) for the NEXT turn; raises ``ValueError("invalid_model")`` otherwise.
        Price-card and provider checks stay with the caller (operator console)."""
        if not isinstance(model, str) or not model:
            raise ValueError("invalid_model")
        self.model = model

    def model_options(self):
        """Model suggestions for the operator console's ``set_model`` control, or None (the console then offers its
        own default list). Each item: ``{"value": model id, "name": display name or None, "image_input": True |
        False | None}`` (False = the provider reports no image input). Suggestions only: ``set_model`` still accepts
        any well-formed id and the backend refuses what it cannot run. Operator data, never model context.
        Default: None."""
        return None

    def enter_replay(self, thread_id):
        """Turn this unstarted session into a read-only replay viewer of ``thread_id`` (str | None):
        sets ``thread_id`` and ``state="replay"``; starts no native process."""
        self.thread_id = thread_id
        self.state = STATE_REPLAY

    def owned_processes(self):
        """Native processes this session owns right now, as ``[(role, process)]``.

        ``role`` is the stable string recorded in ``run.startup_process_signalled`` events; ``process`` is
        Popen-like (``poll``, ``terminate``, ``kill``, ``pid``, ``returncode``). Not-yet-created processes are
        omitted; the gateway calls this repeatedly while aborting startup. Default: ``[]``.
        """
        return []

    def terminal_attach_spec(self):
        """How a read-only native TUI attaches to the live session, or None (``capabilities.terminal_attach``).

        Returns ``{"cwd": Path, "command_prefix": list[str], "env": dict, "attach_uri": str, "thread_id": str}``
        and optionally ``"command": list[str]``: the TUI runs in ``cwd`` with exactly ``env`` and attaches to
        ``attach_uri`` (recorded in ``terminal.json``; ``unix://``/``ws://``/``wss://``, or also ``http(s)://`` with
        ``command``) for ``thread_id``. Without ``command`` the argv is Codex's ``command_prefix + ["resume",
        "--remote", attach_uri, "--no-alt-screen", thread_id]``; with it, ``command`` is the complete argv
        (``command_prefix`` then names the executable part, for observers). The browser view is read-only: the
        ``Terminal`` never writes to the PTY, so a spec must never need input to show the session, and attaching
        must not start a turn or a model call. Default: None (no terminal; the manifest records
        ``terminal_session_id=null`` and the snapshot ``terminal: null``).
        """
        return None

    def evidence_agent(self, **options):
        """Evidence helper for visual review / self-briefing (``review_evidence`` API) built for this session.

        ``options`` are the helper's keyword options (``model``, ``base_instructions``, ``developer_instructions``,
        ``source``; for the relay client also ``upstream`` and ``effort``). Default: a ``RelayEvidenceClient``
        (``ni_rua.cortex.agents.evidence``) bound to this session: one fresh, streamed, tool-free request per call
        through ``model_proxy`` (design §9.2). A backend with its own helper (``capabilities.native_evidence_agent``)
        overrides this and honours ``agent.evidence.client``.
        """
        from .evidence import RelayEvidenceClient
        return RelayEvidenceClient(self, **options)

    def provenance(self):
        """The manifest ``agent_backend`` block (design §7.3) as a JSON-serialisable dict. Observer data only.

        Default: backend kind, counting semantics and capability flags.
        """
        return dict(kind=self.backend_kind, counting=self.counting, capabilities=self.capabilities.to_dict())


def session_backend_kind(session):
    """``session.backend_kind`` when it is a non-empty str, else ``"codex"``.

    Callers read the backend through this so duck-typed, Codex-shaped sessions (test doubles and objects written
    before the protocol) keep the Codex behaviour.
    """
    kind = getattr(session, "backend_kind", None)
    return kind if isinstance(kind, str) and kind else "codex"


def session_turn_blocks_operator(session):
    """True while ``session`` runs a turn that keeps operator commands waiting (resume, go_home, calibration, handheld
    capture, observer operations, reset and the stop wait): a turn of an agent NI-RUA runs, which ends by itself.
    False for a backend declaring ``operator_turn_release`` (external): its turn belongs to an agent NI-RUA does not
    run and may stay open indefinitely, so it never blocks the operator; motion stays protected by motion admission
    (``Harness.request_stop``), the robot lock and the active-action and active-program checks of each command."""
    return getattr(session, "state", None) == STATE_RUNNING and not getattr(session, "operator_turn_release", False)


def session_capabilities(session, default):
    """``session.capabilities`` when it is an ``AgentCapabilities``, else ``default`` (the caller's fallback)."""
    capabilities = getattr(session, "capabilities", None)
    return capabilities if isinstance(capabilities, AgentCapabilities) else default
